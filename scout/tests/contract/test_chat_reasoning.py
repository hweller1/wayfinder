"""Reasoning retention behind the `ChatModel` port (owner decision 2026-10-27; matches Wayfinder's training).

The SFT data always shows the model its earlier assistant reasoning (the Glimmer chat template renders
`reasoning_content` on every assistant turn), so the explore loop keeps each turn's reasoning in the
history and the OpenAI-compatible adapter re-sends it by default. A host turns it off with
`retain_reasoning=False` or `SCOUT_LLM_RETAIN_REASONING=0`. Non-reasoning models never return the field,
so they are unaffected. The stub is an `httpx.MockTransport`; no network.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.adapters.gateway_http.client import _turn_from  # pyright: ignore[reportPrivateUsage]
from scout.adapters.openai_chat import OpenAIChat, retain_reasoning_from_env
from scout.core.contracts.types import ExploreRequest
from scout.core.contracts.wire import StatusReq
from scout.core.explore.loop import ExploreContext, explore
from scout.core.explore.profiles import PROFILES
from scout.core.ports.chat import ChatRequest, ChatTurn, ToolCall
from scout.surfaces.gateway_http.app import completion_json

REPO = Path(__file__).resolve().parents[1] / "fixtures" / "repo_small"
BASE = "https://llm.test/v1"
THINK_1 = "I should grep for the identifier first."
THINK_2 = "The hit is in session.py, so read it."


def _call(id_: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _completion(*calls: dict[str, Any], **extra: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": None, "tool_calls": list(calls), **extra},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


REPORT = {
    "results": [{"path": "src/auth/session.py", "start_line": 6, "end_line": 12, "why": "defines it"}],
    "summary": "defined in session.py",
    "gaps": [],
}


def _script(field: str = "reasoning_content") -> list[dict[str, Any]]:
    return [
        _completion(_call("c1", "grep", {"pattern": "getUserSession"}), **{field: THINK_1}),
        _completion(
            _call("c2", "read_file", {"path": "src/auth/session.py", "start_line": 6, "end_line": 12}),
            **{field: THINK_2},
        ),
        _completion(_call("c3", "report", REPORT)),
    ]


class Stub:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self.script.pop(0))

    def chat(self, **kw: Any) -> OpenAIChat:
        return OpenAIChat(BASE, None, "m", transport=httpx.MockTransport(self.handler), **kw)


async def _explore(chat: Any) -> Any:
    files = {p.relative_to(REPO).as_posix(): p.read_bytes() for p in REPO.rglob("*") if p.is_file()}
    clock = FakeClock()
    ctx = ExploreContext(
        gateway=make_fake_gateway(chat=chat, clock=clock),
        fs=FakeFileSystem(files),
        clock=clock,
        status_req=StatusReq(repo_id="a" * 32, branch_id="a" * 16),
    )
    return await explore(ExploreRequest(brief="where is the session lookup implemented?"), ctx)


def _assistant_reasoning(body: dict[str, Any]) -> list[str | None]:
    return [m.get("reasoning_content") for m in body["messages"] if m["role"] == "assistant"]


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
async def test_adapter_returns_reasoning_from_either_field(field: str) -> None:
    stub = Stub(_script(field))
    turn = await stub.chat().complete(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert turn.reasoning == THINK_1


async def test_non_reasoning_model_yields_none_and_sends_no_reasoning_field() -> None:
    stub = Stub(
        [_completion(_call("c1", "grep", {"pattern": "x"})), _completion(_call("c2", "report", REPORT))]
    )
    chat = stub.chat()
    first = await chat.complete(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert first.reasoning is None
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None}]
    await chat.complete(ChatRequest(messages=msgs))
    assert "reasoning_content" not in json.dumps(stub.bodies[1])


async def test_loop_resends_every_earlier_assistant_turns_reasoning_by_default() -> None:
    stub = Stub(_script())
    result = await _explore(stub.chat())
    assert [h.path for h in result.results] == ["src/auth/session.py"]
    assert _assistant_reasoning(stub.bodies[0]) == []
    assert _assistant_reasoning(stub.bodies[1]) == [THINK_1]
    assert _assistant_reasoning(stub.bodies[2]) == [THINK_1, THINK_2]


async def test_retain_reasoning_off_keeps_it_off_the_wire() -> None:
    stub = Stub(_script())
    result = await _explore(stub.chat(retain_reasoning=False))
    assert [h.path for h in result.results] == ["src/auth/session.py"]  # the loop still completes
    assert all("reasoning_content" not in json.dumps(b) for b in stub.bodies)
    assert _assistant_reasoning(stub.bodies[2]) == [None, None]


def test_retain_reasoning_env_switch() -> None:
    assert retain_reasoning_from_env({}) is True
    assert retain_reasoning_from_env({"SCOUT_LLM_RETAIN_REASONING": "1"}) is True
    for off in ("0", "false", "No", "OFF", " 0 "):
        assert retain_reasoning_from_env({"SCOUT_LLM_RETAIN_REASONING": off}) is False
    assert OpenAIChat.from_env({"SCOUT_LLM_URL": BASE}).retain_reasoning is True
    assert (
        OpenAIChat.from_env({"SCOUT_LLM_URL": BASE, "SCOUT_LLM_RETAIN_REASONING": "0"}).retain_reasoning
        is False
    )


async def test_the_port_carries_reasoning_for_any_chat_model() -> None:
    """ScriptedChat is a non-OpenAI ChatModel: reasoning reaches its recorded history, no adapter."""
    think = ChatTurn(
        None, (ToolCall("c1", "grep", {"pattern": "getUserSession"}),), 10, 5, "tool_calls", reasoning=THINK_1
    )
    done = ChatTurn(None, (ToolCall("c2", "report", REPORT),), 10, 5, "tool_calls")
    chat = ScriptedChat([think, done])
    await _explore(chat)
    assert [m.get("reasoning_content") for m in chat.requests[1].messages if m["role"] == "assistant"] == [
        THINK_1
    ]


def test_gateway_wire_round_trips_reasoning() -> None:
    turn = ChatTurn("c", (ToolCall("c1", "grep", {"pattern": "x"}),), 3, 2, "tool_calls", reasoning=THINK_1)
    body = completion_json(turn)
    assert body["choices"][0]["message"]["reasoning_content"] == THINK_1
    assert _turn_from(json.dumps(body)).reasoning == THINK_1
    plain = dataclasses.replace(turn, reasoning=None)
    assert "reasoning_content" not in completion_json(plain)["choices"][0]["message"]
    assert _turn_from(json.dumps(completion_json(plain))).reasoning is None


# --- the context budget counts reasoning ------------------------------------------------------------------
BIG = {"big.py": "".join(f"needle line {i} padding padding padding\n" for i in range(150))}


def _grep_turn(i: int, reasoning: str | None) -> ChatTurn:
    return ChatTurn(
        None, (ToolCall(f"g{i}", "grep", {"pattern": "needle"}),), 100, 10, "tool_calls", reasoning=reasoning
    )


async def _requests_with(reasoning: str | None, limit: int) -> list[ChatRequest]:
    chat = ScriptedChat(
        [_grep_turn(i, reasoning) for i in range(4)]
        + [
            ChatTurn(
                None,
                (ToolCall("r", "report", {"results": [], "summary": "s", "gaps": []}),),
                1,
                1,
                "tool_calls",
            )
        ]
    )
    clock = FakeClock()
    ctx = ExploreContext(
        gateway=make_fake_gateway(chat=chat, clock=clock),
        fs=FakeFileSystem(BIG),
        clock=clock,
        status_req=StatusReq(repo_id="a" * 32, branch_id="a" * 16),
        profile=dataclasses.replace(PROFILES["product"], context_tokens=limit),
    )
    await explore(ExploreRequest(brief="needle"), ctx)
    return chat.requests


def _elided(req: ChatRequest) -> int:
    return sum(1 for m in req.messages if m["role"] == "tool" and str(m["content"]).startswith("[elided: "))


async def test_reasoning_tokens_count_toward_the_context_budget() -> None:
    """A limit that fits the tool results alone is exceeded once the reasoning is in the history, so the
    oldest tool results are elided and the prompt stays under the limit."""
    from scout.core.explore.loop import _prompt_tokens  # pyright: ignore[reportPrivateUsage]

    limit = 3000
    lean = await _requests_with(None, limit)
    heavy = await _requests_with("think " * 400, limit)
    assert _elided(lean[-1]) == 0
    assert _elided(heavy[-1]) > 0
    for req in heavy:
        assert _prompt_tokens([dict(m) for m in req.messages]) <= limit


async def test_retain_reasoning_off_leaves_it_out_of_the_loop_history_and_budget() -> None:
    """Off means the loop never holds the reasoning, so the context estimate matches what is sent."""
    stub = Stub(_script())
    turn = await stub.chat(retain_reasoning=False).complete(
        ChatRequest(messages=[{"role": "user", "content": "x"}])
    )
    assert turn.reasoning is None
