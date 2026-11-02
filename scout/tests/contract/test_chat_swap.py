"""`OpenAIChat` swap test (task 7.3): the explore loop runs unchanged over a stub OpenAI server.

AC-5.2, FR-P4. The stub is an `httpx.MockTransport` replaying OpenAI-shaped `/chat/completions`
responses (vLLM `qwen3_coder` style: structured `tool_calls` with JSON-string arguments); no network.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.adapters.openai_chat import OpenAIChat
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ExploreRequest
from scout.core.contracts.wire import StatusReq
from scout.core.explore.loop import ExploreContext, explore
from scout.core.ports.chat import ChatRequest

REPO = Path(__file__).resolve().parents[1] / "fixtures" / "repo_small"
IDENT = "getUserSession"
BASE = "https://llm.test/v1"


def _tool_call(id_: str, name: str, args: Any) -> dict[str, Any]:
    raw = args if isinstance(args, str) else json.dumps(args)
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": raw}}


def _completion(
    *calls: dict[str, Any], content: str | None = None, pin: int = 100, pout: int = 10
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "tool_calls": list(calls)},
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": pin, "completion_tokens": pout},
    }


class StubServer:
    """Replays `script` (JSON bodies, or `(status, body)` pairs) and records every request."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0)
        status, body = item if isinstance(item, tuple) else (200, item)
        return (
            httpx.Response(status, json=body)
            if not isinstance(body, str)
            else httpx.Response(status, text=body)
        )

    def body(self, i: int) -> dict[str, Any]:
        return json.loads(self.requests[i].content)

    def chat(self, **kw: Any) -> OpenAIChat:
        return OpenAIChat(
            BASE,
            "sk-test",
            "scout-llm",
            {"Modal-Key": "k"},
            transport=httpx.MockTransport(self.handler),
            **kw,
        )


def _req(**kw: Any) -> ChatRequest:
    return ChatRequest(messages=[{"role": "user", "content": "hi"}], **kw)


async def _explore(chat: OpenAIChat) -> Any:
    files = {p.relative_to(REPO).as_posix(): p.read_bytes() for p in REPO.rglob("*") if p.is_file()}
    clock = FakeClock()
    ctx = ExploreContext(
        gateway=make_fake_gateway(chat=chat, clock=clock),
        fs=FakeFileSystem(files),
        clock=clock,
        status_req=StatusReq(repo_id="a" * 32, branch_id="a" * 16),
    )
    return await explore(ExploreRequest(brief="where is the session lookup implemented?"), ctx)


REPORT_ARGS = {
    "results": [{"path": "src/auth/session.py", "start_line": 6, "end_line": 12, "why": "defines it"}],
    "summary": "getUserSession is defined in src/auth/session.py.",
    "gaps": [],
}


async def test_explore_completes_with_swapped_chat_model() -> None:
    """AC-5.2: the loop runs over `OpenAIChat` in place of the gateway's scripted chat, no core change."""
    server = StubServer(
        [
            _completion(_tool_call("c1", "grep", {"pattern": IDENT}), pin=120, pout=12),
            _completion(
                _tool_call(
                    "c2", "read_file", {"path": "src/auth/session.py", "start_line": 6, "end_line": 12}
                ),
                pin=240,
                pout=18,
            ),
            _completion(_tool_call("c3", "report", REPORT_ARGS), pin=360, pout=40),
        ]
    )
    result = await _explore(server.chat())
    assert [h.path for h in result.results] == ["src/auth/session.py"]
    assert result.usage.turns == 3
    assert result.usage.tool_calls == {"grep": 1, "read_file": 1, "report": 1}
    assert (result.usage.model_tokens.in_, result.usage.model_tokens.out) == (720, 70)
    assert len(server.requests) == 3
    last = server.body(2)["messages"]  # tool results were fed back as OpenAI tool messages
    assert [m["role"] for m in last][:2] == ["system", "user"]
    assert {m["tool_call_id"] for m in last if m["role"] == "tool"} == {"c1", "c2"}


async def test_parse_error_is_fed_back_and_the_loop_recovers() -> None:
    """A tool call whose arguments are not JSON arrives with `parse_error`; the loop counts it malformed,
    tells the model, and the next valid call completes the explore."""
    server = StubServer(
        [
            _completion(_tool_call("bad", "grep", '{"pattern": "getUser')),
            _completion(_tool_call("r", "report", REPORT_ARGS)),
        ]
    )
    result = await _explore(server.chat())
    assert result.usage.turns == 2
    tool_msgs = [m["content"] for m in server.body(1)["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "not valid JSON" in tool_msgs[0]


async def test_parse_error_fields_on_the_turn() -> None:
    server = StubServer(
        [
            _completion(
                _tool_call("a", "grep", "{not json"),
                _tool_call("b", "grep", "[1, 2]"),
                _tool_call("c", "grep", ""),
                _tool_call("d", "grep", {"pattern": "x"}),  # object instead of a JSON string
            )
        ]
    )
    turn = await server.chat().complete(_req())
    a, b, c, d = turn.tool_calls
    assert a.parse_error is not None
    assert a.arguments == {}
    assert b.parse_error == "arguments must be a JSON object"
    assert (c.parse_error, c.arguments) == (None, {})
    assert (d.parse_error, d.arguments) == (None, {"pattern": "x"})
    assert turn.finish == "tool_calls"


async def test_request_carries_a1_sampling_headers_and_thinking_flag() -> None:
    """Agents-A1 sampling (0.85 / 0.95 / top_k 20 / presence 1.1) passes through; `thinking` maps to
    `chat_template_kwargs.enable_thinking`; auth and extra headers are sent."""
    server = StubServer([_completion(content="ok"), _completion(content="ok")])
    chat = server.chat()
    tools = [{"type": "function", "function": {"name": "grep", "parameters": {"type": "object"}}}]
    turn = await chat.complete(_req(tools=tools, tool_choice="required", max_tokens=777, thinking=False))
    await chat.complete(_req())
    assert turn.content == "ok"
    assert turn.tool_calls == ()
    assert turn.finish == "stop"
    body = server.body(0)
    assert body["model"] == "scout-llm"
    assert body["temperature"] == 0.85
    assert body["top_p"] == 0.95
    assert body["top_k"] == 20
    assert body["presence_penalty"] == 1.1
    assert body["max_tokens"] == 777
    assert body["stream"] is False
    assert body["tools"] == tools
    assert body["tool_choice"] == "required"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert server.body(1)["chat_template_kwargs"] == {"enable_thinking": True}
    assert "tools" not in server.body(1)
    assert "tool_choice" not in server.body(1)  # vLLM rejects tool_choice without tools
    r = server.requests[0]
    assert r.url == "https://llm.test/v1/chat/completions"
    assert r.headers["authorization"] == "Bearer sk-test"
    assert r.headers["modal-key"] == "k"


async def test_strict_endpoint_omits_vllm_extensions_and_zero_penalty() -> None:
    server = StubServer([_completion(content="ok")])
    await server.chat(vllm_extensions=False).complete(_req(presence_penalty=0.0))
    body = server.body(0)
    assert "top_k" not in body
    assert "chat_template_kwargs" not in body
    assert "presence_penalty" not in body


async def test_sft_sampling_profile_sends_only_temperature_and_top_p() -> None:
    """wayfinder-v3 was evaluated with temperature and top_p only (no top_k, penalty or thinking flag)."""
    server = StubServer([_completion(content="ok")])
    chat = OpenAIChat(
        BASE, "k", "wayfinder-v3", sampling="sft", transport=httpx.MockTransport(server.handler)
    )
    await chat.complete(_req(presence_penalty=1.1))
    body = server.body(0)
    assert body["model"] == "wayfinder-v3"
    assert (body["temperature"], body["top_p"]) == (0.85, 0.95)
    for absent in ("top_k", "presence_penalty", "chat_template_kwargs"):
        assert absent not in body


async def test_a1_sampling_profile_is_the_default() -> None:
    server = StubServer([_completion(content="ok")])
    await server.chat().complete(_req(presence_penalty=1.1))
    body = server.body(0)
    assert body["top_k"] == 20
    assert body["presence_penalty"] == 1.1
    assert "chat_template_kwargs" in body


def test_sampling_profile_from_env() -> None:
    env = {"SCOUT_LLM_URL": BASE}
    assert OpenAIChat.from_env(env)._sft is False  # pyright: ignore[reportPrivateUsage]
    assert OpenAIChat.from_env({**env, "SCOUT_LLM_SAMPLING": "sft"})._sft is True  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ScoutError):
        OpenAIChat.from_env({**env, "SCOUT_LLM_SAMPLING": "nope"})


async def test_no_api_key_sends_no_authorization() -> None:
    server = StubServer([_completion(content="ok")])
    chat = OpenAIChat(BASE + "/", None, "m", transport=httpx.MockTransport(server.handler))
    await chat.complete(_req())
    assert "authorization" not in server.requests[0].headers
    assert str(server.requests[0].url) == "https://llm.test/v1/chat/completions"


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (429, "slow down", ErrorCode.RATE_LIMITED),
        (500, "boom", ErrorCode.UPSTREAM_UNAVAILABLE),
        (503, "boom", ErrorCode.UPSTREAM_UNAVAILABLE),
        (504, "boom", ErrorCode.UPSTREAM_TIMEOUT),
        (401, "SECRET-VENDOR-TEXT", ErrorCode.INTERNAL),
        (404, "no such model", ErrorCode.INTERNAL),
        (400, "bad field", ErrorCode.INVALID_ARGUMENT),
    ],
)
async def test_http_errors_map_to_typed_errors_without_vendor_text(
    status: int, body: str, code: ErrorCode
) -> None:
    server = StubServer([(status, body + " SECRET-VENDOR-TEXT")])
    with pytest.raises(ScoutError) as ei:
        await server.chat().complete(_req())
    assert ei.value.code == code
    assert "SECRET-VENDOR-TEXT" not in str(ei.value)
    assert ei.value.details.get("reason") is None


async def test_rate_limit_retry_after_and_context_length() -> None:
    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"}, text="x")

    chat = OpenAIChat(BASE, None, "m", transport=httpx.MockTransport(limited))
    with pytest.raises(ScoutError) as ei:
        await chat.complete(_req())
    assert ei.value.code == ErrorCode.RATE_LIMITED
    assert ei.value.details == {"retry_after": 7.0}

    server = StubServer(
        [(400, {"error": {"message": "This model's maximum context length is 65536 tokens."}})]
    )
    with pytest.raises(ScoutError) as ei2:
        await server.chat().complete(_req())
    assert ei2.value.code == ErrorCode.INVALID_ARGUMENT
    assert ei2.value.details["reason"] == "context_length"


async def test_context_overflow_ends_explore_as_budget_exhaustion() -> None:
    """The adapter's context-length error is what the loop's compaction path keys on (gateway-mediated)."""
    server = StubServer([(400, "maximum context length exceeded")])
    result = await _explore(server.chat())
    assert result.budget_exhausted


@pytest.mark.parametrize(
    "make_exc", [lambda: httpx.ReadTimeout("t"), lambda: httpx.ConnectError("c")], ids=["timeout", "connect"]
)
async def test_transport_failures(make_exc: Callable[[], Exception]) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise make_exc()

    chat = OpenAIChat(BASE, None, "m", transport=httpx.MockTransport(boom))
    with pytest.raises(ScoutError) as ei:
        await chat.complete(_req())
    assert ei.value.code in (ErrorCode.UPSTREAM_TIMEOUT, ErrorCode.UPSTREAM_UNAVAILABLE)
    assert (ei.value.code == ErrorCode.UPSTREAM_TIMEOUT) == isinstance(make_exc(), httpx.TimeoutException)


@pytest.mark.parametrize("body", [{"choices": []}, {"nope": 1}, "not json at all"])
async def test_malformed_response_is_upstream_unavailable(body: Any) -> None:
    server = StubServer([body])
    with pytest.raises(ScoutError) as ei:
        await server.chat().complete(_req())
    assert ei.value.code == ErrorCode.UPSTREAM_UNAVAILABLE


def test_stream_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        OpenAIChat(BASE, None, "m").stream(_req())


def test_from_env() -> None:
    with pytest.raises(ScoutError):
        OpenAIChat.from_env({})
    chat = OpenAIChat.from_env(
        {
            "SCOUT_LLM_URL": BASE,
            "SCOUT_LLM_API_KEY": "k",
            "MODAL_PROXY_TOKEN_ID": "i",
            "MODAL_PROXY_TOKEN_SECRET": "s",
        }
    )
    assert chat._headers["Modal-Key"] == "i"  # pyright: ignore[reportPrivateUsage]
    assert chat._model == "scout-llm"  # pyright: ignore[reportPrivateUsage]
