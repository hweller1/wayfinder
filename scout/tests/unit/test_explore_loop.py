"""Explore tools and loop (task 1.11), grep-only, over ScriptedChat + FakeFileSystem.

AC-1.1 (result shape), AC-1.4 (budget fallback), FR-C4 (loop, 3-strike fallback), AC-2.3 (PATH_REFUSED stays
in-loop and never leaks through `report`).
"""

import dataclasses
import json
from collections.abc import Sequence
from typing import Any

import pytest

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import Budget, ExploreRequest, ExploreResult
from scout.core.contracts.wire import StatusReq
from scout.core.explore.loop import ExploreContext, explore
from scout.core.explore.profiles import PROFILES, ExploreProfile
from scout.core.explore.tools import tool_specs
from scout.core.ports.chat import ChatRequest, ChatTurn, ToolCall

AUTH = "def get_user_session(token):\n    s = lookup(token)\n    return s\n\n\nclass Session:\n    pass\n"
FILES = {
    "src/auth.py": AUTH,
    "src/other.py": "x = 1\ny = 2\n",
    ".env": "SECRET=get_user_session\n",
    "README.md": "get_user_session is documented here\n",
}
ID = "a" * 16


def call(
    name: str, args: dict[str, Any] | None = None, id_: str = "c1", parse_error: str | None = None
) -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=args or {}, parse_error=parse_error)


def turn(*calls: ToolCall, tin: int = 100, tout: int = 10, content: str | None = None) -> ChatTurn:
    return ChatTurn(content=content, tool_calls=calls, tokens_in=tin, tokens_out=tout, finish="tool_calls")


def report(
    results: list[dict[str, Any]], summary: str = "found it", gaps: list[str] | None = None
) -> ToolCall:
    return call("report", {"results": results, "summary": summary, "gaps": gaps or []}, id_="r")


class AdvancingChat:
    """ScriptedChat that moves the FakeClock on every model call (a slow model)."""

    def __init__(self, inner: ScriptedChat, clock: FakeClock, seconds: float) -> None:
        self.inner, self.clock, self.seconds = inner, clock, seconds

    async def complete(self, req: ChatRequest) -> ChatTurn:
        self.clock.advance(self.seconds)
        return await self.inner.complete(req)

    def stream(self, req: ChatRequest) -> Any:
        return self.inner.stream(req)


async def run(
    script: Sequence[ChatTurn | ScoutError],
    *,
    budget: Budget | None = None,
    files: dict[str, str] | None = None,
    seconds_per_turn: float = 0.0,
    max_results: int = 12,
    profile: ExploreProfile | None = None,
    backfill_to: int = 0,  # off here: these pin the model's own result (see test_explore_backfill)
) -> tuple[ExploreResult, ScriptedChat, FakeClock]:
    clock = FakeClock()
    chat = ScriptedChat(script)
    model = AdvancingChat(chat, clock, seconds_per_turn)
    ctx = ExploreContext(
        gateway=make_fake_gateway(chat=model, clock=clock),
        fs=FakeFileSystem(files or FILES),
        clock=clock,
        status_req=StatusReq(repo_id=ID * 2, branch_id=ID),
        profile=profile or PROFILES["product"],
        backfill_to=backfill_to,
    )
    req = ExploreRequest(
        brief="where is get_user_session", budget=budget or Budget(), max_results=max_results
    )
    return await explore(req, ctx), chat, clock


def tool_messages(chat: ChatRequest | ScriptedChat) -> list[str]:
    assert isinstance(chat, ScriptedChat)
    return [str(m["content"]) for m in chat.requests[-1].messages if m["role"] == "tool"]


# --- tool specs ----------------
def test_tool_specs_generated_from_models_and_hide_root() -> None:
    specs = tool_specs(semantic_enabled=False)
    assert [s["function"]["name"] for s in specs] == ["grep", "glob", "read_file", "report"]
    for s in specs:
        assert s["type"] == "function"
        assert "root" not in s["function"]["parameters"]["properties"]
    assert specs[0]["function"]["parameters"]["required"] == ["pattern"]
    assert "semantic_search" in [s["function"]["name"] for s in tool_specs(semantic_enabled=True)]


# --- happy path ----------------
async def test_happy_path_grep_read_report_shape_and_usage() -> None:
    """AC-1.1: schema-valid result; usage matches the script exactly."""
    script = [
        turn(call("grep", {"pattern": "get_user_session"}), tin=100, tout=10),
        turn(call("read_file", {"path": "src/auth.py", "start_line": 1, "end_line": 3}), tin=200, tout=20),
        turn(
            report(
                [{"path": "src/auth.py", "start_line": 1, "end_line": 3, "why": "defines it"}],
                summary="Session lookup is get_user_session.",
                gaps=["no tests found"],
            ),
            tin=300,
            tout=30,
        ),
    ]
    res, chat, _ = await run(script, seconds_per_turn=2.0)
    assert [(h.path, h.start_line, h.end_line, h.source) for h in res.results] == [
        ("src/auth.py", 1, 3, "read")
    ]
    assert res.summary == "Session lookup is get_user_session."
    assert res.budget_exhausted is False
    assert res.index.state == "absent"
    assert res.gaps[0] == "no tests found"
    assert any("grep-only" in g for g in res.gaps)
    u = res.usage
    assert u.turns == 3
    assert u.tool_calls == {"grep": 1, "read_file": 1, "report": 1}
    assert (u.model_tokens.in_, u.model_tokens.out) == (600, 60)
    assert u.wall_ms == 6000
    ExploreResult.model_validate(json.loads(res.model_dump_json()))
    # the tool results the model saw are numbered text, and the secret file never matched
    grep_out = chat.requests[1].messages[-1]["content"]
    assert "src/auth.py:1:" in grep_out
    assert ".env" not in grep_out
    assert chat.requests[2].messages[-1]["content"].splitlines()[1] == "1\tdef get_user_session(token):"
    assert chat.remaining == 0


async def test_request_carries_prompt_tools_and_scope() -> None:
    res, chat, _ = await run([turn(report([]))])
    first = chat.requests[0]
    assert first.messages[0]["role"] == "system"
    assert "report" in first.messages[0]["content"]
    assert "where is get_user_session" in first.messages[1]["content"]
    assert [t["function"]["name"] for t in first.tools] == ["grep", "glob", "read_file", "report"]
    assert res.results == []


async def test_glob_tool_and_unmatched_grep_are_ok_text() -> None:
    script = [
        turn(call("glob", {"pattern": "src/*.py"}), call("grep", {"pattern": "zzz_nothing"}, id_="c2")),
        turn(report([])),
    ]
    res, chat, _ = await run(script)
    assert tool_messages(chat) == ["src/auth.py\nsrc/other.py", "no matches"]
    assert res.usage.tool_calls == {"glob": 1, "grep": 1, "report": 1}


# --- report validation ----------------
async def test_report_paths_go_through_the_guard_and_secrets_are_dropped() -> None:
    bad = [
        {"path": ".env", "start_line": 1, "end_line": 1, "why": "secret"},
        {"path": "../etc/passwd", "start_line": 1, "end_line": 1, "why": "escape"},
        {"path": "/etc/passwd", "start_line": 1, "end_line": 1, "why": "absolute"},
        {"path": "nope/missing.py", "start_line": 1, "end_line": 5, "why": "missing"},
        {"path": "src/other.py", "start_line": 50, "end_line": 60, "why": "past the end"},
    ]
    good = {"path": "./src/auth.py", "start_line": 1, "end_line": 2, "why": "ok"}
    res, _, _ = await run([turn(report([*bad, good]))])
    assert [h.path for h in res.results] == ["src/auth.py"]  # normalised
    assert any("5 reported location(s) dropped" in g for g in res.gaps)


async def test_report_line_ranges_are_clamped_to_the_file() -> None:
    items = [
        {"path": "src/other.py", "start_line": 0, "end_line": 99, "why": "too wide"},
        {"path": "src/auth.py", "start_line": 5, "end_line": 2, "why": "inverted"},
        {"path": "src/auth.py", "start_line": -4, "end_line": 1, "why": "negative"},
    ]
    res, _, _ = await run([turn(report(items))])
    assert [(h.path, h.start_line, h.end_line) for h in res.results] == [
        ("src/other.py", 1, 2),
        ("src/auth.py", 5, 5),
        ("src/auth.py", 1, 1),
    ]


async def test_report_hits_on_gitignored_and_oversize_files_dropped_and_max_results_applied() -> None:
    files = {**FILES, ".gitignore": "gen/\n", "gen/out.py": "a\nb\n", "a.py": "1\n2\n", "b.py": "1\n2\n"}
    items = [{"path": p, "start_line": 1, "end_line": 1, "why": "w"} for p in ["gen/out.py", "a.py", "b.py"]]
    res, _, _ = await run([turn(report(items))], files=files, max_results=1)
    assert [h.path for h in res.results] == ["a.py"]


# --- malformed tool calls ----------------
async def test_bad_json_arguments_become_a_tool_error_not_a_crash() -> None:
    script = [turn(call("grep", parse_error="Expecting value: line 1 column 1")), turn(report([]))]
    res, chat, _ = await run(script)
    msg = tool_messages(chat)[0]
    assert msg.startswith("error:")
    assert "not valid JSON" in msg
    assert "Expecting value" in msg
    assert res.usage.tool_calls == {"grep": 1, "report": 1}


async def test_unknown_tool_becomes_a_tool_error_listing_the_real_tools() -> None:
    res, chat, _ = await run([turn(call("rm_rf", {"path": "/"})), turn(report([]))])
    msg = tool_messages(chat)[0]
    assert "unknown tool 'rm_rf'" in msg
    assert "grep, glob, read_file, report" in msg
    assert "rm_rf" not in res.usage.tool_calls


async def test_semantic_search_is_unknown_while_grep_only() -> None:
    _, chat, _ = await run([turn(call("semantic_search", {"query": "x"})), turn(report([]))])
    assert "unknown tool 'semantic_search'" in tool_messages(chat)[0]


@pytest.mark.parametrize(
    ("name", "args", "needle"),
    [
        ("grep", {}, "pattern"),
        ("grep", {"pattern": "x", "context": 99}, "context"),
        ("grep", {"pattern": "x", "bogus": 1}, "bogus"),
        ("read_file", {"path": "src/auth.py", "start_line": "one"}, "start_line"),
        ("glob", {"pattern": 5}, "pattern"),
        ("grep", {"pattern": "x", "root": "/etc"}, "root"),
        ("report", {"summary": "no results key"}, "results"),
    ],
)
async def test_wrong_arguments_become_a_tool_error(name: str, args: dict[str, Any], needle: str) -> None:
    res, chat, _ = await run([turn(call(name, args)), turn(report([]))])
    msg = tool_messages(chat)[0]
    assert msg.startswith("error: invalid arguments")
    assert needle in msg
    assert res.budget_exhausted is False
    assert res.usage.turns == 2


async def test_three_consecutive_malformed_calls_fall_back_to_best_hits() -> None:
    script = [
        turn(call("grep", {"pattern": "get_user_session"})),
        turn(call("nope")),
        turn(call("grep", {})),
        turn(call("grep", parse_error="bad")),
    ]
    res, chat, _ = await run(script)
    assert chat.remaining == 0
    assert res.usage.turns == 4
    assert res.budget_exhausted is False
    assert any("3 consecutive malformed" in g for g in res.gaps)
    assert {h.path for h in res.results} == {"README.md", "src/auth.py"}  # .env never
    assert all(h.source == "grep" for h in res.results)


async def test_a_good_call_resets_the_malformed_streak() -> None:
    script = [
        turn(call("nope")),
        turn(call("nope")),
        turn(call("glob", {"pattern": "*.md"})),
        turn(call("nope")),
        turn(call("nope")),
        turn(report([{"path": "README.md", "start_line": 1, "end_line": 1, "why": "doc"}])),
    ]
    res, chat, _ = await run(script)
    assert chat.remaining == 0
    assert [h.path for h in res.results] == ["README.md"]
    assert not any("malformed" in g for g in res.gaps)


async def test_text_only_turns_count_as_strikes_and_get_a_nudge() -> None:
    script = [turn(content="thinking out loud") for _ in range(3)]
    res, chat, _ = await run(script)
    assert chat.requests[1].messages[-1] == {
        "role": "user",
        "content": "Reply with a tool call. When you have enough evidence, call report.",
    }
    assert res.results == []
    assert any("no candidate locations" in g for g in res.gaps)


# --- PATH_REFUSED / NOT_FOUND stay in-loop ----------------
async def test_path_refused_and_not_found_stay_in_loop() -> None:
    script = [
        turn(
            call("read_file", {"path": ".env"}, id_="a"),
            call("read_file", {"path": "../x"}, id_="b"),
            call("read_file", {"path": "ghost.py"}, id_="c"),
            call("grep", {"pattern": "x", "path": "/etc"}, id_="d"),
            call("glob", {"pattern": "../*"}, id_="e"),
        ),
        turn(report([])),
    ]
    res, chat, _ = await run(script)
    msgs = tool_messages(chat)
    assert [m.split(":")[1].strip() for m in msgs[:2]] == [ErrorCode.PATH_REFUSED] * 2
    assert msgs[2].startswith("error: NOT_FOUND")
    assert msgs[3].startswith("error: PATH_REFUSED")
    assert msgs[4].startswith("error: PATH_REFUSED")
    assert "SECRET" not in "".join(msgs)
    assert res.usage.turns == 2
    assert res.usage.tool_calls["read_file"] == 3


# --- budget exhaustion ----------------
async def test_turn_budget_exhaustion_returns_best_hits_flagged() -> None:
    """AC-1.4: valid result, budget_exhausted, ranked from what was seen, gaps entry."""
    script = [
        turn(call("grep", {"pattern": "get_user_session"})),
        turn(call("grep", {"pattern": "s"})),  # dense in src/auth.py: outranks the single README match
        turn(call("grep", {"pattern": "never reached"})),  # beyond the budget: must not be requested
    ]
    res, chat, _ = await run(script, budget=Budget(max_turns=2, max_seconds=120))
    assert res.budget_exhausted is True
    assert res.usage.turns == 2
    assert chat.calls == 2
    assert chat.remaining == 1
    assert res.results
    assert all(h.source == "grep" for h in res.results)
    assert res.results[0].path == "src/auth.py"
    assert any("budget exhausted (turns)" in g for g in res.gaps)
    ExploreResult.model_validate(json.loads(res.model_dump_json()))


async def test_clock_budget_exhaustion_returns_best_hits_flagged() -> None:
    script = [
        turn(call("grep", {"pattern": "get_user_session"})),
        turn(call("read_file", {"path": "src/auth.py", "start_line": 1, "end_line": 3})),
        turn(call("glob", {"pattern": "*"})),
    ]
    res, chat, clock = await run(script, budget=Budget(max_turns=40, max_seconds=5), seconds_per_turn=3.0)
    assert res.budget_exhausted is True
    assert chat.calls == 2
    assert chat.remaining == 1
    assert any("budget exhausted (seconds)" in g for g in res.gaps)
    assert res.usage.wall_ms == 6000
    assert [(h.path, h.source) for h in res.results][0] == ("src/auth.py", "read")  # read outranks grep


async def test_budget_fallback_with_nothing_seen_is_still_valid() -> None:
    res, _, _ = await run(
        [turn(call("glob", {"pattern": "*.py"}))], budget=Budget(max_turns=1, max_seconds=120)
    )
    assert res.budget_exhausted is True
    assert res.results == []
    assert any("no candidate locations" in g for g in res.gaps)


async def test_fallback_hits_exclude_secrets_and_respect_max_results() -> None:
    files = {f"m{i}.py": f"needle {i}\n" for i in range(5)} | {".env": "needle\n"}
    script = [turn(call("grep", {"pattern": "needle"})), turn(call("glob", {"pattern": "*"}))]
    res, _, _ = await run(script, budget=Budget(max_turns=1, max_seconds=120), files=files, max_results=3)
    assert len(res.results) == 3
    assert ".env" not in [h.path for h in res.results]


# --- errors that are not the model's doing ----------------
async def test_chat_errors_propagate_without_a_partial_result() -> None:
    with pytest.raises(ScoutError) as ei:
        await run([ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "down")])
    assert ei.value.code is ErrorCode.UPSTREAM_UNAVAILABLE


async def test_requests_are_deterministic_across_runs() -> None:
    script = [turn(call("grep", {"pattern": "get_user_session"})), turn(report([]))]
    a, ca, _ = await run(script)
    b, cb, _ = await run(script)
    assert a.model_dump() == b.model_dump()
    assert [r.messages for r in ca.requests] == [r.messages for r in cb.requests]


# --- context-window guard (task 1.11a) ----------------
BIG = {**FILES, "big.py": "".join(f"needle line {i} padding padding padding\n" for i in range(150))}


def small_ctx(tokens: int, base: str = "product") -> ExploreProfile:
    return dataclasses.replace(PROFILES[base], context_tokens=tokens)


def test_context_tokens_default_for_both_profiles() -> None:
    assert PROFILES["product"].context_tokens == 60_000
    assert PROFILES["parity"].context_tokens == 60_000


async def test_long_grep_outputs_are_elided_oldest_first_and_the_loop_continues() -> None:
    """FR-C2: oldest tool results shrink to a stub; the loop goes on to a valid report."""
    script = [
        turn(call("grep", {"pattern": "needle"}, id_="g1")),
        turn(call("grep", {"pattern": "needle line 1"}, id_="g2")),
        turn(call("grep", {"pattern": "needle line 2"}, id_="g3")),
        turn(report([{"path": "big.py", "start_line": 1, "end_line": 2, "why": "w"}])),
    ]
    res, chat, _ = await run(script, files=BIG, profile=small_ctx(1500))
    assert res.budget_exhausted is False
    assert [h.path for h in res.results] == ["big.py"]
    last = chat.requests[-1].messages
    tools = [str(m["content"]) for m in last if m["role"] == "tool"]
    assert tools[0] == "[elided: 21 lines from grep]"
    assert not tools[-1].startswith("[elided:")  # the latest turn is intact
    assert "needle line 2 " in tools[-1]
    assert chat.remaining == 0


async def test_system_prompt_and_brief_are_never_elided() -> None:
    script = [turn(call("grep", {"pattern": "needle"})) for _ in range(6)]
    res, chat, _ = await run(script, files=BIG, profile=small_ctx(1500))
    for r in chat.requests:
        assert r.messages[0]["content"] == PROFILES["product"].prompt
        assert r.messages[1]["content"].startswith("where is get_user_session")
    assert res.budget_exhausted is True


async def test_growing_script_ends_as_context_exhaustion_with_valid_hits() -> None:
    script = [turn(call("grep", {"pattern": "needle"}, id_=f"g{i}")) for i in range(30)]
    res, chat, _ = await run(script, files=BIG, profile=small_ctx(1200))
    assert res.budget_exhausted is True
    assert any("budget exhausted (context)" in g for g in res.gaps)
    assert res.results
    assert all(h.path == "big.py" for h in res.results)
    assert chat.remaining > 0  # stopped before the script ran out
    ExploreResult.model_validate(json.loads(res.model_dump_json()))
    # usage is exact: only calls that were made count
    assert res.usage.turns == chat.calls
    assert res.usage.tool_calls == {"grep": chat.calls}
    assert res.usage.model_tokens.in_ == 100 * chat.calls


async def test_upstream_context_length_error_maps_to_context_exhaustion() -> None:
    err = ScoutError(
        ErrorCode.INVALID_ARGUMENT,
        "This model's maximum context length is 65536 tokens. However, you requested 70000 tokens",
    )
    script = [turn(call("grep", {"pattern": "get_user_session"})), err]
    res, chat, _ = await run(script)
    assert res.budget_exhausted is True
    assert any("budget exhausted (context)" in g for g in res.gaps)
    assert {h.path for h in res.results} == {"README.md", "src/auth.py"}
    assert res.usage.turns == 1  # the rejected call produced no turn
    assert (res.usage.model_tokens.in_, res.usage.model_tokens.out) == (100, 10)
    assert chat.remaining == 0


async def test_upstream_context_flag_in_details_is_recognised() -> None:
    err = ScoutError(ErrorCode.INVALID_ARGUMENT, "bad request", details={"reason": "context_length"})
    res, _, _ = await run([err])
    assert res.budget_exhausted is True
    assert any("(context)" in g for g in res.gaps)


async def test_other_upstream_400s_still_propagate() -> None:
    with pytest.raises(ScoutError):
        await run([ScoutError(ErrorCode.INVALID_ARGUMENT, "unsupported parameter")])
