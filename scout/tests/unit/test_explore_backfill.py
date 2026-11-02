# ruff: noqa: PT018
"""Deterministic backfill (`explore.backfill_to`): order of the three tiers, dedup against the model's report,
the flag, an empty index, a failing semantic tier, rendering and the config setting."""

from pathlib import Path
from typing import Any

import pytest

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ExploreRequest, ExploreResult
from scout.core.contracts.wire import StatusReq
from scout.core.crypto import keys
from scout.core.explore.backfill import brief_identifiers
from scout.core.explore.loop import ExploreContext, explore
from scout.core.explore.render import FURTHER_HEADING, render_text
from scout.core.explore.semantic import make_backend
from scout.core.explore.tools import SemanticBackend
from scout.core.ports.chat import ChatTurn, ToolCall
from scout.core.sync import merkle
from scout.core.sync.engine import SyncEngine
from scout.core.sync.state import StateDir
from scout.wiring import backfill_setting

TENANT = "dev"
BRIEF = "where is `get_user_session` defined and who renders it?"
FILES = {
    "src/auth/session.py": "def get_user_session(token):\n    return lookup(token)\n",
    "src/auth/token.py": "def lookup(token):\n    return token\n",
    "src/web/views.py": "from auth import get_user_session\nget_user_session(1)\n",
    "docs/notes.md": "the session lookup is described here\n",
    "tests/test_session.py": "def test_it():\n    get_user_session('x')\n",
    "src/billing/invoice.py": "def render_invoice(order):\n    return order.total\n",
}


def _call(name: str, args: dict[str, Any], id_: str = "c") -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=args)


def _turn(*calls: ToolCall) -> ChatTurn:
    return ChatTurn(content=None, tool_calls=calls, tokens_in=10, tokens_out=1, finish="tool_calls")


def _report(*paths: str) -> ChatTurn:
    results = [{"path": p, "start_line": 1, "end_line": 2, "why": "model's pick"} for p in paths]
    return _turn(_call("report", {"results": results, "summary": "done", "gaps": []}, "r"))


def _read(path: str, id_: str) -> ToolCall:
    return _call("read_file", {"path": path, "start_line": 1, "end_line": 2}, id_)


async def _run(
    tmp_path: Path,
    script: list[ChatTurn],
    *,
    backfill_to: int = 12,
    files: dict[str, str] | None = None,
    sync: bool = True,
    semantic: bool = True,
    max_results: int = 12,
    broken_search: bool = False,
) -> ExploreResult:
    chat, clock = ScriptedChat(script), FakeClock()
    gw = make_fake_gateway(TENANT, chat=chat, clock=clock)
    k = keys.derive(await gw.team_key())
    fs = FakeFileSystem(FILES if files is None else files, state_dir=str(tmp_path / "git" / "scout"))
    status_req = StatusReq(repo_id="0" * 32, branch_id="0" * 16)
    if sync:
        await SyncEngine(fs, gw, k, clock).sync()  # type: ignore[arg-type]
        st = StateDir.open(fs).read_state()
        assert st.repo_id
        assert st.last_branch
        status_req = StatusReq(repo_id=st.repo_id, branch_id=st.last_branch)
    backend: SemanticBackend | None = None
    if semantic:
        backend = make_backend(gw, k, TENANT, status_req, lambda d: merkle.leaf_hash(k, d))
    if broken_search and backend is not None:
        real = backend

        async def _boom(query: str, glob: str | None, n: int) -> Any:
            if query == BRIEF:  # only the backfill's own query fails, not the model's tool calls
                raise ScoutError(ErrorCode.RATE_LIMITED, "slow down")
            return await real.search(query, glob, n)

        backend = SemanticBackend(search=_boom, leaf_hash=real.leaf_hash)
    ctx = ExploreContext(gw, fs, clock, status_req, semantic=backend, backfill_to=backfill_to)
    return await explore(ExploreRequest(brief=BRIEF, max_results=max_results), ctx)


def _paths(res: ExploreResult) -> list[str]:
    return [h.path for h in res.results]


async def test_model_ranking_first_untouched_then_opened_then_semantic_then_grep(tmp_path: Path) -> None:
    script = [
        _turn(_read("src/auth/token.py", "a"), _read("docs/notes.md", "b"), _read("src/auth/token.py", "c")),
        _report("src/auth/session.py"),
    ]
    res = await _run(tmp_path, script)
    paths = _paths(res)
    assert paths[0] == "src/auth/session.py"
    assert res.results[0].verified
    assert paths[1:3] == ["src/auth/token.py", "docs/notes.md"]  # (a) opened, not reported, first-seen order
    assert all(not h.verified for h in res.results[1:])
    assert all(h.source == "read" for h in res.results[1:3])
    assert len(paths) == len(set(paths)) == len(FILES)  # every file once; semantic + grep supplied the rest
    assert {"src/web/views.py", "tests/test_session.py", "src/billing/invoice.py"} <= set(paths)


async def test_dedup_against_the_models_report_and_across_tiers(tmp_path: Path) -> None:
    script = [_turn(_read("src/auth/session.py", "a")), _report("src/auth/session.py", "src/web/views.py")]
    res = await _run(tmp_path, script)
    paths = _paths(res)
    assert paths.count("src/auth/session.py") == 1
    assert paths.count("src/web/views.py") == 1
    assert paths[:2] == ["src/auth/session.py", "src/web/views.py"]
    assert len(paths) == len(set(paths))


async def test_grep_tier_ranks_by_identifiers_then_matches_then_path(tmp_path: Path) -> None:
    res = await _run(tmp_path, [_report("src/auth/token.py")], semantic=False, sync=False)
    paths = _paths(res)
    assert paths[0] == "src/auth/token.py"
    # `get_user_session` appears 1x in session.py, 2x in views.py, 1x in test_session.py
    assert paths[1:] == ["src/web/views.py", "src/auth/session.py", "tests/test_session.py"]
    assert all(h.source == "grep" and not h.verified for h in res.results[1:])


async def test_semantic_tier_follows_score_order_and_is_unverified(tmp_path: Path) -> None:
    res = await _run(tmp_path, [_report("src/auth/session.py")])
    extra = [h for h in res.results if not h.verified]
    assert extra
    sem = [h for h in extra if h.source == "semantic"]
    scores = [h.score or 0.0 for h in sem]
    assert sem
    assert scores == sorted(scores, reverse=True)


async def test_flag_zero_leaves_the_models_list_alone(tmp_path: Path) -> None:
    script = [_turn(_read("src/auth/token.py", "a")), _report("src/auth/session.py")]
    res = await _run(tmp_path, script, backfill_to=0)
    assert _paths(res) == ["src/auth/session.py"]
    assert all(h.verified for h in res.results)


async def test_target_caps_the_list_and_never_trims_the_models_own(tmp_path: Path) -> None:
    res = await _run(tmp_path, [_report("src/auth/session.py")], backfill_to=3)
    assert len(res.results) == 3
    many = [_report(*FILES)]
    res = await _run(tmp_path, many, backfill_to=2)
    assert _paths(res) == list(FILES)  # the model named more than the target: untouched, no extras
    assert all(h.verified for h in res.results)


async def test_max_results_bounds_the_backfill(tmp_path: Path) -> None:
    res = await _run(tmp_path, [_report("src/auth/session.py")], max_results=2)
    assert len(res.results) == 2


async def test_empty_index_and_empty_repo(tmp_path: Path) -> None:
    res = await _run(tmp_path, [_report()], files={"README.md": "# demo\n"}, sync=False, semantic=False)
    assert res.results == []  # nothing in the brief matches anything: no padding, no crash
    res = await _run(tmp_path, [_report()], files={}, sync=False, semantic=False)
    assert res.results == []
    # an index that was never synced: semantic is off, grep tier still fills
    res = await _run(tmp_path, [_report("src/auth/token.py")], sync=False)
    assert _paths(res)[0] == "src/auth/token.py"
    assert len(res.results) > 1
    assert all(h.source == "grep" for h in res.results[1:])


async def test_a_failing_semantic_tier_is_skipped_not_raised(tmp_path: Path) -> None:
    res = await _run(tmp_path, [_report("src/auth/token.py")], broken_search=True)
    assert _paths(res)[0] == "src/auth/token.py"
    assert len(res.results) > 1
    assert all(h.source == "grep" for h in res.results[1:])


async def test_refused_and_ignored_files_are_never_backfilled(tmp_path: Path) -> None:
    files = {**FILES, ".env": "get_user_session=secret\n"}
    res = await _run(tmp_path, [_turn(_read(".env", "a")), _report("src/auth/session.py")], files=files)
    assert ".env" not in _paths(res)


async def test_rendering_marks_backfilled_entries_apart_in_order(tmp_path: Path) -> None:
    res = await _run(tmp_path, [_report("src/auth/session.py")])
    text = render_text(res, 1500)
    lines = text.splitlines()
    i = lines.index(FURTHER_HEADING)
    assert lines[i - 1].startswith("src/auth/session.py:1-2")
    assert "(unverified)" not in lines[i - 1]
    tail = [ln for ln in lines[i + 1 :] if not ln.startswith("gap:")]
    assert len(tail) == len(res.results) - 1
    assert all("(unverified)" in ln for ln in tail)
    assert [ln.split(":")[0] for ln in tail] == [h.path for h in res.results[1:]]


def test_render_never_ends_on_a_bare_heading() -> None:
    res = ExploreResult.model_validate(
        {
            "results": [
                {"path": "a.py", "start_line": 1, "end_line": 1, "why": "x", "source": "read"},
                {
                    "path": "b.py",
                    "start_line": 1,
                    "end_line": 1,
                    "why": "y",
                    "source": "grep",
                    "verified": False,
                },
            ],
            "summary": "s",
            "gaps": [],
            "index": {"state": "absent", "coverage": 0.0, "semantic_enabled": False},
            "usage": {"turns": 1, "tool_calls": {}, "model_tokens": {"in": 1, "out": 1}, "wall_ms": 1},
        }
    )
    full = render_text(res, 1500).splitlines()
    assert full[-2:] == [FURTHER_HEADING, "b.py:1-1 (unverified)  y"]
    for cap in range(1, 40):  # whatever the cap drops, a heading is never the last line
        lines = render_text(res, cap).splitlines()
        assert not lines or lines[-1] != FURTHER_HEADING


def test_brief_identifiers_pick_code_like_tokens_only() -> None:
    got = brief_identifiers(
        "Fix `parse_args` in cli/main.py so getUserSession handles the error case when it fails"
    )
    assert got[0] == "parse_args"
    assert "getUserSession" in got
    assert "cli/main.py" in got
    assert "handles" not in got
    assert "error" not in got
    assert brief_identifiers("make the thing work better") == []


def test_backfill_setting_precedence_and_validation() -> None:
    assert backfill_setting({}) == 0  # off by default (Addendum E)
    assert backfill_setting({}, {"explore": {"backfill_to": 5}}) == 5
    assert backfill_setting({"SCOUT_EXPLORE_BACKFILL_TO": "0"}, {"explore": {"backfill_to": 5}}) == 0
    assert backfill_setting({"SCOUT_EXPLORE_BACKFILL_TO": "8"}) == 8
    for bad in ("-1", "x", "41"):
        with pytest.raises(ScoutError) as e:
            backfill_setting({"SCOUT_EXPLORE_BACKFILL_TO": bad})
        assert e.value.code is ErrorCode.INVALID_ARGUMENT
