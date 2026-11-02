# ruff: noqa: S603, S607, E501  (fixed argv in tests; long fixture lines)
"""The retrieval spec (docs/retrieval-spec.md): parsing, three-layer precedence, the index-description
generator, the example miner and its leak guard, the Claude Code hooks, and the core's use of the spec
(prompt addendum, exclude/boost/recency filters, k, index off)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.contracts.types import ExploreRequest
from scout.core.contracts.wire import StatusReq
from scout.core.explore import spec as core_spec
from scout.core.explore.candidates import RawHit
from scout.core.explore.loop import ExploreContext, explore
from scout.core.ports.chat import ChatTurn, ToolCall

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "integrations" / "claude-code"
HOOK = PLUGIN / "hooks" / "spec_hooks.py"
REWRITE = PLUGIN / "hooks" / "rewrite_explore.py"

_s = importlib.util.spec_from_file_location("scout_spec", PLUGIN / "lib" / "scout_spec.py")
assert _s
assert _s.loader
sp = importlib.util.module_from_spec(_s)
sys.modules["scout_spec"] = sp
_s.loader.exec_module(sp)

BASE_ENV = {"PATH": os.environ["PATH"], "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def git(cwd: Path, *args: str) -> str:
    env = {
        **BASE_ENV,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


def commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        git(repo, "add", rel)
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    return r


# --- parsing -------------------------------------------------------------------------------------------
FULL = """
instructions = "Prefer src/ over vendored code."
k = 8

[index]
enabled = true
symbols = true

[paths]
boost = ["src/", "!docs/"]
exclude = ["legacy/**", "*.snap"]
boost_weight = 0.5

[recency]
boost = 0.2

[examples]
mine = 2
before = "abc"

[delivery]
search_first = true

[[example]]
query = "add retry to the http client"
files = ["src/http.py", "tests/test_http.py"]
"""


def test_parse_full_spec() -> None:
    flat, warns = sp.normalize(sp.parse_toml(FULL))
    assert warns == []
    assert flat["k"] == 8
    assert flat["index_symbols"] is True
    assert flat["boost_weight"] == 0.5
    assert flat["boost"] == ["src/", "!docs/"]
    assert flat["exclude"] == ["legacy/**", "*.snap"]
    assert flat["recency_boost"] == 0.2
    assert flat["mine"] == 2
    assert flat["before"] == "abc"
    assert flat["search_first"] is True
    assert flat["examples"] == [
        {"query": "add retry to the http client", "files": ["src/http.py", "tests/test_http.py"]}
    ]


def test_subset_parser_matches_tomllib_on_a_spec() -> None:
    import tomllib

    text = FULL + '\n# comment\n[delivery2]\nx = [1, 2.5, "a # b", true]\n'
    assert sp._parse_subset(text) == tomllib.loads(text)  # the 3.9 fallback parses what 3.11+ parses
    multi = 'instructions = """\\\nline one\nline two\n"""\nk = 3\n'
    assert sp._parse_subset(multi) == tomllib.loads(multi)


def test_invalid_values_are_ignored_and_reported() -> None:
    flat, warns = sp.normalize(
        {
            "k": "five",
            "paths": {"boost": "src/"},
            "bogus": 1,
            "index": {"enabled": "yes"},
            "example": [{"query": "x"}],
        }
    )
    assert flat == {}
    assert len(warns) == 6


def test_k_and_weights_are_clamped() -> None:
    flat, _ = sp.normalize(
        {"k": 99, "recency": {"boost": 7}, "examples": {"mine": 50}, "paths": {"boost_weight": -1}}
    )
    assert (flat["k"], flat["recency_boost"], flat["mine"], flat["boost_weight"]) == (
        30,
        1.0,
        sp.MAX_EXAMPLES,
        0.0,
    )


def test_unreadable_spec_never_raises(repo: Path) -> None:
    (repo / ".scout").mkdir()
    (repo / ".scout" / "spec.toml").write_text("this is = = not toml [")
    commit(repo, {"src/a.py": "x = 1\n"}, "initial commit of a")
    res = sp.resolve(str(repo), {**BASE_ENV}, state_dir=str(repo / ".git" / "scout"))
    assert res["layers"]["user"]
    assert any("not readable TOML" in w for w in res["warnings"])
    assert res["k"] == 5  # fell back to the lower layers


# --- precedence ------------------------------------------------------------------------------------------
def test_system_defaults() -> None:
    s = sp.merge(sp.SYSTEM)
    assert s["k"] == 5
    assert s["index_enabled"] is True
    assert "vendor/" in s["exclude"]
    assert "package-lock.json" in s["exclude"]
    assert sp.SYSTEM_FACTS["open_at_coverage"] == 0.8
    assert sp.SYSTEM_FACTS["sync_minutes"] == 5


def test_later_layers_override_earlier() -> None:
    harness = {"k": 6, "instructions": "harness says A", "search_first": False, "boost": ["lib/"]}
    user = {
        "k": 9,
        "instructions": "user says B",
        "search_first": True,
        "boost": ["!lib/", "src/"],
        "exclude": ["gen/", "!vendor/"],
    }
    s = sp.merge(sp.SYSTEM, harness, user)
    assert s["k"] == 9
    assert s["search_first"] is True
    assert s["instructions"] == "harness says A\n\nuser says B"  # concatenated, user last
    assert s["boost"] == ["src/"]  # the user's !lib/ removed the harness boost
    assert "vendor/" not in s["exclude"]
    assert "gen/" in s["exclude"]
    assert "node_modules/" in s["exclude"]


def test_curated_examples_user_first() -> None:
    h = {"examples": [{"query": "h", "files": ["h.py"]}]}
    u = {"examples": [{"query": "u", "files": ["u.py"]}]}
    assert [e["query"] for e in sp.merge(sp.SYSTEM, h, u)["examples"]] == ["u", "h"]


def test_shipped_harness_defaults_parse_cleanly() -> None:
    flat, warns = sp.normalize(sp.parse_toml(sp.HARNESS_FILE.read_text()))
    assert warns == []
    assert flat["workflow_step"] is True
    assert flat["search_first"] is False
    assert flat["mine"] == 3
    assert "vendored" in flat["instructions"]


def test_user_spec_file_overrides_harness_in_resolve(repo: Path) -> None:
    commit(repo, {"src/a.py": "x = 1\n"}, "initial commit of a")
    (repo / ".scout").mkdir()
    (repo / ".scout" / "spec.toml").write_text(
        "k = 7\n[delivery]\nsearch_first = true\nworkflow_step = false\n[index]\nenabled = false\n"
    )
    res = sp.resolve(str(repo), {**BASE_ENV}, state_dir=str(repo / ".git" / "scout"))
    assert res["k"] == 7
    assert res["index_enabled"] is False
    assert res["delivery"] == {
        "session_context": True,
        "workflow_step": False,
        "search_first": True,
        "rewrite_explore": True,
        "report_guidance": "",
        "background": False,
        "handoff": "agent",
        "search_policy": "deny",
    }
    assert res["layers"]["harness"]
    assert res["layers"]["user"].endswith("spec.toml")
    assert (repo / ".git" / "scout" / "spec_resolved.json").exists()


def test_spec_disable_env_silences_everything(repo: Path) -> None:
    commit(repo, {"src/a.py": "x = 1\n"}, "initial commit of a")
    s, info = sp.load_layers(str(repo), {"SCOUT_SPEC_DISABLE": "1"})
    assert info["disabled"]
    assert not (s["workflow_step"])
    assert not (s["search_first"])
    assert not (s["session_context"])
    assert not (s["rewrite_explore"])


# --- index description ----------------------------------------------------------------------------------------
def test_index_description_languages_dirs_packages_symbols(repo: Path) -> None:
    files = {
        "src/core/engine.py": "class Engine:\n    pass\n\n" * 3,
        "src/core/util.py": "def f():\n    pass\n",
        "src/api/routes.py": "class Router:\n    pass\n",
        "web/app.ts": "export class AppShell {}\n",
        "tests/test_engine.py": "def test(): pass\n",
        "README.md": "# hi\n",
        "vendor/lib/x.py": "class Vendored: pass\n",
    }
    commit(repo, files, "add everything to the repository")
    tracked = sp.tracked_files(str(repo))
    kept = sp.filter_excluded(str(repo), tracked, sp.SYSTEM["exclude"])
    assert "vendor/lib/x.py" not in kept
    assert "src/core/engine.py" in kept
    desc = sp.index_description(str(repo), kept, name="demo", symbols=True)
    assert desc.startswith("Repository index: demo (6 indexed files).")
    assert "Languages (source files): Python 4, TypeScript 1" in desc
    assert "src 3" in desc
    assert "tests 1" in desc
    assert "web 1" in desc
    assert "src/core 2" in desc
    assert "Engine" in desc
    assert "Vendored" not in desc
    assert "Top symbols" not in sp.index_description(str(repo), kept, symbols=False)


def test_scoutignore_shapes_the_description(repo: Path) -> None:
    commit(repo, {"src/a.py": "x=1\n", "gen/b.py": "y=2\n", ".scoutignore": "gen/\n"}, "add a generated tree")
    kept = sp.filter_excluded(str(repo), sp.tracked_files(str(repo)), [])
    assert "gen/b.py" not in kept
    assert "src/a.py" in kept


def test_description_is_cached_per_commit(repo: Path) -> None:
    commit(repo, {"src/a.py": "x = 1\n"}, "first commit of the repo")
    state = str(repo / ".git" / "scout")
    a = sp.resolve(str(repo), {**BASE_ENV}, state_dir=state)
    b = sp.resolve(str(repo), {**BASE_ENV}, state_dir=state)
    assert a["cached"] is False
    assert b["cached"] is True
    assert a["description"] == b["description"]
    commit(repo, {"src/b.py": "y = 2\n"}, "second commit of the repo")
    c = sp.resolve(str(repo), {**BASE_ENV}, state_dir=state)
    assert c["cached"] is False
    assert "2 indexed files" in c["description"]
    assert len(list((repo / ".git" / "scout" / "spec_cache").glob("*.json"))) == 2


# --- example miner and its leak guard ----------------------------------------------------------------------------
@pytest.fixture
def history(repo: Path) -> dict[str, str]:
    sha = {}
    sha["c1"] = commit(repo, {"lib/a.py": "1\n", "test/a.py": "1\n"}, "add the alpha parser to the library")
    sha["c2"] = commit(repo, {"lib/b.py": "1\n"}, "add the beta parser to the library")
    sha["c3"] = commit(repo, {"README.md": "x\n"}, "docs only change to the readme")
    sha["c4"] = commit(
        repo, {"lib/gold.py": "1\n", "lib/c.py": "1\n"}, "touch the gold file and another file"
    )
    sha["base"] = commit(repo, {"lib/d.py": "1\n"}, "the base commit of the task itself")
    sha["after"] = commit(repo, {"lib/e.py": "1\n"}, "a commit after the base commit")
    return sha


def test_miner_is_strictly_before_the_base_commit(repo: Path, history: dict[str, str]) -> None:
    ex = sp.mine_examples(str(repo), history["base"], n=10)
    shas = {e["sha"] for e in ex}
    assert history["base"] not in shas
    assert history["after"] not in shas
    assert history["c1"] in shas
    assert history["c2"] in shas
    assert history["c3"] not in shas  # no source file touched
    files = {f for e in ex for f in e["files"]}
    assert "lib/d.py" not in files
    assert "lib/e.py" not in files


def test_miner_never_touches_gold_or_held_out_files(repo: Path, history: dict[str, str]) -> None:
    ex = sp.mine_examples(str(repo), history["base"], avoid=["lib/gold.py", "test/*.py"], n=10)
    shas = {e["sha"] for e in ex}
    assert history["c4"] not in shas  # touches gold
    assert history["c1"] not in shas  # touches test/a.py, a held-out glob
    assert shas == {history["c2"]}


def test_miner_is_deterministic_and_respects_n_and_max_files(repo: Path, history: dict[str, str]) -> None:
    a = sp.mine_examples(str(repo), history["base"], n=1)
    b = sp.mine_examples(str(repo), history["base"], n=1)
    assert a == b
    assert len(a) == 1
    assert a[0]["sha"] == history["c4"]
    assert sp.mine_examples(str(repo), history["base"], n=5, max_files=1) != []
    assert all(len(e["files"]) <= 1 for e in sp.mine_examples(str(repo), history["base"], n=5, max_files=1))
    assert sp.mine_examples(str(repo), history["base"], n=0) == []


def test_resolve_uses_before_and_avoid_from_the_spec(repo: Path, history: dict[str, str]) -> None:
    (repo / ".scout").mkdir()
    (repo / ".scout" / "spec.toml").write_text(
        f'[examples]\nmine = 5\nbefore = "{history["base"]}"\navoid = ["lib/gold.py"]\n'
    )
    res = sp.resolve(str(repo), {**BASE_ENV}, state_dir=str(repo / ".git" / "scout"))
    mined = {e["sha"] for e in res["examples"] if e["source"] == "mined"}
    assert history["after"][:12] not in mined
    assert history["base"][:12] not in mined
    assert history["c4"][:12] not in mined
    assert history["c2"][:12] in mined
    assert all("lib/gold.py" not in e["files"] for e in res["examples"])


# --- delivery text and hook decisions (pure) ----------------------------------------------------------------------
RESOLVED: dict[str, Any] = {
    "instructions": "Prefer src/.", "description": "Repository index: r (3 indexed files).", "boost": ["src/"],
    "examples": [{"query": "add retry", "files": ["a.py", "b.py"], "source": "mined"}],
    "delivery": {"session_context": True, "workflow_step": True, "search_first": True, "rewrite_explore": True},
}  # fmt: skip
PFX = "mcp__plugin_scout_scout__"


def test_session_context_carries_instructions_description_examples() -> None:
    ctx = sp.session_context(RESOLVED, "scout:scout-explorer")
    assert ctx is not None
    assert "Prefer src/." in ctx
    assert "Repository index: r" in ctx
    assert "add retry -> a.py, b.py" in ctx
    assert "Prefer paths: src/" in ctx


def test_session_context_toggle() -> None:
    off = {**RESOLVED, "delivery": {**RESOLVED["delivery"], "session_context": False}}
    assert sp.session_context(off, "Explore") is None


def test_workflow_step_names_the_target_and_is_toggleable() -> None:
    out = sp.prompt_output("fix the bug", RESOLVED, "scout:scout-explorer")
    assert out is not None
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert ctx.startswith("Before reading or editing code, delegate to the `scout:scout-explorer` subagent")
    assert (
        "`Explore`" in sp.prompt_output("x", RESOLVED, "Explore")["hookSpecificOutput"]["additionalContext"]
    )  # type: ignore[index]
    assert sp.prompt_output("<task-notification>done", RESOLVED, "Explore") is None
    off = {**RESOLVED, "delivery": {**RESOLVED["delivery"], "workflow_step": False}}
    assert sp.prompt_output("x", off, "Explore") is None


def test_search_first_denies_main_grep_until_first_delegation() -> None:
    tgt = "scout:scout-explorer"
    out, label, mapped = sp.pretool_decision({"tool_name": "Grep"}, False, RESOLVED, tgt, PFX)
    assert label == "grep_denied"
    assert mapped is False
    assert out is not None
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert tgt in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert sp.pretool_decision({"tool_name": "Glob"}, False, RESOLVED, tgt, PFX)[1] == "grep_denied"
    _, label, mapped = sp.pretool_decision(
        {"tool_name": "Agent", "tool_input": {"subagent_type": "Explore"}}, False, RESOLVED, tgt, PFX
    )
    assert label == "delegation"
    assert mapped is True
    out, label, _ = sp.pretool_decision({"tool_name": "Grep"}, True, RESOLVED, tgt, PFX)
    assert out is None
    assert label == "grep_allowed"


def test_search_first_exempts_subagents_and_scout_calls_map_the_session() -> None:
    tgt = "Explore"
    assert sp.pretool_decision({"tool_name": "Grep", "agent_id": "a1"}, False, RESOLVED, tgt, PFX)[0] is None
    _, label, mapped = sp.pretool_decision(
        {"tool_name": "Agent", "agent_id": "a1"}, False, RESOLVED, tgt, PFX
    )
    assert label == "delegation_sub"
    assert mapped is False
    _, label, mapped = sp.pretool_decision({"tool_name": PFX + "semantic_search"}, False, RESOLVED, tgt, PFX)
    assert label == "search_main"
    assert mapped is True
    _, _, mapped = sp.pretool_decision(
        {"tool_name": PFX + "semantic_search", "agent_id": "a1"}, False, RESOLVED, tgt, PFX
    )
    assert mapped is False


def test_search_first_off_never_denies() -> None:
    off = {**RESOLVED, "delivery": {**RESOLVED["delivery"], "search_first": False}}
    assert sp.pretool_decision({"tool_name": "Grep"}, False, off, "Explore", PFX)[0] is None


# --- the hook scripts end to end ---------------------------------------------------------------------------------
def run_hook(
    mode: str, payload: dict[str, Any], repo: Path, data: Path, extra: dict[str, str] | None = None
) -> tuple[str, str]:
    env = {
        **BASE_ENV,
        "HOME": str(data.parent),
        "CLAUDE_PLUGIN_DATA": str(data),
        "SCOUT_KEYSTORE": "file",
        "SCOUT_NO_KEYCHAIN_PROBE": "1",
        **(extra or {}),
    }
    p = subprocess.run(
        [sys.executable, str(HOOK), mode],
        input=json.dumps({"cwd": str(repo), "session_id": "s1", **payload}),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert p.returncode == 0
    return p.stdout, p.stderr


def events(data: Path) -> list[dict[str, Any]]:
    f = data / "spec_hooks.jsonl"
    return [json.loads(ln) for ln in f.read_text().splitlines()] if f.exists() else []


def test_session_start_hook_injects_context_and_writes_resolved_file(repo: Path, tmp_path: Path) -> None:
    commit(repo, {"src/a.py": "class A: pass\n", "lib/b.py": "x=1\n"}, "initial commit of two files")
    commit(repo, {"src/a.py": "class A:\n    x = 1\n"}, "make class A carry a field")
    data = tmp_path / "data"
    out, _ = run_hook("session_start", {}, repo, data, {"SCOUT_SPEC_TARGET": "Explore"})
    ctx = json.loads(out)["hookSpecificOutput"]
    assert ctx["hookEventName"] == "SessionStart"
    text = ctx["additionalContext"]
    assert "Repository index:" in text
    assert "Prefer the project's own source" in text
    assert "initial commit of two files -> lib/b.py, src/a.py" in text
    res = json.loads((repo / ".git" / "scout" / "spec_resolved.json").read_text())
    assert res["k"] == 5
    assert res["layers"]["harness"] is True
    [ev] = events(data)
    assert ev["hook"] == "SessionStart"
    assert ev["injected"]
    assert ev["has_description"]
    assert ev["n_examples"] >= 1


def test_session_start_hook_is_silent_outside_a_repo_and_when_disabled(repo: Path, tmp_path: Path) -> None:
    commit(repo, {"src/a.py": "x=1\n"}, "initial commit of one file")
    plain = tmp_path / "plain"
    plain.mkdir()
    assert run_hook("session_start", {}, plain, tmp_path / "d1")[0] == ""
    assert run_hook("session_start", {}, repo, tmp_path / "d2", {"SCOUT_SPEC_DISABLE": "1"})[0] == ""


def test_prompt_hook_on_by_default_and_user_toggle_off(repo: Path, tmp_path: Path) -> None:
    commit(repo, {"src/a.py": "x=1\n"}, "initial commit of one file")
    data = tmp_path / "data"
    run_hook("session_start", {}, repo, data, {"SCOUT_SPEC_TARGET": "Explore"})
    out, _ = run_hook("prompt", {"prompt": "add a flag"}, repo, data, {"SCOUT_SPEC_TARGET": "Explore"})
    assert "delegate to the `Explore` subagent" in json.loads(out)["hookSpecificOutput"]["additionalContext"]
    (repo / ".scout").mkdir()
    (repo / ".scout" / "spec.toml").write_text("[delivery]\nworkflow_step = false\n")
    run_hook(
        "session_start", {}, repo, data, {"SCOUT_SPEC_TARGET": "Explore"}
    )  # re-resolves with the user layer
    assert run_hook("prompt", {"prompt": "add a flag"}, repo, data, {"SCOUT_SPEC_TARGET": "Explore"})[0] == ""


def test_pretool_hook_denies_grep_then_allows_after_delegation(repo: Path, tmp_path: Path) -> None:
    commit(repo, {"src/a.py": "x=1\n"}, "initial commit of one file")
    (repo / ".scout").mkdir()
    (repo / ".scout" / "spec.toml").write_text("[delivery]\nsearch_first = true\n")
    data = tmp_path / "data"
    extra = {"SCOUT_SPEC_TARGET": "Explore"}
    run_hook("session_start", {}, repo, data, extra)
    out, _ = run_hook("pretool", {"tool_name": "Grep", "tool_input": {"pattern": "x"}}, repo, data, extra)
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert (
        run_hook("pretool", {"tool_name": "Grep", "agent_id": "sub", "tool_input": {}}, repo, data, extra)[0]
        == ""
    )
    assert (
        run_hook(
            "pretool", {"tool_name": "Agent", "tool_input": {"subagent_type": "Explore"}}, repo, data, extra
        )[0]
        == ""
    )
    assert (
        run_hook("pretool", {"tool_name": "Grep", "tool_input": {"pattern": "x"}}, repo, data, extra)[0] == ""
    )
    evs = [e["event"] for e in events(data) if e["hook"] == "PreToolUse"]
    assert evs == ["grep_denied", "other", "delegation", "grep_allowed"]


def test_pretool_default_is_not_search_first(repo: Path, tmp_path: Path) -> None:
    commit(repo, {"src/a.py": "x=1\n"}, "initial commit of one file")
    data = tmp_path / "data"
    run_hook("session_start", {}, repo, data, {"SCOUT_SPEC_TARGET": "Explore"})
    assert (
        run_hook(
            "pretool", {"tool_name": "Grep", "tool_input": {}}, repo, data, {"SCOUT_SPEC_TARGET": "Explore"}
        )[0]
        == ""
    )


def test_target_is_explore_when_scout_is_not_ready_and_scout_when_index_off_is_false(
    repo: Path, tmp_path: Path
) -> None:
    commit(repo, {"src/a.py": "x=1\n"}, "initial commit of one file")
    data = tmp_path / "data"
    out, _ = run_hook("prompt", {"prompt": "go"}, repo, data)  # no key, no index: stock Explore
    assert "`Explore`" in json.loads(out)["hookSpecificOutput"]["additionalContext"]
    state = repo / ".git" / "scout"
    state.mkdir(exist_ok=True)
    (state / "status.json").write_text(json.dumps({"state": "ready", "coverage": 1.0}))
    creds = data.parent / ".config" / "scout"
    creds.mkdir(parents=True)
    (creds / "credentials").write_text("k")
    out, _ = run_hook("prompt", {"prompt": "go"}, repo, data)
    assert "`scout:scout-explorer`" in json.loads(out)["hookSpecificOutput"]["additionalContext"]
    (repo / ".scout").mkdir()
    (repo / ".scout" / "spec.toml").write_text("[index]\nenabled = false\n")
    run_hook("session_start", {}, repo, data)
    out, _ = run_hook("prompt", {"prompt": "go"}, repo, data)
    assert (
        "`Explore`" in json.loads(out)["hookSpecificOutput"]["additionalContext"]
    )  # index off: grep-only Explore


def test_rewrite_hook_respects_the_spec(repo: Path, tmp_path: Path) -> None:
    commit(repo, {"src/a.py": "x=1\n"}, "initial commit of one file")
    data, home = tmp_path / "data", tmp_path
    state = repo / ".git" / "scout"
    state.mkdir(exist_ok=True)
    from datetime import UTC, datetime

    (state / "status.json").write_text(
        json.dumps(
            {
                "state": "ready",
                "coverage": 1.0,
                "gateway_ok_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    )
    (state / "mcp.pid").write_text(json.dumps({"pid": os.getpid(), "started_at": "x"}))
    (home / ".config" / "scout").mkdir(parents=True)
    (home / ".config" / "scout" / "credentials").write_text("k")
    env = {
        **BASE_ENV,
        "HOME": str(home),
        "CLAUDE_PLUGIN_DATA": str(data),
        "SCOUT_KEYSTORE": "file",
        "SCOUT_NO_KEYCHAIN_PROBE": "1",
    }

    def rewrite() -> str:
        payload = {
            "tool_name": "Agent",
            "cwd": str(repo),
            "tool_input": {"subagent_type": "Explore", "prompt": "find"},
        }
        return subprocess.run(
            [sys.executable, str(REWRITE)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        ).stdout

    assert "scout:scout-explorer" in rewrite()
    (state / "spec_resolved.json").write_text(
        json.dumps({"version": 1, "delivery": {"rewrite_explore": False}, "index_enabled": True})
    )
    assert rewrite() == ""
    (state / "spec_resolved.json").write_text(
        json.dumps({"version": 1, "delivery": {"rewrite_explore": True}, "index_enabled": False})
    )
    assert rewrite() == ""
    assert "spec-off" in (data / "hook.log").read_text()


def test_hooks_json_wires_the_spec_hooks() -> None:
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    cmds = [h["command"] for ev in hooks.values() for g in ev for h in g["hooks"]]
    assert sum("spec_hooks.py" in c for c in cmds) == 3
    assert "UserPromptSubmit" in hooks
    pre = next(g for g in hooks["PreToolUse"] if "spec_hooks.py" in g["hooks"][0]["command"])
    import re

    for tool in ("Agent", "Task", "Grep", "Glob", PFX + "semantic_search"):
        assert re.fullmatch(pre["matcher"], tool)


def test_scout_explorer_description_is_sharp() -> None:
    text = (PLUGIN / "agents" / "scout-explorer.md").read_text()
    desc = next(ln for ln in text.splitlines() if ln.startswith("description:"))
    assert "Delegate to it FIRST" in desc
    assert "retrieval spec" in desc
    assert "## Retrieval spec" in text


# --- the core's use of the spec ---------------------------------------------------------------------------------------
def hit(path: str, score: float | None = None) -> RawHit:
    return RawHit(path, 1, 5, "h", score)


def test_core_spec_load_and_tolerance(tmp_path: Path) -> None:
    assert core_spec.load(str(tmp_path)) is None
    (tmp_path / "spec_resolved.json").write_text("{nope")
    assert core_spec.load(str(tmp_path)) is None
    (tmp_path / "spec_resolved.json").write_text(json.dumps({"version": 1, "layers": {"disabled": True}}))
    assert core_spec.load(str(tmp_path)) is None
    (tmp_path / "spec_resolved.json").write_text(json.dumps({"version": 1, "k": 99, "exclude": ["vendor/"], "index_enabled": False,
                                                              "examples": [{"query": "q", "files": ["a.py"]}, {"query": 3}], "layers": {"harness": True, "user": "x"}}))  # fmt: skip
    s = core_spec.load(str(tmp_path))
    assert s is not None
    assert s.k == 30
    assert s.index_enabled is False
    assert s.examples == (("q", ("a.py",)),)
    assert s.layers == ("system", "harness", "user")


def test_excludes_drop_hits() -> None:
    f = core_spec.Filters(("vendor/", "*.lock"), (), 0.3, 0.0)
    out = f.apply([hit("vendor/x.py"), hit("src/a.py"), hit("deps.lock"), hit("src/vendor_notes.py")])
    assert [h.path for h in out] == ["src/a.py", "src/vendor_notes.py"]


def test_boost_reranks_close_scores_but_not_distant_ones() -> None:
    f = core_spec.Filters((), ("src/",), 0.3, 0.0)
    close = f.apply([hit("docs/a.md", 1.0), hit("src/a.py", 0.9)])
    assert [h.path for h in close] == ["src/a.py", "docs/a.md"]
    far = f.apply([hit("docs/a.md", 1.0), hit("src/a.py", 0.5)])
    assert [h.path for h in far] == ["docs/a.md", "src/a.py"]
    unscored = f.apply([hit("docs/a.md"), hit("src/a.py")])  # rank scores 1.0 and 0.5
    assert [h.path for h in unscored] == ["docs/a.md", "src/a.py"]


def test_recency_boost_prefers_newer_files() -> None:
    f = core_spec.Filters((), (), 0.3, 0.2)
    mt = {"old.py": 1_000, "new.py": 2_000}
    out = f.apply([hit("old.py", 1.0), hit("new.py", 0.95)], lambda p: mt.get(p))
    assert [h.path for h in out] == ["new.py", "old.py"]
    same = f.apply([hit("old.py", 1.0), hit("new.py", 0.95)], lambda p: None)
    assert [h.path for h in same] == ["old.py", "new.py"]  # no mtimes: order unchanged


def test_addendum_has_instructions_description_examples_and_k() -> None:
    s = core_spec.from_dict({**RESOLVED, "version": 1, "k": 4, "exclude": ["vendor/"], "layers": {}})
    assert s is not None
    text = s.addendum()
    for needle in (
        "Prefer src/.",
        "Repository index: r",
        "add retry -> a.py, b.py",
        "Prefer these paths",
        "vendor/",
        "4 results per call",
    ):
        assert needle in text
    assert core_spec.RetrievalSpec().addendum().startswith("Semantic search returns 5")


AUTH = "def get_user_session(token):\n    return token\n"


def _ctx(spec: core_spec.RetrievalSpec | None, chat: ScriptedChat) -> ExploreContext:
    clock = FakeClock()
    return ExploreContext(
        gateway=make_fake_gateway(chat=chat, clock=clock), fs=FakeFileSystem({"src/auth.py": AUTH, "vendor/auth.py": AUTH}),
        clock=clock, status_req=StatusReq(repo_id="a" * 32, branch_id="a" * 16), backfill_to=0, spec=spec,
    )  # fmt: skip


def _report(*paths: str) -> ChatTurn:
    results = [{"path": p, "start_line": 1, "end_line": 2, "why": "w"} for p in paths]
    call = ToolCall(id="r", name="report", arguments={"results": results, "summary": "s", "gaps": []})
    return ChatTurn(content=None, tool_calls=(call,), tokens_in=1, tokens_out=1, finish="tool_calls")


async def test_explorer_prompt_carries_the_spec_and_excludes_are_dropped_from_the_report() -> None:
    spec = core_spec.RetrievalSpec(
        instructions="Prefer src/ over vendor.", description="Repository index: demo.", exclude=("vendor/",)
    )
    chat = ScriptedChat([_report("vendor/auth.py", "src/auth.py")])
    res = await explore(ExploreRequest(brief="where is get_user_session"), _ctx(spec, chat))
    system = chat.requests[0].messages[0]["content"]
    assert "## Retrieval spec" in system
    assert "Prefer src/ over vendor." in system
    assert "Repository index: demo." in system
    assert [h.path for h in res.results] == ["src/auth.py"]


async def test_no_spec_leaves_the_prompt_untouched() -> None:
    chat = ScriptedChat([_report("src/auth.py")])
    await explore(ExploreRequest(brief="where"), _ctx(None, chat))
    assert "## Retrieval spec" not in chat.requests[0].messages[0]["content"]


async def test_index_off_in_the_spec_is_grep_only_and_says_so() -> None:
    spec = core_spec.RetrievalSpec(index_enabled=False)
    chat = ScriptedChat([_report("src/auth.py")])
    res = await explore(ExploreRequest(brief="where"), _ctx(spec, chat))
    assert "semantic_search" not in [t["function"]["name"] for t in chat.requests[0].tools]
    assert any("retrieval spec turned the index off" in g for g in res.gaps)


# --- hill-climb options: handoff, report guidance, search policy, similarity-selected examples -------------
def test_handoff_targets():
    S = sp

    assert S.apply_handoff("scout:scout-explorer", "agent") == "scout:scout-explorer"
    assert S.apply_handoff("scout:scout-explorer", "verbatim") == "scout:super-explore"
    assert S.apply_handoff("scout:scout-explorer", "direct") == S.DIRECT_TOOL
    assert S.apply_handoff("Explore", "direct") == "Explore"
    assert "yourself" in S.workflow_text(S.DIRECT_TOOL, "Aim for 5.")
    assert S.workflow_text("Explore", "Aim for 5.").endswith("Aim for 5.")


def test_new_options_validated():
    S = sp

    flat, warns = S.normalize(
        {
            "delivery": {"handoff": "bogus", "search_policy": "warn", "report_guidance": "x"},
            "examples": {"select": "similar", "pool": 40},
        }
    )
    assert flat["search_policy"] == "warn"
    assert flat["select"] == "similar"
    assert "handoff" not in flat
    assert any("handoff" in w for w in warns)


def test_similar_examples_prefers_overlap():
    S = sp

    pool = [
        {"query": "Bump docs version", "files": ["README.md"]},
        {"query": "Add retry to http client", "files": ["src/http/client.py"]},
    ]
    got = S.similar_examples("make the http client retry on timeout", pool, 1)
    assert got
    assert got[0]["files"] == ["src/http/client.py"]
    assert S.similar_examples("zzz qqq", pool, 1) == []


def test_search_policy_warn_does_not_deny():
    S = sp

    res = {"delivery": {"search_policy": "warn", "search_first": False}}
    out, label, _ = S.pretool_decision({"tool_name": "Grep"}, False, res, "Explore", "mcp__x__")
    assert label == "grep_warned"
    assert "permissionDecision" not in str(out)
    res["delivery"]["search_policy"] = "deny"
    out, label, _ = S.pretool_decision({"tool_name": "Grep"}, False, res, "Explore", "mcp__x__")
    assert label == "grep_denied"
