# ruff: noqa: S603, S607, E501  (fixed argv in tests; long fixture lines)
"""The retrieval spec on Muse Code (the retrieval spec, not in this release): one shared implementation, delivered through
Muse's SessionStart context, UserPromptSubmit workflow step, PreToolUse search-first policy and the explorer
agent description. Hooks run as subprocesses with a scratch HOME and plugin data dir (no network, no Muse)."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CC = REPO / "integrations" / "claude-code"
BUNDLE = REPO / "integrations" / "muse-code" / "scout-wayfinder"
HOOKS = BUNDLE / "hooks"
EXPLORER = "scout-wayfinder/scout-explorer"
GIT_ENV = {
    "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def shared_spec():  # noqa: ANN201
    """The Claude Code plugin's scout_spec module, by path (the Muse copy is checked equal in another test)."""
    s = importlib.util.spec_from_file_location("cc_scout_spec", CC / "lib" / "scout_spec.py")
    assert s
    assert s.loader
    mod = importlib.util.module_from_spec(s)
    sys.modules["cc_scout_spec"] = mod
    s.loader.exec_module(mod)
    return mod


def git(cwd: Path, *args: str) -> str:
    env = {
        **GIT_ENV,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


def make_repo(path: Path, spec: str | None = None) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    for i, (rel, subject) in enumerate(
        [
            ("src/app.py", "add the app entry point"),
            ("src/http/client.py", "add retry to the http client"),
            ("tests/test_client.py", "test the http client"),
        ]
    ):
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"x = {i}\n")
        git(path, "add", rel)
        git(path, "commit", "-q", "-m", subject)
    if spec is not None:
        (path / ".scout").mkdir()
        (path / ".scout" / "spec.toml").write_text(spec)
    return path


def scratch(tmp_path: Path, repo: Path, **extra: str) -> dict[str, str]:
    """A ready demo-lane registry for `repo`, so the explorer target is scout-explorer."""
    home = tmp_path / "demohome"
    home.mkdir(parents=True, exist_ok=True)
    (home / "repos.json").write_text(
        json.dumps({"tiny": {"root": str(repo), "state": "ready", "coverage": 1.0}})
    )
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(tmp_path / "home"),
        "PATH": GIT_ENV["PATH"],
        "MUSE_PLUGIN_DATA_DIR": str(tmp_path / "data"),
        "SCOUT_LANE": "demo",
        "SCOUT_DEMO_HOME": str(home),
        "SCOUT_NO_WARM": "1",
        "MUSE_SESSION_ID": "s1",
        "MUSE_THREAD_ID": "s1",
        **extra,
    }


def hook(name: str, env: dict[str, str], payload: dict) -> dict | None:
    proc = subprocess.run(
        [sys.executable, str(HOOKS / name)],
        env=env,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def session(env: dict, repo: Path) -> dict | None:
    return hook("spec_session.py", env, {"hook_event_name": "SessionStart", "cwd": str(repo)})


def prompt(env: dict, repo: Path, text: str = "Where is the retry logic?") -> dict | None:
    return hook(
        "spec_prompt.py", env, {"hook_event_name": "UserPromptSubmit", "prompt": text, "cwd": str(repo)}
    )


def pretool(env: dict, repo: Path, tool: str, **tool_input: object) -> dict | None:
    return hook(
        "spec_pretool.py",
        env,
        {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input, "cwd": str(repo)},
    )


def ctx(out: dict | None) -> str:
    assert out is not None
    return out["hookSpecificOutput"]["additionalContext"]


SPEC = """instructions = "Prefer src/ over vendored code."
[delivery]
search_first = true
[examples]
mine = 2
"""


# --- one implementation, two hosts --------------------------------------------------------------------
def test_the_muse_plugin_carries_a_verbatim_copy_of_the_shared_spec_code() -> None:
    for src, dst in (
        (CC / "lib" / "scout_spec.py", BUNDLE / "lib" / "scout_spec.py"),
        (CC / "spec" / "harness_defaults.toml", BUNDLE / "spec" / "harness_defaults.toml"),
    ):
        assert dst.read_bytes() == src.read_bytes(), f"{dst.name} drifted: run scripts/sync_muse_spec.py"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "sync_muse_spec.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout


def test_agent_frontmatter_is_strict_yaml_and_describes_the_spec_driven_delegation() -> None:
    text = (BUNDLE / "agents" / "scout-explorer.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m
    fm = yaml.safe_load(
        m.group(1)
    )  # Muse's parser is strict: a stray ": " in the description drops the agent
    assert fm["name"] == "scout-explorer"
    assert "Delegate to it FIRST" in fm["description"]
    assert "retrieval spec" in fm["description"]
    assert "Retrieval spec." in text


def test_claude_code_wording_is_unchanged_by_the_host_table() -> None:
    sp = shared_spec()

    assert (
        sp.workflow_text("scout:scout-explorer")
        == "Before reading or editing code, delegate to the `scout:scout-explorer` subagent with a description of what to find; work from the files it returns."
    )
    assert (
        sp.deny_text("Explore")
        == "Map the code first with the `Explore` subagent; Grep and Glob are for exact identifiers you already know."
    )
    assert sp.apply_handoff("scout:scout-explorer", "verbatim") == "scout:super-explore"
    assert sp.apply_handoff("scout:scout-explorer", "direct") == sp.DIRECT_TOOL


# --- SessionStart context ----------------------------------------------------------------------------
def test_session_start_delivers_instructions_description_and_mined_examples(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", SPEC)
    env = scratch(tmp_path, repo)
    text = ctx(session(env, repo))
    assert "Prefer src/ over vendored code." in text  # user layer, last
    assert "Prefer the project's own source" in text  # harness layer, first
    assert "src" in text  # index description
    assert "add retry to the http client -> src/http/client.py" in text  # mined example
    resolved = json.loads((repo / ".git" / "scout" / "spec_resolved.json").read_text())
    assert resolved["delivery"]["search_first"] is True  # the file the scout core reads
    assert resolved["layers"]["user"]
    log = [
        json.loads(x)
        for x in (Path(env["MUSE_PLUGIN_DATA_DIR"]) / "spec_hooks.jsonl").read_text().splitlines()
    ]
    assert log[0]["target"] == EXPLORER
    assert "retry" not in json.dumps(log)  # decisions only


def test_session_start_is_silent_outside_a_repo_and_when_disabled(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", SPEC)
    env = scratch(tmp_path, repo)
    plain = tmp_path / "plain"
    plain.mkdir()
    assert session(env, plain) is None
    assert session({**env, "SCOUT_SPEC_DISABLE": "1"}, repo) is None
    assert prompt({**env, "SCOUT_SPEC_DISABLE": "1"}, repo) is None
    assert not (repo / ".git" / "scout" / "spec_resolved.json").exists()


def test_scout_env_file_carries_the_spec_settings_muse_strips_from_hook_environments(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", 'instructions = "in repo"\n')
    other = tmp_path / "other.toml"
    other.write_text('instructions = "from the pinned file"\n')
    env = scratch(tmp_path, repo)
    (Path(env["MUSE_PLUGIN_DATA_DIR"]) / "scout.env").write_text(
        f"SCOUT_SPEC_FILE={other}\nSCOUT_SPEC_TARGET=general-purpose\n"
    )
    assert "from the pinned file" in ctx(session(env, repo))
    assert "in repo" not in ctx(session(env, repo))
    assert "`general-purpose` subagent" in ctx(prompt(env, repo))


# --- workflow step ------------------------------------------------------------------------------------
def test_workflow_step_names_the_explorer_and_muse_spawn_tools(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", SPEC)
    env = scratch(tmp_path, repo)
    session(env, repo)
    text = ctx(prompt(env, repo))
    assert f"delegate to the `{EXPLORER}` subagent" in text
    assert "subagent_spawn" in text
    assert "subagent_wait" in text
    assert "timeout_ms 300000" in text
    assert prompt(env, repo, "<task-notification>done") is None


def test_workflow_step_falls_back_to_the_built_in_agent_when_scout_cannot_answer(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", SPEC)
    env = scratch(tmp_path, repo, SCOUT_DEMO_HOME=str(tmp_path / "empty"))  # nothing indexed
    text = ctx(prompt(env, repo))
    assert "`general-purpose` subagent" in text
    assert EXPLORER not in text
    off = make_repo(tmp_path / "off", "[index]\nenabled = false\n")
    assert "`general-purpose`" in ctx(prompt(scratch(tmp_path / "x", off), off))


def test_handoff_direct_points_the_parent_at_the_scout_tool_and_verbatim_keeps_the_agent(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path / "d", '[delivery]\nhandoff = "direct"\n')
    text = ctx(prompt(scratch(tmp_path / "a", repo), repo))
    assert "call the `mcp__plugin_scout_wayfinder_scout__explore` tool yourself" in text
    repo2 = make_repo(
        tmp_path / "v", '[delivery]\nhandoff = "verbatim"\nreport_guidance = "List 5 to 8 files."\n'
    )
    text2 = ctx(prompt(scratch(tmp_path / "b", repo2), repo2))  # Muse has no relay subagent
    assert EXPLORER in text2
    assert text2.endswith("List 5 to 8 files.")


# --- search-first policy -------------------------------------------------------------------------------
def test_search_first_denies_main_search_until_delegation_then_allows(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", SPEC)
    env = scratch(tmp_path, repo)
    session(env, repo)
    first = pretool(env, repo, "search", pattern="retry")
    assert first is not None
    out = first["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert EXPLORER in out["permissionDecisionReason"]
    assert "subagent_spawn" in out["permissionDecisionReason"]
    assert pretool(env, repo, "search", pattern="retry") is not None  # still denied: nothing delegated yet
    assert (
        pretool(env, repo, "subagent_spawn", objective="where is retry", subagent_type=EXPLORER) is None
    )  # maps the session
    assert pretool(env, repo, "search", pattern="retry") is None
    other = {**env, "MUSE_SESSION_ID": "s2", "MUSE_THREAD_ID": "s2"}
    assert pretool(other, repo, "search", pattern="retry") is not None  # per session


def test_subagent_searches_are_never_gated_and_do_not_map_the_session(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", SPEC)
    env = scratch(tmp_path, repo)
    session(env, repo)
    child = {**env, "MUSE_THREAD_ID": "t-child"}
    assert pretool(child, repo, "search", pattern="retry") is None
    assert (
        pretool(child, repo, "subagent_spawn", objective="x") is None
    )  # a child's spawn is not the parent's delegation
    assert pretool(env, repo, "search", pattern="retry") is not None


def test_a_main_thread_scout_call_maps_the_session(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", SPEC)
    env = scratch(tmp_path, repo)
    session(env, repo)
    assert pretool(env, repo, "mcp__plugin_scout_wayfinder_scout.explore", brief="x") is None
    assert pretool(env, repo, "search", pattern="retry") is None


@pytest.mark.parametrize(
    ("policy", "expect"),
    [("warn", "additionalContext"), ("off", None), ("deny", "permissionDecision")],
)
def test_search_policy_modes(tmp_path: Path, policy: str, expect: str | None) -> None:
    repo = make_repo(tmp_path / "repo", f'[delivery]\nsearch_policy = "{policy}"\n')
    env = scratch(tmp_path, repo)
    session(env, repo)
    out = pretool(env, repo, "search", pattern="x")
    if expect is None:
        assert out is None
    else:
        assert out is not None
        assert expect in out["hookSpecificOutput"]
        assert "permissionDecision" not in out["hookSpecificOutput"] or policy == "deny"


def test_search_first_is_off_by_default_and_other_tools_pass(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")  # no .scout/spec.toml: harness defaults only
    env = scratch(tmp_path, repo)
    session(env, repo)
    assert pretool(env, repo, "search", pattern="x") is None
    on = make_repo(tmp_path / "on", SPEC)
    env2 = scratch(tmp_path / "e2", on)
    session(env2, on)
    assert pretool(env2, on, "read_file", path="src/app.py") is None


# --- how the spec and the older one-shot guard share the search tool -----------------------------------
def route(env: dict, repo: Path, tool: str = "search") -> dict | None:
    return hook(
        "route_explore.py", env, {"tool_name": tool, "tool_input": {"pattern": "x"}, "cwd": str(repo)}
    )


def test_route_guard_defers_to_a_spec_policy_and_keeps_working_without_one(tmp_path: Path) -> None:
    plain = make_repo(tmp_path / "plain")
    env = scratch(tmp_path / "p", plain)
    session(env, plain)  # resolves the harness-default spec: no search policy set
    assert route(env, plain) is not None  # the one-shot guard still denies the first search
    spec = make_repo(tmp_path / "spec", SPEC)
    env2 = scratch(tmp_path / "s", spec)
    session(env2, spec)
    assert route(env2, spec) is None  # spec_pretool owns it now: no double deny
    assert pretool(env2, spec, "search", pattern="x") is not None
    disabled = {**env2, "SCOUT_SPEC_DISABLE": "1"}
    assert route(disabled, spec) is not None  # disabling the spec gives the guard back


def test_direct_handoff_lets_the_main_thread_call_scout(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", '[delivery]\nhandoff = "direct"\nsearch_policy = "warn"\n')
    env = scratch(tmp_path, repo)
    session(env, repo)
    assert route(env, repo, "mcp__plugin_scout_wayfinder_scout.explore") is None
    agent = make_repo(tmp_path / "agent", SPEC)
    env2 = scratch(tmp_path / "e2", agent)
    session(env2, agent)
    denied = route(env2, agent, "mcp__plugin_scout_wayfinder_scout.explore")
    assert denied is not None
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_rewrite_explore_false_stops_the_spawn_rewrite(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", "[delivery]\nrewrite_explore = false\n")
    env = scratch(tmp_path, repo)
    session(env, repo)
    payload = {
        "tool_name": "subagent_spawn",
        "tool_input": {"objective": "Where does the client retry?", "role": "explorer"},
        "cwd": str(repo),
    }
    assert hook("route_explore.py", env, payload) is None
    repo2 = make_repo(tmp_path / "r2", "")
    env2 = scratch(tmp_path / "e2", repo2)
    session(env2, repo2)
    payload["cwd"] = str(repo2)
    out = hook("route_explore.py", env2, payload)
    assert out is not None
    assert out["hookSpecificOutput"]["updatedInput"]["subagent_type"] == EXPLORER


# --- a broken spec never blocks a session --------------------------------------------------------------
def test_invalid_spec_is_reported_not_raised(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo", "instructions = [oops\n")
    env = scratch(tmp_path, repo)
    text = ctx(session(env, repo))
    assert "Prefer the project's own source" in text  # the lower layers still apply
    assert pretool(env, repo, "search", pattern="x") is None


def test_background_option_lets_the_parent_work_while_the_explorer_runs(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "bg", "[delivery]\nbackground = true\n")
    env = scratch(tmp_path / "e", repo)
    text = ctx(prompt(env, repo))
    assert "Do not wait right away" in text
    assert "subagent_wait only when you need its list" in text
    plain = make_repo(tmp_path / "plain", "")
    assert "Do not wait right away" not in ctx(prompt(scratch(tmp_path / "e2", plain), plain))
    cc = shared_spec().workflow_text("scout:scout-explorer", "", "claude-code", True)
    assert "Do not wait" not in cc  # Claude Code's Agent tool has its own background flag; unchanged
