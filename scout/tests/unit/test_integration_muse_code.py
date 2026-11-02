"""integrations/muse-code: the scout-wayfinder native Muse plugin bundle.

Static checks of the manifest, Agent Definition, skill, commands and marketplace; the lane logic, MCP
launcher and both hooks as subprocesses with a scratch HOME (no network, no Muse); `scout init --muse-code`
against a stub `muse`; and, when a real `muse` binary is installed, `muse plugins validate` plus
`muse plugins hook test` on every fixture inside an isolated XDG_CONFIG_HOME / XDG_DATA_HOME.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOST = REPO / "integrations" / "muse-code"
BUNDLE = HOST / "scout-wayfinder"
MANIFEST = BUNDLE / ".muse-plugin" / "plugin.json"
EXPLORER = "scout-wayfinder/scout-explorer"
ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
AGENT_NAME = re.compile(r"^[a-z]+(?:-[a-z]+)*$")
FAMILIES = ("skills", "commands", "hooks", "mcpServers", "reminders")
MUSE = shutil.which("muse") or str(Path.home() / ".local" / "bin" / "muse")
HAVE_MUSE = os.access(MUSE, os.X_OK)


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def frontmatter(path: Path) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", path.read_text(encoding="utf-8"), re.S)
    assert m, f"{path.name}: no frontmatter"
    return dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)


# --- static ---------------------------------------------------------------------------------------
def test_manifest_is_native_with_every_family_and_valid_ids() -> None:
    doc = manifest()
    assert doc["schemaVersion"] == 1
    assert doc["name"] == "scout-wayfinder"
    assert doc["compat"] == {"source": "native", "manifestDir": ".muse-plugin"}
    assert set(doc["capabilities"]) == set(FAMILIES)  # `agents` is not a manifest family; see agents/
    for family in FAMILIES:
        for cap in doc["capabilities"][family]:
            assert ID.match(cap["id"]), cap["id"]


def test_every_referenced_file_exists_and_hooks_have_no_matcher() -> None:
    caps = manifest()["capabilities"]
    for family in ("skills", "commands"):
        for cap in caps[family]:
            assert (BUNDLE / cap["path"]).is_file(), cap["path"]
    for cap in [*caps["hooks"], *caps["mcpServers"]]:
        assert "matcher" not in cap  # the native validator rejects `matcher` (Claude/Codex field)
        assert (BUNDLE / cap["command"][1]).is_file(), cap["command"]
    events = {h["id"]: h["event"] for h in caps["hooks"]}
    assert events == {
        "session-start": "SessionStart",
        "spec-session": "SessionStart",
        "route-explore": "PreToolUse",
        "spec-pretool": "PreToolUse",
        "spec-prompt": "UserPromptSubmit",
    }
    assert [m["id"] for m in caps["mcpServers"]] == ["scout"]


def test_agent_definition_is_the_instructed_explorer() -> None:
    fm = frontmatter(BUNDLE / "agents" / "scout-explorer.md")
    assert AGENT_NAME.match(fm["name"])
    assert fm["name"] == "scout-explorer"
    assert fm["model"].startswith("muse-spark-")
    assert "tools" not in fm  # a deny list keeps read/search and the scout MCP tools without naming them
    denied = {t.strip() for t in fm["disallowedTools"].split(",")}
    assert {"write_file", "edit_file", "bash"} <= denied
    assert "subagent_spawn" not in denied  # a session control, not a Work tool: the spawn is rejected
    assert "image_generation" not in denied  # absent in some runtimes (--disable-shell): unknown_tool
    body = (BUNDLE / "agents" / "scout-explorer.md").read_text(encoding="utf-8")
    for needle in ("`explore` FIRST", "read_file", "At most 12 entries", "(unchecked)", "scout: used"):
        assert needle in body, needle


def test_skill_and_commands_have_frontmatter() -> None:
    fm = frontmatter(BUNDLE / "skills" / "scout-explore" / "SKILL.md")
    assert fm["name"] == "scout-explore"
    assert "description" in fm
    assert EXPLORER in (BUNDLE / "skills" / "scout-explore" / "SKILL.md").read_text(encoding="utf-8")
    for cmd in manifest()["capabilities"]["commands"]:
        assert "description" in frontmatter(BUNDLE / cmd["path"]), cmd["id"]
    assert "$ARGUMENTS" in (BUNDLE / "commands" / "find.md").read_text(encoding="utf-8")


def test_marketplace_points_at_the_bundle() -> None:
    doc = json.loads((HOST / ".agents" / "plugins" / "marketplace.json").read_text(encoding="utf-8"))
    (entry,) = doc["plugins"]
    assert entry["name"] == "scout-wayfinder"
    assert entry["source"] == {"source": "local", "path": "./scout-wayfinder"}


def test_host_file_installs_through_the_idempotent_installer() -> None:
    doc = json.loads((HOST / "scout-host.json").read_text(encoding="utf-8"))
    assert doc["version_command"] == ["muse", "--version"]
    (step,) = doc["install_steps"]
    assert step[1] == "{integrations}/muse-code/install.py"


# --- lane logic and launcher ----------------------------------------------------------------------
def git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)  # noqa: S603, S607
    return path


def scratch_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    data = tmp_path / "plugin-data"
    data.mkdir(exist_ok=True)
    return {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "MUSE_PLUGIN_DATA_DIR": str(data),
        "SCOUT_KEYSTORE": "file",
        **extra,
    }


def stub(path: Path, text: str = '#!/bin/sh\necho "$0 $@"\n') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def run_py(script: Path, env: dict[str, str], *args: str, stdin: str = "", cwd: Path | None = None) -> str:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(script), *args],
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        cwd=cwd,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def dry_run(tmp_path: Path, env: dict[str, str]) -> dict:
    return json.loads(run_py(BUNDLE / "bin" / "launch_mcp.py", env, "--dry-run", cwd=tmp_path))


def test_launcher_uses_the_production_lane_with_a_key_file_and_never_prints_it(tmp_path: Path) -> None:
    scout = stub(tmp_path / "bin" / "scout")
    key = tmp_path / "key.txt"
    key.write_text("sk_scout_secret_value\n")
    env = scratch_env(tmp_path)
    (Path(env["MUSE_PLUGIN_DATA_DIR"]) / "scout.env").write_text(f"SCOUT_BIN={scout}\nSCOUT_KEY_FILE={key}\n")
    out = dry_run(tmp_path, env)
    assert out["lane"] == "production"
    assert out["argv"][0] == str(scout)
    assert out["argv"][1:3] == ["mcp", "--tools"]
    assert out["has_key"] is True
    assert "sk_scout_secret_value" not in json.dumps(out)


def test_launcher_falls_back_to_the_demo_lane_without_a_key(tmp_path: Path) -> None:
    project = tmp_path / "project"
    demo = stub(project / ".venv" / "bin" / "scout-demo")
    env = scratch_env(tmp_path)
    (Path(env["MUSE_PLUGIN_DATA_DIR"]) / "scout.env").write_text(
        f"SCOUT_BIN={stub(tmp_path / 'bin' / 'scout')}\nSCOUT_DEMO_PROJECT={project}\n"
    )
    assert dry_run(tmp_path, env) == {"lane": "demo", "argv": [str(demo), "mcp"], "has_key": False}


def test_config_file_beats_the_environment_and_ignores_unknown_names(tmp_path: Path) -> None:
    project = tmp_path / "project"
    stub(project / ".venv" / "bin" / "scout-demo")
    scout = stub(tmp_path / "bin" / "scout")
    env = scratch_env(tmp_path, SCOUT_LANE="production", SCOUT_BIN=str(scout))
    (Path(env["MUSE_PLUGIN_DATA_DIR"]) / "scout.env").write_text(
        f"SCOUT_LANE=demo\nSCOUT_DEMO_PROJECT={project}\nEVIL=1\n"
    )
    assert dry_run(tmp_path, env)["lane"] == "demo"


def test_scout_login_credentials_select_production(tmp_path: Path) -> None:
    scout = stub(tmp_path / "bin" / "scout")
    env = scratch_env(tmp_path, SCOUT_BIN=str(scout))
    creds = Path(env["HOME"]) / ".config" / "scout" / "credentials"
    creds.parent.mkdir(parents=True)
    creds.write_text("{}")
    out = dry_run(tmp_path, env)
    assert (out["lane"], out["has_key"]) == ("production", False)  # the key stays in scout's own store


# --- hooks ----------------------------------------------------------------------------------------
def demo_ready(tmp_path: Path, repo: Path) -> dict[str, str]:
    """A demo registry (scratch SCOUT_DEMO_HOME) where `repo` is indexed and ready."""
    home = tmp_path / "demohome"
    home.mkdir(exist_ok=True)
    (home / "repos.json").write_text(
        json.dumps({"tiny": {"root": str(repo), "state": "ready", "coverage": 1.0}})
    )
    env = scratch_env(tmp_path, SCOUT_LANE="demo", SCOUT_DEMO_HOME=str(home), SCOUT_NO_WARM="1")
    return env


def spawn_payload(repo: Path, objective: str, **extra: object) -> str:
    tool_input = {"command_id": "c1", "objective": objective, "role": "explorer", **extra}
    return json.dumps(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "subagent_spawn",
            "tool_input": tool_input,
            "cwd": str(repo),
        }
    )


def route(env: dict[str, str], payload: str) -> dict | None:
    out = run_py(BUNDLE / "hooks" / "route_explore.py", env, stdin=payload)
    return json.loads(out) if out else None


def test_route_rewrites_an_exploration_spawn_when_scout_is_ready(tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "repo")
    env = demo_ready(tmp_path, repo)
    out = route(env, spawn_payload(repo, "Where does the router produce a 405 response?"))
    assert out is not None
    assert out["hookSpecificOutput"]["updatedInput"]["subagent_type"] == EXPLORER
    assert out["hookSpecificOutput"]["updatedInput"]["objective"].startswith("Where does")
    log = (Path(env["MUSE_PLUGIN_DATA_DIR"]) / "route.log").read_text()
    assert "spawn rewrite lane=demo" in log
    assert "405" not in log  # decisions only, never the objective


@pytest.mark.parametrize(
    ("objective", "extra"),
    [
        ("Fix the 405 handler and add a test", {}),  # an edit task
        ("Where is the 405 handler?", {"subagent_type": "general-purpose-x"}),  # an explicit other agent
        ("Where is the 405 handler?", {"worktree_isolation": True}),  # an isolated child may write
    ],
)
def test_route_leaves_other_spawns_alone(tmp_path: Path, objective: str, extra: dict) -> None:
    repo = git_repo(tmp_path / "repo")
    assert route(demo_ready(tmp_path, repo), spawn_payload(repo, objective, **extra)) is None


def test_route_does_nothing_when_scout_is_not_ready(tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "repo")
    env = scratch_env(tmp_path, SCOUT_LANE="demo", SCOUT_DEMO_HOME=str(tmp_path / "empty"))
    assert route(env, spawn_payload(repo, "Where is the 405 handler?")) is None


def search_payload(repo: Path) -> str:
    return json.dumps({"tool_name": "search", "tool_input": {"pattern": "x"}, "cwd": str(repo)})


def test_search_guard_denies_the_first_main_thread_search_once(tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "repo")
    env = demo_ready(tmp_path, repo)
    child = {**env, "MUSE_SESSION_ID": "s1", "MUSE_THREAD_ID": "t-child"}
    assert route(child, search_payload(repo)) is None  # scout-explorer's own gap-filling searches pass
    main = {**env, "MUSE_SESSION_ID": "s1", "MUSE_THREAD_ID": "s1"}
    first = route(main, search_payload(repo))
    assert first is not None
    out = first["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert EXPLORER in out["permissionDecisionReason"]
    assert route(main, search_payload(repo)) is None  # a repeated search runs
    other = {**env, "MUSE_SESSION_ID": "s2", "MUSE_THREAD_ID": "s2"}
    assert route(other, search_payload(repo)) is not None  # per session


def test_search_guard_is_off_after_a_delegation(tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "repo")
    main = {**demo_ready(tmp_path, repo), "MUSE_SESSION_ID": "s1", "MUSE_THREAD_ID": "s1"}
    route(main, spawn_payload(repo, "Where is the 405 handler?", subagent_type=EXPLORER))
    assert route(main, search_payload(repo)) is None


@pytest.mark.parametrize(("mode", "expect"), [("nudge", "additionalContext"), ("off", None)])
def test_search_route_modes(tmp_path: Path, mode: str, expect: str | None) -> None:
    repo = git_repo(tmp_path / "repo")
    env = {**demo_ready(tmp_path, repo), "MUSE_SESSION_ID": "s1", "MUSE_THREAD_ID": "s1", "SCOUT_ROUTE": mode}
    out = route(env, search_payload(repo))
    if expect is None:
        assert out is None
    else:
        assert out is not None
        assert expect in out["hookSpecificOutput"]


@pytest.mark.parametrize(
    "tool", ["mcp__plugin_scout_wayfinder_scout.explore", "mcp__plugin_scout-wayfinder_scout__explore"]
)
def test_direct_scout_explore_is_sent_to_the_subagent_on_the_main_thread_only(
    tmp_path: Path, tool: str
) -> None:
    env = scratch_env(tmp_path)
    payload = json.dumps({"tool_name": tool, "tool_input": {"brief": "where is x"}, "cwd": str(tmp_path)})
    main = route({**env, "MUSE_SESSION_ID": "s1", "MUSE_THREAD_ID": "s1"}, payload)
    assert main is not None
    assert main["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert EXPLORER in main["hookSpecificOutput"]["permissionDecisionReason"]
    assert route({**env, "MUSE_SESSION_ID": "s1", "MUSE_THREAD_ID": "explorer"}, payload) is None
    status = json.dumps({"tool_name": tool.replace("explore", "index_status"), "tool_input": {}})
    assert route({**env, "MUSE_SESSION_ID": "s1", "MUSE_THREAD_ID": "s1"}, status) is None


def test_route_ignores_other_tools_and_bad_input(tmp_path: Path) -> None:
    env = scratch_env(tmp_path)
    assert route(env, json.dumps({"tool_name": "read_file", "tool_input": {"path": "x"}})) is None
    assert route(env, "not json") is None


def test_session_start_reports_readiness_and_rate_limits_the_warm_up(tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "repo")
    env = {**demo_ready(tmp_path, repo)}
    env.pop("SCOUT_NO_WARM")
    project = tmp_path / "project"
    stub(project / ".venv" / "bin" / "scout-demo", '#!/bin/sh\necho "{\\"warm\\": true}"\n')
    (Path(env["MUSE_PLUGIN_DATA_DIR"]) / "scout.env").write_text(f"SCOUT_DEMO_PROJECT={project}\n")
    payload = json.dumps({"hook_event_name": "SessionStart", "cwd": str(repo)})
    out = json.loads(run_py(BUNDLE / "hooks" / "session_start.py", env, stdin=payload))
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert ctx.startswith("scout: index ready")
    assert "demo repo `tiny`" in ctx
    assert EXPLORER in ctx
    stamps = list(Path(env["MUSE_PLUGIN_DATA_DIR"]).glob("warm-*.stamp"))
    assert len(stamps) == 1
    first = stamps[0].read_text()
    run_py(BUNDLE / "hooks" / "session_start.py", env, stdin=payload)
    assert stamps[0].read_text() == first  # second session within 5 minutes: no new warm-up


def test_session_start_outside_a_repo_says_not_ready(tmp_path: Path) -> None:
    env = scratch_env(tmp_path, SCOUT_NO_WARM="1")
    payload = json.dumps({"hook_event_name": "SessionStart", "cwd": str(tmp_path)})
    out = json.loads(run_py(BUNDLE / "hooks" / "session_start.py", env, stdin=payload, cwd=tmp_path))
    assert out["hookSpecificOutput"]["additionalContext"].startswith("scout: not ready")


def test_warm_worker_runs_watch_and_warm_in_production(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    scout = stub(tmp_path / "bin" / "scout", f'#!/bin/sh\necho "$@" >> {log}\n')
    key = tmp_path / "key"
    key.write_text("sk_x\n")
    env = scratch_env(tmp_path)
    (Path(env["MUSE_PLUGIN_DATA_DIR"]) / "scout.env").write_text(f"SCOUT_BIN={scout}\nSCOUT_KEY_FILE={key}\n")
    repo = git_repo(tmp_path / "repo")
    out = run_py(BUNDLE / "hooks" / "warm.py", env, str(repo))
    assert "lane=production watch rc=spawned" in out
    assert "lane=production warm rc=0" in out
    calls = log.read_text()
    assert f"warm --trigger --root {repo}" in calls


# --- scout init --muse-code -----------------------------------------------------------------------
def test_init_muse_code_dry_run_lists_the_install_step(tmp_path: Path) -> None:
    stub(tmp_path / "bin" / "muse", "#!/bin/sh\necho 1.4.14\n")
    env = {**os.environ, "HOME": str(tmp_path), "SCOUT_INTEGRATIONS_DIR": str(REPO / "integrations")}
    env["PATH"] = f"{tmp_path / 'bin'}:/usr/bin:/bin"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", "init", "--muse-code", "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert (out["host"], out["version"], out["ok"]) == ("muse-code", "1.4.14", True)
    assert out["install"]["ran"] is False
    assert out["install"]["steps"][0]["argv"][1].endswith("integrations/muse-code/install.py")


def test_installer_against_a_stub_muse_writes_the_plugin_config(tmp_path: Path) -> None:
    calls = tmp_path / "calls.log"
    muse = stub(
        tmp_path / "bin" / "muse",
        f'#!/bin/sh\necho "$@" >> {calls}\ncase "$2" in inspect) echo \'{{"runtime_capabilities": []}}\';; '
        "*) echo '{}';; esac\n",
    )
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "XDG_DATA_HOME": str(tmp_path / "data")}
    out = json.loads(run_py(HOST / "install.py", env, "--muse", str(muse), "--lane", "demo"))
    assert out["ok"] is True
    assert [s["step"] for s in out["steps"]] == ["marketplace add", "install", "approve"]
    cfg = tmp_path / "data" / "muse" / "plugins" / "data" / "scout-wayfinder" / "scout.env"
    assert "SCOUT_LANE=demo" in cfg.read_text()
    assert oct(cfg.stat().st_mode & 0o777) == "0o600"
    assert "plugins approve scout-wayfinder --json" in calls.read_text()


# --- the real Muse CLI, isolated ------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_MUSE, reason="muse not installed")
def test_muse_validates_installs_approves_and_runs_every_hook_fixture(tmp_path: Path) -> None:
    env = {
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }

    def muse(*args: str) -> dict:
        proc = subprocess.run(  # noqa: S603
            [MUSE, "plugins", *args, "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-500:]
        return json.loads(proc.stdout)

    val = muse("validate", str(BUNDLE))
    assert (val["valid"], val["diagnostics"]) == (True, [])
    muse("marketplace", "add", "voyage-scout", str(HOST))
    muse("install", "scout-wayfinder@voyage-scout")
    approved = muse("approve", "scout-wayfinder")["runtime_capabilities"]
    ids = {c.get("stable_id") or c.get("scoped_definition_id") for c in approved}
    assert ids == {
        "plugin:scout-wayfinder:hook:session-start",
        "plugin:scout-wayfinder:hook:route-explore",
        "plugin:scout-wayfinder:hook:spec-session",
        "plugin:scout-wayfinder:hook:spec-prompt",
        "plugin:scout-wayfinder:hook:spec-pretool",
        "plugin:scout-wayfinder:mcp_server:scout",
        EXPLORER,
    }
    status = {
        c["candidate"].get("stable_id") or c["candidate"].get("scoped_definition_id"): c["status"]
        for c in muse("inspect", "scout-wayfinder")["runtime_capabilities"]
    }
    assert set(status.values()) == {"trusted_enabled"}
    for fixture in sorted((BUNDLE / "hooks" / "fixtures").glob("*.json")):
        hook = {
            "session": "session-start",
            "spec_session": "spec-session",
            "spec_prompt": "spec-prompt",
            "spec_pretool": "spec-pretool",
        }.get(
            fixture.stem.split("_", 1)[0] if not fixture.stem.startswith("spec") else fixture.stem,
            "route-explore",
        )
        res = muse("hook", "test", f"scout-wayfinder:{hook}", "--fixture", str(fixture))
        (term,) = res["terminals"]
        assert (term["status"], term["exit_code"]) == ("completed", 0), fixture.name
        assert res["decision"]["should_block"] is False
        assert res["decision"]["updated_input"] is None  # "/" is no repo: nothing is routed


# --- scout warm (what the SessionStart worker calls) ---
def test_scout_warm_sends_one_tiny_completion_on_the_fake_profile(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "SCOUT_KEYSTORE": "file",
        "SCOUT_PROFILE": "fake",
        "SCOUT_FAKE_CASE": str(REPO / "tests" / "fixtures" / "poc_explore_case.json"),
    }
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", "warm"], env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["warm"] is True
    assert out["seconds"] >= 0


def _warm_app(gateway_warm: bool | None):  # noqa: ANN202
    from scout.adapters.fake_chat import ScriptedChat
    from scout.adapters.fake_clock import FakeClock
    from scout.adapters.fake_fs import FakeFileSystem
    from scout.adapters.fake_gateway import make_fake_gateway
    from scout.core.app import ScoutApp
    from scout.core.ports.chat import ChatTurn

    chat = ScriptedChat([ChatTurn(content="p", tool_calls=(), tokens_in=1, tokens_out=1, finish="length")])
    clock = FakeClock()
    gw = make_fake_gateway(chat=chat, clock=clock)
    if gateway_warm is not None:

        async def warm() -> bool:
            return gateway_warm

        gw.warm = warm  # type: ignore[attr-defined]
    return ScoutApp(gw, FakeFileSystem({}), clock), chat


def _run(coro):  # noqa: ANN001, ANN202
    import asyncio

    async def go():  # noqa: ANN202
        return await coro

    return asyncio.run(go())


def test_app_warm_default_is_one_blocking_one_token_completion() -> None:
    app, chat = _warm_app(True)
    out = _run(app.warm())
    assert (out["mode"], out["warm"]) == ("completion", True)
    (req,) = chat.requests
    assert (req.max_tokens, req.thinking, req.tools) == (1, False, [])


def test_app_warm_trigger_uses_the_gateway_endpoint_and_sends_no_chat() -> None:
    app, chat = _warm_app(True)
    out = _run(app.warm(trigger=True))
    assert (out["mode"], out["warm"], out["warming"]) == ("endpoint", False, True)
    assert chat.requests == []


@pytest.mark.parametrize("gateway_warm", [False, None])
def test_app_warm_trigger_falls_back_to_a_completion(gateway_warm: bool | None) -> None:
    app, chat = _warm_app(gateway_warm)
    assert _run(app.warm(trigger=True))["mode"] == "completion"
    assert len(chat.requests) == 1


def test_warm_url_is_derived_from_a_modal_model_url() -> None:
    from scout.adapters.openai_chat.client import OpenAIChat, warm_url_for

    v3 = "https://myworkspace--wayfinder-v3-serve.us-east.modal.run/v1"
    assert warm_url_for(v3) == "https://myworkspace--wayfinder-v3-serve-warm.us-east.modal.run/"
    assert warm_url_for("https://api.example.com/v1") is None
    env = {"SCOUT_LLM_URL": "https://api.example.com/v1", "SCOUT_LLM_WARM_URL": "https://w.example/"}
    assert OpenAIChat.from_env(env).warm_url == "https://w.example/"


def test_openai_chat_warm_is_one_authorised_get() -> None:
    import httpx

    from scout.adapters.openai_chat.client import OpenAIChat

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"warming": True})

    chat = OpenAIChat(
        "https://ws--m.us-east.modal.run/v1", "k-secret", "m", transport=httpx.MockTransport(handler)
    )
    assert _run(chat.warm()) is True
    (req,) = seen
    assert (req.method, str(req.url)) == ("GET", "https://ws--m-warm.us-east.modal.run/")
    assert req.headers["authorization"] == "Bearer k-secret"


def test_http_gateway_warm_is_false_on_an_old_gateway_without_the_route() -> None:
    import httpx

    from scout.adapters.gateway_http import HttpGateway

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    gw = HttpGateway("https://gw.example", "sk_scout_x", "0", transport=httpx.MockTransport(handler))
    assert _run(gw.warm()) is False
