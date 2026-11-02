"""Headless Claude Code smoke of the scout plugin (fast mode: the shim): rewrite, allow, fallback.

AC-7.2, AC-7.3, AC-7.4, FR-I3, FR-I4, FR-I6.

Production lane (`test_production_lane_*`): a scout key from `SCOUT_LIVE_DEV_KEY` or the 0600 file
`~/.config/scout/live-smoke-keys.txt` (`dev=` line) reaches the plugin only as `SCOUT_KEY` in the environment
(nothing is written to the Keychain or the credentials file). A public repo (gorilla/mux) is cloned to a temp
directory and indexed through the HOSTED gateway with `scout sync --wait`; then `claude -p` must rewrite
Explore, the shim must reach `scout mcp` (launcher lane `production`, tool call without `repo`), and the
answer must carry ranges from that clone. The branch namespaces are deleted afterwards (`delete_branch`).
Skips without a key. Demo-lane cases below keep the fallback honest: with NO key in the environment and no
`~/.config/scout/credentials`, the launcher and the hook choose the demo lane.

Demo lane, case A (ready): cwd is the demo checkout of `gomux` (indexed by `scout-demo index`, so a ready
index exists in `~/.scout-demo/repos.json`). A prompt that makes the main agent delegate to Explore must be
rewritten to `scout:super-explore`, the shim must call the scout MCP tool without a permission
denial, and the answer must contain scout's `path:start-end` ranges. Case B (not ready): an unindexed git
repo; the hook must fall through and the built-in Explore must run (no zero-tools refusal).

Costs a few cents of haiku plus one A1-35B explore on Modal (about $0.03, 30-150 s). Writes
`artifacts/live/claude_smoke.json`. Needs `claude` (logged in) and the demo keys from the .env files.
"""

import asyncio
import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from helpers_claude import (  # type: ignore[import-not-found]
    ARTIFACT,
    HOOK_LOG,
    MCP_TOOL,
    ZERO_TOOLS,
    claude_available,
    hook_log_since,
    hook_log_size,
    run_claude,
)

from scout.adapters.gateway_http import HttpGateway
from scout.core.sync.state import StateDir
from scout.demo.env import demo_home
from scout.demo.registry import Registry

pytestmark = [pytest.mark.live]

RANGE = re.compile(r"\b[\w./-]+\.go:\d+-\d+\b")
KEY_FILE = Path.home() / ".config" / "scout" / "live-smoke-keys.txt"
GATEWAY = os.environ.get("SCOUT_GATEWAY_URL") or "https://scout-gateway.example.com"
SCOUT_BIN = Path(sys.executable).with_name("scout")
RESULTS: dict[str, object] = {}


def _write_artifact() -> None:
    """Merge into the existing artifact so a partial run (one lane) keeps the other lane's last result."""
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    try:
        merged = json.loads(ARTIFACT.read_text())
    except (OSError, ValueError):
        merged = {}
    merged.update(RESULTS)
    ARTIFACT.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def gomux_root() -> Path:
    rec = Registry(demo_home()).get("gomux")
    if rec is None or rec.state not in ("ready", "partial"):
        pytest.skip(
            "demo repo `gomux` is not indexed: `scout-demo index https://github.com/gorilla/mux --name gomux`"
        )
    return Path(rec.root)


@pytest.fixture(scope="module", autouse=True)
def _need_claude() -> None:
    if not claude_available():
        RESULTS["claude_smoke"] = "blocked-no-claude"
        _write_artifact()
        pytest.skip("BLOCKED: `claude` is not on PATH, the headless smoke cannot run")


def test_case_a_ready_rewrites_allows_and_returns_scout_ranges(gomux_root: Path) -> None:
    offset = hook_log_size()
    run = run_claude(
        "Use the Explore subagent to find where the router produces the 405 Method Not Allowed response. "
        "Then list the file ranges it returned.",
        gomux_root,
        timeout=420,
        extra_env={"SCOUT_EXPLORE_MODE": "fast"},
    )
    log = hook_log_since(offset)
    denials = run.result.get("permission_denials", [])
    assert any("rewrite lane=demo" in line for line in log), log
    assert "Explore" in run.agent_types(), "the main agent did not delegate to Explore"
    assert run.subagent_by_type().get("scout:super-explore", 0) >= 1, run.subagent_by_type()
    assert ZERO_TOOLS not in run.raw
    calls = run.mcp_calls()
    assert calls, "the shim never called the scout MCP tool"
    assert calls[0]["input"].get("repo") == "gomux"
    scout_text = run.tool_results_text()
    assert RANGE.search(scout_text), scout_text[:500]
    assert "mux.go" in scout_text
    assert denials == []
    RESULTS.update(
        {
            "rewrite": True,
            "permission_denials": len(denials),
            "scout_ranges": sorted(set(RANGE.findall(scout_text)))[:8],
            "cost_usd": run.result.get("total_cost_usd"),
        }
    )
    _write_artifact()


def test_case_b_not_ready_falls_through_to_builtin_explore(tmp_path: Path) -> None:
    repo = tmp_path / "unindexed"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603, S607
    (repo / "hello.py").write_text("def greet():\n    return 'hi'\n")
    offset = hook_log_size()
    run = run_claude(
        "Use the Explore subagent to find which file in this repo defines greet, then name the file.",
        repo,
        timeout=240,
    )
    log = hook_log_since(offset)
    assert any(line.split()[1] == "fallthrough" for line in log if len(line.split()) > 1), log
    assert not any("rewrite" in line.split()[1:2] for line in log)
    assert "Explore" in run.agent_types()
    assert ZERO_TOOLS not in run.raw
    assert run.subagent_by_type().get("Explore", 0) >= 1, run.subagent_by_type()
    assert "hello.py" in run.result.get("result", "")
    RESULTS.update({"fallback_ok": True, "fallback_builtin_explore_ran": True})
    _write_artifact()


# --- production lane: scout key + hosted gateway ----------------------------------------------------------
def _dev_key() -> str:
    key = os.environ.get("SCOUT_LIVE_DEV_KEY", "")
    if not key and KEY_FILE.exists():
        for line in KEY_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name == "dev" and value:
                key = value.strip()
    return key


def _kill_watcher(data_dir: Path) -> None:
    """The SessionStart hook leaves a detached `scout watch --ensure` per project; stop this run's."""
    for pidfile in data_dir.glob("watch-*.pid"):
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(pidfile.read_text().strip()), 15)


async def _cleanup(key: str, clone: Path) -> None:
    """Delete this run's own branch (chunks and tree) through the gateway. The tenant's shared cache and meta
    namespaces belong to the gateway and are left alone."""
    state = StateDir(str(clone / ".git" / "scout")).read_state()
    gw = HttpGateway(GATEWAY, key)
    try:
        if state.repo_id and state.last_branch:
            await gw.delete_branch(state.repo_id, state.last_branch)
    finally:
        await gw.aclose()


def test_production_lane_rewrites_and_answers_through_scout_mcp_and_the_hosted_gateway(
    tmp_path: Path,
) -> None:
    """The plugin on the production path: `scout login` key in the environment, `scout mcp` against the hosted
    gateway, no demo registry involved (SCOUT_DEMO_HOME points at an empty directory, so a demo rewrite is
    impossible and `lane=production` in the hook log proves the lane)."""
    key = _dev_key()
    if not key:
        pytest.skip("no live gateway key (SCOUT_LIVE_DEV_KEY or the live-smoke-keys file)")
    if not SCOUT_BIN.exists():
        pytest.skip("`scout` entry point missing from the venv: uv sync")
    clone = tmp_path / "mux"
    subprocess.run(  # noqa: S603
        ["git", "clone", "--depth", "1", "--quiet", "https://github.com/gorilla/mux", str(clone)],  # noqa: S607
        check=True,
        capture_output=True,
        timeout=180,
    )
    demo_home = tmp_path / "empty-demo-home"
    demo_home.mkdir()
    env = {
        "SCOUT_EXPLORE_MODE": "fast",
        "SCOUT_KEY": key,
        "SCOUT_GATEWAY_URL": GATEWAY,
        "SCOUT_BIN": str(SCOUT_BIN),
        "SCOUT_DEMO_HOME": str(demo_home),
        "SCOUT_KEYSTORE": "file",  # nothing is written to the Keychain; HOME stays real for claude's auth
    }
    data_dir = HOOK_LOG.parent
    try:
        sync = subprocess.run(  # noqa: S603
            [str(SCOUT_BIN), "sync", "--wait", "--root", str(clone)],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        assert sync.returncode == 0, sync.stdout[-300:]
        status = json.loads(sync.stdout)
        assert (status["state"], status["coverage"]) == ("ready", 1.0)
        assert key not in sync.stdout + sync.stderr

        offset = hook_log_size()
        run = run_claude(
            "Use the Explore subagent to find where the router produces the 405 Method Not Allowed response. "
            "Then list the file ranges it returned.",
            clone,
            timeout=480,
            extra_env=env,
        )
        log = hook_log_since(offset)
        denials = run.result.get("permission_denials", [])
        assert any("rewrite lane=production" in line for line in log), log
        assert not any("lane=demo" in line and "rewrite" in line for line in log), log
        assert "Explore" in run.agent_types(), "the main agent did not delegate to Explore"
        assert run.subagent_by_type().get("scout:super-explore", 0) >= 1, run.subagent_by_type()
        assert ZERO_TOOLS not in run.raw
        calls = run.mcp_calls()
        assert calls, "the shim never called the scout MCP tool"
        assert (
            "repo" not in calls[0]["input"]
        )  # the production tool has no `repo` (the demo tool requires it)
        scout_text = run.tool_results_text()
        assert RANGE.search(scout_text), scout_text[:500]
        assert "mux.go" in scout_text
        # the main agent may try Bash itself (headless cannot approve it); the scout tool must never be denied
        assert [d for d in denials if d.get("tool_name") == MCP_TOOL] == []
        assert key not in run.raw
        RESULTS.update(
            {
                "production_lane": True,
                "production_gateway": GATEWAY,
                "production_scout_tool_denials": 0,
                "production_other_denials": [d.get("tool_name") for d in denials],
                "production_scout_ranges": sorted(set(RANGE.findall(scout_text)))[:8],
                "production_cost_usd": run.result.get("total_cost_usd"),
            }
        )
        _write_artifact()
    finally:
        _kill_watcher(data_dir)
        asyncio.run(_cleanup(key, clone))
