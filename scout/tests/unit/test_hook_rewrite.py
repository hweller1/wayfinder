"""rewrite_explore.py and allow_explore_tool.py: routing, modes, fall-through, readiness, latency, logs.

FR-I3, FR-I4, FR-I6, NFR-14, AC-7.2, AC-7.4.
"""

import json
import os
import resource
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOKS = REPO / "integrations" / "claude-code" / "hooks"
REWRITE = HOOKS / "rewrite_explore.py"
ALLOW = HOOKS / "allow_explore_tool.py"
TARGET = "scout:scout-explorer"  # default mode hybrid
FAST = "scout:super-explore"
TOOL = "mcp__plugin_scout_scout__"
SECRET_BRIEF = "find the zebra-quasar-7731 handler"  # noqa: S105  (a brief, not a password)


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str, cwd: Path) -> str:
    out = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


class Env:
    def __init__(self, tmp: Path) -> None:
        self.repo = tmp / "repo"
        self.repo.mkdir()
        git("init", "-q", cwd=self.repo)
        (self.repo / "a.txt").write_text("x")
        self.home = tmp / "home"
        self.home.mkdir()
        self.data = tmp / "plugin-data"
        self.demo = tmp / "demo-home"
        self.demo.mkdir()
        self.state = Path(git("rev-parse", "--absolute-git-dir", cwd=self.repo)) / "scout"
        self.state.mkdir()

    def production_ready(
        self, *, state: str = "ready", ok_age: float | None = 5.0, pid: int | None = None
    ) -> None:
        status: dict[str, object] = {"schema": "1.0", "state": state, "coverage": 1.0}
        if ok_age is not None:
            status["gateway_ok_at"] = iso(time.time() - ok_age)
        (self.state / "status.json").write_text(json.dumps(status))
        (self.state / "mcp.pid").write_text(json.dumps({"pid": pid or os.getpid(), "started_at": "x"}))
        creds = self.home / ".config" / "scout"
        creds.mkdir(parents=True, exist_ok=True)
        (creds / "credentials").write_text("k")

    def demo_index(self, *, state: str = "ready", name: str = "myrepo") -> None:
        rec = {"state": state, "root": str(self.repo)}
        (self.demo / "repos.json").write_text(
            json.dumps({name: rec, "other": {"state": "ready", "root": "/nope"}})
        )

    def run(
        self,
        tool_input: dict[str, object],
        *,
        cwd: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.home),
            "SCOUT_DEMO_HOME": str(self.demo),
            "SCOUT_KEYSTORE": "file",  # never probe the real macOS Keychain from a test
            "CLAUDE_PLUGIN_DATA": str(self.data),
            **(extra_env or {}),
        }
        payload = {"tool_name": "Agent", "tool_input": tool_input, "cwd": str(cwd or self.repo)}
        if "SCOUT_EXPLORE_MODE" not in (extra_env or {}):
            env.pop("SCOUT_EXPLORE_MODE", None)
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(REWRITE)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert proc.returncode == 0
        return proc.stdout, proc.stderr

    def log(self) -> str:
        p = self.data / "hook.log"
        return p.read_text() if p.exists() else ""


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def explore_input(**extra: object) -> dict[str, object]:
    return {
        "description": "find it",
        "prompt": SECRET_BRIEF,
        "subagent_type": "Explore",
        "run_in_background": False,
        **extra,
    }


def updated(stdout: str) -> dict[str, object]:
    out = json.loads(stdout)
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    return out["hookSpecificOutput"]["updatedInput"]


def test_production_ready_rewrites_and_copies_every_other_field(env: Env) -> None:
    env.production_ready()
    ti = explore_input(model="haiku", extra_field={"a": 1})
    out, _ = env.run(ti)
    new = updated(out)
    assert new["subagent_type"] == TARGET
    assert {k: v for k, v in new.items() if k != "subagent_type"} == {
        k: v for k, v in ti.items() if k != "subagent_type"
    }
    assert "rewrite lane=production" in env.log()


def test_non_explore_is_untouched(env: Env) -> None:
    env.production_ready()
    out, _ = env.run({**explore_input(), "subagent_type": "general-purpose"})
    assert out == ""
    assert env.log() == ""


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(lambda e: None, id="nothing-indexed"),
        pytest.param(lambda e: e.production_ready(state="absent"), id="index-absent"),
        pytest.param(lambda e: e.production_ready(ok_age=601), id="gateway-stale"),
        pytest.param(lambda e: e.production_ready(ok_age=None), id="gateway-never"),
        pytest.param(lambda e: e.production_ready(pid=2**22 + 12345), id="mcp-dead"),
    ],
)
def test_readiness_failure_falls_through(env: Env, setup) -> None:  # type: ignore[no-untyped-def]
    setup(env)
    out, _ = env.run(explore_input())
    assert out == ""
    assert "fallthrough" in env.log()


@pytest.mark.parametrize("var", ["SCOUT_KEY", "SCOUT_API_KEY"])
def test_key_in_the_environment_counts_as_credentials(env: Env, var: str) -> None:
    """The launcher and the hook share one rule: a key in the environment selects the production lane."""
    env.production_ready()
    (env.home / ".config" / "scout" / "credentials").unlink()
    assert env.run(explore_input())[0] == ""
    out, _ = env.run(explore_input(), extra_env={var: "sk_scout_test"})
    assert updated(out)["subagent_type"] == TARGET
    assert "lane=production" in env.log()


def test_missing_creds_falls_through(env: Env) -> None:
    env.production_ready()
    (env.home / ".config" / "scout" / "credentials").unlink()
    assert env.run(explore_input())[0] == ""


@pytest.mark.parametrize("content", ["{not json", "[]", '"str"', ""])
def test_corrupt_status_file_falls_through(env: Env, content: str) -> None:
    env.production_ready()
    (env.state / "status.json").write_text(content)
    assert env.run(explore_input())[0] == ""


def test_not_a_git_repo_falls_through(env: Env, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert env.run(explore_input(), cwd=plain)[0] == ""


def test_demo_lane_rewrites_and_names_the_repo(env: Env) -> None:
    env.demo_index()
    out, _ = env.run(explore_input())
    new = updated(out)
    assert new["subagent_type"] == TARGET
    assert new["prompt"] == f"[scout repo: myrepo]\n{SECRET_BRIEF}"
    assert new["run_in_background"] is False
    assert "rewrite lane=demo" in env.log()


def test_demo_lane_needs_a_ready_index_for_this_repo(env: Env, tmp_path: Path) -> None:
    env.demo_index(state="absent")
    assert env.run(explore_input())[0] == ""
    env.demo_index()
    other = tmp_path / "other"
    other.mkdir()
    git("init", "-q", cwd=other)
    assert env.run(explore_input(), cwd=other)[0] == ""


def test_demo_lane_works_from_a_subdirectory(env: Env) -> None:
    env.demo_index()
    sub = env.repo / "pkg"
    sub.mkdir()
    assert updated(env.run(explore_input(), cwd=sub)[0])["subagent_type"] == TARGET


def test_log_never_contains_brief_text(env: Env) -> None:
    env.demo_index()
    env.run(explore_input())
    env.run({**explore_input(), "subagent_type": "Explore"}, cwd=env.repo / ".git")
    assert env.log()
    assert SECRET_BRIEF not in env.log()
    assert "zebra" not in env.log()


def test_garbage_stdin_is_silent() -> None:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(REWRITE)], input="not json", capture_output=True, text=True, check=False
    )
    assert (proc.returncode, proc.stdout) == (0, "")


@pytest.mark.slow
def test_median_runtime_under_150ms(env: Env) -> None:
    """NFR-14: the bound is 150 ms. CPU time of the hook child (user+sys, git included) is measured, not
    wall time, so a loaded machine cannot flake it; slow-marked so `make check` skips it."""
    env.production_ready()
    ti = explore_input()
    times: list[float] = []
    for _ in range(20):
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        out, _ = env.run(ti)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        times.append((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime))
        assert updated(out)["subagent_type"] == TARGET
    assert statistics.median(times) < 0.150, statistics.median(times)


def allow(tool_name: object) -> str:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(ALLOW)],
        input=json.dumps({"tool_name": tool_name, "tool_input": {}}),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


@pytest.mark.parametrize("tool", ["explore", "semantic_search", "read_range", "index_status"])
def test_allow_hook_allows_the_plugins_own_read_only_tools(tool: str) -> None:
    out = json.loads(allow(TOOL + tool))["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "allow"


@pytest.mark.parametrize(
    "name",
    ["Bash", "mcp__other_server__explore", TOOL + "delete_everything", "mcp__plugin_scout_x__explore", None],
)
def test_allow_hook_is_silent_for_any_other_tool(name: object) -> None:
    assert allow(name) == ""


def test_allow_hook_garbage_stdin_is_silent() -> None:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(ALLOW)], input="nope", capture_output=True, text=True, check=False
    )
    assert (proc.returncode, proc.stdout) == (0, "")


# --- modes and the gateway probe ---------------------------------------------------------------------
def settings(env: Env, **values: str) -> None:
    env.data.mkdir(parents=True, exist_ok=True)
    proj = {os.path.realpath(env.repo): values}
    (env.data / "settings.json").write_text(json.dumps({"projects": proj}))


def test_default_mode_routes_to_the_hybrid_explorer(env: Env) -> None:
    env.production_ready()
    assert updated(env.run(explore_input())[0])["subagent_type"] == TARGET
    assert f"target={TARGET}" in env.log()


def test_fast_mode_routes_to_the_shim(env: Env) -> None:
    env.production_ready()
    settings(env, mode="fast")
    assert updated(env.run(explore_input())[0])["subagent_type"] == FAST


def test_mode_off_never_rewrites(env: Env) -> None:
    env.production_ready()
    settings(env, mode="off")
    assert env.run(explore_input())[0] == ""
    assert "reason=mode-off" in env.log()


def test_env_mode_wins_over_the_project_setting(env: Env) -> None:
    env.production_ready()
    settings(env, mode="off")
    out, _ = env.run(explore_input(), extra_env={"SCOUT_EXPLORE_MODE": "fast"})
    assert updated(out)["subagent_type"] == FAST


def test_unknown_mode_falls_back_to_hybrid(env: Env) -> None:
    env.production_ready()
    out, _ = env.run(explore_input(), extra_env={"SCOUT_EXPLORE_MODE": "bogus"})
    assert updated(out)["subagent_type"] == TARGET


def warm(
    env: Env, state: str, finished_age: float, url: str = "https://scout-gateway.example.com"
) -> None:
    env.data.mkdir(parents=True, exist_ok=True)
    rec = {"state": state, "finished_at": time.time() - finished_age}
    (env.data / "warm.json").write_text(json.dumps({url: rec}))


def test_gateway_probe_failure_after_the_last_ping_falls_through(env: Env) -> None:
    env.production_ready(ok_age=60)
    warm(env, "unreachable", finished_age=5)
    assert env.run(explore_input())[0] == ""
    assert "reason=gateway-unreachable" in env.log()


def test_gateway_probe_failure_before_a_newer_ping_does_not_block(env: Env) -> None:
    env.production_ready(ok_age=5)
    warm(env, "unreachable", finished_age=120)
    assert updated(env.run(explore_input())[0])["subagent_type"] == TARGET


def test_probe_failure_on_another_model_does_not_block(env: Env) -> None:
    env.production_ready(ok_age=60)
    warm(env, "unreachable", finished_age=5, url="https://scout-gateway-a1.example.com")
    assert updated(env.run(explore_input())[0])["subagent_type"] == TARGET


def test_probe_failure_on_the_selected_model_blocks(env: Env) -> None:
    env.production_ready(ok_age=60)
    settings(env, model="a1")
    warm(env, "unreachable", finished_age=5, url="https://scout-gateway-a1.example.com")
    assert env.run(explore_input())[0] == ""


def warm_failed(env: Env, error: str, finished_age: float) -> None:
    env.data.mkdir(parents=True, exist_ok=True)
    rec = {"state": "failed", "error": error, "finished_at": time.time() - finished_age}
    (env.data / "warm.json").write_text(json.dumps({"https://scout-gateway.example.com": rec}))


def test_a_failed_model_warm_up_falls_through_even_after_a_fresh_gateway_ping(env: Env) -> None:
    """The watcher pings the gateway, not the model: a model that rejected the warm-up blocks routing."""
    env.production_ready(ok_age=1)
    warm_failed(env, "INVALID_ARGUMENT", finished_age=30)
    assert env.run(explore_input())[0] == ""
    assert "reason=model-failed" in env.log()


def test_a_model_failure_expires_so_a_recovered_model_is_retried(env: Env) -> None:
    env.production_ready(ok_age=1)
    warm_failed(env, "INVALID_ARGUMENT", finished_age=700)
    assert updated(env.run(explore_input())[0])["subagent_type"] == TARGET


@pytest.mark.parametrize("error", ["timeout", "UPSTREAM_TIMEOUT"])
def test_a_slow_cold_start_does_not_block_routing(env: Env, error: str) -> None:
    env.production_ready(ok_age=1)
    warm_failed(env, error, finished_age=30)
    assert updated(env.run(explore_input())[0])["subagent_type"] == TARGET


def test_demo_index_is_ignored_when_a_key_is_configured(env: Env) -> None:
    """A user with a key whose production index is not ready gets the stock Explore, not the demo lane."""
    env.production_ready(state="absent")
    env.demo_index()
    assert env.run(explore_input())[0] == ""
