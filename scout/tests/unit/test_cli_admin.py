"""Task 7.8: `login`, `logout`, `sync`, `watch`, `init` over the fake profile with a temporary HOME (FR-I8,
FR-I13, AC-7.7, AC-3.9).

Host-specific facts come only from `scout-host.json` (a stub under `tests/fixtures/integrations_stub/`), so no
host string lives in `src/scout`: `tests/unit/test_no_host_names.py` stays green beside this file.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from scout.adapters.keychain import KeychainStore
from scout.core.contracts.errors import ErrorCode, ScoutError, cli_exit
from scout.surfaces.cli import admin

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tests" / "fixtures" / "poc_explore_case.json"
STUB = ROOT / "tests" / "fixtures" / "integrations_stub"


def run(
    *argv: str, home: Path, env: dict[str, str] | None = None, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    full = {
        **os.environ,
        "HOME": str(home),
        "SCOUT_KEYSTORE": "file",
        "SCOUT_PROFILE": "fake",
        "SCOUT_FAKE_CASE": str(CASE),
        "SCOUT_ROOTS_FILE": str(home / "roots.json"),
        "SCOUT_INTEGRATIONS_DIR": str(STUB),
        **(env or {}),
    }
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", *argv],
        capture_output=True,
        text=True,
        input=stdin,
        env=full,
        check=False,
    )


def store(home: Path) -> KeychainStore:
    return KeychainStore(env={"HOME": str(home), "SCOUT_KEYSTORE": "file"})


def git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)  # noqa: S603, S607
    return path.resolve()


# --- login / logout ---------------------------------------------------------------------------------------
def test_login_stores_the_key_the_tenant_and_the_team_key(tmp_path: Path) -> None:
    p = run("login", "--key", "sk_scout_test", home=tmp_path)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert (out["logged_in"], out["tenant"]) == (True, "dev")
    assert out["daemon"]["installed"] is False  # the fake profile never loads a daemon (see test_launchd.py)
    ks = store(tmp_path)
    assert ks.get("scout_key") == b"sk_scout_test"
    assert ks.get("scout_tenant") == b"dev"
    k_team = ks.get("k_team:dev")
    assert k_team is not None
    assert len(k_team) == 64
    assert "sk_scout_test" not in p.stdout + p.stderr  # the key is never echoed
    creds = tmp_path / ".config" / "scout" / "credentials"
    assert creds.stat().st_mode & 0o777 == 0o600


def test_login_reads_the_key_from_stdin_or_the_environment(tmp_path: Path) -> None:
    assert run("login", "--key", "-", home=tmp_path, stdin="sk_from_stdin\n").returncode == 0
    assert store(tmp_path).get("scout_key") == b"sk_from_stdin"
    assert run("login", home=tmp_path, env={"SCOUT_KEY": "sk_from_env"}).returncode == 0
    assert store(tmp_path).get("scout_key") == b"sk_from_env"


def test_login_without_a_key_is_invalid_argument(tmp_path: Path) -> None:
    p = run("login", home=tmp_path)
    assert p.returncode == cli_exit(ErrorCode.INVALID_ARGUMENT)
    assert json.loads(p.stdout)["error"]["code"] == "INVALID_ARGUMENT"
    assert store(tmp_path).get("scout_key") is None


def test_a_rejected_key_stores_nothing(tmp_path: Path) -> None:
    case = tmp_path / "case.json"
    base = json.loads(CASE.read_text())
    case.write_text(
        json.dumps(
            {
                **base,
                "repo": str(ROOT / "tests/fixtures/repo_small"),
                "gateway_error": {"code": "UNAUTHENTICATED"},
            }
        )
    )
    p = run("login", "--key", "sk_bad", home=tmp_path, env={"SCOUT_FAKE_CASE": str(case)})
    assert p.returncode == cli_exit(ErrorCode.UNAUTHENTICATED)
    assert json.loads(p.stdout)["error"]["code"] == "UNAUTHENTICATED"
    assert "sk_bad" not in p.stdout + p.stderr
    assert store(tmp_path).get("scout_key") is None
    assert store(tmp_path).get("k_team:dev") is None


def test_logout_removes_everything_login_stored(tmp_path: Path) -> None:
    run("login", "--key", "sk_x", home=tmp_path)
    p = run("logout", home=tmp_path)
    assert p.returncode == 0
    assert json.loads(p.stdout) == {"logged_in": False, "was_logged_in": True}
    ks = store(tmp_path)
    assert (ks.get("scout_key"), ks.get("scout_tenant"), ks.get("k_team:dev")) == (None, None, None)
    again = run("logout", home=tmp_path)  # idempotent
    assert again.returncode == 0
    assert json.loads(again.stdout)["was_logged_in"] is False


# --- sync -------------------------------------------------------------------------------------------------
def test_sync_wait_indexes_and_prints_the_status(tmp_path: Path) -> None:
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps({"repo": str(ROOT / "tests/fixtures/repo_small"), "state_dir": str(tmp_path / "st")})
    )
    p = run("sync", "--wait", home=tmp_path, env={"SCOUT_FAKE_CASE": str(case)})
    assert p.returncode == 0, p.stderr
    status = json.loads(p.stdout)
    assert (status["state"], status["coverage"]) == ("ready", 1.0)
    text = run("sync", "--wait", "--text", home=tmp_path, env={"SCOUT_FAKE_CASE": str(case)})
    assert text.stdout.startswith("index ready coverage=1.00")


def test_sync_without_wait_starts_a_detached_run_and_returns() -> None:
    spawned: list[argparse.Namespace] = []

    def spawn(args: argparse.Namespace) -> int:
        spawned.append(args)
        return 4242

    out = _capture(
        lambda: admin.sync(argparse.Namespace(wait=False, full=True, text=False), lambda a: None, spawn)
    )
    assert json.loads(out) == {"started": True, "pid": 4242}
    assert spawned
    assert spawned[0].full is True


def _capture(fn: Any) -> str:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert fn() == 0
    return buf.getvalue()


# --- watch ------------------------------------------------------------------------------------------------
def test_watch_add_and_remove_keep_a_roots_registry(tmp_path: Path) -> None:
    a, b = git_repo(tmp_path / "a"), git_repo(tmp_path / "b")
    assert json.loads(run("watch", "--add", str(a), home=tmp_path).stdout) == {"roots": [str(a)]}
    assert json.loads(run("watch", "--add", str(b), home=tmp_path).stdout)["roots"] == sorted(
        [str(a), str(b)]
    )
    assert json.loads(run("watch", "--add", str(a), home=tmp_path).stdout)["roots"] == sorted(
        [str(a), str(b)]
    )
    assert json.loads(run("watch", "--remove", str(a), home=tmp_path).stdout) == {"roots": [str(b)]}
    assert admin.read_roots(tmp_path / "roots.json") == [str(b)]


def test_watch_add_refuses_a_non_checkout(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    p = run("watch", "--add", str(plain), home=tmp_path)
    assert p.returncode == cli_exit(ErrorCode.NOT_A_REPO)
    gone = run("watch", "--add", str(tmp_path / "missing"), home=tmp_path)
    assert gone.returncode == cli_exit(ErrorCode.NOT_FOUND)
    assert admin.read_roots(tmp_path / "roots.json") == []


def test_watch_ensure_runs_the_single_instance_watcher(tmp_path: Path) -> None:
    p = run("watch", "--ensure", home=tmp_path)  # the fake source ends at once, so the watcher returns
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert out["started"] is True
    assert out["pid"] > 0
    assert run("watch", home=tmp_path).returncode == 0  # `--ensure` is the default


def test_a_second_watcher_for_the_same_checkout_reports_the_running_one() -> None:
    from scout.core.sync.watch import WatchResult

    class Held:
        async def watch(self, stop: threading.Event) -> WatchResult:
            return WatchResult(False, 777)

    out = _capture(lambda: admin.watch_one(argparse.Namespace(), {}, lambda a, e: Held()))
    assert json.loads(out) == {"started": False, "pid": 777}


async def test_the_supervisor_follows_the_roots_registry(tmp_path: Path) -> None:
    roots = tmp_path / "roots.json"
    log: list[tuple[str, str]] = []

    class FakeApp:
        def __init__(self, root: str) -> None:
            self.root = root

        async def watch(self, stop: threading.Event) -> None:
            log.append(("start", self.root))
            await asyncio.to_thread(stop.wait)
            log.append(("stop", self.root))

    async def until(pred: Any) -> None:
        for _ in range(200):
            if pred():
                return
            await asyncio.sleep(0.02)
        raise AssertionError(f"timed out: {log}")

    stop = threading.Event()
    admin.write_roots(roots, ["/r/one"])
    task = asyncio.create_task(admin.supervise(roots, FakeApp, stop, poll_s=0.05))
    await until(lambda: ("start", "/r/one") in log)
    admin.write_roots(roots, ["/r/one", "/r/two"])
    await until(lambda: ("start", "/r/two") in log)
    admin.write_roots(roots, ["/r/two"])
    await until(lambda: ("stop", "/r/one") in log)
    assert ("stop", "/r/two") not in log
    stop.set()
    await asyncio.wait_for(task, timeout=5)
    assert ("stop", "/r/two") in log
    assert log.count(("start", "/r/one")) == 1


async def test_the_supervisor_survives_a_watcher_that_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roots = tmp_path / "roots.json"
    admin.write_roots(roots, ["/r/bad"])
    tries: list[int] = []

    class Failing:
        async def watch(self, stop: threading.Event) -> None:
            tries.append(1)
            raise ScoutError(ErrorCode.UNAUTHENTICATED, "login first")

    stop = threading.Event()
    task = asyncio.create_task(admin.supervise(roots, lambda r: Failing(), stop, poll_s=0.05))
    for _ in range(100):
        if len(tries) >= 2:  # restarted on a later poll, not abandoned
            break
        await asyncio.sleep(0.02)
    stop.set()
    await asyncio.wait_for(task, timeout=5)
    assert len(tries) >= 2
    assert "UNAUTHENTICATED" in capsys.readouterr().err


# --- init -------------------------------------------------------------------------------------------------
def host_dir(
    tmp_path: Path, *, version: str = "2.1.304 (Claude Code)", min_version: str | None = None
) -> Path:
    """The stub host directory with `version_command` rewritten to print `version` (no `claude` needed)."""
    doc = json.loads((STUB / "claude-code" / "scout-host.json").read_text())
    doc["version_command"] = [sys.executable, "-c", f"print({version!r})"]
    if min_version:
        doc["min_version"] = min_version
    target = tmp_path / "integrations" / "claude-code"
    target.mkdir(parents=True)
    (target / "scout-host.json").write_text(json.dumps(doc))
    return tmp_path / "integrations"


def init(tmp_path: Path, *argv: str, integrations: Path) -> subprocess.CompletedProcess[str]:
    return run("init", *argv, home=tmp_path, env={"SCOUT_INTEGRATIONS_DIR": str(integrations)})


def test_init_checks_the_host_version_and_prints_install_commands(tmp_path: Path) -> None:
    p = init(tmp_path, "--host", "claude-code", integrations=host_dir(tmp_path))
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert (out["host"], out["version"], out["min_version"], out["ok"]) == (
        "claude-code",
        "2.1.304",
        "2.1.292",
        True,
    )
    assert out["install_commands"][0].startswith("claude plugin marketplace add")
    assert out["override"] is None


def test_init_reports_a_host_that_is_too_old(tmp_path: Path) -> None:
    p = init(tmp_path, "--host", "claude-code", integrations=host_dir(tmp_path, version="2.1.200"))
    out = json.loads(p.stdout)
    assert out["ok"] is False
    assert "older than 2.1.292" in out["reason"]
    assert p.returncode == 0  # information, not a failure: the commands are still printed


def test_init_without_a_runnable_host_binary_says_so(tmp_path: Path) -> None:
    d = host_dir(tmp_path)
    f = d / "claude-code" / "scout-host.json"
    doc = json.loads(f.read_text())
    doc["version_command"] = ["/nonexistent/host-binary", "--version"]
    f.write_text(json.dumps(doc))
    out = json.loads(init(tmp_path, "--host", "claude-code", integrations=d).stdout)
    assert out["ok"] is False
    assert "could not be run" in out["reason"]


def test_override_is_absent_by_default_and_written_only_with_the_flag(tmp_path: Path) -> None:
    """AC-7.7: the override file is written only on explicit opt-in."""
    d = host_dir(tmp_path)
    target = tmp_path / ".claude" / "agents" / "Explore.md"
    init(tmp_path, "--host", "claude-code", integrations=d)
    assert not target.exists()
    p = init(tmp_path, "--host", "claude-code", "--override", integrations=d)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["override"] == {"written": True, "path": str(target)}
    template = json.loads((STUB / "claude-code" / "scout-host.json").read_text())["override"]["template"]
    assert target.read_text() == template
    again = json.loads(init(tmp_path, "--host", "claude-code", "--override", integrations=d).stdout)
    assert again["override"] == {"written": False, "path": str(target)}  # idempotent


def test_override_never_replaces_a_different_file(tmp_path: Path) -> None:
    d = host_dir(tmp_path)
    target = tmp_path / ".claude" / "agents" / "Explore.md"
    target.parent.mkdir(parents=True)
    target.write_text("my own Explore agent\n")
    p = init(tmp_path, "--host", "claude-code", "--override", integrations=d)
    assert p.returncode == cli_exit(ErrorCode.CONFLICT)
    assert target.read_text() == "my own Explore agent\n"


def test_the_claude_code_flag_is_an_alias_for_host(tmp_path: Path) -> None:
    d = host_dir(tmp_path)
    a = json.loads(init(tmp_path, "--claude-code", integrations=d).stdout)
    b = json.loads(init(tmp_path, "--host", "claude-code", integrations=d).stdout)
    assert a == b
    assert not (tmp_path / ".claude").exists()  # the alias alone writes nothing (AC-7.7)


def test_init_errors_are_typed(tmp_path: Path) -> None:
    d = host_dir(tmp_path)
    no_host = init(tmp_path, integrations=d)
    assert no_host.returncode == cli_exit(ErrorCode.INVALID_ARGUMENT)
    assert "claude-code" in json.loads(no_host.stdout)["error"]["message"]  # lists what is available
    unknown = init(tmp_path, "--host", "nope", integrations=d)
    assert unknown.returncode == cli_exit(ErrorCode.NOT_FOUND)
    (d / "claude-code" / "scout-host.json").write_text("not json")
    bad = init(tmp_path, "--host", "claude-code", integrations=d)
    assert bad.returncode == cli_exit(ErrorCode.INVALID_ARGUMENT)
    (d / "claude-code" / "scout-host.json").write_text(json.dumps({"override": {"target": 1}}))
    assert init(tmp_path, "--host", "claude-code", integrations=d).returncode == cli_exit(
        ErrorCode.INVALID_ARGUMENT
    )


def test_the_stub_and_the_real_host_files_pass_the_shape_check() -> None:
    for root in (STUB, ROOT / "integrations"):
        for host in admin.available_hosts(root):
            admin.load_host(root, host)


def test_version_parsing() -> None:
    assert admin.parse_version("2.1.304 (Claude Code)") == (2, 1, 304)
    assert admin.parse_version("v10.0") == (10, 0)
    assert admin.parse_version("no digits") is None
    assert admin.parse_version("2.1.304") >= admin.parse_version("2.1.292")  # type: ignore[operator]
    assert admin.parse_version("2.1.100") < admin.parse_version("2.1.292")  # type: ignore[operator]
