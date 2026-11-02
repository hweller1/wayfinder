# ruff: noqa: E501  (long test names)
"""hooks/session_start.py: returns at once, detaches its children, one watcher per project, warms the model
behind the project's gateway, says nothing outside an indexed checkout. AC-7.5, FR-I5."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "integrations" / "claude-code" / "hooks" / "session_start.py"


def git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)  # noqa: S603, S607
    return path


def index(repo: Path) -> None:
    state = repo / ".git" / "scout"
    state.mkdir(parents=True, exist_ok=True)
    (state / "status.json").write_text(json.dumps({"schema": "1.0", "state": "ready", "coverage": 1.0}))


@pytest.fixture
def fake_scout(tmp_path: Path):  # type: ignore[no-untyped-def]
    """`scout watch` records its argv and sleeps; `scout warm` records its argv and prints a warm result."""
    calls = tmp_path / "calls.log"
    script = tmp_path / "bin" / "scout"
    script.parent.mkdir()
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> {calls}\n'
        'if [ "$1" = "warm" ]; then echo \'{"warm": true, "seconds": 1.5}\'; exit 0; fi\n'
        "exec sleep 30\n"
    )
    script.chmod(0o755)
    yield script, calls
    for pidfile in tmp_path.glob("data/watch-*.pid"):
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(pidfile.read_text().strip()), signal.SIGKILL)


def run_hook(
    tmp_path: Path, project: Path, *, scout: Path | None, key: bool = True, extra: dict | None = None
):  # type: ignore[no-untyped-def]
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "SCOUT_KEYSTORE": "file",
        "CLAUDE_PROJECT_DIR": str(project),
        "CLAUDE_PLUGIN_DATA": str(tmp_path / "data"),
        # the warm worker probes the gateway first; point it at a closed local port unless a test says otherwise
        "SCOUT_GATEWAY_URL": "http://127.0.0.1:9",
        **({"SCOUT_BIN": str(scout)} if scout else {}),
        **({"SCOUT_KEY": "sk_scout_test"} if key else {}),
        **(extra or {}),
    }
    t0 = time.perf_counter()
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(HOOK)],
        input=json.dumps({"hook_event_name": "SessionStart", "cwd": str(project)}),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0
    assert proc.stderr == ""
    return elapsed, proc.stdout


def wait_for(pred, timeout: float = 5.0) -> bool:  # type: ignore[no-untyped-def]
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def calls_of(calls: Path) -> list[list[str]]:
    return [ln.split() for ln in calls.read_text().splitlines()] if calls.exists() else []


def test_returns_fast_and_starts_one_detached_watcher(tmp_path: Path, fake_scout) -> None:  # type: ignore[no-untyped-def]
    script, calls = fake_scout
    proj = git_repo(tmp_path / "proj")
    elapsed, out = run_hook(tmp_path, proj, scout=script)
    assert elapsed < 2.0
    assert out == ""  # not indexed: no context line
    assert wait_for(lambda: bool(calls_of(calls)))
    assert calls_of(calls) == [["watch", "--ensure", "--root", os.path.realpath(proj)]]


def test_second_call_starts_no_second_watcher(tmp_path: Path, fake_scout) -> None:  # type: ignore[no-untyped-def]
    script, calls = fake_scout
    proj = git_repo(tmp_path / "proj")
    run_hook(tmp_path, proj, scout=script)
    assert wait_for(lambda: bool(calls_of(calls)))
    run_hook(tmp_path, proj, scout=script)
    time.sleep(0.3)
    assert [c[0] for c in calls_of(calls)] == ["watch"]


def test_other_project_gets_its_own_watcher(tmp_path: Path, fake_scout) -> None:  # type: ignore[no-untyped-def]
    script, calls = fake_scout
    run_hook(tmp_path, git_repo(tmp_path / "a"), scout=script)
    run_hook(tmp_path, git_repo(tmp_path / "b"), scout=script)
    assert wait_for(lambda: len(calls_of(calls)) == 2)


def test_indexed_checkout_gets_context_and_a_warm_up_that_records_unreachable(
    tmp_path: Path, fake_scout
) -> None:  # type: ignore[no-untyped-def]
    """The gateway (a closed port here) is probed before `scout warm`; unreachable is recorded, not warmed."""
    script, calls = fake_scout
    proj = git_repo(tmp_path / "proj")
    index(proj)
    _, out = run_hook(tmp_path, proj, scout=script)
    ctx = json.loads(out)["hookSpecificOutput"]
    assert ctx["hookEventName"] == "SessionStart"
    assert "scout:scout-explorer" in ctx["additionalContext"]
    warm = tmp_path / "data" / "warm.json"

    def settled() -> bool:
        doc = json.loads(warm.read_text()) if warm.exists() else {}
        return any(r.get("state") == "unreachable" for r in doc.values())

    assert wait_for(settled, 30)
    assert not any(c[0] == "warm" for c in calls_of(calls))


def test_warm_up_runs_scout_warm_when_the_gateway_answers(tmp_path: Path, fake_scout) -> None:  # type: ignore[no-untyped-def]
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        script, calls = fake_scout
        proj = git_repo(tmp_path / "proj")
        index(proj)
        url = f"http://127.0.0.1:{srv.server_port}"
        run_hook(tmp_path, proj, scout=script, extra={"SCOUT_GATEWAY_URL": url})
        warm = tmp_path / "data" / "warm.json"
        assert wait_for(
            lambda: warm.exists() and json.loads(warm.read_text()).get(url, {}).get("state") == "warm", 30
        )
        rec = json.loads(warm.read_text())[url]
        assert rec["seconds"] == 1.5
        assert ["warm", "--root", os.path.realpath(proj)] in calls_of(calls)
        assert "sk_scout_test" not in warm.read_text()
        assert "sk_scout_test" not in (tmp_path / "data" / "warm.log").read_text()
    finally:
        srv.shutdown()


def test_no_warm_and_no_watch_switches(tmp_path: Path, fake_scout) -> None:  # type: ignore[no-untyped-def]
    script, calls = fake_scout
    proj = git_repo(tmp_path / "proj")
    index(proj)
    run_hook(tmp_path, proj, scout=script, extra={"SCOUT_NO_WARM": "1", "SCOUT_NO_WATCH": "1"})
    time.sleep(0.5)
    assert calls_of(calls) == []
    assert not (tmp_path / "data" / "warm.json").exists()


def test_no_key_starts_nothing(tmp_path: Path, fake_scout) -> None:  # type: ignore[no-untyped-def]
    script, calls = fake_scout
    proj = git_repo(tmp_path / "proj")
    index(proj)
    run_hook(tmp_path, proj, scout=script, key=False)
    time.sleep(0.3)
    assert calls_of(calls) == []


def test_no_scout_binary_is_a_silent_noop(tmp_path: Path) -> None:
    proj = git_repo(tmp_path / "proj")
    _, out = run_hook(tmp_path, proj, scout=None)
    assert out == ""


def test_outside_git_is_a_silent_noop(tmp_path: Path, fake_scout) -> None:  # type: ignore[no-untyped-def]
    script, calls = fake_scout
    plain = tmp_path / "plain"
    plain.mkdir()
    _, out = run_hook(tmp_path, plain, scout=script)
    assert out == ""
    time.sleep(0.2)
    assert calls_of(calls) == []
