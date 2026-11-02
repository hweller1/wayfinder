"""The stdlib hook's production-lane decision equals `core.sync.state.readiness` over a fixture matrix.

The hook must not import core, so this test is the only link between the two. FR-I6, AC-7.4.
"""

import itertools
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scout.core.contracts.schema import StatusFile
from scout.core.sync.state import readiness

REPO = Path(__file__).resolve().parents[2]
REWRITE = REPO / "integrations" / "claude-code" / "hooks" / "rewrite_explore.py"
DEAD_PID = 2**22 + 4242

STATUSES: dict[str, dict[str, object] | str | None] = {
    "missing": None,
    "corrupt": "{broken",
    "ready-fresh": {"state": "ready", "coverage": 1.0, "age": 5},
    "partial-fresh": {"state": "partial", "coverage": 0.5, "age": 30},
    "absent-fresh": {"state": "absent", "coverage": 0.0, "age": 5},
    "ready-stale": {"state": "ready", "coverage": 1.0, "age": 700},
    "ready-just-fresh": {"state": "ready", "coverage": 1.0, "age": 590},
    "ready-just-stale": {"state": "ready", "coverage": 1.0, "age": 610},
    "ready-no-ok-at": {"state": "ready", "coverage": 1.0, "age": None},
    "ready-bad-ok-at": {"state": "ready", "coverage": 1.0, "age": "garbage"},
    "bad-state": {"state": "syncing", "coverage": 1.0, "age": 5},
}


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(state_dir: Path, spec: dict[str, object] | str | None, now: float) -> StatusFile | None:
    """Writes the fixture; returns what core would parse from it (None when core would reject it)."""
    path = state_dir / "status.json"
    if spec is None:
        return None
    if isinstance(spec, str):
        path.write_text(spec)
        return None
    raw: dict[str, object] = {"schema": "1.0", "state": spec["state"], "coverage": spec["coverage"]}
    age = spec["age"]
    if isinstance(age, str):
        raw["gateway_ok_at"] = age
    elif age is not None:
        raw["gateway_ok_at"] = iso(now - float(age))  # type: ignore[arg-type]
    path.write_text(json.dumps(raw))
    try:
        return StatusFile.model_validate(raw)
    except ValueError:
        return None


def hook_rewrites(repo: Path, home: Path, demo: Path) -> bool:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "SCOUT_DEMO_HOME": str(demo),
        "SCOUT_KEYSTORE": "file",  # never probe the real macOS Keychain from a test
    }
    payload = {"tool_input": {"subagent_type": "Explore", "prompt": "p"}, "cwd": str(repo)}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(REWRITE)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return bool(proc.stdout.strip())


@pytest.mark.parametrize(
    ("status_key", "creds", "pid_state"),
    list(itertools.product(STATUSES, (True, False), ("alive", "dead", "missing", "garbage"))),
)
def test_hook_decision_equals_core_readiness(
    tmp_path: Path, status_key: str, creds: bool, pid_state: str
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603, S607
    state_dir = repo / ".git" / "scout"
    state_dir.mkdir()
    home = tmp_path / "home"
    (home / ".config" / "scout").mkdir(parents=True)
    demo = tmp_path / "demo"
    demo.mkdir()  # empty: only the production lane can say yes

    now = time.time()
    status = write_status(state_dir, STATUSES[status_key], now)
    if creds:
        (home / ".config" / "scout" / "credentials").write_text("key")
    pid_file = state_dir / "mcp.pid"
    if pid_state == "alive":
        pid_file.write_text(json.dumps({"pid": os.getpid(), "started_at": "t"}))
    elif pid_state == "dead":
        pid_file.write_text(json.dumps({"pid": DEAD_PID, "started_at": "t"}))
    elif pid_state == "garbage":
        pid_file.write_text("not json")

    expected = readiness(status, now, creds_present=creds, mcp_alive=pid_state == "alive")
    assert hook_rewrites(repo, home, demo) is expected
