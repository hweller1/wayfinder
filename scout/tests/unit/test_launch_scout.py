"""scripts/launch-scout.sh lane choice: production `scout mcp` with a key, demo lane without one.

FR-I1, FR-I2, FR-I6. Stub binaries print which lane ran and their arguments; no network, no real keychain
(`SCOUT_KEYSTORE=file` skips the macOS probe), `HOME` is a scratch directory.
"""

import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "integrations" / "claude-code" / "scripts" / "launch-scout.sh"
ARGS = ["mcp", "--tools", "explore", "--render-ceiling", "1500"]


def launcher(tmp_path: Path) -> Path:
    """A copy of the script three directories below `tmp_path`, so its "own checkout" probe (which walks up
    three levels looking for this repo's pyproject) cannot find the real checkout and its `scout-demo`."""
    copy = tmp_path / "plugin" / "claude-code" / "scripts" / "launch-scout.sh"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(SOURCE.read_text())
    return copy


def stub(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "{label} $@"\n')
    path.chmod(0o755)
    return path


def launch(tmp_path: Path, *, env: dict[str, str] | None = None, demo: bool = True) -> str:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    project = tmp_path / "project"
    if demo:
        stub(project / ".venv" / "bin" / "scout-demo", "demo")
    full = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "SCOUT_KEYSTORE": "file",
        "SCOUT_BIN": str(stub(tmp_path / "bin" / "scout", "production")),
        **({"SCOUT_DEMO_PROJECT": str(project)} if demo else {}),
        **(env or {}),
    }
    proc = subprocess.run(  # noqa: S603
        ["sh", str(launcher(tmp_path)), *ARGS],  # noqa: S607
        env=full,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def write_credentials(tmp_path: Path) -> None:
    creds = tmp_path / "home" / ".config" / "scout"
    creds.mkdir(parents=True, exist_ok=True)
    (creds / "credentials").write_text("{}")


@pytest.mark.parametrize("var", ["SCOUT_KEY", "SCOUT_API_KEY"])
def test_a_key_in_the_environment_selects_the_production_lane(tmp_path: Path, var: str) -> None:
    assert launch(tmp_path, env={var: "sk_scout_test"}) == f"production {' '.join(ARGS)}"


def test_scout_login_credentials_select_the_production_lane(tmp_path: Path) -> None:
    (tmp_path / "home").mkdir()
    write_credentials(tmp_path)
    assert launch(tmp_path) == f"production {' '.join(ARGS)}"


def test_no_key_falls_back_to_the_demo_lane_even_with_a_scout_binary(tmp_path: Path) -> None:
    assert launch(tmp_path) == "demo mcp"  # the demo server takes no flags


def test_the_key_is_passed_to_scout_as_scout_key(tmp_path: Path) -> None:
    probe = tmp_path / "probe" / "scout"
    probe.parent.mkdir(parents=True)
    probe.write_text('#!/bin/sh\necho "key=${SCOUT_KEY:+set}"\n')
    probe.chmod(0o755)
    out = launch(tmp_path, env={"SCOUT_API_KEY": "sk_scout_test", "SCOUT_BIN": str(probe)})
    assert out == "key=set"


def test_lane_can_be_forced(tmp_path: Path) -> None:
    write_credentials(tmp_path)
    assert launch(tmp_path, env={"SCOUT_LANE": "demo"}) == "demo mcp"
    assert launch(tmp_path, env={"SCOUT_LANE": "production"}) == f"production {' '.join(ARGS)}"


def test_scout_without_a_key_still_starts_when_there_is_no_demo(tmp_path: Path) -> None:
    assert launch(tmp_path, demo=False) == f"production {' '.join(ARGS)}"


def test_nothing_installed_is_a_hint_and_exit_1(tmp_path: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        ["sh", str(launcher(tmp_path)), *ARGS],  # noqa: S607
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "SCOUT_KEYSTORE": "file",
            "SCOUT_DEMO_PROJECT": "",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 1
    assert "scout not found" in proc.stderr
