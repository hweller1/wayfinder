"""Task 7.9: the launchd installer for `scout watch` (FR-I13, AC-8.4).

The rendered plist passes `plutil -lint`; install writes the file through an injected `launchctl` runner (a
stub: no real agent is ever loaded); `--no-daemon` skips; a failing launchctl leaves the plist in place and
says so.
"""

import json
import os
import plistlib
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.surfaces.cli import launchd
from scout.surfaces.cli.launchd import RunResult

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tests" / "fixtures" / "poc_explore_case.json"
requires_plutil = pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is macOS only")


class Recorder:
    """A `launchctl` stand-in: records every call, answers with scripted return codes."""

    def __init__(self, *, bootstrap_rc: int = 0, stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.bootstrap_rc, self.stderr = bootstrap_rc, stderr

    def __call__(self, args: Sequence[str]) -> RunResult:
        self.calls.append(list(args))
        if args[0] == "bootstrap":
            return RunResult(self.bootstrap_rc, self.stderr)
        return RunResult(0)


def env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        "HOME": str(tmp_path),
        "SCOUT_LAUNCHD_DIR": str(tmp_path / "LaunchAgents"),
        "PATH": "/usr/bin:/bin",
        **extra,
    }


@requires_plutil
def test_rendered_plist_passes_plutil_lint(tmp_path: Path) -> None:
    path = tmp_path / "dev.scout.watch.plist"
    path.write_bytes(launchd.render_plist(env(tmp_path)))
    p = subprocess.run(["plutil", "-lint", str(path)], capture_output=True, text=True, check=False)  # noqa: S603, S607
    assert p.returncode == 0, p.stdout + p.stderr
    assert "OK" in p.stdout


def test_plist_content(tmp_path: Path) -> None:
    doc = plistlib.loads(launchd.render_plist(env(tmp_path, SCOUT_GATEWAY_URL="https://gw.test")))
    assert doc["Label"] == "dev.scout.watch"
    assert doc["ProgramArguments"] == [sys.executable, "-m", "scout", "watch", "--all"]
    assert doc["RunAtLoad"] is True
    assert doc["KeepAlive"] is True
    assert doc["ThrottleInterval"] == launchd.THROTTLE_S
    assert doc["StandardOutPath"] == str(tmp_path / "Library" / "Logs" / "scout" / "watch.out.log")
    assert doc["StandardErrorPath"] == str(tmp_path / "Library" / "Logs" / "scout" / "watch.err.log")
    e = doc["EnvironmentVariables"]
    assert e["SCOUT_GATEWAY_URL"] == "https://gw.test"
    assert e["HOME"] == str(tmp_path)
    assert e["PATH"].startswith(
        "/opt/homebrew/bin"
    )  # a login shell's PATH is not launchd's: git and rg must resolve


def test_the_plist_never_carries_a_secret_or_the_fake_profile(tmp_path: Path) -> None:
    doc = plistlib.loads(
        launchd.render_plist(
            env(tmp_path, SCOUT_KEY="sk_secret", SCOUT_FAKE_CASE="/x", SCOUT_PROFILE="default")
        )
    )
    e = doc["EnvironmentVariables"]
    assert "SCOUT_KEY" not in e
    assert "SCOUT_FAKE_CASE" not in e
    assert "sk_secret" not in json.dumps(doc)
    assert e["SCOUT_PROFILE"] == "default"


def test_label_and_directory_are_overridable(tmp_path: Path) -> None:
    e = env(tmp_path, SCOUT_LAUNCHD_LABEL="dev.scout.test")
    assert launchd.plist_path(e) == tmp_path / "LaunchAgents" / "dev.scout.test.plist"
    assert plistlib.loads(launchd.render_plist(e))["Label"] == "dev.scout.test"
    default = {"HOME": str(tmp_path)}
    assert launchd.plist_path(default) == tmp_path / "Library" / "LaunchAgents" / "dev.scout.watch.plist"


@pytest.mark.parametrize("bad", ["../escape", "a/b", ".hidden", "", "has space", "x\0y"])
def test_a_label_that_names_a_path_is_refused(tmp_path: Path, bad: str) -> None:
    if not bad:
        assert (
            launchd.label(env(tmp_path, SCOUT_LAUNCHD_LABEL="")) == "dev.scout.watch"
        )  # empty means default
        return
    with pytest.raises(ScoutError) as ei:
        launchd.label(env(tmp_path, SCOUT_LAUNCHD_LABEL=bad))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


def test_install_writes_the_file_and_loads_it_through_the_runner(tmp_path: Path) -> None:
    run = Recorder()
    e = env(tmp_path)
    out = launchd.install(e, run)
    path = tmp_path / "LaunchAgents" / "dev.scout.watch.plist"
    assert out == {"installed": True, "label": "dev.scout.watch", "path": str(path), "loaded": True}
    assert plistlib.loads(path.read_bytes())["Label"] == "dev.scout.watch"
    assert path.stat().st_mode & 0o777 == 0o644
    domain = f"gui/{os.getuid()}"
    assert run.calls == [["bootout", f"{domain}/dev.scout.watch"], ["bootstrap", domain, str(path)]]
    assert (tmp_path / "Library" / "Logs" / "scout").is_dir()  # launchd will not create the log directory


def test_install_is_idempotent_and_replaces_a_stale_plist(tmp_path: Path) -> None:
    run = Recorder()
    path = tmp_path / "LaunchAgents" / "dev.scout.watch.plist"
    path.parent.mkdir()
    path.write_text("stale")
    launchd.install(env(tmp_path), run)
    launchd.install(env(tmp_path), run)
    assert plistlib.loads(path.read_bytes())["RunAtLoad"] is True
    assert [c[0] for c in run.calls] == ["bootout", "bootstrap", "bootout", "bootstrap"]
    assert list(path.parent.iterdir()) == [path]  # no temp files left behind


def test_a_failing_launchctl_leaves_the_plist_and_reports_why(tmp_path: Path) -> None:
    out = launchd.install(
        env(tmp_path), Recorder(bootstrap_rc=5, stderr="Bootstrap failed: 5: Input/output error\nmore")
    )
    assert out["installed"] is True
    assert out["loaded"] is False
    assert out["reason"] == "Bootstrap failed: 5: Input/output error"
    assert (tmp_path / "LaunchAgents" / "dev.scout.watch.plist").exists()


def test_uninstall_unloads_and_removes(tmp_path: Path) -> None:
    run = Recorder()
    launchd.install(env(tmp_path), run)
    run.calls.clear()
    out = launchd.uninstall(env(tmp_path), run)
    assert out["removed"] is True
    assert run.calls == [["bootout", f"gui/{os.getuid()}/dev.scout.watch"]]
    assert not (tmp_path / "LaunchAgents" / "dev.scout.watch.plist").exists()
    run.calls.clear()
    assert launchd.uninstall(env(tmp_path), run)["removed"] is False  # nothing to do: no launchctl call
    assert run.calls == []


def test_no_daemon_skips_the_install(tmp_path: Path) -> None:
    """`login --no-daemon`: nothing is written and launchctl is never called."""
    run = Recorder()
    out = launchd.install_for_login(env(tmp_path), enabled=False, profile="default", runner=run)
    assert out == {"installed": False, "reason": "--no-daemon"}
    assert run.calls == []
    assert not (tmp_path / "LaunchAgents").exists()


def test_login_installs_unless_disabled_or_fake(tmp_path: Path) -> None:
    run = Recorder()
    e = env(tmp_path, SCOUT_LAUNCHCTL="/stub")  # lets the install path run on any OS
    fake = launchd.install_for_login(e, enabled=True, profile="fake", runner=run)
    assert fake["installed"] is False
    assert run.calls == []  # a fake-profile login must never load a daemon
    real = launchd.install_for_login(e, enabled=True, profile="default", runner=run)
    assert real["installed"] is True
    assert [c[0] for c in run.calls] == ["bootout", "bootstrap"]


def login(tmp_path: Path, *argv: str, stub: Path) -> subprocess.CompletedProcess[str]:
    """`scout login` over the fake gateway with `launchctl` stubbed (the fake profile never installs)."""
    full = {
        **os.environ,
        **env(tmp_path, SCOUT_LAUNCHCTL=str(stub)),
        "SCOUT_KEYSTORE": "file",
        "SCOUT_PROFILE": "fake",
        "SCOUT_FAKE_CASE": str(CASE),
    }
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", "login", "--key", "sk_x", *argv],
        capture_output=True, text=True, env=full, check=False,
    )  # fmt: skip


def test_login_command_reports_the_daemon_and_honours_no_daemon(tmp_path: Path) -> None:
    stub = tmp_path / "launchctl"
    stub.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/launchctl.log\n')
    stub.chmod(0o755)
    p = login(tmp_path, "--no-daemon", stub=stub)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["daemon"] == {"installed": False, "reason": "--no-daemon"}
    assert not (tmp_path / "launchctl.log").exists()
    assert not (tmp_path / "LaunchAgents").exists()


def test_the_daemon_command_installs_and_removes_with_the_stub_launchctl(tmp_path: Path) -> None:
    stub = tmp_path / "launchctl"
    stub.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/launchctl.log\n')
    stub.chmod(0o755)
    full = {**os.environ, **env(tmp_path, SCOUT_LAUNCHCTL=str(stub)), "SCOUT_KEYSTORE": "file"}

    def daemon(action: str) -> dict[str, object]:
        p = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "scout", "daemon", action],
            capture_output=True,
            text=True,
            env=full,
            check=False,
        )
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)  # type: ignore[no-any-return]

    assert daemon("install")["loaded"] is True
    plist = tmp_path / "LaunchAgents" / "dev.scout.watch.plist"
    assert plistlib.loads(plist.read_bytes())["Label"] == "dev.scout.watch"
    assert "bootstrap" in (tmp_path / "launchctl.log").read_text()
    assert daemon("uninstall")["removed"] is True
    assert not plist.exists()
