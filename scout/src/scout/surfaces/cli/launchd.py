"""launchd user daemon for `scout watch` (design section 11 non-Claude hosts, FR-I13, AC-8.4).

A host without a SessionStart hook still needs a fresh index, so `scout login` installs a per-user
LaunchAgent, `dev.scout.watch`, that runs `scout watch --all`: one watcher per checkout registered with
`scout watch --add` (see `admin.py`). `RunAtLoad` starts it at login and `KeepAlive` restarts it if it dies.

- The plist is rendered with `plistlib`, so it is well-formed by construction (tests also `plutil -lint` it).
- `launchctl` runs behind an injectable `Runner`; `SCOUT_LAUNCHCTL` swaps the executable (subprocess tests),
  `SCOUT_LAUNCHD_LABEL` / `SCOUT_LAUNCHD_DIR` move the label and the LaunchAgents directory.
- The agent's environment is a whitelist of `SCOUT_*` settings plus a PATH that can find `git` and `rg`
  (launchd starts jobs with a minimal one). Secrets (`SCOUT_KEY`) are never copied: the daemon reads the
  key store.
"""

import os
import plistlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scout.core.contracts.errors import ErrorCode, ScoutError

DEFAULT_LABEL = "dev.scout.watch"
THROTTLE_S = 30  # launchd's minimum gap between restarts: a crash loop must not spin
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# `SCOUT_*` settings the daemon inherits; not `SCOUT_KEY` or `SCOUT_FAKE_CASE` (the daemon is the real thing).
PASS_THROUGH = (
    "SCOUT_GATEWAY_URL",
    "SCOUT_CONFIG",
    "SCOUT_ROOTS_FILE",
    "SCOUT_KEYSTORE",
    "SCOUT_PROFILE",
    "SCOUT_ROOT",
)
SAFE_PATH = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin")


@dataclass(frozen=True, slots=True)
class RunResult:
    returncode: int
    stderr: str = ""


type Runner = Callable[[Sequence[str]], RunResult]


def subprocess_runner(executable: str = "launchctl") -> Runner:
    def run(args: Sequence[str]) -> RunResult:
        try:
            p = subprocess.run(  # noqa: S603
                [executable, *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return RunResult(127, type(e).__name__)
        return RunResult(p.returncode, p.stderr.strip())

    return run


def label(env: Mapping[str, str]) -> str:
    value = env.get("SCOUT_LAUNCHD_LABEL") or DEFAULT_LABEL
    if not _LABEL.match(value):  # it names a file: no separators, no leading dot
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"invalid launchd label {value!r}")
    return value


def _home(env: Mapping[str, str]) -> Path:
    return Path(env["HOME"]) if env.get("HOME") else Path.home()


def agents_dir(env: Mapping[str, str]) -> Path:
    return (
        Path(env["SCOUT_LAUNCHD_DIR"])
        if env.get("SCOUT_LAUNCHD_DIR")
        else _home(env) / "Library" / "LaunchAgents"
    )


def plist_path(env: Mapping[str, str]) -> Path:
    return agents_dir(env) / f"{label(env)}.plist"


def log_dir(env: Mapping[str, str]) -> Path:
    return _home(env) / "Library" / "Logs" / "scout"


def program_arguments() -> list[str]:
    """The interpreter that runs this code, by absolute path: stable under `uv tool` and `uv run`, no PATH."""
    return [sys.executable, "-m", "scout", "watch", "--all"]


def agent_environment(env: Mapping[str, str]) -> dict[str, str]:
    out = {k: env[k] for k in PASS_THROUGH if env.get(k)}
    out["PATH"] = ":".join(dict.fromkeys([*SAFE_PATH, *[p for p in env.get("PATH", "").split(":") if p]]))
    out["HOME"] = str(_home(env))
    return out


def render_plist(env: Mapping[str, str], arguments: Sequence[str] | None = None) -> bytes:
    logs = log_dir(env)
    doc: dict[str, Any] = {
        "Label": label(env),
        "ProgramArguments": list(arguments if arguments is not None else program_arguments()),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": THROTTLE_S,
        "ProcessType": "Background",
        "StandardOutPath": str(logs / "watch.out.log"),
        "StandardErrorPath": str(logs / "watch.err.log"),
        "EnvironmentVariables": agent_environment(env),
    }
    return plistlib.dumps(doc, fmt=plistlib.FMT_XML, sort_keys=False)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".plist-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)  # launchd refuses an agent plist that is group- or world-writable
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _domain() -> str:
    return f"gui/{os.getuid()}"


def make_runner(env: Mapping[str, str]) -> Runner:
    return subprocess_runner(env.get("SCOUT_LAUNCHCTL") or "launchctl")


def install(
    env: Mapping[str, str], runner: Runner | None = None, arguments: Sequence[str] | None = None
) -> dict[str, Any]:
    """Write the plist and (re)load the agent. Never raises for a launchctl failure: the plist is on disk and
    the next login retries, so the result says `loaded: false` with the reason."""
    run = runner or make_runner(env)
    path = plist_path(env)
    log_dir(env).mkdir(parents=True, exist_ok=True)
    _write_atomic(path, render_plist(env, arguments))
    run(
        ["bootout", f"{_domain()}/{label(env)}"]
    )  # a stale copy would make bootstrap fail: ignore "not loaded"
    res = run(["bootstrap", _domain(), str(path)])
    out: dict[str, Any] = {
        "installed": True,
        "label": label(env),
        "path": str(path),
        "loaded": res.returncode == 0,
    }
    if res.returncode != 0:
        out["reason"] = (res.stderr.splitlines() or [f"launchctl exited {res.returncode}"])[0][:200]
    return out


def uninstall(env: Mapping[str, str], runner: Runner | None = None) -> dict[str, Any]:
    run = runner or make_runner(env)
    path = plist_path(env)
    existed = path.exists()
    if existed:
        run(["bootout", f"{_domain()}/{label(env)}"])
        path.unlink(missing_ok=True)
    return {"installed": False, "label": label(env), "path": str(path), "removed": existed}


def install_for_login(
    env: Mapping[str, str], *, enabled: bool, profile: str, runner: Runner | None = None
) -> dict[str, Any]:
    """What `scout login` does about the daemon: install unless `--no-daemon`, off macOS, or the fake profile
    (the daemon runs the real profile, so a fake-profile login must never load one)."""
    if not enabled:
        return {"installed": False, "reason": "--no-daemon"}
    if sys.platform != "darwin" and not env.get("SCOUT_LAUNCHCTL"):
        return {
            "installed": False,
            "reason": "launchd is macOS only: run `scout watch --all` under your service manager",
        }
    if profile == "fake":
        return {"installed": False, "reason": "profile fake has no daemon"}
    return install(env, runner)
