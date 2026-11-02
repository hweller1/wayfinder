"""Lane choice and readiness for the scout-wayfinder Muse plugin. Stdlib only; shared by the MCP launcher and
the two hooks so they always agree.

Muse Code (1.4.14) starts plugin hooks and plugin MCP servers with a filtered environment: HOME, PATH, the
MUSE_PLUGIN_* / PLUGIN_* variables and little else (measured). A `SCOUT_KEY` exported in the shell never
arrives, so configuration comes from files, first hit wins per variable:
  1. `$MUSE_PLUGIN_DATA_DIR/scout.env`  (per install, in Muse's data dir; `scout init --muse-code` writes it)
  2. `$HOME/.config/scout/muse.env`
then the process environment (useful under `muse plugins hook test` and in unit tests). Only the names in
`ALLOWED` are read; values are never printed. A key is referenced by file (`SCOUT_KEY_FILE`), or comes from
`scout login` (macOS Keychain or `~/.config/scout/credentials`).

Lanes:
  production  `scout mcp` against the hosted gateway; needs a key and a `scout` binary.
  demo        `scout-demo mcp` (in-process gateway, repos registered with `scout-demo index`); the fallback.
`SCOUT_LANE=production|demo` forces one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ALLOWED = (
    "SCOUT_LANE",
    "SCOUT_BIN",
    "SCOUT_KEY_FILE",
    "SCOUT_KEYSTORE",
    "SCOUT_GATEWAY_URL",
    "SCOUT_DEMO_PROJECT",
    "SCOUT_DEMO_HOME",
    "SCOUT_DEMO_PRESET",
    "SCOUT_RENDER_MAX_TOKENS",
    "SCOUT_NO_WARM",
    "SCOUT_NO_WATCH",
    "SCOUT_ROUTE",
    "SCOUT_SPEC_TARGET",
    "SCOUT_SPEC_DISABLE",
    "SCOUT_SPEC_FILE",
)
READY_STATES = ("ready", "partial")
EXPLORER = "scout-wayfinder/scout-explorer"


def _parse(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if name in ALLOWED and value and name not in out:
            out[name] = value
    return out


def config_files(env: dict[str, str]) -> list[Path]:
    files = []
    if env.get("MUSE_PLUGIN_DATA_DIR"):
        files.append(Path(env["MUSE_PLUGIN_DATA_DIR"]) / "scout.env")
    home = env.get("HOME") or str(Path.home())
    files.append(Path(home) / ".config" / "scout" / "muse.env")
    return files


def load_config(env: dict[str, str] | None = None) -> dict[str, str]:
    """The allowlisted settings: config files first (first file wins), then the environment."""
    environ = dict(os.environ if env is None else env)
    cfg: dict[str, str] = {}
    for f in config_files(environ):
        for k, v in _parse(f).items():
            cfg.setdefault(k, v)
    for k in ALLOWED:
        if environ.get(k) and k not in cfg:
            cfg[k] = environ[k]
    return cfg


def read_key_file(cfg: dict[str, str]) -> str | None:
    path = cfg.get("SCOUT_KEY_FILE")
    if not path:
        return None
    try:
        key = Path(path).expanduser().read_text(encoding="utf-8").strip().split()[0]
    except (OSError, IndexError, UnicodeDecodeError):
        return None
    return key or None


def keychain_has_key(cfg: dict[str, str]) -> bool:
    """`scout login` stores the key in the macOS Keychain (service scout, account scout_key). Attributes only:
    no -w/-g, so the secret is never read and no prompt can appear."""
    if sys.platform != "darwin" or cfg.get("SCOUT_KEYSTORE") == "file":
        return False
    try:
        proc = subprocess.run(  # noqa: S603
            ["security", "find-generic-password", "-s", "scout", "-a", "scout_key"],  # noqa: S607
            capture_output=True,
            timeout=1,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def has_key(cfg: dict[str, str], env: dict[str, str]) -> bool:
    if read_key_file(cfg) or env.get("SCOUT_KEY") or env.get("SCOUT_API_KEY"):
        return True
    home = env.get("HOME") or str(Path.home())
    if (Path(home) / ".config" / "scout" / "credentials").is_file():
        return True
    return keychain_has_key(cfg)


def scout_bin(cfg: dict[str, str], env: dict[str, str]) -> str | None:
    explicit = cfg.get("SCOUT_BIN")
    if explicit and os.access(explicit, os.X_OK):
        return explicit
    found = shutil.which("scout", path=env.get("PATH"))
    if found:
        return found
    home = env.get("HOME") or str(Path.home())
    local = Path(home) / ".local" / "bin" / "scout"
    return str(local) if os.access(local, os.X_OK) else None


def demo_project(cfg: dict[str, str], env: dict[str, str], here: Path) -> str | None:
    """The voyage-scout checkout that has `scout-demo`: config, ~/.scout-demo/project, or the checkout this
    bundle sits in when it is used from source (integrations/muse-code/scout-wayfinder/lib)."""
    if cfg.get("SCOUT_DEMO_PROJECT"):
        return cfg["SCOUT_DEMO_PROJECT"]
    home = env.get("HOME") or str(Path.home())
    pointer = Path(home) / ".scout-demo" / "project"
    try:
        first = pointer.read_text(encoding="utf-8").splitlines()[0].strip()
        if first:
            return first
    except (OSError, IndexError):
        pass
    for parent in here.resolve().parents:
        pp = parent / "pyproject.toml"
        try:
            if pp.is_file() and "scout-demo" in pp.read_text(encoding="utf-8"):
                return str(parent)
        except OSError:
            continue
    return None


def pick_lane(cfg: dict[str, str], env: dict[str, str]) -> str:
    forced = cfg.get("SCOUT_LANE", "")
    if forced in ("production", "demo"):
        return forced
    return "production" if has_key(cfg, env) and scout_bin(cfg, env) else "demo"


def git_paths(cwd: str) -> tuple[str, str] | None:
    """`(toplevel, scout state dir)` from one git call, or None outside a repo."""
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", cwd, "rev-parse", "--show-toplevel", "--git-path", "scout"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.stdout.splitlines()
    if out.returncode != 0 or len(lines) < 2:
        return None
    state = lines[1] if os.path.isabs(lines[1]) else os.path.join(cwd, lines[1])
    return lines[0], os.path.normpath(state)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def production_status(state_dir: str) -> dict:
    """`status.json` written by `scout sync` / `scout watch`; `{"state": "absent"}` when missing."""
    return _read_json(Path(state_dir) / "status.json") or {"state": "absent"}


def demo_repo(top: str, cfg: dict[str, str], env: dict[str, str]) -> dict | None:
    """The demo registry record whose checkout is `top` (with its `name`), else None."""
    home = cfg.get("SCOUT_DEMO_HOME") or os.path.join(env.get("HOME") or str(Path.home()), ".scout-demo")
    reg = _read_json(Path(home) / "repos.json") or {}
    real = os.path.realpath(top)
    for name, rec in sorted(reg.items()):
        if (
            isinstance(rec, dict)
            and isinstance(rec.get("root"), str)
            and os.path.realpath(rec["root"]) == real
        ):
            return {**rec, "name": name}
    return None


def readiness(cwd: str, env: dict[str, str] | None = None) -> dict:
    """What the hooks need, without network: lane, whether scout can answer for this checkout, and why not."""
    environ = dict(os.environ if env is None else env)
    cfg = load_config(environ)
    lane = pick_lane(cfg, environ)
    out: dict = {"lane": lane, "ready": False, "reason": "", "repo": None, "coverage": None, "files": None}
    paths = git_paths(cwd)
    if paths is None:
        out["reason"] = "not a git checkout"
        return out
    top, state_dir = paths
    out["root"] = top
    if lane == "production":
        st = production_status(state_dir)
        out.update(coverage=st.get("coverage"), files=st.get("files"), state=st.get("state"))
        if st.get("state") in READY_STATES:
            out["ready"] = True
        else:
            out["reason"] = "index " + str(st.get("state")) + ": run `scout sync --wait` in the checkout"
        return out
    rec = demo_repo(top, cfg, environ)
    if rec and rec.get("state") in READY_STATES:
        out.update(ready=True, repo=rec["name"], coverage=rec.get("coverage"), state=rec.get("state"))
    else:
        out["reason"] = "checkout not indexed in the demo lane: run `scout-demo index <path>`"
    return out
