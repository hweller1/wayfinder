#!/usr/bin/env python3
# ruff: noqa: S603, S607  (fixed argv: git, security, scout; no shell)
"""Shared logic of the scout Claude Code plugin: lane, readiness, settings, warm-up and the slash-command
reports. Stdlib only and Python 3.9+ (hooks run under whatever `python3` the user has), never imports scout.

Used three ways:
  * imported by the hooks (`rewrite_explore.py`, `session_start.py`, `allow_explore_tool.py`);
  * run by the slash commands: `python3 scout_cc.py <cmd> --data "$CLAUDE_PLUGIN_DATA" [args]`
    (`status`, `index`, `model`, `mode`, `doctor`, `statusline`, `gateway-url`);
  * run detached by the SessionStart hook: `scout_cc.py warm-worker` (gateway probe, then `scout warm`).

State it reads (never writes):  `<git-path scout>/status.json` and `mcp.pid` (written by `scout`).
State it owns, under the plugin data dir (`${CLAUDE_PLUGIN_DATA}`):
  settings.json  per-project `{"model": "v3"|"a1", "mode": "hybrid"|"fast"|"off"}`
  warm.json      per-gateway last warm-up: state, timestamps, seconds, error code (never a key or a brief)
  hook.log       routing decisions (lane, target, reason; never the brief)
  watch-*.pid, index-*.log, index-*.pid
No function here prints or logs a key: only whether one is configured.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLUGIN = "scout"
SERVER = "scout"
TOOL_PREFIX = f"mcp__plugin_{PLUGIN}_{SERVER}__"
TARGETS = {"hybrid": f"{PLUGIN}:scout-explorer", "fast": f"{PLUGIN}:super-explore"}
MODES = ("hybrid", "fast", "off")
DEFAULT_MODE = "hybrid"
# Gateway profiles: same keys and index, different chat model behind /v1/chat (docs/PRODUCT_STATUS.md).
GATEWAYS = {
    "v3": "https://scout-gateway.example.com",
    "a1": "https://scout-gateway-a1.example.com",
}
MODEL_LABEL = {"v3": "wayfinder-glimmer (Muse-Glimmer-30B SFT)", "a1": "Agents-A1-35B"}
# Display-name aliases accepted by /scout:model and SCOUT_MODEL; the gateway profile id stays `v3`.
MODEL_ALIASES = {"glimmer": "v3", "wayfinder-glimmer": "v3"}
DEFAULT_MODEL = "v3"
GATEWAY_FRESH_S = 600.0  # status.json gateway_ok_at must be younger than this (same rule as core readiness)
WARM_FRESH_S = 600.0  # wayfinder-v3-serve scaledown window: a model answered within this is still warm
WARM_TIMEOUT_S = 900
READY_STATES = ("ready", "partial")
CHARS_PER_TOKEN = 3.5  # the sync engine's estimate (core/sync/engine.py)
# voyage-code list price per 1M tokens (an upper-bound estimate: the tenant cache skips unchanged chunks)
EMBED_USD_PER_M = 0.18
UTC = timezone.utc  # noqa: UP017  (Python 3.9 has no datetime.UTC)
MAX_FILE_BYTES = 1_048_576  # files above this are not indexed (core/local/ignore.py)


# --- small helpers ------------------------------------------------------------------------------------
def read_json(path: str | Path) -> dict[str, Any] | None:
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """tmp file + rename: a concurrent reader sees the old or the new file, never half of one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True)
    os.replace(tmp, path)


def epoch(ts: object) -> float | None:
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()


def stamp(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ago(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    s = int(max(0, seconds))
    if s < 90:
        return f"{s}s ago"
    if s < 5400:
        return f"{s // 60}m ago"
    if s < 172800:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def data_dir(env: dict[str, str], explicit: str | None = None) -> Path:
    raw = (
        explicit or env.get("CLAUDE_PLUGIN_DATA") or os.path.join(tempfile.gettempdir(), "scout-claude-code")
    )
    return Path(raw)


def project_key(root: str) -> str:
    """Stable short id of a project path (file names under the data dir)."""
    import hashlib

    return hashlib.sha256(os.path.realpath(root).encode("utf-8")).hexdigest()[:12]


def log_decision(env: dict[str, str], line: str) -> None:
    raw = env.get("CLAUDE_PLUGIN_DATA")
    if not raw:
        return
    try:
        os.makedirs(raw, exist_ok=True)
        with open(os.path.join(raw, "hook.log"), "a", encoding="utf-8") as f:
            f.write(f"{stamp()} {line}\n")
    except OSError:
        pass


# --- keys, binaries, git ------------------------------------------------------------------------------
def keychain_has_key(env: dict[str, str]) -> bool:
    """macOS only: `scout login` stores the key in the Keychain (service `scout`, account `scout_key`).
    Attributes only (no `-w`/`-g`): the secret is never read and no prompt can appear."""
    if sys.platform != "darwin" or env.get("SCOUT_KEYSTORE") == "file" or env.get("SCOUT_NO_KEYCHAIN_PROBE"):
        return False
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", "scout", "-a", "scout_key"],
            capture_output=True,
            timeout=1,
            check=False,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def creds_present(env: dict[str, str]) -> bool:
    """A scout key is configured: `$SCOUT_KEY` / `$SCOUT_API_KEY`, `~/.config/scout/credentials` or the
    macOS Keychain item. Same rule as scripts/launch-scout.sh."""
    if env.get("SCOUT_KEY") or env.get("SCOUT_API_KEY"):
        return True
    home = env.get("HOME") or str(Path.home())
    if os.path.isfile(os.path.join(home, ".config", "scout", "credentials")):
        return True
    return keychain_has_key(env)


def scout_bin(env: dict[str, str]) -> str | None:
    explicit = env.get("SCOUT_BIN")
    if explicit and os.access(explicit, os.X_OK):
        return explicit
    path = (
        env.get("PATH", "")
        + os.pathsep
        + os.pathsep.join(
            [
                "/opt/homebrew/bin",
                "/usr/local/bin",
                os.path.join(env.get("HOME") or str(Path.home()), ".local/bin"),
            ]
        )
    )
    return shutil.which("scout", path=path)


def lane(env: dict[str, str]) -> str:
    forced = env.get("SCOUT_LANE")
    if forced in ("demo", "production"):
        return forced
    return "production" if creds_present(env) else "demo"


def git_paths(cwd: str) -> tuple[str, str] | None:
    """`(toplevel, state_dir)` from one git call, or None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel", "--git-path", "scout"],
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


def git_head(top: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", top, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def mcp_alive(state_dir: str) -> bool:
    data = read_json(os.path.join(state_dir, "mcp.pid"))
    return pid_alive(data.get("pid") if data else None)


# --- settings (model profile, routing mode) ----------------------------------------------------------
def load_settings(data: Path, top: str) -> dict[str, str]:
    doc = read_json(data / "settings.json") or {}
    proj = (
        doc.get("projects", {}).get(os.path.realpath(top), {})
        if isinstance(doc.get("projects"), dict)
        else {}
    )
    return {k: v for k, v in proj.items() if isinstance(v, str)} if isinstance(proj, dict) else {}


def save_setting(data: Path, top: str, name: str, value: str) -> None:
    path = data / "settings.json"
    doc = read_json(path) or {}
    projects = doc.get("projects") if isinstance(doc.get("projects"), dict) else {}
    proj = projects.get(os.path.realpath(top)) or {}
    proj[name] = value
    projects[os.path.realpath(top)] = proj
    doc["projects"] = projects
    write_json(path, doc)


def model_name(env: dict[str, str], data: Path, top: str) -> str:
    """`SCOUT_MODEL` (v3|glimmer|a1) > the project setting from /scout:model > v3 (wayfinder-glimmer)."""
    m = env.get("SCOUT_MODEL") or load_settings(data, top).get("model") or DEFAULT_MODEL
    m = MODEL_ALIASES.get(m.strip().lower(), m) if isinstance(m, str) else m
    return m if m in GATEWAYS else DEFAULT_MODEL


def mode_name(env: dict[str, str], data: Path, top: str) -> str:
    """`SCOUT_EXPLORE_MODE` (hybrid|fast|off) > the project setting from /scout:mode > hybrid."""
    m = env.get("SCOUT_EXPLORE_MODE") or load_settings(data, top).get("mode") or DEFAULT_MODE
    return m if m in MODES else DEFAULT_MODE


def _config_toml_gateway(env: dict[str, str]) -> str | None:
    path = env.get("SCOUT_CONFIG") or os.path.join(
        env.get("HOME") or str(Path.home()), ".config/scout/config.toml"
    )
    try:
        import tomllib  # 3.11+

        with open(path, "rb") as f:
            val = tomllib.load(f).get("gateway_url")
    except (ImportError, OSError, ValueError):
        return None
    return val.strip() if isinstance(val, str) and val.strip() else None


def gateway(env: dict[str, str], data: Path, top: str) -> tuple[str, str, str]:
    """`(url, model, source)`. `SCOUT_GATEWAY_URL` wins (model then reads `custom` unless it is a known URL),
    then an explicit model choice (`SCOUT_MODEL`, /scout:model), then scout's config.toml, then v3."""
    explicit = (env.get("SCOUT_GATEWAY_URL") or "").strip()
    if explicit:
        known = next((m for m, u in GATEWAYS.items() if u.rstrip("/") == explicit.rstrip("/")), "custom")
        return explicit, known, "SCOUT_GATEWAY_URL"
    model = model_name(env, data, top)
    if env.get("SCOUT_MODEL") or load_settings(data, top).get("model"):  # an explicit choice, v3 included
        return GATEWAYS[model], model, "model profile"
    cfg = _config_toml_gateway(env)
    if cfg:
        known = next((m for m, u in GATEWAYS.items() if u.rstrip("/") == cfg.rstrip("/")), "custom")
        return cfg, known, "config.toml"
    return GATEWAYS[DEFAULT_MODEL], DEFAULT_MODEL, "default"


def child_env(env: dict[str, str], data: Path, top: str) -> dict[str, str]:
    """Environment for a `scout` child: the session's, plus the gateway of the project's model profile."""
    url, _, source = gateway(env, data, top)
    out = dict(env)
    if source == "model profile":
        out["SCOUT_GATEWAY_URL"] = url
    return out


# --- warm record ---------------------------------------------------------------------------------------
def warm_record(data: Path, url: str) -> dict[str, Any]:
    doc = read_json(data / "warm.json") or {}
    rec = doc.get(url.rstrip("/"))
    return rec if isinstance(rec, dict) else {}


def set_warm(data: Path, url: str, **fields: Any) -> None:
    path = data / "warm.json"
    doc = read_json(path) or {}
    key = url.rstrip("/")
    rec = doc.get(key) if isinstance(doc.get(key), dict) else {}
    rec.update(fields)
    doc[key] = rec
    write_json(path, doc)


def warm_state(rec: dict[str, Any], now: float) -> tuple[str, str]:
    """`(label, detail)` of the endpoint behind a gateway, from our own last warm-up (the client cannot see
    Modal's containers): warm = the model answered within the scaledown window; warming = a warm-up runs."""
    state = rec.get("state")
    fin = rec.get("finished_at") if isinstance(rec.get("finished_at"), (int, float)) else None
    if state == "warming" and pid_alive(rec.get("pid")):
        started = rec.get("started_at") if isinstance(rec.get("started_at"), (int, float)) else now
        return "warming", f"warm-up running for {int(now - started)}s (cold start is 3.5-5 min)"
    if state == "warm" and fin is not None:
        secs = rec.get("seconds")
        took = f", took {secs:.0f}s" if isinstance(secs, (int, float)) else ""
        if now - fin < WARM_FRESH_S:
            return "warm", f"model answered {ago(now - fin)}{took}"
        return "cold?", f"last answer {ago(now - fin)}; it scales to zero after 10 min idle"
    if state == "unreachable":
        return "unreachable", f"gateway did not answer {ago(now - fin) if fin else ''}".strip()
    if state == "failed":
        return (
            "failed",
            f"warm-up failed ({rec.get('error') or 'error'}) {ago(now - fin) if fin else ''}".strip(),
        )
    if state == "unsupported":
        return "unknown", "this scout has no `warm` command; upgrade scout"
    return "unknown", "no warm-up yet in this project"


# --- readiness and routing -----------------------------------------------------------------------------
def production_reason(state_dir: str, now: float, env: dict[str, str]) -> str | None:
    """None when the production lane can answer, else why not. Same rule as `core.sync.state.readiness`."""
    if not creds_present(env):
        return "no-creds"
    status = read_json(os.path.join(state_dir, "status.json"))
    if status is None:
        return "no-status"
    if status.get("state") not in READY_STATES:
        return "index-" + str(status.get("state"))
    ok_at = epoch(status.get("gateway_ok_at"))
    if ok_at is None or now - ok_at >= GATEWAY_FRESH_S:
        return "gateway-stale"
    if not mcp_alive(state_dir):
        return "mcp-down"
    return None


MODEL_DOWN_S = (
    600.0  # a failed model warm-up blocks routing this long, then routing (and a new warm-up) retries
)


def gateway_down(data: Path, url: str, state_dir: str, now: float | None = None) -> str | None:
    """Why scout cannot answer although the index looks ready, else None, from the last SessionStart probe:
    gateway-unreachable  /v1/healthz failed AFTER the watcher's last good ping;
    model-failed         the 1-token chat through the gateway failed (the model behind it is down or
                         rejecting requests; the watcher's ping does not reach the model) less than
                         MODEL_DOWN_S ago. Timeouts do not count: a cold start may simply be slow."""
    rec = warm_record(data, url)
    fin = rec.get("finished_at")
    if not isinstance(fin, (int, float)):
        return None
    if rec.get("state") == "unreachable":
        status = read_json(os.path.join(state_dir, "status.json")) or {}
        ok_at = epoch(status.get("gateway_ok_at")) or 0.0
        return "gateway-unreachable" if fin >= ok_at else None
    if rec.get("state") == "failed" and rec.get("error") not in (
        "timeout",
        "UPSTREAM_TIMEOUT",
        "no-scout-binary",
    ):
        return "model-failed" if (time.time() if now is None else now) - fin < MODEL_DOWN_S else None
    return None


def demo_repo(top: str, env: dict[str, str]) -> str | None:
    """Registry name of a ready demo index whose checkout is `top`, else None."""
    home = env.get("SCOUT_DEMO_HOME") or os.path.join(env.get("HOME") or str(Path.home()), ".scout-demo")
    reg = read_json(os.path.join(home, "repos.json"))
    if not reg:
        return None
    real = os.path.realpath(top)
    for name, rec in sorted(reg.items()):
        if (
            isinstance(rec, dict)
            and rec.get("state") in READY_STATES
            and isinstance(rec.get("root"), str)
            and os.path.realpath(rec["root"]) == real
        ):
            return str(name)
    return None


def spec_blocks_rewrite(cwd: str) -> bool:
    """True when the repo's resolved retrieval spec (lib/scout_spec.py) says Explore must stay stock:
    `[delivery] rewrite_explore = false` or `[index] enabled = false`. No spec file: False."""
    paths = git_paths(cwd)
    if paths is None:
        return False
    doc = read_json(os.path.join(paths[1], "spec_resolved.json"))
    if not doc or (doc.get("layers") or {}).get("disabled"):
        return False
    delivery = doc.get("delivery") if isinstance(doc.get("delivery"), dict) else {}
    return delivery.get("rewrite_explore") is False or doc.get("index_enabled") is False


def route(
    tool_input: dict[str, Any], cwd: str, now: float, env: dict[str, str], data: Path
) -> tuple[dict[str, Any] | None, str, str, str]:
    """`(updated_input | None, lane, target, reason)` for a built-in Explore call; None = stock Explore."""
    paths = git_paths(cwd)
    if paths is None:
        return None, "-", "-", "not-a-repo"
    top, state_dir = paths
    mode = mode_name(env, data, top)
    if mode == "off":
        return None, "-", "-", "mode-off"
    target = TARGETS[mode]
    reason = production_reason(state_dir, now, env)
    if reason is None:
        url, _, _ = gateway(env, data, top)
        reason = gateway_down(data, url, state_dir, now)
        if reason is None:
            return {**tool_input, "subagent_type": target}, "production", target, "ready"
    name = demo_repo(top, env) if lane(env) == "demo" or reason == "no-creds" else None
    if name is not None:
        prompt = str(tool_input.get("prompt", ""))
        updated = {**tool_input, "subagent_type": target, "prompt": f"[scout repo: {name}]\n{prompt}"}
        return updated, "demo", target, "ready"
    return None, "-", "-", reason


# --- background children ------------------------------------------------------------------------------
def spawn(argv: list[str], env: dict[str, str], log: Path | None = None, cwd: str | None = None) -> int:
    """Detached child (own session, all fds redirected) so neither a hook nor Claude Code waits on it."""
    out: Any = subprocess.DEVNULL
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        out = open(log, "ab")  # noqa: SIM115  handed to the child
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT if log else out,
            start_new_session=True, env=env, cwd=cwd,
        )  # fmt: skip
    finally:
        if log is not None:
            out.close()
    return proc.pid


def ensure_watcher(scout: str, top: str, env: dict[str, str], data: Path) -> bool:
    """`scout watch --ensure` once per project (pid file); True when one was started now."""
    pidfile = data / f"watch-{project_key(top)}.pid"
    try:
        old = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        old = 0
    if old and pid_alive(old):
        return False
    pid = spawn([scout, "watch", "--ensure", "--root", top], child_env(env, data, top))
    data.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(pid))
    return True


def start_warm(top: str, env: dict[str, str], data: Path) -> bool:
    """Start the warm worker for this project's gateway unless one already runs, the model is warm, or
    SCOUT_NO_WARM is set (session start, /scout:index, /scout:model and the routing hook all obey it)."""
    if env.get("SCOUT_NO_WARM"):
        return False
    url, _, _ = gateway(env, data, top)
    rec = warm_record(data, url)
    now = time.time()
    if rec.get("state") == "warming" and pid_alive(rec.get("pid")):
        return False
    fin = rec.get("finished_at")
    if rec.get("state") == "warm" and isinstance(fin, (int, float)) and now - fin < WARM_FRESH_S / 2:
        return False
    argv = [sys.executable, os.path.abspath(__file__), "warm-worker", "--project", top, "--data", str(data)]
    pid = spawn(argv, env, log=data / "warm.log")
    set_warm(data, url, state="warming", pid=pid, started_at=now)
    return True


def retry_warm(cwd: str, env: dict[str, str], data: Path, min_gap_s: float = 120.0) -> bool:
    """After a failed model warm-up, start another one if the last finished more than `min_gap_s` ago."""
    paths = git_paths(cwd)
    if paths is None:
        return False
    url, _, _ = gateway(env, data, paths[0])
    fin = warm_record(data, url).get("finished_at")
    if isinstance(fin, (int, float)) and time.time() - fin < min_gap_s:
        return False
    return start_warm(paths[0], env, data)


def healthz(url: str, timeout: float = 20.0) -> str | None:
    """None when `GET /v1/healthz` answers ok, else a short reason (the route is public: no key sent)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/healthz", timeout=timeout) as res:  # noqa: S310
            body = json.loads(res.read() or b"{}")
        return None if isinstance(body, dict) and body.get("ok") else "bad-healthz"
    except urllib.error.HTTPError as e:
        return f"http-{e.code}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return type(getattr(e, "reason", e)).__name__


def warm_worker(top: str, env: dict[str, str], data: Path) -> int:
    """Probe the gateway, then one 1-token chat through it (`scout warm`) so Wayfinder's cold start is paid
    before the first search. Records the outcome in warm.json. Prints nothing secret."""
    url, model, _ = gateway(env, data, top)
    set_warm(data, url, state="warming", pid=os.getpid(), started_at=time.time(), model=model)
    down = healthz(url)
    if down is not None:
        set_warm(data, url, state="unreachable", finished_at=time.time(), error=down, pid=None)
        print(f"{stamp()} warm {model}: gateway unreachable ({down})", flush=True)
        return 0
    scout = scout_bin(env)
    if scout is None:
        set_warm(data, url, state="failed", finished_at=time.time(), error="no-scout-binary", pid=None)
        return 0
    t0 = time.time()
    try:
        proc = subprocess.run(
            [scout, "warm", "--root", top], capture_output=True, text=True, timeout=WARM_TIMEOUT_S,
            check=False, env=child_env(env, data, top), stdin=subprocess.DEVNULL,
        )  # fmt: skip
        out = proc.stdout.strip().splitlines()[-1:] or [""]
        doc = json.loads(out[0]) if out[0].startswith("{") else {}
    except subprocess.TimeoutExpired:
        set_warm(data, url, state="failed", finished_at=time.time(), error="timeout", pid=None)
        return 0
    except (OSError, ValueError):
        doc = {}
        proc = None
    if doc.get("warm"):
        secs = float(doc.get("seconds") or time.time() - t0)
        set_warm(
            data, url, state="warm", finished_at=time.time(), seconds=round(secs, 1), error=None, pid=None
        )
        print(f"{stamp()} warm {model}: ok in {secs:.1f}s", flush=True)
        return 0
    code = (doc.get("error") or {}).get("code") if isinstance(doc.get("error"), dict) else None
    msg = ((doc.get("error") or {}).get("message") or "") if isinstance(doc.get("error"), dict) else ""
    if code == "INVALID_ARGUMENT" and "warm" in msg:
        set_warm(data, url, state="unsupported", finished_at=time.time(), error=code, pid=None)
    elif code in ("UPSTREAM_UNAVAILABLE",) and "unreachable" in msg:
        set_warm(data, url, state="unreachable", finished_at=time.time(), error=code, pid=None)
    else:
        rc = proc.returncode if proc is not None else -1
        set_warm(data, url, state="failed", finished_at=time.time(), error=code or f"rc={rc}", pid=None)
    print(f"{stamp()} warm {model}: {code or 'failed'}", flush=True)
    return 0


# --- reports for the slash commands ---------------------------------------------------------------------
def _project(args: argparse.Namespace, env: dict[str, str]) -> tuple[str, str] | None:
    return git_paths(args.project or env.get("CLAUDE_PROJECT_DIR") or os.getcwd())


def status_report(top: str, state_dir: str, env: dict[str, str], data: Path, now: float) -> list[str]:
    ln = lane(env)
    url, model, source = gateway(env, data, top)
    mode = mode_name(env, data, top)
    status = read_json(os.path.join(state_dir, "status.json"))
    lines = [f"scout status for {top}"]
    key = "configured" if creds_present(env) else "not configured (`scout login`, or SCOUT_KEY)"
    lines.append(f"  lane       {ln}   key {key}")
    if ln == "production":
        lines.append(f"  model      {model} - {MODEL_LABEL.get(model, url)}   ({source})")
        lines.append(f"  gateway    {url}")
        label, detail = warm_state(warm_record(data, url), now)
        lines.append(f"  endpoint   {label} - {detail}")
    if status is None:
        lines.append("  index      none for this checkout - run /scout:index")
    else:
        head = git_head(top)
        commit = status.get("commit")
        fresh = (
            "at HEAD"
            if commit and head and commit == head
            else f"indexed {str(commit)[:8]}, HEAD {str(head)[:8]}"
        )
        cov = status.get("coverage")
        cov_s = f"{float(cov):.0%}" if isinstance(cov, (int, float)) else "?"
        syncing = "  (sync running)" if status.get("syncing") else ""
        lines.append(
            f"  index      {status.get('state')} {cov_s}, {fresh}, "
            f"last sync {ago(_age(status.get('last_sync_at'), now))}{syncing}"
        )
        lines.append(
            f"  gateway ok {ago(_age(status.get('gateway_ok_at'), now))} (watcher ping; must be under 10m)"
        )
        if status.get("last_error"):
            lines.append(f"  last error {status.get('last_error')}")
    lines.append(f"  mcp server {'up' if mcp_alive(state_dir) else 'down'}")
    target = TARGETS.get(mode, "-")
    _, rlane, rtarget, reason = route({"subagent_type": "Explore"}, top, now, env, data)
    if rtarget != "-":
        lines.append(f"  routing    mode {mode}: Explore -> {rtarget} ({rlane} lane)")
    else:
        lines.append(
            f"  routing    mode {mode}: Explore runs built-in (scout not ready: {reason}); "
            f"target when ready {target}"
        )
    return lines


def _age(ts: object, now: float) -> float | None:
    t = epoch(ts)
    return None if t is None else now - t


def estimate(top: str) -> tuple[int, int]:
    """`(files, tokens)` the first index of this checkout would embed (upper bound: unchanged chunks hit the
    tenant cache, and the sync engine also skips ignored, secret and binary files)."""
    try:
        out = subprocess.run(
            ["git", "-C", top, "ls-files", "-z"], capture_output=True, timeout=30, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0, 0
    files = total = 0
    for rel in out.split(b"\0"):
        if not rel:
            continue
        try:
            size = os.path.getsize(os.path.join(top.encode(), rel))
        except OSError:
            continue
        if 0 < size <= MAX_FILE_BYTES:
            files += 1
            total += size
    return files, int(total / CHARS_PER_TOKEN)


def index(
    top: str, state_dir: str, env: dict[str, str], data: Path, wait_s: float, poll_s: float = 2.0
) -> list[str]:
    scout = scout_bin(env)
    if lane(env) != "production" or scout is None:
        return [
            "scout: production lane not available (no scout key or no `scout` binary).",
            "  Install: uv tool install <voyage-scout>; then `scout login --key <key>` (Keychain).",
            "  Demo lane instead: uv run scout-demo index <this checkout> --name <name>",
        ]
    files, tokens = estimate(top)
    usd = tokens / 1e6 * EMBED_USD_PER_M
    url, model, _ = gateway(env, data, top)
    lines = [
        f"scout index: {top}",
        f"  gateway {url} (model {model})",
        f"  estimate {files} files, ~{tokens / 1e6:.2f}M tokens, at most ~${usd:.2f} of embedding "
        "(unchanged chunks are cached; code is chunked locally, paths encrypted)",
    ]
    key = project_key(top)
    log = data / f"index-{key}.log"
    pidfile = data / f"index-{key}.pid"
    try:
        old = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        old = 0
    if old and pid_alive(old):
        lines.append(f"  a sync is already running (pid {old}); progress:")
        pid = old
    else:
        with contextlib.suppress(OSError):
            log.unlink()
        pid = spawn([scout, "sync", "--wait", "--root", top], child_env(env, data, top), log=log)
        data.mkdir(parents=True, exist_ok=True)
        pidfile.write_text(str(pid))
        lines.append(f"  sync started (pid {pid})")
    t0 = time.time()
    last_report = -1.0
    while pid_alive(pid) and not _reaped(pid) and time.time() - t0 < wait_s:
        el = time.time() - t0
        if el - last_report >= 15:
            st = read_json(os.path.join(state_dir, "status.json")) or {}
            cov = st.get("coverage")
            lines.append(f"  {el:4.0f}s  syncing  index {st.get('state', 'absent')} {float(cov or 0):.0%}")
            last_report = el
        time.sleep(poll_s)
    el = time.time() - t0
    if pid_alive(pid) and not _reaped(pid):
        lines.append(
            f"  still indexing after {el:.0f}s; it continues in the background. /scout:status shows progress."
        )
        return lines
    result = _last_json(log)
    if result is None:
        lines.append(f"  sync ended after {el:.0f}s without a result; log: {log}")
        return lines
    if "error" in result:
        err = result["error"] if isinstance(result["error"], dict) else {}
        lines.append(f"  sync failed: {err.get('code')}: {err.get('message')}")
        return lines
    lines.append(
        f"  done in {el:.0f}s: index {result.get('state')} {float(result.get('coverage') or 0):.0%}, "
        f"{result.get('files')} files, commit {str(result.get('commit'))[:8]}"
    )
    if result.get("state") in READY_STATES and scout_bin(env) and start_warm(top, env, data):
        lines.append("  warming the search model in the background (first cold start takes 3.5-5 min)")
    return lines


def _reaped(pid: int) -> bool:
    """Reap our own child if it exited (a zombie still answers kill 0)."""
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    return done == pid


def _last_json(log: Path) -> dict[str, Any] | None:
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        if line.startswith("{"):
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            if isinstance(doc, dict):
                return doc
    return None


def doctor(env: dict[str, str], data: Path, top: str | None) -> dict[str, Any]:
    scout = scout_bin(env)
    url, model, source = gateway(env, data, top or os.getcwd())
    out: dict[str, Any] = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "scout_binary": scout,
        "key_configured": creds_present(env),
        "lane": lane(env),
        "gateway": url,
        "model": model,
        "gateway_source": source,
        "gateway_healthz": None,
    }
    if out["lane"] == "production":
        out["gateway_healthz"] = healthz(url) or "ok"
    out["ok"] = bool(scout) and (out["lane"] == "demo" or out["gateway_healthz"] == "ok")
    return out


def statusline(payload: dict[str, Any], env: dict[str, str], data: Path, now: float) -> str:
    """One compact line for a statusLine command (plugins cannot set statusLine; see README)."""
    cwd = str((payload.get("workspace") or {}).get("current_dir") or payload.get("cwd") or os.getcwd())
    paths = git_paths(cwd)
    if paths is None:
        return ""
    top, state_dir = paths
    status = read_json(os.path.join(state_dir, "status.json"))
    if status is None:
        return "scout: not indexed"
    url, model, _ = gateway(env, data, top)
    label, _ = warm_state(warm_record(data, url), now)
    cov = status.get("coverage")
    return f"scout {model} {status.get('state')} {float(cov or 0):.0%} {label} {mode_name(env, data, top)}"


# --- CLI ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scout_cc.py")
    ap.add_argument(
        "cmd",
        choices=["status", "index", "model", "mode", "doctor", "statusline", "warm-worker", "gateway-url"],
    )
    ap.add_argument("value", nargs="?", default=None)
    ap.add_argument("--data", default=None, help="plugin data dir (default $CLAUDE_PLUGIN_DATA)")
    ap.add_argument("--project", default=None, help="project dir (default $CLAUDE_PROJECT_DIR, then cwd)")
    ap.add_argument("--wait-s", type=float, default=float(os.environ.get("SCOUT_INDEX_WAIT_S", "100")))
    args = ap.parse_args(argv)
    env = dict(os.environ)
    data = data_dir(env, args.data)
    now = time.time()
    if args.cmd == "statusline":
        try:
            payload = json.load(sys.stdin)
        except ValueError:
            payload = {}
        print(statusline(payload if isinstance(payload, dict) else {}, env, data, now))
        return 0
    paths = _project(args, env)
    if args.cmd == "doctor":
        print(json.dumps(doctor(env, data, paths[0] if paths else None), sort_keys=True))
        return 0
    if paths is None:
        print("scout: not inside a git checkout; scout indexes git repositories.")
        return 0
    top, state_dir = paths
    if args.cmd == "gateway-url":
        url, _, source = gateway(env, data, top)
        if source == "model profile":
            print(url)
        return 0
    if args.cmd == "warm-worker":
        return warm_worker(top, env, data)
    if args.cmd == "status":
        print("\n".join(status_report(top, state_dir, env, data, now)))
        return 0
    if args.cmd == "index":
        print("\n".join(index(top, state_dir, env, data, args.wait_s)))
        return 0
    if args.cmd == "model":
        return _set_model(args.value, top, env, data)
    return _set_mode(args.value, top, env, data)


def _set_model(value: str | None, top: str, env: dict[str, str], data: Path) -> int:
    value = (value or "").strip().lower()
    value = MODEL_ALIASES.get(value, value)
    if not value:
        url, model, source = gateway(env, data, top)
        print(
            f"scout model: {model} ({source}) via {url}. "
            "Choose with /scout:model glimmer|a1 (glimmer = wayfinder-glimmer, profile v3)."
        )
        return 0
    if value not in GATEWAYS:
        print(
            f"scout model: unknown {value!r}; choose glimmer or v3 (default, wayfinder-glimmer) "
            "or a1 (Agents-A1-35B)."
        )
        return 0
    save_setting(data, top, "model", value)
    url, model, source = gateway(env, data, top)
    lines = [f"scout model for this project: {value} - {MODEL_LABEL[value]}", f"  gateway {GATEWAYS[value]}"]
    if source == "SCOUT_GATEWAY_URL":
        lines.append(
            f"  note: SCOUT_GATEWAY_URL is set in this session and wins ({url}); unset it to use the profile."
        )
    lines.append("  routing, warm-up, /scout:index and /scout:status use it now.")
    lines.append("  the scout MCP server of THIS session keeps its gateway until it restarts:")
    lines.append("  run /mcp, select plugin:scout:scout, Reconnect (or start a new session).")
    warmable = (
        source != "SCOUT_GATEWAY_URL"
        and lane(env) == "production"
        and not env.get("SCOUT_NO_WARM")
        and git_paths(top)
    )
    if warmable and start_warm(top, env, data):
        lines.append(f"  warming {value} in the background (cold start 3.5-5 min).")
    print("\n".join(lines))
    return 0


def _set_mode(value: str | None, top: str, env: dict[str, str], data: Path) -> int:
    value = (value or "").strip().lower()
    if value not in MODES:
        print(f"scout mode: {mode_name(env, data, top)}. Choose /scout:mode hybrid|fast|off:")
        print("  hybrid  Explore -> scout-explorer: scout explore first, then verified reads (default)")
        print("  fast    Explore -> super-explore: one scout call, relayed (cheaper, lower recall)")
        print("  off     never reroute Explore")
        return 0
    save_setting(data, top, "mode", value)
    note = "  (SCOUT_EXPLORE_MODE is set in this session and wins)" if env.get("SCOUT_EXPLORE_MODE") else ""
    print(f"scout mode for this project: {value}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
