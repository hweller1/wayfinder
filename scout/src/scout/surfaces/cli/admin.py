"""Admin commands of the CLI: `login`, `logout`, `sync`, `watch`, `init` (design section 11, FR-I8, FR-I13).

Host-generic: no host name lives here. `init` reads everything host-specific (the version command and minimum,
the override file, the install commands, and for hosts that declare them the install steps to run)
from `<integrations>/<host>/scout-host.json`, so adding a host is a directory, not a code change.
`--claude-code` and `--muse-code` are argparse aliases for `--host <name>` in `main.py`.

`watch --add/--remove` keep a roots registry (`~/.config/scout/roots.json`) that the launchd daemon
(`watch --all`, see `launchd.py`) supervises: one watcher per root, picked up within `ROOTS_POLL_S` of a
change. Credentials: `login` verifies the key with `whoami`, then stores `scout_key`, `scout_tenant` and
`k_team:<tenant>` in the key store; nothing is echoed.
"""

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.surfaces.cli.exit import EXIT_OK

ROOTS_POLL_S = 5.0
TENANT_KEY = "scout_tenant"  # the tenant of the stored key, so `logout` can drop `k_team:<tenant>` offline
VERSION_TIMEOUT_S = 10.0
_VERSION = re.compile(r"\d+(?:\.\d+)+")


def _print(doc: Mapping[str, Any]) -> None:
    print(json.dumps(doc, separators=(",", ":")))


# --- login / logout -------------------------------------------------------------------------------------
def _read_key(args: argparse.Namespace, env: Mapping[str, str]) -> str:
    """`--key K`, `--key -` (stdin, keeps the key out of `ps`) or `SCOUT_KEY`."""
    key = args.key if args.key is not None else env.get("SCOUT_KEY", "")
    if key == "-":
        key = sys.stdin.readline()
    key = key.strip()
    if not key:
        raise ScoutError(
            ErrorCode.INVALID_ARGUMENT, "login needs a key: --key K, --key - (stdin) or SCOUT_KEY"
        )
    return key


async def _verify_and_fetch(gateway: Any) -> tuple[str, bytes]:
    try:
        tenant = (await gateway.whoami()).tenant
        k_team = await gateway.team_key()
    except ScoutError as e:
        if e.code in (ErrorCode.UNAUTHENTICATED, ErrorCode.LOGIN_REQUIRED):
            raise ScoutError(
                ErrorCode.UNAUTHENTICATED, "the gateway rejected this key: check it and run scout login again"
            ) from None
        raise
    finally:
        close = getattr(gateway, "aclose", None)
        if close is not None:
            await close()
    return tenant, k_team


def login(
    args: argparse.Namespace, env: Mapping[str, str], build_login: Callable[..., Any]
) -> dict[str, Any]:
    """Verify and store the key; returns the result document (the caller adds the daemon's part)."""
    from scout.core.app import KEY_NAME
    from scout.surfaces.cli import commands
    from scout.wiring import resolve_config

    key = _read_key(args, env)
    cfg = resolve_config({**commands.flags(args), "key": key}, env, case=env.get("SCOUT_FAKE_CASE") or None)
    gateway, keystore = build_login(cfg, env)
    tenant, k_team = asyncio.run(_verify_and_fetch(gateway))
    keystore.set(KEY_NAME, key.encode("utf-8"))
    keystore.set(TENANT_KEY, tenant.encode("utf-8"))
    keystore.set(f"k_team:{tenant}", k_team)
    return {"logged_in": True, "tenant": tenant}


def logout(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    from scout.adapters.keychain import KeychainStore
    from scout.core.app import KEY_NAME

    keystore = KeychainStore(env=env)
    tenant = (keystore.get(TENANT_KEY) or b"").decode("utf-8")
    was_logged_in = keystore.get(KEY_NAME) is not None
    for name in (KEY_NAME, TENANT_KEY, *([f"k_team:{tenant}"] if tenant else [])):
        keystore.delete(name)
    _print({"logged_in": False, "was_logged_in": was_logged_in})
    return EXIT_OK


# --- sync -------------------------------------------------------------------------------------------------
def spawn_background_sync(args: argparse.Namespace) -> int:
    """A detached `scout sync --wait` (all fds redirected, own session) so the caller returns at once; the
    in-process background mode of `ScoutApp.sync(wait=False)` would die with this short-lived process."""
    from scout.surfaces.cli import commands

    argv = [sys.executable, "-m", "scout", "sync", "--wait"]
    if getattr(args, "full", False):
        argv.append("--full")
    for name, value in commands.flags(args).items():
        if value:
            argv += [f"--{name.replace('_', '-')}", str(value)]
    proc = subprocess.Popen(  # noqa: S603
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


def sync(
    args: argparse.Namespace, build_app: Callable[..., Any], spawn: Callable[..., int] | None = None
) -> int:
    from scout.surfaces import ops

    if not args.wait:
        _print({"started": True, "pid": (spawn or spawn_background_sync)(args)})
        return EXIT_OK
    status = asyncio.run(build_app(args).sync(True, full=args.full))
    print(ops.render_primitive("index_status", status) if args.text else ops.result_json(status))
    return EXIT_OK


# --- watch ------------------------------------------------------------------------------------------------
def roots_file(env: Mapping[str, str]) -> Path:
    if env.get("SCOUT_ROOTS_FILE"):
        return Path(env["SCOUT_ROOTS_FILE"])
    home = Path(env["HOME"]) if env.get("HOME") else Path.home()
    return home / ".config" / "scout" / "roots.json"


def read_roots(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [r for r in data if isinstance(r, str)] if isinstance(data, list) else []


def write_roots(path: Path, roots: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".roots-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(sorted(set(roots)), f)
    os.replace(tmp, path)


def checkout_root(raw: str) -> str:
    """The real path of a git checkout (`NOT_A_REPO` otherwise): the identity a root is registered under."""
    from scout.adapters.fs_local import git

    root = os.path.realpath(os.path.expanduser(raw))
    if not os.path.isdir(root):
        raise ScoutError(ErrorCode.NOT_FOUND, "root is not a directory")
    git.state_dir(root)
    return root


def watch_roots(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    path = roots_file(env)
    roots = read_roots(path)
    if args.add:
        roots = [*roots, checkout_root(args.add)]
    else:
        gone = os.path.realpath(os.path.expanduser(args.remove))
        roots = [r for r in roots if r != gone]
    write_roots(path, roots)
    _print({"roots": sorted(set(roots))})
    return EXIT_OK


def _install_stop_signals(stop: threading.Event) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())


def watch_one(args: argparse.Namespace, env: Mapping[str, str], build_app: Callable[..., Any]) -> int:
    """Run the watcher of one checkout in the foreground until SIGINT/SIGTERM. A second launch for the same
    checkout reports the running pid and returns at once (`ScoutApp.watch` holds the single-instance lock)."""
    stop = threading.Event()
    _install_stop_signals(stop)
    result = asyncio.run(build_app(args, env).watch(stop))
    _print({"started": result.started, "pid": result.pid})
    return EXIT_OK


async def supervise(
    roots_path: Path,
    app_for: Callable[[str], Any],
    stop: threading.Event,
    *,
    poll_s: float = ROOTS_POLL_S,
) -> None:
    """One watcher task per registered root; roots added to or removed from the registry start or stop theirs
    within `poll_s`; a watcher that ended on its own is restarted on the next poll. Returns once `stop` is set
    and every watcher has finished."""
    running: dict[str, tuple[asyncio.Task[Any], threading.Event]] = {}

    async def watcher(root: str, stop_root: threading.Event) -> None:
        try:
            await app_for(root).watch(stop_root)
        except ScoutError as e:
            print(f"scout watch {root}: {e.code}: {e.message}", file=sys.stderr, flush=True)

    while not stop.is_set():
        wanted = set(read_roots(roots_path))
        for root in sorted(wanted - running.keys()):
            ev = threading.Event()
            running[root] = (asyncio.create_task(watcher(root, ev)), ev)
        for root in sorted(running.keys() - wanted):
            running.pop(root)[1].set()
        for root, (task, _) in list(running.items()):
            if task.done():
                del running[root]  # restarted by the next iteration if still wanted
        await asyncio.to_thread(stop.wait, poll_s)
    for _, ev in running.values():
        ev.set()
    await asyncio.gather(*(t for t, _ in running.values()), return_exceptions=True)


def watch_all(args: argparse.Namespace, env: Mapping[str, str], build_app: Callable[..., Any]) -> int:
    stop = threading.Event()
    _install_stop_signals(stop)

    def app_for(root: str) -> Any:
        return build_app(argparse.Namespace(**{**vars(args), "root": root}), env)

    asyncio.run(supervise(roots_file(env), app_for, stop))
    return EXIT_OK


# --- init -------------------------------------------------------------------------------------------------
def integrations_dir(env: Mapping[str, str]) -> Path:
    """`SCOUT_INTEGRATIONS_DIR`, else `integrations/` of the source checkout this code runs from."""
    if env.get("SCOUT_INTEGRATIONS_DIR"):
        return Path(env["SCOUT_INTEGRATIONS_DIR"])
    here = Path(__file__).resolve().parents[4] / "integrations"
    if here.is_dir():
        return here
    raise ScoutError(
        ErrorCode.NOT_FOUND, "no integrations directory next to this install: set SCOUT_INTEGRATIONS_DIR"
    )


def available_hosts(root: Path) -> list[str]:
    return sorted(p.parent.name for p in root.glob("*/scout-host.json"))


def load_host(root: Path, host: str) -> dict[str, Any]:
    path = root / host / "scout-host.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        names = ", ".join(available_hosts(root)) or "none"
        raise ScoutError(ErrorCode.NOT_FOUND, f"no host {host!r} (available: {names})") from None
    except (OSError, ValueError):
        raise ScoutError(
            ErrorCode.INVALID_ARGUMENT, f"{host}: scout-host.json is not readable JSON"
        ) from None
    if not isinstance(doc, dict):
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"{host}: scout-host.json must be an object")
    _check_shape(host, doc)
    return doc


def _check_shape(host: str, doc: dict[str, Any]) -> None:
    def bad(field: str) -> ScoutError:
        return ScoutError(ErrorCode.INVALID_ARGUMENT, f"{host}: scout-host.json field {field!r} is malformed")

    for field in ("min_version",):
        if field in doc and not isinstance(doc[field], str):
            raise bad(field)
    for field in ("version_command", "install_commands"):
        value = doc.get(field, [])
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise bad(field)
    steps = doc.get("install_steps", [])
    if not isinstance(steps, list) or not all(
        isinstance(s, list) and s and all(isinstance(x, str) for x in s) for s in steps
    ):
        raise bad("install_steps")
    override = doc.get("override")
    if override is not None and not (
        isinstance(override, dict)
        and isinstance(override.get("target"), str)
        and isinstance(override.get("template"), str)
    ):
        raise bad("override")


def parse_version(text: str) -> tuple[int, ...] | None:
    m = _VERSION.search(text)
    return tuple(int(p) for p in m.group(0).split(".")) if m else None


def check_version(doc: Mapping[str, Any]) -> dict[str, Any]:
    """`{"version", "min_version", "ok", "reason"?}` from the host's own version command; `ok` is None when
    the host file names no version command (nothing to check)."""
    command: list[str] = doc.get("version_command") or []
    minimum: str = doc.get("min_version", "")
    out: dict[str, Any] = {"min_version": minimum or None, "version": None, "ok": None}
    if not command:
        out["reason"] = "scout-host.json has no version_command: skipped"
        return out
    try:
        proc = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        out.update(ok=False, reason=f"{command[0]} could not be run")
        return out
    found = parse_version(proc.stdout + proc.stderr)
    if found is None:
        out.update(ok=False, reason=f"{command[0]} printed no version")
        return out
    out["version"] = ".".join(map(str, found))
    need = parse_version(minimum) if minimum else ()
    out["ok"] = need is not None and found >= need
    if not out["ok"]:
        out["reason"] = f"{command[0]} {out['version']} is older than {minimum}"
    return out


def write_override(doc: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Write the host's override file. Only called for `--override`; never replaces a different file."""
    override = doc.get("override")
    if not override:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "this host has no override file")
    raw: str = override["target"]
    home = env.get("HOME")
    target = Path(home) / raw[2:] if home and raw.startswith("~/") else Path(raw).expanduser()
    template: str = override["template"]
    try:
        existing = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = None
    if existing is not None and existing != template:
        raise ScoutError(ErrorCode.CONFLICT, f"{target} already exists and differs: move it aside first")
    if existing is None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template, encoding="utf-8")
    return {"written": existing is None, "path": str(target)}


def placeholders(root: Path) -> dict[str, str]:
    """Values for `{name}` in `install_steps`."""
    return {"integrations": str(root.resolve()), "repo": str(root.resolve().parent), "python": sys.executable}


def run_install(doc: Mapping[str, Any], root: Path, *, dry_run: bool) -> dict[str, Any]:
    """Run the host's `install_steps` (argv lists, no shell; `{integrations}`, `{repo}`, `{python}` filled in)
    with this process's environment, so `XDG_*` scope the host's config. Stops at the first failing step. The
    report holds each step's argv, exit code and the last line of its output (steps are fixed commands from
    scout-host.json and print no secrets)."""
    values = placeholders(root)
    steps = [[_fill(a, values) for a in step] for step in doc.get("install_steps", [])]
    if dry_run:
        return {"ran": False, "steps": [{"argv": s} for s in steps]}
    report: dict[str, Any] = {"ran": True, "steps": [], "ok": True}
    for step in steps:
        try:
            proc = subprocess.run(  # noqa: S603
                step, capture_output=True, text=True, timeout=300, check=False, stdin=subprocess.DEVNULL
            )
            rc, tail = proc.returncode, (proc.stdout + proc.stderr).strip().splitlines()[-1:]
        except (OSError, subprocess.TimeoutExpired) as e:
            rc, tail = -1, [type(e).__name__]
        report["steps"].append({"argv": step, "rc": rc, "tail": tail[0][:200] if tail else ""})
        if rc != 0:
            report["ok"] = False
            break
    return report


def _fill(text: str, values: Mapping[str, str]) -> str:
    for k, v in values.items():
        text = text.replace("{" + k + "}", v)
    return text


def init(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    root = integrations_dir(env)
    if not args.host:
        names = ", ".join(available_hosts(root)) or "none"
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"init needs --host NAME (available: {names})")
    doc = load_host(root, args.host)
    result: dict[str, Any] = {"host": args.host, **check_version(doc)}
    result["install_commands"] = list(doc.get("install_commands", []))
    result["override"] = write_override(doc, env) if args.override else None
    if doc.get("install_steps"):
        dry = bool(getattr(args, "dry_run", False)) or result["ok"] is False
        result["install"] = run_install(doc, root, dry_run=dry)
    _print(result)
    return EXIT_OK
