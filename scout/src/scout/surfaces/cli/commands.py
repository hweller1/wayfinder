"""CLI commands. Imported only after argument parsing (`main.py`), so heavy modules load per command.

Every query command is the same three steps: build the contract request from the flags, call the one facade
(`ScoutApp`, built by `scout.wiring`), print the result (JSON by default, plain text with `--text`). The
profile comes from `--profile` / `SCOUT_PROFILE`; profile `fake` reads its case file from `SCOUT_FAKE_CASE`.
"""

import argparse
import asyncio
import json
import os
from collections.abc import Callable, Mapping
from typing import Any

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.surfaces import ops

Handler = Callable[[argparse.Namespace], int]


def flags(args: argparse.Namespace) -> dict[str, str | None]:
    """The composition flags the user gave (absent ones are not in the namespace: see `_common`)."""
    return {k: getattr(args, k, None) for k in ("profile", "root", "gateway_url")}


def profile(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> str:
    """The profile name these flags resolve to (flag > `SCOUT_PROFILE` > default)."""
    environ = os.environ if env is None else env
    return flags(args)["profile"] or environ.get("SCOUT_PROFILE") or "default"


def build_app(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> Any:
    """The `ScoutApp` for these flags (flag > env `SCOUT_*` > config file > default)."""
    from scout.wiring import build_app as wiring_build_app
    from scout.wiring import resolve_config

    environ = os.environ if env is None else env
    cfg = resolve_config(flags(args), environ, case=environ.get("SCOUT_FAKE_CASE") or None)
    return wiring_build_app(cfg, environ)


def _payload(args: argparse.Namespace, op: str) -> dict[str, Any]:
    """The request fields the user set; defaults are the contract's, so unset flags are left out."""
    g = lambda name: getattr(args, name, None)  # noqa: E731
    out: dict[str, Any]
    if op == "explore":
        out = {"brief": g("brief"), "paths": g("paths"), "max_results": g("max_results"),
               "render_max_tokens": g("render_max_tokens")}  # fmt: skip
        budget = {"max_turns": g("max_turns"), "max_seconds": g("max_seconds")}
        out["budget"] = {k: v for k, v in budget.items() if v is not None} or None
    elif op == "semantic_search":
        out = {"query": g("query"), "path_glob": g("path_glob"), "k": g("k")}
    elif op == "grep":
        out = {"pattern": g("pattern"), "path": g("path"), "context": g("context"),
               "regex": False if g("literal") else None}  # fmt: skip
    elif op == "read_range":
        out = {"path": g("path"), "start_line": g("start"), "end_line": g("end")}
    else:
        out = {}
    return {k: v for k, v in out.items() if v is not None}


def _query(op: str) -> Handler:
    def run(args: argparse.Namespace) -> int:
        req = ops.parse_request(op, _payload(args, op))
        app = build_app(args)
        result = asyncio.run(getattr(app, op)(req))
        print(ops.render(op, req, result) if getattr(args, "text", False) else ops.result_json(result))
        return 0

    return run


def _schema(args: argparse.Namespace) -> int:
    from scout.core.contracts.schema import render_schemas

    files = render_schemas()
    if args.name is None:
        print("\n".join(sorted(n.removesuffix(".json") for n in files)))
        return 0
    text = files.get(f"{args.name}.json")
    if text is None:
        raise ScoutError(
            ErrorCode.NOT_FOUND, f"no schema named {args.name!r} (run `scout schema` to list them)"
        )
    print(json.dumps(json.loads(text), indent=2))
    return 0


def _surface(module: str) -> Any:
    """Another surface, imported by name at run time: the surfaces never import each other statically
    (import-linter contract 5), and a command that does not need one never pays for it. `__import__` rather
    than `importlib.import_module`: only the C import path is timed and nested by `-X importtime`, which the
    startup guard reads to find who imported what."""
    return __import__(f"scout.surfaces.{module}", fromlist=["_"])


def _state_dir(args: argparse.Namespace) -> str | None:
    """The state dir (`git rev-parse --git-path scout`) of the resolved root, or None outside git."""
    from scout.adapters.fs_local import git
    from scout.wiring import resolve_config

    try:
        root = resolve_config(flags(args), os.environ).root or os.getcwd()
        return git.state_dir(os.path.realpath(root))
    except ScoutError:
        return None


def _mcp(args: argparse.Namespace) -> int:
    server = _surface("mcp.server")
    demo = profile(args) == "demo"
    tools = (server.select_demo_tools if demo else server.select_tools)(
        args.tools
    )  # fail early on a bad name
    if demo:
        # `--profile demo` serves the demo lane's own server (explore(brief, repo), index_repo, list_repos),
        # so it is a drop-in for `scout-demo mcp`; it renders at its own fixed size, so --render-ceiling and
        # --structured do not apply to it.
        from scout.wiring import build_demo_mcp

        built = server.restrict_demo_tools(build_demo_mcp(), tools)
    else:
        render_default = int(os.environ.get("SCOUT_RENDER_MAX_TOKENS") or server.DEFAULT_RENDER_MAX_TOKENS)
        built = server.build_mcp(
            lambda: build_app(args),
            tools,
            args.render_ceiling,
            structured=args.structured,
            default_render_tokens=render_default,
        )
    state = None if profile(args) == "fake" else _state_dir(args)
    server.serve(built, http_port=args.http, state_dir=state)
    return 0


def _serve(args: argparse.Namespace) -> int:
    import socket
    import sys
    from pathlib import Path

    daemon_mod = _surface("daemon.server")
    host = daemon_mod.check_loopback(args.host)
    token_path = Path(args.token_file) if args.token_file else daemon_mod.default_token_path()
    token = daemon_mod.load_or_create_token(token_path)
    port = args.port
    if port == 0:  # a free port, announced on stderr so a launcher can read it
        with socket.socket() as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]
    daemon = daemon_mod.create_daemon(lambda: build_app(args), token)
    print(
        f"scout daemon listening on http://{host}:{port} (token file {token_path})",
        file=sys.stderr,
        flush=True,
    )
    daemon_mod.serve(daemon, host, port)
    return 0


def _admin() -> Any:
    from scout.surfaces.cli import admin

    return admin


def _login(args: argparse.Namespace) -> int:
    from scout.surfaces.cli import launchd
    from scout.wiring import build_login

    doc = _admin().login(args, os.environ, build_login)
    doc["daemon"] = launchd.install_for_login(os.environ, enabled=not args.no_daemon, profile=profile(args))
    print(json.dumps(doc, separators=(",", ":")))
    return 0


def _daemon(args: argparse.Namespace) -> int:
    from scout.surfaces.cli import launchd

    out = launchd.install(os.environ) if args.action == "install" else launchd.uninstall(os.environ)
    print(json.dumps(out, separators=(",", ":")))
    return 0


def _sync(args: argparse.Namespace) -> int:
    rc: int = _admin().sync(args, lambda a: build_app(a))
    return rc


def _watch(args: argparse.Namespace) -> int:
    admin = _admin()
    if args.add or args.remove:
        rc: int = admin.watch_roots(args, os.environ)
    elif args.all:
        rc = admin.watch_all(args, os.environ, build_app)
    else:
        rc = admin.watch_one(args, os.environ, build_app)
    return rc


def _warm(args: argparse.Namespace) -> int:
    out = asyncio.run(build_app(args).warm(trigger=bool(getattr(args, "trigger", False))))
    print(json.dumps({**out, "seconds": round(float(str(out["seconds"])), 2)}))
    return 0


COMMANDS: dict[str, Handler] = {
    "warm": _warm,
    "login": _login,
    "logout": lambda a: _admin().logout(a, os.environ),
    "daemon": _daemon,
    "sync": _sync,
    "watch": _watch,
    "init": lambda a: _admin().init(a, os.environ),
    "serve": _serve,
    "mcp": _mcp,
    "explore": _query("explore"),
    "search": _query("semantic_search"),
    "grep": _query("grep"),
    "read": _query("read_range"),
    "status": _query("index_status"),
    "schema": _schema,
}


def run(args: argparse.Namespace) -> int:
    return COMMANDS[args.cmd](args)
