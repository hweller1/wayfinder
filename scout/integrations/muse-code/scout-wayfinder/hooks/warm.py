#!/usr/bin/env python3
"""Detached worker started by session_start.py (never by Muse directly): warm Wayfinder and keep the index
fresh, then exit. Runs with stdout/stderr appended to `$MUSE_PLUGIN_DATA_DIR/warm.log`; prints no secrets.

  production  `scout watch --ensure` (one watcher per checkout; a second launch is a no-op) unless
              SCOUT_NO_WATCH, then `scout warm --trigger`: the gateway's `/v1/warm` hits the model's Modal
              `/warm` endpoint and returns in about a second while the GPU boots. Gateways without the route,
              or models without a warm endpoint, fall back to one blocking 1-token chat.
  demo        `scout-demo warm --trigger` (the demo model's Modal `/warm`, else a 1-token call).
The key comes from SCOUT_KEY_FILE when configured, else from `scout login`'s key store.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, rel: str):  # a sibling by path, so this file imports only the stdlib
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / rel)
    if spec is None or spec.loader is None:
        raise ImportError(rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lanes = _load("lanes", "../lib/lanes.py")
launch_mcp = _load("launch_mcp", "../bin/launch_mcp.py")

WARM_TIMEOUT_S = 600


def commands(cwd: str, env: dict[str, str]) -> tuple[str, list[list[str]], dict[str, str]]:
    lane, argv, child = launch_mcp.plan(env)
    cfg = lanes.load_config(env)
    if lane == "production":
        cmds = [] if cfg.get("SCOUT_NO_WATCH") else [[argv[0], "watch", "--ensure", "--root", cwd]]
        return lane, [*cmds, [argv[0], "warm", "--trigger", "--root", cwd]], child
    # demo: argv is `<scout-demo> mcp` or `uv run ... scout-demo mcp`
    return lane, [[*argv[:-1], "warm", "--trigger"]], child


def _code(stdout: str) -> str:
    """`mode=endpoint|completion` of a successful warm, or the error code of a failed one (no message)."""
    try:
        doc = json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return ""
    if isinstance(doc.get("error"), dict):
        return f"code={doc['error'].get('code')}"
    return f"mode={doc.get('mode')}" if doc.get("mode") else ""


def main() -> int:
    cwd = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    lane, cmds, child = commands(cwd, dict(os.environ))
    for cmd in cmds:
        is_watch = "watch" in cmd[1:3]
        for _attempt in (1, 2):  # a just-booted model can fail its first request; one retry
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            t0 = time.monotonic()
            code = ""
            try:
                if is_watch:
                    subprocess.Popen(  # noqa: S603
                        cmd, cwd=cwd, env=child, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, start_new_session=True,
                    )  # fmt: skip
                    rc: int | str = "spawned"
                else:
                    proc = subprocess.run(  # noqa: S603
                        cmd, cwd=cwd, env=child, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=WARM_TIMEOUT_S, check=False,
                    )  # fmt: skip
                    rc, code = proc.returncode, _code(proc.stdout)
            except (OSError, subprocess.SubprocessError) as e:
                rc = type(e).__name__
            name = "watch" if is_watch else "warm"
            extra = f" {code}" if code else ""
            print(f"{stamp} lane={lane} {name} rc={rc}{extra} {time.monotonic() - t0:.1f}s", flush=True)
            if rc in (0, "spawned"):
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
