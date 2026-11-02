#!/usr/bin/env python3
"""MCP server launcher for the scout-wayfinder Muse plugin (manifest `mcpServers[scout]`).

Muse runs this from the plugin cache with the session workspace as cwd and a filtered environment. It picks a
lane with `lib/lanes.py` and replaces itself with the server, so the server owns stdin/stdout:
  production  `scout mcp --tools explore,index_status --render-ceiling 1500` (hosted gateway; repo = cwd)
  demo        `scout-demo mcp` (explore(brief, repo), list_repos, index_repo)
The key, when it comes from `SCOUT_KEY_FILE`, is handed to the child as `SCOUT_KEY` and never printed.
`--dry-run` prints the chosen lane and argv (no key) and exits; the unit tests use it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
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

PROD_ARGS = ["mcp", "--tools", "explore,index_status", "--render-ceiling", "1500"]
EXTRA_PATH = ["/opt/homebrew/bin", "/usr/local/bin"]


def plan(env: dict[str, str]) -> tuple[str, list[str], dict[str, str]]:
    home = env.get("HOME") or str(Path.home())
    env = {**env, "PATH": os.pathsep.join([env.get("PATH", ""), *EXTRA_PATH, f"{home}/.local/bin"])}
    cfg = lanes.load_config(env)
    lane = lanes.pick_lane(cfg, env)
    child = dict(env)
    for k in ("SCOUT_GATEWAY_URL", "SCOUT_KEYSTORE", "SCOUT_DEMO_HOME", "SCOUT_DEMO_PRESET"):
        if cfg.get(k):
            child[k] = cfg[k]
    child["SCOUT_RENDER_MAX_TOKENS"] = cfg.get("SCOUT_RENDER_MAX_TOKENS", "1200")
    if lane == "production":
        binary = lanes.scout_bin(cfg, env)
        if binary:
            key = lanes.read_key_file(cfg)
            if key:
                child["SCOUT_KEY"] = key
            return lane, [binary, *PROD_ARGS], child
        if cfg.get("SCOUT_LANE") == "production":
            raise SystemExit(
                "scout-wayfinder: SCOUT_LANE=production but no scout binary (uv tool install ...)"
            )
    project = lanes.demo_project(cfg, env, HERE)
    if project:
        venv = Path(project) / ".venv" / "bin" / "scout-demo"
        if os.access(venv, os.X_OK):
            return "demo", [str(venv), "mcp"], child
        uv = shutil.which("uv", path=env["PATH"])
        if uv:
            return "demo", [uv, "run", "--quiet", "--project", project, "scout-demo", "mcp"], child
    found = shutil.which("scout-demo", path=env["PATH"])
    if found:
        return "demo", [found, "mcp"], child
    binary = lanes.scout_bin(cfg, env)
    if binary:  # no demo either: a keyless `scout mcp` still starts and its tools answer LOGIN_REQUIRED
        return "production", [binary, *PROD_ARGS], child
    raise SystemExit("scout-wayfinder: neither scout nor scout-demo found; see INSTALL.md")


def main() -> None:
    lane, argv, child = plan(dict(os.environ))
    if "--dry-run" in sys.argv[1:]:
        print(json.dumps({"lane": lane, "argv": argv, "has_key": "SCOUT_KEY" in child}))
        return
    os.execvpe(argv[0], argv, child)  # noqa: S606


if __name__ == "__main__":
    main()
