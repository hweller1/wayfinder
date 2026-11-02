#!/usr/bin/env python3
"""PreToolUse hook: route exploration to scout-explorer.

Muse Code 1.4.14 native plugin hooks have no `matcher` (the validator rejects it), so this runs before every
tool call and returns at once for anything but these:

  subagent_spawn  When the parent spawns a generic child (no `subagent_type`, or `general-purpose`) whose
                  objective reads as read-only exploration ("where is...", "how does...", "find the code
                  that...") and scout is ready for this checkout, the call is rewritten to
                  `subagent_type: scout-wayfinder/scout-explorer` through `updatedInput`. Objectives that ask
                  for edits, runs or tests, isolated children and explicit agent types are left alone.
  search          Main thread only (MUSE_THREAD_ID == MUSE_SESSION_ID), scout ready, nothing delegated to
                  scout-explorer yet this session. Muse has no built-in Explore subagent to replace, and a
                  reminder alone was ignored in the live test (the parent grepped by hand), so by default
                  (`SCOUT_ROUTE=guard`) the FIRST such search is denied once with a reason that names the
                  subagent; every later search, and every search after a delegation, runs unchanged, so an
                  exact-string lookup costs at most one retry. `SCOUT_ROUTE=nudge` only adds a reminder,
                  `SCOUT_ROUTE=off` does nothing.

  scout explore  Main thread only, guard mode: denied with a pointer to the subagent, which makes the same
                 call and then verifies its ranges (a direct call returns an unverified, budget-capped list).

Decisions (never the objective text) go to `$MUSE_PLUGIN_DATA_DIR/route.log`. Any error: no output, exit 0.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
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

GENERIC = ("", "general-purpose")
EXPLORE = re.compile(
    r"\b(where|which (files?|modules?|functions?|classes?)|how (does|do|is|are)|find|locate"
    r"|look (for|into|up)|explor\w*|trace|map out|search|investigate|understand|identify"
    r"|what (code|files?|calls?))\b",
    re.I,
)
ACT = re.compile(
    r"\b(implement|fix|patch|write|edit|modify|refactor|rename|delete|remove|create|add|update|bump|commit"
    r"|run|execute|install|build|deploy|test that|benchmark)\b",
    re.I,
)


def is_exploration(objective: str) -> bool:
    return bool(EXPLORE.search(objective)) and not ACT.search(objective)


def _log(env: dict[str, str], line: str) -> None:
    d = env.get("MUSE_PLUGIN_DATA_DIR")
    if not d:
        return
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        with open(Path(d) / "route.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + line + "\n")
    except OSError:
        pass


def _flag(env: dict[str, str], name: str) -> Path:
    d = Path(env.get("MUSE_PLUGIN_DATA_DIR") or env.get("TMPDIR") or tempfile.gettempdir())
    session = re.sub(r"[^A-Za-z0-9_-]", "", env.get("MUSE_SESSION_ID", "")) or "nosession"
    return d / f"{name}-{session}"


def _set(flag: Path) -> bool:
    try:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("1", encoding="utf-8")
    except OSError:
        return False
    return True


def route_spawn(tool_input: dict, cwd: str, env: dict[str, str]) -> dict | None:
    kind = str(tool_input.get("subagent_type") or "")
    if kind == lanes.EXPLORER:
        _set(_flag(env, "delegated"))
    if kind not in GENERIC:
        _log(env, f"spawn keep type={kind}")
        return None
    if tool_input.get("worktree_isolation"):
        _log(env, "spawn keep isolation")
        return None
    objective = str(tool_input.get("objective") or "")
    if not is_exploration(objective):
        _log(env, "spawn keep not-exploration")
        return None
    r = lanes.readiness(cwd, env)
    if not r["ready"]:
        _log(env, f"spawn keep lane={r['lane']} not-ready")
        return None
    if spec_delivery(cwd).get("rewrite_explore") is False:
        _log(env, "spawn keep spec rewrite_explore=false")
        return None
    _log(env, f"spawn rewrite lane={r['lane']}")
    _set(_flag(env, "delegated"))
    updated = {**tool_input, "subagent_type": lanes.EXPLORER}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": updated,
            "additionalContext": f"scout-wayfinder: routed this exploration child to {lanes.EXPLORER}.",
        }
    }


def spec_active(cwd: str, env: dict[str, str]) -> bool:
    """True when the resolved spec turns the search-first policy on (`search_policy` deny or warn, or
    `search_first`), so the retrieval-spec hooks (hooks/spec_hooks.py) own the main thread's `search` and the
    one-shot guard below must not double up. The resolved spec reads "off" both for an explicit off and for
    the default, so "off" leaves the guard as it was (`SCOUT_ROUTE=off` turns that off).
    `SCOUT_SPEC_DISABLE=1` gives the guard back."""
    if lanes.load_config(env).get("SCOUT_SPEC_DISABLE"):
        return False
    d = spec_delivery(cwd)
    return d.get("search_policy") in ("deny", "warn") or bool(d.get("search_first"))


def spec_delivery(cwd: str) -> dict:
    """`[delivery]` of the resolved spec (spec_resolved.json, written by spec_hooks.py); `{}` when absent."""
    paths = lanes.git_paths(cwd)
    resolved = lanes._read_json(Path(paths[1]) / "spec_resolved.json") if paths else None
    delivery = (resolved or {}).get("delivery")
    return delivery if isinstance(delivery, dict) else {}


def spec_handoff(cwd: str, env: dict[str, str]) -> str:
    return str(spec_delivery(cwd).get("handoff", "agent"))


def route_search(cwd: str, env: dict[str, str]) -> dict | None:
    session, thread = env.get("MUSE_SESSION_ID", ""), env.get("MUSE_THREAD_ID", "")
    if session and thread and session != thread:
        return None  # inside a subagent (scout-explorer itself searches to fill gaps)
    if spec_active(cwd, env):
        return None
    mode = lanes.load_config(env).get("SCOUT_ROUTE", "guard")
    once = _flag(env, "routed")
    if mode == "off" or once.exists() or _flag(env, "delegated").exists():
        return None
    r = lanes.readiness(cwd, env)
    if not r["ready"] or not _set(once):
        return None
    _log(env, f"search {mode} lane={r['lane']}")
    if mode == "nudge":
        text = (
            "scout-wayfinder: this checkout is indexed. For a broad where/how question, spawn "
            f"`{lanes.EXPLORER}` (subagent_spawn) instead of searching by hand; keep using search for exact "
            "strings."
        )
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": text}}
    reason = (
        f"scout-wayfinder: this repository is indexed. Delegate the exploration first: call subagent_spawn "
        f'with subagent_type "{lanes.EXPLORER}" and the question as the objective, then subagent_wait with '
        "timeout_ms 300000 (it takes 1-3 minutes; wait again on timeout, do not cancel it). It returns "
        "verified path:start-end ranges. If you only need one exact string, repeat this search: it "
        "will run."
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def is_scout_explore(tool: str) -> bool:
    """The plugin's scout `explore` tool, under any of the names Muse shows it by."""
    return tool.startswith("mcp__plugin_scout") and re.search(r"(__|\.)explore$", tool) is not None


def route_direct_explore(env: dict[str, str], cwd: str = "") -> dict | None:
    """Main thread calling scout `explore` itself gets the unverified, budget-capped list and pays a second
    cold call when it later delegates. In guard mode it is sent to the subagent instead; inside a subagent
    (the explorer) it always runs."""
    session, thread = env.get("MUSE_SESSION_ID", ""), env.get("MUSE_THREAD_ID", "")
    if session and thread and session != thread:
        return None
    if lanes.load_config(env).get("SCOUT_ROUTE", "guard") != "guard":
        return None
    if cwd and spec_active(cwd, env) and spec_handoff(cwd, env) == "direct":
        return None  # the spec asked the parent to call scout itself
    _log(env, "explore deny main-thread")
    reason = (
        "scout-wayfinder: call scout through the subagent: subagent_spawn with subagent_type "
        f'"{lanes.EXPLORER}" and the question as the objective, then subagent_wait with timeout_ms 300000. '
        "It runs this explore call and verifies the ranges it returns."
    )
    out = {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}
    return {"hookSpecificOutput": out}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        tool = str(payload.get("tool_name") or "")
        env = dict(os.environ)
        if is_scout_explore(tool):
            out = route_direct_explore(env, str(payload.get("cwd") or os.getcwd()))
            if out is not None:
                sys.stdout.write(json.dumps(out) + "\n")
            return 0
        if tool not in ("subagent_spawn", "search"):
            return 0
        cwd = str(payload.get("cwd") or os.getcwd())
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            return 0
        out = route_spawn(tool_input, cwd, env) if tool == "subagent_spawn" else route_search(cwd, env)
        if out is not None:
            sys.stdout.write(json.dumps(out) + "\n")
    except Exception:  # noqa: BLE001  never break the user's session
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
