#!/usr/bin/env python3
# ruff: noqa: E501
"""Retrieval-spec hooks for Muse Code (the retrieval spec, not in this release): `spec_hooks.py <session_start|prompt|pretool>`.

The Muse twin of integrations/claude-code/hooks/spec_hooks.py. Both read ONE spec: `lib/scout_spec.py` here is
a byte-identical copy of the Claude Code plugin's file (the bundle is installed on its own, so it cannot import
across directories; `scripts/sync_muse_spec.py --check` and a unit test keep the two equal), and so is
`spec/harness_defaults.toml`. Parsing, the three layers, the index description and the git-history miner are
that shared code; only the delivery vocabulary differs (`scout_spec.HOSTS["muse-code"]`).

  session_start  SessionStart: resolve the spec (writes `<git-path scout>/spec_resolved.json`, which the scout
                 core reads), then add instructions + index description + examples as additionalContext.
  prompt         UserPromptSubmit: the workflow step "before reading or editing code, delegate to the search
                 subagent" (`[delivery] workflow_step`), worded for subagent_spawn / subagent_wait.
  pretool        PreToolUse (Muse hooks have no matcher: this runs on every tool call and returns at once for
                 all but spawns, scout tools and `search`): the search-first policy (`[delivery] search_first` /
                 `search_policy`): the MAIN thread's `search` is denied until its first delegation or scout call;
                 subagents (MUSE_THREAD_ID != MUSE_SESSION_ID) are never gated.

Delegation target: `scout-wayfinder/scout-explorer` when scout can answer for this checkout and the spec leaves
the index on; otherwise Muse's built-in `general-purpose` agent (Muse has no Explore subagent).
`SCOUT_SPEC_TARGET` pins it (the spec-only ablation sets `general-purpose`), `SCOUT_SPEC_DISABLE=1` silences
everything, `SCOUT_SPEC_FILE` points at another spec file. Muse filters hook environments, so these three are
read from `scout.env` like the other settings (lib/lanes.py). Stdlib only, silent on any error. Decisions (event
names, never prompts or paths) go to `$MUSE_PLUGIN_DATA_DIR/spec_hooks.jsonl`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOST = "muse-code"
SPEC_ENV = ("SCOUT_SPEC_TARGET", "SCOUT_SPEC_DISABLE", "SCOUT_SPEC_FILE")
SCOUT_PREFIX = "mcp__plugin_scout"


def _load(name: str, rel: str):  # a sibling by path, so this file imports only the stdlib
    spec = importlib.util.spec_from_file_location(name, HERE / rel)
    if spec is None or spec.loader is None:
        raise ImportError(rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lanes = _load("lanes", "../lib/lanes.py")
scout_spec = _load("scout_spec", "../lib/scout_spec.py")


def spec_env(env: dict[str, str]) -> dict[str, str]:
    """The process env plus the spec settings from scout.env (Muse strips them from hook environments)."""
    cfg = lanes.load_config(env)
    return {**env, **{k: cfg[k] for k in SPEC_ENV if cfg.get(k)}}


def target_for(cwd: str, resolved: dict, env: dict[str, str]) -> str:
    """Where the workflow step and the deny message point (`[delivery] handoff` picks agent or direct)."""
    h = scout_spec.HOSTS[HOST]
    pinned = env.get("SCOUT_SPEC_TARGET")
    if pinned:
        base = pinned
    elif not resolved.get("index_enabled", True):
        base = h["fallback"]
    else:
        base = h["agent"] if lanes.readiness(cwd, env)["ready"] else h["fallback"]
    return scout_spec.apply_handoff(base, resolved["delivery"].get("handoff", "agent"), HOST)


def _log(env: dict[str, str], rec: dict) -> None:
    d = env.get("MUSE_PLUGIN_DATA_DIR")
    if not d:
        return
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        with open(Path(d) / "spec_hooks.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 2), **rec}) + "\n")
    except OSError:
        pass


def _is_sub(env: dict[str, str]) -> bool:
    session, thread = env.get("MUSE_SESSION_ID", ""), env.get("MUSE_THREAD_ID", "")
    return bool(session and thread and session != thread)


def _state_file(env: dict[str, str], payload: dict) -> Path | None:
    d = env.get("MUSE_PLUGIN_DATA_DIR")
    sid = re.sub(r"[^A-Za-z0-9_-]", "", str(env.get("MUSE_SESSION_ID") or payload.get("session_id") or ""))[
        :64
    ]
    return Path(d) / "spec-state" / sid if d and sid else None


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    env = spec_env(dict(os.environ))
    try:
        try:
            payload = json.load(sys.stdin)
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        cwd = str(payload.get("cwd") or os.getcwd())
        got = scout_spec.git_top_and_state(cwd)
        if got is None or env.get("SCOUT_SPEC_DISABLE"):
            return 0
        top, state_dir = got
        if mode == "session_start":
            resolved = scout_spec.resolve(top, env, state_dir)
            target = target_for(cwd, resolved, env)
            ctx = scout_spec.session_context(resolved, target)
            _log(env, {"hook": "SessionStart", "target": target, "handoff": resolved["delivery"].get("handoff"),
                       "has_description": bool(resolved["description"]), "n_examples": len(resolved["examples"]),
                       "cached": resolved["cached"], "warnings": len(resolved["warnings"]), "injected": bool(ctx)})  # fmt: skip
            if ctx:
                out = {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}
                sys.stdout.write(json.dumps(out) + "\n")
            return 0
        resolved = scout_spec._read_json(os.path.join(state_dir, scout_spec.RESOLVED)) or scout_spec.resolve(
            top, env, state_dir, write=False
        )
        target = target_for(cwd, resolved, env)
        if mode == "prompt":
            out = scout_spec.prompt_output(str(payload.get("prompt", "")), resolved, target, HOST)
            _log(env, {"hook": "UserPromptSubmit", "target": target, "injected": out is not None})
            if out:
                sys.stdout.write(json.dumps(out) + "\n")
        elif mode == "pretool":
            name = str(payload.get("tool_name") or "")
            if name not in scout_spec.HOSTS[HOST]["gated_tools"] + scout_spec.HOSTS[HOST][
                "delegate_tools"
            ] and not name.startswith(SCOUT_PREFIX):
                return 0  # most calls: nothing to decide, nothing to log
            sf = _state_file(env, payload)
            mapped = bool(sf and sf.exists())
            sub = _is_sub(env)
            out, label, after = scout_spec.pretool_decision(
                payload, mapped, resolved, target, SCOUT_PREFIX, HOST, sub
            )
            if after and not mapped and sf:
                sf.parent.mkdir(parents=True, exist_ok=True)
                sf.write_text(label, encoding="utf-8")
            ti = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
            _log(env, {"hook": "PreToolUse", "tool": name[:60], "event": label, "subagent": sub,
                       "subagent_type": ti.get("subagent_type"), "denied": bool(out and "permissionDecision" in str(out))})  # fmt: skip
            if out:
                sys.stdout.write(json.dumps(out) + "\n")
    except Exception as e:  # noqa: BLE001  never break the user's session
        _log(env, {"hook": mode, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
