#!/usr/bin/env python3
# ruff: noqa: E501
"""Retrieval-spec hooks (the retrieval spec, not in this release): `spec_hooks.py <session_start|prompt|pretool>`.

  session_start  SessionStart: resolve the spec (writes `<git-path scout>/spec_resolved.json`, which the scout
                 core reads), then add instructions + index description + examples as additionalContext.
  prompt         UserPromptSubmit: the workflow step "before reading or editing code, delegate to the search
                 subagent" (`[delivery] workflow_step`).
  pretool        PreToolUse on Agent|Task|Grep|Glob|scout tools: the search-first policy (`[delivery]
                 search_first`): the MAIN agent's Grep/Glob are denied until its first delegation or scout
                 call; subagents are never gated.

The delegation target is `scout:scout-explorer` when scout can answer for this checkout (the same readiness
rule as rewrite_explore.py) and the spec leaves the index on; otherwise the built-in `Explore`.
$SCOUT_SPEC_TARGET pins it (the spec-only ablation sets `Explore`); $SCOUT_SPEC_DISABLE=1 silences everything.
Stdlib only, silent on any error: a spec problem must never break a session. Decisions (event, never prompts or
paths) go to ${CLAUDE_PLUGIN_DATA}/spec_hooks.jsonl.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
EXPLORE = "Explore"


def target_for(top, cwd, resolved, env):
    """Where the workflow step and the deny message point (`[delivery] handoff` picks agent, relay or direct)."""
    import scout_spec

    return scout_spec.apply_handoff(
        _base_target(top, cwd, resolved, env), resolved["delivery"].get("handoff", "agent")
    )


def _base_target(top, cwd, resolved, env):
    import scout_cc

    pinned = env.get("SCOUT_SPEC_TARGET")
    if pinned:
        return pinned
    if not resolved.get("index_enabled", True):
        return EXPLORE
    if scout_cc.mode_name(env, scout_cc.data_dir(env), top) == "off":
        return EXPLORE
    # At SessionStart and on the first prompt the MCP server may not be up and the gateway ping may be a few
    # minutes old: those two do not make the target Explore. Anything else (no key, no index) does.
    paths = scout_cc.git_paths(cwd)
    reason = scout_cc.production_reason(paths[1], time.time(), env) if paths else "not-a-repo"
    if reason in (None, "mcp-down", "gateway-stale"):
        return scout_cc.TARGETS[scout_cc.mode_name(env, scout_cc.data_dir(env), top)]
    updated, _lane, _target, _why = scout_cc.route(
        {"prompt": ""}, cwd, time.time(), env, scout_cc.data_dir(env)
    )
    return updated["subagent_type"] if updated else EXPLORE


def _log(env, rec):
    raw = env.get("CLAUDE_PLUGIN_DATA")
    if not raw:
        return
    try:
        os.makedirs(raw, exist_ok=True)
        with open(os.path.join(raw, "spec_hooks.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 2), **rec}) + "\n")
    except OSError:
        pass


def _state_file(env, payload):
    raw = env.get("CLAUDE_PLUGIN_DATA")
    sid = str(payload.get("session_id") or "")
    if not raw or not sid:
        return None
    safe = "".join(c for c in sid if c.isalnum() or c in "-_")[:64]
    return os.path.join(raw, "spec-state", safe)


def main(argv):
    mode = argv[1] if len(argv) > 1 else ""
    env = dict(os.environ)
    try:
        import scout_spec

        try:
            payload = json.load(sys.stdin)
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        cwd = str(env.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd())
        got = scout_spec.git_top_and_state(cwd)
        if got is None or env.get("SCOUT_SPEC_DISABLE"):
            return 0
        top, state_dir = got
        if mode == "session_start":
            resolved = scout_spec.resolve(top, env, state_dir)
            target = target_for(top, cwd, resolved, env)
            ctx = scout_spec.session_context(resolved, target)
            _log(env, {"hook": "SessionStart", "target": target, "layers": resolved["layers"]["harness"] and ["system", "harness"] + (["user"] if resolved["layers"]["user"] else []),
                       "handoff": resolved["delivery"].get("handoff"), "has_description": bool(resolved["description"]), "n_examples": len(resolved["examples"]),
                       "cached": resolved["cached"], "warnings": len(resolved["warnings"]), "injected": bool(ctx)})  # fmt: skip
            if ctx:
                sys.stdout.write(
                    json.dumps(
                        {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}
                    )
                )
            return 0
        resolved = scout_spec._read_json(os.path.join(state_dir, scout_spec.RESOLVED)) or scout_spec.resolve(
            top, env, state_dir, write=False
        )
        target = target_for(top, cwd, resolved, env)
        if mode == "prompt":
            out = scout_spec.prompt_output(str(payload.get("prompt", "")), resolved, target)
            _log(env, {"hook": "UserPromptSubmit", "target": target, "injected": out is not None})
            if out:
                sys.stdout.write(json.dumps(out))
        elif mode == "pretool":
            sf = _state_file(env, payload)
            mapped = bool(sf and os.path.exists(sf))
            out, label, after = scout_spec.pretool_decision(
                payload, mapped, resolved, target, scout_cc_prefix()
            )
            if after and not mapped and sf:
                os.makedirs(os.path.dirname(sf), exist_ok=True)
                with open(sf, "w", encoding="utf-8") as f:
                    f.write(label)
            ti = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
            _log(env, {"hook": "PreToolUse", "tool": str(payload.get("tool_name"))[:60], "event": label, "subagent": bool(payload.get("agent_id")),
                       "subagent_type": ti.get("subagent_type"), "denied": bool(out and "permissionDecision" in str(out))})  # fmt: skip
            if out:
                sys.stdout.write(json.dumps(out))
    except Exception as e:  # noqa: BLE001  never break the user's session
        _log(env, {"hook": mode, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return 0


def scout_cc_prefix():
    import scout_cc

    return scout_cc.TOOL_PREFIX


if __name__ == "__main__":
    sys.exit(main(sys.argv))
