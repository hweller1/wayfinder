#!/usr/bin/env python3
"""PreToolUse hook (matcher `Agent|Task`): route the built-in Explore subagent to scout.

Only a call with `subagent_type == "Explore"` is considered. When scout can answer for this checkout it is
rewritten (every other input field copied through) to the subagent of the project's mode:
  hybrid (default)  scout:scout-explorer   scout `explore` first, then verified reads and gap-filling
  fast              scout:super-explore    one scout call relayed verbatim (cheaper, lower recall)
  off               never rewritten
"Can answer" (lib/scout_cc.py `route`):
  production  a key configured AND status.json ready/partial AND gateway_ok_at < 10 min AND the scout MCP
              server alive AND the last SessionStart probe of the gateway did not fail after that ping.
  demo        no key: a ready `scout-demo` index whose checkout is this repo (the name is passed on a first
              prompt line `[scout repo: NAME]`).
Anything else: no output, so Claude Code runs the stock Explore and the user sees no failure. Stdlib only,
silent on any error. The decision (lane, target, reason; never the brief) goes to
${CLAUDE_PLUGIN_DATA}/hook.log.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))


def main() -> int:
    env = dict(os.environ)
    try:
        import scout_cc

        payload = json.load(sys.stdin)
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict) or tool_input.get("subagent_type") != "Explore":
            return 0
        cwd = str(payload.get("cwd") or os.getcwd())
        data = scout_cc.data_dir(env)
        if scout_cc.spec_blocks_rewrite(cwd):
            scout_cc.log_decision(env, "fallthrough lane=- reason=spec-off")
            return 0
        updated, lane, target, reason = scout_cc.route(tool_input, cwd, time.time(), env, data)
        if updated is None:
            scout_cc.log_decision(env, f"fallthrough lane={lane} reason={reason}")
            if reason == "model-failed" and not env.get("SCOUT_NO_WARM"):
                scout_cc.retry_warm(cwd, env, data)  # detached; a later Explore routes once the model is back
            return 0
        scout_cc.log_decision(env, f"rewrite lane={lane} target={target} reason={reason}")
        out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": updated}}
        sys.stdout.write(json.dumps(out))
    except Exception:  # noqa: BLE001  never break the user's session
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
