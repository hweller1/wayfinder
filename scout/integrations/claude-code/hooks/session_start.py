#!/usr/bin/env python3
"""SessionStart hook: keep the index fresh, pay Wayfinder's cold start early, tell the agent scout's state.

Returns at once (children are detached, all fds redirected):
  1. production lane with a `scout` binary: `scout watch --ensure` for this checkout (one per project; it
     refreshes status.json and gateway_ok_at, which the routing hook reads);
  2. when this checkout has an index: a warm-up worker (lib/scout_cc.py warm-worker) probes the gateway and
     sends one 1-token chat through it (`scout warm`), so the 3.5-5 min scale-to-zero cold start of the
     model is paid before the first search. Skipped while a warm-up runs or the model answered < 5 min ago.
     SCOUT_NO_WARM=1 disables it; SCOUT_NO_WATCH=1 disables the watcher.
  3. one line of additionalContext when the checkout is indexed (so the agent knows scout-explorer exists).
Outside a git checkout, without a key and without a demo index it does nothing and prints nothing.
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

        try:
            payload = json.load(sys.stdin)
        except ValueError:
            payload = {}
        cwd = (
            env.get("CLAUDE_PROJECT_DIR")
            or (payload.get("cwd") if isinstance(payload, dict) else None)
            or os.getcwd()
        )
        paths = scout_cc.git_paths(str(cwd))
        if paths is None:
            return 0
        top, state_dir = paths
        data = scout_cc.data_dir(env)
        lane = scout_cc.lane(env)
        scout = scout_cc.scout_bin(env)
        indexed = os.path.exists(os.path.join(state_dir, "status.json"))
        if lane == "production" and scout and scout_cc.creds_present(env):
            if not env.get("SCOUT_NO_WATCH"):
                scout_cc.ensure_watcher(scout, top, env, data)
            if indexed and not env.get("SCOUT_NO_WARM"):
                scout_cc.start_warm(top, env, data)
        ready = indexed or (lane == "demo" and scout_cc.demo_repo(top, env) is not None)
        mode = scout_cc.mode_name(env, data, top)
        if ready and mode != "off":
            ctx = (
                "scout is indexed for this repository: broad where/how questions can go to the "
                "scout:scout-explorer subagent (Explore calls are routed to it when scout is ready). "
                "/scout:status shows its state."
            )
            out = {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}
            sys.stdout.write(json.dumps(out))
        scout_cc.log_decision(
            env, f"session lane={lane} indexed={int(indexed)} mode={mode} t={time.time():.0f}"
        )
    except Exception:  # noqa: BLE001  never break the session
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
