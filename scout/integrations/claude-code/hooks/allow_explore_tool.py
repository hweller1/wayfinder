#!/usr/bin/env python3
"""PreToolUse hook for the plugin's own scout MCP tools (matcher `mcp__plugin_scout_scout__.*`): allow them.

They are read-only (explore, semantic_search, read_range, index_status). In the default permission
mode a subagent's call to a plugin MCP tool is denied, and a plugin cannot ship `permissions.allow`, so
the plugin allows exactly its own tools here. Any other tool name gets no output (the normal permission
flow). Stdlib only.
"""

from __future__ import annotations

import json
import sys

PREFIX = "mcp__plugin_scout_scout__"
ALLOWED = ("explore", "semantic_search", "read_range", "index_status", "grep")


def decide(payload: object) -> dict | None:
    name = payload.get("tool_name") if isinstance(payload, dict) else None
    if not isinstance(name, str) or not name.startswith(PREFIX) or name[len(PREFIX) :] not in ALLOWED:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": "scout: read-only code search tool",
        }
    }


if __name__ == "__main__":
    try:
        out = decide(json.load(sys.stdin))
    except (OSError, ValueError):
        out = None
    if out is not None:
        sys.stdout.write(json.dumps(out))
