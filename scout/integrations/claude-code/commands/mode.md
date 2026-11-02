---
description: Choose how Explore is routed to scout (hybrid = verified, fast = one relayed call, off)
argument-hint: hybrid|fast|off
allowed-tools: Bash(python3:*)
---
!`python3 "${CLAUDE_PLUGIN_ROOT}/lib/scout_cc.py" mode "$ARGUMENTS" --data "${CLAUDE_PLUGIN_DATA}"`

Repeat the output above to the user verbatim inside a code block and nothing else.
