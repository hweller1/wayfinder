---
description: Switch scout's search model for this project (glimmer or v3 = wayfinder-glimmer, a1 = Agents-A1-35B)
argument-hint: glimmer|v3|a1
allowed-tools: Bash(python3:*)
---
!`python3 "${CLAUDE_PLUGIN_ROOT}/lib/scout_cc.py" model "$ARGUMENTS" --data "${CLAUDE_PLUGIN_DATA}"`

Repeat the output above to the user verbatim inside a code block and nothing else. Do not run any other
command.
