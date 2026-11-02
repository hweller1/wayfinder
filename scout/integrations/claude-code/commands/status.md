---
description: Show scout's index freshness, lane, model and whether the search endpoint is warm
allowed-tools: Bash(python3:*)
---
!`python3 "${CLAUDE_PLUGIN_ROOT}/lib/scout_cc.py" status --data "${CLAUDE_PLUGIN_DATA}"`

Repeat the status above to the user verbatim inside a code block. Then, only if something is not ready,
add one sentence naming the fix (/scout:index for no or stale index, `scout login` for no key, wait a few
minutes for a warming endpoint). Do not run any other command.
