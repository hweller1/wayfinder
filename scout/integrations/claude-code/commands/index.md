---
description: Index (sync) this repository with scout and show progress and estimated cost
allowed-tools: Bash(python3:*)
---
!`python3 "${CLAUDE_PLUGIN_ROOT}/lib/scout_cc.py" index --data "${CLAUDE_PLUGIN_DATA}"`

Above is the output of `scout sync` for this repository. Repeat it to the user verbatim inside a code block,
then add one short sentence: if the index finished and is ready, say Explore and /scout:find now use scout;
if it is still running, say it continues in the background and /scout:status shows progress; if it failed or
the production lane is unavailable, state the next step the output gives. Do not run any other command.
