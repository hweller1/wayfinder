---
description: Show scout's index state and lane for this repository
---
Report scout's status for the current repository in three short lines (name the search model wayfinder-glimmer when it is the default Scout model): lane (production or demo), index state
and coverage, and whether the scout-explorer subagent will use scout or fall back to plain search.

Use the scout MCP tools: `index_status` (production lane) or `list_repos` (demo lane; find the entry whose `root`
is this checkout). Do not run an explore call. If no scout tool is available, say the plugin's MCP server is not
running and suggest `muse plugins inspect scout-wayfinder` to check its approval.
