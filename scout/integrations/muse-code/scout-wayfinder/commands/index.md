---
description: Index (or refresh) this repository for scout
argument-hint: "[path or git URL]"
---
Index the repository for scout so the scout-explorer subagent can search it. Target: `$ARGUMENTS` if given,
otherwise the current git checkout.

1. Call the scout MCP tool `index_status` if it exists (production lane). If the state is `ready` and coverage
   is above 0.8, say the index is current and stop.
2. Production lane (a `scout` binary and key are configured): run `scout sync --wait` from the repository root
   with the shell tool and report the final `state`, `coverage` and `files` from its JSON output.
3. Demo lane (the scout tools are `explore`, `list_repos`, `index_repo`): call `index_repo` with `source` set to
   the target path or URL and report the returned `state`, `files` and `est_cost_usd`.
Only index public code or code the user owns and has asked to index; embeddings go to Voyage AI.
