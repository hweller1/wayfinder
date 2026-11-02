# scout for Claude Code

The `scout` plugin (marketplace `voyage-scout`, version 0.2.0) puts scout's code search inside Claude Code. scout
is Wayfinder, a search model, running over an encrypted voyage-code index of the repository. When Claude
delegates a search to Explore and scout is ready for the repository, the plugin routes the call to
**scout-explorer**. That subagent sweeps the repository with one scout `explore` call, reads the ranges it
returns to verify them, fills gaps with Grep and semantic search, and hands back at most 12 ranked
`path:start-end` ranges with a one-line reason each. When scout is not ready, Explore runs unchanged.

Install: [INSTALL.md](INSTALL.md).

## Why a hybrid subagent

The design follows the measurements in `docs/scout_vs_explore.md`:

- **Replacing Explore with a relay lost recall.** A Haiku shim that makes one scout call scored 0.403 file
  recall@12, against 0.490 for the built-in Sonnet Explore. The shim could not check scout's answer or follow
  up on it.
- **Making scout tools available was not enough.** Built-in Explore with the scout tools attached called them
  0 times in 50 sessions.
- **Instructions worked.** Told to call `explore`, Explore did.

scout-explorer therefore runs on Sonnet with both scout tools and Claude Code's own Read, Grep and Glob, and it
is instructed to call scout first and verify afterwards. The relay shim is kept as a fast, cheap mode.

## Components

| Path | What it does |
|---|---|
| `agents/scout-explorer.md` | The hybrid subagent (Sonnet). Its tools are scout `explore`, `semantic_search`, `read_range`, `index_status`, plus Read, Grep and Glob, all read-only. Protocol: explore first, verify the top ranges by reading, fill gaps, return at most 12 verified ranges with no padding. If scout fails it says so and searches with Grep and Read. |
| `agents/super-explore.md` | The 0.1 relay shim (Haiku, one `explore` call returned verbatim). Used in `fast` mode. |
| `hooks/rewrite_explore.py` | PreToolUse on `Agent\|Task`. Routes `subagent_type: "Explore"` to the subagent of the project's mode when scout can answer; otherwise there is no output and the stock Explore runs. |
| `hooks/session_start.py` | SessionStart. Starts `scout watch --ensure` (keeps the index and `gateway_ok_at` fresh) and a detached warm-up worker. The worker probes `/v1/healthz` and then sends one 1-token chat (`scout warm`), so the 3.5 to 5 minute cold start of the scale-to-zero model is paid before the first search. Adds one line of context when the repository is indexed. Returns in well under a second. |
| `hooks/allow_explore_tool.py` | PreToolUse on `mcp__plugin_scout_scout__.*`. Allows the plugin's own read-only scout tools: a plugin cannot ship `permissions.allow`, and the default mode would deny a subagent's MCP call. |
| `commands/` | `/scout:index`, `/scout:status`, `/scout:find <question>`, `/scout:model glimmer\|v3\|a1`, `/scout:mode hybrid\|fast\|off`. |
| `skills/scout-search/SKILL.md` | Tells the main agent when to delegate to scout-explorer (broad where or how questions, unfamiliar repositories, flows across files) and when not to (a known file, a single-symbol grep). |
| `lib/scout_cc.py` | Shared stdlib code (Python 3.9+, never imports scout): lane, readiness, routing, settings, warm-up, and the command reports. |
| `scripts/launch-scout.sh` | MCP launcher. Production lane is `scout mcp --tools explore,semantic_search,read_range,index_status` on the project's gateway; demo lane is `scout-demo mcp`. |
| `install.py` | Idempotent installer and verifier; `scout init --claude-code` runs it. |

## Routing

The hook reroutes Explore only when all of the following hold. Otherwise the stock Explore runs and the user
sees no error.

- A scout key is configured: `SCOUT_KEY`, `SCOUT_API_KEY`, `~/.config/scout/credentials`, or the Keychain item
  that `scout login` writes. The hook checks attributes only; it never reads the secret.
- `status.json` (under `git rev-parse --git-path scout`) is `ready` or `partial`.
- The watcher's `gateway_ok_at` is under 10 minutes old.
- The scout MCP server's `mcp.pid` is alive.
- The last SessionStart probe of the project's gateway did not fail after that ping.

Without a key, a ready `scout-demo` index of the checkout routes the same way in the demo lane. Every decision
is logged to `${CLAUDE_PLUGIN_DATA}/hook.log` with lane, target and reason; the brief is never logged.

## Modes and models

`/scout:mode` chooses the routing target per project:

- `hybrid` (default): scout-explorer.
- `fast`: the Haiku shim.
- `off`: never reroute.

`SCOUT_EXPLORE_MODE` overrides the setting for a session.

`/scout:model v3|a1` chooses the gateway profile per project: `glimmer` (alias of the profile id `v3`, which still works) is wayfinder-glimmer on `scout-gateway`, a1 is
Agents-A1-35B on `scout-gateway-a1`. Both use the same keys and the same index. wayfinder-qwen is not yet served, so there is no `qwen` profile. Routing, warm-up, `/scout:index`
and `/scout:status` pick up the change at once, and it warms the new model in the background. The MCP server
that is already running keeps its gateway until it restarts: run `/mcp`, select `plugin:scout:scout` and
choose Reconnect, or start a new session; the launcher then exports the profile's `SCOUT_GATEWAY_URL`. An
explicit `SCOUT_GATEWAY_URL` in the environment always wins. `SCOUT_MODEL` overrides the setting for a
session. Settings live in `${CLAUDE_PLUGIN_DATA}/settings.json`.

## Status line

A plugin cannot set Claude Code's status line: only the `agent` and `subagentStatusLine` keys of a plugin's
settings are honoured. To show scout's state, add this to your own `settings.json`. The line it prints
looks like `scout v3 ready 100% warm hybrid`.

```json
{ "statusLine": { "type": "command",
  "command": "python3 /path/to/voyage-scout/integrations/claude-code/lib/scout_cc.py statusline --data ~/.claude/plugins/data/scout-voyage-scout" } }
```

A local-directory marketplace loads the plugin in place, so the path is the checkout's.

## Switches

| Variable | Effect |
|---|---|
| `SCOUT_EXPLORE_MODE` | Routing mode for the session (`hybrid`, `fast`, `off`). |
| `SCOUT_MODEL` | Gateway profile for the session (`v3` or its alias `glimmer`, `a1`). |
| `SCOUT_GATEWAY_URL` | Explicit gateway URL; overrides the profile. |
| `SCOUT_NO_WARM=1` | No warm-up call at session start or on a model switch. |
| `SCOUT_NO_WATCH=1` | No watcher at session start. |
| `SCOUT_LANE` | Force the `demo` or `production` lane. |
| `SCOUT_INDEX_WAIT_S` | How long `/scout:index` waits before leaving the sync in the background (default 100 s). |

## Verification

- `make check` runs the hook, command, skill and manifest tests offline: `tests/unit/test_hook_rewrite.py`,
  `test_session_start_hook.py`, `test_claude_code_commands.py`, `test_integration_claude_code_static.py` and
  `test_hook_readiness_parity.py`.
- The live smoke is `uv run python scripts/smoke_claude_plugin.py`. It runs headless `claude -p` with isolated
  config directories on gorilla/mux through the hosted gateway, and writes its artifacts to
  `artifacts/live/claude_plugin_smoke/`.
