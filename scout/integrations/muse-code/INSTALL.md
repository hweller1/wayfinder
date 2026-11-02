# scout-wayfinder for Muse Code

A native Muse Code plugin (`.muse-plugin/plugin.json`, Muse Code 1.4.13+, verified on 1.4.14). It adds a
read-only subagent, `scout-wayfinder/scout-explorer`. The subagent asks scout first (Wayfinder over a
voyage-code index of the repository), reads the top ranges to check them, fills gaps with search, and hands
back up to 12 ranked `path:start-end  why` lines, checked files first and scout's other files after them.

## Zero to working (three commands)

```bash
uv tool install /path/to/voyage-scout && scout login --key <scout key>     # 1. scout CLI + key (Keychain)
scout init --muse-code                                                      # 2. marketplace add, install, approve
cd /path/to/public/checkout && scout sync --wait && muse                    # 3. index the repo, start Muse
```

`scout init --muse-code` checks `muse --version` and then runs `integrations/muse-code/install.py`. The installer
is idempotent:

1. `muse plugins marketplace add voyage-scout <this dir>`, or `update` if the marketplace exists.
2. `muse plugins install scout-wayfinder@voyage-scout`, or `update` if installed.
3. `muse plugins approve scout-wayfinder`, which trusts the two hooks, the MCP server and the Agent Definition.
4. Writes the plugin's `scout.env` into Muse's plugin data dir.

`scout init --muse-code --dry-run` prints the steps without running them. Run it from a voyage-scout checkout, or
set `SCOUT_INTEGRATIONS_DIR=/path/to/voyage-scout/integrations` for an installed `scout`.

Without the scout CLI, use the demo lane:

```bash
uv run scout-demo index /path/to/checkout
python3 integrations/muse-code/install.py --lane demo
```

Isolated install, which leaves `~/.config/muse` untouched:

```bash
XDG_CONFIG_HOME=$T/config XDG_DATA_HOME=$T/data python3 install.py ...
```

Then run every `muse` command with the same two variables set.

Muse sends any code the agent reads to Meta. Use public repositories, or code you are allowed to share.
Embeddings go to Voyage AI.

## What the bundle contains

`scout-wayfinder/` is the plugin root. `.agents/plugins/marketplace.json` is the marketplace entry.

| Path | Muse capability | What it does |
|---|---|---|
| `.muse-plugin/plugin.json` | manifest (native, schemaVersion 1) | Declares the skill, three commands, two hooks and one MCP server. `muse plugins validate` returns valid with zero diagnostics. |
| `agents/scout-explorer.md` | Agent Definition | Runs on `muse-spark-1.3` at effort `low`. Write, edit, bash, web and memory tools are disallowed, and it is valid under `--disable-write --disable-shell` too. Procedure: scout `explore` first, check the top ~5 ranges with `read_file`, one or two `search`es for named identifiers, at most ~8 tool calls in all. It returns up to 12 files: checked ones first, then scout's unchecked ones (marked), wrong ones last only to fill. Then `Gaps:` and `scout: used/unavailable`. It is spawned as `subagent_type: scout-wayfinder/scout-explorer` and inherits the plugin MCP tools. |
| `skills/scout-explore/SKILL.md` | skill (`plugin:scout-wayfinder:scout-explore`) | Tells the main agent when to delegate: broad where/how questions, unfamiliar repos, multi-file answers. It also says when not to: exact strings, open files, trivial edits. It says how long to wait (1 to 3 minutes; `subagent_wait` with `timeout_ms: 300000`). |
| `commands/{index,status,find}.md` | commands `/scout-wayfinder:index`, `:status`, `:find` | `index` runs `scout sync --wait`, or `index_repo` in the demo lane. `status` reports the lane and index state. `find <what>` delegates to scout-explorer. |
| `hooks/session_start.py` | `SessionStart` hook | Reads the index status from local files (no network) and adds one context line: ready or not, lane, coverage. It starts `hooks/warm.py` detached, at most once per 5 minutes per checkout. In production, warm.py runs `scout watch --ensure` and `scout warm --trigger`; in the demo lane, `scout-demo warm --trigger`. The trigger hits the model's Modal `/warm` endpoint (through the gateway's `POST /v1/warm`, or directly in the demo lane), which returns in about a second and starts Wayfinder's 2 to 3 minute cold start while the user types. A gateway without the route falls back to one blocking 1-token completion. |
| `hooks/spec_session.py`, `spec_prompt.py`, `spec_pretool.py` (shared `spec_hooks.py`) | `SessionStart`, `UserPromptSubmit` and `PreToolUse` hooks | The retrieval spec (the retrieval spec, not in this release): resolves `.scout/spec.toml` over the harness and system layers (shared `lib/scout_spec.py`, a verbatim copy of the Claude Code plugin's), writes `spec_resolved.json` for the scout core, adds instructions, index description and mined examples as session context, adds the delegate-first workflow step to each prompt, and applies the search-first policy to the main thread's `search`. `SCOUT_SPEC_DISABLE=1` silences all three. |
| `hooks/route_explore.py` | `PreToolUse` hook | When scout is ready: (a) a generic `subagent_spawn` whose objective reads as read-only exploration is rewritten to scout-explorer through `updatedInput`; (b) the first main-thread `search` of a session, before any delegation, is denied once with a reason that names the subagent. A repeated search runs, so an exact-string lookup costs one retry. (c) A main-thread call to scout `explore` is sent to the subagent, which makes the same call and verifies it; inside the subagent it runs. `SCOUT_ROUTE=nudge` only adds a reminder; `SCOUT_ROUTE=off` disables (b). Decisions go to `route.log` in the plugin data dir. |
| `bin/launch_mcp.py` | MCP server `scout` (stdio) | Chooses the lane. Production runs `scout mcp --tools explore,index_status --render-ceiling 1500` against the hosted gateway; the repo is the session cwd. Demo runs `scout-demo mcp` (`explore(brief, repo)`, `list_repos`, `index_repo`). It mirrors the Claude Code launcher. |
| `lib/lanes.py` | (shared, stdlib only) | Config, lane choice and readiness, shared by the launcher and both hooks. |
| `hooks/fixtures/*.json` | `muse plugins hook test` fixtures | `{event, matcher_input, cwd, stdin}`, one per hook path. |

Tool names inside Muse: `mcp__plugin_scout_wayfinder_scout.explore` in the event stream. Agent `tools` selectors
use `mcp__plugin_scout-wayfinder_scout__explore`.

### Configuration

Muse starts plugin hooks and plugin MCP servers with a filtered environment. They get `HOME`, `PATH`,
`MUSE_PLUGIN_*` / `PLUGIN_*` and little else (measured). A `SCOUT_KEY` exported in your shell never reaches
them. Settings are therefore read from files, first match wins per name:

1. `$XDG_DATA_HOME/muse/plugins/data/scout-wayfinder/scout.env`, written by the installer.
2. `~/.config/scout/muse.env`.
3. The process environment.

Only these names are read:

- `SCOUT_LANE`
- `SCOUT_BIN`
- `SCOUT_KEY_FILE`
- `SCOUT_KEYSTORE`
- `SCOUT_GATEWAY_URL`
- `SCOUT_DEMO_PROJECT`
- `SCOUT_DEMO_HOME`
- `SCOUT_DEMO_PRESET`
- `SCOUT_RENDER_MAX_TOKENS`
- `SCOUT_ROUTE`
- `SCOUT_NO_WARM`
- `SCOUT_NO_WATCH`

The key comes from `scout login` (Keychain or `~/.config/scout/credentials`) or from a file named by
`SCOUT_KEY_FILE`. A key value is never written to `scout.env`.

### What Muse's plugin format cannot do, and what the bundle uses instead

- **No `agents` manifest family.** Native manifests support `skills`, `commands`, `hooks`, `mcpServers` and
  `reminders`; `agents` is reported as unsupported. Agent Definitions are discovered from the conventional
  `agents/*.md` directory instead. They are inventoried, need `muse plugins approve`, and spawn as
  `<plugin>/<name>`.
- **No hook `matcher`.** Native hooks reject the field. Both hooks run for every event of their type and filter
  on `tool_name` themselves (about 80 ms per call).
- **No built-in Explore subagent to replace.** Muse's built-ins are `general-purpose` and `workflow-subagent`.
  Interception therefore works at two points:
  - generic exploration spawns are rewritten;
  - the first main-thread `search` is gated once.

  In the live test a reminder alone was ignored: the parent grepped by hand.
- **`disallowedTools` may name only tools present in the session.** `subagent_spawn` is a session control and is
  rejected with `tool_classification`. `image_generation` is absent under `--disable-shell` and is rejected with
  `unknown_tool`. Either rejection makes the whole Agent Definition "invalid for the current runtime", so the deny
  list holds only tools that exist in every runtime.
- **Explicit `tools` grants accept MCP tools only as `mcp__plugin_<plugin>_<server>__<tool>`.** Other forms are
  rejected as `unknown_tool`. The explorer uses a deny list instead, so it keeps the read and search tools and
  the scout tools without naming them.
- **`muse exec` does not expand slash commands.** The commands work in the TUI. Headless runs use the skill and
  the subagent directly.

## Verify

```bash
muse plugins validate integrations/muse-code/scout-wayfinder --json      # valid, diagnostics []
muse plugins inspect scout-wayfinder --json                              # all runtime capabilities trusted_enabled
muse plugins hook test scout-wayfinder:route-explore \
  --fixture integrations/muse-code/scout-wayfinder/hooks/fixtures/route_spawn_explore.json --json
```

`make check` runs `tests/unit/test_integration_muse_code.py`, which contains:

- static checks;
- the lane, launcher and hook logic, using a scratch HOME;
- `scout init --muse-code` against a stub `muse`;
- when `muse` is installed, validate, marketplace install, approve and every hook fixture, in an isolated XDG.

The live headless runs are under `artifacts/live/muse_*` (see `docs/PRODUCT_STATUS.md`).
