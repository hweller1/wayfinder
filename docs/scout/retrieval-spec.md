# Retrieval spec

The retrieval spec tells scout and the agent that calls it how to search your repository. It has three layers. Later layers override earlier ones, so you only write what you want to change.

| Layer | Who sets it | Where it lives | What it holds |
|---|---|---|---|
| System defaults | scout | built into `integrations/claude-code/lib/scout_spec.py` | index what git tracks, skip `.gitignore` matches, binaries, lockfiles and vendored code; syntax-aware chunks; re-sync changed files every few minutes; 5 results per search; search opens at 80% indexed |
| Harness defaults | the plugin | `integrations/claude-code/spec/harness_defaults.toml` | instructions such as "prefer the project's own source over vendored code"; the delivery toggles; 3 examples mined from git history |
| User options | you | `.scout/spec.toml` and `.scoutignore` in the repository | instructions, path boosts and excludes, a recency boost, curated examples, `k`, and the switch that turns the index off |

## Where scout looks

The plugin reads `<repo>/.scout/spec.toml`. Commit it so the team shares it. `SCOUT_SPEC_FILE=/path/spec.toml` points at another file (used by the eval). `SCOUT_SPEC_DISABLE=1` silences the whole spec. `.scoutignore` keeps its existing meaning (gitignore syntax, read by sync, grep and read): a file listed there is never indexed and never described.

## The file

```toml
instructions = """
Prefer src/ over vendored code. Tests are supporting evidence.
"""
k = 5                          # results per semantic search, 1 to 30

[index]
enabled = true                 # false: grep only, Explore is not routed to scout
symbols = false                # add top type names to the index description

[paths]
boost = ["src/", "lib/"]       # rank these higher when scores are close
exclude = ["legacy/**", "*.snap"]   # never returned; adds to the built-in vendor/lockfile list
boost_weight = 0.3             # 0.3 means a boosted hit counts 30% more

[recency]
boost = 0.0                    # 0 to 1: favour recently modified files

[examples]
mine = 3                       # past commits shown as "description -> files touched", 0 to 5
max_files = 8                  # skip commits touching more files than this
avoid = []                     # never mine a commit touching these paths
before = ""                    # mine only commits before this one (default HEAD)
select = "recent"              # "recent" | "similar": similar picks the `mine` examples closest to each prompt
pool = 40                      # candidates mined when select = "similar"

[delivery]
session_context = true         # index description and examples at SessionStart
workflow_step = true           # first-step instruction on each prompt
search_first = false           # deny the main agent's Grep/Glob until its first delegation
search_policy = "deny"         # "deny" | "warn" | "off"; overrides search_first when set
handoff = "agent"              # "agent" | "verbatim" | "direct": how the parent receives scout's result
report_guidance = ""           # text added to the workflow step, e.g. "report the 5 to 8 most relevant files"
rewrite_explore = true         # route built-in Explore calls to scout

[explore]
max_turns = 0                  # 0 = no spec cap; 6 or 8 shortens the explorer's loop (lower of this and the request)

[[example]]                    # curated examples; they come before mined ones
query = "add retry to the http client"
files = ["src/http/client.py", "tests/test_http_client.py"]
```

Every key is optional. An unknown key, a value of the wrong type, or a file that is not valid TOML is ignored and reported under `warnings` in the resolved spec; the layers below still apply and the session is never blocked.

## Precedence rules

Scalars (`k`, the toggles, `boost_weight`, `recency.boost`, `index.enabled`) take the last layer that sets them. `instructions` are concatenated in layer order, so your text comes last and wins any conflict in wording. `paths.boost` and `paths.exclude` are concatenated; a `!pattern` in a later layer removes the same `pattern` from an earlier one, so `exclude = ["!vendor/"]` re-includes vendored code. Curated examples come first (yours, then the harness's), and mined examples fill the remaining slots up to five.

## The index description

Scout builds it from the files git tracks, minus the system excludes, your `paths.exclude` and `.scoutignore`: repository name and file count, languages by source file count, top-level directories with file counts, the largest packages (up to three path levels), and with `index.symbols = true` the type names declared in the largest source files. It is cached per commit and spec under `<git-dir>/scout/spec_cache/`, so a session start on an unchanged commit costs a file read.

## Mined examples and the leak guard

An example is a past commit: its subject line is the query and the files it touched are the answer. The miner walks the ancestors of `before` (default `HEAD`), never includes `before` itself, skips merges, commits touching more than `max_files` files, commits without a source file, and any commit touching a path in `avoid`. The result is deterministic. For an evaluation, set `before` to the task's base commit and `avoid` to the gold and held-out files, so no example describes the change being searched for.

## What reaches the agent

The plugin resolves the spec at SessionStart and writes `<git-dir>/scout/spec_resolved.json`. Four things read it.

1. **SessionStart context** (`session_context`): your instructions, the index description, the examples and the preferred paths, as additional context for the session.
2. **Workflow step** (`workflow_step`, on by default): on each prompt, "Before reading or editing code, delegate to the `scout:scout-explorer` subagent with a description of what to find; work from the files it returns." The target is built-in `Explore` when scout cannot answer for this checkout (no key, no index) or `index.enabled = false`.
3. **Search-first policy** (`search_first`, off by default): a PreToolUse hook denies the main agent's `Grep` and `Glob` until its first delegation or scout call, with a message pointing at the subagent. Subagents are never gated, and after the first delegation the main agent searches freely.
4. **The `scout-explorer` agent** has a description that tells the parent to delegate to it first, and reads the spec to word its brief. A built-in `Explore` call is rewritten to it when `rewrite_explore` is true and scout is ready, as before.

## The same spec on Muse Code

The `scout-wayfinder` plugin for Muse Code (`integrations/muse-code/`) reads the same `.scout/spec.toml` and `.scoutignore` with the same precedence. It carries a byte-identical copy of `scout_spec.py` and `spec/harness_defaults.toml` (a Muse plugin is installed on its own, so it cannot import across directories); `scripts/sync_muse_spec.py` refreshes the copy and a unit test fails when they differ. Parsing, the three layers, the index description and the git-history miner are that shared code. Only the delivery vocabulary differs, in `scout_spec.HOSTS`.

| Layer of delivery | Claude Code | Muse Code 1.4.14 |
|---|---|---|
| Session context | SessionStart hook | SessionStart hook `spec-session` (`context` effect, role developer) |
| Workflow step | UserPromptSubmit hook | UserPromptSubmit hook `spec-prompt`, worded for `subagent_spawn` then `subagent_wait` |
| Search-first policy | PreToolUse on Agent, Task, Grep, Glob | PreToolUse hook `spec-pretool` on `subagent_spawn`, scout tools and `search`; subagents are told apart by `MUSE_THREAD_ID != MUSE_SESSION_ID` |
| Delegate | `scout:scout-explorer` (built-in `Explore` as the fallback) | `scout-wayfinder/scout-explorer` (built-in `general-purpose` as the fallback: Muse has no Explore) |
| Agent description | `agents/scout-explorer.md` | `agents/scout-explorer.md`, same wording |

Differences to know. Muse hooks have no `matcher`, so each runs on every tool call and returns at once for tools it does not decide on. Muse filters hook environments, so `SCOUT_SPEC_FILE`, `SCOUT_SPEC_TARGET` and `SCOUT_SPEC_DISABLE` are read from the plugin's `scout.env`. `handoff = "verbatim"` has no relay subagent in the Muse plugin and behaves as `"agent"`; `"direct"` points the parent at scout's `explore` tool. The older one-shot search guard of `route_explore.py` steps aside when the resolved spec turns the search-first policy on, so a search is never denied twice. The scout core is unchanged: it reads `spec_resolved.json` written by `spec-session`, so instructed retrieval inside `explore` works the same on both hosts.

## Instructed retrieval inside scout

The scout core reads the same file on every call, so edits apply to the next search. The explorer model gets the instructions, index description and examples appended to its system prompt, so it plans its queries with them. Excludes drop semantic hits and reported ranges; boosts and the recency boost rerank semantic hits by a score adjustment (`score * (1 + boost_weight)` for a boosted path, plus `recency.boost` scaled by file modification time across the hits); the gateway asks for up to three times `k` hits when a filter can change the list. `k` caps the results of a search. `index.enabled = false` makes `semantic_search` answer "use grep" and the explorer grep-only. Nothing is retrained and the gateway is unchanged.

## Delivery options added by the hill-climb

These options exist so the delivery can be tuned; none changes a default. `handoff = "agent"` is the current Sonnet scout-explorer. `"verbatim"` points the parent at `scout:super-explore`, a relay subagent that returns scout's ranked report unedited. `"direct"` has the parent call scout's `explore` tool itself. `search_policy = "warn"` adds a hint to a main-agent Grep or Glob instead of denying it. `report_guidance` appends text to the workflow step and the session context. `examples.select = "similar"` ranks a mined pool by IDF-weighted token overlap with each prompt (same leak guard, applied when the pool is mined). `[explore] max_turns` caps the explorer loop. The 2026-10-30 hill-climb (wayfinder `docs/spec_hillclimb.md`) found no config that beat built-in Explore with the same spec on held-out CONFIRM tasks, so the defaults stay as listed in the table at the top: `handoff = "agent"`, search-first on in the evaluation, 3 recent mined examples, `k = 5`, no turn cap.

## Try it

```bash
python3 integrations/claude-code/lib/scout_spec.py resolve --root . --no-write
```

prints the resolved spec: every layer merged, the description, the examples and any warnings.

## Provenance

Passing one specification (instructions, labelled examples, an index description) to every retrieval step follows Databricks' Instructed Retriever. The delivery layers (workflow step, search-first policy, index description, history examples) reproduce round-4 arm D of the benchmark-repo benchmark, where delegation rose from 0% to 98% of runs and gold recall among files read rose by 0.26 with no change in pass rate.
