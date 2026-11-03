# scout product status (2026-10-28)

This document says what works today in voyage-scout, how to install and run it for each host, how the hosted gateway is operated, and what is still missing. The spec of record is `docs/PRD.md`; task state is in `specs/super-explore/tasks.md`. Numbers here come from runs in this repository (artifacts under `artifacts/live/` and `artifacts/eval/`). The eval ran on 2026-10-27; its result is in the section below and in the Scout-versus-Explore study write-up (not in this release).

## What works

The client, the gateway and the integrations run end to end on real services. A user holds one scout key. `scout login` verifies it against the hosted gateway and stores it in the macOS Keychain (file fallback `~/.config/scout/credentials`, mode 0600). `scout sync` indexes a git checkout: files are chunked on the client, paths and line ranges are encrypted (AES-SIV and AES-GCM), identifier tokens are HMAC-hashed, and only ciphertext and vectors reach turbopuffer. Chunk text goes to the gateway for embedding and is not stored. `scout search`, `scout explore`, `scout grep` and `scout read` query the index through the same facade, and every returned range is checked against the file on disk before it is shown.

Evidence, all live:

| Claim | Evidence |
|---|---|
| Store contract holds on real turbopuffer | `tests/live/test_store_suite_live.py` runs all 19 cases of `tests/contract/store_suite.py` (including conditional-upsert CAS and the vector-less cases) on `TurbopufferStore`; 19 passed, zero `scout_dev_suite_` namespaces left (`ops/ns_janitor.py --verify-empty`). FakeStore needed no change. |
| Voyage returns 1024-dim unit-norm vectors | `tests/live/test_voyage_live.py` on three synthetic strings |
| Hosted gateway works from a key alone | `artifacts/live/gateway_smoke.json`: sync of gorilla/mux (27 files, 29 s), semantic search, explore (25 s, 7 turns), tenant isolation with a second key, `delete_branch` cleanup |
| Claude Code plugin, production lane | `artifacts/live/claude_smoke.json`: headless `claude -p` with `SCOUT_KEY` set rewrites Explore to `super-explore:super-explore` (hook log `rewrite lane=production`), the shim calls `scout mcp`, answers with `mux.go:220-222` style ranges, no permission denial on the scout tool |
| Gateway on wayfinder-glimmer | `artifacts/live/gateway_smoke_v3.json` (sync 65 s, explore 66 s over 5 turns, isolation holds); A1 gateway: `gateway_smoke_a1.json` |
| Claude Code plugin on wayfinder-glimmer | `artifacts/live/claude_smoke_v3.json` (3 of 3 cases pass; A1 gateway production lane: `claude_smoke_a1.json`) |
| Claude Code plugin, demo fallback and not-ready fall-through | same file, `rewrite` and `fallback_ok` entries |
| No plaintext reaches the store | `tests/security/test_no_plaintext.py` runs the full client flow through a `RecordingStore` on `repo_small` plus `declaration_real` and scans for path segments, identifiers and 20-byte source windows; three planted leaks (plaintext path, raw BM25 token, chunk text) are caught |

The default suite (`make check`) runs about 1,900 tests with no network.

## Install and run

### Claude Code

`integrations/claude-code/` holds the native `scout` plugin, version 0.2.0 (marketplace `voyage-scout`; it was `super-explore` 0.1). Install, after `uv tool install` and `scout login`:

```bash
SCOUT_INTEGRATIONS_DIR=/path/to/voyage-scout/integrations scout init --claude-code   # validate, add, install, verify
cd /path/to/public/checkout && scout sync --wait && claude
```

`scout init --claude-code` runs `install.py`. It validates the plugin, adds or updates the marketplace, installs `scout@voyage-scout`, then verifies the install: the plugin is listed and enabled, and a doctor checks python, the `scout` binary, the key and the gateway's healthz. `CLAUDE_CONFIG_DIR` scopes the install, so tests never touch `~/.claude`.

**Routing.** When Explore is called and scout can answer for the checkout, the PreToolUse hook routes it to **scout-explorer**. "Can answer" means all of:

- a key is configured;
- `status.json` is ready or partial;
- `gateway_ok_at` is under 10 minutes old;
- the MCP server is alive;
- the last SessionStart probe of the gateway and model did not fail.

scout-explorer is a Sonnet subagent. Its tools are scout `explore`, `semantic_search`, `read_range` and `index_status`, plus Read, Grep and Glob. It calls `explore` first, verifies by reading, fills gaps, and returns at most 12 verified `path:start-end` ranges. Otherwise the stock Explore runs and the user sees no failure. A model that rejected the warm-up blocks routing for 10 minutes, and a retry warm-up starts in the background.

**Other pieces.**

- The SessionStart hook starts `scout watch --ensure` and a detached warm-up: a healthz probe, then one 1-token chat through `scout warm`. The 3.5 to 5 minute cold start is paid before the first search.
- `/scout:mode hybrid|fast|off` chooses the routing target. `fast` is the 0.1 Haiku relay shim, kept as a cheap mode.
- Slash commands: `/scout:index` (sync with progress and an embedding-cost estimate), `/scout:status` (index freshness, lane, model, endpoint warm or cold, routing), `/scout:find <question>`, `/scout:model v3|a1`.
- `/scout:model v3|a1` switches the gateway profile per project (`glimmer` is accepted as an alias for `v3`; `/scout:status` shows the model as wayfinder-glimmer). The profile id `v3` is an internal id and was not renamed. The running MCP server picks the change up on `/mcp` Reconnect or in a new session.
- A `scout-search` skill tells the main agent when to delegate and when not to.
- A plugin cannot set the status line, so `scout_cc.py statusline` is documented for the user's own settings.

Without a key, the demo lane (`scout-demo`) routes the same way. All plugin logic is stdlib Python 3.9+ in `lib/scout_cc.py`.

**Verification.** `make check` covers the hooks (routing, modes, fall-through, model failure, readiness parity with core), the command scripts, the skill, agent and command frontmatter, and the installer.

The live smoke, `scripts/smoke_claude_plugin.py`, runs headless `claude -p` with isolated config dirs on gorilla/mux through the hosted gateway. Artifacts are in `artifacts/live/claude_plugin_smoke*/`. It ran on 2026-10-28 with search model A1, because wayfinder-glimmer was down:

- Explore was routed to `scout:scout-explorer`, which called scout `explore` first, then Grep and three Reads. It returned 7 ranked `mux.go`/`route.go` ranges with reasons in 92 s, for $0.095.
- With the gateway unreachable, the SessionStart probe recorded it, the hook fell through, and built-in Explore answered in 9 s.
- `/scout:status`, `/scout:model a1|v3`, `/scout:mode fast|hybrid`, `/scout:index` (27 files, about $0.01) and `/scout:find` all worked.

**Calibration (Addendum C of the Scout-versus-Explore study write-up (not in this release), 5 tasks, A1 gateway).** `hybrid_a1` scored recall@12 0.167, against 0.436 for built-in Explore in the same pass and 0.085 for the A1 relay `super_a1`. The pre-registered gate therefore failed and no 50-task run was made.

Verification on the same model doubled recall over the relay. But the subagent mostly Grepped instead of reading (0.6 reads per session), returned short lists (6.2 files), and took about 145 s, mostly in the A1 explore call. The agent file was tightened afterwards to make Read the verification step; that version is untested. **Calibration on wayfinder-glimmer (Addendum D, 5 tasks).** After wayfinder-glimmer was fixed, the subagent was revised:

- up to 12 files, verified first, scout's unread hits below;
- Read the top ~5;
- about 8 tool calls in total.

`hybrid_v3` scored recall@12 0.309 against 0.349 for built-in Explore. It was ahead on precision@12 (0.408 against 0.343) and recall@5 (0.284 against 0.257), with 7.4 files against 10.4, in about 157 s against 21 s. The gate was "at least even", so it failed narrowly and no 50-task run was made.

### Muse Code

`integrations/muse-code/` holds a native Muse Code plugin, `scout-wayfinder`. It was verified end to end on Muse
Code 1.4.14 on 2026-10-28, in both lanes. The install is three commands:

```bash
uv tool install /path/to/voyage-scout && scout login --key <scout key>
scout init --muse-code      # marketplace add, install and approve, all idempotent
cd <public checkout> && scout sync --wait && muse
```

The bundle uses every capability family Muse's plugin format offers:

- **Agent Definition `scout-wayfinder/scout-explorer`.** Runs on `muse-spark-1.3` at effort `low`, read-only. It
  calls scout `explore` first, verifies the top ranges with `read_file`, fills gaps with `search`, and returns at
  most 12 ranked `path:start-end` lines.
- **Skill.** Tells the main agent when to delegate and how long to wait.
- **Commands.** `/scout-wayfinder:index`, `:status` and `:find`.
- **SessionStart hook.** Reports index readiness. It starts `scout watch --ensure` and `scout warm` in the
  background; `scout warm` is new, a one-token chat through the gateway that begins Wayfinder's cold start.
- **PreToolUse hook.** Rewrites generic exploration spawns to scout-explorer. It denies the first main-thread
  `search` of a session once, with a pointer to the subagent, and sends a main-thread scout `explore` call to the
  subagent.
- **MCP server.** The production lane runs `scout mcp`; the demo lane `scout-demo mcp` is the fallback.
- **Marketplace entry.** At `integrations/muse-code/.agents/plugins/marketplace.json`.

`muse plugins validate` reports valid with zero diagnostics. After `approve`, `muse plugins inspect` shows all
four runtime capabilities `trusted_enabled`: two hooks, the MCP server and the Agent Definition. Every hook
passes `muse plugins hook test` against its fixtures, and `make check` runs those tests in an isolated XDG.

Muse's format has gaps. It has no `agents` manifest family (definitions come from `agents/*.md`), no hook
`matcher`, no built-in Explore subagent to replace, and `muse exec` does not expand slash commands. INSTALL.md
lists each gap and the mechanism used instead.

Live headless runs used `muse exec --json`, an isolated XDG and a temp clone of gorilla/mux. The prompt was
"where does the router produce a 405 and how can a user customise it". Artifacts are under
`artifacts/live/muse_demo/`, `artifacts/live/muse_production/` and `artifacts/live/muse_smoke.json`.

- **Production lane** (hosted gateway, wayfinder-glimmer): this closes the earlier "production lane unverified" gap.
  1. The parent read the skill and spawned `scout-wayfinder/scout-explorer`.
  2. The explorer's first call was `mcp__plugin_scout_wayfinder_scout.explore`, followed by 8 verification reads
     and 4 gap searches.
  3. It handed back 12 ranked ranges, the first being `mux.go:669-676` (default 405 handler), and ended with
     `scout: used`.
  4. Wall time was 181 s. wayfinder-glimmer was cold: the session-start warm-up took 107 s, and the explore call hit its 120 s
     budget and returned partial candidates, which the explorer's verification turned into a correct list.
  5. In an earlier run without the search guard, the parent ignored the reminder and grepped by hand. The guard
     came from that run.
- **Demo lane** (`scout-demo`, A1-35B):
  1. The explorer ran `list_repos`, then `explore`, then 12 verification reads and one gap search.
  2. It handed back 11 ranked ranges (`route.go:52-60`, `mux.go:17-20`, ..., `mux.go:164-171`, `mux.go:220-222`, `mux.go:669-676`)
     and ended with `scout: used`. Wall time was 131 s.

**Calibration** (`artifacts/live/muse_calibration/`, `eval/harness/muse_calibration.py`). This is not a claim:
5 tasks, no CIs.

- **Setup.** 5 public tasks from `eval/eval_tasks.jsonl`, the first one of each repo: mlx-vlm, deepagents,
  Genexus18MCP, mteb and skypilot (Python and C#). Each task ran at its `base_commit`.
- **Arms.** Muse Code with the plugin (production lane, wayfinder-glimmer) against Muse Code alone. Both used
  `muse-spark-1.3` at effort `low`, `--disable-write --disable-shell`, and the same read-only brief, which ends by
  asking for a ranked JSON file list.
- **Scoring.** File recall@12 of the final answer.

| Arm | recall@12 (final) | recall@12 (explorer handback) | wall s mean (median) | tool calls mean | parent input tokens mean | files listed mean |
|---|---|---|---|---|---|---|
| Muse + scout-wayfinder | 0.274 | 0.274 | 205 (198) | 36.6 | 291k | 8.6 |
| Muse alone | 0.365 | n/a | 39 (40) | 25.6 | 630k | 11.0 |

Per task, plugin against alone:

| Task | Plugin | Alone |
|---|---|---|
| mlx-vlm | 0.50 | 0.50 |
| deepagents | 0.25 | 0.38 |
| Genexus18MCP | 0.00 | 0.00 |
| mteb | 0.29 | 0.29 |
| skypilot | 0.33 | 0.67 |

The plugin path behaved as designed in all 5 runs:

- the parent delegated to scout-explorer;
- the explorer called scout `explore` once and then verified;
- in 4 of 5 runs the parent first tried scout `explore` itself, and the route hook sent it to the subagent.

On these tasks the plugin did not raise recall and was about 5 times slower. Wall time is dominated by the
2 to 3 minute scout call on wayfinder-glimmer. The plugin parent read half as many input tokens, and its lists were shorter
(8.6 files against 11.0), which recall@12 does not reward.

This is the same direction as the Claude Code eval: Muse's own grep-and-read explorer is fast and competitive.
The instructed-explorer pattern now works mechanically in Muse; whether it helps needs a powered run, ideally
with wayfinder-glimmer warm and a precision metric.

**Iteration (2026-10-28, later).** Same 5 tasks, same plain-Muse rows reused, plugin arm only. wayfinder-glimmer was warm
from other traffic. Artifacts are in `artifacts/live/muse_calibration_v2/` and `_v3/`.

- **Explorer v2.** Returns up to 12 files: checked files first, then scout's unchecked files in its order.
  Checking reads are limited to the top ~5, with ~8 tool calls in total.
- **Explorer v3.** Also fills spare slots from cheap `files_with_matches` searches. The skill tells the parent
  to keep the explorer's list whole and in order.

| Arm | recall@12 final | recall@12 handback | handback files | wall s mean | parent input tokens |
|---|---|---|---|---|---|
| Muse alone | 0.365 | n/a | n/a | 39 | 630k |
| plugin v1 (first run) | 0.274 | 0.274 | 7.2 | 205 | 291k |
| plugin v2 | 0.315 | 0.307 | 9.2 | 186 | 354k |
| plugin wayfinder-glimmer | 0.302 | 0.327 | 8.8 | 223 | 298k |

The handback improved (0.274 to 0.327) but is still below plain Muse, so the 20-task arm was not run; its task
set is defined as `--set ext20`.

Two causes remain:

- Scout's own list often has fewer than 12 files, and the explorer rarely fills the rest.
- The parent still trims the handback in its final answer (deepagents: handback 0.38, final 0.25).

Wall time is dominated by the scout `explore` loop itself, about 2 to 3 minutes even on a warm wayfinder-glimmer, not by cold
start.

**Warm fix.** Code done, deploy blocked:

- The gateway has a new `POST /v1/warm`. It calls the model's Modal `/warm` endpoint, which returns
  `{"warming": true}` in about 0.75 s (measured once).
- `scout warm --trigger` uses that route; the plugin worker calls it. Without the route it falls back to the old
  blocking 1-token call.
- Redeploying `scout-gateway` was refused by this session's permission check, as was a deploy under a separate
  app name. The live gateway therefore still has the old behaviour.
- The 3-cold-start confirmation did not run either. Stopping wayfinder-glimmer containers to force zero was refused, and
  waiting for wayfinder-glimmer to scale to zero never happened because others kept using it. The script for it is
  `ops/warm_cold_check.py`.
- `scout warm` without `--trigger` keeps the blocking semantics, because the Claude Code plugin's worker reads
  `warm: true` as "model answered".

`scout warm` returned INVALID_ARGUMENT after about 151 s on a cold wayfinder-glimmer in 3 of 7 warm-ups. It still triggers the
cold start, and later calls succeeded. The 4xx is probably the upstream proxy timing out, and it is not
diagnosed.

Muse sends any code the agent reads to Meta, so index and read public repositories only. Muse filters the
environment of plugin hooks and MCP servers (measured: `HOME`, `PATH`, `MUSE_PLUGIN_*` and little else), so the
plugin reads its settings from `scout.env` in its data dir. Keys come from `scout login` or a key file and are
never written there.

### OpenCode and any other MCP host

No OpenCode, Codex or Cursor directory exists yet (tasks 11.7 to 11.9), and none of these hosts has been run. The server is a standard MCP stdio server, so any host that accepts a command can use it:

```json
{ "mcpServers": { "scout": { "command": "scout", "args": ["mcp", "--tools", "explore"],
  "env": { "SCOUT_KEY": "<scout key>" } } } }
```

Exposed tools: `explore`, `semantic_search`, `grep`, `read_range`, `index_status` (`--tools` narrows the set; `--structured` returns JSON; `--http PORT` serves streamable HTTP on loopback). Output is one plain-text block, capped by `--render-ceiling`. Run the host from inside the git checkout, because the server derives the repo from its working directory. For hosts that cannot speak MCP, `scout serve` exposes the same operations over a loopback HTTP daemon with a bearer token, and `scout explore --json` works from any script.

## Gateway

The gateway is one CPU ASGI app on Modal, `deploy/modal/gateway.py`, at `https://scout-gateway.example.com`. `GET /v1/healthz` is unauthenticated and returns `{"ok":true,...}`; everything else needs `Authorization: Bearer <scout key>`. It scales to zero, so the first request after idle takes a few seconds (7.7 s in the smoke).

Operations:

- Deploy with `modal deploy deploy/modal/gateway.py`. The deploy builds a wheel of the working tree, so the deployed code is whatever is checked out.
- Issue a key with `modal run deploy/modal/gateway.py::issue_key --tenant <name>`. The key is printed once; only its HMAC tag is stored. A key maps to exactly one tenant, and tenants cannot see each other's namespaces.
- Vendor credentials (turbopuffer, Voyage, the chat upstream) live only in the Modal secrets `scout-gateway` and `scout-llm`. Clients never hold them.
- Explore calls the chat model through `/v1/chat`. The default model is wayfinder-glimmer (a post-trained Muse Glimmer 30B), served by `deploy/modal/serve_glimmer.py` on one H100 with scale to zero, or by the GCP kit in `deploy/gcp`. The gateway reads its upstream from `SCOUT_LLM_URL`, `SCOUT_LLM_API_KEY`, `SCOUT_LLM_MODEL` and `SCOUT_LLM_SAMPLING` in the single Modal secret `scout-gateway` (created by `deploy/modal/make_secrets.py`). `SCOUT_LLM_SAMPLING=sft` makes the adapter send only temperature and top_p, the protocol wayfinder-glimmer was trained and evaluated under.

## Known gaps

The trust boundary is as stated in the PRD: the index (turbopuffer) never holds code, but chunk text passes through the gateway and Voyage's embedding API to be embedded, so the gateway operator can read it in transit and unwrap `K_team`. Embeddings can be partly inverted, and deterministic path encryption shows which rows share a path segment. The README trust-boundary section (task 11.10) is not written.

Open work, in rough order of consequence:

1. Voyage zero-retention opt-out on the `al-` key is an author action (task 12.1). Until it is confirmed, only public or synthetic code should be embedded; the live tests that send real code are gated on `SCOUT_VOYAGE_OPTOUT_CONFIRMED=1`.
2. The scout-vs-Explore eval ran on 2026-10-27 (see "Eval result" below) and super-explore did not beat the built-in Explore on retrieval. The formal parity run (13.8), the Phase 0 GPU spike (10.6) and the latency and budget defaults that depend on it have not run. Explore takes about 2 to 3 minutes per session under 20-way concurrent load on one H100 (wayfinder-glimmer median 163 s, A1 122 s in the eval).
3. The teammate seed has been tested on fakes and in the contract suites, not on two real clones against the hosted gateway (12.4). The PRD target of under 5 s to first query is unmeasured live.
4. OpenCode, Codex, Cursor and generic integration directories are missing. Muse Code is verified in both lanes on gorilla/mux. It has a 5-task calibration but no powered eval.
5. The headless Claude smoke needs a logged-in `claude` and a dev key in `~/.config/scout/live-smoke-keys.txt`; it cannot run in CI. The demo-lane cases need `gomux` indexed under `~/.scout-demo`.
6. `scout init` finds `integrations/` only next to a source checkout. Packaging them into the wheel is not done.
7. The demo lane leaves `scout_dev_demo_*` namespaces in turbopuffer (see the demo smoke-test notes (not in this release)); delete them with the janitor when finished. The tenant `cache` and `meta` namespaces persist by design.
8. No PR exists and CI has not run; many `tasks.md` boxes are unchecked because the author marks them, not because the work is missing.

## Eval result (2026-10-27)

Full write-up: the Scout-versus-Explore study write-up (not in this release) (pre-registered in a66e4a2 before the full run; rows and statistics in `artifacts/eval/`). Setup: 50 public held-out tasks in 10 repos, headless `claude -p` (Sonnet 5.5 parent through an LLM API gateway, isolated config dirs), three arms: built-in Explore, super-explore on wayfinder-glimmer, super-explore on A1-35B. Metric: file recall@12 of the Explore handback, paired by task, 10,000-resample task bootstrap.

| Arm | recall@12 [95% CI] | wall s (median) | session cost |
|---|---|---|---|
| built-in Explore | 0.490 [0.414, 0.564] | 29 | $0.090 |
| super-explore, wayfinder-glimmer | 0.403 [0.325, 0.486] | 163 | $0.041 |
| super-explore, A1 | 0.352 [0.269, 0.437] | 122 | $0.040 |

The pre-registered contrast, wayfinder-glimmer minus built-in Explore, is -0.086 [-0.153, -0.021] (repo-cluster CI [-0.135, -0.037]): built-in Explore retrieved more. wayfinder-glimmer minus A1 is +0.051 [-0.015, +0.116], not significant. Scout sessions were cheaper (the Haiku shim replaces a Sonnet subagent; GPU cost is not included) and 4 to 5 times slower under 20-way concurrent load. There is no coding-outcome metric. The eval found and fixed a sync bug (embed requests over 500 entries on boilerplate shared by many files).

### Augment arm (2026-10-27, later the same day)

Addendum A of the Scout-versus-Explore study write-up (not in this release) (pre-registered in cc5b3f7) tested scout as a tool for the built-in Explore instead of a replacement: arm `augment_v3`, the same 50 tasks, built-in Explore with `scout mcp --tools explore,semantic_search` attached and nothing rewritten or blocked. Result: recall@12 0.485 [0.401, 0.566] against 0.490 for built-in Explore alone, paired difference -0.005 [-0.041, +0.030] (cluster CI [-0.039, +0.031]), no difference; wall time 35 s and cost $0.093 per session, both within noise of built-in Explore. The reason is that the subagent never called a scout tool in any of the 50 sessions (it used Grep, Bash, Read and Glob, 5.3 calls per task), also not with the tool schemas loaded eagerly on three probe tasks. When told explicitly to call `explore` it did and the call took about 2 minutes. So the open question has a narrow answer: left to choose, Explore prefers its own fast search; whether scout helps an Explore that is instructed to use it is untested. The harness gained the arm (`eval/harness/arms.py`, usage statistics in `report.py`);

### GRPO arm a step-20 as the scout model (2026-10-28)

Addendum B of the Scout-versus-Explore study write-up (not in this release) (pre-registered in b6132e5) reran the replace setup with a new candidate checkpoint, `super_grpo_a20` (merged `/data/grpo/a/s20_merged`, served by `wayfinder/serve/wayfinder_v3.py` under the app name `wayfinder-a20-serve`, behind `scout-gateway-a20`), on the same 50 tasks. Primary contrast against `super_v3` on handback recall@12: -0.034 [-0.087, +0.016] (cluster CI [-0.082, +0.014]), no difference, point estimate negative. recall@12 is 0.370 [0.299, 0.441] against 0.403 for wayfinder-glimmer and 0.490 for built-in Explore (a20 minus built-in Explore -0.120 [-0.187, -0.058]). The handback is the longest of any arm (10.2 files against 9.6 for wayfinder-glimmer) with lower precision at 12 (0.268 against 0.299), and recall@5 and recall@7 are not above wayfinder-glimmer, so the dev-eval gain looks like longer reports, not better retrieval. All-gold fell to 0.02 from 0.10. Wall time 152 s median, session cost $0.042. `wayfinder-a20-serve` back at zero containers, `scout_eval_*` namespaces deleted and verified empty. The default model stays wayfinder-v3.

## Backfill to 12 files (2026-10-28)

`explore.backfill_to` (default 12, 0 is off; env `SCOUT_EXPLORE_BACKFILL_TO`, config `[explore]`) pads every `explore` result in code, inside the core use case: files the model opened but did not report, then top semantic hits for the brief, then grep hits for identifiers in the brief. The model's ranking stays first; added entries carry `verified: false` and render under "Further candidates (unverified)". MCP, CLI, SDK, daemon, gateway-backed and demo lanes all go through `ScoutApp.explore`, so all return it. The Claude Code and Muse scout-explorer prompts pass the section through in order. `make check` is green with unit and contract tests for dedup, order, the flag, an empty index and a failing semantic tier.

Addendum E (5 tasks, wayfinder-glimmer): the gate failed for both arms. `hybrid_v3_bf` recall@12 0.311 against 0.372 for fresh `explore_default` (it was 0.309 without backfill); files 11.8 (was 7.4); precision@12 0.239 (was 0.408). `super_v3_bf` 0.235. Longer lists did not buy recall: the filler is mostly not gold. No 50-task run. The backfill is client-side, so it needs no gateway redeploy; the default is on, and it can be switched off with `SCOUT_EXPLORE_BACKFILL_TO=0` if the author prefers the shorter, more precise list. The `deploy/modal/gateway.py` redeploy was denied by the permission check and is left to the author.
