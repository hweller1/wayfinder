# scout super-explore vs Claude Code's built-in Explore

Pre-registered 2026-10-27, before any full-50 run. The sections down to "Results" are fixed; results are appended below them and the text above is not edited afterwards.

## Question

Does Explore replaced by super-explore (the scout MCP tool, run by a hosted model on the scout gateway) hand the parent agent better file retrieval than the built-in Explore, and what does it cost in wall time and dollars? Two scout models are compared: wayfinder-glimmer (SFT, post-trained Muse-Glimmer-30B) and Agents-A1-35B.

## Design

Three arms, same 50 tasks, same brief, same parent.

| Arm | Explore subagent | Search backend |
|---|---|---|
| `explore_default` | built-in Explore | grep, glob, read and shell tools of the subagent |
| `super_v3` | `super-explore:super-explore` (rewritten from Explore by the plugin hook) | scout `explore` on the hosted gateway, chat model wayfinder-glimmer |
| `super_a1` | same | same gateway code, chat model Agents-A1-35B |

Tasks: the 50 tasks of `eval/eval_tasks.jsonl` (10 public repos, 5 each: 25 Python, 10 Go, 10 C++, 5 C#). All 10 repos are in Wayfinder's `eval_holdout` split, so neither model trained on them; no train-split task and no private repo is used. Each task is run at its own `base_commit`.

The parent is `claude -p` on claude-sonnet-5-5 with the same fixed prompt in every arm. Its only permitted tool is `Agent` (a PreToolUse gate hook denies everything else for the parent, not for the subagent), and it must call `Agent` once with `subagent_type: "Explore"` and the brief. The brief is the task question plus the suffix "End with a JSON list of the repo-relative files you consider most relevant, best first." The parent then repeats the handback. The built-in Explore resolves to claude-sonnet-5-5 under this routing (it inherits the parent's model; verified in `tool_use_result.resolvedModel`). The shim's `model: haiku` resolves to claude-haiku-5-5 through `ANTHROPIC_DEFAULT_HAIKU_MODEL`. Settings are identical across arms except that the scout arms load the plugin (`--plugin-dir integrations/claude-code`) and carry a scout key and gateway URL (production lane). Each session has its own `CLAUDE_CONFIG_DIR`; the user's global Claude config is never read. The parent runs through the LLM gateway to Anthropic models, not through the user's Fireworks settings.

The scout arms index each repo once per task commit with `scout sync --wait` under tenant `eval` (namespaces `scout_eval_*`). The two scout arms use separate checkouts so each has its own readiness state, and the same index content (same master key, same embeddings). Task outcome metrics (a coding result) are not part of the spec, which defines retrieval only; none is measured.

## Metrics

Primary, fixed in advance: **file recall@12 of the handback**, `|gold ∩ top12| / |gold|` at file level, where the candidate list is the repo files named in the Explore handback text (what the parent receives), validated against `git ls-files` at `base_commit` after stripping a checkout-root prefix, `./` and `:start-end` suffixes, ranked by first mention. The primary contrast is **`super_v3` minus `explore_default`**, paired by task, 95% percentile bootstrap CI over tasks (10,000 resamples, seed 20261025).

Secondary, also fixed now:
- `super_a1` minus `explore_default`, and `super_v3` minus `super_a1`, same statistics.
- All-gold-found rate (recall@12 equal to 1) with paired deltas.
- Parent final-answer recall@12 (same extraction on the parent's last message).
- Wall time per session (mean, median, p90), Explore-subagent duration, session cost in USD (`total_cost_usd`), parent cost (recomputed from stream-json usage at the benchmark's Sonnet 5.5 rates), subagent cost (total minus parent), parent turns.
- Repo-cluster bootstrap CI (resampling the 10 repos) as a sensitivity check, since tasks within a repo are correlated.
- Recall delta per language (Python, Go, C++, C#), reported as labelled strata with their n; no pooled claim stronger than the per-language picture supports.

Decision rule: the primary contrast counts as a result only if its task-bootstrap CI excludes zero; the cluster CI is reported next to it and a disagreement is stated. Everything else is descriptive.

## Row validity

A row enters the scored set only if it carries the brief hash, wall time of at least 2 s, a non-empty handback, a cost, a resolved subagent model, and no zero-tools refusal. A scout-arm row counts as scored only if the plugin hook logged `rewrite lane=production` and the subagent type was `super-explore:super-explore` (or a scout MCP call is in the stream); otherwise it is a fallthrough row, retried once, and if it still falls through it is reported as such and excluded from the paired comparison. Paired comparisons use only tasks with a valid row in all three arms. Failed or excluded rows are listed in `artifacts/eval/results.json`.

## Budget

Anthropic usage is estimated before the run. One session is a parent Sonnet call or two plus the subagent: smoke probes on gorilla/mux cost $0.04 to $0.09 per built-in Explore session (Sonnet subagent with a few tool calls) and about $0.02 to $0.04 for a shim session (Haiku shim plus a short Sonnet parent). Larger repos push the built-in Explore higher. Modal GPU cost for the two scout models is accounted by the serving side and is not part of this document's tally; the tally of Anthropic and Voyage spend is kept in `artifacts/eval/spend.json`.

## Caveats stated in advance

1. Public repos only, and all in the held-out split; the 10 repos were chosen in `eval/SELECTION.md` using an earlier benchmark, and that document's held-out check shows no selection edge, so this is a sample, not a set where scout was expected to win.
2. The parent is Sonnet 5.5 routed through an LLM API gateway, not the Fireworks route configured on the author's machine; absolute costs and the built-in Explore's model (the parent's model) depend on this routing. A deployment where the built-in Explore runs on a smaller model would shift the comparison.
3. The coordinator parent is constrained to relay the handback; this measures retrieval that reaches the parent, not what a free-running coding agent would do with it.
4. No coding-outcome metric; retrieval recall at file level is a proxy.
5. The scout models share one gateway code path and the same index, so the wayfinder-glimmer versus A1 contrast is the chat model only. Gateway cold starts of the model servers can inflate wall time of the first sessions; models are warmed before the run and wall time is reported with medians as well as means.
6. Sample size is 50 tasks in 10 clusters; CIs are wide and several secondary contrasts will not separate from zero.

## Results

Run 2026-10-27. The pre-registration above was committed as a66e4a2 before any full-50 run (a 2-repo, 2-task smoke ran before it was final; its rows are in `artifacts/eval/smoke/` and are not scored). Rows: `artifacts/eval/rows_<arm>.jsonl`; statistics: `artifacts/eval/results.json` and `REPORT.md` from `eval/harness/report.py` (10,000 resamples, seed 20261025).

**Primary result.** `super_v3` minus `explore_default` on handback file recall@12 is **-0.086 [-0.153, -0.021]** (task bootstrap; repo-cluster CI [-0.135, -0.037]). The interval excludes zero and the two CIs agree, so under the pre-registered rule built-in Explore (Sonnet 5.5, grep and reads) retrieved more of the gold files than super-explore on wayfinder-glimmer. The scout arms did not beat built-in Explore on retrieval in this run.

Tasks with a valid row in every arm: 50

| Arm | recall@12 | all-gold | wall s (mean) | wall median / p90 | cost USD | parent USD | subagent USD |
|---|---|---|---|---|---|---|---|
| explore_default | 0.490 [0.414, 0.564] | 0.10 [0.02, 0.18] | 33.1 [28.5, 39.3] | 29 / 46 | 0.0897 [0.0824, 0.0974] | 0.0485 [0.0454, 0.0519] | 0.0412 [0.0354, 0.0473] |
| super_v3 | 0.403 [0.325, 0.486] | 0.10 [0.02, 0.20] | 166.9 [161.9, 172.4] | 163 / 196 | 0.0408 [0.0382, 0.0436] | 0.0397 [0.0372, 0.0423] | 0.0011 [0.0010, 0.0013] |
| super_a1 | 0.352 [0.269, 0.437] | 0.06 [0.00, 0.14] | 124.1 [116.4, 131.9] | 122 / 161 | 0.0402 [0.0377, 0.0430] | 0.0392 [0.0367, 0.0418] | 0.0011 [0.0010, 0.0012] |

| Paired delta (a minus b) | metric | mean [95% task CI] | repo-cluster CI |
|---|---|---|---|
| super_v3_minus_explore_default | recall12 | -0.086 [-0.153, -0.021] | [-0.135, -0.037] |
| super_v3_minus_explore_default | all_gold | +0.000 [-0.060, +0.060] | [-0.060, +0.060] |
| super_v3_minus_explore_default | wall_s | +133.806 [+126.390, +140.870] | [+126.907, +140.912] |
| super_v3_minus_explore_default | cost_usd | -0.049 [-0.056, -0.042] | [-0.055, -0.042] |
| super_v3_minus_explore_default | final_recall12 | -0.084 [-0.151, -0.020] | [-0.131, -0.037] |
| super_a1_minus_explore_default | recall12 | -0.137 [-0.199, -0.079] | [-0.198, -0.075] |
| super_a1_minus_explore_default | all_gold | -0.040 [-0.100, +0.000] | [-0.100, +0.000] |
| super_a1_minus_explore_default | wall_s | +90.997 [+80.548, +100.117] | [+81.631, +100.521] |
| super_a1_minus_explore_default | cost_usd | -0.049 [-0.056, -0.043] | [-0.056, -0.042] |
| super_a1_minus_explore_default | final_recall12 | -0.127 [-0.185, -0.073] | [-0.180, -0.070] |
| super_v3_minus_super_a1 | recall12 | +0.051 [-0.015, +0.116] | [-0.028, +0.119] |
| super_v3_minus_super_a1 | all_gold | +0.040 [+0.000, +0.100] | [+0.000, +0.100] |
| super_v3_minus_super_a1 | wall_s | +42.809 [+34.130, +51.514] | [+30.415, +55.880] |
| super_v3_minus_super_a1 | cost_usd | +0.001 [-0.001, +0.002] | [-0.001, +0.002] |
| super_v3_minus_super_a1 | final_recall12 | +0.043 [-0.025, +0.112] | [-0.031, +0.106] |

Per-language recall@12 delta (mean, n tasks):

- super_v3_minus_explore_default: C# -0.204 (n=5), C++ -0.142 (n=10), Go -0.055 (n=10), Python -0.053 (n=25)
- super_a1_minus_explore_default: C# -0.204 (n=5), C++ +0.015 (n=10), Go -0.102 (n=10), Python -0.199 (n=25)
- super_v3_minus_super_a1: C# +0.000 (n=5), C++ -0.157 (n=10), Go +0.047 (n=10), Python +0.146 (n=25)

Fallthrough or invalid rows per arm (reported, not scored): explore_default 0 (fallthrough 0), super_v3 0 (fallthrough 0), super_a1 0 (fallthrough 0)

Fifty tasks, all three arms valid on every one; no fallthrough rows, no invalid rows (every scout row has `rewrite lane=production` in the hook log, subagent type `super-explore:super-explore`, one scout MCP call, shim resolved to claude-haiku-5-5; every built-in row resolved to claude-sonnet-5-5). Seven rows (4 `super_v3`, 3 `super_a1`) needed a second attempt for a stream or session problem before a valid row was written; only the valid attempt is scored.

## Reading

Recall. Built-in Explore averaged 0.490 [0.414, 0.564], super-explore on wayfinder-glimmer 0.403 [0.325, 0.486], on A1-35B 0.352 [0.269, 0.437]. wayfinder-glimmer minus A1 is +0.051 [-0.015, +0.116], which does not separate from zero on recall (cluster CI [-0.028, +0.119]); the all-gold difference of +0.04 has a CI starting at 0.00. Per language, wayfinder-glimmer trails built-in Explore in every stratum (C# -0.20 on 5 tasks, C++ -0.14 on 10, Go -0.06 on 10, Python -0.05 on 25); A1 trails by more in Python (-0.20) and ties in C++ (+0.02). The wayfinder-glimmer gain over A1 sits in Python (+0.15 on 25 tasks) while A1 is ahead in C++ (+0.16 on 10); these strata are small and are descriptive only. All-gold-found is 0.10 for built-in Explore and wayfinder-glimmer, 0.06 for A1; none of the all-gold deltas excludes zero.

Time. Wall time per session is 33 s for built-in Explore (median 29, p90 46), 167 s for super-explore on wayfinder-glimmer (median 163, p90 196) and 124 s on A1 (median 122, p90 161). The scout arms are 4 to 5 times slower on this setup. Caveat on the cause: the 10 repo lanes ran their three arms concurrently, so up to 20 scout explore loops hit the two model servers at once while the built-in arm only waits on Anthropic; wayfinder-glimmer wall time in particular includes queueing on a single H100 (the serving side reports it autoscales to more replicas under load). The numbers are what a user would get under this load, not the unloaded latency of either model.

Cost. A session costs $0.090 with built-in Explore, $0.041 with super-explore on wayfinder-glimmer and $0.040 on A1 (paired difference about -$0.049 [-$0.056, -$0.042]). The saving is the subagent: built-in Explore's Sonnet subagent costs $0.041 per session against $0.001 for the Haiku shim. The scout model's own serving cost (Modal GPU) is not in these figures. The parent cost is about $0.04 to $0.05 in every arm.

Retrieval content. Built-in Explore made 5.5 tool calls per task on average and listed 9.0 distinct repo files in its handback; the shim made one MCP call and the handback lists 9.6 (wayfinder-glimmer) and 7.8 (A1) files. The MCP output is capped at 1,200 rendered tokens by the plugin, so the handback is bounded by design.

## Deviations from the pre-registered setup

1. The parent runs with `--allowedTools Bash Read Grep Glob` so that the subagent's tools run without an interactive permission prompt in headless mode (the first smoke showed built-in Explore denied Bash). The gate hook still denies every non-`Agent` tool for the parent itself. Smoke rows before this change are not scored.
2. `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` is set in every arm; without it the parent returned before a backgrounded Explore finished.
3. The SessionStart watcher is stubbed (the harness syncs explicitly) so no watcher process outlives a session; the plugin's readiness files are real (status.json after `scout sync --wait`, a live `mcp.pid` written by `scout mcp`), and `SCOUT_LANE=production` forces the launcher onto the gateway lane.
4. Product fix found while indexing: `scout sync` failed on three repos (redex, terraform-google-conversion, velox) because boilerplate text shared by more than 500 files produced an embed request over the 500-entry `replace` cap. `src/scout/core/sync/engine.py` now splits a chunk key's rows by file count as well as row count (`_split_group`, unit-tested). The index used for all arms was built with the fix.
5. The two scout arms are served by two gateways (wayfinder-glimmer and A1) that share keys, index and chat code path; ten eval keys were issued under tenant `eval` and rotated across repo lanes to stay under the two-concurrent-chat cap per key.

## Spend and hygiene

Details in `artifacts/eval/spend.json`. All eval namespaces were deleted (`ops/ns_janitor.py --prefix scout_eval_ --delete`, then `--verify-empty` reported none). Only public repositories were indexed.

## Limits of this result

Fifty tasks in ten repos; the primary CI is wide enough that a true gap anywhere from about -0.02 to -0.15 is compatible with the data. The comparison is one parent model and one routing (the LLM gateway); a built-in Explore running on a smaller model would be cheaper and probably weaker. The task set and gold files are those of the file-level benchmark, so recall measures overlap with files changed by the gold commit, not whether the handback would have let a coder succeed. Results are not pooled across languages beyond the labelled split above.

## Addendum A: augment arm (pre-registered 2026-10-27, before any augment row is scored)

Everything above this heading, including the Results and Reading sections, is unchanged. The first three arms replaced Explore with a Haiku shim that relays one scout call, so scout's output was the ceiling. This addendum asks the other question: does scout help when it is one more tool in the built-in Explore's toolbox?

### Arm

`augment_v3`: the built-in Explore subagent exactly as in `explore_default` (same parent model and prompt, same gate hook, same `--allowedTools Bash Read Grep Glob`, same settings, same environment variables), plus scout's MCP server in the session: `scout mcp --tools explore,semantic_search` against the production gateway (`https://scout-gateway.example.com`, default chat model wayfinder-glimmer), the same tenant `eval`, the same namespaces `scout_eval_*` and a key rotated across repo lanes as before. The server is attached with `--mcp-config` plus `--strict-mcp-config` (server name `scout`, tools `mcp__scout__explore` and `mcp__scout__semantic_search`, both added to `--allowedTools`); the key reaches it through `${SCOUT_KEY}` expansion from the session environment and is never written to disk. The super-explore plugin is not loaded, so nothing rewrites Explore, no built-in tool is removed or blocked, and the subagent chooses freely among grep, glob, read, shell and the two scout tools. The subagent is not told that scout exists beyond the tool descriptions Claude Code gives it. Each session has its own isolated `CLAUDE_CONFIG_DIR`; the user's global Claude config is never read. The render ceiling is the server default (1,500 tokens; the plugin's 1,200 override is not applied), a difference from the replace arms that is accepted and stated.

Tasks, briefs, repos, commits, validation rules and the scorer are those of the main pre-registration. The 50 tasks run once; a session with a stream or session failure gets one retry, as before. Run conditions: the 10 repo lanes run concurrently (one augment session per lane at a time, so up to 10 simultaneous scout explore loops against wayfinder-glimmer, against up to 20 in the original run), with `wayfinder-v3-serve` warm and allowed to scale to 3 replicas. The `explore_default` and `super_v3` rows of the original run are reused as the comparison arms; they are not re-run.

### Row validity (additional)

An `augment_v3` row is valid only if the init event reports the `scout` MCP server as `connected` and the subagent type is the built-in `Explore`. A row where the subagent never called a scout tool is valid and scored, because "the subagent did not reach for scout" is part of the result; it is counted in the usage statistics. Rows are otherwise validated by the rules above. Paired comparisons use the tasks valid in all compared arms.

### Metrics

Primary, fixed now: **`augment_v3` minus `explore_default` on handback file recall@12**, paired by task, 95% percentile bootstrap over tasks (10,000 resamples, seed 20261025), with the repo-cluster bootstrap CI reported next to it. The same decision rule applies: the contrast counts as a result only if the task-bootstrap CI excludes zero, the cluster CI is stated beside it, and a disagreement is stated.

Secondary, fixed now, all descriptive:
- `augment_v3` minus `super_v3` on recall@12 (same statistics).
- Wall time per session (mean, median, p90) and the paired wall-time delta against `explore_default`; subagent duration.
- Cost: session cost, parent cost and subagent cost, as above, and the paired cost delta against `explore_default` and `super_v3`. Parent plus subagent Anthropic cost only; Modal GPU time is not in it.
- Scout-tool use: the share of tasks in which the subagent called at least one scout tool, mean calls per task by tool (`explore`, `semantic_search`), mean subagent tool calls, and recall@12 split by tasks where scout was and was not called (a within-arm description, not a causal comparison).
- All-gold-found rate and final-answer recall@12 with paired deltas, and the per-language recall delta against `explore_default`, as labelled strata.

### Caveats stated in advance

1. The comparison arms were run earlier the same day; model, gateway and load conditions may have drifted. There is no replicate of `explore_default`, so the run-to-run noise of one arm is not measured and a small delta cannot be separated from it beyond what the bootstrap shows.
2. Claude Code lists MCP tools to the subagent behind its tool-search mechanism in this version (a smoke probe showed `ToolSearch` before the first `mcp__scout__explore` call), which may lower how often scout is chosen. That is part of the setting under test, not a harness defect, and the usage statistic reports it.
3. `scout explore` runs wayfinder-glimmer as an inner agent loop; its call is slow (about 2 minutes in the probe) relative to a grep, so wall time rises whenever it is used. Load differs from the original run (10 instead of 20 concurrent loops), so wall time is not comparable to the `super_v3` figure without that qualification.
4. All caveats of the main pre-registration (public held-out repos, Sonnet 5.5 through an LLM API gateway, handback-only relay parent, file-level recall as a proxy, 50 tasks in 10 clusters) apply.

### Budget

Voyage: the eval namespaces were deleted after the first run and are re-synced; the tenant embedding cache persists, so re-embedding should be small.

### Addendum A results

Run 2026-10-27 after the addendum was committed (cc5b3f7) and the harness arm was added (125bc5e). Rows: `artifacts/eval/rows_augment_v3.jsonl`; statistics: `artifacts/eval/results.json` and `REPORT.md` (same code, same seed). The first 48 rows ran in one pass; a sync of two repos hit an HTTP 408 from the gateway and the run was resumed once for the last two rows. No row was retried for a validity problem and none was excluded: 50 of 50 rows are valid, all with the `scout` MCP server `connected` and the subagent type `Explore` (resolved to claude-sonnet-5-5).

**The subagent never called a scout tool.** In 50 of 50 sessions the built-in Explore made zero calls to `mcp__scout__explore` and `mcp__scout__semantic_search`; it used Grep (43 calls in total), Bash (32), Read (16) and Glob (3), 5.3 tool calls per task, against 5.5 in the replace-arm comparison. `augment_v3` is therefore, in effect, a second sample of the `explore_default` arm with an unused extra MCP server attached, and its result is not evidence about whether scout helps when used.

| Arm | recall@12 | all-gold | wall s (mean) | wall median / p90 | cost USD | parent USD | subagent USD |
|---|---|---|---|---|---|---|---|
| explore_default | 0.490 [0.414, 0.564] | 0.10 [0.02, 0.18] | 33.1 [28.5, 39.3] | 29 / 46 | 0.0897 [0.0824, 0.0974] | 0.0485 [0.0454, 0.0519] | 0.0412 [0.0354, 0.0473] |
| super_v3 | 0.403 [0.325, 0.486] | 0.10 [0.02, 0.20] | 166.9 [161.9, 172.4] | 163 / 196 | 0.0408 [0.0382, 0.0436] | 0.0397 [0.0372, 0.0423] | 0.0011 [0.0010, 0.0013] |
| augment_v3 | 0.485 [0.401, 0.566] | 0.14 [0.06, 0.24] | 35.4 [31.1, 40.1] | 31 / 57 | 0.0934 [0.0851, 0.1024] | 0.0526 [0.0488, 0.0568] | 0.0408 [0.0349, 0.0471] |

| Paired delta (a minus b) | metric | mean [95% task CI] | repo-cluster CI |
|---|---|---|---|
| augment_v3 minus explore_default (primary) | recall12 | -0.005 [-0.041, +0.030] | [-0.039, +0.031] |
| augment_v3 minus explore_default | all_gold | +0.040 [+0.000, +0.100] | [+0.000, +0.120] |
| augment_v3 minus explore_default | wall_s | +2.3 [-4.7, +7.8] | [-5.3, +8.3] |
| augment_v3 minus explore_default | cost_usd | +0.004 [-0.003, +0.010] | [-0.002, +0.010] |
| augment_v3 minus explore_default | final_recall12 | -0.005 [-0.041, +0.030] | [-0.039, +0.031] |
| augment_v3 minus super_v3 | recall12 | +0.081 [+0.012, +0.154] | [+0.030, +0.139] |
| augment_v3 minus super_v3 | wall_s | -131.6 [-138.8, -124.4] | [-139.3, -123.4] |
| augment_v3 minus super_v3 | cost_usd | +0.053 [+0.045, +0.061] | [+0.043, +0.063] |

Per-language recall@12 delta, augment_v3 minus explore_default: C# -0.033 (n=5), C++ +0.018 (n=10), Go -0.039 (n=10), Python +0.005 (n=25).

**Reading.** The primary contrast is -0.005 [-0.041, +0.030] (cluster CI [-0.039, +0.031]): it does not exclude zero and the two CIs agree, so under the pre-registered rule there is no difference between the built-in Explore with scout's tools available and the built-in Explore alone. That is the expected outcome when the tools are never used, and the 0.485 against 0.490 is a measure of the run-to-run noise of this setup (one rerun of the same arm on the same 50 tasks moved the mean by 0.005, though individual tasks moved more: the paired CI half-width is about 0.035). The contrast against `super_v3` (+0.081 [+0.012, +0.154]) restates the original result from the other side: the built-in Explore working alone beats the scout-only handback, and it does so in 35 s against 167 s. The all-gold difference (+0.04, CI starting at 0.00) is not a finding.

**Why scout was not chosen.** Two checks, both on 1 to 3 tasks, so indicative only. First, with the brief extended by "first call the mcp__scout__explore tool", the subagent did load the tool through `ToolSearch` and call it once (137 s subagent duration), so the server, the permission setting and the key all work. Second, with `ENABLE_TOOL_SEARCH=false` set (intended to load tool schemas eagerly; the stream shows no ToolSearch call, but I did not inspect the subagent's tool list directly) the subagent still made zero scout calls on three tasks and went straight to Grep and Bash (8 to 27 s). So the subagent, given the description `Explore` receives for these tools, prefers its fast built-in search and the choice is not an artefact of hidden tool schemas. A deployment that wants Explore to use scout has to instruct it (the agent definition or the tool description), which is a different, instructed arm and was not run or pre-registered here. This answers the open question as asked, "chosen freely": it does not, on these 50 tasks; it does not say whether scout would help a subagent that is told to call it.

**Deviations.** (1) `augment_v3` ran with up to 10 concurrent sessions (one per repo lane) against the original 20; since no scout chat call occurred this has no effect on the measurements. (2) `wayfinder-v3-serve` was kept warm at one container (deployed with a maximum of three); it never needed to scale. (3) The two-row resume after a gateway HTTP 408 on sync is noted above.

`wayfinder-v3-serve` was redeployed with `min_containers=0` and `max_containers=1` after the run (the previous deploy default). The `scout_eval_*` namespaces of the earlier run had been deleted and were re-synced for this run (a dry run before the sync found none); they were deleted again (`ops/ns_janitor.py --prefix scout_eval_ --delete`) and `--verify-empty` reported none. Only public repositories were indexed. Details in `artifacts/eval/spend.json`.

## Addendum B: GRPO arm a step-20 as the scout model (pre-registered 2026-10-28, before any `super_grpo_a20` row is run)

Everything above this heading, including the Results, Reading and Addendum A sections, is unchanged. This addendum tests one new candidate model in the setup of `super_v3`.

### Why this candidate

Wayfinder's GRPO arm a (precision coefficient 0) at step 20 is a candidate because on the 50-dev-task Wayfinder eval its reported recall is +0.03 above wayfinder-glimmer and +0.10 to +0.13 above A1-35B. The caveat is that it reports about 0.7 more files than wayfinder-glimmer, and at a fixed report length (the first 5 or 7 files) it does not beat wayfinder-glimmer. This eval's primary metric, handback recall@12, caps the report length at 12 files and the plugin caps the rendered output, so it is the product-side test of whether the gain survives a bounded handback or was a longer-report effect.

### Arm

`super_grpo_a20`: the replace setup of `super_v3`, unchanged in every respect except the chat model. The same parent (`claude -p`, claude-sonnet-5-5, the same fixed prompt, the same gate hook, the same `--allowedTools Bash Read Grep Glob`, the same settings and environment variables), the same plugin (`--plugin-dir integrations/claude-code`, hook rewrite of Explore to `super-explore:super-explore`, Haiku shim relaying one scout call, `SCOUT_LANE=production`), the same 50 tasks at their own `base_commit`, the same briefs, the same validation rules (including the fallthrough rule and one retry), the same tenant `eval`, the same namespaces `scout_eval_*` and the same rotation of ten eval keys across the ten repo lanes. The chat model is the merged checkpoint `/data/grpo/a/s20_merged` on Modal Volume `wayfinder-data`, served by the unmodified `serve/wayfinder_v3.py` configuration (same vLLM flags, `muse_glimmer` tool and reasoning parsers, bearer-key proxy and scale to zero) as app `wayfinder-a20-serve`, model name `wayfinder-grpo-a20`, and reached through a third gateway `scout-gateway-a20` (`SCOUT_GATEWAY_APP=scout-gateway-a20 SCOUT_LLM_PROFILE=a20`, Modal secret `scout-llm-a20`, `SCOUT_LLM_SAMPLING=sft` as for wayfinder-glimmer). The existing gateways, secrets and keys are not touched. The `explore_default`, `super_v3` and `super_a1` rows of the original run are reused as the comparison arms and are not re-run.

### Metrics

Primary, fixed now: **`super_grpo_a20` minus `super_v3` on handback file recall@12**, paired by task, 95% percentile bootstrap over tasks (10,000 resamples, seed 20261025), with the repo-cluster bootstrap CI next to it. The decision rule is that of the main pre-registration: the contrast counts as a result only if the task-bootstrap CI excludes zero, the cluster CI is stated beside it, and a disagreement is stated.

Secondary, fixed now:
- **`super_grpo_a20` minus `explore_default`** on handback recall@12 (same statistics).
- **Handback length and precision, so padding is visible.** Per arm and as paired deltas against `super_v3` and `explore_default`: the number of repo files named in the handback (`n_paths`, ranked list validated against `git ls-files` as for recall), the number of those inside the top 12, and precision `|gold ∩ first k| / |first k|` at k = 5, 7 and 12 (0 for an empty list). Recall@5 and recall@7 on the same ranked list are reported alongside, since the training-side caveat concerns the first 5 and 7 files. A recall gain at recall@12 that comes with a larger `n_paths` and no gain in recall@5, recall@7 or precision@12 is read as longer reports, not better retrieval, and stated as such.
- `super_grpo_a20` minus `super_a1`, all-gold-found rate, parent final-answer recall@12, wall time (mean, median, p90), session, parent and subagent cost, and the per-language recall delta as labelled strata, all descriptive as in the main pre-registration.

### Row validity

As in the main pre-registration for the scout arms: a row is scored only if it carries the brief hash, wall time of at least 2 s, a non-empty handback, a cost, a resolved subagent model and no zero-tools refusal, the plugin hook logged `rewrite lane=production` and the subagent type is `super-explore:super-explore` (or a scout MCP call is in the stream). A fallthrough row is retried once and, if it still falls through, reported and excluded from paired comparisons. Paired comparisons use the tasks valid in all compared arms.

### Caveats stated in advance

1. The comparison arms were run on 2026-10-27; model serving, gateway and load conditions may have drifted, and there is no replicate of `super_v3`, so the run-to-run noise of this arm is not measured. Addendum A measured one rerun of a Sonnet arm moving 0.005 on the mean (paired CI half-width about 0.035); the noise of a scout arm may differ.
2. Load differs from the original run: the new arm runs alone, one session per repo lane, so up to 10 simultaneous scout explore loops against up to 2 replicas of one server, against up to 20 loops in the original wayfinder-glimmer run. Wall time is not comparable to the `super_v3` figure without that qualification; recall is not expected to depend on it.
3. The scout models are sampled at temperature 0.85, top_p 0.95; each task runs once, so a task-level difference includes sampling variance.
4. The step-20 checkpoint was picked because it looked best on the Wayfinder dev eval. That is selection on a different set (Wayfinder dev tasks), not on these 50 held-out tasks, but the two sets are drawn from related distributions, so a positive result here is a single test of one selected candidate, not a ranking of checkpoints.
5. All caveats of the main pre-registration (public held-out repos, Sonnet 5.5 through an LLM API gateway, handback-only relay parent, file-level recall as a proxy, 50 tasks in 10 clusters) apply.

### Budget

Voyage: the `scout_eval_*` namespaces were deleted after Addendum A and are re-synced; the tenant embedding cache persists. The namespaces are deleted again afterwards and `--verify-empty` run.

### Addendum B results

Run 2026-10-28 after the addendum was committed (b6132e5) and the arm was added to the harness (6ab90f9). Rows: `artifacts/eval/rows_super_grpo_a20.jsonl`; statistics: `artifacts/eval/results.json` and `REPORT.md` (same code, same seed; the report gained handback length, precision and recall@5/7 for this addendum, computed from the stored `paths` and the task gold files, so the earlier arms now carry those columns too and their other figures are unchanged). The 50 rows were produced in one pass with no retry and no failure: 50 of 50 are valid, none a fallthrough (every row has `rewrite lane=production`, subagent type `super-explore:super-explore`, one scout MCP call, shim resolved to claude-haiku-5-5). No smoke run preceded it. The checkpoint serves with tool calls parsing (client check on 20 scout-shaped requests: 19 parsed cleanly, 20 made a tool call, 0 leaked markup, `wayfinder/artifacts/serve/grpo_a20_check.json`; cold start 254 s).

**Primary result.** `super_grpo_a20` minus `super_v3` on handback recall@12 is **-0.034 [-0.087, +0.016]** (task bootstrap; repo-cluster CI [-0.082, +0.014]). The interval includes zero and the two CIs agree, so under the pre-registered rule there is no difference, and the point estimate is negative. The step-20 checkpoint's reported-recall edge over wayfinder-glimmer on the Wayfinder dev eval did not reproduce as a gain in the product handback.

| Arm | recall@12 | all-gold | paths per handback | precision@5 | precision@7 | precision@12 | recall@5 | recall@7 | wall s median / p90 | cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| explore_default | 0.490 [0.414, 0.564] | 0.10 | 9.00 | 0.488 | 0.449 | 0.434 | 0.380 | 0.430 | 29 / 46 | 0.0897 |
| super_v3 | 0.403 [0.325, 0.486] | 0.10 | 9.56 | 0.368 | 0.342 | 0.299 | 0.303 | 0.364 | 163 / 196 | 0.0408 |
| super_a1 | 0.352 [0.269, 0.437] | 0.06 | 7.82 | 0.347 | 0.341 | 0.325 | 0.273 | 0.330 | 122 / 161 | 0.0402 |
| super_grpo_a20 | 0.370 [0.299, 0.441] | 0.02 [0.00, 0.06] | 10.22 [9.62, 10.78] | 0.360 | 0.300 | 0.268 [0.208, 0.333] | 0.275 | 0.309 | 152 / 186 | 0.0422 |

| Paired delta (a minus b) | metric | mean [95% task CI] | repo-cluster CI |
|---|---|---|---|
| super_grpo_a20_minus_super_v3 (primary) | recall12 | -0.034 [-0.087, +0.016] | [-0.082, +0.014] |
| super_grpo_a20_minus_super_v3 | n_paths | +0.660 [-0.040, +1.440] | [+0.020, +1.520] |
| super_grpo_a20_minus_super_v3 | precision12 | -0.031 [-0.078, +0.012] | [-0.067, +0.003] |
| super_grpo_a20_minus_super_v3 | recall5 | -0.028 [-0.089, +0.026] | [-0.089, +0.026] |
| super_grpo_a20_minus_super_v3 | recall7 | -0.055 [-0.121, +0.002] | [-0.126, +0.007] |
| super_grpo_a20_minus_super_v3 | all_gold | -0.080 [-0.160, -0.020] | [-0.160, +0.000] |
| super_grpo_a20_minus_super_v3 | wall_s | -9.8 [-16.8, -2.4] | [-17.9, -1.0] |
| super_grpo_a20_minus_super_v3 | cost_usd | +0.001 [-0.001, +0.004] | [-0.001, +0.005] |
| super_grpo_a20_minus_explore_default (secondary) | recall12 | -0.120 [-0.187, -0.058] | [-0.190, -0.048] |
| super_grpo_a20_minus_explore_default | n_paths | +1.220 [-0.240, +2.500] | [-0.300, +2.600] |
| super_grpo_a20_minus_explore_default | precision12 | -0.167 [-0.226, -0.107] | [-0.240, -0.096] |
| super_grpo_a20_minus_explore_default | wall_s | +124.0 [+115.6, +132.2] | [+115.6, +132.0] |
| super_grpo_a20_minus_explore_default | cost_usd | -0.048 [-0.054, -0.041] | [-0.054, -0.041] |

Per-language recall@12 delta, `super_grpo_a20` minus `super_v3`: C# -0.100 (n=5), C++ +0.063 (n=10), Go -0.022 (n=10), Python -0.064 (n=25). Against `explore_default`: C# -0.304, C++ -0.079, Go -0.077, Python -0.117.

**Reading.** The candidate reports the longest handbacks of any arm, 10.2 files against 9.6 for wayfinder-glimmer (+0.66 [-0.04, +1.44]; the cluster CI excludes zero, the task CI does not), and its precision at 12 is lower (0.268 against 0.299), so the extra files are mostly misses. At a fixed report length it is also not ahead: recall@5 is -0.028 and recall@7 -0.055 against wayfinder-glimmer, neither excluding zero. That is the pattern the caveat predicted, more files without better files, and here it also fails to convert into recall@12 at all. All-gold-found fell from 0.10 to 0.02 (-0.08 [-0.16, -0.02], cluster CI touching zero; descriptive, not a decision-rule result). Against built-in Explore the arm is -0.120 [-0.187, -0.058] (cluster CI [-0.190, -0.048]), a larger gap than wayfinder-glimmer's -0.086 and the largest after A1's -0.137; the cluster and task CIs agree. The one stratum where the arm leads wayfinder-glimmer is C++ (+0.063 on 10 tasks), too small to read. Wall time, 157 s mean and 152 s median, is 10 s below wayfinder-glimmer's, within what the lighter load (10 concurrent loops, one replica warm) explains (caveat 2); cost per session is $0.042, the same as wayfinder-glimmer within noise.

**Deviations.** (1) `wayfinder-a20-serve` was deployed with `max_containers=2` and `min_containers=1`; one container served the run. (2) Gateway `scout-gateway-a20` is served at `https://myworkspace--scout-gateway-a20-api.us-east.modal.run` (the `us-east` host form of the current Modal URL scheme); the earlier gateways' URLs are untouched. (3) The pre-registered budget line said no smoke run would be needed; none was run. (4) Report code: the handback length, precision and recall@5/7 columns were added to `eval/harness/report.py` after the addendum was committed and before any a20 row was scored; they are computed only from fields the addendum names.

`wayfinder-a20-serve` was redeployed with `min_containers=0` and `max_containers=1` after the run. The `scout_eval_*` namespaces (21, from the re-sync) were deleted and `ops/ns_janitor.py --prefix scout_eval_ --verify-empty` reported none. Only public repositories were indexed. Details in `artifacts/eval/spend.json`.

## Addendum C: hybrid scout-explorer, 5-task calibration (pre-registered 2026-10-28, before any `hybrid_v3` row is run)

Everything above this heading is unchanged. This addendum tests the instructed setup that Addendum A left open, in the form the plugin now ships (scout plugin 0.2.0): a Sonnet subagent that is told to call scout `explore` first and then verify.

### Arm

`hybrid_v3`: the parent, prompt, gate hook, `--allowedTools`, settings, environment, briefs, tenant, namespaces and key rotation of `super_v3`, unchanged, except the plugin routes Explore to `scout:scout-explorer` (`SCOUT_EXPLORE_MODE=hybrid`). That subagent runs on `model: sonnet` (claude-sonnet-5-5, the model of the built-in Explore here). Its tools are scout `explore`, `semantic_search`, `read_range` and `index_status` on the wayfinder-glimmer gateway (`scout-gateway`, chat model wayfinder-glimmer), plus Read, Grep and Glob. Its instructions are fixed in `integrations/claude-code/agents/scout-explorer.md` at the commit that adds this addendum:

1. Call `explore` once, first.
2. Read the top ranges (up to 8) to verify them.
3. Fill gaps with Grep, semantic_search and Read (about 8 extra calls).
4. Return at most 12 ranked `path:start-end` ranges, with no padding.

`SCOUT_NO_WARM=1` (the harness warms the model itself, not each session). `explore_default` is run again in the same pass on the same tasks, so the two arms share load and time of day. The stored `super_v3` and `explore_default` rows of the original run are reported beside them for the same tasks.

### Tasks

Calibration: 5 tasks, the harness's `--tasks 5` selection, which is the first task of each of the first 5 repos in `eval/eval_tasks.jsonl`. These are the tasks of Blaizzy/mlx-vlm, langchain-ai/deepagents, GoogleCloudPlatform/terraform-google-conversion, temporalio/temporal and lennix1337/Genexus18MCP, which are Python, Python, Go, Go and C#. Each runs at its own `base_commit`.

### Metrics

Reported per arm for the calibration, all descriptive (5 tasks cannot support a decision):

- handback file recall@12
- handback precision@12 and the number of paths in the handback
- recall@5 and recall@7
- wall time per session
- session cost in USD (`total_cost_usd`)
- for `hybrid_v3`: whether scout `explore` was the subagent's first tool, the number of explore calls, and the reads after it

Row validity is that of `super_v3`. A `hybrid_v3` row is scored only if the hook logged `rewrite lane=production target=scout:scout-explorer` and the subagent type was `scout:scout-explorer`. A fallthrough row is retried once.

**Gate to the full run (fixed now).** The full 50-task `hybrid_v3` arm runs only if, on the 5 tasks, all three of these hold:

- mean recall@12 of `hybrid_v3` is at least that of the fresh `explore_default` rows minus 0.05;
- scout `explore` was the first tool in at least 4 of 5 sessions;
- no row is invalid after its retry.

If the gate passes, the full run is a further addendum, written and committed before its first row. If it fails, the calibration is reported as is and the full run is not made.

### Caveats stated in advance

1. Five tasks in five repos. The CIs span most of the range, and one task can move a mean by 0.2. The calibration exists to catch a broken protocol or a large loss before spending on 50 tasks. It is not a result.
2. The hybrid subagent sees scout's answer and the repository. A recall gain over `super_v3` may come from Sonnet's own search rather than from scout. The tool sequence shows how much of the final list came after the explore call, but this design cannot separate the two.
3. Load is the harness default (up to 5 concurrent sessions per arm against one wayfinder-glimmer replica). Wall time includes the scout call's 1 to 3 minutes.
4. The caveats of the main pre-registration apply: public held-out repos, Sonnet 5.5 through an LLM API gateway, a relay-only parent, and file-level recall as a proxy.

### Budget

Anthropic: 10 sessions, at about $0.10 for `explore_default` and an estimated $0.15 to $0.30 for `hybrid_v3` (a Sonnet subagent with a scout call and about 10 reads). That is under $3 with retries, against $25 for this addendum and any full run; sessions are capped at $1.

Modal: `wayfinder-v3-serve` is kept warm (`min_containers=1`) only during the run and scaled to zero afterwards.

The `scout_eval_*` namespaces are deleted afterwards and `--verify-empty` is run. Only public repositories are indexed.

### Addendum C deviation, recorded before the first row (2026-10-28, 18:55 PT)

The wayfinder-glimmer upstream is down. `wayfinder-v3-serve` fails at vLLM engine init on every container start (`RuntimeError: vllm exited with code 1`). This follows an uncommitted cold-start rework of `serve/wayfinder_v3.py` that another session is deploying (GPU memory snapshot on by default, read-only cache mount). Every chat through `scout-gateway` therefore fails after about 150 s, and the gateway reports the failure as `INVALID_ARGUMENT: the chat request was rejected by the model`. This was observed on `scout warm`, on a direct `scout explore`, and in the plugin smoke. This eval does not redeploy another session's serving app.

The calibration therefore runs the same arm on the A1 gateway, as **`hybrid_a1`**. Only the gateway changes: `scout-gateway-a1`, chat model Agents-A1-35B, the same keys, the same index and the same agent file. A1 answered a warm-up in 42 s. The comparison arms become:

- the fresh `explore_default` rows;
- the stored `super_a1` rows for the same 5 tasks, which are the relay on the same model and so the like-for-like test of whether verification helps;
- the stored `super_v3` rows, for reference.

The gate to a full run, its thresholds and every other rule above are unchanged. The gate applies to `hybrid_a1`. A full run, if the gate passes, uses whichever model is serving when its addendum is written, and that addendum says which. `hybrid_v3` stays defined in the harness for when wayfinder-glimmer serves again.

### Addendum C results (calibration, 2026-10-28)

The run used the A1 gateway, per the deviation above. Rows: `artifacts/eval/hybrid/rows_{explore_default,hybrid_a1}.jsonl`. Table: `artifacts/eval/hybrid/calibration.json`, from `eval/harness/calibration_c.py`, using the same validation, path extraction and length statistics as `report.py`.

All 10 sessions are valid. No row is a fallthrough. Every `hybrid_a1` row carries `rewrite lane=production target=scout:scout-explorer`, subagent type `scout:scout-explorer`, resolved model claude-sonnet-5-5, and exactly one scout `explore` call. One lane (temporalio/temporal) failed its first sync with a gateway HTTP 408, as in Addendum A. It was resumed once and both its rows were produced in the resume.

| Arm (same 5 tasks) | recall@12 | precision@12 | paths | recall@5 | wall s median | cost USD |
|---|---|---|---|---|---|---|
| explore_default, fresh | 0.436 | 0.422 | 10.4 | 0.248 | 20 | 0.104 |
| **hybrid_a1** | **0.167** | **0.286** | **6.2** | 0.167 | 145 | 0.086 |
| explore_default, stored (10-27) | 0.394 | 0.407 | 9.8 | 0.304 | 38 | 0.089 |
| super_v3, stored | 0.252 | 0.209 | 9.8 | 0.232 | 165 | 0.038 |
| super_a1, stored (the relay on the same model) | 0.085 | 0.104 | 4.0 | 0.065 | 147 | 0.035 |

Per-task recall@12 for `hybrid_a1` against fresh `explore_default`:

- mlx-vlm: 0.50 against 0.50
- terraform-google-conversion: 0.00 against 0.33
- deepagents: 0.00 against 0.50
- temporal: 0.00 against 0.40
- Genexus18MCP: 0.33 against 0.44

**Gate: not passed. No full run.**

- Recall@12 0.167 against 0.436 fails the "within 0.05" rule.
- Scout `explore` was the first tool in 5 of 5 sessions, which passes.
- No row was invalid, which passes.

**Reading.** These are descriptive numbers on 5 tasks. The hybrid did what the protocol fixes first: it called scout once, as its first tool, every time. It then mostly did not verify by reading. Reads after explore averaged 0.6 per session, against the protocol's "up to 8". The subagent checked scout's names with Grep (5.6 tool calls per session in all) and returned short lists (6.2 paths, against 10.4 for Explore). Against the relay on the same model (`super_a1`), verification doubled recall (0.167 against 0.085) and nearly tripled precision (0.286 against 0.104).

Against built-in Explore, it lost on both, in about 7 times the wall time. The A1 explore call alone took 108 to 149 s. On these feature-request briefs, whose gold sets include new and peripheral files, the "do not pad" rule and a verifier that drops what it cannot confirm work against recall@12.

Two changes follow from this. Neither is tested here:

1. Wording that makes Read the verification step. It was added to the agent file after this run (the git history shows it).
2. Allowing the subagent to keep scout's unverified-but-plausible files below the verified ones, up to the cap.

Whether wayfinder-glimmer behind the hybrid closes the gap is untested, because wayfinder-glimmer was down.

**Minor deviation.** During the run, a working-tree edit to the agent file existed for about 20 seconds (18:47:31 to 18:47:51 PT). It added "a Grep hit is not verification: read the lines" and "drop what you could not Read". Three `hybrid_a1` sessions started within the same minute (18:47:01 to 18:47:14). The edit was reverted before the run continued. If any of them read it, it would have pushed toward more reads. The observed read rate is low in all five sessions, including the one that ran 15 minutes later, so the edit did not produce the result.

**Spend and hygiene.**

- Namespaces: the five branch namespaces this run synced were deleted through the gateway (`delete_branch` with each checkout's own `repo_id` and `branch_id`). A tenant-wide `ns_janitor --prefix scout_eval_ --verify-empty` was not run: another session's Muse calibration uses the same tenant and prefix concurrently.
- The eval used its own checkout cache (`SCOUT_EVAL_CACHE=.cache/eval_repos_hybrid`), so that concurrent `git clean` calls could not collide. Only public repositories were indexed.

## Addendum D: hybrid scout-explorer on wayfinder-glimmer with a 12-file handback (pre-registered 2026-10-28, before any row of this addendum)

Everything above is unchanged. wayfinder-glimmer serving is fixed: commit 5279c65 in the wayfinder repo. `scripts/v3_check.py` against the endpoint gave 20/20 tool calls parsed, 20/20 on turn 2, and reasoning present.

Addendum C and the Muse calibration failed the same way: handbacks of 5 to 7 files lose recall@12 against Explore's lists of about 10 files. This addendum therefore tests a revised subagent on wayfinder-glimmer.

### Arm

`hybrid_v3`: the arm defined in Addendum C, on the wayfinder-glimmer gateway, with the agent file `integrations/claude-code/agents/scout-explorer.md` as committed with this addendum. The changes from Addendum C:

1. The handback has up to 12 ranked files. Verified ranges come first; scout's remaining hits follow, marked `(unverified)`, instead of being dropped. Only ranges that were read and found irrelevant are dropped.
2. Read verification of the top ~5 ranges, instead of "up to 8".
3. A budget of about 8 tool calls in total, `explore` included.

Everything else is unchanged: parent, prompt, gate hook, settings, briefs, tenant, the `SCOUT_EVAL_CACHE` checkout cache, `SCOUT_NO_WARM=1` in sessions, validity rules and retries. Before any row counts, wayfinder-glimmer is warmed: the gateway's `/v1/warm` trigger (`scout warm --trigger`), then one blocking `scout warm` that must return `warm: true`.

### Calibration and gate

The 5 tasks of Addendum C. `explore_default` runs fresh in the same pass. The metrics are those of Addendum C, plus the share of unverified entries in each handback. The stored `super_v3` rows and the Addendum C rows are reported beside them.

**Gate to the full run (fixed now):** mean recall@12 of `hybrid_v3` on the 5 tasks is at least that of the fresh `explore_default` rows ("at least even"), and no row is invalid after its retry. If the gate fails, the calibration is reported and nothing more runs.

### Full run, if the gate passes

The full run covers all 50 tasks of `eval/eval_tasks.jsonl` with arms `hybrid_v3` and a fresh `explore_default`, in one pass. The stored `super_v3` rows are reported beside them. The 5 calibration tasks are run again inside the 50, and the calibration rows are not reused.

Primary, fixed now: **`hybrid_v3` minus `explore_default` on handback recall@12**, paired by task, 95% percentile task bootstrap (10,000 resamples, seed 20261025), with the repo-cluster CI beside it. The decision rule is that of the main pre-registration.

Secondary, descriptive:

- `hybrid_v3` minus `super_v3`;
- handback length (`n_paths`), precision@5/7/12, recall@5/7;
- all-gold, wall time and session cost;
- explore-first share, reads after explore, unverified share;
- per-language deltas.

### Caveats stated in advance

1. The agent wording was revised after a failed calibration on the same 5 tasks. Those tasks are therefore not held out for this revision. The 50-task run is the test; the 5 tasks are a gate.
2. Listing scout's unverified hits trades precision for recall by design. Precision@12 is reported next to recall so that the trade is visible.
3. All earlier caveats apply.

### Budget

- Anthropic: calibration under $2. A full run is 100 sessions at about $0.10 each, so $10 to $15 with retries, against $25. Sessions are capped at $1.
- Namespaces are left for the author's janitor run.

### Addendum D results (calibration, 2026-10-28)

The run used the agent file of f6f1d7a. Before any row, wayfinder-glimmer was warm: `scout warm --trigger`, then a blocking `scout warm` that returned `warm: true` in 0.7 s, because the coordinator's `v3_check.py` had already started a container. Rows: `artifacts/eval/hybrid_v3/rows_{explore_default,hybrid_v3}.jsonl`. Table: `artifacts/eval/hybrid_v3/calibration.json`, from `eval/harness/calibration_c.py hybrid_v3 hybrid_v3`. All 10 rows are valid, with no retry and no fallthrough. Every `hybrid_v3` row has `target=scout:scout-explorer` and exactly one scout `explore` call.

| Arm (same 5 tasks) | recall@12 | precision@12 | files | recall@5 | wall s median | cost USD |
|---|---|---|---|---|---|---|
| explore_default, fresh | 0.349 | 0.343 | 10.4 | 0.257 | 21 | 0.089 |
| **hybrid_v3** | **0.309** | **0.408** | **7.4** | **0.284** | 157 | 0.081 |
| explore_default, stored (10-27) | 0.394 | 0.407 | 9.8 | 0.304 | 38 | 0.089 |
| super_v3, stored | 0.252 | 0.209 | 9.8 | 0.232 | 165 | 0.038 |
| hybrid_a1 (Addendum C) | 0.167 | 0.286 | 6.2 | 0.167 | 145 | 0.086 |

Per-task recall@12, `hybrid_v3` against fresh `explore_default`:

- mlx-vlm: 0.62 against 0.50
- Genexus18MCP: 0.44 against 0.44
- deepagents: 0.38 against 0.50
- temporal: 0.10 against 0.30
- terraform-google-conversion: 0.00 against 0.00

**Gate: not passed (0.309 against 0.349). No 50-task run.**

**Reading.** These are descriptive numbers on 5 tasks. On wayfinder-glimmer and with the revised wording, the hybrid comes close to built-in Explore on recall@12: a gap of -0.04, against -0.27 for `hybrid_a1`. It is ahead on precision@12 (0.408 against 0.343) and recall@5 (0.284 against 0.257). It is also above the wayfinder-glimmer relay `super_v3` on every column. It loses recall@12 through list length: 7.4 files against 10.4. The instruction to keep scout's unread hits below the verified ones did not produce 12-file lists, partly because scout's own answer usually names fewer than 12 files.

Reads after explore averaged 0.4 per session (3.2 tool calls in all), so the subagent still mostly checks with Grep rather than Read. Each session took about 157 s, most of it in the wayfinder-glimmer explore call. The fresh `explore_default` mean (0.349) is 0.045 below its stored value on the same tasks (0.394), which is noise of the size Addendum A measured. The gate was fixed against the fresh rows.

**Spend.**

- Anthropic: $0.85 for the 10 sessions.
- Modal: $1 recorded (session `scout-hybrid-v3`). `wayfinder-v3-serve` was already warm, was not redeployed, and scales to zero after its 10-minute idle window. A watcher logs its container count to `artifacts/eval/hybrid_v3/v3_watch.log`.
- Namespaces are left for the author's janitor run.


## Addendum E: deterministic backfill to 12 files in the explore use case (pre-registered 2026-10-28, before any row of this addendum)

Everything above is unchanged. wayfinder-glimmer serving is fixed (commit 5279c65 in the wayfinder repo).

Addendum D showed that `hybrid_v3` beats built-in Explore on precision@12 and recall@5 and loses recall@12 only because its lists are short (7.4 files against 10.4). Asking the model to add files did not work: scout's own answer names 6 to 7 files, and the subagent rarely adds more. This addendum moves the padding into code.

### Change under test (commit 5b5c920)

`explore.backfill_to` (default 12; 0 is off; env `SCOUT_EXPLORE_BACKFILL_TO`, config table `[explore]`) in `scout.core.explore.backfill`. After the model's `report`, the use case appends files not already listed, one entry per file, in this order: (a) files the model opened with `read_file` but did not report, first-seen; (b) the top `semantic_search` hits for the brief, by score; (c) files containing identifiers extracted from the brief (grep), by identifiers matched, then matches, then path. The model's own ranking stays first and untouched. Appended entries have `verified: false` and render under `Further candidates (unverified):`. It is pure post-processing (index and grep lookups, no model turn), runs inside the scout process, and needs no gateway change. Both scout-explorer prompts pass the section through in order.

### Arms

Same 5 tasks, harness, parent, gate hook, settings, briefs, `SCOUT_EVAL_CACHE` checkouts, `SCOUT_NO_WARM=1`, validity rules and retries as Addendum D. Both arms set `SCOUT_EXPLORE_BACKFILL_TO=12` and run against the wayfinder-glimmer gateway.

- `super_v3_bf`: the pure relay (Haiku `super-explore`, one `explore` call returned verbatim) plus backfill.
- `hybrid_v3_bf`: the Sonnet `scout-explorer` of Addendum D, with the revised prompt, plus backfill.

A fresh `explore_default` runs in the same pass. Stored `explore_default`, `super_v3` and `hybrid_v3` (Addendum D) rows are reported beside them. Before any row, wayfinder-glimmer is warmed with `scout warm --trigger`, then a blocking `scout warm` that must return `warm: true`.

### Gate and full run

**Gate (fixed now):** an arm passes if its mean recall@12 on the 5 tasks is at least that of the fresh `explore_default` rows and none of its rows is invalid after retry. If neither arm passes, nothing more runs. If both pass, the full run is for the one with the higher recall@12 (ties: the cheaper).

Full run, if an arm passes, pre-registered here: all 50 tasks of `eval/eval_tasks.jsonl`, the passing arm and a fresh `explore_default` in one pass (the stored `explore_default` rows are not reused: different day, and the 5 calibration tasks are rerun inside the 50). Primary: **arm minus `explore_default` on handback recall@12**, paired by task, 95% percentile task bootstrap (10,000 resamples, seed 20261025), with the repo-cluster CI beside it; decision rule as in the main pre-registration. Secondary, descriptive: precision@5/7/12, recall@5/7, handback length, all-gold, unverified share, wall time, cost per session, the arm minus the stored `super_v3`.

### Caveats stated in advance

1. The backfill pads the list with entries the model did not judge. It trades precision for recall by design; precision@12 is reported next to recall@12.
2. Backfill (b) queries the same index the explorer used, so it can repeat what the model already saw. It cannot repeat a file already listed.
3. The Claude Code parent may trim a handback in its own final answer (seen on deepagents in the Muse run: handback 0.38, final 0.25). The harness scores the handback; `final_recall12` is stored in every row.
4. The 5 calibration tasks were used by Addenda C and D; they are a gate, not held out.
5. The production gateway is unchanged by this work; the backfill runs in the local scout MCP process (`explore` through the Claude Code plugin). The lane tested is production-lane scout on the wayfinder-glimmer gateway.

### Budget

`wayfinder-v3-serve` is scaled to zero afterwards unless the Muse calibration is using it. Namespaces are left for the author.

### Addendum E results (calibration, 2026-10-28)

wayfinder-glimmer was warm before any row (`scout warm --trigger`, then a blocking `scout warm` that returned `warm: true`). Rows: `artifacts/eval/backfill/rows_{explore_default,super_v3_bf,hybrid_v3_bf}.jsonl`; table: `artifacts/eval/backfill/calibration.json` (`eval/harness/calibration_e.py`). All 15 rows are valid, with no fallthrough.

| Arm (same 5 tasks) | recall@12 | precision@12 | files | recall@5 | wall s median | cost USD |
|---|---|---|---|---|---|---|
| explore_default, fresh | 0.372 | 0.308 | 12.0 | 0.282 | 21 | 0.092 |
| super_v3_bf | 0.235 | 0.200 | 9.2 | 0.190 | 143 | 0.038 |
| hybrid_v3_bf | 0.311 | 0.239 | 11.8 | 0.217 | 183 | 0.078 |
| hybrid_v3 (Addendum D) | 0.309 | 0.408 | 7.4 | 0.284 | 157 | 0.081 |
| super_v3 (stored) | 0.252 | 0.209 | 9.8 | 0.232 | 165 | 0.038 |

**Gate: not passed by either arm** (0.235 and 0.311 against 0.372). No 50-task run.

**Reading.** The backfill did what it was built for on length: `hybrid_v3_bf` hands back 11.8 files (7.4 before), 77% of them unverified. Recall@12 did not move (0.311 against 0.309), and precision@12 fell from 0.408 to 0.239. The extra files were mostly not gold: the fill is less relevant than the model's own picks by construction. `super_v3_bf` reached 12 files on three of five tasks and only 4 and 6 on Genexus and deepagents; I did not trace why the backfill stopped short there (candidates: the relay's render token cap, few identifiers in the brief, hits dropped by the disk check). Task-level noise is large: `hybrid_v3_bf` on temporal scored 0.00 against 0.10 before, and fresh `explore_default` moved from 0.349 to 0.372 on the same tasks.

Takeaways: list length alone does not buy recall@12; the filler needs a better source than brief-derived heuristics (for example a second model-ranked pass or a path-neighbourhood tier). The parent may also trim a handback in its final answer (`final_recall12` equals handback recall here, so it did not on these rows).

**Spend.** Anthropic about $1.1 for the 15 sessions. `scripts/v3_warm.sh off` was run after the rows. The gateway redeploy (`modal deploy deploy/modal/gateway.py`) was denied by the permission check and was not needed for the test (the backfill runs in the client process); the author will deploy it. Namespaces `scout_eval_*` are left for the author.
