# Run B: wayfinder-glimmer against Agents-A1-35B on languages out of domain for wayfinder-glimmer, on repos A1 cannot have trained on (pre-registration)

Written 2026-10-28 (author approved Run B of `docs/a1_domain_audit.md` section 4 on 2026-10-28), after the task set and indexes were built and before any rollout. The protocol is a copy of `docs/test_split_eval.md`; only the task set, the language groups and the app names change. Nothing in this file is changed after the first rollout. The result is scored once, reported as is, and never used to choose between models, recipes or checkpoints. Results are never pooled with the dev, test or E2E sets.

## Question

The test split showed wayfinder-glimmer ahead of A1-35B on repos neither model trained on. wayfinder-glimmer's SFT data has only 6 Java/Kotlin, 8 C# and 9 Swift tasks (of 357), so those three languages are thin for wayfinder-glimmer. A1's training data is undisclosed, so for A1 the only fact that can be asserted is a date: A1 was released 2026-06-26. Run B asks whether wayfinder-glimmer's lead over A1 survives in the three languages wayfinder-glimmer saw least, on code written after A1's release.

## Tasks

120 retrieval tasks, `data/ood_langs_tasks.json` (the frozen eval copies are `artifacts/ood_langs/tasks_ood.json` and four repo-disjoint shards `tasks_shard{0..3}.json`, 30 tasks each, built from it). 40 C#, 40 Java/Kotlin, 40 Swift, over 111 distinct repos (C# 36, Java/Kotlin 40, Swift 38 repos), at most 2 tasks per repo (cap was 6; round-robin over repos before any repo gets a second task). Gold is the implementation (non-test) source files the commit changed, 2 to 10 per task, in the same sense as every task in the existing pool; tests are excluded from gold (`gold_files_all` in the data keeps them). 91% of tasks have every gold file in the task's own language, 97% of gold files on average; the rest are build scripts or web assets changed in the same commit.

Eligibility, applied before any model saw a task:

- **Repo:** public GitHub repo, `created_at` after 2026-06-26 (the earliest in the set was created 2026-06-27), language C#, Java, Kotlin or Swift by the GitHub language field, not a fork, not archived, at least 5 stars, pushed after 2026-09-15, 100 KB to 800 MB. Not in `data/splits.json` (316 repos, case-insensitive), not in the E2E v4 manifest, the RL-pool or SFT files or any other Wayfinder data file (a name grep over `artifacts/`, `data/`, `eval/e2e` and `docs/`), and no commit in its history authored before 2026-06-26. The grep found one candidate (Zesty0wl/mac-performance-monitor, which sits in a repo-provenance candidate list but in no split, task or training file); to keep the rule literal it was dropped and its single task replaced from another repo of the same language (recorded in `artifacts/ood_langs/funnel.json`, `sweep_replacements`). This happened before any index build for the replacement, before any run.
- **Commit:** authored on or after 2026-06-27, non-merge, at least 2 non-test source files changed (at most 10), at least 2 of them in the target language.
- **Pipeline:** the same miner, classifiers and leak guards as the existing pools: `scripts/ood_langs/mine_ood.py` is a copy (not an import) of `benchmark-repo/scripts/mine_expanded_tasks.py` via the E2E v4 copy `scripts/e2e_v4/mine_v4.py`, using the vendored byte-identical audited modules `asa.v3.categories.build_candidate` (the category classifiers), `asa.v3.leakcheck.measure_bridge` (commit scope, template slot, whole path and basename guards) and a copy of `verify_gold_at_base`. Each source commit yields one task (first accepting category). The only additions are a stricter test-path filter (directories such as `Foo.Tests`, `androidTest`, `FooTests`, on top of the miner's own `is_test_path`) and the gold-exists-at-base rule below. Discovery is `scripts/ood_langs/discover.py`, selection `scripts/ood_langs/select_ood.py` (seed 20261028).
- **Gold exists at base:** every gold file exists in the tree at the base commit (the pool's `verifiable_src` rule). A task whose gold includes a file its own commit creates is dropped, because it can never reach all-gold. This removed 1,508 of 2,886 miner-accepted tasks.
- **Question text:** the commit subject with the conventional-commit prefix and trailing PR reference stripped, 4 or more words, first letter upper-cased: exactly how the existing commit-mined pool generates questions. Subjects that are less than 70% ASCII (Chinese or Japanese commit messages; 3 of 1,378) were dropped because the pool's questions are English.
- **Funnel** (`artifacts/ood_langs/funnel.json`): 5,968 repos discovered, 4,582 eligible, 259 mined in random order (stopped when supply was ample), 238 mined cleanly (15 had pre-release history, 6 failed to clone); 47,988 commits, 37,297 on or after the cutoff, 16,338 with at least 2 non-test source files, 13,374 within the size cap, 12,456 with at least 2 files in the target language, 2,886 accepted by a category classifier and the leak guards (156 repos), 1,378 with all gold at base, 1,373 after the question-language and 2 to 10 gold rules (C# 292 tasks / 36 repos, Java/Kotlin 573 / 57, Swift 508 / 39), 120 selected. The categories of the 120 follow what the classifiers accept: implementing_new_features 93, cybersecurity_bug_finding 8, modernize 6, paying_tech_debt 6, refactoring 5, splitting_monolith 2.
- **Reachability and privacy:** `gh api repos/<repo>/commits/<base_commit>` returned the sha and `private` was false for all 120 repo@commits (`artifacts/ood_langs/exclusions.json`, empty). The exclusion rule for anything that fails at run time is the test split's: a task is excluded from every arm if a checkout or harness infrastructure error remains in any arm after one identical re-run of the missing rollouts; the list is written to `exclusions.json` before any metric is computed and reported. Model-level failures are never exclusions.
- **Licences:** the repos' licences are mixed (MIT 48, GPL-3.0 27, Apache-2.0 11, AGPL-3.0 8, none or NOASSERTION 23, others). Publish only aggregate numbers, as for every other set. Public repos only; no private code is read.

Caveats on the set, stated now: the repos are young (created within 4 months), mostly small hobby or one-developer projects, many written with AI assistance, and 58 of 120 commits are from October 2026. That is the price of a hard guarantee that A1 cannot have seen them, and it means absolute recall levels are not comparable to the test split; only contrasts are. Commit subjects in this population are often multi-clause ("Rename the Dock tab, add Fix TizenTube and an on-page preview"), which makes questions broader than in the older pool. The 40 per language is the target the author gave; per-language contrasts are descriptive, not decision-making.

## Indexes

All 120 repo@commits (111 repos) are `ready` with coverage 1.0 in the Modal Dict `wayfinder-index-registry`, built by `wayfinder/rollout/index_app.py::launch_units` (scout's secure-indexing path, public repos only; Voyage tokens at list about $15, `artifacts/ood_langs/index_registry.json`). None is `too_big`, so the semantic arm offers the index on every task.

## Arms

As in `docs/test_split_eval.md`, unchanged: `wayfinder-v3` (`/data/sft/v3_merged`, served as `wayfinder-v3`, T 0.85, top_p 0.95, `muse_glimmer` parsers, `--max-num-seqs 64`), Agents-A1-35B (`/weights/Agents-A1-FP8`, served as `a1`, `wayfinder/rollout/a1_eval_app.py` unchanged, A1's own sampling T 0.85, top_p 0.95, top_k 20, presence_penalty 1.1, thinking on), and the untrained Muse-Glimmer-30B (`/weights/Muse-Glimmer-30B`, served as `glimmer`, `WF_EVAL_VARIANT=base`). Each runs `grep` (grep, glob, read_file, report) and `sem` (plus `semantic_search`, `semantic=auto`), two seeds (0 and 1), one rollout per (task, seed, arm).

Launchers: `wayfinder/rollout/test_eval_app.py` for wayfinder-glimmer and the base, `wayfinder/rollout/a1_eval_app.py` for A1, both unmodified; `scripts/ood_langs/launch.sh` starts 4 repo-disjoint shards per model, 12 H100 apps in parallel, both arms sequentially on one server per app, concurrency 24 (wayfinder-glimmer, base) and 25 (A1). App names `wayfinder-oodeval-<model>-s<i>`. Startup checks must pass on every app (served name, tool call parses, wayfinder-glimmer logprobs differ from base, base logprobs match the recorded base), or the app is relaunched.

## Harness (identical across models, identical to the test-split eval)

Scout product prompt with the convergence nudge (`run.TEACHER_NUDGE`), the tools above, `max_turns` 40, `max_seconds` 3600 (scout's Budget cap lifted at runtime), `max_model_len` 65536 (32768 only if 65536 fails to start), at most 12 reported files, one H100 per app, `retain_reasoning` on, scorer `wayfinder/rollout/score.py::score_row`. No prompt, tool or budget change.

## Metrics

Unit of analysis: the task, seeds averaged within task. Paired contrasts use tasks with at least one valid row in both arms. All CIs are 95% percentile bootstrap, 10,000 resamples, seed 11: (a) task bootstrap; (b) repo-cluster bootstrap, resampling the 111 repos with replacement and pooling their tasks' paired deltas (with at most 2 tasks per repo the two intervals are expected to be close). Scorer: `scripts/score_ood_langs.py`, a copy of `scripts/score_test_split.py` with the task set, directory and language groups changed; metric code is unchanged.

**Primary:** reported recall (|reported gold| / |gold|, at most 12 files, deduped), wayfinder-glimmer minus A1, grep arm, with the task-bootstrap CI and the repo-cluster CI.

**Secondary** (each with both CIs wherever a paired contrast is shown): recall@5 and recall@7 at fixed report length (wayfinder-glimmer minus A1 and wayfinder-glimmer minus base, both arms); precision; wayfinder-glimmer minus A1 on the semantic arm; wayfinder-glimmer minus base on both arms (A1 minus base descriptive); all-gold in the report, turns, tokens, ended-by-report, number of reported files, touched-file recall; per-language results (C#, Java/Kotlin, Swift; 40 tasks each) with task CIs, win/loss/tie by task and by repo, and the semantic-arm table; the semantic index is offered on all tasks. Tool-call health is counted as in the test split.

## Decision rule

**wayfinder-glimmer beats A1 out of domain** if the primary contrast's task-bootstrap lower bound is above 0 and the repo-cluster lower bound is also above 0. If either bound is at or below 0, the claim is reported as not established, whatever the point estimate; if the point estimate is at or below 0, it is reported plainly as wayfinder-glimmer not beating A1 on these languages (and the sign by language is shown). No second look, no model selection, no rerun of a model or seed to improve a number. The only permitted reruns are identical re-launches of rollouts that died of infrastructure causes (checkout, harness exception, function deadline), done before scoring and counted in the results. The rule is about the pooled 120 tasks; per-language numbers do not enter the decision.

## Validation (before scoring)

Each model by arm must have: the right model name and weights on every row, `retain_reasoning` true, `max_turns` 40, `max_seconds` 3600, seeds {0, 1}, every (task, seed) of the non-excluded tasks exactly once, the sampling profile above. Errors and tool-parse failures are counted from the rows and reported per arm: rows with `error`, malformed tool calls (invalid JSON arguments, unknown tool, invalid arguments), turns with no tool call, rollouts stopped by three consecutive malformed calls, leaked tool markup in content. A cell that fails validation is reported as failed, not silently scored.

## Budget, watcher, guards

Watcher `ops/ood_eval_watch.py` (prefix `wayfinder-oodeval`, copy of `ops/test_eval_watch.py` with an auth canary and a 150-minute age limit), armed before the first launch; it stops only apps whose name starts with the prefix. `wayfinder-grpo-*` and `asa-*` are never touched.

## Files

`data/ood_langs_tasks.json`, `scripts/ood_langs/{discover,mine_ood,select_ood}.py`, `scripts/ood_langs/launch.sh`, `scripts/score_ood_langs.py`, `ops/ood_eval_watch.py`, and under `artifacts/ood_langs/`: `funnel.json`, `discovered.json`, `mined_raw.jsonl`, `exclusions.json`, `index_registry.json`, `tasks_ood.json`, `tasks_shard*.json`.

## Results

Run 2026-10-28 (launched 11:46 PT, last app finished about 12:14 PT). Scored once with `scripts/score_ood_langs.py` exactly as pre-registered in 66c521d; full output in `artifacts/ood_langs/summary.json` and `score.txt`, gzipped rows in `artifacts/ood_langs/rows/`. Nothing in the protocol above was changed after the first rollout, and no rollout was rerun.

**Primary result: wayfinder-glimmer beats A1 out of domain, by the pre-registered rule.** wayfinder-glimmer minus A1, grep arm, reported recall: **+0.073, task-bootstrap 95% CI [+0.038, +0.107], repo-cluster CI [+0.036, +0.108]** (120 tasks, 111 repos). Both lower bounds are above 0. On the test split the same contrast was +0.080 [+0.049, +0.112], so the lead in the three languages wayfinder-glimmer saw least is about the same size (-0.007), on repos A1 cannot have trained on.

### Validation

Six model-by-arm cells, each 240 rows (120 tasks x seeds 0 and 1), all healthy: 0 error rows, 0 infrastructure errors, every (task, seed) exactly once, `retain_reasoning` true, `max_turns` 40, `max_seconds` 3600, no 429s, reasoning content in every rollout. Model and weights on every row: `wayfinder-v3` / `/data/sft/v3_merged`, `a1` / `/weights/Agents-A1-FP8` (T 0.85, top_p 0.95, top_k 20, presence_penalty 1.1, thinking on), `glimmer` / `/weights/Muse-Glimmer-30B`. Startup checks passed on all 12 apps (served name correct, tool call parsed, wayfinder-glimmer T=0 logprobs differ from base by L1 0.65, base apps matched the recorded base logprobs at L1 0.0 to 0.002). Exclusions: none; all 120 tasks scored in every arm. No relaunch. The index was offered in all 240 sem rollouts per model; `semantic_search` was called in 182 (wayfinder-glimmer), 216 (A1) and 232 (base) of them, so wayfinder-glimmer uses the index least.

Tool-call health, counted from the rows:

| Arm | Tool calls | Malformed | Rate | Turns with no tool call | Rollouts stopped by 3 consecutive malformed | Leaked markup |
|---|---|---|---|---|---|---|
| wayfinder-glimmer grep | 6,485 | 8 (invalid args) | 0.12% | 0 | 0 | 0 |
| wayfinder-glimmer sem | 6,264 | 7 | 0.11% | 1 | 0 | 0 |
| A1 grep | 8,432 | 79 (52 unknown tool, 27 invalid) | 0.94% | 23 | 4 | 0 |
| A1 sem | 8,144 | 63 | 0.77% | 19 | 2 | 0 |
| base grep | 7,548 | 97 (31 unknown, 66 invalid) | 1.29% | 0 | 9 | 0 |
| base sem | 7,377 | 61 | 0.83% | 0 | 8 | 0 |

No bad-JSON arguments in any arm. Unknown-tool calls in the grep arms are `semantic_search` calls, which that arm does not offer. Reasoning is re-rendered into the next prompt for all three models (median prompt growth over previous output 1.07 for wayfinder-glimmer, 1.04 for A1, 1.07 for base).

### Arms (95% task-bootstrap CI; seeds averaged within task; 120 tasks)

| Arm | Reported recall | recall@5 | recall@7 | Precision | All-gold | Turns mean (p50) | Tokens | Ended by report | Files reported |
|---|---|---|---|---|---|---|---|---|---|
| wayfinder-glimmer grep | 0.686 [0.637, 0.734] | 0.581 [0.530, 0.632] | 0.655 [0.605, 0.705] | 0.436 [0.392, 0.481] | 0.34 [0.27, 0.42] | 8.6 (8) | 209k | 100% | 7.0 |
| A1 grep | 0.613 [0.563, 0.662] | 0.498 [0.450, 0.548] | 0.570 [0.519, 0.619] | 0.399 [0.353, 0.445] | 0.27 [0.20, 0.34] | 32.6 (39) | 737k | 53% [45, 60] | 7.3 |
| base grep | 0.556 [0.505, 0.608] | 0.510 [0.459, 0.563] | 0.545 [0.494, 0.597] | 0.473 [0.423, 0.524] | 0.21 [0.15, 0.28] | 28.6 (29) | 542k | 83% [77, 89] | 5.4 |
| wayfinder-glimmer sem | 0.698 [0.649, 0.745] | 0.587 [0.533, 0.640] | 0.660 [0.608, 0.711] | 0.455 [0.409, 0.500] | 0.35 [0.27, 0.42] | 8.1 (8) | 197k | 100% | 6.8 |
| A1 sem | 0.651 [0.601, 0.700] | 0.543 [0.492, 0.593] | 0.598 [0.547, 0.649] | 0.438 [0.391, 0.485] | 0.31 [0.23, 0.38] | 32.3 (38) | 765k | 55% [48, 62] | 7.1 |
| base sem | 0.586 [0.537, 0.636] | 0.532 [0.482, 0.583] | 0.567 [0.517, 0.618] | 0.508 [0.456, 0.561] | 0.23 [0.16, 0.31] | 26.4 (26) | 527k | 86% [81, 91] | 5.2 |

Stop reasons (rollouts of 240): wayfinder-glimmer grep and sem 240 report; A1 grep 126 report, 114 turn cap; A1 sem 133 and 107; base grep 199 and 41; base sem 207 and 33.

### Paired contrasts (point estimate, task-bootstrap CI / repo-cluster CI; 120 tasks, 111 repos)

| Contrast | Reported recall | recall@5 | recall@7 | Precision | All-gold | Turns | Tokens |
|---|---|---|---|---|---|---|---|
| **wayfinder-glimmer - A1, grep (primary)** | **+0.073 [+0.038, +0.107] / [+0.036, +0.108]** | +0.083 [+0.047, +0.119] / [+0.047, +0.120] | +0.085 [+0.049, +0.122] / [+0.048, +0.123] | +0.037 [+0.008, +0.064] / [+0.009, +0.064] | +0.071 [+0.008, +0.133] / [+0.004, +0.138] | -24.0 [-25.4, -22.5] | -528k [-580k, -476k] |
| wayfinder-glimmer - A1, sem | +0.047 [+0.013, +0.081] / [+0.014, +0.081] | +0.044 [+0.007, +0.081] / [+0.007, +0.082] | +0.062 [+0.025, +0.100] / [+0.026, +0.101] | +0.016 [-0.015, +0.046] / [-0.015, +0.048] | +0.037 [-0.021, +0.096] / [-0.021, +0.097] | -24.3 [-25.5, -23.0] | -568k [-617k, -518k] |
| wayfinder-glimmer - base, grep | +0.130 [+0.092, +0.169] / [+0.092, +0.169] | +0.071 [+0.028, +0.114] / [+0.027, +0.116] | +0.110 [+0.073, +0.149] / [+0.072, +0.150] | -0.037 [-0.067, -0.006] / [-0.067, -0.006] | +0.129 [+0.071, +0.192] / [+0.070, +0.194] | -20.0 [-21.3, -18.6] | -334k [-381k, -289k] |
| wayfinder-glimmer - base, sem | +0.112 [+0.076, +0.148] / [+0.077, +0.145] | +0.055 [+0.018, +0.093] / [+0.017, +0.092] | +0.093 [+0.058, +0.131] / [+0.059, +0.128] | -0.053 [-0.082, -0.025] / [-0.081, -0.025] | +0.113 [+0.058, +0.171] / [+0.058, +0.168] | -18.3 [-19.6, -16.9] | -330k [-375k, -287k] |
| A1 - base, grep (descriptive) | +0.057 [+0.009, +0.105] / [+0.010, +0.104] | -0.012 [-0.059, +0.036] | +0.025 [-0.021, +0.072] | -0.074 [-0.108, -0.038] | +0.058 [-0.008, +0.125] | +4.0 | +195k |
| A1 - base, sem (descriptive) | +0.065 [+0.024, +0.106] / [+0.025, +0.104] | +0.011 [-0.035, +0.057] | +0.031 [-0.014, +0.076] | -0.070 [-0.106, -0.034] | +0.075 [+0.008, +0.142] | +6.0 | +238k |

Semantic minus grep within model: wayfinder-glimmer +0.011 [-0.011, +0.034], A1 +0.037 [+0.003, +0.072], base +0.030 [-0.010, +0.069]. Unlike on the test split, the index lifts A1 enough to be separable from zero (barely), and the wayfinder-glimmer lead on the sem arm is smaller (+0.047) though still clear of 0 on both CIs. The lead survives fixed report length: wayfinder-glimmer reports fewer files than A1 (7.0 against 7.3) and leads by +0.083 at recall@5 and +0.085 at recall@7, so it is not padding. wayfinder-glimmer's precision is 0.037 above A1's and 0.037 below the base's (the base reports 1.7 fewer files).

By repo and task, wayfinder-glimmer minus A1 on grep: repos 57 win, 19 loss, 35 tie (of 111); tasks 62 / 19 / 39 (of 120). wayfinder-glimmer minus A1 on sem: repos 44 / 20 / 47; wayfinder-glimmer minus base on grep: repos 62 / 11 / 38.

### Per language (reported recall, descriptive; 40 tasks each, not powered, no decision uses them)

| Language | Tasks / repos | wayfinder-glimmer grep | A1 grep | base grep | wayfinder-glimmer - A1 grep [task CI] | wayfinder-glimmer - A1 sem [task CI] | wayfinder-glimmer - base grep [task CI] |
|---|---|---|---|---|---|---|---|
| C# | 40 / 36 | 0.668 | 0.583 | 0.521 | +0.086 [+0.035, +0.140] | +0.019 [-0.031, +0.071] | +0.147 [+0.068, +0.230] |
| Java/Kotlin | 40 / 40 | 0.680 | 0.597 | 0.552 | +0.083 [+0.015, +0.149] | +0.057 [-0.009, +0.124] | +0.128 [+0.067, +0.189] |
| Swift | 40 / 38 | 0.710 | 0.661 | 0.596 | +0.049 [-0.012, +0.107] | +0.064 [+0.002, +0.121] | +0.114 [+0.065, +0.166] |

wayfinder-glimmer is ahead of A1 on the grep and sem arms in every language. The grep interval excludes 0 for C# and Java/Kotlin, and includes 0 for Swift (the smallest lead, +0.049, where wayfinder-glimmer has the most training tasks of the three); the sem interval excludes 0 only for Swift. At 40 tasks per language these are consistent with one common effect of about +0.07.

### Reading and caveats

This is the result the pre-registration was built to test, and it came out the same way as the test split: wayfinder-glimmer beats A1 by +0.07 reported recall with both CIs clear of 0, with a quarter of the turns (8.6 against 32.6) and 28% of the tokens, because A1 reaches `report` in only 53% of grep rollouts against 100% for wayfinder-glimmer. Out-of-domain language did not erase the lead, and the thin Swift, Java/Kotlin and C# coverage in wayfinder-glimmer's SFT (6 to 9 tasks each) did not hurt it relative to A1. wayfinder-glimmer also beats the untrained base by +0.13, so the post-training gain holds on these languages. Caveats: (1) A1 and the base are single prompted configurations at the 40-turn cap and part of their score is budget-fallback credit; (2) wayfinder-glimmer is one training run, so this measures the checkpoint, not the recipe; (3) the repos are young, small and mostly one-developer projects, 58 of 120 commits are from October 2026, so absolute recall (wayfinder-glimmer 0.69 against 0.52 on the test split) is not comparable across sets, only contrasts are; (4) questions are raw commit subjects and are broad in this population; (5) this set does not separate "wayfinder-glimmer did not train on these languages" from "wayfinder-glimmer did not train on these repos": the test split already showed repo-level generalization, and the language-level claim here rests on the three languages being thin, not absent, in wayfinder-glimmer's SFT. This set is now spent: it is not used to choose between models or recipes.

### Spend

12 H100 apps estimated at Modal list price from per-app GPU seconds in the meta files (`artifacts/ood_langs/shards/*/*.meta.json`): $26.12 (wayfinder-glimmer $11.89, A1 $7.42, base $6.81). Voyage indexing of the 120 repo@commits: $15.17 at list price (`index_registry.json`). Watcher `ops/ood_eval_watch.py` ran for the whole eval (pid in `artifacts/ops/ood_eval_watch.pid`); no `wayfinder-oodeval*` app is live, and `wayfinder-grpo-*` and `asa-*` were not touched.
