# wayfinder-qwen: results

These are the results for wayfinder-qwen, the model in the final submission. Intervals are 95% bootstrap intervals over tasks. The raw task rows come from a private benchmark and are not in this repository, so each table below names where its numbers come from. "Private evaluation data, summarised in docs/HF_RELEASE_PLAN.md" means the rows are not here and the summary and its source are listed in section 7 of that file. The first model, wayfinder-glimmer, is not part of these results; its write-ups are kept in [glimmer_era/](glimmer_era/).

## 1. The cascade on 200 held-out tasks

200 tasks from 47 repositories never seen in training, 2 seeds per task, seeds averaged, task as the unit. Recall@12 is the share of the right files in the top 12 of the list.

| List | recall@12 |
|---|---|
| Model's own list | 0.414 |
| After the hybrid merge | 0.493 |
| After the rerank (final list) | 0.531 |

The whole cascade adds +0.117 [+0.095, +0.140] recall@12. The cascade also makes the list longer and lowers precision@12 from 0.416 to 0.303 (-0.113 [-0.142, -0.085]). It finds more of the right files and returns more files that are not right. Recall@5 moves from 0.336 to 0.345, which is not a significant change, so these data show no gain at the top of the list.

Source: private evaluation data, summarised in docs/HF_RELEASE_PLAN.md (section 7, "200-task strata"). The same numbers are in [../MODEL_CARD_QWEN.md](../MODEL_CARD_QWEN.md) under "200 held-out tasks: the model's own list, and after the cascade". The model's own reported recall on the same tasks, 0.415 on the grep arm (400 rollouts), is in the same card; the scorers are in [../../eval/scorers/](../../eval/scorers/).

## 2. Per-stratum results

Copied from [../MODEL_CARD_QWEN.md](../MODEL_CARD_QWEN.md), section "Where it does well and where it does not". The strata were computed for that card (see docs/HF_RELEASE_PLAN.md, section 7) and are not recomputed here. Intervals are paired bootstrap intervals over tasks, not adjusted for the number of strata, so read the small ones as direction only. A stratum with fewer than 8 tasks is "not enough tasks". "Change" is final list minus the model's own list.

### 200 held-out tasks, recall@12

| Stratum | Tasks | Own list | After hybrid | After rerank | Change |
|---|---|---|---|---|---|
| All | 200 | 0.414 | 0.493 | 0.531 | +0.117 [+0.095, +0.140] |
| **Category** | | | | | |
| implementing_new_features | 119 | 0.392 | 0.459 | 0.503 | +0.112 [+0.083, +0.142] |
| paying_tech_debt | 36 | 0.466 | 0.558 | 0.593 | +0.127 [+0.079, +0.177] |
| refactoring | 25 | 0.477 | 0.570 | 0.627 | +0.151 [+0.092, +0.217] |
| modernize | 17 | 0.388 | 0.478 | 0.467 | +0.080 [+0.027, +0.136] |
| cybersecurity_bug_finding, db_schema_rot | 2, 1 | not enough tasks | | | |
| **Language** | | | | | |
| TypeScript/JavaScript | 50 | 0.414 | 0.508 | 0.548 | +0.134 [+0.088, +0.185] |
| Java/Kotlin | 28 | 0.389 | 0.454 | 0.494 | +0.105 [+0.045, +0.173] |
| Go | 18 | 0.428 | 0.417 | 0.508 | +0.080 [+0.000, +0.173] |
| Other | 18 | 0.340 | 0.415 | 0.430 | +0.090 [+0.032, +0.159] |
| C# | 18 | 0.509 | 0.605 | 0.652 | +0.143 [+0.102, +0.183] |
| Swift | 16 | 0.453 | 0.495 | 0.525 | +0.073 [+0.018, +0.136] |
| Rust | 16 | 0.390 | 0.556 | 0.554 | +0.164 [+0.109, +0.223] |
| C/C++ | 15 | 0.469 | 0.513 | 0.583 | +0.113 [+0.027, +0.209] |
| Python | 13 | 0.326 | 0.462 | 0.476 | +0.151 [+0.086, +0.221] |
| Ruby | 8 | 0.434 | 0.516 | 0.521 | +0.087 [+0.000, +0.191] |
| **Right files per task** | | | | | |
| 3 to 5 | 77 | 0.485 | 0.552 | 0.567 | +0.082 [+0.052, +0.116] |
| 6 to 8 | 62 | 0.405 | 0.497 | 0.545 | +0.141 [+0.096, +0.190] |
| 9 or more | 61 | 0.334 | 0.415 | 0.472 | +0.138 [+0.105, +0.172] |

No held-out task has fewer than 3 right files. Where it ties: in Go and Ruby the gain is not separable from zero (18 and 8 tasks). For Rust and modernize the hybrid merge did the work and the rerank added nothing (Rust 0.556 to 0.554, modernize 0.478 to 0.467). The final list is not below the model's own list in any stratum shown. The model's own list is weakest on Python (0.326, 13 tasks), Other languages (0.340, 18 tasks) and tasks with 9 or more right files (0.334, 61 tasks); part of that is the 12-file report cap.

### 26 tasks in Muse Code, files found with and without Wayfinder

From the same card, section "26 tasks in Muse Code: Muse Code alone against Muse Code with Wayfinder". One run per task. "More / same / fewer" counts tasks where Muse Code with Wayfinder found more, the same or fewer right files than Muse Code alone.

| Stratum | Tasks | More / same / fewer | Files found, alone to with |
|---|---|---|---|
| All | 26 | 6 / 11 / 9 | 94 to 90 (of 171) |
| implementing_new_features | 10 | 2 / 3 / 5 | 30 to 24 |
| paying_tech_debt | 8 | 2 / 4 / 2 | 30 to 31 |
| TypeScript/JavaScript | 9 | 2 / 6 / 1 | 34 to 36 |
| 3 to 5 right files | 11 | 1 / 5 / 5 | 31 to 25 |
| 6 to 8 right files | 9 | 3 / 3 / 3 | 32 to 31 |

Smaller strata (refactoring, modernize, db_schema_rot, C#, Java/Kotlin, Other, Go, Swift, 9 or more right files) have too few tasks to report. It loses ground in implementing_new_features and in tasks with 3 to 5 right files, and ties in the other strata shown. These samples are too small for per-stratum intervals.

## 3. Coding pass rate on 34 tasks

34 tasks that fail without search, 2 seeds. The coder is Opus 5.5 with no search tools, network blocked; it can open only the files the searcher returned, and held-out tests are absent from its sandbox.

| Searcher | Pass rate |
|---|---|
| No search | 0.074 |
| wayfinder-qwen with the cascade | 0.824 [0.71, 0.93] |
| Right files given | 0.971 |
| Sonnet 5.5 high, run without the cascade | 0.868 |
| GPT-6.1 Sol, run without the cascade | 0.853 |
| Agents-A1-35B, run without the cascade | 0.853 |

Search lifts the pass rate by about 0.8 over no search. The differences between searchers are unresolved on 34 tasks. The other three searchers were run on 2026-11-02 without the cascade, so the comparison is not like for like. The 34 tasks come from 8 repositories and 16 of them come from one, so the effective sample is closer to 8.

Source: private evaluation data, summarised in README.md and docs/HF_RELEASE_PLAN.md. The harness code is in [../../eval/e2e/](../../eval/e2e/). An earlier run of the same design without the cascade (wayfinder-qwen 0.87, 59 of 68 runs) is in [../MODEL_CARD_QWEN.md](../MODEL_CARD_QWEN.md), section "End to end"; the first-model write-up is [glimmer_era/e2e_final_results.md](glimmer_era/e2e_final_results.md).

## 4. Search benchmark against frontier models

The same 200 held-out tasks (2 seeds, 400 rollouts per arm), grep arm, same explore loop and 40-turn cap. Reported recall is the share of right files in the model's final report (at most 12 files). Cost is per search task.

| Model | Reported recall | Cost per task |
|---|---|---|
| wayfinder-qwen, own list | 0.415 | about $0.045 (open-price equivalent) |
| Agents-A1-35B | 0.313 | $0.113 |
| Sonnet 5.5 high | 0.529 | $0.120 |
| GPT-6.1 Sol | 0.580 | $0.062 (rate unconfirmed) |

With the cascade, wayfinder-qwen reaches recall@12 of 0.531, level with Sonnet and a little under GPT. The lists have different lengths, so this is not a like-for-like comparison. On recall@5 it trails the frontier models. The wayfinder-qwen cost is the open-model list price applied to its tokens; it is not a bill. The GPT rate is an assumption that was not confirmed against the gateway catalog.

Source: Agents-A1, Sonnet and GPT rows are in [../../eval/results/heldout_hard__score_4way.txt](../../eval/results/heldout_hard__score_4way.txt) (200 tasks per arm). The wayfinder-qwen row is private evaluation data, summarised in [../MODEL_CARD_QWEN.md](../MODEL_CARD_QWEN.md). The cost definitions are in [glimmer_era/heldout_hard_v1_frontier.md](glimmer_era/heldout_hard_v1_frontier.md).

## 5. Inside Muse Code

26 tasks from 18 repositories, one run per task, default configuration, direct hand-off and the cascade, wayfinder-qwen behind Scout. The main agent is muse-spark-1.3-contributor on both sides, so a pair differs only by the Wayfinder `explore` call.

| | Muse Code alone | With Wayfinder |
|---|---|---|
| Cost per task (list price, excludes GPU) | $0.342 | $0.195 |
| Muse Code input tokens read | 887K | 441K |
| precision@12 | 0.366 | 0.355 |
| recall@12 | 0.590 | 0.536 |
| recall@5 | 0.449 | 0.383 |
| Median wall time | 80 s | 105 s |

Cost is about 43 percent lower and the main agent reads about half the tokens. Precision@12 and recall@12 differ within noise (recall@12 difference -0.053 [-0.127, +0.015]). Recall@5 is lower with Wayfinder and we have not tested why. Wayfinder is about 25 seconds slower in median time. With Wayfinder, more right files were found on 6 tasks, the same on 11 and fewer on 9. The claim is cost and tokens, not better accuracy.

Source: private evaluation data, summarised in docs/HF_RELEASE_PLAN.md (section 7, "26-task Muse Code counts"). The same table is in [../MODEL_CARD_QWEN.md](../MODEL_CARD_QWEN.md). The harness is in [../../scout/eval/harness/](../../scout/eval/harness/).

### The demo video

The demo uses the three tasks jellyfin, excalidraw and mongoose, five runs each. The video shows the best of the 5 runs per task, chosen after the batch finished. That is a selection, so the video does not represent a typical run. The median run for each task is listed in the demo notes (kept with the private deck source, not in this repository). The 26-task table above is the representative result. See [../demo.md](../demo.md).

## 6. Limits

- The Muse Code study has one run per task (n = 26), so it is calibration-scale evidence.
- Wayfinder finds fewer right files than Muse Code alone on 9 of 26 tasks, most often in implementing_new_features and in tasks with 3 to 5 right files. On excalidraw it did worse in 4 of 5 batch runs.
- Wayfinder is slower in wall time (about 25 s in the median on the 26 tasks).
- The cascade lowers precision@12 (0.416 to 0.303) and shows no significant gain at recall@5.
- The 34-task coding set has 16 tasks from one repository. The other searchers were run without the cascade on a different day, so the pass-rate differences are unresolved.
- The cost figures for wayfinder-qwen are open-model list prices, not a bill, and the GPT rate is unconfirmed.
- The held-out numbers were produced with MTP off and the coding and Muse Code results with MTP on. Whether thinking was on in every Muse Code call was not verified from a trace. More limits are in [../MODEL_CARD_QWEN.md](../MODEL_CARD_QWEN.md).
- Evaluated on public repositories and English task wording only.
