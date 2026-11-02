# Wayfinder-Glimmer SFT eval (v2b recipe on spark_v1 + spark_v2 data), dev set

Run 2026-10-27. wayfinder-glimmer passes all five pre-registered PRD section 15 criteria on the grep-only arm, with larger margins than v2b. Against v2b the improvement is real in direction but small: reported recall is +0.033 [-0.003, +0.072] on the grep arm (not significant) and +0.032 [+0.005, +0.060] on the semantic arm (significant, exploratory). wayfinder-glimmer is the best model so far and sits 0.036 below the teacher on reported recall.

## What was trained

Data v3 (`wayfinder-data:/sft/teacher_v3/`, summary `artifacts/sft/teacher_v3_summary.json`): all rows of teacher passes `spark_v1` (plus the `spark_v1_baseten` rows v2 also read) and `spark_v2` (seeds 4-7; 2,822 rows when converted). `spark_v3` was not included. Filters are v2's, unchanged: train split only, the 17 E2E eval tasks excluded, one clean report with budget not exhausted, all gold found (min_recall 1.0), evidence covered, at most the task's p75 turns, at most 49,152 tokens, dedupe on task and seed, plus the 33 strict A1 examples. Dev, test and eval-holdout repos never enter (the split filter dropped none because the volume holds train-split tasks only). Conversion ran through `convert_teacher_modal.py`, which gained a `--teacher-dirs` option; nothing else in the converter changed.

| Funnel step | Rows |
|---|---|
| rows in (spark_v1 + spark_v1_baseten + spark_v2) | 5,683 |
| train split | 5,683 |
| not an E2E eval task | 5,627 (56 dropped) |
| converted (clean protocol) | 4,824 |
| all gold found | 2,372 |
| evidence covered | 2,210 |
| at most p75 turns | 1,844 |
| fits 49,152 tokens | 1,353 |
| deduped | 1,350 |

Drop reasons (rows): not all gold 2,452; budget exhausted 520; fits 49k 491; over p75 turns 366; assistant without tool call 268; evidence not covered 162; E2E eval task 56; no report 14; duplicate task-seed 3; multi report 1.

Final set: 1,383 examples (1,350 teacher + 33 A1), 41.56M tokens (v2: 20.6M), 7.02M assistant tokens, 344 tasks (v2: 300), 153 repos (v2: 141), semantic tool offered in 1,062 examples = 76.8% (v2: 60%).

Training: v2b recipe unchanged. LoRA r64 alpha 128, lr 5e-5, 2 epochs, warmup 5, pack length 49,152, one packed row per replica per step, 4x H200 data parallel (`train_v1.py --gpu h200x4`), seed 0. 239 steps per epoch, 478 steps, 180 min wall, loss first5 to last5 0.70 to 0.39 (v2b 0.66 to 0.39). Gradient-norm spikes up to 24 on 7 steps (56, 86, 87, 164, 180, 265, 289), clipped at 1.0, no loss divergence. Merged as for v1 and v2 (416 modules, scale 2.0, fp32 merge then bf16) to `/data/sft/v3_merged`.

## Eval protocol

Identical to v2: the 50 dev tasks x 2 seeds (`artifacts/teacher_test/stageB_tasks.json`), scout product prompt, convergence nudge, tools, T=0.85, top_p=0.95, `max_turns` 40, `max_seconds` 3600, one H100 container per arm at concurrency 14, vLLM 0.32.0, max_model_len 65536. Rows: `artifacts/sft_v3/sft_v3_{grep,sem}.jsonl`; metrics and paired task-bootstrap CIs: `artifacts/sft_v3/summary.json` (`scripts/score_sft_v3.py`, which reuses the v2 scorer's metric definitions).

Row validation (before treating the run as healthy): both arms have 100 rows, 50 tasks, 0 errors, model `wayfinder-v3` on every row; turns min/median/max 5/11/40 (grep) and 5/11/39 (sem); wall time median 395 s and 387 s. Adapter check passed on both servers (served as `wayfinder-v3`, T=0 logprobs differ from base, tool call parsed). The semantic tool was offered in all 100 sem rows and called in 78 of them (1.21 calls per rollout); in the grep arm it was offered in none. The two H100 servers ran 59 and 61 minutes.

## Results (95% task-bootstrap CI, seeds averaged within task)

| Metric | Base | v2b grep | wayfinder-glimmer grep | v2b sem | wayfinder-glimmer sem | Teacher |
|---|---|---|---|---|---|---|
| Reported recall | 0.246 [0.177, 0.318] | 0.362 [0.279, 0.447] | 0.396 [0.314, 0.482] | 0.382 [0.296, 0.469] | 0.415 [0.334, 0.496] | 0.440 [0.354, 0.529] |
| All-gold (reported) | 0.02 [0.00, 0.06] | 0.05 [0.00, 0.11] | 0.08 [0.02, 0.16] | 0.09 [0.02, 0.17] | 0.06 [0.01, 0.13] | 0.10 [0.03, 0.19] |
| Touched-file recall | 0.463 [0.368, 0.558] | 0.512 [0.418, 0.604] | 0.549 [0.462, 0.636] | 0.519 [0.432, 0.604] | 0.554 [0.466, 0.642] | 0.553 [0.465, 0.642] |
| All-gold (touched) | 0.15 [0.06, 0.25] | 0.16 [0.07, 0.26] | 0.19 [0.09, 0.30] | 0.14 [0.06, 0.24] | 0.19 [0.10, 0.29] | 0.19 [0.09, 0.31] |
| Ended by `report` before cap | 21% [12, 30] | 95% [89, 99] | 97% [92, 100] | 99% [97, 100] | 98% [94, 100] | 95% [89, 99] |
| Turns mean (p50) | 37.4 (40) | 13.8 (11) | 12.2 (11) | 12.4 | 12.8 (11) | 11.5 |
| Tokens per rollout | 821k | 490k | 423k | 428k | 451k | 388k |
| Tool calls per turn | 1.15 | 3.22 | 3.18 | 3.32 | 3.22 | 3.05 |
| Model latency per turn | 5.6 s | 34.9 s | 36.2 s | 33.0 s | 37.3 s | 30.3 s |

## PRD section 15, grep arm vs base (paired by task, 50 tasks)

| Criterion | v2b | wayfinder-glimmer | Result |
|---|---|---|---|
| Reported recall delta, CI lower bound > 0 (target >= +0.10) | +0.117 [+0.072, +0.167] | +0.150 [+0.096, +0.209] | PASS, target met |
| Ends by `report` before the cap >= 80% | 95% | 97% | PASS |
| Turns p50 <= 20 | 11 | 11 | PASS |
| Touched-file recall guardrail, CI lower bound > -0.10 | +0.049 [+0.012, +0.087] | +0.086 [+0.045, +0.128] | PASS (touched recall rose) |
| Tokens per rollout down >= 30% | -40% | -48% | PASS |
| Criteria passed | 5 of 5 | 5 of 5 | |

Stop reasons for wayfinder-glimmer grep: 97 report, 2 context budget, 1 turn cap. The stretch target (teacher 0.44) is not met on the grep arm: wayfinder-glimmer minus teacher on reported recall is -0.036 [-0.073, -0.004], touched recall +0.007 [-0.029, +0.038], turns +0.7 [-0.4, +1.8], tokens +37k [-23k, +93k]. The v2b gap was -0.070.

Semantic arm (exploratory) against base: reported recall +0.169 [+0.117, +0.225], touched +0.090 [+0.051, +0.133].

## wayfinder-glimmer against v2b (paired by task, 50 tasks, wayfinder-glimmer minus v2b)

| Metric | Grep arm | Semantic arm |
|---|---|---|
| Reported recall | +0.033 [-0.003, +0.072] | +0.032 [+0.005, +0.060] |
| All-gold (reported) | +0.03 [0.00, +0.08] | -0.03 [-0.08, 0.00] |
| Touched-file recall | +0.036 [-0.004, +0.076] | +0.034 [+0.005, +0.063] |
| All-gold (touched) | +0.03 [0.00, +0.07] | +0.05 [+0.01, +0.09] |
| Ended by `report` | +0.02 [0.00, +0.05] | -0.01 [-0.03, 0.00] |
| Turns | -1.56 [-2.92, -0.32] | +0.40 [-1.03, +1.87] |
| Tokens per rollout | -66k [-140k, -1k] | +22k [-47k, +95k] |
| Model latency per turn | +1.3 s [-1.3, +3.9] | +4.2 s [+1.2, +7.7] |

wayfinder-glimmer semantic minus wayfinder-glimmer grep: reported recall +0.019 [-0.007, +0.045], turns +0.5 [-0.9, +1.9], tokens +27k [-38k, +91k]; the index still gives no separable gain over grep.

## Reading

Doubling the data (701 to 1,383 examples, 20.6M to 41.6M tokens) at the same recipe moved reported recall by about +0.03 on both arms, in the direction of the teacher. The grep-arm interval touches zero, so on its own that arm does not establish a wayfinder-glimmer over v2b gain; the semantic arm's does exclude zero, and the two point estimates agree. The grep-arm token and turn reductions (-66k, -1.6 turns) have CIs that exclude zero narrowly. The remaining gap to the teacher (0.036 grep, 0.025 semantic) is smaller than v2b's (0.078 and 0.058) and its CI now barely excludes zero on the grep arm. All-gold in the report stays between 0.05 and 0.09 for every variant at n=100 rollouts, so differences there are counts of a few rollouts.

## Caveats

1. Dev set, 50 tasks, 18 repos, 2 seeds; CIs treat tasks as the sample. The wayfinder-glimmer vs v2b comparison is two single training runs (seed 0), so run-to-run training variance is not measured; a +0.03 difference is within what a reseed could plausibly move.
2. The `spark_v2` pass shares task ids with `spark_v1` (the same 709 train tasks with new seeds), so wayfinder-glimmer sees more trajectories per task, not more tasks beyond 344 versus 300 in the filtered set.
3. Turn latency and wall time come from a 14-way-concurrent single-H100 server and are not model properties. The wayfinder-glimmer sem arm shows +4.2 s per turn against v2b sem, which is a serving difference at this noise level.
4. Gradient-norm spikes (peak 24.4, 7 steps over 8) occurred, clipped; the final loss matches v2b. Merged bf16 weights are what was scored.
5. The 17 E2E-eval tasks were excluded, but other tasks from the same train-split repos were trained on (as in v2), so any E2E result on wayfinder-glimmer is repo-seen, task-held-out.
6. `spark_v3` (seeds 8-11) was still collecting and was not used.


Data conversion about $0.3; wayfinder-glimmer train $54.5 (4x H200, 10,803 s wall, 43,213 GPU-s at $4.54/h); merge about $1; two eval arms $7.9 (H100, 3,518 s and 3,667 s). Watcher `ops/sftv3_watch.py` armed before launch (auth canary, 360-minute kill age, prefix `wayfinder-sftv3`); apps `wayfinder-sftv3-train`, `-merge-v3`, `-eval-v3`. No `wayfinder-sftv3*` apps remain live.
