# SFT v2 eval (two hedged variants), dev set

Run 2026-10-26/27. Both variants pass all five pre-registered PRD section 15 criteria on the grep-only arm. v2b (2 epochs at lr 5e-5) is the winner by a small margin; the two are statistically indistinguishable on recall.

## What was trained

Data v2 (`wayfinder-data:/sft/teacher_v2/`, summary `artifacts/sft/teacher_v2_summary.json`): every Spark Open teacher row on the volume (2,861 rows at conversion time; the collection kept running after the 2,703 in the brief), train split only, same filters as v1 (converted, all-gold min_recall 1.0, evidence-covered, at most the task's p75 turns, max 49,152 tokens, dedupe on task and seed), plus the 33 strict A1 examples. The 17 round-3 E2E headroom tasks are excluded (28 teacher rows dropped; ids in `artifacts/sft/e2e_excluded_tasks.json` and the summary), so an E2E eval of the trained model does not see them in training.

| Funnel step | Rows |
|---|---|
| rows in (all on volume) | 2,861 |
| train split | 2,861 (the volume holds train tasks only) |
| not an E2E eval task | 2,833 |
| converted (one report, budget not exhausted, clean protocol) | 2,443 |
| all gold found | 1,205 |
| evidence covered | 1,115 |
| at most p75 turns | 938 |
| fits 49,152 tokens | 673 |
| deduped | 668 |

Final set: 701 examples (668 teacher + 33 A1), 20.6M tokens, 3.5M assistant tokens, 300 tasks, 141 repos, 60% of examples with the semantic tool offered (423 of 701). v1 had 157 examples, 4.5M tokens.

Training: LoRA r64 alpha 128, `train.py` on 4x H200 data-parallel (`train_v1.py --gpu h200x4`), pack length 49,152, one packed row per replica per step, effective batch about 190k tokens, 119 steps per epoch. `train.py` gained `--epochs` (fresh shuffle and packing each epoch).

| Variant | Recipe | Steps | Wall | Loss first5 to last5 | Merged checkpoint |
|---|---|---|---|---|---|
| v2a | 1 epoch, lr 1e-4 | 119 | 46 min | 0.63 to 0.42 | `/data/sft/v2a_merged` |
| v2b | 2 epochs, lr 5e-5 | 238 | 89 min | 0.66 to 0.39 | `/data/sft/v2b_merged` |

Each adapter was merged into the base weights as for v1 (`sft_merge_app.py`, fp32 merge, bf16 cast, 416 modules).

## Eval protocol

Same 50 dev tasks x 2 seeds (`artifacts/teacher_test/stageB_tasks.json`), same scout product prompt, nudge, tools and sampling as the baseline and v1. Changes against v1: `max_seconds` 3600 (no wall-clock cap; the longest rollout was 1,395 s, none ended on the clock) and `max_turns` 40, one H100 container per arm at concurrency 14. Rows: `artifacts/sft_v2/sft_{v2a,v2b}_{grep,sem}.jsonl`; metrics and paired task-bootstrap CIs: `artifacts/sft_v2/summary.json` and `artifacts/sft_v2/score.txt` (`scripts/score_sft_v2.py`). All 400 rollouts completed with zero errors. Grep arm is the pre-registered comparison; the semantic arm is exploratory (the index tool was offered in all 200 rows and called in 69% of v2a and 78% of v2b rollouts, 1.3 and 1.2 calls per rollout, against 3.2 for v1).

## Results (95% task-bootstrap CI, seeds averaged within task)

| Metric | Base | v1 grep | v2a grep | v2b grep | Teacher (Spark, ref) |
|---|---|---|---|---|---|
| Rollouts | 100 | 99 | 100 | 100 | 98 |
| Reported recall | 0.246 [0.177, 0.318] | 0.345 [0.265, 0.427] | 0.357 [0.272, 0.444] | 0.362 [0.279, 0.447] | 0.440 [0.354, 0.529] |
| Touched-file recall | 0.463 [0.368, 0.558] | 0.516 [0.424, 0.609] | 0.501 [0.411, 0.592] | 0.512 [0.418, 0.604] | 0.553 [0.465, 0.642] |
| All-gold in report | 0.02 [0.00, 0.06] | 0.05 [0.00, 0.11] | 0.09 [0.03, 0.16] | 0.05 [0.00, 0.11] | 0.10 [0.03, 0.19] |
| Ended by `report` (not a budget stop) | 21% | 35% | 88% | 95% | 95% |
| Turns p50 / mean | 40 / 37.4 | 24 / 23.1 | 13 / 15.0 | 11 / 13.8 | 9.5 / 11.5 |
| Tokens per rollout | 821k | 865k | 521k | 490k | 388k |
| Tool calls per turn | 1.14 | 3.50 | 3.05 | 3.22 | 3.05 |
| Model latency per turn (mean) | 5.6 s | 34.6 s | 33.3 s | 34.9 s | 30.3 s |
| Wall p50 | 190 s | 802 s | 417 s | 429 s | 278 s |
| Stop reasons (report / turn cap / context / wall), rollouts | 21 / 79 / 0 / 0 | 34 / 15 / 5 / 45 | 88 / 6 / 6 / 0 | 95 / 2 / 3 / 0 | not tabulated |

Semantic arm (exploratory, same protocol): reported recall v1 0.333, v2a 0.362 [0.287, 0.440], v2b 0.382 [0.296, 0.469]; ended by `report` 54%, 94%, 99%; turns p50 21, 12, 11; tokens 827k, 509k, 428k. Semantic minus grep within a variant is not significant on reported recall (v2a +0.005 [-0.032, +0.042], v2b +0.020 [-0.013, +0.059]).

Per-turn latency is a serving property (concurrency 14 and prefix-cache hit rate about 48% on one H100, against 95% and 5.6 s in the baseline at a different load), not a model property; it is within noise across v1, v2a and v2b and similar to the teacher's. Read the per-turn cost as unchanged and the per-rollout cost as down, because turns fell.

## PRD section 15 criteria, grep arm vs base (paired by task, 50 tasks)

| Criterion | v1 | v2a | v2b |
|---|---|---|---|
| Reported recall delta, CI lower bound > 0 (target >= +0.10) | +0.099 [+0.046, +0.156] PASS, target missed by 0.001 | +0.111 [+0.065, +0.162] PASS, target met | +0.117 [+0.072, +0.167] PASS, target met |
| Ends by `report` before the cap >= 80% | 35% FAIL | 88% PASS | 95% PASS |
| Turns p50 <= 20 | 24 FAIL | 13 PASS | 11 PASS |
| Touched-file recall guardrail, CI lower bound > -0.10 | +0.053 [+0.010, +0.099] PASS | +0.038 [-0.001, +0.076] PASS | +0.049 [+0.012, +0.087] PASS |
| Tokens per rollout down >= 30% | +5% FAIL | -37% PASS | -40% PASS |
| Criteria passed | 2 of 5 | 5 of 5 | 5 of 5 |

The stretch target (the teacher's reported recall 0.44) is not reached: v2a is -0.076 [-0.118, -0.038] and v2b -0.070 [-0.110, -0.036] below the teacher on reported recall, and 2 to 3.5 turns and 100 to 130k tokens above it.

## Which v2 wins

v2b, narrowly. Head to head the two do not differ on reported recall (v2a minus v2b -0.006 [-0.034, +0.019]), touched recall (-0.012 [-0.036, +0.011]), turns (+1.2 [-0.5, +2.9]) or tokens (+32k [-47k, +113k]). v2b ends by `report` more often (95% against 88%, difference -0.07 [-0.14, -0.01] for v2a), runs 2 fewer turns at the median and uses 6% fewer tokens, and its guardrail interval excludes zero while v2a's touches it. v2a reports all gold more often (0.09 against 0.05, +0.04 [+0.01, +0.08]), which is the one metric that favours it, and that is a small count (9 and 5 of 100). The extra epoch at a lower rate buys a little more convergence and a safer guardrail for twice the training cost ($27 against $14). v2b is the checkpoint to carry forward; v2a is a fine cheaper fallback.

Against v1 the gain is entirely in stopping (ended-by-report +0.53 to +0.60, turns -8 to -9, tokens -344k to -375k, all CIs exclude zero); reported recall is not distinguishable from v1 (v2a +0.012 [-0.030, +0.055], v2b +0.018 [-0.023, +0.060]). The v1 decision rule held: v1 did not converge for lack of signal, and 4.5x the data with a comparable epoch count fixed convergence. More data did not by itself lift reported recall past v1's level, so the remaining gap to the teacher (0.07 to 0.08) is about which files are named when it stops, not about when it stops.

## Caveats

1. Dev set, 50 tasks, 18 repos, 2 seeds; CIs treat tasks as the sample. None of the v2 training repos are dev repos (train split only).
2. Reported recall delta is about +0.11 to +0.12 with a CI of roughly +0.07 to +0.17, so the +0.10 target is met on the point estimate only; the lower bound is below +0.10.
3. All-gold in the report stays near 0.05 to 0.09 for every variant, versus 0.10 for the teacher; the claim here is convergence plus a modest recall gain, not that the model finds every gold file.
4. Turn-latency and wall columns come from a 14-way-concurrent single-H100 server with a 48% prefix-cache hit rate and are not a property of the model.
5. Merged bf16 weights are what was scored.
6. 33 strict A1 examples are in the training mix; the lossy-render A1 tier was not used.
7. The 17 E2E-eval tasks were excluded from the SFT data, but other tasks from the same 10 train-split repos (reqwest, mashumaro, gh-aw-mcpg, OpenBiliClaw, go-micro, celestia-app, cplt, helm, code-graph-rag) were trained on, so the E2E result on those repos is repo-seen, task-held-out. See `docs/e2e_eval_v2.md`.


Data conversion about $0.2; v2a train $14 (4x H200, 46 min), v2b train $27.5 (89 min), three merges about $1, four eval arms $15.6 (H100, about $3.7 to $4.1 each). Zero `wayfinder-sftv2*` apps and no containers from this track at the end; the watcher `ops/sftv2_watch.py` was armed at launch with an auth canary and a 200-minute kill age.
