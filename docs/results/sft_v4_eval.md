# SFT v4 eval (wayfinder-glimmer recipe on spark_v1 + spark_v1_baseten + spark_v2 + spark_v3 data), dev set, retain reasoning on

Run 2026-10-27. v4 passes all five PRD section 15 criteria on the grep arm, but it is not distinguishable from wayfinder-glimmer at the same retain-on protocol. Reported recall, v4 minus wayfinder-glimmer, is +0.013 [-0.015, +0.042] on the grep arm and -0.005 [-0.031, +0.023] on the semantic arm. v4 uses +0.8 turns and +24k tokens more than wayfinder-glimmer on the grep arm (CIs exclude zero narrowly). Adding 30% more data (1,383 to 1,802 examples) at the same recipe did not move recall. Starting GRPO from wayfinder-glimmer or v4 is a coin flip on this evidence; wayfinder-glimmer is cheaper to justify (already the reference everywhere).

## What was trained

Data v4 (`wayfinder-data:/sft/teacher_v4/`, summary `artifacts/sft/teacher_v4_summary.json`): all rows of `spark_v1`, `spark_v1_baseten`, `spark_v2` and `spark_v3` present at 10:11 PT (spark_v3 was about 87% collected; its hard-task tail was not waited for). Filters unchanged from wayfinder-glimmer: train split, E2E-eval tasks excluded (`artifacts/sft/e2e_excluded_tasks.json`; `eval/e2e/tasks_v3.json` did not exist yet, and the new E2E task set comes from dev/test/eval_holdout repos, which the split filter already excludes), one clean report with budget not exhausted, all gold found, evidence covered, at most the task's p75 turns, at most 49,152 tokens, dedupe on task and seed, plus 33 strict A1 examples.

| Funnel step | Rows |
|---|---|
| rows in | 8,190 |
| not an E2E eval task | 8,106 |
| converted (clean protocol) | 6,404 |
| all gold found | 3,143 |
| evidence covered | 2,915 |
| at most p75 turns | 2,360 |
| fits 49,152 tokens | 1,772 |
| deduped | 1,769 |

Final set: 1,802 examples (1,769 teacher + 33 A1), 53.92M tokens (wayfinder-glimmer: 41.56M), 8.95M assistant tokens, 357 tasks (wayfinder-glimmer: 344), 155 repos (wayfinder-glimmer: 153), semantic tool offered in 82.0% (wayfinder-glimmer: 76.8%).

Training: wayfinder-glimmer hyperparameters unchanged (LoRA r64 alpha 128, lr 5e-5, 2 epochs, warmup 5, pack length 49,152, one packed row per replica per step, seed 0). Shape: 8x H200 data parallel, a new `run_h200x8_long` function in `train_v1.py` (`--gpu h200x8`, timeout via `WF_TRAIN_TIMEOUT_H`). One pack per replica per step makes the global batch 8 packs, versus 4 in wayfinder-glimmer, so the step count is 154 per epoch / 308 total (wayfinder-glimmer: 239 / 478) rather than a matched global batch. This was a deliberate choice: the coordinator asked for matching the global batch, but matching needs 2 accumulation steps on 8 GPUs, which would give back none of the speedup relative to 4 replicas at 1 pack; I kept the wayfinder-glimmer per-replica setting. Treat "same hyperparameters" as "same lr, epochs, LoRA and per-replica batch", with half as many optimizer steps at twice the batch. Wall time 7,216 s (120 min; wayfinder-glimmer was 180 min on 4x H200 for fewer tokens), 57,726 GPU-s. Loss first5 to last5 0.705 to 0.373 (wayfinder-glimmer 0.70 to 0.39). Gradient norm peaked at 6.9, no step above 8 (wayfinder-glimmer had 7 spikes up to 24). Merged as before (416 modules, scale 2.0, max relative delta norm 0.0204) to `/data/sft/v4_merged`.

Scheduling: both 8x and 4x H200 functions were queued at once (the 8x was submitted 5 minutes after the 4x); the 8x won at 10:31 PT, about 7 minutes after submission, and the 4x app was stopped before it started.

## Eval protocol

Same as wayfinder-glimmer: 50 dev tasks x 2 seeds (`artifacts/teacher_test/stageB_tasks.json`), scout product prompt, convergence nudge, T=0.85, top_p=0.95, `max_turns` 40, `max_seconds` 3600, concurrency 14 per server, H100, vLLM 0.32.0, max_model_len 65536. Differences: `WF_EVAL_RETAIN=1` (retain reasoning, default since 22b2bcb) for v4 and the wayfinder-glimmer comparison column, and each arm is sharded across 8 H100 apps (`--shard i/8`, tasks i, i+8, ...; concatenated by `scripts/score_sft_v4.py`) instead of one. Sharding changes concurrency per server (about 14 per server over 6 to 7 tasks x 2 seeds), which may lower per-turn latency versus the single-server runs; latency is not a model property.

The wayfinder-glimmer retain-on comparison: grep arm rows are the OPD agent's `artifacts/opd/round1/eval_d/sft_v3_grep.jsonl` (config verified by the scorer: wayfinder-glimmer weights, retain true, T 0.85, max_turns 40, max_seconds 3600, same task file, 100 rows). The semantic arm for wayfinder-glimmer retain-on did not exist; it was re-run here on 4 H100 shards (`artifacts/sft_v4/v3_sem_shard*`, about $12). Base and teacher columns are retain-off (their original runs); only the wayfinder-glimmer and v4 columns share the retain-on setting.

Row validation: v4 grep and sem each have 100 rows, 50 tasks, 0 errors, model `wayfinder-v4` on every row, `retain_reasoning` true on every row; turns min/median/max 4/10/21 (grep), 5/9/19 (sem); wall time median 260 s and 284 s. wayfinder-glimmer retain-on: 5/9/14 (grep), 5/8/20 (sem), all 100 rows, 0 errors. Rows: `artifacts/sft_v4/sft_v4_{grep,sem}.jsonl`, `sft_v3r_sem.jsonl`; metrics, CIs and per-row validation: `artifacts/sft_v4/summary.json`.

## Results (95% task-bootstrap CI, seeds averaged within task)

| Metric | Base | wayfinder-glimmer grep (retain on) | v4 grep | wayfinder-glimmer sem (retain on) | v4 sem | Teacher |
|---|---|---|---|---|---|---|
| Reported recall | 0.246 [0.177, 0.318] | 0.408 [0.327, 0.490] | 0.421 [0.335, 0.508] | 0.398 [0.319, 0.479] | 0.393 [0.309, 0.477] | 0.440 [0.354, 0.529] |
| All-gold (reported) | 0.02 [0.00, 0.06] | 0.07 [0.02, 0.14] | 0.09 [0.03, 0.17] | 0.07 [0.01, 0.14] | 0.07 [0.01, 0.14] | 0.10 [0.03, 0.19] |
| Touched-file recall | 0.463 [0.368, 0.558] | 0.537 [0.449, 0.624] | 0.531 [0.437, 0.623] | 0.524 [0.437, 0.611] | 0.519 [0.430, 0.606] | 0.553 [0.465, 0.642] |
| All-gold (touched) | 0.15 [0.06, 0.25] | 0.16 [0.07, 0.27] | 0.17 [0.07, 0.28] | 0.16 [0.07, 0.26] | 0.15 [0.06, 0.25] | 0.19 [0.09, 0.31] |
| Ended by `report` before cap | 21% [12, 30] | 96% [91, 100] | 98% [94, 100] | 98% [94, 100] | 98% [94, 100] | 95% [89, 99] |
| Turns mean (p50) | 37.4 (40) | 9.2 (9) | 10.0 (10) | 9.2 (9) | 9.6 (9) | 11.5 |
| Tokens per rollout | 821k | 269k | 293k | 280k | 284k | 388k |

For reference, wayfinder-glimmer under the original retain-off eval (docs/sft_v3_eval.md): grep 0.396, 12.2 turns, 423k tokens; sem 0.415. Retain on cuts wayfinder-glimmer's turns from 12.2 to 9.2 and tokens by about a third at the same recall (+0.012 grep, -0.017 sem, within noise).

## PRD section 15, grep arm vs base (paired by task, 50 tasks)

| Criterion | wayfinder-glimmer (retain on) | v4 | Result |
|---|---|---|---|
| Reported recall delta, CI lower bound > 0 (target >= +0.10) | +0.162 [+0.115, +0.213] | +0.175 [+0.121, +0.234] | PASS, target met |
| Ends by `report` before the cap >= 80% | 96% | 98% | PASS |
| Turns p50 <= 20 | 9 | 10 | PASS |
| Touched-file recall guardrail, CI lower bound > -0.10 | +0.074 [+0.036, +0.111] | +0.068 [+0.025, +0.113] | PASS |
| Tokens per rollout down >= 30% | -67% | -64% | PASS |
| Criteria passed | 5 of 5 | 5 of 5 | |

v4 grep stop reasons: 98 report, 2 context budget. Semantic arm (exploratory) against base: reported recall +0.147 [+0.094, +0.204], touched +0.056 [+0.021, +0.093]. v4 grep minus teacher: reported recall -0.011 [-0.038, +0.014] (wayfinder-glimmer retain-on: not computed; the gap to the teacher is no longer separable from zero), turns -1.6, tokens -100k.

## v4 against wayfinder-glimmer (both retain on; paired by task, v4 minus wayfinder-glimmer)

| Metric | Grep arm | Semantic arm |
|---|---|---|
| Reported recall | +0.013 [-0.015, +0.042] | -0.005 [-0.031, +0.023] |
| All-gold (reported) | +0.02 [-0.02, +0.07] | 0.00 [-0.03, +0.03] |
| Touched-file recall | -0.006 [-0.031, +0.022] | -0.005 [-0.036, +0.022] |
| All-gold (touched) | +0.01 [-0.05, +0.07] | -0.01 [-0.03, 0.00] |
| Ended by `report` | +0.02 [0.00, +0.05] | 0.00 [0.00, 0.00] |
| Turns | +0.80 [+0.22, +1.41] | +0.37 [-0.12, +0.85] |
| Tokens per rollout | +23.9k [+0.2k, +48.0k] | +4.2k [-20.5k, +28.8k] |
| Model latency per turn | -2.5 s [-4.8, -0.3] | -3.6 s [-5.7, -1.5] |

v4 semantic minus v4 grep: reported recall -0.028 [-0.060, +0.002]; the index again gives no gain over grep.

## Reading

More data at the same recipe stopped paying. wayfinder-glimmer over v2b (701 to 1,383 examples) moved recall by about +0.03; v4 over wayfinder-glimmer (1,383 to 1,802 examples, 77% to 82% semantic) moves it by +0.013 on grep and -0.005 on semantic, both with CIs centered near zero. v4's grep-arm point estimate is the highest of any checkpoint (0.421), but the interval on the difference spans zero, so this does not establish v4 over wayfinder-glimmer. The small turn and token increases on grep are the only CI-excluding differences and they point against v4 (slightly longer trajectories). Because the same dev tasks were reused across v1 to v4 and checkpoint-to-checkpoint differences are now smaller than the paired CIs, further SFT-data scaling on this eval has little power left to discriminate; the remaining headroom to the teacher is small (-0.011 [-0.038, +0.014] on grep).

## Caveats

1. Dev set, 50 tasks, 18 repos, 2 seeds; CIs treat tasks as the sample. wayfinder-glimmer and v4 are single training runs (seed 0); run-to-run training variance is not measured.
2. The v4 optimizer ran 308 steps at global batch 8 packs, versus 478 steps at 4 packs for wayfinder-glimmer (see Training). Data volume and batch size changed together, so this is not a clean data-only ablation.
3. wayfinder-glimmer's retain-on grep rows come from a different launcher (the OPD agent's, one H100 server) and the wayfinder-glimmer retain-on semantic rows from 4 shards; v4 rows come from 8 shards per arm. Latency and wall time differ with server load and are not model properties. Quality metrics are insensitive to this.
4. spark_v3 was about 87% collected at conversion; the missing 13% are the hard tasks that stalled, so v4's extra data skews toward easier tasks. The `spark_v2` and `spark_v3` passes share task ids with `spark_v1`.
5. The 17 E2E-eval tasks were excluded, but other tasks from the same train-split repos were trained on, as in v2 and wayfinder-glimmer. `eval/e2e/tasks_v3.json` did not exist at conversion time; its tasks come from dev/test/eval_holdout repos, which never enter SFT.
6. Base and teacher columns are retain-off runs; the comparison criteria against base therefore mix protocols (the base model barely uses retained reasoning because it rarely reports before the cap, so this likely understates v4's margin if anything, but it was not measured).

## Spend and operations

Conversion about $0.3; train 57,726 GPU-s on 8x H200 at $4.54/h = about $73; merge about $1; v4 evals 16 H100 shards plus the wayfinder-glimmer semantic re-run (4 shards), about $24 list in total. Watcher `ops/sftv4_watch.py` armed before the first launch (auth canary, 330-minute kill age, prefix `wayfinder-sftv4`). Apps used: `wayfinder-sftv4-train8`, `-eval-v3sem-s*`, `-merge-v4`, `-eval-grep-s*`, `-eval-sem-s*`; a duplicate queued 4x H200 train app was stopped before start. No `wayfinder-sftv4*` apps remain live except the watcher process.
