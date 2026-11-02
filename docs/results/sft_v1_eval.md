# SFT v1 eval (LoRA r64 on Muse-Glimmer-30B), dev set

Run 2026-10-26/27. Adapter `wayfinder-data:/sft/v1` (157 examples, 1 epoch, 25 steps), evaluated on the baseline's 50 dev tasks x 2 seeds (`artifacts/teacher_test/stageB_tasks.json`), same prompt, convergence nudge, tools and sampling (T=0.85, top_p=0.95, max_turns 40, max_seconds 900) as `docs/glimmer_baseline.md`. Rows: `artifacts/sft_v1/sft_v1_{grep,sem}.jsonl`, `artifacts/sft_v1/rerun_*/`; metrics with paired task-bootstrap CIs: `artifacts/sft_v1/summary.json` (`scripts/score_sft_v1.py`). Entry points: `wayfinder/rollout/sft_eval_app.py` (copy of the baseline app, baseline untouched), `wayfinder/rollout/sft_merge_app.py`.

## Headline

v1 learned part of the behaviour but not enough to pass the pre-registered bar. On the grep-only arm, reported recall rose from 0.25 to 0.34 (+0.099, CI [+0.046, +0.156]), touched-file recall did not drop (+0.053), and the share of rollouts that end by calling `report` rose from 16% to 35% (59% once wall-capped rollouts are rerun with a long clock). It did not learn to stop early: turns p50 is 24 to 26 against a target of 20, and tokens per rollout are flat to higher (865k to 973k against base 821k, target -30%). PRD section 15 result: 2 of 5 criteria pass (the two recall criteria), 3 fail. The teacher's profile (95% report, p50 9.5 turns, 388k tokens) is far away.

## Setup and how the adapter was served

- vLLM 0.32.0, 1x H100 per arm, `--tool-call-parser muse_glimmer --reasoning-parser muse_glimmer --enable-auto-tool-choice --language-model-only --enable-prefix-caching`, max_model_len 65536, served name `wayfinder-v1`. Grep arm: `semantic=off`. Semantic arm: `semantic=auto` (99 of 99 completed rows had `semantic_search` offered; 71 of 99 called it, 3.2 calls per rollout on average).
- Adapter load, first attempt (runtime LoRA: `--enable-lora --max-lora-rank 64 --lora-modules wayfinder-v1=/data/sft/v1`): vLLM logged "Loaded new LoRA adapter: name 'wayfinder-v1'", `/v1/models` listed both names, a T=0 request with `model=wayfinder-v1` differed from `model=glimmer` on the same prompt (12-token logprob L1 distance 1.10, different reasoning text, 3 tool calls against 1), and tool calls parsed under LoRA. Evidence: `artifacts/sft_v1/lora_attempt/sft_v1.meta.json`.
- That run was aborted after 2 rollouts because per-turn latency was 50 to 250 s (base: about 5 s), roughly 10x slower, so rollouts would have hit the 900 s wall clock.
- All reported results therefore use the adapter merged into the base weights (`wayfinder-data:/sft/v1_merged`, `W + (alpha/r) B A`, 416 modules, scale 2.0, done in fp32 then cast to bf16). The merged server reproduces the runtime-LoRA first-token logprobs (L1 0.09 against 0.09 to 1.10 from base) and its tool calls parse. The merged model is what was scored; runtime LoRA was only used for the load check.
- Concurrency 32 on one merged server thrashed the KV and prefix cache (prefix hit rate 35% against 95% in the baseline, KV use 93%), so the final runs use one container per arm at concurrency 14. This is a deviation from the baseline's concurrency of 32 and is the main reason turn latency is high (below).

## Results (95% task-bootstrap CI; seeds averaged within task)

| Metric | Base | v1 grep-only | v1 semantic auto (exploratory) | Teacher Stage B (reference) |
|---|---|---|---|---|
| Rollouts scored | 100 | 99 (1 harness timeout) | 99 (1 harness timeout) | 98 |
| Recall, benchmark convention (`recall_seen_k5`) | 0.463 [0.368, 0.558] | 0.516 [0.424, 0.609] | 0.523 [0.437, 0.608] | 0.553 [0.465, 0.642] |
| Reported recall | 0.246 [0.177, 0.318] | 0.345 [0.265, 0.427] | 0.333 [0.254, 0.414] | 0.440 [0.354, 0.529] |
| All-gold (seen) | 0.15 [0.06, 0.25] | 0.17 [0.08, 0.27] | 0.17 [0.08, 0.28] | 0.19 [0.09, 0.31] |
| All-gold (reported) | 0.02 [0.00, 0.06] | 0.05 [0.00, 0.11] | 0.06 [0.01, 0.13] | 0.10 [0.03, 0.19] |
| Turns p50 / p90 | 40 / 40 | 24 / 40 | 21 / 40 | 9.5 / 18 |
| Hit turn cap (or budget exhausted) | 79% | 65% | 46% | 5% |
| Called `report` | 16% [8, 24] | 35% [24, 47] | 55% [44, 66] | 95% [89, 99] |
| Tokens per rollout (in + out) | 821k | 865k | 827k | 388k |
| Wall time p50 | 190 s | 802 s | 587 s | 278 s |
| Tool calls per turn | 1.14 | 3.50 | 3.74 | 3.05 |
| Model latency per turn (mean) | 5.6 s | 34.5 s | 33.2 s | 30.2 s |
| GPU $ per rollout | $0.011 | $0.062 | $0.059 | n/a (Baseten) |

"Called `report`" counts `report` tool calls in the transcript (for v1 it equals ended without budget exhaustion). Base `report` calls: 16 of 100.

Paired per-task deltas against base (95% task-bootstrap CI, 50 tasks):

| Metric | v1 grep minus base | v1 semantic minus base | v1 semantic minus v1 grep |
|---|---|---|---|
| Recall (seen k5) | +0.053 [+0.010, +0.099] | +0.060 [+0.020, +0.105] | +0.007 [-0.035, +0.046] |
| Reported recall | +0.099 [+0.046, +0.156] | +0.088 [+0.035, +0.143] | -0.011 [-0.049, +0.028] |
| All-gold (seen) | +0.02 [-0.03, +0.08] | +0.02 [0.00, +0.06] | n/a |
| Called `report` | +0.19 [+0.08, +0.30] | +0.39 [+0.27, +0.51] | +0.20 [+0.07, +0.33] |
| Turns | -14.3 [-16.7, -11.8] | -15.1 [-17.4, -12.8] | -0.9 [-3.5, +1.8] |
| Tokens per rollout | +43k [-58k, +143k] | +6k [-84k, +100k] | -38k [-146k, +75k] |

v1 grep against the teacher: reported recall -0.089 [-0.132, -0.047], turns +11.8, tokens +488k, report rate -60 points. Seen recall is statistically level with the teacher (-0.026 [-0.063, +0.011]).

## PRD section 15 criteria, grep-only arm vs base (pre-registered; paired by task)

Primary view: all completed rollouts, exactly as pre-registered. A rollout ended by the 900 s wall clock cannot call `report`, so this view mixes behaviour and serving speed; the two views below separate them.

| Criterion | Primary (99 rollouts) | Sensitivity: not wall-capped (54 rollouts, 37 tasks) | Turn-cap only, wall-capped pairs rerun at 3600 s (100 rollouts, 50 tasks) |
|---|---|---|---|
| Reported recall delta, CI lower bound > 0 (target >= +0.10) | +0.099 [+0.046, +0.156] PASS; target missed by 0.001 | +0.065 [+0.008, +0.129] PASS; target missed | +0.072 [+0.018, +0.129] PASS; target missed |
| Calls `report` before the cap, >= 80% | 35% FAIL | 58% FAIL | 59% FAIL |
| Turns p50 <= 20 | 24 FAIL | 18.5 PASS | 26.5 FAIL |
| Touched-file recall delta, CI lower bound > -0.10 | +0.053 [+0.010, +0.099] PASS | +0.013 [-0.030, +0.059] PASS | +0.044 [-0.001, +0.090] PASS |
| Tokens per rollout, down >= 30% | +5% (865k) FAIL | +3% (846k) FAIL | +18% (973k) FAIL |

Reading the three views. The primary view is the pre-registered one: 2 of 5 pass. The not-wall-capped view is biased, because it keeps only rollouts that finished quickly (it selects for short ones, so its p50 turns of 18.5 is not a fair median). The turn-cap-only view is the closest to the baseline protocol (the baseline never hit the wall clock): it still fails 3 of 5, so serving speed does not explain the failures. In it, 41% of rollouts still reach the 40-turn cap without reporting, p50 turns is 26.5, and tokens are 18% above base because rollouts that were cut off at 900 s now run on to the turn cap or to `report`.

Wall-capped rows: 45 of 99 grep rollouts ended by the 900 s clock in the primary run (mean 24 turns at the cut), 24 of 99 in the semantic arm. Rerun: 46 pairs (the 45 plus the 1 harness-timeout row) at `max_seconds=3600`, everything else identical, concurrency 16 per container, 2 containers; 1 still hit 3600 s. The rerun needed a runtime lift of scout's `Budget.max_seconds <= 900` validator in the eval app (scout source untouched). Other end states in the primary grep run: 15 turn cap, 5 context-budget exhausted, 34 reported.

## Per-turn behaviour and latency against baseline

v1 issues 3.5 tool calls per turn against base 1.14, close to the teacher's 3.05, so SFT did transfer the parallel-call style. Per-turn model latency is 34.5 s against 5.6 s. That is a serving effect, not a model effect: the teacher's latency per turn in its own run was 30 s with the same calls-per-turn, large parallel tool results inflate prefill, and our merged server ran 14 concurrent rollouts with a 35 to 60% prefix-cache hit rate (base: 95%). Do not read the latency or the GPU $ column as a property of v1; they depend on run concurrency and cache hit rate. At the baseline's 32-way concurrency the merged server thrashed, so this comparison could not be run like for like.

## Answers

1. Did it learn to stop and report? Partly. Report rate 16% to 35% (59% with an unconstrained clock), turns p50 40 to 24 to 26. It does not reach the teacher's 95% and p50 9.5. The semantic arm gets there faster (55% report, p50 21, wall-capped 24%), but that arm is exploratory.
2. Did exploration survive? Yes. Touched-file recall is up, not down (+0.05 [+0.01, +0.10]); reported recall +0.10, still 0.09 short of the teacher.
3. Semantic index: no reliable gain over grep-only for v1 on recall (reported recall -0.011 [-0.049, +0.028]); it does raise the share that report (+20 points [+7, +33]), which is partly a speed effect (fewer wall-capped rollouts).

## Caveats

1. Concurrency and cache: runs used concurrency 14 per container instead of 32 and per-turn latency was 6x base, so 45% of grep rollouts hit the 900 s clock. The rerun view addresses this for the grep arm only; the semantic arm has no rerun (24 wall-capped rows).
2. Merged weights, not runtime LoRA, were scored (runtime LoRA was too slow). bf16 merge adds rounding; the T=0 logprobs matched the LoRA run within L1 0.09 over 12 tokens.
3. The grep rerun replaces the wall-capped pairs only. Non-capped pairs keep their primary-run rows, so the rerun view mixes two server runs.
4. n = 50 tasks, 18 repos, 2 seeds; CIs treat tasks as the sample. 1 harness TimeoutError per arm is excluded from scoring.
5. Dev set only; v1 trained on 157 examples for 25 steps, so the "did not converge" branch of the PRD decision rule applies on stopping (more data, 2 epochs). Reported recall improved but the +0.10 target was missed by 0.001 in the primary view, and by more in the other two, which is "converges but reported recall flat" on the margin.
6. Row-level `cost_usd` uses list prices with no cache accounting and is not comparable; use the GPU $ row.

## Spend

Zero `wayfinder-sft-eval*` apps and zero containers at the end.
