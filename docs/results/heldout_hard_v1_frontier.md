# Held-out hard v1: wayfinder-glimmer and A1 against Sonnet 5.5 and GPT-6.1 Sol

This document adds two frontier models to the one-time evaluation in `docs/heldout_hard_v1_eval.md`. Sonnet 5.5 (reasoning effort high) and GPT-6.1 Sol (default reasoning) were run through the LLM gateway inside scout's own explore loop on `data/heldout_hard_v1.json` (200 tasks, 47 repositories), so their rows compare directly with the wayfinder-glimmer and A1 rows. The author approved the run on 2026-10-30.

## Protocol

Only the model endpoint changed. The loop, product prompt, convergence nudge, tools (grep, glob, read_file, report, plus semantic_search in the sem arm on the same turbopuffer namespaces), 40-turn cap, 3,600 s cap, backfill-off setting, report format and scorer are those of `wayfinder/rollout/test_eval_app.py` (wayfinder-glimmer) and `a1_eval_app.py` (A1). Seeds 0 and 1, arms grep and sem, so 800 rollouts per model. The runner is `wayfinder/rollout/frontier_eval.py` with the the LLM gateway client in `wayfinder/rollout/frontier_chat.py`; it ran on the laptop because rollouts are API-bound and grep is CPU only. Nothing was run on Modal.

The two models are driven through their native wire formats, because the OpenAI chat shape does not carry their reasoning. Sonnet uses the gateway's Anthropic `/v1/messages` route with adaptive thinking, `output_config.effort = "high"` and summarized thinking display; each assistant turn's native blocks (thinking with signature, text, tool_use) are replayed verbatim, so thinking is retained across turns. Anthropic fixes temperature at 1 when thinking is on, so no temperature, top_p or top_k is sent. GPT-6.1 Sol uses `/openai/v1/responses` statelessly (`store: false`, `include: reasoning.encrypted_content`) and replays its encrypted reasoning items; the response reports effort medium and provider-default sampling (temperature 1.0, top_p 0.98). Neither route supports a seed, so the seed value labels the row but does not control sampling. Both use a 16,000-token per-call output limit including reasoning, against 8,192 for the open models. The exact settings are in each row's `meta.settings`.

Rows use the wayfinder-glimmer/A1 schema, with `model`, `weights` (`gateway:<model id>`), `usage`, `cost_usd` and `meta` added or redefined for the frontier rows; `usage.billed` holds the four billed token buckets. `scripts/score_heldout_hard_4way.py` is a copy of the wayfinder-glimmer/A1 scorer with the two models and the extra contrasts added, and writes `artifacts/heldout_hard/summary_4way.json` and `score_4way.txt`. Metrics and bootstrap settings are unchanged: the task is the unit, and intervals are 95% percentile intervals from 10,000 task-bootstrap and repository-cluster resamples. The wayfinder-glimmer and A1 numbers reproduce `summary.json` exactly.

Calibration came first: 5 tasks (two MongoDB-ecosystem), seed 0, both arms, both models, 20 rollouts for $2.07. The full run then cost $142.71 including calibration.

## Validation

All eight frontier and open arms returned 400 of 400 rows with no errors, both seeds covering all 200 tasks, a single model id, and `meta.settings` and billed cost present on every frontier row. Turns ranged from 3 to 23 for Sonnet and 5 to 14 for GPT; grep-arm wall time per rollout had a median of 77 s (Sonnet) and 104 s (GPT) with a maximum of 364 s, and there were zero 429s. Every frontier rollout ended by calling report, against 47% and 53% for A1 in the grep and sem arms. Sonnet returned thinking in 99% of grep rollouts; GPT's reasoning is encrypted, so its transcripts carry none. In the sem arm semantic_search was offered in all 400 rollouts of each model and called in 252 Sonnet rollouts (63%) and 380 GPT rollouts (95%).

Parse health differs by model. GPT made no malformed calls. Sonnet had 122 malformed calls over 6,185 grep tool calls (2%), almost all of them report calls missing the `why` field of a later result, or a grep `context` above 5; the loop returns these as tool errors and Sonnet corrects on the next turn, and no rollout stopped on three consecutive malformed calls. The malformed turns count toward its turns and cost.

## Result

Both frontier models beat wayfinder-glimmer on reported recall in both arms, and every interval excludes zero. GPT-6.1 Sol leads on recall and Sonnet 5.5 on precision. wayfinder-glimmer stays level with GPT on precision in the grep arm and is ahead of A1 by a wide margin everywhere.

Reported recall, recall@5, precision, files reported, turns and billed dollars per task, mean over 200 tasks with seeds averaged within task (95% task-bootstrap CIs for recall in brackets):

| Arm | Model | Reported recall | Recall@5 | Precision | n reported | Turns | Tokens per task | Cost per task |
|---|---|---|---|---|---|---|---|---|
| grep | wayfinder-glimmer | 0.452 [0.412, 0.491] | 0.360 | 0.415 | 7.6 | 9.9 | 268k | $0.043 open-price equiv. |
| grep | A1 | 0.313 [0.280, 0.348] | 0.247 | 0.339 | 7.8 | 34.1 | 764k | $0.113 open-price equiv. |
| grep | Sonnet 5.5 high | 0.529 [0.491, 0.567] | 0.427 | 0.491 | 7.5 | 7.5 | 136k | $0.120 billed |
| grep | GPT-6.1 Sol | 0.580 [0.541, 0.619] | 0.455 | 0.434 | 9.0 | 7.8 | 95k | $0.062 billed |
| sem | wayfinder-glimmer | 0.446 [0.406, 0.484] | 0.350 | 0.413 | 7.7 | 9.4 | 254k | $0.041 open-price equiv. |
| sem | A1 | 0.352 [0.319, 0.387] | 0.283 | 0.378 | 7.6 | 33.5 | 746k | $0.110 open-price equiv. |
| sem | Sonnet 5.5 high | 0.518 [0.480, 0.556] | 0.414 | 0.488 | 7.5 | 7.1 | 125k | $0.112 billed |
| sem | GPT-6.1 Sol | 0.576 [0.538, 0.614] | 0.448 | 0.441 | 8.9 | 7.1 | 84k | $0.058 billed |

Tokens per task are input plus output including cached input. The cost column mixes kinds only in this summary row for readability; the next section defines each and they must not be added or ranked as one quantity.

Contrasts on reported recall with task CI and repository-cluster CI (positive means the first model is better):

| Contrast | Grep | Sem |
|---|---|---|
| wayfinder-glimmer minus A1 | +0.139 [+0.111, +0.166] / [+0.116, +0.164] | +0.093 [+0.068, +0.120] / [+0.067, +0.119] |
| wayfinder-glimmer minus Sonnet | -0.078 [-0.104, -0.052] / [-0.108, -0.047] | -0.072 [-0.100, -0.044] / [-0.101, -0.044] |
| wayfinder-glimmer minus GPT | -0.128 [-0.157, -0.099] / [-0.156, -0.101] | -0.130 [-0.160, -0.101] / [-0.156, -0.106] |
| A1 minus Sonnet | -0.216 [-0.247, -0.186] / [-0.254, -0.180] | -0.166 [-0.194, -0.138] / [-0.197, -0.134] |
| A1 minus GPT | -0.267 [-0.300, -0.234] / [-0.302, -0.233] | -0.223 [-0.255, -0.192] / [-0.258, -0.190] |
| Sonnet minus GPT | -0.051 [-0.074, -0.027] / [-0.073, -0.029] | -0.058 [-0.082, -0.034] / [-0.082, -0.035] |

On recall@5 the gaps are wayfinder-glimmer minus Sonnet -0.067 [-0.089, -0.044] (grep) and -0.064 [-0.089, -0.039] (sem), wayfinder-glimmer minus GPT -0.094 [-0.122, -0.067] and -0.098 [-0.125, -0.072], and Sonnet minus GPT -0.027 [-0.050, -0.006] and -0.035 [-0.057, -0.013]. On precision wayfinder-glimmer trails Sonnet by 0.075 [0.050, 0.101] in grep and 0.076 in sem, trails GPT by 0.019 [-0.006, +0.044] in grep (level) and 0.028 in sem, and Sonnet leads GPT by 0.056 [0.036, 0.077] and 0.048. GPT reports 9.0 files against 7.5 to 7.7 for the others, which raises its recall and lowers its precision; wayfinder-glimmer minus GPT in files reported is -1.4 [-1.7, -1.1]. Recall@5 uses report order and removes that advantage for GPT only partly: GPT still leads wayfinder-glimmer by 0.094 at k=5.

In the grep arm wayfinder-glimmer closes 64% of the reported-recall gap between A1 and Sonnet (0.139 of 0.216) and 52% of the gap to GPT (0.139 of 0.267). It does this in 9.9 turns against 7.5 and 7.8 for the frontier models, and at 268k tokens per task against 136k (Sonnet) and 95k (GPT), so wayfinder-glimmer takes about 2 and 2.8 times their tokens.

Semantic search again does not help any model with a meaningful recall gain except A1. Sem minus grep on reported recall is -0.011 [-0.027, +0.004] for Sonnet, -0.004 [-0.018, +0.010] for GPT and -0.006 [-0.022, +0.010] for wayfinder-glimmer, each with fewer turns (0.4 to 0.7) and fewer tokens (11k to 14k); A1 gains +0.039 [+0.017, +0.064]. Even GPT, which calls semantic_search in 95% of sem rollouts, gains nothing in recall. This repeats the dev-set and wayfinder-glimmer/A1 pattern: the index saves cost, not recall, on this set.

## Breakdowns

Reported recall, grep arm, with recall@5 contrasts for the two pairs that matter. Intervals are task-bootstrap and groups are small, so these are descriptive. Groups with fewer than 13 tasks (Ruby, and two categories with 3 tasks) are in `summary_4way.json` only.

| Group | Tasks / repos | wayfinder-glimmer | A1 | Sonnet | GPT | wayfinder-glimmer minus Sonnet, recall@5 | wayfinder-glimmer minus GPT, recall@5 |
|---|---|---|---|---|---|---|---|
| MongoDB ecosystem | 24 / 7 | 0.420 | 0.305 | 0.543 | 0.562 | -0.076 [-0.127, -0.029] | -0.079 [-0.120, -0.043] |
| TS/JS | 50 / 11 | 0.446 | 0.328 | 0.491 | 0.552 | -0.036 [-0.080, +0.006] | -0.083 [-0.130, -0.042] |
| Java/Kotlin | 28 / 8 | 0.426 | 0.270 | 0.567 | 0.590 | -0.111 [-0.184, -0.035] | -0.109 [-0.212, -0.006] |
| C# | 18 / 4 | 0.544 | 0.371 | 0.582 | 0.612 | -0.035 [-0.086, +0.015] | -0.019 [-0.089, +0.048] |
| Go | 18 / 5 | 0.465 | 0.325 | 0.545 | 0.619 | -0.066 [-0.125, -0.013] | -0.073 [-0.144, +0.015] |
| Rust | 16 / 3 | 0.468 | 0.344 | 0.570 | 0.568 | -0.067 [-0.118, -0.028] | -0.036 [-0.091, +0.024] |
| Swift | 16 / 3 | 0.465 | 0.358 | 0.581 | 0.593 | -0.120 [-0.205, -0.045] | -0.122 [-0.223, -0.029] |
| Other | 18 / 5 | 0.452 | 0.249 | 0.448 | 0.598 | -0.027 [-0.129, +0.083] | -0.182 [-0.298, -0.070] |
| C/C++ | 15 / 3 | 0.498 | 0.346 | 0.584 | 0.657 | -0.088 [-0.174, -0.008] | -0.126 [-0.218, -0.045] |
| Python | 13 / 5 | 0.356 | 0.248 | 0.476 | 0.477 | -0.053 [-0.144, +0.025] | -0.070 [-0.168, +0.013] |
| New features | 119 / 43 | 0.418 | 0.290 | 0.517 | 0.566 | -0.077 [-0.109, -0.046] | -0.117 [-0.158, -0.079] |
| Tech debt | 36 / 21 | 0.525 | 0.369 | 0.611 | 0.632 | -0.075 [-0.135, -0.020] | -0.054 [-0.106, -0.007] |
| Refactoring | 25 / 18 | 0.523 | 0.367 | 0.534 | 0.632 | -0.027 [-0.072, +0.018] | -0.080 [-0.124, -0.040] |
| Modernize | 17 / 13 | 0.443 | 0.307 | 0.451 | 0.501 | -0.014 [-0.058, +0.024] | -0.032 [-0.093, +0.022] |

The MongoDB-ecosystem result matches the aggregate: both frontier models lead wayfinder-glimmer by about 0.08 recall@5, and wayfinder-glimmer leads A1 by 0.089. wayfinder-glimmer is level with Sonnet (recall@5 interval including zero) in TS/JS, C#, Other, Python and the refactoring and modernize categories and is furthest behind on Swift and Java/Kotlin. The gap to GPT is largest on Other languages (0.182) and C/C++. The per-group semantic-arm numbers and all 26 repository types are in `artifacts/heldout_hard/summary_4way.json`.

## Cost per task

Three costs are reported and must not be mixed. All are dollars per task, mean over tasks of the seed-averaged rollout.

Billed cost applies to Sonnet and GPT only: the LLM gateway tokens times a rate table (fresh input, cache read, cache write, output), because the LLM gateway returns no price. Sonnet is $2.00 / $0.20 / $2.50 / $10.00 per million, from the the LLM gateway catalog figures in `benchmark-repo/scripts/score_frontier.py`. GPT-6.1 Sol is $1.48 / $0.074 / $1.85 / $7.40, an assumption (the list price scaled by the 0.74 the LLM gateway factor of gpt-6-sol) that has not been confirmed against the catalog, so the GPT figures could be off by that scaling. Cache reads were 82% of Sonnet's input tokens and 75% of GPT's in seed 0, with cache writes 18% and 24%, which is why billed cost is far below fresh-token cost.

Open-price equivalent applies to wayfinder-glimmer and A1: their tokens at $0.14 per million input and $1.00 per million output, the DeepInfra Qwen3.5-35B-A3B list price used by `scripts/score_value_props.py` and the e2e cost tables (wayfinder-glimmer is priced the same; it is a 30B-A3B). The wayfinder-glimmer and A1 rows record zero cached tokens, so every input token is priced fresh and these figures are upper bounds. This is what an API at an open-model price would charge, not what the evaluation cost.

GPU cost applies to wayfinder-glimmer and A1 only: Modal H100 list price ($3.95 per hour) of each arm's container seconds from the meta files, including vLLM load, divided by 400 rollouts. It depends on how many rollouts shared a server, so it is a serving-cost estimate for this run rather than a per-token price.

| Arm | Model | Billed (the LLM gateway) | Open-price equivalent | GPU (Modal H100, per rollout) |
|---|---|---|---|---|
| grep | wayfinder-glimmer | n/a | $0.043 | $0.029 (arm total $11.77) |
| grep | A1 | n/a | $0.113 | $0.012 (arm total $4.87) |
| grep | Sonnet 5.5 high | $0.120 (arm total $47.92) | n/a | n/a |
| grep | GPT-6.1 Sol | $0.062 (arm total $24.73) | n/a | n/a |
| sem | wayfinder-glimmer | n/a | $0.041 | $0.028 (arm total $11.26) |
| sem | A1 | n/a | $0.110 | $0.012 (arm total $4.79) |
| sem | Sonnet 5.5 high | $0.112 (arm total $44.63) | n/a | n/a |
| sem | GPT-6.1 Sol | $0.058 (arm total $23.36) | n/a | n/a |

At like-for-like token pricing the self-hosted-class models are cheaper per task than the frontier models: wayfinder-glimmer at an open price costs about $0.043 against $0.062 for GPT and $0.120 for Sonnet in the grep arm, while A1 at the same open price ($0.113) costs more than GPT because it uses 8 times GPT's tokens. That comparison pairs two different price lists (open-model list against the LLM gateway) and says what a task costs under each model's own price, not that wayfinder-glimmer is cheaper at equal margins. The A1 GPU figure being lower than wayfinder-glimmer's reflects A1 shards sharing a server across 25 concurrent rollouts, and should not be read as A1 being cheaper to serve.

## Spend

Modal: $0, since the run was on the laptop and no app was launched. A `--kind gateway` flag on `record` writes it.

## Caveats

The two frontier models were not tuned to the harness, and Sonnet's malformed report calls cost it a few turns that a stricter tool description might have removed. The output limit is 16,000 tokens for the frontier models and 8,192 for the open ones; no frontier rollout hit it (every rollout ended by report), so it does not change the rows. Neither frontier route supports a seed, so the two seeds are independent samples rather than matched ones. Sonnet's summarized thinking is a summary of its reasoning and GPT's reasoning is encrypted, so the stored transcripts understate reasoning content; billed output tokens include all of it. The GPT rate card is assumed. No call finished on the length limit. Provider-side prompt caching differs between the two models, which affects billed cost but not retrieval. The set is the same one that is now spent for wayfinder-glimmer and A1 (nothing was selected on it), and the caveats in `heldout_hard_v1_eval.md` about question wording, mislabelled categories and the near-absence of security and schema tasks apply here unchanged.
