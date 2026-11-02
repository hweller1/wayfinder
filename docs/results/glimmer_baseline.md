# Prompted Glimmer baseline ("before training")

Run 2026-10-26. Muse-Glimmer-30B, untrained, on the same 50 dev tasks (`artifacts/teacher_test/stageB_tasks.json`, 18 repos) as the Spark teacher's Stage B, 2 seeds each (100 rollouts, 0 errors). Rows: `artifacts/baseline/glimmer_dev.jsonl`; summary with task-bootstrap CIs: `artifacts/baseline/glimmer_dev_summary.json` (`scripts/score_glimmer_baseline.py`); server and run metadata: `artifacts/baseline/glimmer_dev.meta.json`.

## Setup

- Serving: vLLM 0.32.0, 1x H100, bf16, `--tool-call-parser muse_glimmer --reasoning-parser muse_glimmer --enable-auto-tool-choice --language-model-only --enable-prefix-caching`, max_model_len 65536 (started first try; KV cache 761k tokens), 64 seqs.
- Harness: scout product profile via the shared `run_rollout` (not edited). Entry point is new, `wayfinder/rollout/glimmer_baseline_app.py`, which runs vLLM and the rollouts in the same Modal container because the laptop cannot resolve `*.modal.host`.
- Same as the teacher run: scout product prompt and tools, convergence nudge appended to the system prompt, T=0.85, top_p=0.95, max_turns 40, max_seconds 900, grep-only (`semantic=false`: scout's loop has no semantic backend wired in the harness). Concurrency 32.
- Nudge text used, taken from the teacher run's working-tree code (`run.TEACHER_NUDGE`, not yet committed when this ran; it equals the fallback text): "Report as soon as the evidence covers the brief; do not exhaust the turn budget." It is recorded in every row.
- Reasoning strength: the Glimmer template defaults to `high`; not overridden.

## Results (50 tasks x 2 seeds, 95% task-bootstrap CI)

| Metric | Glimmer base |
|---|---|
| File recall, benchmark convention (`recall_seen_k5`) | 0.463 [0.368, 0.559] |
| All-gold-found, same convention | 0.15 [0.06, 0.25] |
| File recall, reported by `report` (what a parent receives) | 0.246 [0.177, 0.318] |
| All-gold in the report | 0.02 [0.00, 0.06] |
| Turns p50 / mean | 40 / 37.4 |
| Hit the 40-turn cap | 79% |
| Tokens per rollout (in + out) | 821k (818k in, 3.6k out) |
| Wall time p50 | 190 s |
| GPU $ per rollout | $0.011 |
| Cost at Baseten list price, as the teacher runs report it | $1.04 (see caveat 3) |

Seed 0 and 1 agree (recall 0.447 / 0.480, all-gold 0.14 / 0.16).

## Comparison

Benchmark per-task recall from `data/task_filter.json` on the same 50 tasks, paired difference with task-bootstrap CI. Benchmark arms: b2 = grep, c = index, g = grep + index.

| Arm (benchmark) | Recall on these tasks | Glimmer minus arm |
|---|---|---|
| A1-35B b2 (grep) | 0.182 | +0.282 [+0.212, +0.356] |
| Sonnet 5.5 high b2 (grep) | 0.336 | +0.127 [+0.067, +0.193] |
| A1-35B c (index) | 0.432 | +0.031 [-0.012, +0.078] |
| Sonnet 5.5 high c (index) | 0.424 | +0.040 [-0.008, +0.086] |
| A1-35B g (grep + index) | 0.473 | -0.010 [-0.074, +0.056] |
| Spark teacher, self-hosted Stage B (98 rows so far) | 0.553 | -0.080 [-0.121, -0.040] (49 tasks) |

The teacher's p50 is 9.5 turns and 0.39M tokens per rollout, against Glimmer's 40 turns and 0.82M. Training headroom is therefore in stopping, not in raw search.

## Caveats

1. The benchmark rows used their own harness and prompt (neutral prompt, grep k=5, 63-turn limit, no `report` tool). Scout's grep shows up to 200 files per call. `recall_seen_k5` approximates the benchmark by counting only the first 5 files of each grep/glob result plus opened files, but the comparison with b2/c/g is indicative, not like for like. Do not read "Glimmer beats Sonnet + grep" off this table.
2. Glimmer mostly fails to converge. 79% of rollouts ran out of turns without calling `report` (they fall back to scout's best-candidates list), which is why reported recall (0.25) is half of seen recall (0.46) and only 2% of reports hold all gold. Rollouts that did report (21) had p50 29 turns. The nudge did not fix this. The base model reports 16 times in 100 rollouts, as measured by `report` tool calls. 18 tool results were malformed-argument errors.
3. Cost. vLLM did not return `cached_tokens`, so row-level `cost_usd` assumes no caching ($1.04). The server's own counters show 95.3% of prompt tokens were prefix-cache hits (77.9M of 81.8M), which puts list-price cost near $0.18. The real spend was one H100 for 983 s, about $1.08 for all 100 rollouts. Compare with the teacher's figures only on tokens and turns.
4. Grep-only. The product's semantic search is not in the harness yet, so this is the grep arm of the product profile. The same flag setting applies to the teacher run.
5. Dev set only (50 of 153 dev tasks, 18 repos, chosen by the teacher plan). Task-bootstrap CIs treat the 50 tasks as the sample; seeds are averaged within task first.
6. `semantic_search` was called 64 times although it was not offered in the tool list; the harness returned an error to the model for those calls each time.
