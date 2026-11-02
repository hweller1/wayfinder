# End-to-end coding eval: does the search subagent change whether a coder passes held-out tests?

Run 2026-10-26. Code: `eval/e2e/` (copied from benchmark-repo `scripts/e2e/`, which is untouched). Records: `eval/e2e/runs/issue/<condition>/` (coder runs, with transcripts and patches), `eval/e2e/search/issue/<condition>/` (the search subagent output each coder was handed), raw search rows with full transcripts in `artifacts/e2e/search_*.jsonl` (gitignored), analysis `eval/e2e/analyze.py` -> `eval/e2e/results.json`.

## Setup

Same coder and harness as E2E round 3: Opus 5.5 medium through an LLM API gateway, in a Modal sandbox with the network blocked. The coder has no search tools. `read_file` opens only the files the search subagent returned, and the held-out tests are absent from the sandbox and asserted so. The diff (test paths dropped) is scored in a separate sandbox against the held-out tests (pass = every fail-to-pass test passes and no regression), on the shared cache volume `asa-e2e-cache-v2`. The 17 round-3 headroom tasks (oracle passes and no-search fails on the seed-0 screen) are used with seeds 1 and 2, 34 runs per condition. Seed 0 is not used.

| Condition | What the coder is handed |
|---|---|
| none (floor) | nothing; the round-3 records are reused |
| oracle (ceiling) | the gold source files; the round-3 records are reused |
| glimmer_base | output of untrained Muse-Glimmer-30B (vLLM 0.32.0, 1x H100 in the same Modal container as the rollouts) running scout's explore loop |
| a1_35b_scout | output of Agents-A1-35B (the deployed Modal endpoint) through scout's explore loop |
| spark_teacher | output of Muse Spark Open (`meta-models/muse-spark-open` on Baseten, reasoning effort high) through scout's explore loop |

The search conditions use the same loop settings as the teacher Stage B and the Glimmer baseline: scout product prompt and tools (grep, glob, read_file, report; no semantic search, so `semantic=false`), the convergence nudge, max 40 turns, 900 s, T 0.85, top_p 0.95 (A1 uses its recommended top_k 20 and presence penalty 1.1). The brief is the round-3 issue text. The coder is given the subagent's summary and reported locations, plus excerpts of the first 5 reported files (the lines the subagent reported), and may open any file the subagent reported. Rollouts ran inside Modal because the laptop cannot resolve `*.modal.host`.

"none" and "oracle" were not rerun: the records are the round-3 records for the same 17 tasks and seeds (68 files), produced by the same coder code. The only change to `run_coder.py` is the condition table, paths, and an inlined Opus price function.

## Pass rate (17 tasks x seeds 1-2, 95% task-bootstrap CI)

| Condition | Pass | Rate | 95% CI | Coder $/run | Search $/run |
|---|---|---|---|---|---|
| none | 8/34 | 0.24 | [0.06, 0.44] | 0.11 | 0 |
| glimmer_base | 22/34 | 0.65 | [0.44, 0.85] | 0.28 | 0.018 |
| a1_35b_scout | 20/34 | 0.59 | [0.38, 0.79] | 0.31 | 0.032 |
| spark_teacher | 24/34 | 0.71 | [0.50, 0.88] | 0.36 | 0.914 |
| oracle | 27/34 | 0.79 | [0.62, 0.94] | 0.44 | 0 |

Paired contrasts (per-task mean over the two seeds, task bootstrap, 17 tasks):

| Contrast | Diff | 95% CI |
|---|---|---|
| glimmer_base - none | +0.41 | [+0.18, +0.65] |
| a1_35b_scout - none | +0.35 | [+0.12, +0.59] |
| spark_teacher - none | +0.47 | [+0.26, +0.68] |
| a1_35b_scout - glimmer_base | -0.06 | [-0.18, +0.06] |
| spark_teacher - glimmer_base | +0.06 | [+0.00, +0.15] |
| spark_teacher - a1_35b_scout | +0.12 | [+0.03, +0.24] |
| oracle - glimmer_base | +0.15 | [-0.03, +0.35] |
| oracle - none | +0.56 | [+0.26, +0.82] |

Every search subagent, including untrained Glimmer, lifts the coder well above no search, and none is distinguishable from the oracle ceiling at this sample size. Between subagents the only difference whose interval excludes zero is Spark over A1-35B (+0.12), and it is one of eight unadjusted contrasts. Spark over Glimmer sits on the edge. Search quality has little room to show at this n because the oracle is only 0.79 and 6 of 17 tasks are saturated (all conditions pass) or hopeless (no condition passes) once any files are given.

## Search metrics vs pass

Recall is file-level against the gold source files (test files and files created by the change excluded, as in round 3). "Reported" is the set the coder is handed (at most 12 files); "seen k5" is the benchmark-style view (opened files plus the first 5 files of each grep or glob result).

| Condition | Recall (reported) | All-gold (reported) | All-gold (seen k5) | Turns, mean | Hit 40-turn cap | Files handed, mean | Search $/run |
|---|---|---|---|---|---|---|---|
| glimmer_base | 0.68 [0.54, 0.82] | 0.44 [0.21, 0.68] | 0.50 [0.26, 0.74] | 36.1 | 71% | 6.5 | 0.018 |
| a1_35b_scout | 0.64 [0.51, 0.76] | 0.38 [0.21, 0.59] | 0.50 [0.29, 0.71] | 30.9 | 32% | 5.3 | 0.032 |
| spark_teacher | 0.78 [0.67, 0.88] | 0.53 [0.29, 0.76] | 0.74 [0.53, 0.91] | 32.8 | 68% | 6.4 | 0.914 |

Pooled over the three search conditions (102 runs), the coder passed 80% (37/46) when the report contained every gold file and 52% (29/56) when it missed one. It passed 67% (65/97) when any gold file was reported and 20% (1/5) when none was. The ordering of search recall (Spark > Glimmer > A1) roughly matches the ordering of pass rate (Spark > Glimmer > A1), but with three points and wide intervals this is not a test. Per-run rows are at the end.

Glimmer and Spark hit the turn cap in about 70% of rollouts and A1 in 32%; for capped runs scout returns its best-candidates list instead of a report, and the coder is handed that. Even so, capped Glimmer reports carry recall 0.68 here, much higher than the 0.25 "reported" recall in `glimmer_baseline.md` (different tasks and briefs: the 17 oracle-solvable tasks with issue-text briefs, and gold here excludes test files and new files; the cause of the gap was not isolated).

Pass rate per task (passes out of the two seeds):

| task | none | glimmer_base | a1_35b_scout | spark_teacher | oracle |
|---|---|---|---|---|---|
| temporal_02ec16 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 |
| mcp-client-for-ollama_cfa805 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 |
| reqwest_0aca66 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| mashumaro_43fe69 | 0/2 | 2/2 | 1/2 | 2/2 | 2/2 |
| google_workspace_mcp_778a00 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| gh-aw-mcpg_257862 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| OpenBiliClaw_69c829 | 0/2 | 0/2 | 1/2 | 1/2 | 2/2 |
| go-micro_017552 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| claude-agent-sdk-python_e9449b | 1/2 | 2/2 | 1/2 | 2/2 | 1/2 |
| mashumaro_56d0d2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| temporal_67689b | 2/2 | 1/2 | 1/2 | 2/2 | 0/2 |
| celestia-app_c4d20b | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| cplt_0885a0 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| helm_aa40c6 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| temporal_7dc679 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 |
| code-graph-rag_83ce76 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| mcp-client-for-ollama_3e98bf | 0/2 | 1/2 | 0/2 | 1/2 | 2/2 |

## Cost

Recorded $ per run, mean over 34 runs. Coder is Opus at list price ($4 in, $20 out, cache read $0.20, write $5 per M; not confirmed against the the LLM gateway catalog, as in round 3). Sandbox CPU is added to the totals in `results.json` at roughly $0.03 per run. Search cost differs in kind by condition: Spark is the Baseten list price of its own tokens ($1.25 / $4.25 / $0.15 per M), so it overstates a self-hosted cost (its server did not report cached tokens either, see `teacher_test.md`); Glimmer is the H100 container seconds (vLLM load 216 s plus rollouts, 563 s in all, $0.62 for 34 runs) split evenly; A1 is the launcher wall time on the shared warm H100 endpoint at the same H100 rate, split evenly ($1.08 for 34 runs). Spark's coder cost is also higher ($0.36 vs $0.28-0.31 per run); I did not investigate why.

The none and oracle records cost nothing new.

## Caveats

- n is small: 17 tasks, 2 seeds, 34 runs per condition. The task is the unit of resampling; seeds are averaged within task first. Intervals are wide, eight contrasts are unadjusted, and only the "search vs none" results are robust.
- Selection. The 17 tasks were chosen from a seed-0 screen where oracle passed and no-search failed, so the floor is low by construction and every search condition has headroom. The result says "when the coder lacks the files, handing it files helps", not what an average task would show. The none and oracle rows are repeated from round 3 and share this selection.
- Not the round-3 search harness. Round 3 ran the benchmark's solver loop (grep k=5 or Atlas index with 512-token snippets, neutral prompt). Here the subagents run scout's loop with a report tool and line ranges, grep-only, with the convergence nudge, so these conditions are not comparable with the round-3 35b_grep or 35b_index rows beyond rough scale. None of the conditions here uses the vector index, which is the main thing Wayfinder is meant to add.
- Different endpoints and samplers across conditions (A1 with top_k 20 and presence penalty 1.1, Spark with reasoning effort high, Glimmer at its default high reasoning), so the comparison is of three "as served" subagents, not a clean model ablation. Spark rollouts queued behind Baseten's shared rate limit (wall times in the rows are not model time).
- Five A1 rollouts failed on the first pass (Modal web endpoint answered long turns with a 303 redirect that the client did not follow, reported by scout as INVALID_ARGUMENT); `search_app.py` now follows redirects and those five were rerun, so the A1 records are complete (34/34). Two of the 34 Spark records come from a first attempt that was stopped before it finished; the settings were identical apart from the token-gate level.
- Task overlap with training. Of the 17 tasks, 10 are in the `train` split of `data/splits.json`, 3 in `dev`, 3 in `eval_holdout`, 1 in `test`. PRD section 5 reserves the E2E gated tasks for rubric calibration and the final end-to-end eval, so the 10 train-split tasks (and the other 7) must be excluded from SFT and GRPO data before this eval is rerun on a trained model, or the final number will be contaminated.
- The Glimmer row is the untrained baseline. The question for Wayfinder is whether training moves the Glimmer row toward or beyond Spark, and the interval around the current gap (about 0.06) says a trained model cannot be distinguished from base on this task set unless it moves recall substantially, because the ceiling (oracle) is only 0.15 above Glimmer.
- Search metrics vs pass are correlational: pooled runs, tasks differ in difficulty, and a coder can pass without the gold files (the none condition passes 24%).
- Coder price and the oracle/none records are round-3 values; coder caps (60 tool calls or 400k prompt tokens) counted as in round 3.

## Per-run search metrics and outcome

| task | cond | seed | pass | recall_rep | all-gold rep | turns | files | stop |
|---|---|---|---|---|---|---|---|---|
| temporal-d20b02 | glimmer_base | 1 | 0 | 0.14 | 0 | 40 | 7 | cap |
| temporal-d20b02 | glimmer_base | 2 | 0 | 0.57 | 0 | 40 | 12 | cap |
| mcp-client-for-ollama-8bfdcf | glimmer_base | 1 | 0 | 0.67 | 1 | 31 | 3 | report |
| mcp-client-for-ollama-8bfdcf | glimmer_base | 2 | 0 | 0.67 | 1 | 40 | 5 | cap |
| reqwest | glimmer_base | 1 | 1 | 1.00 | 1 | 40 | 7 | cap |
| reqwest | glimmer_base | 2 | 1 | 1.00 | 1 | 40 | 5 | cap |
| mashumaro | glimmer_base | 1 | 1 | 1.00 | 1 | 40 | 9 | cap |
| mashumaro | glimmer_base | 2 | 1 | 1.00 | 1 | 40 | 8 | cap |
| google_workspace_mcp | glimmer_base | 1 | 1 | 1.00 | 1 | 17 | 3 | report |
| google_workspace_mcp | glimmer_base | 2 | 1 | 1.00 | 1 | 27 | 3 | report |
| gh-aw-mcpg-774b25 | glimmer_base | 1 | 1 | 0.60 | 0 | 40 | 12 | cap |
| gh-aw-mcpg-774b25 | glimmer_base | 2 | 1 | 0.60 | 0 | 40 | 9 | cap |
| openbiliclaw-411469 | glimmer_base | 1 | 0 | 0.33 | 0 | 40 | 11 | cap |
| openbiliclaw-411469 | glimmer_base | 2 | 0 | 0.33 | 0 | 40 | 4 | cap |
| go-micro | glimmer_base | 1 | 0 | 0.60 | 0 | 40 | 8 | cap |
| go-micro | glimmer_base | 2 | 0 | 0.20 | 0 | 40 | 12 | cap |
| claude-agent-sdk-python-0a69e9 | glimmer_base | 1 | 1 | 0.80 | 0 | 34 | 5 | report |
| claude-agent-sdk-python-0a69e9 | glimmer_base | 2 | 1 | 0.80 | 0 | 40 | 7 | cap |
| mashumaro-31b756 | glimmer_base | 1 | 1 | 0.75 | 0 | 40 | 4 | cap |
| mashumaro-31b756 | glimmer_base | 2 | 1 | 0.50 | 0 | 40 | 3 | cap |
| temporal-79c467 | glimmer_base | 1 | 0 | 1.00 | 1 | 35 | 5 | report |
| temporal-79c467 | glimmer_base | 2 | 1 | 0.75 | 0 | 30 | 4 | report |
| celestia-app-ae5cc4 | glimmer_base | 1 | 1 | 1.00 | 1 | 17 | 5 | report |
| celestia-app-ae5cc4 | glimmer_base | 2 | 1 | 1.00 | 1 | 23 | 4 | report |
| cplt-930808 | glimmer_base | 1 | 1 | 0.50 | 0 | 40 | 4 | cap |
| cplt-930808 | glimmer_base | 2 | 1 | 0.25 | 0 | 36 | 2 | report |
| helm | glimmer_base | 1 | 1 | 1.00 | 1 | 40 | 8 | cap |
| helm | glimmer_base | 2 | 1 | 1.00 | 1 | 40 | 8 | cap |
| temporal-ef607d | glimmer_base | 1 | 0 | 0.00 | 0 | 40 | 4 | cap |
| temporal-ef607d | glimmer_base | 2 | 0 | 0.00 | 0 | 40 | 9 | cap |
| code-graph-rag-4cce83 | glimmer_base | 1 | 1 | 1.00 | 1 | 40 | 10 | cap |
| code-graph-rag-4cce83 | glimmer_base | 2 | 1 | 1.00 | 1 | 40 | 10 | cap |
| mcp-client-for-ollama-22343e | glimmer_base | 1 | 0 | 0.57 | 0 | 18 | 4 | report |
| mcp-client-for-ollama-22343e | glimmer_base | 2 | 1 | 0.57 | 0 | 40 | 6 | cap |
| temporal-d20b02 | a1_35b_scout | 1 | 0 | 0.57 | 0 | 40 | 10 | cap |
| temporal-d20b02 | a1_35b_scout | 2 | 0 | 0.14 | 0 | 40 | 12 | cap |
| mcp-client-for-ollama-8bfdcf | a1_35b_scout | 1 | 0 | 0.67 | 1 | 26 | 3 | report |
| mcp-client-for-ollama-8bfdcf | a1_35b_scout | 2 | 0 | 0.67 | 1 | 40 | 7 | cap |
| reqwest | a1_35b_scout | 1 | 1 | 0.67 | 0 | 21 | 2 | report |
| reqwest | a1_35b_scout | 2 | 1 | 1.00 | 1 | 30 | 3 | report |
| mashumaro | a1_35b_scout | 1 | 1 | 1.00 | 1 | 40 | 7 | cap |
| mashumaro | a1_35b_scout | 2 | 0 | 0.00 | 0 | 28 | 5 | report |
| google_workspace_mcp | a1_35b_scout | 1 | 1 | 1.00 | 1 | 40 | 5 | cap |
| google_workspace_mcp | a1_35b_scout | 2 | 1 | 1.00 | 1 | 21 | 4 | report |
| gh-aw-mcpg-774b25 | a1_35b_scout | 1 | 1 | 0.80 | 0 | 40 | 9 | cap |
| gh-aw-mcpg-774b25 | a1_35b_scout | 2 | 1 | 0.80 | 0 | 17 | 5 | report |
| openbiliclaw-411469 | a1_35b_scout | 1 | 1 | 0.67 | 0 | 32 | 2 | report |
| openbiliclaw-411469 | a1_35b_scout | 2 | 0 | 0.33 | 0 | 28 | 2 | report |
| go-micro | a1_35b_scout | 1 | 0 | 0.30 | 0 | 40 | 8 | cap |
| go-micro | a1_35b_scout | 2 | 0 | 0.50 | 0 | 25 | 7 | report |
| claude-agent-sdk-python-0a69e9 | a1_35b_scout | 1 | 0 | 0.80 | 0 | 25 | 5 | report |
| claude-agent-sdk-python-0a69e9 | a1_35b_scout | 2 | 1 | 0.80 | 0 | 29 | 5 | report |
| mashumaro-31b756 | a1_35b_scout | 1 | 1 | 1.00 | 1 | 37 | 5 | report |
| mashumaro-31b756 | a1_35b_scout | 2 | 1 | 0.75 | 0 | 37 | 4 | report |
| temporal-79c467 | a1_35b_scout | 1 | 0 | 0.50 | 0 | 30 | 3 | report |
| temporal-79c467 | a1_35b_scout | 2 | 1 | 0.00 | 0 | 40 | 12 | cap |
| celestia-app-ae5cc4 | a1_35b_scout | 1 | 1 | 0.67 | 0 | 12 | 5 | report |
| celestia-app-ae5cc4 | a1_35b_scout | 2 | 1 | 1.00 | 1 | 27 | 6 | report |
| cplt-930808 | a1_35b_scout | 1 | 1 | 0.25 | 0 | 16 | 1 | report |
| cplt-930808 | a1_35b_scout | 2 | 1 | 0.25 | 0 | 30 | 1 | report |
| helm | a1_35b_scout | 1 | 1 | 1.00 | 1 | 28 | 4 | report |
| helm | a1_35b_scout | 2 | 1 | 1.00 | 1 | 15 | 3 | report |
| temporal-ef607d | a1_35b_scout | 1 | 0 | 0.67 | 1 | 34 | 10 | report |
| temporal-ef607d | a1_35b_scout | 2 | 0 | 0.00 | 0 | 40 | 9 | cap |
| code-graph-rag-4cce83 | a1_35b_scout | 1 | 1 | 1.00 | 1 | 28 | 4 | report |
| code-graph-rag-4cce83 | a1_35b_scout | 2 | 1 | 1.00 | 1 | 34 | 4 | report |
| mcp-client-for-ollama-22343e | a1_35b_scout | 1 | 0 | 0.29 | 0 | 40 | 2 | cap |
| mcp-client-for-ollama-22343e | a1_35b_scout | 2 | 0 | 0.57 | 0 | 40 | 7 | cap |
| temporal-d20b02 | spark_teacher | 1 | 0 | 0.43 | 0 | 40 | 7 | cap |
| temporal-d20b02 | spark_teacher | 2 | 0 | 0.29 | 0 | 40 | 11 | cap |
| mcp-client-for-ollama-8bfdcf | spark_teacher | 1 | 0 | 0.67 | 1 | 31 | 3 | report |
| mcp-client-for-ollama-8bfdcf | spark_teacher | 2 | 0 | 0.67 | 1 | 40 | 6 | cap |
| reqwest | spark_teacher | 1 | 1 | 1.00 | 1 | 40 | 5 | cap |
| reqwest | spark_teacher | 2 | 1 | 1.00 | 1 | 40 | 3 | cap |
| mashumaro | spark_teacher | 1 | 1 | 1.00 | 1 | 40 | 6 | cap |
| mashumaro | spark_teacher | 2 | 1 | 1.00 | 1 | 40 | 6 | cap |
| google_workspace_mcp | spark_teacher | 1 | 1 | 1.00 | 1 | 12 | 3 | report |
| google_workspace_mcp | spark_teacher | 2 | 1 | 1.00 | 1 | 40 | 4 | cap |
| gh-aw-mcpg-774b25 | spark_teacher | 1 | 1 | 0.80 | 0 | 40 | 9 | cap |
| gh-aw-mcpg-774b25 | spark_teacher | 2 | 1 | 0.80 | 0 | 40 | 11 | cap |
| openbiliclaw-411469 | spark_teacher | 1 | 0 | 0.67 | 0 | 40 | 6 | cap |
| openbiliclaw-411469 | spark_teacher | 2 | 1 | 0.67 | 0 | 17 | 4 | report |
| go-micro | spark_teacher | 1 | 0 | 0.40 | 0 | 40 | 9 | cap |
| go-micro | spark_teacher | 2 | 0 | 0.40 | 0 | 40 | 9 | cap |
| claude-agent-sdk-python-0a69e9 | spark_teacher | 1 | 1 | 1.00 | 1 | 40 | 10 | cap |
| claude-agent-sdk-python-0a69e9 | spark_teacher | 2 | 1 | 0.80 | 0 | 40 | 10 | cap |
| mashumaro-31b756 | spark_teacher | 1 | 1 | 1.00 | 1 | 14 | 5 | report |
| mashumaro-31b756 | spark_teacher | 2 | 1 | 1.00 | 1 | 25 | 5 | report |
| temporal-79c467 | spark_teacher | 1 | 1 | 0.50 | 0 | 40 | 9 | cap |
| temporal-79c467 | spark_teacher | 2 | 1 | 0.75 | 0 | 40 | 7 | cap |
| celestia-app-ae5cc4 | spark_teacher | 1 | 1 | 1.00 | 1 | 19 | 5 | report |
| celestia-app-ae5cc4 | spark_teacher | 2 | 1 | 1.00 | 1 | 7 | 7 | report |
| cplt-930808 | spark_teacher | 1 | 1 | 0.50 | 0 | 40 | 6 | cap |
| cplt-930808 | spark_teacher | 2 | 1 | 0.75 | 0 | 40 | 4 | cap |
| helm | spark_teacher | 1 | 1 | 1.00 | 1 | 17 | 3 | report |
| helm | spark_teacher | 2 | 1 | 1.00 | 1 | 12 | 3 | report |
| temporal-ef607d | spark_teacher | 1 | 0 | 0.67 | 1 | 40 | 8 | cap |
| temporal-ef607d | spark_teacher | 2 | 0 | 0.33 | 0 | 40 | 10 | cap |
| code-graph-rag-4cce83 | spark_teacher | 1 | 1 | 1.00 | 1 | 14 | 4 | report |
| code-graph-rag-4cce83 | spark_teacher | 2 | 1 | 1.00 | 1 | 40 | 9 | cap |
| mcp-client-for-ollama-22343e | spark_teacher | 1 | 1 | 0.86 | 0 | 27 | 6 | report |
| mcp-client-for-ollama-22343e | spark_teacher | 2 | 0 | 0.57 | 0 | 40 | 6 | cap |
