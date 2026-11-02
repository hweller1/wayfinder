# E2E coding task set v4: 83 headroom tasks from held-out repos

Built 2026-10-27 (author approved the larger set and the mining that extends it). Manifest: `eval/e2e/tasks_v4.json` (a superset of `tasks_v3.json`: the 8 v3 tasks are kept unchanged). Reference rows: `eval/e2e/results_v4.json`. Interim manifest at 33 tasks: `tasks_v4_interim.json`. No search-model condition was run on this set by this build.

n is 83 (75 new + the 8 from v3), above the 65 target. Every task and repo is outside the train split, so no `e2e_only` split was needed and `data/splits.json` is untouched.

## Mining

Pool: the 35 Python, Go and Rust repos in the dev, test and eval_holdout splits of `data/splits.json` (the 18 whose clones were missing were cloned blob:none with full history under `eval/e2e/data/e2e/repos/`). The miner is `scripts/e2e_v4/mine_v4.py`, a copy of `benchmark-repo/scripts/mine_expanded_tasks.py` with the audited classifiers and leak guards vendored under `scripts/e2e_v4/vendor/asa` (copied, never imported across repos). Filters: source commit on or after 2024-07-01, at least 2 non-test source files changed, a runnable test unit (the gate's own `units_for`), the miner's commit-scope, template-slot, whole-path and basename guards, one task per source commit, and the gold-at-base tree check (0 discrepancies). Commits the v3 gate had already tried are skipped.

| Stage | Count |
|---|---|
| Non-merge commits since 2024-07-01, 35 repos | 101,924 |
| At least 2 non-test source files | 35,885 |
| Has a runnable test unit | 14,676 |
| Within the size cap (at most 10 source files) | 12,834 |
| Accepted by a category classifier and the leak guards | 3,641 |
| After gold-at-base check (candidates, 30 repos) | 3,523 |

The 3,523 candidates exceeded the 1,000 target, so the held-out repos alone supplied enough and no new public repos were added.

## Screen funnel

The gate was run in waves in held-out-split order over the repos that gated in a pilot (4 tasks per repo, 117 tasks), then over the rest of the productive repos, and was stopped once the screen had ample supply (1,442 of the 3,523 attempted; the machine load limit was reached at about 330 concurrent sandboxes).

| Stage | Tasks |
|---|---|
| Gate attempted | 1,442 |
| Gated (fail-to-pass fail at parent, pass at the fix, no-network sandbox) | 647 |
| Issue text passes the five leak guards (3 tries, never loosened) | 363 |
| Scorer self-check (real patch passes, empty patch fails) | 329 of 336 run |
| Screened at seed 0 (oracle and no-search) | 236 |
| Oracle passes | 114 |
| Oracle passes and no-search fails (headroom) | 75 |

Gate losses: 425 no touched test passes at the fix (environment cannot import the package), 90 touch no test function, 83 exceptions (lock compile), 81 prepare failures, 50 already pass at the parent, 34 flaky, 32 too slow. 93 validated tasks were not screened (gate stopped early, a deliberate stop), so more headroom exists in the pool. Seed 0 is the screening seed and is never reported.

## Final set

83 tasks in 12 repos: dev 38, test 43, eval_holdout 2. Python 49, Go 17, Rust 17. Largest repos: agno 19, kara 16, google_workspace_mcp 13, gentle-ai 10, nucs 9. Tasks cluster by repo, so the effective sample is smaller than 83 for a repo-level effect; a paired comparison across tasks within the repo set is unaffected, but the set is dominated by a few repos and eval_holdout is thin (2 tasks, temporal from v3), because pandas, skypilot, mteb, deepagents and temporal mostly fail the gate on environment.

## Reference rows (seeds 1-2, 95% task-bootstrap CI)

| Condition | Pass | Rate | 95% CI |
|---|---|---|---|
| none (no search) | 10/166 | 0.06 | [0.02, 0.11] |
| oracle (gold files) | 156/166 | 0.94 | [0.90, 0.98] |

83 tasks, 2 seeds each. For the 8 v3 tasks these are the v3 rows (themselves round-3 records for 6 of them). For the 75 new tasks: no-search 9/150 (0.06, [0.01, 0.11]), oracle 143/150 (0.95, [0.91, 0.99]). The oracle ceiling is higher than v3's 0.81 because the screen keeps only tasks the oracle solved at seed 0. The coder is Opus 5.5 medium in the no-network Modal sandbox with the same harness code as v3.

## Power

Paired sd of 0.283 from round 3 (`power_v3.json`). Minimum detectable paired difference at 80% power: 0.087 at n=83 (0.098 at 65). Power for a +0.10 difference is 0.90 at n=83 and 0.81 at n=65, from a normal approximation, two-sided alpha 0.05.

## Contamination

Asserted by `scripts/e2e_v4/contamination.py` (output `eval/e2e/experiments/e2e_v4/contamination.json`, clean): split is not train for every task; the repo and the repo name are not a train-split repo; no task id and no source commit appears in the SFT v1-v4 teacher files, the A1 converted files, or the OPD round-1 and round-2 rollouts (2,116 files scanned; summary and drop files that list excluded tasks are skipped by design); and no task id or repo appears in `artifacts/rl_pool/*.json` (47,252 task ids) or `data/rl_pool_*.json` (none exist). Membership in `e2e_excluded_tasks.json` and `opd_excluded_tasks.json` is recorded, not a failure: those files list tasks kept out of training. Only the 8 v3 tasks can be in them, and they are in no training file.

## How it was built

`scripts/e2e_v4/mine_v4.py` mines; `eval/e2e/run_coder_v4_pipeline.py` (copy of the v3 driver, workspace `eval/e2e/experiments/e2e_v4`, runs in `eval/e2e/runs_v4`) gates, writes issues (Sonnet 5.5 high, 16 in parallel), self-checks and runs the coder; `run_coder_v4_loop.py`, `run_coder_v4_screen.py` and `run_coder_v4_ref.py` shard the screen over many processes (the run_coder prefix keeps `ops/e2e_watch.py` from stopping the sandbox app); `analyze_v4.py` writes the manifest. Changes to shared code: `modal_sandbox.save_toolchain` now takes a file lock and renames atomically (parallel gates clobbered `modal_toolchains.json`), and `write_issues.py` runs tasks in parallel under `E2E_ISSUE_WORKERS`. Defaults for v3 are unchanged.

## Spend

About $368 recorded under `e2e-taskset-v4`: $286 for 921 coder run records (Opus plus sandbox CPU), about $40 of Modal CPU for gates and self-checks, about $45 of Sonnet issue writing. That is slightly over the $350 allotment. No GPU, no search-model condition and no Baseten call in this build.

## Caveats

- Oracle 0.94 and no-search 0.06 are set by selection on seed 0; the rows are seeds 1-2, which were not used to select.
- 12 repos; agno is 23% of the set.
- Issue texts are model-written under leak guards that reject about 43% of gated tasks, so the set favours changes whose behaviour can be described without naming existing code.
