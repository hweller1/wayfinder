# Final E2E eval: pre-registration

Written 2026-10-30 15:10 PT, before any E2E data on the clean task set. This plan fixes task eligibility, screening, the reporting seeds and the analysis. Changes after data collection starts are recorded below as dated addenda and labelled post hoc.

## Question

Does a better search subagent raise the pass rate of a fixed coder on tasks that cannot be solved without finding the code? Design: handed files (`docs/e2e_eval.md`). The coder is Opus 5.5 via an LLM API gateway with no search tools; `read_file` opens only files the subagent returned; held-out tests are absent from the sandbox.

Conditions: `none` (nothing handed), `oracle` (gold source files), and four subagents in scout's loop: `v3`, `a1`, `sonnet` (Sonnet 5.5 high), `gpt` (GPT-6.1 Sol).

## Task eligibility (author rule 2026-10-30)

A task the coder can pass without search measures memorisation or triviality, not retrieval, and is excluded. Eligibility is decided before any subagent condition runs.

1. **Structural filter (build time).** The gold patch modifies at least one existing non-test file at the base commit. Tasks whose gold patch only adds new files are dropped, since a blind coder can write them from the issue alone.
2. **Contamination probe (no tools).** Opus 5.5 gets the repo name and the issue, no repository access, and is asked to list the files it would change. A task is flagged memorised if it names at least half of the gold non-test paths exactly. Flagged tasks are excluded and counted.
3. **Two-seed blind screen.** `none` runs on seeds 0 and 1 and must fail both. `oracle` runs on seed 0 and must pass. Tasks meeting both are the headroom set.
4. Tasks from repos in the trained-on set (`eval/e2e/experiments/e2e_clean/trained_on_repos.json`) or in dev, test, eval_holdout, hard_dev_v1 or hard_v2 are not eligible (enforced by the task builder).

## Size and curation

Target: at least 50 headroom tasks from at least 15 repos (raised from 25 by the author on 2026-10-30 15:15 PT, before any E2E data). If the first task file yields fewer, curation continues from the remaining mined candidates (gate, issue, probe, screen) in batches, prioritising gold patches that touch more existing files and span more directories, until 50 are reached or the Modal cap is hit. The eligibility rules above do not change to reach the target. If the target is not reached, the result is reported at the size achieved, with its minimum detectable effect.

## Reported runs

All six conditions run on seeds 2 and 3 for every headroom task. Seeds 0 and 1 are screening only and never reported as results. Subagent search runs use the same scout-loop settings as `docs/heldout_hard_v1_frontier.md` (grep and semantic search available, 40 turns), handing the coder the subagent's report and the first five reported files' excerpts, as in `docs/e2e_eval.md`.

## Analysis

- Unit: task; seeds averaged within task.
- Primary family (Holm across 4): each subagent minus `none` on pass rate, exact paired sign-flip test, plus task-bootstrap 95% CI.
- Secondary, descriptive: subagent-vs-subagent contrasts (wayfinder-glimmer minus A1, Sonnet minus wayfinder-glimmer, GPT minus wayfinder-glimmer), `oracle` minus each subagent, gold recall of the handed files, and pass rate by whether every gold file was handed.
- Robustness: the same contrasts on all oracle-solvable tasks regardless of the blind screen, labelled as such.
- Reported alongside: counts dropped at each eligibility step (structural, probe, blind screen), per language.

## Calibration

Five tasks through every stage (probe, screen, one seed of each condition) before scaling. Checks: the coder reaches the LLM gateway, handed files reach the sandbox, subagents return reports, the scorer runs the held-out tests, and `oracle` passes at least 3 of 5.

## Budget


## Addendum 1 (2026-10-30 16:40 PT, before any E2E data)

1. **Interpretation of subagent-vs-subagent contrasts.** wayfinder-glimmer vs A1, Sonnet vs wayfinder-glimmer and GPT vs wayfinder-glimmer on pass rate are descriptive. Each is reported with its minimum detectable effect at the achieved n (two-sided alpha 0.05, power 0.8, paired). A non-significant difference smaller than the MDE is reported as "not resolved at this n", not as "no difference".
2. **Secondary task set: E2E v4.** The 83 already-gated tasks in the E2E v4 set (12 repos, no repo overlap with SFT train or either RL pool; 7 shared with the dev pool and 4 with the test split) go through the same contamination probe and two-seed blind screen. Survivors run the same six conditions on seeds 2 and 3. They are reported as a separate, labelled secondary set and never pooled with the primary clean set.
3. **Cutoff.** Data collection stops at 12:00 PT 2026-10-31. Results are reported at the size reached then, with the MDE. The eligibility rules do not change.
4. **Curation order.** For new candidates, the issue is written and the contamination probe run before Modal gating; only probe-passing candidates are gated. This changes cost, not eligibility.
