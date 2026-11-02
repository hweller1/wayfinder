# Muse Code with Scout: pre-registered calibration plan

Written 2026-10-31 before any arm run (a probe of Muse's hook events with a throwaway plugin and a 1-repo index check were the only things run). This plan fixes the arms, tasks, metrics, comparison and stop rules for a calibration of the Scout retrieval spec on Muse Code. The result will be reported as calibration-scale evidence, not as a confirmed result. The author approved the work on 2026-10-31 so a side-by-side demo can be filmed on Monday 2026-11-02 (Muse Code without Scout against Muse Code with Scout). The model behind Scout stays wayfinder-glimmer (Muse Glimmer 30B base, wayfinder-glimmer).

## Question

Does Muse Code, given the Scout retrieval spec and the Scout plugin, find more of the files that a task touches than stock Muse Code, or than Muse Code given the same spec but only its built-in search? The earlier Muse calibration (voyage-scout `docs/PRODUCT_STATUS.md`, 5 tasks, no spec, no CIs) found the plugin no better than plain Muse and about 5 times slower. The Claude Code spec ablation (`docs/cc_scout_ablation.md`, `docs/spec_hillclimb.md`) found the spec fixes delegation but Scout did not beat built-in Explore with the same spec. This run asks the same question on a second host.

## Arms

All three arms get the same free-parent prompt (`arms.CC_PROMPT`: "Find the files in this repository that are relevant to the task below, and report them. Use whatever tools you have.", with the task text and the one-line instruction to end with a JSON list of repo-relative files, best first). One headless `muse exec --json` session per (task, arm), one seed, `--disable-write --disable-shell` (retrieval only, as in the first Muse calibration, where an unrestricted parent started implementing), Muse's `search` and `read_file` available to every thread.

| Arm | Installed | Delegate | What it isolates |
|---|---|---|---|
| muse_stock | nothing | none | Muse as it ships |
| muse_explore_spec | three spec hooks only (session context, workflow step, search-first), no MCP server, no agent | built-in `general-purpose` agent with Muse's own search and read tools | the effect of the spec's delivery (delegate first), with no index |
| muse_scout_spec | the full `scout-wayfinder` 0.3.0 plugin plus the same spec | `scout-wayfinder/scout-explorer`, which calls Scout `explore` (wayfinder-glimmer over the gateway index) then verifies with reads | the product: spec + Scout + index |

The spec file is written per task, outside the checkout, and pointed at through `SCOUT_SPEC_FILE`. It is the file the Claude Code spec arms used (`arms.write_spec_file`): search-first on, workflow step on, session context on, 3 mined examples. The leak guard is unchanged: examples are mined strictly before `base_commit` and never touch a gold file. Both spec arms receive the identical file.

The parent model is the host's default (no `--model`, no `--reasoning-effort`). A probe on 2026-10-31 showed Muse Code 1.4.14 selecting `muse-spark-1.3-contributor` (profile `tbh`) at the default effort `high`; every row records the model id Muse actually used, and the report states it. The scout-explorer agent pins its own model (`muse-spark-1.3`, effort `low`); that is part of the product under test. API key: the Meta key from `wayfinder/.env`, passed only in the child environment.

Isolation: every session runs with `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_STATE_HOME` and `XDG_CACHE_HOME` pointing at a per-run scratch directory under `/tmp/muse-spec/xdg`. `~/.config/muse`, `~/.local/share/muse` and `~/.claude` are never written. The scratch directory is deleted after each run.

## Tasks

The TUNE side of the pre-registered cc50 split, `data/cc_scout_hillclimb_split.json`: **26 tasks in 18 repos** (the brief for this work said 25; the file holds 26 and all 26 are used). CONFIRM (24 tasks) is not touched. One seed. Checkouts are fresh blobless clones at each task's `base_commit` with no ref reaching a later commit. The repo index for the Scout arm is the existing gateway index: a re-sync of each checkout through the product path (`scout sync --wait`) before the run must report `ready` and is expected to embed zero new tokens; embedded tokens are logged per sync, and if any task would embed more than a trivial amount the run stops (see Budget).

Calibration first: the first 5 TUNE tasks in file order (solidus, akaunting, joomla-cms twice, mongosh), all three arms. The 5 are then reused as part of the 26 if the gates below pass.

## Metrics

All scored on the parent's final answer, the same extraction as every earlier cc50 study (`briefs.extract_paths`: a token counts only if it is exactly a tracked path of the checkout).

- recall@5 and recall@12: share of the task's gold files among the first 5 and 12 reported files.
- precision@12: gold files among the first 12 reported, divided by the number reported (at most 12).
- Delegation rate: share of runs where the parent called `subagent_spawn`.
- Scout tool-call rate: share of runs in which a Scout `explore` call happened in any thread.
- Median wall seconds per run (and p90), end to end, with the Scout `explore` call's own median recorded separately.
- Cost, in separate units, never summed into one number: Muse API dollars per run at the Spark list price ($1.25 input, $0.15 cached input, $4.25 output per million tokens) for the parent and all child threads; Modal GPU dollars (not per call; the warm hours are reported once, from the billing report); Voyage dollars (index re-sync tokens, expected zero).

## Pre-registered comparison

Primary contrasts, paired by task, on recall@12, with recall@5 and precision@12 alongside: **muse_scout_spec minus muse_stock** and **muse_scout_spec minus muse_explore_spec**. 95% percentile interval from a 10,000-resample task bootstrap (seed 20261025, `eval/harness/stats.py`), plus a repo-cluster bootstrap as a sensitivity check. There is no pass or fail threshold and no claim is made from a point estimate: with about 26 tasks the intervals are expected to be wide (about plus or minus 0.1 on recall@12, as on CONFIRM in the Claude Code study) and the report says so. A gain or loss is described only if the interval excludes zero, and even then as calibration-scale evidence.

The secondary reading is descriptive: delegation and Scout-call rates (did the spec work on this host), latency, and Muse dollars per run.

## Calibration gates (5 tasks per arm, before the full run)

1. Rows valid: exit code 0, plausible wall time, model recorded, reported paths parsed, and for both spec arms the SessionStart context and the workflow step actually fired (read from the hook decision log, which holds event names and never prompts or paths). An invalid row is rerun once and, if it stays invalid, excluded from its arm and from the paired contrasts, with the count reported.
2. Scout actually called: `muse_scout_spec` must show a Scout `explore` call in at least 4 of 5 runs. If it does not, the delivery is fixed (hook, agent description, workflow wording) and the 5 are rerun; the 5 failed rows are kept and reported.
3. Stock integrity: `muse_stock` rows carry no plugin and no spec output.
4. Projected full-run cost, from the calibration, is under the caps below.
5. wayfinder-glimmer is warm for the Scout arm (scripted warm-up, scaled to zero afterwards).

Then all 26 tasks, all three arms (78 sessions), width 2 across repos (one task at a time per repo checkout). Concurrency is recorded because it moves Scout's wall time; wayfinder-glimmer is one H100.

## Budget and stop rules

| Item | Cap | Gate |
|---|---|---|
| Muse API (Meta key, billed per token) | $60, logged | reserve recorded as `--kind gateway`; actual cost summed from run tokens and recorded after each stage |
| Voyage | $10, demo repos only | the TUNE re-sync is expected to embed zero tokens; the run aborts if embedded tokens ever cost more than $8 in total |

wayfinder-glimmer and the scout gateway are warm only while a run is in progress and are scaled back to zero afterwards. No retries beyond the single rerun of an invalid row, no second seed and no extra arms: more than that would be new gateway spend and needs the author's explicit approval first. If the paired intervals straddle zero after this run, that is the result.

## Disclosed limits

One seed, 26 tasks, so intervals are wide and a second seed is not run. The parent is Muse's default model at high effort, not the cheaper low-effort setting of the first calibration, and it differs from the Sonnet 5.5 parent of the Claude Code studies, so absolute numbers do not transfer. The three arms differ in more than the index: `muse_explore_spec` delegates to a general-purpose child on the parent's model, `muse_scout_spec` to a `muse-spark-1.3` low-effort explorer plus a wayfinder-glimmer loop. Muse has no Explore subagent, so there is no exact twin of the Claude Code `cc_explore_spec` arm. `--disable-write --disable-shell` removes shell grep in every arm; Muse's `search` tool remains. Recall is scored on the parent's reported list, so report length matters. The spec's mined examples describe past commits of the same repository, which gives both spec arms a vocabulary hint stock does not have. No coding-outcome metric.

## Deviations

Any departure from this plan after the first run is listed here with its reason, and the results document repeats it.
