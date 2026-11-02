# Spec hill-climb: does any delivery of the Scout retrieval spec beat Explore with the same spec?

No. On the 24 held-out CONFIRM tasks the chosen Scout config (verbatim handoff) scored recall@12 0.458 against 0.511 for built-in Explore with the same spec and parent instruction, a paired difference of -0.052 [-0.157, +0.056]. The pre-registered pass rule needs a positive point estimate, so Scout fails. Nothing in the climb separated from noise on TUNE either, apart from one variant that was clearly worse. The prototype keeps its current defaults. Plan, split and rules: `docs/spec_hillclimb_plan.md` (committed as cdfa530 before any variant ran).

## Setup

The 50 cc50 tasks were split by repo into TUNE (26 tasks, 18 repos) and CONFIRM (24 tasks, 17 repos), seeded and balanced on language and mongodb-ecosystem (`scripts/split_cc50.py`, `data/cc_scout_hillclimb_split.json`). Selection used TUNE only. CONFIRM ran once, with one seed. Parent is Sonnet 5.5 through an LLM API gateway, one seed per variant unless stated, repo checkouts at base_commit, all 50 repos already indexed, wayfinder-glimmer served from `wayfinder-v3-serve` while rounds ran. Recall and precision are scored on the parent's final reported list, as in `cc_scout_ablation.md`. Rows that failed to parse a reported path were rerun once; a row that still failed is excluded from that variant and from its paired contrasts (n shows it).

The levers are real spec options, documented in voyage-scout `docs/retrieval-spec.md`: `[delivery] handoff` (agent, verbatim, direct), `report_guidance`, `search_policy` (deny, warn, off), `[examples] select` (recent, similar) and `mine`, and `[explore] max_turns`. The variants differ only in the per-task spec file the harness writes (`VARIANTS` in `eval/harness/arms.py`). The report guidance text was "list only files you have evidence for, ranked most relevant first; aim for the 5 to 8 most relevant files". Variant names: `x_base` is cc_explore_spec, `x_guid` is cc_explore_spec with the guidance, `base` is cc_scout_spec, `h_verbatim` and `h_direct` change the handoff, `g_verbatim` adds the guidance to `h_verbatim`, `v_*` are round-2 variants on `h_verbatim`.

## TUNE results (26 tasks, seed 0)

Delegated counts the runs where the parent used a subagent or called Scout's explore tool. The precision floor is cc_explore_spec with the guidance (`x_guid`, 0.413) minus 0.03, so 0.383.

| Variant | n | recall@5 | recall@12 | precision@12 | n_reported | delegated | median wall (s) | $/run (the LLM gateway) |
|---|---|---|---|---|---|---|---|---|
| x_base (Explore + spec) | 26 | 0.410 | 0.539 | 0.412 | 10.5 | 26/26 | 31 | 0.100 |
| x_guid (Explore + spec + guidance) | 26 | 0.377 | 0.562 | 0.413 | 11.5 | 26/26 | 32 | 0.101 |
| base (cc_scout_spec rerun) | 26 | 0.390 | 0.520 | 0.412 | 9.0 | 25/26 | 159 | 0.073 |
| h_verbatim (2b) | 26 | 0.348 | 0.541 | 0.387 | 8.8 | 26/26 | 155 | 0.045 |
| h_direct (2c) | 26 | 0.343 | 0.496 | 0.414 | 8.0 | 23/26 | 144 | 0.044 |
| g_verbatim (lever 1 on 2b) | 26 | 0.416 | 0.495 | 0.435 | 7.4 | 26/26 | 150 | 0.041 |
| v_warn (verbatim, search-first warns) | 26 | 0.404 | 0.518 | 0.417 | 9.4 | 16/26 | 149 | 0.045 |
| v_off (verbatim, no search-first) | 26 | 0.335 | 0.481 | 0.443 | 9.3 | 13/26 | 82 | 0.045 |
| v_mine0 (verbatim, 0 examples) | 26 | 0.376 | 0.508 | 0.384 | 8.4 | 24/26 | 152 | 0.041 |
| v_mine6 (verbatim, 6 examples) | 25 | 0.385 | 0.535 | 0.394 | 9.5 | 25/25 | 148 | 0.043 |
| v_sim3 (verbatim, 3 similar examples) | 25 | 0.347 | 0.469 | 0.375 | 9.0 | 24/25 | 151 | 0.040 |
| v_t6 (verbatim, explore capped at 6 turns) | 26 | 0.332 | 0.536 | 0.387 | 9.6 | 26/26 | 124 | 0.047 |
| v_t8 (verbatim, 8 turns) | 25 | 0.366 | 0.523 | 0.393 | 9.3 | 25/25 | 148 | 0.044 |

Second seed on TUNE for the top two within 0.02 (`h_verbatim`, `v_t6`) and for `x_guid`: `x_guid` recall@12 0.530, precision 0.355; `h_verbatim` 0.509, 0.394; `v_t6` 0.438, 0.330. `v_t6` against `x_guid` is -0.095 [-0.167, -0.023], so the turn cap loses recall. Seed-to-seed movement of a single variant (0.541 to 0.509 for `h_verbatim`, 0.562 to 0.530 for `x_guid`) is about the size of most between-variant gaps, which is why nothing below is called a win.

Lever 1 was run on the best handoff only (`g_verbatim`) as the plan said; `g_direct` was exercised on four calibration tasks to check that the guidance reaches the parent; `g_agent` and `g_direct` were not run on TUNE.

## Climb path

Round 1. The handoff levers: `h_verbatim` had the best recall@12 among the Scout variants (0.541) while staying above the precision floor (0.387 against 0.383); `base` was 0.520 and `h_direct` 0.496. Against `x_guid` none of them is distinguishable: `h_verbatim` -0.014 [-0.080, +0.050] in the first pass and +0.021 for `x_guid` over it in the final rows, `base` -0.042 [-0.148, +0.049], `h_direct` -0.066 [-0.151, +0.018]. The guidance on the verbatim handoff cut reports to 7.4 files and raised precision to 0.435, but recall@12 fell to 0.495 (-0.067 against `x_guid`, [-0.149, +0.010]), so it failed the selection metric. `h_verbatim` was carried forward.

Round 2, all on `h_verbatim`, differences against it. Search-first warn -0.023 [-0.104, +0.047] and off -0.060 [-0.125, +0.003] (delegation fell to 16/26 and 13/26, and the off variant is faster only because the parent skipped the subagent in half the runs). Zero mined examples -0.033 [-0.098, +0.036], six -0.028 [-0.080, +0.023], three chosen by similarity -0.094 [-0.167, -0.028], the only difference whose interval excludes zero. Turn cap 6 -0.005 [-0.065, +0.053], cap 8 -0.040 [-0.101, +0.016]. No round-2 variant beat the round-1 winner, so `h_verbatim` was the chosen config. Round 3 (index symbols, k 8, excludes) was skipped: Modal was already near its $20 target, and no earlier lever had moved recall.

## CONFIRM (24 tasks, one run, chosen config `h_verbatim`)

| Arm | n | recall@5 | recall@12 | precision@12 | n_reported | delegated | median wall (s) | $/run |
|---|---|---|---|---|---|---|---|---|
| Scout, verbatim handoff (chosen) | 24 | 0.359 | 0.458 | 0.335 | 8.8 | 24/24 | 159 | 0.044 |
| cc_explore_spec + guidance (comparator) | 24 | 0.369 | 0.511 | 0.333 | 11.5 | 24/24 | 31 | 0.098 |
| cc_explore_spec, no guidance | 24 | 0.442 | 0.565 | 0.392 | 10.7 | 24/24 | 33 | 0.106 |
| cc_explore (stock) | 24 | 0.373 | 0.461 | 0.384 | 8.5 | 0/24 | 8 | 0.040 |

Paired contrasts (task bootstrap, 10,000 resamples, 95%):

| Contrast | recall@5 | recall@12 | precision@12 |
|---|---|---|---|
| Scout minus Explore + spec + guidance | -0.010 [-0.125, +0.103] | -0.052 [-0.157, +0.056] | +0.001 [-0.082, +0.077] |
| Scout minus stock cc_explore | -0.014 [-0.069, +0.044] | -0.002 [-0.071, +0.065] | -0.049 [-0.117, +0.012] |
| Explore + spec + guidance minus stock | +0.004 [-0.133, +0.140] | +0.050 [-0.093, +0.186] | -0.051 [-0.135, +0.052] |

## Pass or fail

Fail. The rule required a recall@12 point estimate above zero against Explore with the spec and the same instruction (got -0.052), a lower bound above -0.02 (got -0.157), and precision not more than 0.03 lower (got +0.001, which passes). Scout does not beat Explore here, and on CONFIRM it is no better than stock cc_explore either (-0.002 recall@12, precision 0.049 lower). The CONFIRM interval is wide: 24 tasks cannot rule out a gain or a loss of about 0.1 in either direction, so the honest reading is "no evidence that Scout's retrieval helps this parent, and Scout costs about five times the wall time". Two things worth knowing. The report guidance did not help Explore on CONFIRM (0.565 without it, 0.511 with it, one seed, within noise), so the comparator is, if anything, a weaker one than `x_base`; against `x_base` Scout is further behind. And the guidance did not shorten Explore's list (11.5 reported files with it against 10.7 without); it shortened Scout's only on TUNE (7.4 against 8.8).

## Latency

Median wall: Explore with the spec 31 to 33 s, Scout with the Sonnet agent 159 s, verbatim relay 155 s, direct call 144 s, explore turn cap 6 124 s and cap 8 148 s. Removing the Sonnet middle layer saves roughly 15 s of about 155 s, and a 6-turn cap saves about 30 s at a recall cost (second seed -0.095 against Explore). The time sits in wayfinder-glimmer's own loop inside `explore`, as the latency profile found. Raw stream-json transcripts for every run from round 2 onward are in `voyage-scout/artifacts/eval/cc50/hillclimb/*/transcripts/`; the per-call explore `usage.turns` and `wall_ms` have not been extracted from them yet.

## Recommended default spec for the prototype

Keep the current defaults: `handoff = "agent"`, `search_policy = "deny"` (search-first on), 3 recent mined examples, `k = 5`, index description without symbols, no `report_guidance`, no turn cap. The new options ship as documented, off by default. If Scout is to be recommended on retrieval quality, the result to beat is Explore with the spec, and the latency gap must close first; a verbatim relay is a reasonable choice when cost per run matters, since `h_verbatim` runs at about $0.045 the LLM gateway against $0.10 for Explore with the spec, with recall that is not distinguishable on either split.

## Cost

| Item | Amount |
|---|---|
| the LLM gateway (532 rows including calibration and reruns, from row cost) | about 31 USD, against the 70 cap |
| Modal wayfinder-glimmer warm, 13:26 to 18:22 PT, H100 at about 3.95 USD/h (estimated from warm hours) | about 19.5 USD, against the 20 target and 35 stop |
| Voyage | 0 (no new indexing) |

`wayfinder-v3-serve` was redeployed at `min_containers=0` after the CONFIRM run.

## Caveats

One seed on CONFIRM and mostly one on TUNE, 24 to 26 tasks, so only differences above about 0.07 are separable from noise and one variant (`v_sim3`) is. No pass rate is measured. The first variants ran at width 4 and later ones at width 10 to 12, which changes contention on the gateway and so wall time, not retrieval. Scout rows within a task share a checkout and run sequentially. Similarity selection ranks the mined pool by token overlap with the task text, which is a weak proxy for an embedding; the result does not rule out a better selector. The `x_guid` comparator was chosen in the plan before seeing that guidance hurt Explore on CONFIRM.
