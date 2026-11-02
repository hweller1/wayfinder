# Held-out hard v1: wayfinder-glimmer against Agents-A1-35B

This document reports a one-time evaluation of wayfinder-glimmer and Agents-A1-35B on `data/heldout_hard_v1.json`, a 200-task set from 47 repositories that no Wayfinder model has trained on or been selected against. It tests whether wayfinder-glimmer's retrieval advantage over A1 holds on codebases, languages and repository types outside the training distribution. It does not evaluate the GRPO checkpoints; round 2 ended flat against wayfinder-glimmer (see `docs/grpo_r2_plan.md`, RESULT).

## Set and protocol

The tasks come from the hard_v3 mining pass (`scripts/rlpool/mine_hard_v3.py`): bug-fix and feature commits since 2025 whose gold files span at least two top-level directories and include at least one file a grep over the question's own tokens cannot reach. The held-out split was made by repository before any wayfinder-glimmer screening score was looked at. All 12 MongoDB-ecosystem repositories were held out, and the remaining repositories were drawn round-robin by repository type, with at most 6 tasks per repository. Neither the GRPO pools nor the dev set share a repository with it.

| Dimension | Coverage |
|---|---|
| Languages | TS/JS 50, Java/Kotlin 28, C# 18, Go 18, Other 18, Rust 16, Swift 16, C/C++ 15, Python 13, Ruby 8 |
| Repository types | 26, including MongoDB ecosystem (24 tasks, 7 repos), .NET services, Laravel, iOS and Android apps, ML/LLM tooling, embedded C, editor plugins, frontend monorepos |
| Categories | new features 119, tech debt 36, refactoring 25, modernize 17, security 2, schema 1 |

The protocol matches the OOD-languages evaluation (`docs/ood_langs_eval.md`): 2 seeds, grep and semantic-search arms, 40 turns, 3,600 s, all 200 repo@commits indexed (coverage 1.0). Every arm returned 400 of 400 rows with no errors and the expected model and weights. Scoring is `scripts/score_heldout_hard.py`, a copy of the OOD scorer with the task set and groupings changed. The task is the unit of analysis; CIs are 95% percentile intervals from 10,000 task-bootstrap and repository-cluster-bootstrap resamples.

## Result

wayfinder-glimmer beats A1 on every retrieval metric in both arms, and both confidence intervals exclude zero for each. The pre-registered-style primary contrast (grep arm, reported recall) is +0.139, with task CI [+0.111, +0.166] and repository-cluster CI [+0.116, +0.164].

| Metric | wayfinder-glimmer grep | A1 grep | wayfinder-glimmer − A1 grep (task / cluster CI) | wayfinder-glimmer − A1 sem (task / cluster CI) |
|---|---|---|---|---|
| Reported recall | 0.452 | 0.313 | +0.139 [+0.111, +0.166] / [+0.116, +0.164] | +0.093 [+0.068, +0.120] / [+0.067, +0.119] |
| Recall@5 | 0.360 | 0.247 | +0.113 [+0.088, +0.139] / [+0.092, +0.136] | +0.068 [+0.045, +0.091] / [+0.044, +0.092] |
| Precision | 0.415 | 0.339 | +0.076 [+0.043, +0.107] / [+0.046, +0.105] | +0.035 [+0.006, +0.064] / [+0.006, +0.063] |
| All gold reported | 0.10 | 0.05 | +0.058 [+0.030, +0.087] | +0.035 [+0.010, +0.062] |
| Files reported | 7.6 | 7.8 | −0.18 [−0.55, +0.19] | +0.07 [−0.27, +0.41] |
| Turns | 9.9 | 34.1 | −24.2 | −24.1 |
| Tokens per task | 268k | 764k | −496k | −492k |

wayfinder-glimmer reports the same number of files as A1, so the recall gain is not padding; precision rises with it. It does so in a quarter of the turns and about a third of the tokens. A1 ends by calling its report tool in only 47% (grep) and 53% (sem) of rollouts; it usually runs out of turns.

By repository, wayfinder-glimmer wins 45 of 47 in the grep arm (1 loss, 1 tie) and 39 of 47 in the semantic arm (7 losses, 1 tie).

## Breakdowns

Recall@5, wayfinder-glimmer minus A1, with task-bootstrap CIs. Groups are small, so these are descriptive.

| Group | Tasks / repos | Grep | Sem |
|---|---|---|---|
| MongoDB ecosystem | 24 / 7 | +0.089 [+0.027, +0.160] | +0.046 [−0.004, +0.097] |
| TS/JS | 50 / 11 | +0.105 [+0.061, +0.152] | +0.070 [+0.027, +0.115] |
| Java/Kotlin | 28 / 8 | +0.111 [+0.052, +0.176] | +0.101 [+0.052, +0.150] |
| Go | 18 / 5 | +0.138 [+0.056, +0.222] | +0.139 [+0.066, +0.220] |
| C# | 18 / 4 | +0.118 [+0.051, +0.190] | +0.029 [−0.040, +0.092] |
| Swift | 16 / 3 | +0.086 [−0.004, +0.188] | −0.048 [−0.105, +0.005] |
| C/C++ | 15 / 3 | +0.081 [−0.011, +0.193] | +0.024 [−0.051, +0.093] |
| New features | 119 / 43 | +0.108 [+0.075, +0.144] | +0.076 [+0.043, +0.109] |
| Tech debt | 36 / 21 | +0.130 [+0.070, +0.195] | +0.027 [−0.018, +0.074] |
| Refactoring | 25 / 18 | +0.118 [+0.053, +0.187] | +0.074 [+0.022, +0.128] |

The grep-arm advantage is positive in every language and every category with more than one task. The semantic arm is weaker in two places. On Swift and iOS repositories wayfinder-glimmer is level with or slightly below A1 (−0.048 on Swift, −0.061 on mobile-ios), and on C#/.NET it is level. The full per-group table, including all 26 repository types, is in `artifacts/heldout_hard/summary.json`.

## Semantic search

Semantic search does not help wayfinder-glimmer on this set: wayfinder-glimmer's sem arm minus its grep arm is −0.006 [−0.022, +0.010] in reported recall, with half a turn and 14k tokens saved. A1 gains +0.039 [+0.017, +0.064] from it. wayfinder-glimmer still leads A1 in the semantic arm, but the product claim that the index improves wayfinder-glimmer's retrieval on unfamiliar codebases is not supported here; it saves cost, not recall. This matches the dev-set pattern and is the first thing to investigate if Scout's semantic tool is a demo point.

## Caveats

The set selects for hard, cross-subsystem commits, so absolute recall (0.36 to 0.45 for wayfinder-glimmer) is lower than on dev; the comparison, not the level, is the result. Task questions come from cleaned commit subjects, and a skim found some vague wording and a few mislabelled categories. Security, schema migration and monolith-splitting tasks are nearly absent (3 tasks), so the category breakdown says nothing about them. Each Swift, C/C++ and Ruby group has 3 repositories, which is too few for a language-level claim.

The set is now spent for wayfinder-glimmer and A1. Any future checkpoint evaluated on it should be compared against these rows, and nothing should be selected on it.

Cost: about $41 of Modal GPU (8 H100 apps, about one hour).
