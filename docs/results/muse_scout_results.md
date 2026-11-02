# Muse Code with Scout: calibration results

This is a calibration-scale result for the retrieval spec and the Scout plugin on Muse Code 1.4.14, run on 2026-10-31 under the pre-registered plan in the Muse Scout study plan (not in this release) (committed as b01640a before any arm run). On the 26 TUNE tasks, one seed, Scout did not find more of the right files than stock Muse Code: recall@12 was 0.525 with Scout against 0.600 for stock, a paired difference of -0.075 [-0.140, -0.010], and the run took about three times as long. The spec itself worked as a delivery mechanism: stock Muse delegated in 0 of 26 runs and both spec arms in 26 of 26, and Scout's `explore` was called in all 26 Scout runs. This does not confirm or refute anything beyond these tasks; the intervals are wide and CONFIRM was not touched.

## Setup as run

The tasks are the 26 TUNE tasks of `data/cc_scout_hillclimb_split.json` (the brief said 25; the file holds 26 and all were used). The parent model is Muse's default, recorded in every row as `muse-spark-1.3-contributor` at the default effort (high). The explorer agent in the Scout arm runs `muse-spark-1.3` at effort low, and Scout's `explore` runs wayfinder-glimmer. Every session used the same free-parent prompt, `--disable-write --disable-shell`, and an isolated XDG tree under `/tmp/muse-spec/xdg` that was deleted afterwards; `~/.config/muse` and `~/.local/share/muse` were not touched. Width was 2 across repos, with wayfinder-glimmer held warm (`min_containers=1`) from 12:34 to 14:10 PT. Every one of the 78 rows is valid under the plan's rules (exit 0, model recorded, paths parsed, spec context and workflow step fired in both spec arms, Scout called in every Scout row), and none was rerun. Rows, transcripts and per-run session logs are in voyage-scout `artifacts/live/muse_spec/`; the harness is `eval/harness/muse_spec.py`.

## Calibration (first 5 tasks, all three arms)

The calibration gates were met: 15 of 15 rows valid, Scout called in 5 of 5 Scout rows, stock rows carried no plugin and no spec output, and the projected full-run cost was acceptable. The 5 rows are part of the 26.

| Arm | recall@12 | precision@12 | delegated | Scout called | wall median s | Muse $/run |
|---|---|---|---|---|---|---|
| muse_stock | 0.458 | 0.193 | 0/5 | 0/5 | 66 | 0.338 |
| muse_explore_spec | 0.487 | 0.193 | 5/5 | 0/5 | 77 | 0.276 |
| muse_scout_spec | 0.434 | 0.311 | 5/5 | 5/5 | 189 | 0.243 |

## All 26 TUNE tasks

| Arm | n | recall@5 | recall@12 | precision@12 | files reported | delegated | Scout called | wall median (p90) s | Scout explore median s | Muse $/run |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_stock | 26 | 0.413 | 0.600 | 0.381 | 15.0 | 0/26 | 0/26 | 65 (115) | n/a | 0.334 |
| muse_explore_spec | 26 | 0.394 | 0.566 | 0.333 | 16.7 | 26/26 | 0/26 | 77 (109) | n/a | 0.249 |
| muse_scout_spec | 26 | 0.396 | 0.525 | 0.384 | 10.5 | 26/26 | 26/26 | 189 (231) | 126.2 | 0.237 |

Wall seconds are end to end per session at width 2. Modal GPU dollars are not per call and Voyage dollars were zero.

Paired contrasts, task bootstrap with 10,000 resamples (seed 20261025), 95% interval:

| Contrast | recall@5 | recall@12 | precision@12 |
|---|---|---|---|
| muse_scout_spec minus muse_stock | -0.017 [-0.074, +0.039] | -0.075 [-0.140, -0.010] | +0.003 [-0.047, +0.051] |
| muse_scout_spec minus muse_explore_spec | +0.002 [-0.069, +0.075] | -0.041 [-0.116, +0.035] | +0.051 [+0.000, +0.102] |

Repo-cluster intervals (sensitivity): muse_scout_spec minus muse_stock: recall@12 [-0.130, -0.023]; muse_scout_spec minus muse_explore_spec: recall@12 [-0.105, +0.030].

## Reading

Scout is behind stock Muse on recall@12 and the interval excludes zero (-0.075 [-0.140, -0.010], repo-cluster [-0.130, -0.023]), so the honest reading is that on these tasks Scout lost some recall. Part of the gap is length: stock reports 15.0 files and the Scout arm 10.5, and recall@12 rewards listing more. Precision@12 is level with stock (+0.003 [-0.047, +0.051]) and slightly above the spec-only arm (+0.051 [+0.000, +0.102]). Against the spec-only arm, which gets the same delegate-first instruction but searches with Muse's own tools, Scout is -0.041 [-0.116, +0.035] on recall@12, which does not separate from zero. recall@5 is flat across all three arms.

The spec's delivery did what it was built to do: 0 of 26 stock runs delegated and 26 of 26 spec runs did, with no search denial needed (the workflow step alone was enough). What delegation changed was where the work happens. The stock parent made 32.5 tool calls and read 864 thousand input tokens; with Scout the parent made 16.2 calls and read 397 thousand, and the explorer child 7.9 calls and 115 thousand. That is why the Muse API estimate per run is lower with Scout ($0.237) than stock ($0.334), and also lower than the spec-only arm ($0.249), whose general-purpose child makes 26.7 calls on the parent's model.

The cost is time. The median Scout session took 189 s against 65 s for stock and 77 s for the spec-only arm, and the `explore` call alone had a median of 126 s on a warm wayfinder-glimmer under width-2 load. This is the same direction as the Claude Code study (the spec hill-climb write-up, not in this release), where Scout did not beat built-in Explore with the same spec and was about five times slower.

## Per task

| Repo | Task (id suffix) | Gold | stock r@12 | explore_spec r@12 | scout_spec r@12 | stock s | explore_spec s | scout_spec s |
|---|---|---|---|---|---|---|---|---|
| akaunting/akaunting | 2d5fb3 | 7 | 0.00 | 0.14 | 0.14 | 64 | 72 | 206 |
| solidusio/solidus | 3c5cab | 3 | 0.67 | 0.67 | 0.33 | 131 | 106 | 254 |
| joomla/joomla-cms | 5a3ec9 | 3 | 1.00 | 1.00 | 0.67 | 76 | 131 | 189 |
| mongodb-js/mongosh | 0726b4 | 8 | 0.62 | 0.62 | 0.62 | 50 | 73 | 175 |
| joomla/joomla-cms | 0b434f | 5 | 0.00 | 0.00 | 0.40 | 66 | 77 | 172 |
| mongodb/mongo-c-driver | 334796 | 6 | 0.00 | 0.00 | 0.00 | 77 | 93 | 227 |
| mongodb/mongo-csharp-driver | b50e42 | 4 | 0.50 | 0.00 | 0.50 | 89 | 78 | 253 |
| mongodb/mongodb-atlas-cli | 657937 | 6 | 0.67 | 0.67 | 0.17 | 114 | 102 | 172 |
| nopSolutions/nopCommerce | 83c2fb | 7 | 0.86 | 0.86 | 0.71 | 56 | 105 | 167 |
| spring-projects/spring-security | 9f02bc | 5 | 1.00 | 1.00 | 1.00 | 43 | 111 | 185 |
| wagtail/wagtail | bcb2b2 | 4 | 0.75 | 0.75 | 0.75 | 64 | 104 | 199 |
| home-assistant/iOS | c8cd59 | 5 | 0.40 | 0.40 | 0.40 | 45 | 69 | 220 |
| mongodb-js/mongosh | 44051f | 11 | 0.27 | 0.09 | 0.09 | 116 | 135 | 227 |
| Automattic/mongoose | f700de | 4 | 0.50 | 0.50 | 0.25 | 46 | 77 | 184 |
| Automattic/mongoose | 061b69 | 6 | 0.67 | 0.67 | 0.50 | 76 | 68 | 195 |
| Automattic/mongoose | b9c8bc | 4 | 0.50 | 0.50 | 0.50 | 45 | 53 | 156 |
| excalidraw/excalidraw | 4f02b3 | 11 | 0.91 | 0.73 | 0.73 | 55 | 76 | 171 |
| home-assistant/iOS | 051846 | 4 | 1.00 | 1.00 | 1.00 | 45 | 95 | 152 |
| rust-lang/rust-analyzer | dfb876 | 12 | 0.25 | 0.00 | 0.08 | 204 | 98 | 209 |
| spring-projects/spring-ai | 8987d6 | 9 | 0.78 | 0.78 | 0.78 | 46 | 78 | 234 |
| spring-projects/spring-ai | c86d48 | 8 | 0.50 | 0.62 | 0.50 | 66 | 73 | 180 |
| Automattic/mongoose | 1d0265 | 8 | 0.50 | 0.38 | 0.38 | 71 | 61 | 193 |
| jellyfin/jellyfin | 9a726e | 10 | 0.60 | 0.80 | 0.60 | 36 | 58 | 145 |
| Automattic/mongoose | ac013e | 7 | 1.00 | 1.00 | 1.00 | 69 | 69 | 189 |
| mongodb/mongo-go-driver | 87c305 | 10 | 0.90 | 0.80 | 0.80 | 32 | 51 | 148 |
| vitejs/vite | caa9ea | 4 | 0.75 | 0.75 | 0.75 | 86 | 58 | 198 |

## What these numbers do not say

One seed and 26 tasks. A second seed was not run. The parent here is Muse's default model at high effort, not the low-effort setting of the earlier 5-task Muse calibration and not the Sonnet 5.5 parent of the Claude Code studies, so absolute values do not transfer. Muse has no Explore subagent, so the spec-only arm delegates to `general-purpose` on the parent's model and is not an exact twin of the Claude Code `cc_explore_spec` arm. `--disable-write --disable-shell` removes shell grep in every arm. Recall is scored on the parent's reported list. The mined examples describe earlier commits of the same repository, which gives both spec arms a vocabulary hint stock does not have. Concurrency (width 2 on one H100) moves Scout's wall time. There is no coding-outcome metric.

## Cost (separate units)

| Item | Amount | Basis |
|---|---|---|
| Voyage | $0 | the re-sync of each TUNE checkout through the gateway embedded 0 tokens |

Deviations from the plan: none after the first run, apart from the 26-versus-25 task count noted above, which the plan already stated.
