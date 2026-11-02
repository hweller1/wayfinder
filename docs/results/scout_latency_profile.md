# Why Scout is about 5x slower than built-in Explore in Claude Code

Scout's median wall time on the 50-task retrieval-spec ablation is 164 s against 35 s for built-in Explore. About 80% of the Scout run (roughly 132 s) is spent inside one `explore` MCP call, and almost all of that is wayfinder-glimmer taking turns, not cold starts and not the Sonnet subagent that wraps it. This note decomposes the wall time from the stored rows, labels every estimate, and ranks three fixes. Analysis only: CPU, existing data, no Modal, no the LLM gateway calls. The script is `scripts/scout_latency_profile.py`; the figure is `docs/figures/s11b_latency.png`.

## What was recorded and what was not

The cc50 rows (voyage-scout `artifacts/eval/cc50/full/`, arms `cc_scout_spec` and `cc_explore_spec`, 50 rows each, 1 seed) carry `wall_s` (whole `claude -p` session), `explore_s` (the Agent tool's `totalDurationMs`, so the subagent's duration as the parent sees it), the subagent's tool sequence and counts, parent and subagent turns, and a hook log line with the SessionStart timestamp. The stream-json transcripts were not stored, so there are no per-turn timestamps in the cc50 data. Nothing from the gateway or from Modal was kept locally for those runs, and the Scout `explore` result's own `usage` block (turns, wall_ms) is not in the rows. The parent-side residual (wall minus explore_s) is measured; the splits below it are estimates.

Four sources fill the gaps, each used only where the row says nothing:

1. The `cc_explore_spec` arm, which has the same Sonnet subagent shape with no MCP call. A linear fit on its 50 rows gives Sonnet subagent time = 11.5 s + 1.83 s per tool call. This is the estimate for the Sonnet layer inside Scout.
2. Direct wayfinder-glimmer runs with a per-turn log (`artifacts/heldout_hard/shards/v3_s*/sft_v3_sem.jsonl`, 400 runs, semantic arm, product profile). They give the turns per explore run, the turn latency and the tool-time share. They ran at a different concurrency and on a different task set, so they set shape, not seconds.
3. A one-call gateway smoke (`artifacts/live/gateway_smoke_v3.json`, gorilla/mux): 6 wayfinder-glimmer turns in 102.9 s, first request to the gateway 1.35 s. Used as the gateway round trip.
4. `docs/serving_v3.md`: cold start median 155.5 s from zero, warm tool-call turn 6.2 s median at concurrency 1 and 6.4 s at concurrency 4.

## Stacked breakdown (seconds per run, n=50 per arm)

Each cell is a separate percentile of that component, so p90 values do not add up to the wall p90. Rows marked (est.) are not recorded; see the sources above.

| Component | cc_explore_spec median | cc_explore_spec p90 | cc_scout_spec median | cc_scout_spec p90 | Status |
|---|---|---|---|---|---|
| Parent turns before delegation | 4.9 | 6.1 | 5.2 | 7.1 | est. (45% of the measured residual) |
| Delegation overhead (Agent tool start) | inside the row above | | inside the row above | | not separately recorded |
| Sonnet subagent's own turns | 21.9 | 30.3 | 15.2 | 22.5 | built-in Explore: measured (`explore_s`); Scout: est. from the explore-arm fit |
| `explore` call, gateway round trip | n/a | n/a | 1.4 | 1.4 | est. (smoke first request) |
| `explore` call, tool time (grep, semantic, read) | n/a | n/a | 3.9 | 3.9 | est. (median wall minus model latency, direct wayfinder-glimmer runs) |
| `explore` call, wayfinder-glimmer model turns | n/a | n/a | 126.7 | 151.0 | est. (`explore_s` minus the three rows above) |
| Parent time after Scout returns | 6.0 | 7.5 | 6.4 | 8.7 | est. (55% of the measured residual) |
| Wall time (measured) | 34.7 | 45.1 | 163.8 | 187.3 | measured |
| Measured parent residual (wall minus `explore_s`) | 10.9 | 13.6 | 11.7 | 15.7 | measured |
| Scout `explore_s` (Agent tool duration) | 21.9 | 30.3 | 150.9 | 175.3 | measured |

The Scout component medians sum to 159 s against the 164 s measured median, because medians of parts do not add. The subagent in the Scout arm made a median of 2 non-`explore` tool calls (p90 6) after its one `explore` call, 49 of 50 runs called `explore` exactly once, and the Scout subagent's own output was a median of 2,310 characters.

Two checks that the picture is stable: the measured spread of `explore_s` is narrow (min 99 s, p10 137 s, median 151 s, p90 175 s), and its correlation with the number of Sonnet tool calls after `explore` is 0.11. The wall time is set by the `explore` call, and the Sonnet work around it barely moves it.

## Attribution

Shares are of the 164 s Scout median wall time.

| Cause | Seconds (median) | Share | Basis |
|---|---|---|---|
| Cold starts | about 0, bounded under 5 | about 0% observed | see below |
| Sonnet middle layer | 15 to 21 | 9% to 13% | est.; 15.2 s of subagent turns, plus up to 6 s of the post-Scout parent relay |
| wayfinder-glimmer multi-turn loop (model turns) | 127 | 77% | est. |
| wayfinder-glimmer loop including gateway and tool time | 132 | 80% | est. |
| Parent turns not tied to either | about 6 | 4% | measured residual less the relay share |

**Cold starts.** No cold start is visible in these rows. wayfinder-glimmer was warmed before the run and held up by a watchdog (`/tmp/v3_watchdog.sh` scales it down only after 25 idle minutes), so the 155 s cold start in `docs/serving_v3.md` never occurred during the run. The first four sessions, which started within 10 s of each other against the freshly warmed stack, had `explore_s` of 143, 169, 149 and 141 s, against a run median of 151 s, so there is no first-call penalty. Windowed medians of `explore_s` in 10-minute buckets over the 55-minute run are 155, 151, 143, 160, 157 and 154 s, with no trend and no per-window spike. The only cold component left would be the gateway's CPU container (a few seconds when it scales from zero), which the data cannot separate from noise. The stated caveat in `docs/cc_scout_ablation.md` that Scout wall time includes cold semantic searches therefore does not explain the gap; the time goes to per-turn latency instead (see below). A real demo that starts from zero would add the 155 s once: see fix 3.

**Sonnet middle layer.** Built-in Explore's Sonnet subagent takes 21.9 s median including its own search. The Scout arm's Sonnet subagent spends 15.2 s outside the `explore` call (estimate), reading a few ranges and writing a 2.3k-character handback that the parent then repeats. That is the entire cost of the middle layer on the clock. It also cost $0.019 per run, though the arm is still cheaper than Explore in dollars.

**wayfinder-glimmer's loop.** The `explore` call is about 132 s, which at the 17 s per turn seen in the gateway smoke (102.9 s for 6 turns, tools and gateway included) is about 8 turns. Direct wayfinder-glimmer runs show a median of 9 turns (p90 13) and 3.2 tool calls per turn, with 79% of assistant turns already issuing more than one call. The two sources agree on 8 to 9 turns. Tool time is small: the median gap between wall time and summed model latency in the direct runs is 3.9 s (p90 9.7 s). A turn's latency regresses as 5.4 s + 0.055 s per output token + 0.00013 s per input token (least-squares fit over 3,777 turns), so for the median turn of 476 output tokens and 26k input tokens decode is about 26 s of a 32 s turn under heavy concurrency; the cc50 turns are shorter, which is why they cost 15 to 17 s.

## Top 3 fixes, ranked by expected seconds saved per run

| Rank | Fix | Expected saving (median run) | Basis |
|---|---|---|---|
| 1 | Turn budget on `explore` (cap at 6 turns, forced report on the next) | about 60 s (about 40 s at a cap of 8) | est. from direct wayfinder-glimmer runs |
| 2 | Pass-through handoff: parent calls the Scout `explore` tool itself, no Sonnet subagent | 15 to 21 s | est. from the explore-arm fit |
| 3 | Cut wayfinder-glimmer decode per turn (shorter reasoning, tighter report schema), with `min_containers=1` for demos | about 25 s per run; 155 s on the first call after idle | est.; cold start measured |

**1. Turn budget.** In the 400 direct wayfinder-glimmer runs, the mean share of a run's model latency that falls after turn N is 69% after turn 4, 48% after turn 6 and 27% after turn 8. Applied to the 127 s model component, a cap of 6 saves about 60 s and a cap of 8 saves about 34 s. The price is retrieval: runs that needed more than 6 turns have lower reported recall (0.42 against 0.60 for runs of 6 or fewer), but that gap is confounded by task difficulty, so it overstates the cost of truncation and does not measure it. This fix therefore needs a calibration batch before any full run (about 5 tasks per condition) and a check that recall@12 does not fall outside the interval around the current 0.519. The product profile already carries the nudge "report as soon as the evidence covers the brief" and a 40-turn ceiling, and that ceiling is almost never reached (0.5% of runs), so the budget is not binding today.

**2. Pass-through handoff.** The Sonnet layer costs 15 s of subagent turns that add no search, plus the parent's relay of a 2.3k-character handback. Letting the parent call `mcp__plugin_scout_scout__explore` directly removes the first and shortens the second. Evidence that the layer is overhead and not search: the correlation between `explore_s` and the subagent's post-`explore` tool calls is 0.11, and 49 of 50 runs make exactly one `explore` call. The risk is delegation. The spec arms delegate 98% to 100% because the workflow step tells the parent to hand off to a subagent; a direct-call arm has to re-earn that rate in calibration. A saving that is small against the 164 s but is the only fix that does not touch retrieval quality.

**3. Fewer output tokens per turn, plus a warm container for demos.** Decode dominates a turn (0.055 s per output token against 0.00013 s per input token). A 25% cut in output tokens per turn, from shorter reasoning and a terser tool-call format, would remove about 20% of the 127 s, or about 25 s. This is the least certain number here because the output-token mix inside cc50 explore calls is not recorded. Separately, `min_containers=1` saves nothing on this benchmark, because wayfinder-glimmer was already warm, but it removes the 155.5 s median cold start (runs 115 to 185 s) from the first call after the 10-minute scaledown window, which is the biggest single latency a demo user sees; it costs $3.95 per hour (`docs/serving_v3.md`). Parallel tool calls and a smaller `k` are not on the list: 79% of wayfinder-glimmer's turns are already parallel, and tool time is under 4 s median, so neither can move more than a few seconds.

## Caveats

One seed and 50 tasks. The wayfinder-glimmer turn counts, tool time and decode figures come from a different run (heldout hard v1, 400 runs, product profile) than the cc50 wall times, so the turn counts and per-turn seconds are shapes that agree across two independent sources, not a per-run measurement. The 45/55 split of the parent residual and the 1.4 s gateway round trip are placeholders and move no component by more than a few seconds. To replace the estimates, store the stream-json transcripts and the `explore` result's `usage.turns` and `usage.wall_ms` in the next cc50 run (the core already returns both in `ExploreResult.usage`).
