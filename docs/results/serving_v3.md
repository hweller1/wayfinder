# Serving Wayfinder-Glimmer: cold start, warm path and cost

`serve/wayfinder_v3.py` serves wayfinder-glimmer (the SFT Muse-Glimmer-30B; served id `wayfinder-v3` and Modal app `wayfinder-v3-serve` unchanged) on one H100 with vLLM 0.32, the `muse_glimmer` tool and reasoning parsers and the bearer proxy. The API, parsers, retained reasoning, `V3_SERVED_NAME` and `V3_SECRET` are unchanged. This note records what changed to cut the cold start, the measurements, and what the demo should run.

## What changed

The A1 techniques are now the default path (`V3_SNAPSHOT=1`). The vLLM engine starts inside `@modal.enter(snap=True)` with `--enable-sleep-mode`, serves real-shape warmup traffic (scout system prompt and tool schemas, a 20k-token prefill, a tool call, a 16-way burst), commits the torch.compile, inductor, triton and flashinfer caches to Volume `wayfinder-cache` (first run only), and is put to sleep (level 1) before Modal takes the CPU and GPU memory snapshot. A restore wakes the engine in about 2 s, re-primes the prefix cache with a 12k-token warmup (about 10 s), and then starts the bearer proxy. Weights stay on the `wayfinder-data` Volume. The container reads them once, at snapshot creation, not on restore. `V3_SNAPSHOT=0` restores the previous plain start.

`GET https://myworkspace--wayfinder-v3-serve-warm.us-east.modal.run/` with `Authorization: Bearer $V3_API_KEY` is the SessionStart hook call. It runs on a small CPU function, spawns a no-op input on the GPU class and returns `{"warming": true}` in under a second when the CPU function is already up (about 3 s if it must itself cold start, still non-blocking for a hook with a 5 s curl timeout). It returns 401 without the key. Any request to the model URL, including `/health` or `/v1/models`, also starts the cold start, but it blocks until the container is up. Hooks should call `/warm` and not wait.

## Measurements (5 runs from zero, `scripts/v3_coldstart_bench.py`, `artifacts/serve/v3_coldstart_bench.json`)

Before is commit 198c018 and `artifacts/serve/v3_startup_cold.json`. Cold start is the time from the first request, sent to a stopped app, to its first streamed token.

| Metric | Before | After (5 runs) |
|---|---|---|
| Cold start to first token | 220 s (ready, startup log), 323 s (client, `v3_check_deployed.json`) | median 155.5 s; runs 184.6, 155.3, 176.9, 155.5, 115.5 |
| Warm scout tool-call turn, concurrency 1 (median of 15) | not measured | 6.2 s (2.3 s for a short call; up to 13 s for long reasoning) |
| Warm scout tool-call turn, concurrency 4 (median of 20) | 5.4 s median over 20 briefs at c4 | 6.4 s |
| Warm TTFT, short prompt (median of 5 runs) | 0.47 s | 0.31 to 0.46 s |
| Tool-call parse, 20 briefs | 19/20 (0.95), turn 2 19/19, no leaked markup | 19/20 (0.95), turn 2 19/19, no leaked markup, 20/20 with reasoning |

The warm numbers match the old ones within noise: the engine is the same. The one unparsed call is a schema failure on one brief, as before.

The cold start is a wash of two effects. A restore alone takes 12 s (2 s wake plus 10 s warmup). But Modal rebuilt the memory snapshot on each scale from zero (a `snapshot_create` of 83 to 112 s with warm compile caches precedes each `snapshot_restore`), so a cold start still pays boot, weight load (about 30 s) and CUDA-graph capture. The persistent caches took torch.compile from 53 s (plain start, first cold start) to 0 s, and capture from 84 s to about 8 s. Before the caches were warm the first snapshot creation took 345 s. I tried a read-only cache mount (`V3_CACHE_RO=1`) to keep the Volume version stable between starts and reuse the snapshot. It did not stop the rebuild, so it is off by default. Cold starts after the last redeploy measured 147 to 266 s, because a redeploy invalidates the snapshot and the container that follows rebuilds it. Getting under 60 s needs the snapshot to be reused, which we did not manage here. A1 sees 1 to 2 minutes with the same mechanism. Treat the honest figure for wayfinder-glimmer as 2 to 3 minutes from zero, with a median of 2.6.

## Cost

The H100 is billed at about $3.95/h while a container is up (the figure in `docs/STATUS.md`). A scaled-to-zero app bills nothing while idle.

| Mode | Cost |
|---|---|
| `min_containers=1`, always warm | $3.95/h, $94.80/day |
| Scale to zero, `scaledown_window=600` | one session = cold start (about 155 s, $0.17) + 10 min idle tail ($0.66) + active time, so about $0.85 plus $0.066 per active minute |
| Break-even | warm is cheaper than scale to zero above about 4.6 isolated sessions per hour |

## Demo-day recommendation

Run `scripts/v3_warm.sh on` (redeploys with `min_containers=1`) about 15 minutes before the demo window, and `scripts/v3_warm.sh off` right after. For an 8-hour window this costs 8 x $3.95 = $31.60, plus 15 minutes of lead time ($0.99), so $32.59 total. A 2-hour session costs $7.90 plus the lead. Note that redeploying invalidates the snapshot, so run it before the window, not during it. With the app warm the first scout call takes about 0.5 s and a tool-call turn takes 2 to 6 s. If you do not pin a container, have the SessionStart hook call `/warm` so the 2 to 3 minute cold start overlaps the user's first minutes, and lengthen `V3_SCALEDOWN_S` (the default is 600) if the demo has gaps.
