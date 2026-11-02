# Muse Code with and without Scout: side-by-side demo runbook

This package is for filming two Muse Code panes on the same task, one without Scout and one with the Scout plugin, with wayfinder-glimmer behind Scout. Read the next paragraph before choosing what to say on camera. On the 26-task calibration run (`docs/results/muse_scout_results.md`), Scout did not find more of the right files than stock Muse Code, and no task showed Scout finding the right code where stock missed it and finishing in under two minutes. The video can honestly show that Scout works end to end, delegates correctly, cites line ranges, and used less parent reading and fewer Muse dollars per task. It cannot honestly show that Scout is faster or more accurate.

This runbook is adapted to the single-VM deployment in `deploy/gcp/` (wayfinder-glimmer and the Scout gateway on one A100). The numbers, task choices and the rehearsal below were produced earlier against a Modal-hosted gateway on an H100; the GCP path has not been rehearsed, and an A100 is slower than the H100 behind those numbers.

## The aggregate numbers go next to every example

All 26 TUNE tasks, one seed, Muse's default parent model (`muse-spark-1.3-contributor`), calibration-scale evidence and not a confirmed result:

| Arm | recall@12 | precision@12 | files reported | Scout called | wall median s | Muse $/run (list-price estimate) |
|---|---|---|---|---|---|---|
| Muse Code, stock | 0.600 | 0.381 | 15.0 | 0/26 | 65 | 0.334 |
| Muse Code + spec, built-in search | 0.566 | 0.333 | 16.7 | 0/26 | 77 | 0.249 |
| Muse Code + spec + Scout (wayfinder-glimmer + index) | 0.525 | 0.384 | 10.5 | 26/26 | 189 | 0.237 |

Scout minus stock on recall@12 is -0.075 [-0.140, -0.010]; Scout minus the spec-only arm is -0.041 [-0.116, +0.035]. The parent read 864 thousand input tokens per task in stock and 397 thousand with Scout (the Scout explorer child read another 115 thousand). The `explore` call alone took a median of 126 s on a warm wayfinder-glimmer.

## (a) Task selection

The four candidates below come from the 26-task run. They are illustrative examples chosen from that run, not typical results, and they were chosen for how easy they are to explain, not for how well Scout did. Numbers are recall@12 against the task's gold files, session wall time, number of files in the final answer, and the Muse dollar estimate, in stock / spec-only / Scout order.

| # | Repo and task | Gold | recall@12 | wall s | files | Muse $ |
|---|---|---|---|---|---|---|
| 1 | `mongodb-js/mongosh`: "Add support for fetching a greeting CTA as part of the update flow" | 8 | 0.62 / 0.62 / 0.62 | 50 / 73 / 175 | 10 / 12 / 7 | 0.23 / 0.22 / 0.17 |
| 2 | `Automattic/mongoose`: "Export traced channels as self-contained objects" | 7 | 1.00 / 1.00 / 1.00 | 69 / 69 / 189 | 8 / 9 / 12 | 0.18 / 0.19 / 0.23 |
| 3 | `joomla/joomla-cms`: "Error handling: Adding new shouldUseException()" | 5 | 0.00 / 0.00 / 0.40 | 66 / 77 / 172 | 21 / 14 / 15 | 0.34 / 0.19 / 0.20 |
| 4 | `mongodb/mongo-go-driver`: "Refactor replaceErrors to always wrap errors instead of replacing them" | 10 | 0.90 / 0.80 / 0.80 | 32 / 51 / 148 | 17 / 14 / 12 | 0.17 / 0.16 / 0.22 |

Candidate 1 is the one to film. The question is a plain sentence, the repo is the MongoDB shell, and all three arms land on the same five of eight gold files (`update-notification-manager.ts`, `cli-repl.ts`, `mongosh-repl.ts`, `types/src/index.ts` and the update-manager spec), so the picture is "same answer, different route". Scout's answer is the shortest (7 files against 10) and carries line anchors; the Scout parent read 192 thousand input tokens against 478 thousand for stock and cost $0.17 against $0.23. Scout took 175 s against 50 s.

Candidate 2 is a MongoDB driver with a clean question where every arm finds all 7 gold files; it shows delegation working and nothing else, and Scout was the slower and slightly dearer arm. Candidate 3 is the only task in the 26 where Scout found gold files that stock missed (2 of 5 against 0 of 5, the trait file `LegacyErrorHandlingTrait.php` and its unit test), but the question names an internal method that is hard to explain on camera, and the spec-only arm also found none, so it is a single task and could be noise. Candidate 4 is the fastest Scout run in the MongoDB set and is a small loss on recall (0.80 against 0.90).

Scout wall time under about two minutes was not met by any task: the shortest Scout sessions in the 26 were 145 s (jellyfin) and 148 s (candidate 4), and candidate 1 took 175 s. If the video needs to stay short, run the two panes at the same time and cut the wait, and say that it was cut.

### What to say if there is no visible win

The strongest honest framing is about how Scout fits Muse, not about beating it. Three claims are supported by the 26 rows. First, delegation works: stock Muse never delegated, and with the spec Muse delegated in 26 of 26 runs and Scout's `explore` was called in all 26, with no denial needed. Second, the parent reads less: roughly half the input tokens, 16 tool calls instead of 33, and a shorter ranked answer with `path:line` anchors, at a lower Muse cost per task ($0.237 against $0.334). Third, the index is cheap to keep: re-syncing an unchanged checkout through the gateway embedded zero tokens in every repo synced for the run (18 repos) and took 16 to 26 seconds in the five I timed, and the one-time embedding cost of the demo repos was between $0.11 (mongosh) and $0.80 (mongoose). What not to say: that Scout is faster, that it finds more files, or that it beats Muse's own search. It was about three times slower and its recall was lower by 0.075 with an interval that excludes zero.

## (b) Latency

The demo window needs the VM up and the model loaded. `scripts/demo_warm.sh on [HOURS]` starts the VM (`gcloud compute instances start`), checks gcloud auth first, and arms a detached teardown watcher in the same step (default 4 hours) so a forgotten GPU stops itself. `scripts/demo_warm.sh wait` blocks until a real Scout call returns; vLLM needs 3 to 8 minutes after boot to load the weights. `scripts/demo_warm.sh off` stops the VM and cancels the watcher. The VM bills about $5 an hour while it runs (`deploy/gcp/README.md`). The Scout pane waits about two minutes inside one `explore` call on an H100, and longer on an A100.

Muse Code's `subagent_spawn` is asynchronous, so a background Scout search is possible: the parent spawns the explorer, keeps working, and calls `subagent_wait` when it needs the list. The retrieval spec now has `[delivery] background = true` (shared `scout_spec.py`, copied into the Muse plugin), which adds that instruction to the workflow step. It is covered by a unit test. In the live test on candidate 1 the parent still called `subagent_wait` 3.5 s after the spawn and the run took 182 s against 157 s without the option, because a pure file-finding task gives the parent nothing else to do. The option is off by default. For filming, expect the blocking behaviour: the Scout pane waits about two minutes inside one `explore` call while the stock pane finishes in about a minute.

## (c) Replay

Every calibration session was recorded in full during the study (parent and child threads, tool calls, results, timestamps) so that each candidate had all three arms stored as a fallback if live filming failed. Those recordings are raw transcripts and are not part of this repository. `python3 scripts/demo_replay.py <session directory> [-o out.md]` renders any recorded session directory as a readable timeline; record your own rehearsal with the `--headless` form below and keep it as the fallback.

## (d) Runbook

Two panes, each with its own isolated Muse tree under `~/demo-muse`, so neither touches `~/.config/muse` or `~/.local/share/muse`. All commands run from the `scout/` directory of this repository. Public repositories only: Muse sends the code it reads to Meta.

One-time setup, before the day. Deploy the VM first (`deploy/gcp/README.md`) and issue a scout key with `gateway_vm.py issue-key --tenant demo`. The scout key must belong to the tenant that holds the index. The Meta key is the one Muse Code uses (`META_API_KEY`).

```bash
cd scout
uv sync                                                    # builds .venv/bin/scout (or: uv tool install .)
mkdir -p ~/.config/scout
printf '%s\n' "<scout key from issue-key>" > ~/.config/scout/demo.key && chmod 600 ~/.config/scout/demo.key
printf '%s\n' "<your Muse / Meta API key>"  > ~/.config/scout/meta.key && chmod 600 ~/.config/scout/meta.key
```

On the day, in a terminal off camera.

```bash
cd scout
export GCP_PROJECT=<your-project> GCP_ZONE=us-central1-a
export SCOUT_GATEWAY_URL=https://<host>          # the gateway, e.g. https://34-1-2-3.sslip.io
scripts/demo_warm.sh on 4                                  # start the VM, teardown watcher armed
scripts/demo_warm.sh wait                                  # blocks until the model answers
scripts/demo_panes.sh setup mongodb-js/mongosh 33f85ff2f0f7de0de12dce508a7e515263b384eb
                                                           # clones the repo twice, installs and approves the plugin in the
                                                           # scout pane only, writes its settings, then `scout sync --wait`
                                                           # (login is the key file; `scout login` is not needed here)
scripts/demo_panes.sh status                               # expect "state":"ready" and the plugin enabled
```

On a gateway that already holds the index, setup takes about 20 seconds because sync embeds 0 tokens. On a fresh gateway (new master key) the first sync embeds the whole repository (see the costs at the end). Do not start filming until `wait` returns.

The prompt, pasted into both panes:

```text
Find the files in this repository that are relevant to the task below, and report them with a one-line reason each. Do not change anything.

Task: Add support for fetching a greeting CTA as part of the update flow
```

Left pane, Muse Code without Scout, and right pane, Muse Code with Scout:

```bash
scripts/demo_panes.sh stock        # then paste the prompt
scripts/demo_panes.sh scout        # then paste the prompt, within a second of the other pane
```

Both open the Muse TUI in their own clone with `--trust-workspace`. The same commands with `--headless "<prompt>"` run the session without the TUI and print JSON events; the earlier rehearsal used that form because a TUI needs a terminal, and the TUI itself was not driven during rehearsal. Do one dry run on the filming machine and watch for any first-start prompt from Muse. Show the line "Built with Muse Spark Open" on screen during the demo (a slide or the video description). wayfinder-glimmer is a Muse Spark Open derivative and the license requires it wherever a product uses the model (`NOTICE`).

What to point at. In the left pane, the stock parent searches and reads on its own (about 30 tool calls) and finishes in about a minute with 10 to 15 files. In the right pane, the first thing after the prompt is a `subagent_spawn` of `scout-wayfinder/scout-explorer`: the spec told the parent to delegate before reading code. Then one long `explore` call (about two minutes; this is wayfinder-glimmer searching the index), a handful of reads, and an answer of about 7 files with `path:line` anchors and a reason each. Say the time difference out loud. For the cost line, run `/scout-wayfinder:status` in the right pane (index freshness, lane, endpoint state) and quote the index cost: embedding mongosh once cost $0.11 (636,704 voyage-code-4 tokens, 152 s), and a re-sync of an unchanged tree embeds nothing. The Muse API cost of the session is not shown in the TUI; the calibration rows for this task show about $0.23 stock and $0.17 Scout by list price, taken from the run's token counts. The slash commands work in the TUI and do not expand under `muse exec`, so they were not exercised in rehearsal.

The same runbook works for the other candidates by swapping the repo, commit and task line: mongoose `Automattic/mongoose` at `030504d9ad926c099eb4615e1d601e91c38f5853` ("Export traced channels as self-contained objects"), joomla `joomla/joomla-cms` at `fb908bac4e03e4fd0485efe10cd980a4a97dde71` ("Error handling: Adding new shouldUseException()"), go driver `mongodb/mongo-go-driver` at `4a4e6dcfbe4bb57b95772d3ea1339211a3ab5ab6` ("Refactor replaceErrors to always wrap errors instead of replacing them").

One thing that will bite: `muse plugins install` copies the plugin into the pane's cache, so editing the plugin source does nothing until `muse plugins update scout-wayfinder` (and `approve`) is run with the pane's XDG variables set. `demo_panes.sh setup` installs fresh each time.

Teardown, after filming:

```bash
scripts/demo_panes.sh teardown                             # removes both panes' Muse state and stops the watcher
scripts/demo_warm.sh off                                   # stops the VM (GPU billing ends; the disk keeps billing)
scripts/demo_warm.sh status                                # confirm TERMINATED
rm -f ~/.config/scout/demo.key ~/.config/scout/meta.key    # if you do not want the key files left on disk
```

The rehearsal was run end to end with these commands on candidate 1 against the earlier Modal gateway (setup, status, both panes headless, the background variant, teardown) and the one thing it fixed was the plugin listing in `status`.

## (e) Pre-indexed repos

Indexing is per gateway: the VM reads an index only if it runs with the same `SCOUT_MASTER_KEY`, `SCOUT_PEPPER` and `SCOUT_KEYS_NS` as the gateway that wrote it (`deploy/gcp/README.md`, step 1). With those values reused, the four candidate repos were already indexed during the calibration run (`scout sync --wait` reported state ready and 0 embedded tokens for every repo). With new values, run `scout sync --wait` once per repo before the day. One-time embedding costs measured during the study: mongosh $0.11, mongo-go-driver $0.34, mongoose $0.80, joomla $0.30.
