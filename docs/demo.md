# Demo runbook: Muse Code with and without Wayfinder

This is the runbook for the demo of wayfinder-qwen. Two Muse Code panes work the same task, one without Scout and one with the Scout plugin, with wayfinder-qwen behind Scout. The main agent in both panes is `muse-spark-1.3-contributor`. The demo uses three tasks, from the repositories jellyfin, excalidraw and mongoose, and each task is run 5 times per arm.

## What the demo shows, and how to label it

The video shows the best of the 5 runs per task, chosen after the batch finished. It is a selection and not a typical run, and it must be labelled that way on screen or in the description. The representative result is the 26-task study: with Wayfinder the main agent costs $0.195 per task against $0.342, reads 441K input tokens against 887K, finds files within noise on recall@12 (0.536 against 0.590), is lower on recall@5 (0.383 against 0.449) and is about 25 s slower in the median (n = 26, one run each). See [results/README.md](results/README.md), section 5. The demo can honestly show that Wayfinder works end to end, delegates, cites line ranges and costs less. It cannot honestly show that Wayfinder is faster or finds more files. On excalidraw it did worse in 4 of 5 batch runs.

## Commands

| Command | What it does | Where it is |
|---|---|---|
| `scripts/demo_batch.sh` | Runs the 5 runs per task for the three tasks, automatically | Not in this repository. It belongs to the private voyage-scout checkout and was not copied here |
| `scripts/demo_replay_three.sh` with `SPEEDS="8 8 8"` | Replays the three chosen runs at 8x speed for the video | Not in this repository, same reason |
| `scout/scripts/demo_panes.sh` | Sets up and opens the two Muse Code panes (steps below) | In this repository |
| `scout/scripts/demo_replay.py <session dir>` | Renders a recorded session directory as a readable timeline | In this repository |
| `scout/scripts/demo_warm.sh` | Starts and stops the GCP VM. The GCP kit serves wayfinder-glimmer, so it does not apply to the qwen demo; serve wayfinder-qwen with `deploy/modal/serve_qwen.py` ([MODEL_CARD_QWEN.md](MODEL_CARD_QWEN.md)) | In this repository, glimmer-era |

The best-of-5 labelling rule: for each task the shown run is the best of the 5, picked after seeing all 5, and the video says so. The median run for each task is the fallback and the honest summary of a typical run. The median runs are listed in the demo speaker notes, which are part of the private deck source and not in this repository. The raw takes (one `result.json` per run) are in the private voyage-scout checkout under `artifacts/demo/takes/`. I have not copied either here, so I cannot point to a file in this repository for them.

The five excalidraw runs found 10, 10, 10, 5 and 8 of 11 right files without Wayfinder and 3, 8, 4, 3 and 8 with it (median 10 against 4); the source is the demo takes above, summarised in [MODEL_CARD_QWEN.md](MODEL_CARD_QWEN.md). The demo task for each repository is set in the batch script, which is not here, so this runbook does not restate the three task lines. The older single-task rehearsal (mongosh) used the prompt below.

## Setup and filming

Two panes, each with its own isolated Muse tree under `~/demo-muse`, so neither touches `~/.config/muse` or `~/.local/share/muse`. All commands run from the `scout/` directory of this repository. Public repositories only: Muse sends the code it reads to Meta.

One-time setup, before the day. Serve wayfinder-qwen and the gateway first ([RUN.md](RUN.md) and [MODEL_CARD_QWEN.md](MODEL_CARD_QWEN.md); the GCP kit in `deploy/gcp/` serves wayfinder-glimmer) and issue a scout key for the tenant (`gateway_vm.py issue-key --tenant demo` on the GCP gateway). The scout key must belong to the tenant that holds the index. The Meta key is the one Muse Code uses (`META_API_KEY`).

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
export SCOUT_GATEWAY_URL=https://<host>          # your gateway; the model must already answer
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

Both open the Muse TUI in their own clone with `--trust-workspace`. The same commands with `--headless "<prompt>"` run the session without the TUI and print JSON events; the earlier rehearsal used that form because a TUI needs a terminal, and the TUI itself was not driven during rehearsal. Do one dry run on the filming machine and watch for any first-start prompt from Muse. Show the line "Built with Muse Spark Open" on screen during the demo (a slide or the video description). wayfinder-qwen is a Muse Spark Open derivative and the license requires it wherever a product uses the model (`NOTICE`).

What to point at. In the left pane, the stock parent searches and reads on its own. In the right pane, the first thing after the prompt is a `subagent_spawn` of `scout-wayfinder/scout-explorer`: the spec told the parent to delegate before reading code. Then one `explore` call (the wayfinder-qwen search), a handful of reads, and an answer with `path:line` anchors and a reason each. Say any time difference out loud: Wayfinder is slower, about 25 s in the median over 26 tasks. For the cost line, run `/scout-wayfinder:status` in the right pane (index freshness, lane, endpoint state) and quote the 26-task cost, $0.195 against $0.342 per task by list price. A re-sync of an unchanged tree embeds nothing; embedding mongosh once cost $0.11 (636,704 voyage-code-4 tokens, 152 s). The slash commands work in the TUI and do not expand under `muse exec`.

The same commands work for any repository: swap the repo, commit and task line. The jellyfin, excalidraw and mongoose commits and task lines used for the video are set in `scripts/demo_batch.sh`, which is not in this repository, so they are not restated here. The mongosh prompt above is the one this repository's rehearsal used.

One thing that will bite: `muse plugins install` copies the plugin into the pane's cache, so editing the plugin source does nothing until `muse plugins update scout-wayfinder` (and `approve`) is run with the pane's XDG variables set. `demo_panes.sh setup` installs fresh each time.

Teardown, after filming (stop the Modal apps as described in [RUN.md](RUN.md), "Stop paying"):

```bash
scripts/demo_panes.sh teardown                             # removes both panes' Muse state
rm -f ~/.config/scout/demo.key ~/.config/scout/meta.key    # if you do not want the key files left on disk
```

The commands in this section were rehearsed earlier end to end on one task against a Modal gateway, with the glimmer-era model behind it. They have not been rerun with wayfinder-qwen from this repository.


## Pre-indexed repos

Indexing is per gateway: a gateway reads an index only if it runs with the same `SCOUT_MASTER_KEY`, `SCOUT_PEPPER` and `SCOUT_KEYS_NS` as the gateway that wrote it. With those values reused, `scout sync --wait` reports state ready and 0 embedded tokens. With new values, run `scout sync --wait` once per repository before the day. One-time embedding costs measured earlier: mongosh $0.11, mongoose $0.80; the jellyfin and excalidraw costs were not measured in this repository's data.
