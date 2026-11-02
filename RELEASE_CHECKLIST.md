# Release checklist (for the coordinator)

Local repo only. Nothing has been pushed and no GitHub repo or cloud resource was created. Sources were read, never modified: `voyage-scout` (Git history from 2026-10-25) and `wayfinder` (Git history from 2026-10-26). Nothing from `agentic_search` was copied.

## File counts

991 files, about 15 MB: `scout/` 744 (of which 41 are the llama.cpp test fixtures and 40 the wire schemas), `training/` 151, `eval/` 36, `deploy/` 18, `docs/` 34, `licenses/` 4, plus README, LICENSE, NOTICE, this file.

## Tests and scans (run on the public copy, offline)

| Check | Result |
|---|---|
| `scout` full default suite (`uv run pytest -q`, no network, live/eval/spike deselected) | 2089 passed, 5 skipped, 120 deselected (live, eval and spike markers) |
| `deploy/gcp/tests` (flag parity with the Modal app, config parsing, gateway import and healthz, no secrets in env examples, scripts parse) | 16 passed |
| `training/tests` | 161 passed, 6 skipped (need withheld task manifests or cluster specs) |
| Secret scan (gitleaks is not installed; regex scan for `sk-`, `hf_`, `tpuf_`, `pa-`, `AKIA`, `ghp_`, `xox*`, `al-` keys, private-key blocks, `*KEY/TOKEN/SECRET/PEPPER=` lines with long values, base64 literals, and any `.env`, `*.key`, `*.pem` file) | No secrets. Only synthetic hits: a fixture PEM (`SYNTHETIC_PEM_SECRET`), `fixture-not-a-real-secret` in `scout/tests/fixtures/repo_small/.env` (a trap file the test suite needs, un-ignored on purpose), and test constants. Run gitleaks once before pushing |
| Sensitive-term grep (`/Users/`, Modal workspace name, internal gateway URL, mongodb.com, 10gen, Glean, docs.google, PRFAQ, customer names) | Clean |

Skipped tests in `scout/` (5): they read task manifests or a Phase-0 spike artifact that are not included.

## Built before October 26 versus during the event

`voyage-scout` history: 42 commits on 2026-10-25 (the last at 20:07 PT), 131 on 10-26, then 8, 20, 4, 3 and 1 on the following days. `wayfinder` history starts 2026-10-26. Of the 414 `scout/` source files carried over (before the GCP kit), 132 were first added in the October 25 commits. They are listed below by directory; many were substantially extended during the event, and the README says so. Files first added on or after October 26 are event work: the encryption layer, sync engine, gateway service, both plugins (53 files), retrieval spec, demo runtime, evaluation harness, most of the tests, and everything in `training/`, `eval/` and `deploy/gcp/`.

Pre-event, first added 2026-10-25:

- `scout/Makefile` (1 file)
- `scout/ops/ns_janitor.py` (1 file)
- `scout/pyproject.toml` (1 file)
- `scout/schema/v1` (38 files)
- `scout/scripts/gen_schema.py` (1 file)
- `scout/src/scout/__init__.py` (1 file)
- `scout/src/scout/adapters` (6 files)
- `scout/src/scout/adapters/fs_local` (1 file)
- `scout/src/scout/adapters/gateway_http` (1 file)
- `scout/src/scout/adapters/keychain` (1 file)
- `scout/src/scout/adapters/openai_chat` (1 file)
- `scout/src/scout/adapters/turbopuffer` (1 file)
- `scout/src/scout/adapters/voyage` (1 file)
- `scout/src/scout/core` (2 files)
- `scout/src/scout/core/contracts` (5 files)
- `scout/src/scout/core/crypto` (1 file)
- `scout/src/scout/core/explore` (9 files)
- `scout/src/scout/core/local` (6 files)
- `scout/src/scout/core/ports` (8 files)
- `scout/src/scout/core/service` (1 file)
- `scout/src/scout/core/sync` (1 file)
- `scout/src/scout/surfaces` (2 files)
- `scout/src/scout/surfaces/cli` (1 file)
- `scout/src/scout/surfaces/daemon` (1 file)
- `scout/src/scout/surfaces/gateway_http` (1 file)
- `scout/src/scout/surfaces/mcp` (1 file)
- `scout/src/scout/wiring.py` (1 file)
- `scout/tests/conformance/.gitkeep` (1 file)
- `scout/tests/conftest.py` (1 file)
- `scout/tests/contract/.gitkeep` (1 file)
- `scout/tests/contract/test_poc_explore.py` (1 file)
- `scout/tests/eval/.gitkeep` (1 file)
- `scout/tests/fixtures/.gitkeep` (1 file)
- `scout/tests/fixtures/poc_explore_case.json` (1 file)
- `scout/tests/fixtures/repo_small` (15 files)
- `scout/tests/live/.gitkeep` (1 file)
- `scout/tests/security/.gitkeep` (1 file)
- `scout/tests/spike/.gitkeep` (1 file)
- `scout/tests/unit/test_contracts_types.py` (1 file)
- `scout/tests/unit/test_explore_loop.py` (1 file)
- `scout/tests/unit/test_explore_support.py` (1 file)
- `scout/tests/unit/test_fake_store_core.py` (1 file)
- `scout/tests/unit/test_fakes_poc.py` (1 file)
- `scout/tests/unit/test_local_grep.py` (1 file)
- `scout/tests/unit/test_local_tools.py` (1 file)
- `scout/tests/unit/test_no_host_names.py` (1 file)
- `scout/tests/unit/test_ns_janitor.py` (1 file)
- `scout/tests/unit/test_spend_gate.py` (1 file)
- `scout/tests/unit/test_wire_schema.py` (1 file)
- `scout/uv.lock` (1 file)

Also pre-existing and not claimed: Muse Glimmer 30B, Muse Spark Open, Agents-A1-35B and its transcripts, voyage-code-4, turbopuffer, vLLM, Muse Code, Claude Code, and a private pre-event benchmark of coding tasks (the training pool and all held-out sets derive from it). Some `training/` and `eval/` scripts are adapted copies of tooling for that benchmark (their docstrings say "copy of"): `eval/e2e/gate.py`, `eval/e2e/anchor_audit.py`, `eval/scorers/score_heldout_hard*.py`, `score_ood_langs.py`, `training/wayfinder/rollout/a1_eval_app.py`. Ask whether the judges need that stated per file.

## Excluded, and why

| Excluded | Why |
|---|---|
| `docs/PRD*`, `STATUS.md`, `PRFAQ*`, deck sources and `final_deck*`, `deck_audit`, `prfaq_value_prop_edits`, `task_composition`, `tpuf_cost_trace` (invoice reconciliation), opportunity docs, `docs/research/` | Internal business and planning material |
| `artifacts/` entirely (spend ledgers `artifacts/ops/spend.jsonl`, rows, logs, transcripts, `ops/*` watchers, `ops/spend_gate.py`, `ops/billing_trueup.py`) | Spend ledgers, raw transcripts, large artifacts. Only 6 small summary files are kept in `eval/results/` |
| `specs/` (ralph specs with owner quotes and Glean-sourced research), `voyage-scout/docs/PRD.md`, `CLAUDE.md`, `spikes/` | Internal process and internal links |
| All task manifests and datasets: `wayfinder/data/*.json` (splits, task filter, RL pools, held-out sets), `eval/e2e` task and result JSON, `voyage-scout/eval/*.jsonl`, `cc50_*`, `_rows.json`, candidate repo lists | They derive from the pre-event private benchmark (ids start `v3__`; `splits.json` names it as its source). Only repo names and licenses are published (`eval/results/eval_repos_license.csv`). `data/rubric_weights.json` is kept (fitted weights, no task text) |
| The 358 private 10gen tasks | None are in any copied file (grep for `10gen` over the copied tree finds nothing; the only mention was in the excluded PRD) |
| Teacher outputs and SFT data (`teacher_v*`, a1 conversions, OPD rollouts) | Muse Spark Open outputs; the release is code plus model. `training/README.md` documents regeneration |
| wayfinder-qwen: `*qwen*` files, `docs/qwen_moe_sft.md`, `score_heldout_hard_5way.py`, `score_qwen_dev.py`, Qwen eval apps | Out of scope for this release. The architecture figure was regenerated without it. The only remaining mentions are the `display_names.py` label helper, one integrations README line ("not yet served"), a closing "next" line, and A1/vLLM parser names |
| `deploy/g4` (GKE specs with internal namespaces and secret names), `final_deck*`, `docs/figures/final_deck` (apart from 4 result PNGs checked visually) | Internal infrastructure |
| `eval/e2e` orchestration and watchers, `asa/` vendored from the private benchmark, `grove_spend*.jsonl`, run, search and dataset folders | Entangled with the private benchmark, spend ledgers, or raw outputs |
| `scout/tests/unit/test_spend_gate.py`, `training/tests/test_mine_hard.py`, `test_grpo_rank5.py`, `release/smoke_test.py`, `release/README.md` | Depend on excluded internal tooling, or hold internal approval and legal notes |

Scrubbed in what was kept: the Modal workspace name became `myworkspace`; the gateway URL default became `https://scout-gateway.example.com` (code and tests), and `SCOUT_GATEWAY_URL` is the way to set it (`demo_panes.sh` now requires it); the internal LLM gateway ("Grove") URL and key became `LLM_GATEWAY_URL` and `LLM_GATEWAY_API_KEY`, and the name was removed from prose; absolute `/Users/...` paths became `<repo>` placeholders; `agentic_search` became `benchmark-repo`; `.env` search paths into sibling repos were removed from the harness; sentences about budgets, caps, ledgers, billing true-ups and invoices were dropped from the result docs (list kept by the author in a scratch log); "the owner" became "the author". `demo_warm.sh` now starts and stops the GCP VM instead of scaling Modal apps.

## Open items for the owner

1. **Model repo visibility.** `hweller1/wayfinder-glimmer-30b` was private when last checked (`release/README.md`, 2026-10-28). The README links it and the GCP kit downloads it; make it public (and confirm the card, `docs/MODEL_CARD.md`) or set `HF_TOKEN` on the VM.
2. **Copyright holder and legal review.** NOTICE says "the authors of this repository". Confirm with MongoDB legal that a public personal repo of work done for Voyage AI is cleared, who holds copyright, and the earlier open licence questions (Spark Open derivative terms, Agents-A1 attribution, Model-as-a-Service clause if anyone hosts this for third parties).
3. **Licences of the evaluation repositories.** The data are not all permissive: of 47 held-out and 111 out-of-distribution evaluation repositories, 7 and 32 are GPL, AGPL or MPL, and 6 and 23 have no or unrecognised licence (see `eval/results/eval_repos_license.csv`). Train, dev and test splits are almost all permissive (235 of 244 train repos; the rest unavailable or unlicensed at the API). The model card says "permissively licensed" for training data: re-check against the 344 training tasks. One repository (`Emmy123222/Stellar-GreenPay`) is gone from GitHub.
4. **Task manifests are withheld** because they derive from the private benchmark. Decide whether to publish the event-mined subsets (the v4 and clean end-to-end tasks, held-out hard v1); that needs a licence check per repository and a view on "pre-event data".
5. **Privacy wording.** README states "never stored" with the trust boundary. Voyage zero-retention for the embedding account was unresolved in the source docs (`docs/scout/PRODUCT_STATUS.md`, known gaps).
6. **Latency wording.** The brief said Scout takes "about 2.5 to 3 minutes". The sources give 126 s median for the `explore` call, 189 s per Muse Code session (65 s stock), and about 175 s per search in the model card; the README says "about 2 to 3 minutes" with those figures.
7. **GCP kit is untested on a VM.** Verify the Deep Learning image family name (`create_vm.sh` default is a guess), the driver version for vLLM 0.32.0's CUDA 12.9 wheels, the A100 list price (README quotes about $5.07 per hour from memory), `--max-run-duration` support in your gcloud, and run `healthcheck.sh` before the demo. FP8 on an A100 is unmeasured; bf16 leaves about 13 GB of KV cache.
8. **Reuse of the existing index.** To keep the repos already indexed during the studies, reuse the Modal gateway's `SCOUT_MASTER_KEY`, `SCOUT_PEPPER` and `SCOUT_KEYS_NS` (in the Modal secret and `~/.config/scout/gateway-master.json`); otherwise every repo re-embeds (costs in `docs/demo.md`).
9. **Repository name and URL.** The model card points to `github.com/hweller1/wayfinder`; `REPO_URL` in the GCP kit is a placeholder.
10. **Docs are historical.** `docs/results` and `docs/scout` still mention Modal, internal ids (`wayfinder-v3`), and documents that are not included (plans, status, specs). The README and `docs/README.md` say so. Skim them once for anything you do not want public (for example task titles of public repos in the per-task tables, and MongoDB-ecosystem repo names).
11. **Run gitleaks** (not installed here) on the final tree, and re-read `git log` of the first commit's author line.
12. `scout/pyproject.toml` still lists `modal` in the `gateway` extra, so the VM venv installs it unused. Harmless, not changed because it would invalidate `uv.lock`.
