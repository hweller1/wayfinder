# Release checklist (for the coordinator)

Local repo only. Nothing has been pushed and no GitHub repo or cloud resource was created. Sources were read, never modified: `voyage-scout` (Git history from 2026-10-25) and `wayfinder` (Git history from 2026-10-26). Nothing from `agentic_search` was copied.

## File counts

About 970 files after pruning: `scout/` 408 tracked files (of which 72 are the llama.cpp-derived declaration fixtures and 40 the wire schemas), `training/`, `eval/`, `deploy/`, `docs/` (now 3 result write-ups, the product status, the demo runbook, the model card, model names, README and figures), `licenses/`, plus README, LICENSE, NOTICE, `.gitleaks.toml` and this file. Run `git ls-files | wc -l` for the exact number.

## Tests and scans (run on the public copy, offline, re-run 2026-11-02 after the edits below)

| Check | Result |
|---|---|
| `scout` full default suite (`uv run pytest -q`, no network, live/eval/spike deselected) | 2089 passed, 5 skipped, 120 deselected. Same as before |
| `deploy/gcp/tests` | 16 passed. Same as before |
| `training/tests` | 161 passed, 6 skipped. Same as before |
| `deploy/modal/tests` (added 2026-11-02, offline: vLLM flags equal the reference apps' for glimmer and qwen, modules import without Modal credentials, the qwen key proxy enforces the key and renames `reasoning_content`, secrets helper writes no value to stdout) | 21 passed |
| gitleaks 8.30.1, `--no-git` on the working tree | 3 hits, all false positives: a test constant (`k-other-1234567890`) in `scout/tests/unit/test_make_secrets.py`, hex test vectors in `scout/tests/unit/test_crypto_keys_ids.py`, and a docstring in `training/wayfinder/opd/score.py` that the generic-api-key rule matches. `.gitleaks.toml` allowlists those three files for that rule only. The synthetic fixture PEM and the fixture `.env` were not flagged |
| gitleaks, git history | Run on the final commit; see the report that came with it |
| Sensitive-term grep (`/Users/`, Modal workspace name, internal gateway URL, mongodb.com, 10gen, Glean, docs.google, PRFAQ, customer names, unreleased model names) | Clean |

Skipped tests in `scout/` (5): they read task manifests or a Phase-0 spike artifact that are not included.

## Built before October 26 versus during the event

`voyage-scout` history: 42 commits on 2026-10-25 (the last at 20:07 PT), 131 on 10-26, then 8, 20, 4, 3 and 1 on the following days. `wayfinder` history starts 2026-10-26. Recounted 2026-11-02 against the 408 tracked `scout/` files: 131 were first added in the October 25 commits (the earlier 132 of 414 came from a different file count). The README has a table by component, including which of those files were empty packages on October 25 (the encryption, sync and service packages and the CLI, daemon, MCP and gateway surfaces, and the real Voyage, turbopuffer and keychain adapters were not written yet). They are listed below by directory; many were substantially extended during the event, and the README says so. Files first added on or after October 26 are event work: the encryption layer, sync engine, gateway service, both plugins (53 files), retrieval spec, demo runtime, evaluation harness, most of the tests, and everything in `training/`, `eval/` and `deploy/gcp/`.

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
| `deploy/g4` (GKE specs with internal namespaces and secret names), `final_deck*`, `docs/figures/final_deck` (apart from 4 result PNGs checked visually) | Internal infrastructure |
| `eval/e2e` orchestration and watchers, `asa/` vendored from the private benchmark, `grove_spend*.jsonl`, run, search and dataset folders | Entangled with the private benchmark, spend ledgers, or raw outputs |
| `scout/tests/unit/test_spend_gate.py`, `training/tests/test_mine_hard.py`, `test_grpo_rank5.py`, `release/smoke_test.py`, `release/README.md` | Depend on excluded internal tooling, or hold internal approval and legal notes |

Scrubbed in what was kept: the Modal workspace name became `myworkspace`; the gateway URL default became `https://scout-gateway.example.com` (code and tests), and `SCOUT_GATEWAY_URL` is the way to set it (`demo_panes.sh` now requires it); the internal LLM gateway ("Grove") URL and key became `LLM_GATEWAY_URL` and `LLM_GATEWAY_API_KEY`, and the name was removed from prose; absolute `/Users/...` paths became `<repo>` placeholders; `agentic_search` became `benchmark-repo`; `.env` search paths into sibling repos were removed from the harness; sentences about budgets, caps, ledgers, billing true-ups and invoices were dropped from the result docs (list kept by the author in a scratch log); "the owner" became "the author". `demo_warm.sh` now starts and stops the GCP VM instead of scaling Modal apps.

## Done on 2026-11-02

1. Prior-work disclosure: README table by component (state on October 25, work during the event), derived from `git log` of `voyage-scout`.
2. Privacy wording: README, architecture figure and `PRODUCT_STATUS.md` now say "Our index never stores your code": turbopuffer holds vectors, keyed hashes and encrypted paths, while chunk text passes through the gateway and Voyage's embedding API. No claim of Voyage zero retention.
3. Internal docs pruned: `docs/results/` keeps `e2e_final_results.md`, `heldout_hard_v1_frontier.md`, `muse_scout_results.md`; `docs/scout/` keeps `PRODUCT_STATUS.md`. References to the deleted write-ups in code comments, tests and READMEs now say "not in this release". Spend, ledger and cost-table passages were removed from the kept results.
4. The unreleased second model is no longer named in the README, `docs/MODEL_NAMES.md`, `display_names.py` and the Claude Code README.
5. Training-data license wording: the card no longer says "permissively licensed". Of the 244 repositories in the training split, 242 are permissive (MIT 129, Apache-2.0 98, BSD-3-Clause 12, BSD-2-Clause 2, ISC 1), one is AGPL-3.0 (`turbo-devv/openOMSI`) and one is gone from GitHub (`Emmy123222/Stellar-GreenPay`); checked with the GitHub license API on 2026-11-02, list in `eval/results/train_repos_license.csv`. The 344 SFT tasks come from 153 of those repositories; the task list is on the Modal volume and was not read, so the counts are for the whole split. The same sentence is on the Hugging Face card (commit 36bdf89d).
6. Hugging Face model `hweller1/wayfinder-glimmer-30b` is public (verified 2026-11-02, commit d8813963; card updated to 36bdf89d).
7. gitleaks installed and run (see above).

## Open items for the owner

1. **Repository name and URL.** Confirm `github.com/hweller1/wayfinder`; the model card points to it and `REPO_URL` in the GCP kit is a placeholder.
2. **Modal path written, not deployed from a fresh account.** `deploy/modal/serve_glimmer.py`, `deploy/modal/make_secrets.py` and the simplified `scout/deploy/modal/gateway.py` (one secret, `scout-gateway`; the A1 and step-20 profiles and the `scout-llm*` secrets were removed from it) were checked offline only (`deploy/modal/tests`, no Modal call, no secret or resource created). Before relying on the README's "Run on Modal" steps, run them once on an empty workspace: the first start downloads about 60 GB, the app starts vLLM without GPU memory snapshots (the reference app's snapshot mode was not ported, so cold start is probably longer than the 2 to 3 minutes measured), and the URL pattern `https://<workspace>--<label>.modal.run` (and `-warm`) assumes Modal's default environment. The $3.95 per hour H100 price and $1.72 per million lines are the author's figures.
3. **GCP kit never run on a real VM.** Run `healthcheck.sh` and the demo path once before relying on it. FP8 on an A100 is unmeasured; bf16 leaves about 13 GB of KV cache.
4. **Placeholders to verify.** The Deep Learning image family name (`create_vm.sh` default is a guess), the driver version for vLLM 0.32.0's CUDA 12.9 wheels, the A100 list price quoted in the README (about $5.07 per hour, from memory), and `--max-run-duration` support in your gcloud.
5. **Push.** Nothing has been pushed. Re-run `gitleaks detect --source . --redact` right before pushing and re-read the author line of the commits.
6. **Legal and data questions still with the owner.** Copyright holder and legal clearance for a public personal repo (NOTICE says "the authors of this repository"); Spark Open derivative terms and Agents-A1 attribution; whether to publish the event-mined task subsets; and the licenses of the evaluation repositories (GPL, AGPL, MPL or unlicensed: 7 of 47 held-out and 32 of 111 out-of-distribution repositories, see `eval/results/eval_repos_license.csv`).
7. **Reuse of the existing index.** To keep repositories already indexed, reuse the gateway's `SCOUT_MASTER_KEY`, `SCOUT_PEPPER` and `SCOUT_KEYS_NS`; otherwise every repo re-embeds.
8. `scout/pyproject.toml` still lists `modal` in the `gateway` extra (harmless; changing it would invalidate `uv.lock`).
