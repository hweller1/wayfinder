# Semantic index for rollouts

Every rollout repo@base_commit (train, dev, test in data/splits.json: 306 repos, 1511 repo@commits) is indexed into turbopuffer through scout's real secure-indexing path (`scout.demo.runtime.DemoRuntime`, imported from ../voyage-scout, not edited) by `wayfinder/rollout/index_app.py` (Modal app `wayfinder-index`, `modal run --detach wayfinder/rollout/index_app.py::launch`). Tier order: rl-eligible train, the 50 Stage-B dev tasks, remaining dev, everything else. Resumable: finished repo@commits are skipped via the Modal Dict `wayfinder-index-registry`; spend estimate (Voyage tokens x $0.18/M) in Dict `wayfinder-index-spend`; progress in `wayfinder-index-progress`. The job stops starting new commits at `cap_usd` (default 140).

Namespaces: `scout_dev_wf_<repo-slug>_<commit8>` and `__tree` per repo@commit; `scout_dev_wf_<repo-slug>__{cache,meta,keys}` per repo. Slug = lowercase `owner__name`, non-alphanumerics to `_`.

Keys: Modal secret `wayfinder-index-keys` holds TURBOPUFFER_API_KEY, VOYAGE_API_KEY and `WF_INDEX_MASTER_KEY_HEX` (the 32-byte gateway master key; the per-repo team key is wrapped under it in `__keys`). Local copy: `~/.scout-wayfinder/master.key` (0600). Lose it and every index is unreadable (re-index). Rebuild the secret with `ops/make_index_secret.py`. Any Modal function that runs rollouts with semantic on needs `secrets=[modal.Secret.from_name("wayfinder-index-keys")]` and the pip packages cryptography, keyring, turbopuffer, voyageai (done in modal_app.py and baseten_bulk_app.py).

Each rollout checkout gets a deterministic scout `clone_id` (`semantic.clone_id_for`) so it addresses the namespace the indexer wrote.

Rollout flag: `run_rollout(..., semantic="auto"|"on"|"off")`; CLI/launchers `--semantic`. auto (default) = semantic_search when an index exists, else grep-only with `semantic_reason` recorded; on = fail (infra error, retried) if no index; off = grep-only. Rows carry `semantic`, `semantic_requested`, `semantic_reason`, `semantic_index`, `semantic_query_tokens`.

Switch a bulk job: stop the running app, relaunch with a new `--name` (the harness is re-uploaded at launch), e.g. `modal run --detach wayfinder/rollout/baseten_bulk_app.py::launch --name spark_v2_sem --semantic auto`.

Wrappers around scout (scout source untouched): the demo's pre-index size scan is made tolerant of dangling symlinks, and scout's per-batch row cap is set to 500 so a chunk repeated in >500 files cannot overflow `EmbedReq.replace` (500 files).

Validation (5 dev tasks, Baseten Spark Open, seed 0, nudge, 40 turns): artifacts/semantic_validation/*.jsonl.
