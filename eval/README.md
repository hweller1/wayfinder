# Evaluation

Scorers, harness code and result summaries. There are no raw transcripts, no per-task rows and no task records in this directory: those derive from a pre-event private benchmark or are large, and are not part of the release. The code is here so the methods can be read and re-run on your own task set; several scripts expect inputs (row files, task manifests, a sibling benchmark checkout) that you have to produce.

| Path | What it is |
|---|---|
| `scorers/` | The scorers that produced the tables in `docs/results/` and `docs/results/glimmer_era/`: reported recall, recall@5, precision, turns, paired task and repository-cluster bootstraps. `score_heldout_hard_4way.py` (wayfinder-glimmer, Agents-A1-35B, Sonnet 5.5, GPT-6.1 Sol on the 200-task held-out set), `score_test_split.py`, `score_ood_langs.py`, `score_sft_v*.py`, `scout_latency_profile.py` |
| `e2e/` | The end-to-end coding study: a coder model with no search tools can open only the files the searcher returned, held-out tests are absent from its sandbox, and the diff is scored in a separate sandbox (`run_coder.py`, `sandbox.py`, `modal_sandbox.py`, `score.py`, `gate.py`, `analyze_final.py`, `power_v3.py`). Orchestration and launch scripts for the full study are not included. Protocol: the end-to-end final study plan (not in this release); results: [../docs/results/README.md](../docs/results/README.md) (wayfinder-qwen; the first-model write-up is in `../docs/results/glimmer_era/`) |
| `results/` | Summary JSON and text of the held-out, test-split, out-of-distribution and baseline evaluations, and `eval_repos_license.csv` and `train_repos_license.csv` (every repository in the evaluation sets with its SPDX license as reported by the GitHub API on 2026-11-02) |
| `../scout/eval/harness/` | The harness for the Claude Code and Muse Code plugin studies (arms, briefs, paired statistics, reports) |

The summary files in `results/` are from the glimmer-era runs (wayfinder-glimmer): `a1_baseline__summary.json`, `sft_v3__summary.json`, `test_split__summary.json`, `ood_langs__summary.json` and `heldout_hard__score_4way.txt` (the 200-task held-out comparison; its wayfinder-glimmer rows are the `v3_*` arms, and its Agents-A1, Sonnet and GPT rows are also quoted in the final results). There are no wayfinder-qwen summary files here: the qwen numbers are in [../docs/MODEL_CARD_QWEN.md](../docs/MODEL_CARD_QWEN.md) and [../docs/results/README.md](../docs/results/README.md). The two license CSVs are not model results.

Frontier models (Sonnet 5.5, GPT-6.1 Sol, Opus 5.5) are reached through an LLM API gateway whose URL and key you set in `LLM_GATEWAY_URL` and `LLM_GATEWAY_API_KEY`. Open models run behind vLLM on Modal (`../training/wayfinder/rollout/*_eval_app.py`).

The evaluation sets include repositories under copyleft or unstated licenses (see the CSV). They are used for evaluation only; the training split (repository-level) is separate and was not evaluated on.
