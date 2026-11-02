# Training wayfinder-glimmer

**Built with Muse Spark Open.** wayfinder-glimmer is Muse Glimmer 30B (Meta, Apache-2.0) fine-tuned on trajectories produced by Muse Spark Open 816B-A42B (Meta, Muse Model License) and on a few Agents-A1-35B transcripts, so it is a Muse Spark Open derivative. Read [../NOTICE](../NOTICE) before you redistribute weights.

The code runs on Modal (GPU jobs) and Hugging Face; the launchers need your own Modal workspace and secrets. Names such as `wayfinder-data` (a Modal Volume) and `wayfinder-v3` (the served name of wayfinder-glimmer, kept for continuity with the result files) are internal ids, listed in [docs/MODEL_NAMES.md](../docs/MODEL_NAMES.md).

## What the model is

LoRA supervised fine-tuning on the language model only (rank 64, alpha 128, all attention and MLP projections), merged into the base weights in fp32 and saved as bf16, so serving needs no LoRA support. 1,383 examples (1,350 Muse Spark Open teacher trajectories and 33 Agents-A1-35B trajectories), 41.56M tokens, 7.02M assistant tokens, 344 tasks from 153 public repositories. Only assistant tokens carry loss. 2 epochs, learning rate 5e-5 with cosine decay, 5 warmup steps, packed rows of up to 49,152 tokens, 4x H200 data parallel, 478 steps, about 3 hours, seed 0. One run; the results describe this checkpoint, not the recipe. Write-up: [../docs/results/sft_v3_eval.md](../docs/results/sft_v3_eval.md) (the model is "v3" there and in file names).

## Pipeline, in order

| Step | Code | Notes |
|---|---|---|
| Task pool and splits | `scripts/dataprep/build_filter.py`, `build_splits.py`, `common.py` | Repository-level train, dev, test and held-out splits. The pool is a set of coding tasks mined from public repositories; the records are not included (see below) |
| Index the repositories | `wayfinder/rollout/index_app.py` | Builds the same encrypted voyage-code-4 index Scout uses, so rollouts can call `semantic_search` |
| Serve the teacher | `serve/spark_open.py`, `serve/spark_client.py`, `serve/spark_download.py` | Muse Spark Open 816B-A42B (NVFP4) with vLLM on 8x B200 as an ephemeral Modal app, several replicas in parallel |
| Roll out the teacher | `wayfinder/rollout/modal_app.py`, `run.py`, `chat.py`, `nudges.py` | Runs Scout's own explore loop (grep, glob, read_file, semantic_search, report), up to 40 turns, temperature 0.85, top_p 0.95, with the teacher's reasoning retained between turns |
| Filter and convert | `wayfinder/sft/convert_teacher.py`, `convert_teacher_modal.py`, `convert.py` (Agents-A1 transcripts) | Keeps runs that found every gold file, opened every file they reported, called `report` within budget, stayed under the task's p75 turn count and 49,152 tokens, deduped on task and seed; the evaluation tasks are excluded. Funnel in `docs/results/sft_v3_eval.md` |
| Train | `wayfinder/sft/train_v1.py` (launcher, `--gpu h200x4`), `train.py` | LoRA with packed sequences, flash-attention 3, gradient checkpointing |
| Merge | `wayfinder/rollout/sft_merge_app.py` | fp32 merge of the 416 LoRA modules, saved as bf16 |
| Release | `release/stage.py`, `release/publish.py` | Stages weights with the license files and the "Built with Muse Spark Open" notice, checks the folder, uploads to the Hub (private by default) |
| Serve | `serve/wayfinder_v3.py` (Modal), [../deploy/gcp](../deploy/gcp) (one A100 VM) | vLLM 0.32.0 with the `muse_glimmer` tool-call and reasoning parsers and the checkpoint's own chat template |

Experiments beyond SFT, with write-ups in [../docs/results/](../docs/results/): GRPO with verifiable rewards on top of the SFT checkpoint (`wayfinder/grpo`, `scripts/grpo`, `docs/results/grpo.md`) and on-policy distillation (`wayfinder/opd`). Neither produced the released model.

## The teacher trajectories are not published

The SFT data is outputs of Muse Spark Open. This repository releases code and the model, not those trajectories (the license permits it; we chose to keep the release small and to avoid republishing a teacher's outputs). To regenerate comparable data you need your own access to the Muse Spark Open weights (the repository is gated on Hugging Face), a Modal account with 8x B200 or H200 capacity, and a task pool:

1. Build a pool of tasks (a repository at a commit, a plain-language question, the gold files it touches) from public repositories whose licenses you accept, and split by repository (`scripts/dataprep`).
2. Index each repository (`wayfinder/rollout/index_app.py`) with your Voyage and turbopuffer keys.
3. Start teacher replicas (`modal run serve/spark_open.py`, with `SPARK_REPLICA` suffixes for more than one) and roll out (`wayfinder/rollout/modal_app.py`); keep every row.
4. Convert and filter with `wayfinder/sft/convert_teacher_modal.py` using the filters above, then train with `wayfinder/sft/train_v1.py`.

The task records used for the released model derive from a pre-event private benchmark and are not included either. `../eval/results/eval_repos_license.csv` lists the repositories of the evaluation sets with their licenses.

## Tests

```bash
cd training && PYTHONPATH=.:../scout/src python3 -m pytest tests -q
```

Tests that need task manifests or cluster specs that are not in this repository are skipped with a reason. The GRPO tests import the on-policy-distillation span code, which is why `wayfinder/opd` is here.
