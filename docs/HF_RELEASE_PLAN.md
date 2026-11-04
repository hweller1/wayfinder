# Release plan: gated training-data dataset for wayfinder-qwen

Status: DRAFT for the owner. Nothing has been uploaded, no Hugging Face repository has been created or changed, and no token was used. Every command below is for after the owner approves. The dataset card is `docs/DATASET_CARD_TRAINING_DATA.md`; the model card edits are in `docs/MODEL_CARD_QWEN.md` and mention the dataset link as "not yet published".

## 1. What would be uploaded

The 1,383 training examples are not in this repository or in the local checkouts. They sit on the training volume (`wayfinder-data:/sft/qwen_v3/replica/train-*.jsonl` for the Qwen render, `wayfinder-data:/sft/teacher_v3` for the Glimmer render). The first step is to copy the Qwen render to a local staging folder.

| File in the dataset repo | Content | Count |
|---|---|---|
| `README.md` | The dataset card | 1 |
| `data/train-*.jsonl` | One example per line: `id`, `tools_variant`, `messages`, `meta` | 1,383 rows (344 tasks, 153 repositories) |
| `repo_licenses.csv` | `repo, license_spdx` for the 153 repositories | 153 rows |
| `tools.json` | The tool schemas (`serve/scout_tool_specs.json`) | 1 |
| `LICENSE-MUSE-SPARK-OPEN.md`, `USE_POLICY-MUSE-SPARK-OPEN.md`, `NOTICE` | Copied from `licenses/` and the repository root | 3 |
| `tasks.jsonl` (optional, decision D2) | `task_id, repo, base_commit, request, gold_files` | 344 rows |

Size: not measured, because the data is on the volume. For scale, 47.2M Qwen tokens (6.9M assistant tokens) in 1,383 examples (`artifacts/sft/qwen_v3_summary.json`); the three 100-row rollout files in `artifacts/sft_v4/` average 0.23 MB per row, which would be about 320 MB for 1,383 rows. Measure the staged folder before upload (`du -sh`).

Fields kept: those in the card schema. Stripped before upload, so nothing derived from the serving stack or the machine ships:

| Stripped | Why |
|---|---|
| `input_ids`, `assistant_spans`, `n_tokens` (as top-level fields) | Derived from the tokenizer and template; anyone can recompute them. Large |
| `meta.teacher_endpoint`, `meta.teacher_model` host details | Can hold a Modal workspace name or a third-party host URL. Keep only `meta.source` |
| Any absolute path or host name inside tool results (`/Users/`, `/private/`, `/data/`, `/tmp/`, clone directories) | Machine paths; the path/PII scan below must find none |
| `meta.cost*`, latency, rate-limit and gateway fields, if any are present | Not needed, and some name internal services |
| Any field not in the card schema | Allow-list the schema; drop everything else |

## 2. Commands to run after the owner approves

Authenticate with your own token, interactively or from an environment variable set in the shell. Never write a token into a file, a command history you share or this repository.

```bash
# 0. Who am I, and is it the right account? (read-only)
hf auth whoami

# 1. Create the dataset repository, private first.
hf repo create hweller1/wayfinder-training-data --repo-type dataset --private

# 2. Upload the staged folder (README.md, data/, repo_licenses.csv, tools.json, licenses). The repo is private, so nobody can see it yet.
hf upload hweller1/wayfinder-training-data ./hf_stage . --repo-type dataset \
  --commit-message "Add wayfinder training data (gated, manual approval)"
```

```python
# 3. Switch to gated, manual approval. Do this BEFORE making the repository public.
from huggingface_hub import HfApi
api = HfApi()   # uses the logged-in token
api.update_repo_settings(repo_id="hweller1/wayfinder-training-data",
                         repo_type="dataset", gated="manual")

# 4. Check the gating took effect while the repository is still private.
info = api.dataset_info("hweller1/wayfinder-training-data")
print(info.gated, info.private)          # expect: manual True

# 5. Make it public (the files stay behind the access request).
api.update_repo_settings(repo_id="hweller1/wayfinder-training-data",
                         repo_type="dataset", private=False)

# 6. Review requests (the owner decides who gets in).
for r in api.list_pending_access_requests("hweller1/wayfinder-training-data", repo_type="dataset"):
    print(r.username, r.fullname, r.fields)      # name, affiliation, country, intended use
# api.accept_access_request("hweller1/wayfinder-training-data", "<username>", repo_type="dataset")
# api.reject_access_request("hweller1/wayfinder-training-data", "<username>", repo_type="dataset")
```

After publishing, verify from a logged-out shell: `curl -sI https://huggingface.co/datasets/hweller1/wayfinder-training-data/resolve/main/data/train-00000-of-00001.jsonl` must not return the file (expect a 401 or 403 or a redirect to the gate), and the dataset page must show the terms and the request button. Then, and only then, update the model card link (remove "not yet published" in `docs/MODEL_CARD_QWEN.md`, upload that file to the model repository) and fix the lines that still say the trajectories are not published (section 5).

## 3. Licence and data constraints that bear on this release

What the files say, read from `licenses/LICENSE-MUSE-SPARK-OPEN.md`, `NOTICE`, `eval/results/train_repos_license.csv` and `training/README.md`.

1. The Muse Spark Open licence grants permission "to use, copy, modify (including fine-tune), create derivative works, distill, merge, publish, distribute, sublicense, and/or sell copies of the Software", subject to its conditions. Its text has no clause that bars training another model on the outputs, including a competing hosted service; it was searched for "train", "compet" and "distill" and only the grant above matches. Nothing in the dataset card should say otherwise.
2. Condition 1: the notice, the licence and the Usage Policy "shall be included in all copies or substantial portions of the Software and Derivatives". "Derivatives" is defined as "any AI model created, trained, modified (including fine-tuned), distilled, or otherwise improved using the Software or any outputs or results thereof". A dataset is not an AI model, so on a literal reading the dataset is not a Derivative and condition 1 does not strictly attach to it. The card includes the licence files and the notice anyway, and puts the obligation on anyone who trains a model on it, which is where the definition bites. Counsel should confirm this reading (decision D5).
3. Condition 2: "Built with Muse Spark Open" must be shown prominently. The dataset card shows it under the title and the gate prompt quotes it. It is also required in the README and NOTICE of the GitHub repository, already present.
4. Section 3 (Model as a Service, US$20 million aggregate gross revenue over twelve months) attaches to offering Derivatives through an API, not to publishing data. The gate prompt repeats it so downstream users know it follows any model they train.
5. Section 4 ("Meta claims no ownership of outputs") supports publishing the outputs. Section 5 disclaims warranty. The Usage Policy item that bears on this data is the bar on actions that "infringe, misappropriate, or otherwise violate any third-party rights, including the outputs or results of any products or services using Muse Spark Open". The trajectories contain excerpts of third-party source code that the model read, so this is the clause to check against the repositories' licences.
6. Teacher provenance (`docs/qwen_moe_sft.md` in the private repository): of 1,383 rows, 1,347 came from self-hosted replicas of the open weights, 3 from a third-party host of the same open weights and 33 from Agents-A1-35B. None came from Meta's API. The Meta API terms (section 10.1(ix), as recorded in the PRD) bar training a "competing" model on API outputs without written authorisation, and that does not reach these rows. The 3 third-party-host rows rest on the host's model id and its documentation, not a byte comparison of weights (decision D4).
7. The 33 Agents-A1-35B rows: Agents-A1-35B is Apache-2.0 (InternScience). Keep the attribution in the card and NOTICE; the Apache licence needs the licence text and notices with redistribution of the model, and the card carries both.
8. Source repositories: 242 of the 244 training-split repositories are permissive (MIT 129, Apache-2.0 98, BSD-3-Clause 12, BSD-2-Clause 2, ISC 1), one is AGPL-3.0 (`turbo-devv/openOMSI`) and one could not be checked and is no longer on GitHub (`Emmy123222/Stellar-GreenPay`), from `eval/results/train_repos_license.csv`. For the 155 repositories of the 357-task union, the split is MIT 85, Apache-2.0 59, BSD-3-Clause 9, BSD-2-Clause 1, AGPL-3.0 1, and the unchecked repository is not among them. Which 153 repositories the 344 tasks use was not saved outside the training volume, so whether the AGPL repository is in the released rows must be read from the staged data (decision D3). MIT, BSD and ISC require keeping the copyright and licence notice with copies or substantial portions; Apache-2.0 requires the licence and any NOTICE text. Tool results are excerpts, many of them whole files, so ship `repo_licenses.csv` and the notices, and have counsel decide whether more is needed (decision D6).
9. Task records: `README.md` and `training/README.md` say the task records "derive from a private benchmark" and are not included, and `data/splits.json` records that `scripts/dataprep/assert_no_private.py` is still "pending" ("private ids never leave the laptop"). The requests, repositories, commits and gold lists are in the trajectories themselves, so releasing the data releases the task text for the 344 tasks. This needs the owner's decision before anything else (D2).
10. The public repository today says the trajectories are not published, in `NOTICE` ("are not published, in this repository or elsewhere"), `training/README.md` ("we chose to keep the release small and to avoid republishing a teacher's outputs"), `docs/MODEL_CARD.md` and the Limitations of the qwen card. Releasing the data reverses a stated choice, so those lines have to change at the same time.

## 4. Pre-release checklist

Run on the staged folder `./hf_stage` before step 1. Each item needs a recorded pass.

| Check | How | Pass condition |
|---|---|---|
| Licence review | Counsel reads section 3 above, the gate prompt and the card | Written sign-off on D2, D3, D5 and D6 |
| Secret and key scan | `gitleaks detect --no-git --source ./hf_stage`, plus a grep for `hf_[A-Za-z0-9]{20,}`, `sk-[A-Za-z0-9]{20,}`, `AKIA[0-9A-Z]{16}`, `-----BEGIN`, `Bearer `, `api[_-]?key`, `token=`, `.modal.run`, `modal.com`, `baseten`, `mongodb.com`, `10gen` | No hit that is a live credential or an internal host. Triage every hit by hand; test constants and public-repo fixtures can be allow-listed with a note |
| Path and PII scan | grep for `/Users/`, `/private/`, `/home/`, `/data/`, `/tmp/`, `henryweller`, e-mail patterns `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}`, phone patterns | No machine path. Any e-mail address must come from a public-repository excerpt (author lines, CODEOWNERS); the owner decides whether to redact those (D8) |
| Test split exclusion | See the script below | Every `task_id` has split `train` in `data/splits.json`; no repository of the dev, test, 200-task held-out, OOD-language or E2E-v4 sets appears; none of the 17 E2E excluded task ids appears |
| Row count and fields | `wc -l`, and a schema check against the allow-list | 1,383 rows, 344 distinct tasks, 153 distinct repositories, 1,350 with `meta.source == teacher` and 33 with `a1`, no field outside the schema |
| Round trip | Re-render 20 random rows with the Qwen template and compare to `messages` | Identical (the training build did this for all 1,383) |
| Gate dry run | After step 3 with the repository still private, request access from a second account | The prompt and fields show; files are blocked until accepted |

Test-split exclusion script (read-only, local files; run it against the staged `data/*.jsonl`):

```python
import json, glob
sp = json.load(open("wayfinder/data/splits.json"))
tasks = sp["tasks"]
held = json.load(open("wayfinder/data/heldout_hard_v1.json"))
excl = {t["task_id"] for t in json.load(open("wayfinder/artifacts/sft/e2e_excluded_tasks.json"))["tasks"]}
rows = [json.loads(l) for f in glob.glob("hf_stage/data/*.jsonl") for l in open(f)]
ids = {r["meta"]["task_id"] for r in rows}
bad_split = [i for i in ids if tasks[i]["split"] != "train"]
repos = {tasks[i]["repo_full"] for i in ids}
other = {s: {t["repo_full"] for t in tasks.values() if t["split"] == s} for s in ("dev", "test", "eval_holdout")}
overlap = {s: sorted(repos & r) for s, r in other.items()}
overlap["heldout_hard"] = sorted(repos & {t["repo"] for t in held["tasks"]})
print(len(rows), len(ids), len(repos), bad_split, overlap, sorted(ids & excl))
# expect: 1383 344 153 [] all-empty lists []
```

Already checked on the 357-task union that the tables use (`wayfinder/artifacts/rl_pool/sft_used_tasks.json`): all 357 are split `train`, none of their 155 repositories is in the dev, test or `eval_holdout` splits or in the 200-task held-out set, and none of the 17 E2E excluded task ids is among them. The 344 tasks are a subset of those within 13 tasks, so rerun it on the staged rows.

## 5. Other files to change at release (not edited now)

- `NOTICE`, `training/README.md`, `docs/MODEL_CARD.md` (the glimmer card) and the Limitations of `docs/MODEL_CARD_QWEN.md`: remove or rewrite the statements that trajectories and task text are not published, and the 344-task sentence in the Training bullet.
- `README.md` line "Task records are not included, because they derive from a private benchmark" if `tasks.jsonl` ships.
- The qwen card line that links the dataset: remove "not yet published".

## 6. Open decisions for the owner

| # | Decision | Why it blocks |
|---|---|---|
| D1 | Publish the data at all? | The repository and cards say trajectories are not published. A memory note records a publication hold with the 2026-11-02 hackathon release as the only exception; this dataset was not part of that release. Needs an explicit yes |
| D2 | May the task text, repository, commit and gold-file lists of the 344 tasks be released? | They derive from a private benchmark, and `assert_no_private.py` has not been run (`private_check: pending`). If no, the data cannot ship in this form |
| D3 | The AGPL-3.0 repository (`turbo-devv/openOMSI`): drop its rows? | Copyleft code excerpts redistributed in a dataset. Dropping them is the safe default, and would lower the counts in both cards |
| D4 | Keep or drop the 3 third-party-host rows? | They rest on a model id and docs, not on checked weights. Dropping gives 1,380 rows and changes the counts |
| D5 | Does counsel agree the dataset itself is not a "Derivative" of Muse Spark Open? | The card works either way (it carries the notice and licence), but the answer sets what the gate can honestly say |
| D6 | Are `repo_licenses.csv` plus the card enough for the MIT, BSD, ISC and Apache-2.0 notice requirements on whole-file excerpts? | The excerpts are not accompanied by per-file copyright headers beyond what the tool output happened to show |
| D7 | Which account owns it (`hweller1`) and is a company review needed before publishing under it? | The affiliation shown in the card and the author line is Voyage AI |
| D8 | Redact e-mail addresses and names found in public-code excerpts? | Public, but personal data; redaction changes the text the model saw |
| D9 | Ship the 33 Agents-A1 rows? | Apache-2.0, no obstacle found; listed so it is a conscious yes |
| D10 | Who approves access requests, on what rule, and how fast? | Manual gating needs a named person and a rule (for example: a named person, a stated purpose, a corporate or academic affiliation) |
| D11 | Release the Glimmer render as well as the Qwen render? | The card says the data is the same examples; only the Qwen render is staged here |

## 7. Where each number in the cards comes from

| Number | Source |
|---|---|
| 1,383 examples, 344 tasks, 153 repositories, 1,350 teacher, 33 A1, 5,683 teacher rows in and 1,350 out, 41.56M Glimmer tokens, 6.9M assistant tokens | `wayfinder/artifacts/sft/teacher_v3_summary.json` |
| 47.18M Qwen tokens, 754 examples that called `semantic_search`, 1,049 teacher examples with five tools | `wayfinder/artifacts/sft/qwen_v3_summary.json`; 1,062 = 1,049 teacher + 13 A1 with `scout5` (`teacher_v3_summary.json`, `semantic_examples_total`) |
| 1,347 self-hosted, 3 third-party host, none from Meta's API | `wayfinder/docs/qwen_moe_sft.md` (provenance paragraph) |
| LoRA rank 64, alpha 128, 250 modules, 76.7M parameters, 2 epochs, lr 5e-5, 460 steps, 4x H200, 88 minutes, seed 0 | `wayfinder/docs/qwen_moe_sft.md` and `wayfinder/artifacts/qwen_sft/train_summary_replica.json` (`seed` 0, `lora_params_m` 76.677, `n_examples` 1383, `docs_per_step` 6, `epochs` 2, `lr` 5e-05). The deck and `training/README.md` say "all attention and MLP projections"; `qwen_moe_sft.md` lists attention projections, the DeltaNet projections and the shared expert only, and the card uses that list |
| 357-task category, language, right-file and request-length tables, 155 repositories, licence split of 155 | `wayfinder/artifacts/rl_pool/sft_used_tasks.json` joined to `wayfinder/data/splits.json` (fields `category`, `language`, `gold_files`, `task_text`) and `wayfinder-public/eval/results/train_repos_license.csv` |
| 200-task strata (model, hybrid, rerank) | Recomputed from `wayfinder/artifacts/fusion_v2/heldout_s{0..3}/sft_replica_semfus.jsonl` (400 rows, 200 tasks, seeds 0 and 1) with the functions in `wayfinder/scripts/analyze_heldout_cascade.py`, joined to `wayfinder/data/heldout_hard_v1.json` for category, language and gold count; overall 0.414, 0.493, 0.531, +0.117 [+0.0949, +0.1392] and precision@12 change -0.1132 [-0.1422, -0.0846] match `heldout_analysis.json`. Strata intervals: 4,000-resample paired bootstrap, my own run |
| 26-task Muse Code counts, files found, tokens, cost, time | `voyage-scout/artifacts/live/muse_v2_full2/rows.jsonl` (52 rows; pairs by `repo` and `prompt_sha`; files found = round(`recall12` x `n_gold`); parent input tokens = `tokens_parent.in`; cost = `muse_usd`; time = `wall_s`). The cost interval and the 23-of-26 and 22-of-26 counts: the first from `wayfinder/docs/hackathon_v2/build_v3.py` speaker notes, the second and third recomputed from the rows |
| Demo takes, excalidraw 10, 10, 10, 5, 8 against 3, 8, 4, 3, 8 | `voyage-scout/artifacts/demo/takes/excalidraw-*/result.json` |
| Loss-pattern check (rho -0.46, p = 0.018; +0.49, p = 0.012; +0.39, p = 0.048) | Spearman with a 20,000-permutation test over the 26 pairs, using `frac_unreachable_nontest`, `n_dirs` and `n_gold` from `heldout_hard_v1.json` (definition in `wayfinder/scripts/rlpool/mine_hard.py`) |
| Tool mix of held-out rollouts (read_file 53.5%, grep 32.5%, glob 6.2%, semantic_search 4.2%, report 3.6%) | `tool_usage` in the 400 rollout rows above; this is model behaviour, not the training mix |

## 8. What could not be computed

The per-tool call mix inside the training trajectories, the exact 344 task ids (so category, language and file-count tables are for the 357-task union), repository type for training tasks (recorded for the held-out set only), the on-disk size of the data, and which repositories the 344 tasks use (so the AGPL question in D3 is open). All of these are one pass over the staged files once they are copied off the training volume.
