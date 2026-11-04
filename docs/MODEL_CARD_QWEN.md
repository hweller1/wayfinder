---
license: other
license_name: apache-2.0-and-muse-model-license
license_link: https://huggingface.co/hweller1/wayfinder-qwen-35b-a3b/blob/main/LICENSE-MUSE-SPARK-OPEN.md
library_name: transformers
pipeline_tag: text-generation
language:
  - en
base_model:
  - Qwen/Qwen3.6-35B-A3B
tags:
  - code-search
  - agentic
  - tool-use
  - subagent
  - lora-merged
  - moe
  - muse
---

# wayfinder-qwen

**Built with Muse Spark Open.**

wayfinder-qwen is Qwen3.6-35B-A3B (35B parameters, 3B active per token) post-trained with LoRA supervised fine-tuning, merged into the base weights, to work as a code-search subagent. A coding agent hands it a brief such as "where does the router decide that a request matches a route with path variables?". The model plans, searches an unfamiliar repository with `grep`, `glob` and `read_file`, optionally calls `semantic_search` over a voyage-code-4 vector index, and returns a ranked list of `path:start-end` ranges with a one-line reason each, a summary and the gaps it could not establish. The parent agent reads that report instead of searching itself.

The weights are bf16 safetensors (26 shards) with the config, tokenizer and chat template of Qwen3.6-35B-A3B unchanged. This repository is a weights release: it holds no training data, trajectories or task text. The training data is not released. A gated release (access by request) is planned.

## Latest results

The full pipeline has three steps. First, the model searches the repository with its own tools and reports a ranked list of files. Second, one hybrid search (voyage-code-4 vectors plus BM25 on hashed identifiers, top 100 chunks rolled up to files) is merged with the model's list by reciprocal rank fusion. Third, Voyage rerank-3 reorders positions 6 to 40 with the instruction "Rank source files a developer must edit or test for this change", and the first 5 are kept as they are.

Recall@12 on 200 held-out tasks (47 repositories never seen in training, 2 seeds per task, task as the unit):

| Step | Recall@12 |
|---|---|
| 1. The model's own list | 0.414 |
| 2. After the hybrid merge | 0.493 |
| 3. After the rerank | 0.531 |

The whole cascade adds +0.117 [+0.095, +0.140], paired by task. Recall@5 moves from 0.336 to 0.345, which is not significant, so these data show no gain at the top of the list. The list gets longer, and precision@12 falls from 0.416 to 0.303. The cascade adds about 0.4 s per search.

These are properties of the Scout pipeline around the weights, which needs the voyage-code-4 index and the rerank call. The weights alone give the 0.414 line.

Coding pass rate (34 tasks from 8 repositories, 2 seeds, a coder with no search tools that can open only the files it is handed): 0.824 [0.71, 0.93] with wayfinder-qwen and the cascade, against 0.074 with no search and 0.971 when handed the right files. Three other searchers (Sonnet 5.5 high, GPT-6.1 Sol and Agents-A1-35B) scored 0.85 to 0.87 in an earlier run without the cascade. The differences between searchers are unresolved on 34 tasks, and 16 of the tasks come from one repository.

Inside Muse Code (26 tasks from 18 repositories, one run per task, main agent muse-spark-1.3-contributor), adding Wayfinder cut cost per task from $0.342 to $0.195 and the tokens the main agent reads from 887K to 441K. The right files found and precision@12 are the same within noise (recall@12 0.590 to 0.536, difference -0.053 [-0.127, +0.015]). Recall@5 is lower, and the median search is about 25 s slower. The claim is cost and tokens, not better accuracy.

## Serving

Tested with vLLM 0.32.0, one H100, online FP8 quantization of the bf16 checkpoint (`--quantization fp8`, not Qwen's own FP8 release) and, for the speed numbers below, multi-token prediction with 2 draft tokens. Every number in this card was measured with FP8. bf16 serving was not evaluated. The tool-call and reasoning parsers are required; without `--tool-call-parser qwen3_coder` the tool calls come back as plain text. The checkpoint's own chat template is used (no `--chat-template`).

```bash
pip install "vllm==0.32.0"
vllm serve hweller1/wayfinder-qwen-35b-a3b \
  --served-model-name wayfinder-qwen \
  --quantization fp8 --max-model-len 65536 --gpu-memory-utilization 0.92 \
  --enable-prefix-caching --language-model-only --max-num-seqs 32 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --override-generation-config '{"temperature": 0.85, "top_p": 0.95, "top_k": 20}' \
  --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
  --port 8000
```

Drop the `--speculative-config` line to serve without multi-token prediction. The prediction head is the one Qwen shipped. It was not fine-tuned. Its acceptance rate was 70.2% of draft tokens (60,421 of 86,045) in the 12-rollout latency run and 70.5% and 72.5% in two end-to-end launches. The held-out retrieval numbers below were produced with it off. Quality with it on was checked on 12 rollouts, on the end-to-end coding study and on the Muse Code run, not on the held-out set.

The model reasons before each tool call and was trained and evaluated with its own earlier reasoning sent back on every request. The chat template renders earlier reasoning, but vLLM 0.32.0 reads only the `reasoning` key of an assistant message and silently drops `reasoning_content`, which gives empty think blocks in the rendered prompt. vLLM also returns reasoning under the `reasoning` key, so copy it back unchanged into `reasoning` when you append the assistant turn. A client that drops earlier reasoning, or sends it under `reasoning_content` only, is outside what the model was trained on. The chat template needs tool-call arguments as a dict. Sampling used in the evaluations: temperature 0.85, top_p 0.95, top_k 20, thinking on (the template default).

Example request (one tool shown; the full harness offers `grep`, `glob`, `read_file`, `semantic_search` and a terminal `report` tool):

```python
import json, urllib.request
tools = [{"type": "function", "function": {
    "name": "grep",
    "description": "Search the repository with ripgrep; files are ranked by match density.",
    "parameters": {"type": "object",
                   "properties": {"pattern": {"type": "string"},
                                  "context": {"type": "integer", "maximum": 5}},
                   "required": ["pattern"]}}}]
body = {"model": "wayfinder-qwen", "tools": tools, "tool_choice": "auto",
        "max_tokens": 1024, "temperature": 0.85, "top_p": 0.95,
        "messages": [
            {"role": "system", "content": "You are a code-search subagent. Find the files that answer the brief, then report ranked path:start-end ranges with reasons. Report as soon as the evidence covers the brief; do not exhaust the turn budget."},
            {"role": "user", "content": "Where is the HTTP retry backoff maximum delay configured?"}]}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
msg = json.load(urllib.request.urlopen(req))["choices"][0]["message"]
print([c["function"]["name"] for c in msg.get("tool_calls", [])])   # expect ['grep', ...]
```

The tool schemas and system prompt used in training and evaluation come from the reference harness in the GitHub repository (`serve/scout_tool_specs.json`, `serve/scout_product_prompt.md`, also in this repository). Keep the convergence line "Report as soon as the evidence covers the brief; do not exhaust the turn budget." in the system prompt, because the training examples carry it.

## Intended use

Use it as the explore or search subagent under a stronger parent model, inside a harness that executes its tool calls and offers the five tools (`grep`, `glob`, `read_file`, `semantic_search`, `report`). Its output is a list of leads for the parent to verify. It suits cases where latency and GPU cost per search matter.

Out of scope:

- Use outside that tool surface (other tool names, no `report` tool, other prompts). It is untested there.
- General chat, general code generation, or any task where retention of general skills matters. Retention was not measured.
- Image or video input. The vision encoder of the base is present in the weights but was never trained or tested here (`--language-model-only`).
- Use as the only safeguard on code changes.
- Searching repositories you are not permitted to read.
- Any use prohibited by the Muse Spark Open Usage Policy.

## How it was trained

Tasks are mined from real commits in public repositories. The commit subject line is the question and the files the commit changed are the answers (the gold files). Muse Spark Open, the teacher, searched each task with `grep`, `glob`, `read_file` and `report` (1,062 of the 1,383 examples also offered `semantic_search` over a voyage-code-4 index). Only runs that found every gold file, opened every file they reported and called `report` in time were kept, and the 1,350 teacher examples that survived also stayed under the task's 75th-percentile turn count and 49,152 tokens (5,683 teacher rows in, 1,350 out). With 33 examples from Agents-A1-35B the set is 1,383 examples from 344 tasks in 153 of the 244 training-split repositories, 242 of those 244 under permissive licenses.

Qwen3.6-35B-A3B was then fine-tuned on those examples with LoRA supervised fine-tuning: rank 64, alpha 128, on the attention projections, the linear-attention projections and the shared expert of every MoE block (250 modules, 76.7M trainable parameters; the routed experts and the router are frozen), assistant-only loss, 2 epochs at learning rate 5e-5, 460 steps of 6 documents, 4x H200 data parallel, 88 minutes of wall time, then merged into the base weights. This is one training run with one seed. The recipe was not ablated: no run varied the data, rank or schedule against the released one, apart from the dev-set-only variant described under Training.

At inference time the Scout pipeline can refine the model's list. After the model reports, one hybrid search (voyage-code-4 vectors plus BM25) is merged with its list, and a Voyage rerank-3 pass reorders positions 6 to 40. On the 200 held-out tasks (47 repositories, two seeds) recall@12 goes from 0.414 for the model's own list to 0.493 after the hybrid merge and 0.531 after the rerank (+0.117 [+0.095, +0.140], paired by task). This is a property of the Scout pipeline and not of the weights alone: the same weights without the cascade score the first number. It also lengthens the list, so precision@12 falls from 0.416 to 0.303. It was measured in its own run of the held-out tasks, so it is not directly comparable to the reported-recall table under Evaluation.

## Training

- Base model: Qwen3.6-35B-A3B (`Qwen/Qwen3.6-35B-A3B`, Apache-2.0), a hybrid of Gated DeltaNet and gated-attention layers with 256 routed experts per MoE layer.
- Method: LoRA supervised fine-tuning, rank 64, alpha 128, on the attention projections of the 10 gated-attention layers, the `in_proj_qkv`, `in_proj_z` and `out_proj` projections of the 30 Gated DeltaNet layers, and the shared expert of all 40 MoE blocks (250 modules, 76.7M trainable parameters). The 256 routed experts and the router are frozen. The adapter was merged into the base shards and saved as bf16, so serving needs no LoRA support.
- Data: 1,383 examples rendered with the official Qwen3.6 chat template (47.2M tokens, 6.9M of them assistant tokens, all carrying loss; every assistant turn keeps its reasoning). They come from 344 tasks in public repositories, all from the training split, none from the test, development or held-out sets. The tasks come from 153 of the 244 repositories in the training split. Of those 244, 242 are under permissive licenses (MIT 129, Apache-2.0 98, BSD-3-Clause 12, BSD-2-Clause 2, ISC 1), one is AGPL-3.0 and one is no longer on GitHub, so its license could not be checked (GitHub license API, 2026-11-02; list in `eval/results/train_repos_license.csv` in the GitHub repository). The 344 tasks are not listed in this release, so these counts cover the whole training split, and either of those two repositories may be among the 153. 1,350 examples are Muse Spark Open teacher trajectories (1,347 from self-hosted replicas of the open weights, 3 from a third-party host of the same open weights; none from the Meta API) and 33 are Agents-A1-35B trajectories. Only runs that found every gold file, opened every file they reported and called `report` in time were kept.
- Schedule: 2 epochs (460 steps of 6 documents), learning rate 5e-5 with cosine decay to 10%, 5 warmup steps, one document per sequence (no packing, because the recurrent layers would leak state between packed documents), 4x H200, 88 minutes.
- Single training run (seed 0). Results describe this checkpoint, not the recipe. A second variant that upweighted examples where semantic search found a gold file grep had missed was trained and evaluated on the development set only. It was no better and is not released.

The trajectories and task text are not in this repository. The training data is not released. A gated release (access by request) is planned.

## Task shapes it was trained on

What the examples look like: one user message holding a short request, then a loop of assistant turns (reasoning, then one or more tool calls) and tool results, ending in a `report` call. The request is the subject line of a real commit, for example "Sanitize username used in proxy job names by splitting on underscores" or "Remove deprecations from OllamApi and AnthropicApi". Subject lines are one line each, median 8 words (range 3 to 98). The right files are the files that commit touched, including test files.

One limit on these tables: the list of the 344 tasks actually used was not saved outside the training volume. The tables below are for the 357 training-split tasks used across all versions of this training data (155 repositories), which differ from the 344 by at most 13 tasks. All 357 are in the training split.

| Task category (357 tasks) | Tasks | Share |
|---|---|---|
| implementing_new_features | 138 | 38.7% |
| refactoring | 74 | 20.7% |
| paying_tech_debt | 54 | 15.1% |
| modernize | 36 | 10.1% |
| cybersecurity_bug_finding | 29 | 8.1% |
| onboarding | 19 | 5.3% |
| db_schema_rot | 4 | 1.1% |
| qa_over_codebases | 3 | 0.8% |
| splitting_monolith | 0 | 0.0% |

Categories come from the benchmark the tasks were mined from and describe the commit, not a hand label. No category is named bug fixing; cybersecurity_bug_finding is the nearest, at 8% of the tasks.

| Language (majority gold-file extension) | Tasks | Share |
|---|---|---|
| Python | 83 | 23.2% |
| TypeScript/JavaScript | 72 | 20.2% |
| Rust | 71 | 19.9% |
| C/C++ | 41 | 11.5% |
| Go | 38 | 10.6% |
| Ruby | 15 | 4.2% |
| Other | 14 | 3.9% |
| Swift | 9 | 2.5% |
| C# | 8 | 2.2% |
| Java/Kotlin | 6 | 1.7% |

| Right files per task | Tasks | Share |
|---|---|---|
| 1 | 62 | 17.4% |
| 2 to 3 | 132 | 37.0% |
| 4 to 6 | 116 | 32.5% |
| 7 or more | 47 | 13.2% |

The median is 3 right files per task (range 1 to 12, mean 3.8). The 200-task held-out set used in Evaluation is harder in shape: 3 to 15 right files, median 6, at least one non-test right file that neither the words of the request nor a grep for its rarest terms reaches.

Repository type was not recorded for the training tasks (it exists only for the held-out set). Tasks per repository: 155 repositories, at most 13 tasks from one repository (starkillerOG/reolink_aio), then 10 each from github/gh-aw-mcpg and pixlcore/xyops. Repository licenses of those 155: MIT 85, Apache-2.0 59, BSD-3-Clause 9, BSD-2-Clause 1, AGPL-3.0 1.

Tool mix. What is known from the build summaries: of the 1,383 examples, 1,062 offered `semantic_search` alongside the other four tools and 754 of them called it at least once; the rest used `grep`, `glob` and `read_file` only. The per-tool call counts inside the training trajectories were not recomputed, because the trajectories are on the training volume and not in the repositories used for this card. For orientation only, and not the training mix: in the model's own held-out rollouts (400 rollouts, median 8 turns) `read_file` was 53.5% of tool calls, `grep` 32.5%, `glob` 6.2%, `semantic_search` 4.2% and `report` 3.6%.

## Evaluation

All numbers use the same explore loop, system prompt, 40-turn cap and two seeds, with earlier reasoning retained and verified to reach the prompt. "Reported recall" is the share of gold files in the `report` call (at most 12 files), which is what a parent agent receives. Intervals are 95% bootstrap intervals over tasks. The semantic arm adds `semantic_search` over a voyage-code-4 index. Every number below is from the FP8 serving described above.

### Held-out retrieval: 200 tasks, 47 repositories never seen in training

Two seeds per task, so 400 rollouts per arm and 800 in total, with no errors. Run once, with MTP off, and not used to choose between models.

| Model | Reported recall, grep | Reported recall, with semantic search | Recall@5, grep | Turns per task (grep) | Ended by report (grep) |
|---|---|---|---|---|---|
| wayfinder-qwen | 0.415 [0.377, 0.452] | 0.427 [0.388, 0.465] | 0.335 [0.301, 0.368] | 9.5 | 0.96 |
| Agents-A1-35B, same tasks | 0.313 [0.280, 0.348] | 0.352 [0.319, 0.387] | 0.247 [0.217, 0.279] | 34.1 | 0.47 |

Paired by task, wayfinder-qwen minus Agents-A1: +0.102 [+0.078, +0.127] on the grep arm and +0.075 [+0.051, +0.098] with semantic search. The all-gold rate (every gold file in the report) is 0.07 on the grep arm. The malformed tool-call rate was 0.05% of calls, and no tool markup leaked into message content.

On a 50-task development set with two seeds (the gate before the held-out run, so a selection set), reported recall was 0.375 [0.293, 0.458] on the grep arm and 0.385 [0.305, 0.468] with semantic search.

### Speed and cost per search

12 development tasks, seed 0, grep arm, one stream, one H100 container, same harness. This is a small sample that sets the shape and not tight intervals.

| Setup | Median wall time per search | Median model time per turn (p90) | Median turns | Decode rate |
|---|---|---|---|---|
| wayfinder-qwen, FP8, MTP on (2 draft tokens) | 19.7 s | 2.1 s (6.6 s) | 9.5 | 286 tok/s |
| wayfinder-qwen, FP8, MTP off | 43.5 s | 3.6 s (9.7 s) | 10.5 | 185 tok/s |

Reported recall on these 12 tasks was 0.20 with MTP on and 0.19 with it off. With n=12 no inference is drawn from that.

At a concurrency of 14 on the held-out set, GPU cost per rollout at the H100 list price of $3.95 per hour (vLLM load included) was $0.0150 on the grep arm and $0.0144 with semantic search (400 rollouts per arm).

### End to end: pass rate of a coding agent handed the searcher's files

34 tasks from 8 repositories, two seeds (68 runs per condition), selected because a coder cannot solve them without finding the code. A fixed coding model has no search tools and can only open the files the subagent returned; held-out tests are absent from its sandbox. MTP was on for wayfinder-qwen in this study.

| What the coder is handed | Pass rate |
|---|---|
| No search | 0.07 (5 of 68) |
| wayfinder-qwen | 0.87 (59 of 68) |
| Gold files (ceiling) | 0.97 (66 of 68) |

wayfinder-qwen lifts the pass rate by +0.794 [+0.676, +0.897] over no search; the lift survives a repository-cluster bootstrap ([+0.595, +0.871]). The files handed over contained 0.73 of the gold files on average. Caveat: the set is concentrated, with 16 of the 34 tasks from one repository, so the effective sample is closer to 8 repositories, and the searchers sit near the ceiling. A separate, stricter task set reached only 4 usable tasks (8 runs): wayfinder-qwen passed 2 of 8 against 1 of 8 with no search, and nothing was resolved there. This run did not use the cascade; the cascade run (0.824 [0.71, 0.93], 34 tasks, 2 seeds) is under Latest results.

### Inside Muse Code: 26 tasks, one seed

wayfinder-qwen as the chat model behind a Scout `explore` call in Muse Code, on 26 tasks from 18 repositories, one seed, MTP on. Calibration scale only.

| Arm | Recall@5 | Recall@12 | Precision@12 | Files reported | Median session |
|---|---|---|---|---|---|
| Stock Muse Code, no Scout | 0.413 | 0.600 | 0.381 | 15.0 | 65 s |
| Scout with wayfinder-qwen | 0.358 | 0.496 | 0.365 | 10.9 | 118 s |

wayfinder-qwen is below stock Muse Code on recall@12: -0.104 [-0.163, -0.051] (the repository-cluster interval also excludes zero). Part of that gap is report length (10.9 files against 15.0), and precision@12 is level. A Scout search with wayfinder-qwen is still about 1.8 times slower than stock Muse Code on these tasks.

## Where it does well and where it does not

Two evaluations carry the strata below. Both are hard by construction (see Task shapes), so they say how the model does on tasks where the right files are not named by the request, not how it does on an average task. Intervals are 95% paired bootstrap intervals over tasks and are not adjusted for the number of strata, so read the small ones as direction only. A stratum with fewer than 8 tasks is shown as "not enough tasks".

### 200 held-out tasks: the model's own list, and after the cascade

Recall@12 of three lists on the same rollouts (two seeds per task, seeds averaged, task as the unit): the model's own list, the list after hybrid search merged in, and the list after rerank (the final list). "Change" is final minus the model's own list.

| Stratum | Tasks | Model's own list | After hybrid | After rerank | Change |
|---|---|---|---|---|---|
| All | 200 | 0.414 | 0.493 | 0.531 | +0.117 [+0.095, +0.140] |
| **Category** | | | | | |
| implementing_new_features | 119 | 0.392 | 0.459 | 0.503 | +0.112 [+0.083, +0.142] |
| paying_tech_debt | 36 | 0.466 | 0.558 | 0.593 | +0.127 [+0.079, +0.177] |
| refactoring | 25 | 0.477 | 0.570 | 0.627 | +0.151 [+0.092, +0.217] |
| modernize | 17 | 0.388 | 0.478 | 0.467 | +0.080 [+0.027, +0.136] |
| cybersecurity_bug_finding, db_schema_rot | 2, 1 | not enough tasks | | | |
| **Language** | | | | | |
| TypeScript/JavaScript | 50 | 0.414 | 0.508 | 0.548 | +0.134 [+0.088, +0.185] |
| Java/Kotlin | 28 | 0.389 | 0.454 | 0.494 | +0.105 [+0.045, +0.173] |
| Go | 18 | 0.428 | 0.417 | 0.508 | +0.080 [+0.000, +0.173] |
| Other | 18 | 0.340 | 0.415 | 0.430 | +0.090 [+0.032, +0.159] |
| C# | 18 | 0.509 | 0.605 | 0.652 | +0.143 [+0.102, +0.183] |
| Swift | 16 | 0.453 | 0.495 | 0.525 | +0.073 [+0.018, +0.136] |
| Rust | 16 | 0.390 | 0.556 | 0.554 | +0.164 [+0.109, +0.223] |
| C/C++ | 15 | 0.469 | 0.513 | 0.583 | +0.113 [+0.027, +0.209] |
| Python | 13 | 0.326 | 0.462 | 0.476 | +0.151 [+0.086, +0.221] |
| Ruby | 8 | 0.434 | 0.516 | 0.521 | +0.087 [+0.000, +0.191] |
| **Right files per task** | | | | | |
| 3 to 5 | 77 | 0.485 | 0.552 | 0.567 | +0.082 [+0.052, +0.116] |
| 6 to 8 | 62 | 0.405 | 0.497 | 0.545 | +0.141 [+0.096, +0.190] |
| 9 or more | 61 | 0.334 | 0.415 | 0.472 | +0.138 [+0.105, +0.172] |

(No held-out task has fewer than 3 right files, so there is no 1 to 2 row.)

Where it does better: the model's own list is strongest on C# (0.509, 18 tasks), on tasks with 3 to 5 right files (0.485, 77 tasks) and on refactoring and paying_tech_debt (0.477 and 0.466). The cascade adds recall in every stratum with at least 8 tasks, and the interval excludes zero in all of them except Go and Ruby, where the lower end is 0.000. Where it ties: in Go and Ruby the gain is not separable from zero (18 and 8 tasks), and for Rust and modernize the hybrid merge did the work and the rerank added nothing (Rust 0.556 to 0.554, modernize 0.478 to 0.467). Where it does worse: nowhere in this table is the final list below the model's own list, but the model's own list is weakest on Python (0.326, 13 tasks), Other languages (0.340, 18 tasks) and on tasks with 9 or more right files (0.334, 61 tasks). Recall of its own list falls as the number of right files rises (0.485, 0.405, 0.334), partly because a 12-file report caps recall on large tasks. The cascade also lengthens the list and lowers precision@12 from 0.416 to 0.303 (-0.113 [-0.142, -0.085]). Repository-type strata exist for this set (26 types); only 10 have 8 or more tasks and none is reported here.

### 26 tasks in Muse Code: Muse Code alone against Muse Code with Wayfinder

A second Muse Code run on 26 of the same held-out tasks (18 repositories, one run per task, default configuration with direct hand-off and the cascade, wayfinder-qwen behind Scout). Same parent model on both sides, so a pair differs only by the Wayfinder `explore` call. Files found is the number of right files in the final list of 12 or fewer. The table under Evaluation is an earlier run of the same 26 tasks under a different protocol; both show Muse Code with Scout below stock Muse Code on recall@12.

| | Muse Code alone | With Wayfinder | Difference |
|---|---|---|---|
| Right files found (of 171 across 26 tasks) | 94 | 90 | -4 |
| Recall@12 | 0.590 | 0.536 | -0.053 [-0.127, +0.015] |
| Parent input tokens per task | 887K | 441K | fewer on 23 of 26 tasks |
| Muse cost per task (list price, excludes GPU) | $0.342 | $0.195 | -$0.147 [-$0.205, -$0.096] |
| Median wall time | 80 s | 105 s | slower on 22 of 26 tasks |

Per task: more right files with Wayfinder on 6 tasks, the same on 11, fewer on 9.

| Stratum | Tasks | More / same / fewer | Files found, alone to with |
|---|---|---|---|
| implementing_new_features | 10 | 2 / 3 / 5 | 30 to 24 |
| paying_tech_debt | 8 | 2 / 4 / 2 | 30 to 31 |
| refactoring | 5 | not enough tasks | |
| modernize, db_schema_rot | 2, 1 | not enough tasks | |
| TypeScript/JavaScript | 9 | 2 / 6 / 1 | 34 to 36 |
| C#, Java/Kotlin, Other | 3 each | not enough tasks | |
| Go, Swift | 2 each | not enough tasks | |
| 3 to 5 right files | 11 | 1 / 5 / 5 | 31 to 25 |
| 6 to 8 right files | 9 | 3 / 3 / 3 | 32 to 31 |
| 9 or more right files | 6 | not enough tasks | |

Reading: with Wayfinder the coding agent reads about half as many tokens and spends about 43% less on Muse, and finds about as many right files overall, with wide uncertainty on a 26-task, one-run sample. It is slower. Where it loses ground in these data is implementing_new_features (5 of 10 tasks with fewer files) and tasks with 3 to 5 right files (5 of 11 with fewer); it ties in paying_tech_debt, TypeScript/JavaScript and tasks with 6 to 8 right files. No stratum with 8 or more tasks shows a gain of more than two files, and these samples are too small to put intervals on the per-stratum differences.

Single runs are noisy. In the three-task demo batch (5 runs each, same excalidraw task), Muse Code alone found 10, 10, 10, 5 and 8 of 11 right files and Muse Code with Wayfinder found 3, 8, 4, 3 and 8, a median of 10 against 4. In the 26-task run the same excalidraw task went the other way, 6 to 8. The rust-analyzer "Remove `allow`s that are no longer necessary" task (12 right files) was a tie at 1 file in the 26-task run, and has no scored demo take in the files used here.

Hypothesis about the losses, offline and exploratory, not a tested claim: the tasks where Wayfinder loses are those whose right files are not named after the request (excalidraw "Remove image preview on image insertion", rust-analyzer "Remove `allow`s...": the right files are spread over the codebase and their paths and contents do not contain the words of the request), so grep has no lexical way in, and the Muse parent's broader search may cover more ground. A check we could run offline on the 26 pairs, using the held-out set's own labels for how many non-test right files the request words do not reach: the tasks where Wayfinder found fewer files had a median of 0.67 of their non-test right files unreachable (9 tasks), against 0.40 for ties (11) and 0.37 for tasks where it found more (6). Rank correlation between that share and the change in recall@12 is -0.46 (permutation p = 0.018, n = 26). That points the same way as the hypothesis. It is one run, and it is one of five measures looked at. Two others were nominally significant too: number of directories (rho +0.49, p = 0.012), which runs the opposite way to a scatter explanation, with more directories going with better results for Wayfinder, and number of right files (rho +0.39, p = 0.048). With five looks the -0.46 does not survive a Bonferroni correction (0.018 x 5 = 0.09), the share is entangled with the number of right files, and no pre-registered test was run. Treat it as a lead, and note that the scatter half of the hypothesis is not supported.

## Limitations

- Retrieval is below frontier models in the same loop (see the search benchmark in the GitHub repository), and the model ends by calling `report` on 0.96 of grep-arm rollouts.
- In Muse Code, recall@12 is 0.496 against 0.600 for stock Muse Code without Scout (26 tasks, one seed), and the Scout path is slower than stock end to end (118 s against 65 s median).
- Whether thinking was on in every `explore` call of the Muse Code run was not verified from a trace; the run left thinking at the Qwen template default instead of setting it per request as the held-out harness does. Treat that part of the Muse Code result as unconfirmed.
- The held-out numbers were produced with MTP off, and the speed numbers and the end-to-end and Muse Code results with MTP on. Quality with MTP on has not been measured on the held-out set.
- The speed figures come from 12 tasks.
- The semantic index adds no separable recall: +0.012 over the grep arm on the held-out set, not distinguishable from zero.
- All evaluations used FP8 online quantization. Other precisions and engines were not tested.
- Only the shared expert and attention projections were trained. Routed-expert LoRA, a second seed and other recipes were not tried.
- English-only task wording. Evaluated on public repositories only. Performance on private or unusual codebases is unknown.
- This is one training run; Swift, C# and Java/Kotlin are thin in the training data.
- Trained on public code whose individual licenses were not cleared task by task.
- It inherits the biases and failure modes of the base and teacher models. Test it for your use before deployment.

## License and notices

Three sets of terms apply. Keep all license files in this repository with any copy of the weights.

1. **Qwen3.6-35B-A3B.** The base model is released by Alibaba Cloud under the Apache License 2.0 (`LICENSE`).
2. **Muse Spark Open derivative terms.** Because the model was fine-tuned on outputs of Muse Spark Open, it is a "Derivative" under the Muse Model License, version "Muse Spark Open" (`LICENSE-MUSE-SPARK-OPEN.md`, which includes the Usage Policy as Exhibit A; also provided as `USE_POLICY-MUSE-SPARK-OPEN.md`). The license permits publishing and distributing Derivatives. Its conditions: the license and Usage Policy travel with the weights; **"Built with Muse Spark Open"** must be displayed in the user interface or accompanying documentation of any product or service that uses this model; and any party offering this model as a Model as a Service with aggregate gross revenue above US$20 million in a twelve-month period needs a separate written agreement with Meta. Your use must comply with the Usage Policy.
3. **Agents-A1.** 33 of the 1,383 training examples are trajectories from Agents-A1-35B (InternScience), released under the Apache License 2.0. The training recipe adapts the Agents-A1 work (arXiv 2606.30616).

Meta claims no ownership of outputs generated by the model. The software and outputs are provided as is, without warranty. See `NOTICE`.

## Citation and links

```bibtex
@misc{wayfinder_qwen_2026,
  title  = {wayfinder-qwen: a fast code-search subagent post-trained from Qwen3.6-35B-A3B},
  author = {Weller, Henry},
  year   = {2026},
  note   = {Voyage AI. Built with Muse Spark Open.},
  url    = {https://huggingface.co/hweller1/wayfinder-qwen-35b-a3b}
}
```

- Code, evaluation and serving harness: https://github.com/hweller1/wayfinder
- Voyage AI: https://www.voyageai.com
- Base model: Qwen3.6-35B-A3B, Alibaba Cloud. Teacher: Muse Spark Open 816B-A42B, Meta Platforms, Inc.

## Related models

The first model trained with the same recipe, [wayfinder-glimmer](https://huggingface.co/hweller1/wayfinder-glimmer-30b) (Muse Glimmer 30B base), is a separate repository and is not part of the final results.
