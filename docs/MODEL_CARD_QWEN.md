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

It is the faster sibling of [wayfinder-glimmer](https://huggingface.co/hweller1/wayfinder-glimmer-30b), trained on the same examples with the same recipe. On a single H100 and a single stream it finishes a search in about 20 to 44 seconds where wayfinder-glimmer takes about 175 seconds. In exchange its retrieval is a little weaker: on the 200-task held-out set its reported recall is 0.036 below wayfinder-glimmer on the grep arm. See Evaluation and Limitations before choosing between them.

The weights are bf16 safetensors (26 shards) with the config, tokenizer and chat template of Qwen3.6-35B-A3B unchanged. This repository is a weights release: no training data, trajectories or task text are included.

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

Use it as the explore or search subagent under a stronger parent model, inside a harness that executes its tool calls and offers the five tools (`grep`, `glob`, `read_file`, `semantic_search`, `report`). Its output is a list of leads for the parent to verify. Choose it over wayfinder-glimmer when latency or GPU cost per search matters more than the last few points of recall.

Out of scope:

- Use outside that tool surface (other tool names, no `report` tool, other prompts). It is untested there.
- General chat, general code generation, or any task where retention of general skills matters. Retention was not measured.
- Image or video input. The vision encoder of the base is present in the weights but was never trained or tested here (`--language-model-only`).
- Use as the only safeguard on code changes.
- Searching repositories you are not permitted to read.
- Any use prohibited by the Muse Spark Open Usage Policy.

## Training

- Base model: Qwen3.6-35B-A3B (`Qwen/Qwen3.6-35B-A3B`, Apache-2.0), a hybrid of Gated DeltaNet and gated-attention layers with 256 routed experts per MoE layer.
- Method: LoRA supervised fine-tuning, rank 64, alpha 128, on the attention projections of the 10 gated-attention layers, the `in_proj_qkv`, `in_proj_z` and `out_proj` projections of the 30 Gated DeltaNet layers, and the shared expert of all 40 MoE blocks (250 modules, 76.7M trainable parameters). The 256 routed experts and the router are frozen. The adapter was merged into the base shards and saved as bf16, so serving needs no LoRA support.
- Data: the same 1,383 examples as wayfinder-glimmer, re-rendered with the official Qwen3.6 chat template and not re-selected (47.2M tokens, 6.9M of them assistant tokens, all carrying loss; every assistant turn keeps its reasoning). They come from 344 tasks in public repositories, all from the training split, none from the test, development or held-out sets. The tasks come from 153 of the 244 repositories in the training split. Of those 244, 242 are under permissive licenses (MIT 129, Apache-2.0 98, BSD-3-Clause 12, BSD-2-Clause 2, ISC 1), one is AGPL-3.0 and one is no longer on GitHub, so its license could not be checked (GitHub license API, 2026-11-02; list in `eval/results/train_repos_license.csv` in the GitHub repository). The 344 tasks are not listed in this release, so these counts cover the whole training split, and either of those two repositories may be among the 153. 1,350 examples are Muse Spark Open teacher trajectories (1,347 from self-hosted replicas of the open weights, 3 from a third-party host of the same open weights; none from the Meta API) and 33 are Agents-A1-35B trajectories. Only runs that found every gold file, opened every file they reported and called `report` in time were kept.
- Schedule: 2 epochs (460 steps of 6 documents), learning rate 5e-5 with cosine decay to 10%, 5 warmup steps, one document per sequence (no packing, because the recurrent layers would leak state between packed documents), 4x H200, 88 minutes.
- Single training run (seed 0). Results describe this checkpoint, not the recipe. A second variant that upweighted examples where semantic search found a gold file grep had missed was trained and evaluated on the development set only. It was no better and is not released.

The trajectories and task text are not published.

## Evaluation

All numbers use the same explore loop, system prompt, 40-turn cap and two seeds, with earlier reasoning retained and verified to reach the prompt. "Reported recall" is the share of gold files in the `report` call (at most 12 files), which is what a parent agent receives. Intervals are 95% bootstrap intervals over tasks. The semantic arm adds `semantic_search` over a voyage-code-4 index. Every number below is from the FP8 serving described above.

### Held-out retrieval: 200 tasks, 47 repositories never seen in training

Two seeds per task, so 400 rollouts per arm and 800 in total, with no errors. Run once, with MTP off, and not used to choose between models.

| Model | Reported recall, grep | Reported recall, with semantic search | Recall@5, grep | Turns per task (grep) | Ended by report (grep) |
|---|---|---|---|---|---|
| wayfinder-qwen | 0.415 [0.377, 0.452] | 0.427 [0.388, 0.465] | 0.335 [0.301, 0.368] | 9.5 | 0.96 |
| wayfinder-glimmer, same tasks | 0.452 [0.412, 0.491] | 0.446 [0.406, 0.484] | 0.360 [0.324, 0.399] | 9.9 | 1.00 |
| Agents-A1-35B, same tasks | 0.313 [0.280, 0.348] | 0.352 [0.319, 0.387] | 0.247 [0.217, 0.279] | 34.1 | 0.47 |

Paired by task, wayfinder-qwen minus wayfinder-glimmer on reported recall: -0.036 [-0.055, -0.018] on the grep arm (the repository-cluster interval, [-0.054, -0.019], agrees) and -0.018 [-0.037, -0.000] with semantic search. Minus Agents-A1: +0.102 [+0.078, +0.127] on the grep arm and +0.075 [+0.051, +0.098] with semantic search. The all-gold rate (every gold file in the report) is 0.07 on the grep arm against 0.10 for wayfinder-glimmer. The malformed tool-call rate was 0.05% of calls, and no tool markup leaked into message content.

On a 50-task development set with two seeds (the gate before the held-out run, so a selection set), reported recall was 0.375 [0.293, 0.458] on the grep arm and 0.385 [0.305, 0.468] with semantic search.

### Speed and cost per search

12 development tasks, seed 0, grep arm, one stream, one H100 container, same harness. This is a small sample that sets the shape and not tight intervals.

| Setup | Median wall time per search | Median model time per turn (p90) | Median turns | Decode rate |
|---|---|---|---|---|
| wayfinder-qwen, FP8, MTP on (2 draft tokens) | 19.7 s | 2.1 s (6.6 s) | 9.5 | 286 tok/s |
| wayfinder-qwen, FP8, MTP off | 43.5 s | 3.6 s (9.7 s) | 10.5 | 185 tok/s |
| wayfinder-glimmer, bf16 | 175.4 s | 13.2 s (36.3 s) | 10.0 | 41 tok/s |

Reported recall on these 12 tasks was 0.20 with MTP on, 0.19 with it off and 0.23 for wayfinder-glimmer. With n=12 no inference is drawn from that.

At a concurrency of 14 on the held-out set, GPU cost per rollout at the H100 list price of $3.95 per hour (vLLM load included) was $0.0150 on the grep arm and $0.0144 with semantic search for wayfinder-qwen, against $0.0294 and $0.0281 for wayfinder-glimmer (400 rollouts per arm).

### End to end: pass rate of a coding agent handed the searcher's files

34 tasks from 8 repositories, two seeds (68 runs per condition), selected because a coder cannot solve them without finding the code. A fixed coding model has no search tools and can only open the files the subagent returned; held-out tests are absent from its sandbox. MTP was on for wayfinder-qwen in this study.

| What the coder is handed | Pass rate |
|---|---|
| No search | 0.07 (5 of 68) |
| wayfinder-qwen | 0.87 (59 of 68) |
| wayfinder-glimmer | 0.87 (59 of 68) |
| Gold files (ceiling) | 0.97 (66 of 68) |

wayfinder-qwen lifts the pass rate by +0.794 [+0.676, +0.897] over no search; the lift survives a repository-cluster bootstrap ([+0.595, +0.871]). Against wayfinder-glimmer the difference is 0.000 [-0.044, +0.044], so a pass-rate difference larger than about 0.06 between the two is excluded on these tasks. The files handed over contained 0.73 of the gold files on average (wayfinder-glimmer 0.75). Caveat: the set is concentrated, with 16 of the 34 tasks from one repository, so the effective sample is closer to 8 repositories, and the searchers sit near the ceiling. A separate, stricter task set reached only 4 usable tasks (8 runs): wayfinder-qwen passed 2 of 8 against 1 of 8 with no search, and nothing was resolved there. This arm was added after the other conditions were reported.

### Inside Muse Code: 26 tasks, one seed

wayfinder-qwen as the chat model behind a Scout `explore` call in Muse Code, on 26 tasks from 18 repositories, one seed, MTP on. Calibration scale only.

| Arm | Recall@5 | Recall@12 | Precision@12 | Files reported | Median session |
|---|---|---|---|---|---|
| Stock Muse Code, no Scout | 0.413 | 0.600 | 0.381 | 15.0 | 65 s |
| Scout with wayfinder-glimmer | 0.396 | 0.525 | 0.384 | 10.5 | 189 s |
| Scout with wayfinder-qwen | 0.358 | 0.496 | 0.365 | 10.9 | 118 s |

wayfinder-qwen is below stock Muse Code on recall@12: -0.104 [-0.163, -0.051] (the repository-cluster interval also excludes zero). Part of that gap is report length (10.9 files against 15.0), and precision@12 is level. Against wayfinder-glimmer the difference is -0.029 [-0.090, +0.025], which does not separate from zero at this size. The median `explore` call took 44.8 s with wayfinder-qwen and 126.2 s with wayfinder-glimmer, under different serving setups (FP8 with MTP against bf16 without it), so that ratio is not like for like. A Scout search with wayfinder-qwen is still about 1.8 times slower than stock Muse Code on these tasks.

## Limitations

- Retrieval is below wayfinder-glimmer: -0.036 [-0.055, -0.018] reported recall on the grep arm of the 200-task held-out set, a smaller but borderline gap with semantic search (-0.018 [-0.037, -0.000]), a lower all-gold rate, and it ends by calling `report` slightly less often (0.96 against 1.00). It is also below frontier models in the same loop (see the wayfinder-glimmer card).
- In Muse Code, recall@12 is 0.496 against 0.600 for stock Muse Code without Scout (26 tasks, one seed), and the Scout path is slower than stock end to end (118 s against 65 s median).
- Whether thinking was on in every `explore` call of the Muse Code run was not verified from a trace; the run left thinking at the Qwen template default instead of setting it per request as the held-out harness does. Treat that part of the Muse Code result as unconfirmed.
- The held-out numbers were produced with MTP off, and the speed numbers and the end-to-end and Muse Code results with MTP on. Quality with MTP on has not been measured on the held-out set.
- The speed figures come from 12 tasks. The wayfinder-glimmer comparison in the Muse Code run mixes model and serving differences.
- The semantic index adds no separable recall: +0.012 over the grep arm on the held-out set, not distinguishable from zero.
- All evaluations used FP8 online quantization. Other precisions and engines were not tested.
- Only the shared expert and attention projections were trained. Routed-expert LoRA, a second seed and other recipes were not tried, and the gap to wayfinder-glimmer has no tested explanation.
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
- Sibling model: https://huggingface.co/hweller1/wayfinder-glimmer-30b
- Voyage AI: https://www.voyageai.com
- Base model: Qwen3.6-35B-A3B, Alibaba Cloud. Teacher: Muse Spark Open 816B-A42B, Meta Platforms, Inc.
