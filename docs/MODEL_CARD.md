---
license: other
license_name: apache-2.0-and-muse-model-license
license_link: LICENSE-MUSE-SPARK-OPEN.md
library_name: transformers
pipeline_tag: text-generation
language:
  - en
base_model:
  - meta-models/Muse-Glimmer-30B
tags:
  - code-search
  - agentic
  - tool-use
  - subagent
  - lora-merged
  - muse
---

# wayfinder-glimmer

**Built with Muse Spark Open.**

wayfinder-glimmer is Muse Glimmer 30B post-trained (LoRA supervised fine-tuning, merged into the base weights) to work as a code-search subagent. A coding agent hands it a brief such as "where does the router decide that a request matches a route with path variables?". The model plans, searches an unfamiliar repository with `grep`, `glob` and `read_file`, optionally calls `semantic_search` over a voyage-code-4 vector index, and returns a ranked list of `path:start-end` ranges with a one-line reason each, a summary and the gaps it could not establish. The parent agent reads that report instead of searching itself.

The weights are bf16 safetensors with the config, tokenizer and chat template of Muse Glimmer 30B unchanged. This repository is a weights release: no training data, trajectories or task text are included.

## Serving

Tested with vLLM 0.32.0, one H100, bf16. The tool-call and reasoning parsers are required; without `--tool-call-parser muse_glimmer` the tool calls come back as plain text. The checkpoint's own chat template is used (no `--chat-template`).

```bash
pip install "vllm==0.32.0"
vllm serve hweller1/wayfinder-glimmer-30b \
  --served-model-name wayfinder-glimmer \
  --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.92 \
  --language-model-only --enable-prefix-caching --max-num-seqs 64 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
  --port 8000
```

The model reasons before each tool call and was trained and evaluated with its own earlier reasoning sent back on every request. The chat template renders `reasoning_content` of earlier assistant messages. vLLM 0.32.0 returns the reasoning under the `reasoning` key, so copy it into `reasoning_content` when you append the assistant turn to the conversation. A client that drops earlier reasoning is outside what the model was trained on. The chat template needs tool-call arguments as a dict. Sampling used in the evaluations: temperature 0.85, top_p 0.95.

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
body = {"model": "wayfinder-glimmer", "tools": tools, "tool_choice": "auto",
        "max_tokens": 1024, "temperature": 0.85, "top_p": 0.95,
        "messages": [
            {"role": "system", "content": "You are a code-search subagent. Find the files that answer the brief, then report ranked path:start-end ranges with reasons. Report as soon as the evidence covers the brief; do not exhaust the turn budget."},
            {"role": "user", "content": "Where is the HTTP retry backoff maximum delay configured?"}]}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
msg = json.load(urllib.request.urlopen(req))["choices"][0]["message"]
print([c["function"]["name"] for c in msg.get("tool_calls", [])])   # expect ['grep', ...]
```

The tool schemas and system prompt used in training and evaluation come from the reference harness in the GitHub repository (`serve/scout_tool_specs.json`, `serve/scout_product_prompt.md`). Keep the convergence line "Report as soon as the evidence covers the brief; do not exhaust the turn budget." in the system prompt, because the training examples carry it.

## Intended use

Use it as the explore or search subagent under a stronger parent model, inside a harness that executes its tool calls and offers the five tools (`grep`, `glob`, `read_file`, `semantic_search`, `report`). Its output is a list of leads for the parent to verify.

Out of scope:

- Use outside that tool surface (other tool names, no `report` tool, other prompts). It is untested there.
- General chat, general code generation, or any task where retention of general skills matters. Retention was not measured.
- Use as the only safeguard on code changes.
- Searching repositories you are not permitted to read.
- Any use prohibited by the Muse Spark Open Usage Policy or the Muse Glimmer Usage Policy.

## Training

- Base model: Muse Glimmer 30B.
- Method: LoRA supervised fine-tuning on the language model only, rank 64, alpha 128, all attention and MLP projections, then merged into the base weights in fp32 and saved as bf16. Serving needs no LoRA support.
- Data: 1,383 filtered examples from 344 tasks in public repositories. The tasks come from 153 of the 244 repositories in the training split. Of those 244, 242 are under permissive licenses (MIT 129, Apache-2.0 98, BSD-3-Clause 12, BSD-2-Clause 2, ISC 1), one is AGPL-3.0 and one is no longer on GitHub, so its license could not be checked (GitHub license API, 2026-11-02; list in `eval/results/train_repos_license.csv` in the GitHub repository). The 344 tasks are not listed in this release, so these counts cover the whole training split, and either of those two repositories may be among the 153. 1,350 are Muse Spark Open teacher trajectories (1,347 from self-hosted replicas of the open weights, 3 from a third-party host of the same open weights; none from the Meta API) and 33 are Agents-A1-35B trajectories. Only runs that found every gold file, opened every file they reported and called `report` in time were kept. Only assistant tokens carry loss.
- Schedule: 2 epochs, learning rate 5e-5 with cosine decay, packed rows of up to 49,152 tokens, 4x H200, about 3 hours.
- Single training run (seed 0). Results describe this checkpoint, not the recipe.

The trajectories and task text are not published.

## Evaluation

All numbers use the same explore loop, system prompt, 40-turn cap and two seeds, with earlier reasoning retained. "Reported recall" is the share of gold files in the `report` call (at most 12 files), which is what a parent agent receives. Intervals are 95% bootstrap intervals over tasks. The semantic arm adds `semantic_search` over a voyage-code-4 index.

### Held-out retrieval: 200 tasks, 47 repositories never seen in training

| Model | Reported recall, grep | Reported recall, with semantic search | Turns per task (grep) |
|---|---|---|---|
| wayfinder-glimmer | 0.452 [0.412, 0.491] | 0.446 [0.406, 0.484] | 9.9 |
| Agents-A1-35B | 0.313 [0.280, 0.348] | 0.352 [0.319, 0.387] | 34.1 |
| Sonnet 5.5 (high reasoning effort) | 0.529 [0.491, 0.567] | 0.518 [0.480, 0.556] | 7.5 |
| GPT-6.1 Sol | 0.580 [0.541, 0.619] | 0.576 [0.538, 0.614] | 7.8 |

wayfinder-glimmer is ahead of A1 (+0.139 [+0.111, +0.166] on the grep arm) and behind both frontier models (-0.078 against Sonnet, -0.128 against GPT on the grep arm). The frontier models were run through their own native APIs in the same loop. This set was run once and was not used to choose between models.

### Held-out test split: 152 tasks, 31 repositories

Pre-registered and run once. Reported recall on the grep arm: wayfinder-glimmer 0.519 [0.467, 0.569], Agents-A1-35B 0.439 [0.391, 0.488], untrained Muse Glimmer 30B 0.428 [0.376, 0.480]. Against A1: +0.080 [+0.049, +0.112]. With the semantic index the result is the same in sign (0.521 against 0.456 for A1 and 0.442 for the base).

### Languages with little training data: 120 tasks, 111 repositories

C#, Java/Kotlin and Swift, on repositories created after Agents-A1 was released. Reported recall on the grep arm: wayfinder-glimmer 0.686 [0.637, 0.734], A1 0.613 [0.563, 0.662], untrained base 0.556 [0.505, 0.608]. Against A1: +0.073 [+0.038, +0.107]. Per-language intervals are not powered.

### End to end: pass rate of a coding agent handed the searcher's files

34 tasks from 8 repositories, selected because a coder cannot solve them without finding the code. A fixed coding model has no search tools and can only open the files the subagent returned; held-out tests are absent from its sandbox.

| What the coder is handed | Pass rate |
|---|---|
| No search | 0.07 |
| wayfinder-glimmer | 0.87 |
| Sonnet 5.5 as the searcher | 0.87 |
| Gold files (ceiling) | 0.97 |

Any searcher lifts the pass rate by about 0.8 over no search; the lift survives a repository-cluster bootstrap ([+0.53, +0.89] for wayfinder-glimmer). Differences between searchers are not resolved. Caveat: the set is concentrated, with 16 of the 34 tasks from one repository, so the effective sample is closer to 8 repositories, and the searchers sit near the ceiling. A separate, stricter task set reached only 4 usable tasks, and nothing was resolved there.

## Limitations

- Slower than frontier search in our serving: about 175 s per search on one H100 (single stream, bf16). Frontier models in the same loop had a median grep-arm wall time of 77 s (Sonnet 5.5) and 104 s (GPT-6.1 Sol) per rollout through their APIs.
- The semantic index adds no recall for this model: +0.002 [-0.015, +0.019] on the test split, and no gain on the 200-task set (0.446 with semantic search against 0.452 without).
- Retrieval recall is below frontier models (see the held-out table).
- English-only task wording. No other natural language was tested.
- Evaluated on public repositories only. Performance on private or unusual codebases is unknown.
- This is one training run; Swift, C# and Java/Kotlin are thin in the training data.
- Trained on public code whose individual licenses were not cleared task by task.
- It inherits the biases and failure modes of the base and teacher models. Test it for your use before deployment.

## License and notices

Three sets of terms apply. Keep all license files in this repository with any copy of the weights.

1. **Muse Glimmer 30B.** The base model is released under the Apache License 2.0 (`LICENSE`), with Meta's Muse Glimmer Usage Policy (`USAGE_POLICY.md`).
2. **Muse Spark Open derivative terms.** Because the model was fine-tuned on outputs of Muse Spark Open, it is a "Derivative" under the Muse Model License, version "Muse Spark Open" (`LICENSE-MUSE-SPARK-OPEN.md`, which includes the Usage Policy as Exhibit A; also provided as `USE_POLICY-MUSE-SPARK-OPEN.md`). The license permits publishing and distributing Derivatives. Its conditions: the license and Usage Policy travel with the weights; **"Built with Muse Spark Open"** must be displayed in the user interface or accompanying documentation of any product or service that uses this model; and any party offering this model as a Model as a Service with aggregate gross revenue above US$20 million in a twelve-month period needs a separate written agreement with Meta. Your use must comply with both usage policies.
3. **Agents-A1.** 33 of the 1,383 training examples are trajectories from Agents-A1-35B (InternScience), released under the Apache License 2.0. The training recipe adapts the Agents-A1 work (arXiv 2606.30616).

Meta claims no ownership of outputs generated by the model. The software and outputs are provided as is, without warranty.

## Citation and links

```bibtex
@misc{wayfinder_glimmer_2026,
  title  = {wayfinder-glimmer: a code-search subagent post-trained from Muse Glimmer 30B},
  author = {Weller, Henry},
  year   = {2026},
  note   = {Voyage AI. Built with Muse Spark Open.},
  url    = {https://huggingface.co/hweller1/wayfinder-glimmer-30b}
}
```

- Code, evaluation and serving harness: https://github.com/hweller1/wayfinder
- Voyage AI: https://www.voyageai.com
- Base model: Muse Glimmer 30B, Meta Platforms, Inc. Teacher: Muse Spark Open 816B-A42B, Meta Platforms, Inc.
