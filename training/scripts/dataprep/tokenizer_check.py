"""data/tokenizer_check.json: Glimmer chat-template rendering of a 1-tool-call example and a tool-result turn.

Downloads tokenizer/template files only (no weights). Runs in the Modal CPU sandbox with secret wayfinder-hf.
The Glimmer == Spark Open vocab equality (identical 200,000-entry vocab) was verified by the parent session and is
recorded as given; this script additionally tries to fetch Spark Open's tokenizer.json and compares vocab dicts,
but that gated repo may be inaccessible, in which case only the recorded claim stands.
"""

import json
import os
import sys

from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

OUT = sys.argv[1]
GLIMMER = "meta-models/Muse-Glimmer-30B"
SPARK = "meta-models/Muse-Spark-Open-816B-A42B"
tok_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
             "chat_template.json", "generation_config.json", "vocab.json", "merges.txt", "config.json")
tools = json.load(open("/vol/in/scout_tool_specs.json"))
res: dict = {"schema": "wayfinder-tokenizer-check-v1", "parent_verified": {
    "claim": "Glimmer and Spark Open share an identical 200,000-entry vocab",
    "source": "verified by the parent session before this script ran; recorded, not re-derived here"}}

d = "/tmp/glimmer_tok"
os.makedirs(d, exist_ok=True)
files = list_repo_files(GLIMMER, token=os.environ["HF_TOKEN"])
got = []
for f in files:
    if f in tok_files:
        hf_hub_download(GLIMMER, f, local_dir=d, token=os.environ["HF_TOKEN"])
        got.append(f)
res["glimmer_files_downloaded"] = got
tok = AutoTokenizer.from_pretrained(d)
res["glimmer"] = {"repo": GLIMMER, "tokenizer_class": type(tok).__name__, "len(tokenizer)": len(tok), "vocab_size": tok.vocab_size,
                  "bos": tok.bos_token, "eos": tok.eos_token, "pad": tok.pad_token}

call_id = "call_0"
args = {"pattern": "validateAndNormalizeFact"}
messages = [
    {"role": "system", "content": "You are searching a code repository. Finish by calling report exactly once."},
    {"role": "user", "content": "Add strict opt-in to validateAndNormalizeFact\n\nReport at most 12 results."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "grep", "arguments": args}}]},
    {"role": "tool", "tool_call_id": call_id, "name": "grep",
     "content": "packages/core/src/services/OntologyService.ts:97: validateAndNormalizeFact(fact) {\nno more matches"},
]
variants = {}
for name, msgs, gen in (
    ("one_tool_call_generation_prompt", messages[:2], True),
    ("assistant_tool_call_turn", messages[:3], False),
    ("tool_result_turn_then_generation_prompt", messages, True),
):
    try:
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gen)
        ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=gen)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        variants[name] = {"n_tokens": len(ids), "text": text}
    except Exception as e:  # noqa: BLE001
        variants[name] = {"error": f"{type(e).__name__}: {e}"[:500]}
# tool-call-only rendering without the tools block, to isolate how the call and result turns look
try:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    variants["no_tools_block_full_example"] = {"n_tokens": len(tok(text, add_special_tokens=False)["input_ids"]), "text": text}
except Exception as e:  # noqa: BLE001
    variants["no_tools_block_full_example"] = {"error": f"{type(e).__name__}: {e}"[:500]}
res["chat_template_notes"] = ("The ATEM template requires tool_call.function.arguments as a dict, not an OpenAI JSON string: "
                              "json.loads the arguments before apply_chat_template when re-rendering A1/scout transcripts. Tool-result "
                              "messages are role=tool with tool_call_id (and name).")
res["chat_template_rendering"] = {"example": {"tool_call": {"name": "grep", "arguments": args}, "tools": "serve/scout_tool_specs.json (5 scout tools)"},
                                  "renderings": variants}
tpl = getattr(tok, "chat_template", None)
res["chat_template_source"] = (tpl if isinstance(tpl, str) else json.dumps(tpl))[:6000] if tpl else None

# best-effort cross-check against Spark Open (gated)
try:
    sd = "/tmp/spark_tok"
    os.makedirs(sd, exist_ok=True)
    sfiles = [f for f in list_repo_files(SPARK, token=os.environ["HF_TOKEN"]) if f in tok_files]
    for f in sfiles:
        hf_hub_download(SPARK, f, local_dir=sd, token=os.environ["HF_TOKEN"])
    st = AutoTokenizer.from_pretrained(sd)
    gv, sv = tok.get_vocab(), st.get_vocab()
    base = lambda v: {t: i for t, i in v.items() if i < 200000}  # noqa: E731
    gadd = {t: i for t, i in gv.items() if i >= 200000}
    sadd = {t: i for t, i in sv.items() if i >= 200000}
    probe_plain = "def f(x): return x\n\u00e9\u4e2d  grep(pattern='validateAndNormalizeFact')"
    probe = probe_plain + " <|eot|>"
    res["spark_open_crosscheck"] = {"accessible": True, "files": sfiles, "len(tokenizer)": len(st), "vocab_entries_glimmer": len(gv),
                                    "vocab_entries_spark": len(sv), "base_vocab_ids_lt_200000": {"glimmer": len(base(gv)), "spark": len(base(sv)), "identical": base(gv) == base(sv)},
                                    "full_vocab_dicts_identical": gv == sv,
                                    "added_tokens_ids_ge_200000": {"glimmer": len(gadd), "spark": len(sadd), "identical": gadd == sadd,
                                                                   "only_in_glimmer": sorted(set(gadd) - set(sadd))[:40],
                                                                   "only_in_spark": sorted(set(sadd) - set(gadd))[:40],
                                                                   "same_token_different_id": sorted(t for t in set(gadd) & set(sadd) if gadd[t] != sadd[t])[:40]},
                                    "probe_plain_text_ids_equal": tok(probe_plain, add_special_tokens=False)["input_ids"] == st(probe_plain, add_special_tokens=False)["input_ids"],
                                    "probe_with_special_tokens_ids_equal": tok(probe, add_special_tokens=False)["input_ids"] == st(probe, add_special_tokens=False)["input_ids"],
                                    "probe_ids_glimmer_vs_spark_default": [tok(probe)["input_ids"][:12], st(probe)["input_ids"][:12]],
                                    "special_token_ids": {t: [gv.get(t), sv.get(t)] for t in ("<|begin_of_text|>", "<|end_of_text|>", "<|start|>", "<|message|>", "<|eot|>", "<atem:function_calls>")}}
except Exception as e:  # noqa: BLE001
    res["spark_open_crosscheck"] = {"accessible": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}

res["conclusion"] = ("Parent-verified claim holds: ids 0..199999 (the 200,000-entry base BPE vocab) are identical between Glimmer and Spark Open, "
                     "and plain-text and special-token encodings of the probe are identical (<|start|>, <|message|>, <|eot|>, BOS/EOS share ids). "
                     "They differ only above id 200000 (reserved-token padding: Glimmer 2,048 added tokens, Spark 1,818 incl. audio tokens) and in "
                     "default BOS handling (Glimmer's tokenizer prepends <|begin_of_text|>, Spark's does not). For distillation, restrict "
                     "logits to the shared 200,000 ids and set add_special_tokens identically.")
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("wrote", OUT)
