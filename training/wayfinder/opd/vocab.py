"""Shared-vocabulary bookkeeping for Glimmer (student) and Spark Open (teacher).

Ids 0..199,999 are the identical base BPE vocab. Above that each model has its own added tokens (Glimmer 2,048,
Spark 1,818). A handful of structural specials sit at the same id with the same string in both (BOS/EOS, `<|start|>`,
`<|message|>`, `<|eom|>`, `<|eot|>`, pad, `<|patch|>` ...). The loss may use exactly those and the base ids; every other
added id is masked (it means a different token, or nothing, on the other side).

`data/opd_shared_vocab.json` is produced once by `python -m wayfinder.opd.vocab --glimmer DIR --spark DIR` from the two
tokenizer directories and read at train time by `load_allowed`; no tokenizer is needed in the trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_VOCAB = 200_000
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "opd_shared_vocab.json"


def shared_added_ids(glimmer_vocab: dict[str, int], spark_vocab: dict[str, int]) -> list[int]:
    """Added-token ids (>= BASE_VOCAB) that map to the same string in both tokenizers."""
    gi = {i: t for t, i in glimmer_vocab.items() if i >= BASE_VOCAB}
    si = {i: t for t, i in spark_vocab.items() if i >= BASE_VOCAB}
    return sorted(i for i, t in gi.items() if si.get(i) == t)


def load_allowed(path: str | Path = DEFAULT_PATH) -> dict:
    """{'base_vocab': 200000, 'shared_added_ids': [...], 'glimmer_vocab_size': 202048}."""
    return json.loads(Path(path).read_text())


def allowed_mask(vocab_size: int, spec: dict):  # -> torch.BoolTensor[vocab_size]
    import torch

    m = torch.zeros(vocab_size, dtype=torch.bool)
    m[: spec["base_vocab"]] = True
    for i in spec["shared_added_ids"]:
        m[i] = True
    return m


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--glimmer", required=True)
    ap.add_argument("--spark", required=True)
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    a = ap.parse_args()
    g = AutoTokenizer.from_pretrained(a.glimmer)
    s = AutoTokenizer.from_pretrained(a.spark)
    gv, sv = g.get_vocab(), s.get_vocab()
    base_mismatch = [i for i in range(BASE_VOCAB) if g.convert_ids_to_tokens(i) != s.convert_ids_to_tokens(i)]
    assert not base_mismatch, f"base vocab differs at {len(base_mismatch)} ids"
    ids = shared_added_ids(gv, sv)
    structural = {n: g.convert_tokens_to_ids(t) for n, t in (("start", "<|start|>"), ("message", "<|message|>"), ("eom", "<|eom|>"), ("eot", "<|eot|>"), ("bos", "<|begin_of_text|>"))}
    structural["assistant"] = g("<|start|>assistant to=grep", add_special_tokens=False)["input_ids"][1]  # the bare `assistant` token after <|start|>
    structural["user"] = g("<|start|>user", add_special_tokens=False)["input_ids"][1]
    assert all(structural[k] == s.convert_tokens_to_ids(t) for k, t in (("start", "<|start|>"), ("eom", "<|eom|>"), ("eot", "<|eot|>"), ("bos", "<|begin_of_text|>")))
    spec = {
        "schema": "wayfinder-opd-shared-vocab-v1", "base_vocab": BASE_VOCAB, "glimmer_vocab_size": len(g), "spark_vocab_size": len(s),
        "structural_ids": structural, "shared_added_ids": ids, "shared_added_tokens": [g.convert_ids_to_tokens(i) for i in ids],
        "n_glimmer_added_masked": len(g) - BASE_VOCAB - len(ids),
        "note": "ids < base_vocab and shared_added_ids are usable by the loss; every other id is masked",
    }
    Path(a.out).write_text(json.dumps(spec, indent=1))
    print(json.dumps(spec)[:600])


if __name__ == "__main__":
    main()
