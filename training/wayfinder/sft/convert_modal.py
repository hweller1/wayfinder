"""Modal CPU fan-out for the A1-35B transcript conversion (ephemeral app, no deploy).

    modal run wayfinder/sft/convert_modal.py                # all 11 tarballs x 4 shards, writes /sft/a1_converted
    modal run wayfinder/sft/convert_modal.py --limit 2      # smoke: first 2 shards only, writes /sft/a1_converted_smoke

Reads wayfinder-data:/transcripts/a1_35b/{modal,a100}/*.tgz, the Glimmer tokenizer from wayfinder-weights, and
writes per-shard JSONL (+ per-shard drop log) back to wayfinder-data. The local entrypoint downloads the summaries,
writes artifacts/sft/a1_convert_summary.json and prints the counts. CPU only; cost is a few dollars at most.
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the container the module sits at /root
APP_NAME = "wayfinder-sftconv"
SHARDS_PER_TARBALL = 4
TOKENIZER_DIR = "/weights/Muse-Glimmer-30B"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("inspect-ai", "zstandard", "transformers==5.21.0", "tokenizers", "jinja2", "huggingface_hub")
    .env({"PYTHONPATH": "/root", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_file(REPO / "serve" / "scout_product_prompt.md", "/srv/scout_product_prompt.md", copy=False)
    .add_local_file(REPO / "serve" / "scout_tool_specs.json", "/srv/scout_tool_specs.json", copy=False)
    .add_local_file(REPO / "data" / "splits.json", "/srv/splits.json", copy=False)
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)
app = modal.App(APP_NAME, image=image)
data = modal.Volume.from_name("wayfinder-data")
weights = modal.Volume.from_name("wayfinder-weights")

TARBALLS = [f"modal/pub_s{i}_runlogs.tgz" for i in range(8)] + [
    f"a100/{a}_sophisticated_agents-a1-35b.tgz" for a in ("b2", "c", "h")
]


@app.function(volumes={"/data": data, "/weights": weights}, timeout=40 * 60, cpu=2, memory=8192, max_containers=48)
def convert_shard(tarball: str, shard: int, nshards: int, out_dir: str) -> dict:
    from collections import Counter

    from transformers import AutoTokenizer

    from wayfinder.sft.convert import convert_eval_file

    t0 = time.time()
    splits = json.load(open("/srv/splits.json"))
    system_prompt = Path("/srv/scout_product_prompt.md").read_bytes().decode("utf-8")
    tools = json.load(open("/srv/scout_tool_specs.json"))
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    name = tarball.replace("/", "_").removesuffix(".tgz")
    out = Path("/data") / out_dir.strip("/")
    out.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp()
    n_eval = 0
    ex_n = 0
    drops: Counter = Counter()
    by_arm_tier: Counter = Counter()
    out_lossy = Path("/data") / (out_dir.strip("/") + "_lossy")
    out_lossy.mkdir(parents=True, exist_ok=True)
    with tarfile.open(f"/data/transcripts/a1_35b/{tarball}", "r:gz") as tf, \
            open(out / f"train-{name}-{shard:02d}.jsonl", "w") as fo, \
            open(out_lossy / f"train-{name}-{shard:02d}.jsonl", "w") as fl, \
            open(out / f"drops-{name}-{shard:02d}.jsonl", "w") as fd:
        idx = 0
        for m in tf:
            if not (m.isfile() and m.name.endswith(".eval")):
                continue
            idx += 1
            if idx % nshards != shard:
                continue
            n_eval += 1
            p = os.path.join(tmp, f"{n_eval}.eval")
            with tf.extractfile(m) as src, open(p, "wb") as dst:  # type: ignore[union-attr]
                dst.write(src.read())
            try:
                exs, ds = convert_eval_file(p, splits, system_prompt, tools, tok)
            except Exception as e:  # noqa: BLE001
                exs, ds = [], [{"file": m.name, "reason": f"exception:{type(e).__name__}:{e}"[:200]}]
            os.unlink(p)
            for e in exs:
                e["meta"]["source_tarball"] = tarball
                (fo if e["meta"]["tier"] == "strict" else fl).write(json.dumps(e) + "\n")
                ex_n += 1
                by_arm_tier[(e["meta"]["arm"], e["meta"]["tier"], e["meta"]["all_gold_found"])] += 1
            for d in ds:
                d["source_tarball"] = tarball
                fd.write(json.dumps(d) + "\n")
                drops[d["reason"]] += 1
    data.commit()
    return {"tarball": tarball, "shard": shard, "eval_files": n_eval, "examples": ex_n, "drops": dict(drops),
            "by_arm_tier_allgold": {"|".join(map(str, k)): v for k, v in by_arm_tier.items()},
            "secs": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(limit: int = 0, out_dir: str = "") -> None:
    jobs = [(t, s) for t in TARBALLS for s in range(SHARDS_PER_TARBALL)]
    if limit:
        jobs = jobs[:limit]
    out_dir = out_dir or ("sft/a1_converted_smoke" if limit else "sft/a1_converted")
    t0 = time.time()
    results = list(convert_shard.starmap([(t, s, SHARDS_PER_TARBALL, out_dir) for t, s in jobs]))
    summ_dir = REPO / "artifacts" / "sft"
    summ_dir.mkdir(parents=True, exist_ok=True)
    (summ_dir / f"{'smoke_' if limit else ''}a1_convert_shards.json").write_text(json.dumps(results, indent=1))
    tot_files = sum(r["eval_files"] for r in results)
    tot_ex = sum(r["examples"] for r in results)
    print(f"eval_files={tot_files} examples={tot_ex} wall={time.time() - t0:.0f}s cpu_s={sum(r['secs'] for r in results):.0f}")
