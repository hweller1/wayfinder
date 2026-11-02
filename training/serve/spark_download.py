"""Stage 1: download meta-models/Muse-Spark-Open-816B-A42B-NVFP4 into Modal Volume `wayfinder-spark` (CPU only).

    modal run serve/spark_download.py            # list, shard, download in parallel, verify
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

MODEL_ID = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
DEST = "/spark/Muse-Spark-Open-816B-A42B-NVFP4"
N_SHARDS = 12
HERE = Path(__file__).resolve().parent

app = modal.App("wayfinder-spark-download")
vol = modal.Volume.from_name("wayfinder-spark", create_if_missing=True)
secret = modal.Secret.from_name("wayfinder-hf")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
)


@app.function(image=image, secrets=[secret], timeout=300)
def listing() -> list[tuple[str, int]]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_ID, files_metadata=True)
    return sorted((s.rfilename, s.size or 0) for s in info.siblings)


@app.function(image=image, secrets=[secret], volumes={"/spark": vol}, timeout=55 * 60, cpu=8, memory=16384)
def fetch(files: list[str]) -> dict:
    import os

    from huggingface_hub import hf_hub_download

    t0 = time.time()
    done = 0
    for f in files:
        hf_hub_download(MODEL_ID, f, local_dir=DEST, token=os.environ["HF_TOKEN"])
        done += 1
    vol.commit()
    return {"files": done, "seconds": round(time.time() - t0, 1)}


@app.function(image=image, volumes={"/spark": vol}, timeout=600)
def verify(expected: dict[str, int]) -> dict:
    vol.reload()
    root = Path(DEST)
    have = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts}
    missing = sorted(set(expected) - set(have))
    wrong = sorted(k for k in expected if k in have and have[k] != expected[k])
    return {"expected": len(expected), "have": len(have), "missing": missing, "size_mismatch": wrong, "bytes": sum(have.values())}


@app.local_entrypoint()
def main() -> None:
    t0 = time.time()
    lst = listing.remote()
    print(f"{len(lst)} files, {sum(s for _, s in lst)/1e9:.1f} GB", flush=True)
    for n, s in lst:
        if s > 2e9 or not n.endswith(".safetensors"):
            pass
    # greedy balance by size
    bins: list[list[str]] = [[] for _ in range(N_SHARDS)]
    loads = [0] * N_SHARDS
    for n, s in sorted(lst, key=lambda x: -x[1]):
        i = loads.index(min(loads))
        bins[i].append(n)
        loads[i] += s
    res = list(fetch.map([b for b in bins if b]))
    print(res, flush=True)
    v = verify.remote(dict(lst))
    v["download_wall_s"] = round(time.time() - t0, 1)
    v["shard_results"] = res
    print(json.dumps({k: v[k] for k in v if k != "shard_results"}, indent=1), flush=True)
    (HERE.parent / "artifacts" / "serve" / "spark_download.json").write_text(json.dumps(v, indent=2))
