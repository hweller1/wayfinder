"""OPD trainer on Modal: LoRA on Glimmer (base or an SFT-merged checkpoint) against stored Spark top-20 logprobs.

    modal run wayfinder/opd/train_app.py --scored opd/smoke --out opd/smoke/adapter --gpu h200x2 --steps 16 --eval-chains 2
    modal run wayfinder/opd/train_app.py --scored opd/run1 --out opd/run1/adapter --student /data/sft/v2a_merged --gpu h200x4 --steps 40

The image, torch/transformers/peft/flash-attn pins and the model loading are the SFT trainer's (wayfinder/sft/train.py is shipped as
/srv/train.py and imported by wayfinder/opd/train_opd.py). App prefix `wayfinder-opd` (ops/opd_watch.py).
`--student` is the model whose weights are loaded and adapted: /weights/Muse-Glimmer-30B (base, default) or a merged SFT
checkpoint such as /data/sft/v2a_merged. The adapter is trained on top of it; merge with wayfinder/rollout/sft_merge_app.py logic.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import sys

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # local entrypoint imports wayfinder.opd.*
app = modal.App("wayfinder-opd-train")
weights = modal.Volume.from_name("wayfinder-weights")
data_vol = modal.Volume.from_name("wayfinder-data")
image = (  # same stack as wayfinder/sft/train_v1.py
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.9.1", "transformers==5.21.0", "peft", "accelerate", "kernels>=0.17.2,<0.18", "huggingface_hub", "numpy", "einops")
    .run_commands("pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/srv"})
    .add_local_file(REPO / "wayfinder" / "sft" / "train.py", "/srv/train.py")
    .add_local_dir(str(REPO / "wayfinder" / "opd"), "/srv/wayfinder/opd", ignore=["**/__pycache__"])
    .add_local_file(REPO / "data" / "opd_shared_vocab.json", "/srv/opd_shared_vocab.json")
)
_VOLS = {"/weights": weights, "/data": data_vol}


def _body(scored: str, out: str, student: str, gpus: int, replicas: int, extra: str) -> dict:
    t0 = time.time()
    data_vol.reload()
    Path(f"/data/{out}").mkdir(parents=True, exist_ok=True)
    cmd = ["torchrun", "--standalone", f"--nproc-per-node={replicas}", "/srv/wayfinder/opd/train_opd.py", "--model", student,
           "--scored", f"/data/{scored}/scored", "--vocab", "/srv/opd_shared_vocab.json",
           "--out", f"/data/{out}", "--log-file", f"/data/{out}/train_summary.json", "--gpus-per-replica", str(gpus // replicas), *extra.split()]
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    data_vol.commit()
    return {"rc": rc, "out": out, "gpu_seconds": round((time.time() - t0) * gpus, 1), "wall_s": round(time.time() - t0, 1)}


@app.function(image=image, gpu="H200:2", volumes=_VOLS, timeout=2 * 3600)
def run_h200x2(scored: str, out: str, student: str, extra: str) -> dict:
    return _body(scored, out, student, 2, 2, extra)  # 2 replicas x 1 GPU


@app.function(image=image, gpu="H200:4", volumes=_VOLS, timeout=3 * 3600)
def run_h200x4(scored: str, out: str, student: str, extra: str) -> dict:
    return _body(scored, out, student, 4, 4, extra)  # 4 replicas x 1 GPU


@app.function(image=image, gpu="H200:4", volumes=_VOLS, timeout=3 * 3600)
def run_h200x4r2(scored: str, out: str, student: str, extra: str) -> dict:
    return _body(scored, out, student, 4, 2, extra)  # 2 replicas x 2 GPUs (layers split): room for chains > 49k tokens (UNTESTED above 49k)


@app.function(image=image, gpu="H200", volumes=_VOLS, timeout=2 * 3600)
def run_h200x1(scored: str, out: str, student: str, extra: str) -> dict:
    return _body(scored, out, student, 1, 1, extra)


@app.local_entrypoint()
def main(scored: str = "opd/smoke", out: str = "opd/smoke/adapter", student: str = "/weights/Muse-Glimmer-30B", gpu: str = "h200x2",
         steps: int = 16, chains_per_replica: int = 1, lr: float = 5e-5, eval_chains: int = 2, eval_every: int = 0, extra: str = "") -> None:
    fn = {"h200x1": run_h200x1, "h200x2": run_h200x2, "h200x4": run_h200x4, "h200x4r2": run_h200x4r2}[gpu]
    ex = f"--steps {steps} --chains-per-replica {chains_per_replica} --lr {lr} --eval-chains {eval_chains} --eval-every {eval_every} {extra}"
    res = fn.remote(scored, out, student, ex)
    print("RESULT", json.dumps(res))
    d = REPO / "artifacts" / "opd"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"train_{Path(out).parent.name}.json").write_text(json.dumps(res, indent=1))
