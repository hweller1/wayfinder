"""SFT v1 launcher: LoRA r=64 on Muse-Glimmer-30B over the converted train split, 1 epoch, 8 GPUs. NOT launched by its author.

Config from the dry run (docs/sft_dryrun.md): one replica per GPU on H200 (141 GB; 96 GiB peak at a 32k packed row),
8 data-parallel replicas, flash_attention_2 with packed rows (position_ids restart per document, so no cross-document
attention, sliding-window layers included), gradient checkpointing, assistant-only loss with a chunked lm_head. On H100
(80 GB) one replica does not fit even 1k tokens; use `--gpu h100` which runs 4 replicas x 2 GPUs (layers split).

    # watcher first, in the same shell step (the default 65 min kill age is too short for an epoch):
    SFTDRY_MAX_AGE_MIN=110 nohup python3 ops/sftdry_watch.py > artifacts/ops/sftdry_watch.out 2>&1 &
    modal run wayfinder/sft/train_v1.py --gpu h200 --data "/data/sft/a1_converted/train-*.jsonl" --out /data/sft/v1


Adapters (and a checkpoint every --save-every steps) land in wayfinder-data:/sft/v1/. Effective batch = 8 replicas x
`--packs` packed rows of <= 32,768 tokens (default 2 -> up to 524k tokens/step, about 115 steps for 60M tokens).
Learning rate defaults to 1e-4, NOT the 1e-5 of the PRD: 1e-5 is a full-fine-tune value and is low for LoRA (the dry run at
1e-5 did move the loss, 0.59 -> 0.49 in 50 steps, but a ~100-step epoch wants a larger step). Pass --lr 1e-5 to follow the PRD.
Function timeout is 90 min: 60M tokens would take ~65-70 min on 8x H200; the 1,237 converted examples (~25M tokens) ~30 min.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App(os.environ.get("WF_TRAIN_APP", "wayfinder-sftdry-v1"))  # prefix wayfinder-sftdry so ops/sftdry_watch.py covers it; v2 sets WF_TRAIN_APP=wayfinder-sftv2-train-<variant>
weights = modal.Volume.from_name("wayfinder-weights")
data_vol = modal.Volume.from_name("wayfinder-data")
image = (  # identical to modal_dryrun.image (kept self-contained: modal does not ship sibling modules)
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.9.1", "transformers==5.21.0", "peft", "accelerate", "kernels>=0.17.2,<0.18", "huggingface_hub", "numpy", "einops")
    .run_commands("pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "train.py", "/srv/train.py")
)


def _body(out: str, train_args: list[str], gpus: int, replicas: int, extra: str) -> dict:
    t0 = time.time()
    Path(out).mkdir(parents=True, exist_ok=True)
    base = ["/srv/train.py", "--model", "/weights/Muse-Glimmer-30B", "--out", out, "--log-file", f"{out}/train_summary.json",
            "--gpus-per-replica", str(gpus // replicas)]
    cmd = ["torchrun", "--standalone", f"--nproc-per-node={replicas}", *base, *train_args, *extra.split()]
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    data_vol.commit()
    return {"rc": rc, "out": out, "gpu_seconds": round((time.time() - t0) * gpus, 1), "wall_s": round(time.time() - t0, 1)}


def _run(gpus: int, replicas: int, data: str, out: str, lr: float, packs: int, steps: int, save_every: int, extra: str) -> dict:
    import glob

    files = sorted(glob.glob(data))
    if not files:
        raise SystemExit(f"no files match {data}")
    args = ["--data", *files, "--steps", str(steps), "--packs-per-replica-per-step", str(packs), "--lr", str(lr),
            "--warmup-steps", "5", "--attn", "flash_attention_2", "--save", "--save-every", str(save_every)]
    return _body(out, args, gpus, replicas, extra)


_VOLS = {"/weights": weights, "/data": data_vol}


@app.function(image=image, gpu="H200:8", volumes=_VOLS, timeout=90 * 60)
def run_h200(data: str, out: str, lr: float, packs: int, steps: int, save_every: int, extra: str) -> dict:
    return _run(8, 8, data, out, lr, packs, steps, save_every, extra)  # 8 replicas x 1 GPU


@app.function(image=image, gpu="H100:8", volumes=_VOLS, timeout=90 * 60)
def run_h100(data: str, out: str, lr: float, packs: int, steps: int, save_every: int, extra: str) -> dict:
    return _run(8, 4, data, out, lr, packs, steps, save_every, extra)  # 4 replicas x 2 GPUs (layers split)


@app.function(image=image, gpu="H200:4", volumes=_VOLS, timeout=int(os.environ.get("WF_TRAIN_TIMEOUT_H", "3")) * 3600)  # v3 (1,383 examples, ~2x v2) sets 5
def run_h200x4(data: str, out: str, lr: float, packs: int, steps: int, save_every: int, extra: str) -> dict:
    return _run(4, 4, data, out, lr, packs, steps, save_every, extra)  # 4 replicas x 1 GPU (measured in the dry run: 7.2k tok/s)


@app.function(image=image, gpu="H200:8", volumes=_VOLS, timeout=int(os.environ.get("WF_TRAIN_TIMEOUT_H", "5")) * 3600)  # v4: long-timeout 8-GPU function (run_h200 keeps its 90 min dry-run limit)
def run_h200x8_long(data: str, out: str, lr: float, packs: int, steps: int, save_every: int, extra: str) -> dict:
    return _run(8, 8, data, out, lr, packs, steps, save_every, extra)  # 8 replicas x 1 GPU


@app.local_entrypoint()
def main(gpu: str = "h200", data: str = "/data/sft/a1_converted/train-*.jsonl", out: str = "/data/sft/v1",
         lr: float = 1e-4, packs: int = 2, steps: int = 0, save_every: int = 25, extra: str = "") -> None:
    fn = {"h200": run_h200, "h100": run_h100, "h200x4": run_h200x4, "h200x8": run_h200x8_long}[gpu]
    res = fn.remote(data, out, lr, packs, steps, save_every, extra)
    print("RESULT", json.dumps(res))
