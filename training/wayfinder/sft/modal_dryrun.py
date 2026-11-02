"""SFT dry run on Modal (ephemeral; never `modal deploy`). App name prefix `wayfinder-sftdry`.

    modal run wayfinder/sft/modal_dryrun.py --stage probe_h200      # 1x H200: max seq len, step time (random tokens)
    modal run wayfinder/sft/modal_dryrun.py --stage probe_h100      # 1x H100
    modal run wayfinder/sft/modal_dryrun.py --stage probe_h100x2    # 2x H100, layers split across the pair
    modal run wayfinder/sft/modal_dryrun.py --stage dry_h200        # ~50 steps on 200 real examples, 1x H200
    modal run wayfinder/sft/modal_dryrun.py --stage scale_h200x4    # 4 data-parallel replicas on 4x H200 (allreduce path)

Every stage is one function with timeout <= 60 min; start ops/sftdry_watch.py in the same shell step.
Weights come from Volume wayfinder-weights, data and adapters from wayfinder-data (/sft/...).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "wayfinder-sftdry"
HERE = Path(__file__).resolve().parent
DATA_GLOBS = ["/data/sft/dev_sample_long.jsonl"]  # 200 real examples, >=14k tokens (mean ~19k): the realistic length

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("wayfinder-weights")
data = modal.Volume.from_name("wayfinder-data")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.9.1", "transformers==5.21.0", "peft", "accelerate", "kernels>=0.17.2,<0.18", "huggingface_hub", "numpy", "einops")
    .run_commands("pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "train.py", "/srv/train.py")
)


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    print(f"[exit {rc} after {time.time() - t0:.0f}s]", flush=True)
    return rc


def _have_data() -> list[str]:
    import glob

    return [f for g in DATA_GLOBS for f in sorted(glob.glob(g))]


def _body(stage: str, train_args: list[str], gpus: int, replicas: int, extra: str = "") -> dict:
    t0 = time.time()
    out = f"/data/sft/dryrun/{stage}"
    Path(out).mkdir(parents=True, exist_ok=True)
    base = ["/srv/train.py", "--model", "/weights/Muse-Glimmer-30B", "--out", out, "--log-file", f"{out}/{stage}.json"]
    gpr = gpus // replicas
    base += ["--gpus-per-replica", str(gpr)]
    if replicas > 1:
        cmd = ["torchrun", "--standalone", f"--nproc-per-node={replicas}", *base, *train_args, *extra.split()]
    else:
        cmd = ["python", *base, *train_args, *extra.split()]
    rc = _run(cmd)
    data.commit()
    return {"stage": stage, "rc": rc, "gpu_seconds": round((time.time() - t0) * gpus, 1), "wall_s": round(time.time() - t0, 1)}


PROBE_LENS = "2048,4096,8192,16384,24576,32768"


def _data_args() -> list[str]:
    files = _have_data()
    if not files:
        raise SystemExit("no real data on wayfinder-data (/sft/a1_converted/train-*.jsonl or /sft/dev_sample.jsonl)")
    return ["--data", *files]


@app.function(image=image, gpu="H200", volumes={"/weights": weights, "/data": data}, timeout=40 * 60)
def probe_h200(extra: str = "") -> dict:
    return _body("probe_h200", ["--data", "/dev/null", "--probe-lens", PROBE_LENS, "--verify"], 1, 1, extra)


@app.function(image=image, gpu="H100", volumes={"/weights": weights, "/data": data}, timeout=40 * 60)
def probe_h100(extra: str = "") -> dict:
    return _body("probe_h100", ["--data", "/dev/null", "--probe-lens", "1024,2048,4096,8192,16384,32768"], 1, 1, extra)


@app.function(image=image, gpu="H100:2", volumes={"/weights": weights, "/data": data}, timeout=40 * 60)
def probe_h100x2(extra: str = "") -> dict:
    return _body("probe_h100x2", ["--data", "/dev/null", "--probe-lens", PROBE_LENS], 2, 1, extra)


@app.function(image=image, gpu="H200", volumes={"/weights": weights, "/data": data}, timeout=60 * 60)
def dry_h200(pack: bool = True, steps: int = 50, extra: str = "") -> dict:
    args = [*_data_args(), "--limit-examples", "200", "--steps", str(steps), "--packs-per-replica-per-step", "2",
            "--lr", "1e-5", "--verify", "--save"]
    if not pack:
        args.append("--no-pack")
    return _body("dry_h200", args, 1, 1, extra)


@app.function(image=image, gpu="H200:4", volumes={"/weights": weights, "/data": data}, timeout=45 * 60)
def scale_h200x4(steps: int = 6, extra: str = "") -> dict:
    args = [*_data_args(), "--limit-examples", "200", "--steps", str(steps), "--packs-per-replica-per-step", "1",
            "--lr", "1e-5"]
    return _body("scale_h200x4", args, 4, 4, extra)


@app.function(image=image, gpu="H200:2", volumes={"/weights": weights, "/data": data}, timeout=30 * 60)
def dp_check_h200x2(extra: str = "") -> dict:
    """Data-parallel path check on random tokens (NOT a measured run): 2 replicas, a few steps, allreduce + save."""
    args = ["--data", "/dev/null", "--synthetic-len", "8192", "--steps", "4", "--packs-per-replica-per-step", "1", "--lr", "1e-5", "--warmup-steps", "1"]
    return _body("dp_check_h200x2", args, 2, 2, extra)


STAGES = {"dp_check_h200x2": dp_check_h200x2, "probe_h200": probe_h200, "probe_h100": probe_h100, "probe_h100x2": probe_h100x2, "dry_h200": dry_h200, "scale_h200x4": scale_h200x4}


@app.local_entrypoint()
def main(stage: str, extra: str = "", steps: int = 50, nopack: bool = False) -> None:
    fn = STAGES[stage]
    kw: dict = {}
    if stage == "dry_h200":
        kw = {"pack": not nopack, "steps": steps, "extra": extra}
    if stage == "scale_h200x4":
        kw = {"steps": steps, "extra": extra}
    if stage.startswith("probe") or stage.startswith("dp_check"):
        kw = {"extra": extra}
    res = fn.remote(**kw)
    print("RESULT", json.dumps(res))
