"""Merge the SFT v1 LoRA (Volume wayfinder-data:/sft/v1) into the base weights (Volume wayfinder-weights) and write the
merged checkpoint to wayfinder-data:/sft/v1_merged. CPU only. W' = W + (alpha/r) * B @ A in fp32, cast to bf16.

    modal run wayfinder/rollout/sft_merge_app.py                       # v1
    WF_MERGE_VARIANT=v2a modal run wayfinder/rollout/sft_merge_app.py  # /data/sft/v2a -> /data/sft/v2a_merged
Why: runtime vLLM LoRA made rollouts ~10x slower than base (per-turn latency 50-250 s); merged weights serve at base speed.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import modal

VARIANT = os.environ.get("WF_MERGE_VARIANT", "v1")  # v1 keeps the original app name; v2a/v2b use the sftv2 prefix (watcher)
app = modal.App("wayfinder-sft-merge" if VARIANT == "v1" else f"wayfinder-{'sftv4' if VARIANT.startswith('v4') else 'sftv3' if VARIANT == 'v3' else 'sftv2'}-merge-{VARIANT}")  # v3 uses the sftv3 watcher prefix
weights = modal.Volume.from_name("wayfinder-weights")
data_vol = modal.Volume.from_name("wayfinder-data")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "safetensors", "numpy")


@app.function(image=image, volumes={"/weights": weights, "/data": data_vol}, cpu=8, memory=131072, timeout=40 * 60)
def merge(variant: str = "v1") -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    src, adp, dst = Path("/weights/Muse-Glimmer-30B"), Path(f"/data/sft/{variant}"), Path(f"/data/sft/{variant}_merged")
    cfg = json.loads((adp / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(adp / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        assert k.startswith("base_model.model."), k
        name = k[len("base_model.model."):]
        mod, ab = name.rsplit(".lora_", 1)
        pairs.setdefault(mod + ".weight", {})[ab[0]] = v  # 'A' / 'B'
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and not f.name.startswith("."):
            shutil.copy(f, dst / f.name)
    merged, max_rel = 0, 0.0
    for shard in sorted(src.glob("model-*.safetensors")):
        t0 = time.time()
        tensors = load_file(str(shard))
        for name in list(tensors):
            if name in pairs:
                w = tensors[name]
                ab = pairs[name]
                delta = scale * (ab["B"].float() @ ab["A"].float())
                assert delta.shape == w.shape, (name, delta.shape, w.shape)
                new = (w.float() + delta).to(w.dtype)
                max_rel = max(max_rel, float(delta.norm() / w.float().norm()))
                tensors[name] = new
                merged += 1
        save_file(tensors, str(dst / shard.name), metadata={"format": "pt"})
        print(f"{shard.name}: done in {time.time()-t0:.0f}s, merged so far {merged}", flush=True)
        del tensors
    assert merged == len(pairs), (merged, len(pairs))
    data_vol.commit()
    return {"modules_merged": merged, "scale": scale, "max_rel_delta_norm": max_rel}


@app.local_entrypoint()
def main():
    print(merge.remote(VARIANT))
