"""Merge an OPD LoRA adapter into the checkpoint it was trained on (CPU Modal job).

    modal run wayfinder/opd/merge_app.py --base /data/sft/v2a_merged --adapter /data/opd/run1/adapter --dst /data/opd/run1/merged
    modal run wayfinder/opd/merge_app.py --base /data/sft/v1_merged --adapter /data/opd/smoke/adapter --dry-run   # name check only, writes nothing

W' = W + (alpha/r) * B @ A in fp32, cast to bf16 (same arithmetic as wayfinder/rollout/sft_merge_app.py, but the base is any
checkpoint, because OPD adapts the SFT-merged model, not the raw base). The result serves with vLLM like any merged model
(wayfinder/rollout/sft_eval_app.py takes WF_EVAL_VARIANT / weights dir; wayfinder/opd/rollout_app.py takes --student).
App prefix `wayfinder-opd` (ops/opd_watch.py).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import modal

app = modal.App("wayfinder-opd-merge")
weights = modal.Volume.from_name("wayfinder-weights")
data_vol = modal.Volume.from_name("wayfinder-data")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "safetensors", "numpy")


@app.function(image=image, volumes={"/weights": weights, "/data": data_vol}, cpu=8, memory=131072, timeout=40 * 60)
def merge(base: str, adapter: str, dst: str, dry_run: bool) -> dict:
    import torch
    from safetensors.torch import load_file, save_file

    data_vol.reload()
    src, adp = Path(base), Path(adapter)
    cfg = json.loads((adp / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(adp / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        assert k.startswith("base_model.model."), k
        mod, ab = k[len("base_model.model."):].rsplit(".lora_", 1)
        pairs.setdefault(mod + ".weight", {})[ab[0]] = v  # 'A' / 'B'
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [n for n in pairs if n not in idx]
    out = {"adapter_modules": len(pairs), "scale": scale, "r": cfg["r"], "missing_in_base": len(missing), "missing_examples": missing[:3]}
    if dry_run or missing:
        return out
    d = Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and not f.name.startswith("."):
            shutil.copy(f, d / f.name)
    merged, max_rel = 0, 0.0
    for shard in sorted(src.glob("model-*.safetensors")):
        t0 = time.time()
        tensors = load_file(str(shard))
        for name in list(tensors):
            if name in pairs:
                w, ab = tensors[name], pairs[name]
                delta = scale * (ab["B"].float() @ ab["A"].float())
                assert delta.shape == w.shape, (name, delta.shape, w.shape)
                tensors[name] = (w.float() + delta).to(w.dtype)
                max_rel = max(max_rel, float(delta.norm() / w.float().norm()))
                merged += 1
        save_file(tensors, str(d / shard.name), metadata={"format": "pt"})
        print(f"{shard.name}: done in {time.time() - t0:.0f}s, merged so far {merged}", flush=True)
        del tensors
    assert merged == len(pairs), (merged, len(pairs))
    data_vol.commit()
    return {**out, "modules_merged": merged, "max_rel_delta_norm": max_rel, "dst": dst}


@app.local_entrypoint()
def main(base: str, adapter: str, dst: str = "", dry_run: bool = False):
    print(json.dumps(merge.remote(base, adapter, dst or adapter.rstrip("/") + "_merged", dry_run)))
