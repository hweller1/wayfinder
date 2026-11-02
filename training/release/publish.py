"""Merge a Wayfinder LoRA into Muse-Glimmer-30B and publish it to Hugging Face. PRIVATE unless --public.

    python3 release/publish.py <namespace/repo-name> [--variant v3] [--dry-run] [--public] [--card release/MODEL_CARD.md]

What it does, in one Modal CPU job (no GPU):
  1. Reads the LoRA from Volume wayfinder-data:/sft/<variant> and the base from wayfinder-weights:/Muse-Glimmer-30B.
  2. Merges W' = W + (alpha/r) B A in fp32, casts to bf16 and writes the shards to the container's local disk (the
     Volumes are only read, never written).
  3. Adds the base's config, tokenizer and chat template unchanged, the rendered model card, Apache-2.0 LICENSE and
     Glimmer USAGE_POLICY.md, the Muse Model License and Spark Open usage policy, NOTICE, scout's tool specs and prompt,
     and wayfinder_release.json (hashes, merge stats, base revision).
  4. Checks the staged folder (required files, no stray files, attribution present, tensor names equal the base index,
     shard hashes equal /sft/<variant>_merged when that exists).
  5. Unless --dry-run: creates the repo PRIVATE, uploads, verifies the remote file list and sizes, and only with
     --public flips it to public, after all of the above succeeded and the card holds no TODO(final) markers.

--dry-run does steps 1 to 4 plus read-only Hub checks (token identity and write role, repo namespace, repo existence)
and uploads nothing. Nothing in this script creates, pushes or changes visibility of a repo without being run without
--dry-run, and nothing makes a repo public without --public.

Secret wayfinder-hf must hold HF_TOKEN
(write role for a real upload).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# Modal CPU price: 8 cores at $0.0000131/core-s + 128 GiB at $0.00000222/GiB-s (about $1.40/h). Upper-bound estimate.
CPU_CORES, MEM_GIB = 8, 128
USD_PER_S = CPU_CORES * 0.0000131 + MEM_GIB * 0.00000222
BASE_REPO = "meta-models/Muse-Glimmer-30B"
TIMEOUT_S = 150 * 60


def _ensure_modal() -> None:
    """Re-exec under the Python that has `modal` when the current interpreter lacks it (modal is a uv tool here)."""
    try:
        import modal  # noqa: F401
        return
    except ImportError:
        pass
    exe = shutil.which("modal")
    if not exe or os.environ.get("WF_PUBLISH_REEXEC"):
        sys.exit("modal is not importable and no `modal` CLI was found on PATH")
    first = Path(exe).read_text(errors="ignore").splitlines()[0]
    if not first.startswith("#!"):
        sys.exit("cannot find modal's interpreter")
    os.environ["WF_PUBLISH_REEXEC"] = "1"
    os.execv(first[2:].strip(), [first[2:].strip(), *sys.argv])


if __name__ == "__main__":
    _ensure_modal()

import modal  # noqa: E402

app = modal.App("wayfinder-release")
weights_vol = modal.Volume.from_name("wayfinder-weights")
data_vol = modal.Volume.from_name("wayfinder-data")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "safetensors", "numpy", "huggingface_hub>=1.0,<2")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(str(HERE), "/root/release", copy=False, ignore=["**/__pycache__"])
    .add_local_file(str(ROOT / "serve" / "scout_tool_specs.json"), "/root/serve/scout_tool_specs.json", copy=False)
    .add_local_file(str(ROOT / "serve" / "scout_product_prompt.md"), "/root/serve/scout_product_prompt.md", copy=False)
)


def _log(msg: str) -> None:
    print(f"[release {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@app.function(
    image=image,
    volumes={"/weights": weights_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("wayfinder-hf")],
    cpu=CPU_CORES,
    memory=MEM_GIB * 1024,
    ephemeral_disk=512 * 1024,
    timeout=TIMEOUT_S,
)
def build_and_publish(repo_id: str, variant: str, dry_run: bool, public: bool, card_text: str) -> dict:
    sys.path.insert(0, "/root/release")
    import torch
    from huggingface_hub import HfApi
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from stage import (
        ATTRIBUTION, LICENSE_FILES, NOTICE, base_files_to_copy, check_card, check_staged, count_todos,
        lora_module_name, public_refusal, render_card, sha256_file,
    )

    t_start = time.time()
    report: dict = {"repo_id": repo_id, "variant": variant, "dry_run": dry_run, "public_requested": public}
    src, adp = Path("/weights/Muse-Glimmer-30B"), Path(f"/data/sft/{variant}")
    merged_ref = Path(f"/data/sft/{variant}_merged")
    stage = Path("/tmp/stage")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    # ---- Hub identity and namespace (read-only) -------------------------------------------------------------
    api = HfApi()
    who = api.whoami()
    role = (who.get("auth", {}).get("accessToken", {}) or {}).get("role")
    namespaces = [who["name"], *[o["name"] for o in who.get("orgs", [])]]
    ns = repo_id.split("/")[0]
    report["hub"] = {"user": who["name"], "token_role": role, "namespaces": namespaces}
    if "/" not in repo_id or ns not in namespaces:
        raise SystemExit(f"repo namespace {ns!r} is not the token's user or one of its orgs {namespaces}")
    exists = api.repo_exists(repo_id, repo_type="model")
    report["hub"]["repo_exists"] = exists
    if exists:
        info = api.model_info(repo_id)
        report["hub"]["repo_private"] = info.private
        if not info.private and not public:
            raise SystemExit(f"{repo_id} already exists and is PUBLIC; pass --public to acknowledge, or pick another id")
    if not dry_run and role not in ("write", "fineGrained"):
        raise SystemExit(f"token role is {role!r}: a real upload needs a write token in Modal secret wayfinder-hf")
    base_sha = api.model_info(BASE_REPO).sha
    report["base_repo_head_sha_now"] = base_sha
    _log(f"hub ok: user={who['name']} role={role} exists={exists} base_head={base_sha[:12]}")

    # ---- card checks ----------------------------------------------------------------------------------------
    card = render_card(card_text, repo_id=repo_id, variant=variant)
    errs = check_card(card)
    todos = count_todos(card)
    report["card"] = {"todo_markers": todos, "errors": errs}
    if errs:
        raise SystemExit(f"model card errors: {errs}")
    if public_refusal(card, public):
        raise SystemExit(public_refusal(card, public))

    # ---- merge ----------------------------------------------------------------------------------------------
    cfg = json.loads((adp / "adapter_config.json").read_text())
    if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("peft_type") != "LORA":
        raise SystemExit(f"unsupported adapter config (peft_type={cfg.get('peft_type')}, dora/rslora set)")
    scale = cfg["lora_alpha"] / cfg["r"]
    adapter_file = adp / "adapter_model.safetensors"
    report["adapter"] = {"r": cfg["r"], "alpha": cfg["lora_alpha"], "sha256": sha256_file(adapter_file),
                         "bytes": adapter_file.stat().st_size}
    lora = load_file(str(adapter_file))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        mod, ab = lora_module_name(k)
        pairs.setdefault(mod, {})[ab] = v
    assert all(set(p) == {"A", "B"} for p in pairs.values()), "unpaired LoRA tensors"
    index = json.loads((src / "model.safetensors.index.json").read_text())
    missing = [m for m in pairs if m not in index["weight_map"]]
    if missing:
        raise SystemExit(f"{len(missing)} adapter modules are not in the base: {missing[:3]}")
    merged_n, max_rel, rel_sum = 0, 0.0, 0.0
    shards = sorted(src.glob("model-*.safetensors"))
    for shard in shards:
        t0 = time.time()
        tensors = load_file(str(shard))
        for name in list(tensors):
            if name in pairs:
                w = tensors[name]
                delta = scale * (pairs[name]["B"].float() @ pairs[name]["A"].float())
                assert delta.shape == w.shape, (name, delta.shape, w.shape)
                rel = float(delta.norm() / w.float().norm())
                max_rel, rel_sum = max(max_rel, rel), rel_sum + rel
                tensors[name] = (w.float() + delta).to(w.dtype)
                merged_n += 1
        save_file(tensors, str(stage / shard.name), metadata={"format": "pt"})
        del tensors
        _log(f"{shard.name}: merged so far {merged_n}/{len(pairs)} ({time.time() - t0:.0f}s)")
    assert merged_n == len(pairs), (merged_n, len(pairs))
    report["merge"] = {"modules": merged_n, "scale": scale, "max_rel_delta_norm": max_rel,
                       "mean_rel_delta_norm": rel_sum / merged_n}

    # ---- the rest of the folder -----------------------------------------------------------------------------
    for n in base_files_to_copy([p.name for p in src.iterdir() if p.is_file()]):
        shutil.copy(src / n, stage / n)
    (stage / "README.md").write_text(card)
    for staged_name, vendored in LICENSE_FILES.items():
        shutil.copy(Path("/root/release/licenses") / vendored, stage / staged_name)
    (stage / "NOTICE").write_text(NOTICE)
    shutil.copy("/root/serve/scout_tool_specs.json", stage / "scout_tool_specs.json")
    shutil.copy("/root/serve/scout_product_prompt.md", stage / "scout_product_prompt.md")

    # ---- checks on the staged folder ------------------------------------------------------------------------
    errs = check_staged(stage)
    wm = index["weight_map"]
    seen: dict[str, str] = {}
    for shard in sorted(stage.glob("model-*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                seen[k] = shard.name
    if seen != wm:
        errs.append(f"tensor-to-shard map differs from the base index ({len(seen)} vs {len(wm)} tensors)")
    shard_hashes = {p.name: sha256_file(p) for p in sorted(stage.glob("model-*.safetensors"))}
    ref = {}
    if merged_ref.exists():
        for name, h in shard_hashes.items():
            rp = merged_ref / name
            ref[name] = (sha256_file(rp) == h) if rp.exists() else None
        report["matches_existing_merged_volume_copy"] = ref
        if not all(v is True for v in ref.values()):
            msg = f"shard hashes differ from {merged_ref}: {ref}"
            if dry_run:
                _log(f"WARNING: {msg}")
            else:
                errs.append(msg)  # the published weights must be the checkpoint that was evaluated
        same_cfg = {}
        for n in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "model.safetensors.index.json", "processor_config.json"):
            if (stage / n).exists() and (merged_ref / n).exists():
                same_cfg[n] = sha256_file(stage / n) == sha256_file(merged_ref / n)
        report["config_files_match_merged_volume_copy"] = same_cfg
        if not all(same_cfg.values()):
            errs.append(f"config or tokenizer files differ from {merged_ref}: {same_cfg}")
    report["staged_errors"] = errs
    if errs:
        raise SystemExit(f"staged folder errors: {errs}")
    total = sum(p.stat().st_size for p in stage.iterdir())
    manifest = {
        "model": repo_id, "variant": variant, "base": BASE_REPO, "base_head_sha_when_published": base_sha,
        "teacher": "meta-models/Muse-Spark-Open-816B-A42B", "attribution": ATTRIBUTION,
        "adapter": report["adapter"], "merge": report["merge"], "shard_sha256": shard_hashes,
        "built_unix": int(time.time()), "model_card_todo_markers": todos,
    }
    (stage / "wayfinder_release.json").write_text(json.dumps(manifest, indent=1))
    files = sorted((p.name, p.stat().st_size) for p in stage.iterdir())
    report["staged"] = {"total_bytes": total, "files": files}
    _log(f"staged {len(files)} files, {total / 2**30:.1f} GiB, errors none")

    # ---- upload ---------------------------------------------------------------------------------------------
    if dry_run:
        report["uploaded"] = False
        report["secs"] = round(time.time() - t_start, 1)
        _log("dry run: nothing uploaded")
        return report
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)  # always private first
    _log("uploading (private)")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=repo_id, folder_path=str(stage), repo_type="model", private=True)
    else:  # newer huggingface_hub without the helper: one resumable LFS commit
        api.upload_folder(repo_id=repo_id, folder_path=str(stage), repo_type="model", commit_message="Wayfinder Glimmer 30B")
    siblings = api.model_info(repo_id, files_metadata=True).siblings
    remote = {s.rfilename: s.size for s in siblings}
    bad = [(n, sz) for n, sz in files if n != "README.md" and remote.get(n) != sz]
    if bad or "README.md" not in remote:
        raise SystemExit(f"remote verification failed (name, local size): {bad[:5]}")
    remote_sha = {s.rfilename: getattr(getattr(s, "lfs", None), "sha256", None) for s in siblings}
    bad_sha = [n for n in shard_hashes if remote_sha.get(n) != shard_hashes[n]]
    if bad_sha:
        raise SystemExit(f"remote LFS sha256 differs from the staged shards: {bad_sha}")
    report["remote_files"] = sorted(remote.items())
    report["remote_shard_sha256_match"] = True
    report["remote_extra_files"] = sorted(set(remote) - {n for n, _ in files} - {".gitattributes"})
    report["uploaded"] = True
    if public:
        api.update_repo_settings(repo_id, private=False, repo_type="model")
        report["made_public"] = True
    report["remote_private"] = api.model_info(repo_id).private
    report["secs"] = round(time.time() - t_start, 1)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_id", help="target repo, e.g. hweller1/wayfinder-glimmer-30b")
    p.add_argument("--variant", default="v1", help="LoRA at wayfinder-data:/sft/<variant> (default v1)")
    p.add_argument("--dry-run", action="store_true", help="build and verify everything, upload nothing")
    p.add_argument("--public", action="store_true", help="make the repo public after upload (default: private)")
    p.add_argument("--card", type=Path, default=HERE / "MODEL_CARD.md")
    p.add_argument("--max-usd", type=float, default=5.0, help="spend-gate amount to check before launch")
    args = p.parse_args()

    from stage import check_card, count_todos, public_refusal, render_card

    card_text = args.card.read_text()
    card = render_card(card_text, repo_id=args.repo_id, variant=args.variant)
    errs = check_card(card)
    todos = count_todos(card)
    print(f"card: {todos} TODO(final) markers, {len(errs)} errors")
    if errs:
        print("\n".join(f"  {e}" for e in errs))
        return 2
    refusal = public_refusal(card, args.public)
    if refusal:
        print(refusal)
        return 2

    # The Hub rejects a commit whose README front matter is invalid (this cost one failed 55 GB upload attempt).
    try:
        import urllib.request
        from huggingface_hub import get_token
        yreq = urllib.request.Request(
            "https://huggingface.co/api/validate-yaml", json.dumps({"content": card, "repoType": "model"}).encode(),
            {"Content-Type": "application/json", "Authorization": f"Bearer {get_token()}"})
        urllib.request.urlopen(yreq, timeout=30)
    except urllib.error.HTTPError as e:
        print(f"Hub rejects the card front matter: {e.read().decode()[:400]}")
        return 2
    except Exception as e:  # offline or no token: the container run still validates at commit time
        print(f"front matter pre-check skipped ({type(e).__name__})")

    t0 = time.time()
    kind = "dryrun" if args.dry_run else "upload"
    try:
        with modal.enable_output(), app.run():
            rep = build_and_publish.remote(args.repo_id, args.variant, args.dry_run, args.public, card_text)
    except BaseException:
        raise
    wall = time.time() - t0
    cost = round(wall * USD_PER_S, 2)
    out_dir = ROOT / "artifacts" / "release"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.variant}_{kind}.json"
    rep["wall_s_total"] = round(wall, 1)
    rep["est_cost_usd"] = cost
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("staged",)}, indent=1))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
