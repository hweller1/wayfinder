"""LoRA SFT trainer for Muse-Glimmer-30B: packing with cross-document attention masking, assistant-only loss.

Design (see docs/sft_dryrun.md for measurements):
  * One process per replica (torchrun). A replica owns `--gpus-per-replica` GPUs; layers are split across them
    (naive layer-wise model parallelism through accelerate's device_map), so a replica is 1 GPU on H200 and 2 GPUs on H100.
    Replicas are data-parallel; only the LoRA gradients (fp32, ~430M params) are all-reduced, by hand, once per step.
  * Packing: examples are first-fit packed into bins of <= --pack-len tokens and fed as ONE row with per-document
    position_ids (restarting at 0). transformers detects the restarts (flash-attn varlen via cu_seqlens when a flash
    kernel is selected; a block-diagonal mask otherwise), so no token attends across a document boundary, in the
    full-attention and in the sliding-window layers alike. `--no-pack` runs one example per micro-batch instead.
  * Loss: assistant tokens only (the `assistant_spans` of the data contract). lm_head runs on the assistant positions only,
    in chunks under activation checkpointing, with the model's exact head (logits * output_multiplier, then
    T * tanh(. / T) softcap), so the [T, 202048] logits tensor is never materialised. `--verify` compares the result with
    the model's own forward and packed vs unpacked.
  * LoRA r=64 on q,k,v,o,gate_proj(attention gate) and MLP gate/up/down of the language model; adapters in fp32, base bf16,
    autocast bf16, gradient checkpointing, AdamW, cosine lr with short linear warmup.

Data: JSONL rows with `input_ids`, `assistant_spans` ([start, end) token ranges that carry loss), optional `id`.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LORA_TARGETS = r".*language_model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


# ------------------------------------------------------------------------------------------------ data
def load_examples(patterns: list[str], limit: int | None, max_len: int) -> list[dict[str, Any]]:
    files: list[str] = []
    for p in patterns:
        files += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
    rows: list[dict[str, Any]] = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                ids = r["input_ids"]
                spans = [(int(a), int(b)) for a, b in r["assistant_spans"]]
                if len(ids) > max_len or not spans:
                    continue
                assert all(0 < a < b <= len(ids) for a, b in spans), f"bad span in {r.get('id')}"
                rows.append({"id": r.get("id"), "input_ids": ids, "spans": spans})
                if limit and len(rows) >= limit:
                    return rows
    return rows


def pack_bins(lengths: list[int], order: list[int], pack_len: int) -> list[list[int]]:
    """First-fit packing over a (shuffled) order, with a bounded number of open bins so the order stays near-random."""
    bins: list[list[int]] = []
    free: list[int] = []
    for i in order:
        for b in range(len(bins)):
            if lengths[i] <= free[b]:
                bins[b].append(i)
                free[b] -= lengths[i]
                break
        else:
            bins.append([i])
            free.append(pack_len - lengths[i])
    return bins


def make_batch(rows: list[dict[str, Any]], idxs: list[int]) -> dict[str, torch.Tensor]:
    ids: list[int] = []
    pos: list[int] = []
    label_pos: list[int] = []  # packed index of each assistant token
    for i in idxs:
        r = rows[i]
        base = len(ids)
        ids += r["input_ids"]
        pos += list(range(len(r["input_ids"])))
        for a, b in r["spans"]:
            label_pos += list(range(base + a, base + b))
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "position_ids": torch.tensor([pos], dtype=torch.long),
        "label_pos": torch.tensor(label_pos, dtype=torch.long),
        "n_docs": torch.tensor(len(idxs)),
    }


# ------------------------------------------------------------------------------------------------ model
def device_map_for(n_layers: int, devices: list[int]) -> dict[str, int]:
    g = len(devices)
    dm: dict[str, int] = {}
    # lm_head + final norm + loss live on the last device; the embedding and vision stub on the first.
    for k in ("model.vision_tower", "model.vision_adapter", "model.vision_projection", "model.perception_emb_norm",
              "model.language_model.embed_tokens", "model.language_model.rotary_emb"):
        dm[k] = devices[0]
    # give the last device two fewer layers when g > 1: it also holds lm_head and the loss chunks
    if g == 1:
        counts = [n_layers]
    else:
        base = n_layers / g
        counts = [round(base)] * g
        counts[-1] = max(1, counts[-1] - 1)
        counts[0] += n_layers - sum(counts)
    li = 0
    for d, c in zip(devices, counts):
        for _ in range(c):
            dm[f"model.language_model.layers.{li}"] = d
            li += 1
    dm["model.language_model.norm"] = devices[-1]
    dm["lm_head"] = devices[-1]
    return dm


def load_model(path: str, attn: str, devices: list[int], lora_r: int, lora_alpha: int, grad_ckpt: bool, tiny: bool = False):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, MuseGlimmerForConditionalGeneration

    cfg = AutoConfig.from_pretrained(path)
    n_layers = cfg.text_config.num_hidden_layers
    dm = device_map_for(n_layers, devices)
    model = None
    for impl in [x for x in (attn, "sdpa") if x]:  # fall back to sdpa (block-diagonal mask) if the flash kernel is unavailable
        try:
            model = MuseGlimmerForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, device_map=dm, attn_implementation=impl, allow_all_kernels=impl.startswith("kernels-community/")
            )
            print(f"[train] attention implementation: {impl}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[train] attn_implementation={impl} failed: {e!r}", flush=True)
            if impl == "sdpa":
                raise
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none")
    pm = get_peft_model(model, lcfg)  # peft upcasts adapter weights to fp32 (autocast_adapter_dtype)
    for n, p in pm.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    model.config.use_cache = False
    return pm


def _unwrap(pm: Any) -> Any:
    from peft import PeftModel

    return pm.base_model.model if isinstance(pm, PeftModel) else pm


def head_logits(inner: Any, h: torch.Tensor) -> torch.Tensor:
    tc = inner.config.text_config
    logits = inner.lm_head(h).float()
    logits = logits * tc.output_multiplier
    logits = torch.tanh(logits / tc.final_logit_softcapping) * tc.final_logit_softcapping
    return logits


def _chunk_loss(inner: Any, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(head_logits(inner, h), labels, reduction="sum")


def packed_loss(pm: Any, batch: dict[str, torch.Tensor], first_dev: Any = None, chunk: int = 2048) -> tuple[torch.Tensor, int]:
    """Sum of next-token CE over assistant tokens of one packed row, and the number of such tokens."""
    inner = _unwrap(pm)
    dev = inner.model.language_model.embed_tokens.weight.device
    ids = batch["input_ids"].to(dev)
    pos = batch["position_ids"].to(dev)
    hid = inner.model.language_model(input_ids=ids, position_ids=pos, use_cache=False).last_hidden_state[0]
    lp = batch["label_pos"].to(hid.device)
    # a doc's first token is never an assistant token (bos), so lp-1 never crosses a document boundary
    assert bool((pos[0, lp.to(pos.device)] > 0).all()), "assistant span starts at a document's first token"
    labels = ids[0, lp.to(ids.device)].to(hid.device)
    src = lp - 1
    total = hid.new_zeros((), dtype=torch.float32)
    for s in range(0, src.numel(), chunk):
        h = hid[src[s : s + chunk]]
        total = total + checkpoint(_chunk_loss, inner, h, labels[s : s + chunk], use_reentrant=False)
    return total, int(src.numel())


# ------------------------------------------------------------------------------------------------ verification
@torch.no_grad()
def verify(pm: Any, rows: list[dict[str, Any]], first_dev: Any = None) -> dict[str, float]:
    """(1) chunked head vs the model's own full-logits forward; (2) packed two-doc row vs the docs run separately."""
    pm.eval()
    inner = _unwrap(pm)
    short = [i for i, r in enumerate(rows) if len(r["input_ids"]) <= 3000][:2]
    if len(short) < 2:
        short = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))[:2]
    out: dict[str, float] = {}
    sep_sum = 0.0
    sep_n = 0
    for k, i in enumerate(short):
        b = make_batch(rows, [i])
        mine, n = packed_loss(pm, b, first_dev)
        ids = b["input_ids"].to(inner.model.language_model.embed_tokens.weight.device)
        ref_logits = inner(input_ids=ids, use_cache=False).logits[0].float()
        lp = b["label_pos"].to(ref_logits.device)
        ref = F.cross_entropy(ref_logits[lp - 1], ids[0, lp.to(ids.device)].to(ref_logits.device), reduction="sum")
        out[f"head_vs_model_abs_diff_doc{k}"] = abs(float(mine) - float(ref))
        out[f"head_vs_model_rel_diff_doc{k}"] = abs(float(mine) - float(ref)) / max(1e-9, float(ref))
        sep_sum += float(mine)
        sep_n += n
        del ref_logits
    pk, pn = packed_loss(pm, make_batch(rows, short), first_dev)
    out["packed_sum"] = float(pk)
    out["separate_sum"] = sep_sum
    out["packed_vs_separate_rel_diff"] = abs(float(pk) - sep_sum) / max(1e-9, sep_sum)
    out["n_label_tokens"] = sep_n
    assert pn == sep_n
    pm.train()
    return out


# ------------------------------------------------------------------------------------------------ probe
def _rand_doc(n: int, rng: random.Random) -> dict[str, Any]:
    ids = [200000] + [rng.randrange(1000, 150000) for _ in range(n - 1)]
    return {"id": "synthetic", "input_ids": ids, "spans": [(n - n // 4, n)]}


def probe(pm: Any, lens: list[int], devices: list[int], log_path: str | None, do_verify: bool) -> None:
    """Peak memory and step time of one fwd+bwd at each length (packed row of 2 docs of n/2 when n >= 8192)."""
    rng = random.Random(0)
    out: dict[str, Any] = {"devices": devices, "rows": []}
    pm.train()  # from_pretrained returns eval mode; gradient checkpointing only acts in train mode
    if do_verify:
        vr = verify(pm, [_rand_doc(900, rng), _rand_doc(1300, rng)])
        print("VERIFY", json.dumps(vr), flush=True)
        out["verify"] = vr
        pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    for n in lens:
        for d in devices:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(d)
        rows = [_rand_doc(n // 2, rng), _rand_doc(n - n // 2, rng)] if n >= 8192 else [_rand_doc(n, rng)]
        rec: dict[str, Any] = {"seq_len": n, "docs": len(rows)}
        try:
            times = []
            for it in range(2):
                for d in devices:
                    torch.cuda.synchronize(d)
                t0 = time.time()
                batch = make_batch(rows, list(range(len(rows))))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ls, k = packed_loss(pm, batch)
                (ls / k).backward()
                for d in devices:
                    torch.cuda.synchronize(d)
                times.append(time.time() - t0)
                for p in params:
                    p.grad = None
            rec.update(ok=True, fwd_bwd_s=round(times[-1], 2), first_s=round(times[0], 2), tok_per_s=round(n / times[-1]),
                       peak_gib=[round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in devices])
        except torch.OutOfMemoryError as e:
            rec.update(ok=False, error="OOM " + str(e)[:700])
            for p in params:
                p.grad = None
        print("PROBE", json.dumps(rec), flush=True)
        out["rows"].append(rec)
        if not rec["ok"]:
            break
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/weights/Muse-Glimmer-30B")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="/data/sft/dryrun")
    ap.add_argument("--attn", default="kernels-community/flash-attn3")
    ap.add_argument("--gpus-per-replica", type=int, default=1)
    ap.add_argument("--pack-len", type=int, default=32768)
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--limit-examples", type=int, default=None)
    ap.add_argument("--steps", type=int, default=50, help="optimizer steps (0 = one epoch)")
    ap.add_argument("--packs-per-replica-per-step", type=int, default=2, help="gradient accumulation (micro-batches)")
    ap.add_argument("--epochs", type=int, default=1, help="with --steps 0: run this many epochs, re-shuffling and re-packing each epoch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--save-every", type=int, default=0, help="also save the adapter every N steps (rank 0)")
    ap.add_argument("--synthetic-len", type=int, default=0,
                    help="MEMORY PROBE ONLY: ignore --data and run random tokens of this length (assistant = last 25%%)")
    ap.add_argument("--probe-lens", default="",
                    help="MEMORY/SPEED PROBE ONLY: comma list of sequence lengths; one fwd+bwd on random tokens each (OOM caught), then exit")
    ap.add_argument("--log-file", default=None)
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    gpr = a.gpus_per_replica
    devices = [rank * gpr + i for i in range(gpr)]
    torch.cuda.set_device(devices[0])
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{devices[0]}"))
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(a.seed)
    t_load = time.time()

    def log(msg: str) -> None:
        if rank == 0:
            print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    pm = load_model(a.model, a.attn, devices, a.lora_r, a.lora_alpha, not a.no_grad_ckpt)
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    log(f"loaded in {time.time() - t_load:.0f}s; attn={a.attn} replicas={world} gpus/replica={gpr} lora params={n_train / 1e6:.0f}M")
    params = [p for p in pm.parameters() if p.requires_grad]

    for d in devices:
        log(f"after load: cuda:{d} allocated {torch.cuda.memory_allocated(d) / 2**30:.1f} GiB of {torch.cuda.get_device_properties(d).total_memory / 2**30:.1f}")
    if a.probe_lens:
        probe(pm, [int(x) for x in a.probe_lens.split(",")], devices, a.log_file, a.verify)
        return
    if a.synthetic_len:
        n = a.synthetic_len
        rows = [{"id": "synthetic", "input_ids": [random.randrange(1000, 150000) for _ in range(n)],
                 "spans": [(n - n // 4, n)]}]
        rows[0]["input_ids"][0] = 200000
    else:
        rows = load_examples(a.data, a.limit_examples, a.pack_len)
    lengths = [len(r["input_ids"]) for r in rows]
    log(f"{len(rows)} examples, {sum(lengths) / 1e6:.2f}M tokens, mean {sum(lengths) / max(1, len(lengths)):.0f}")

    if a.verify:
        res = verify(pm, rows, devices[0])
        log("VERIFY " + json.dumps(res))
        if rank == 0 and a.log_file:
            Path(a.log_file).with_suffix(".verify.json").write_text(json.dumps(res, indent=1))

    # micro-batch plan: shuffled, packed; every rank derives the same plan
    order = list(range(len(rows)))
    random.Random(a.seed).shuffle(order)
    if a.no_pack:
        bins = [[i] for i in order]
    else:
        bins = pack_bins(lengths, order, a.pack_len)
    random.Random(a.seed + 1).shuffle(bins)
    per_step = a.packs_per_replica_per_step * world
    n_steps_epoch = len(bins) // per_step
    if a.epochs > 1 and not a.steps:
        full = bins[: n_steps_epoch * per_step]
        for e in range(1, a.epochs):  # fresh order and packing per epoch (every rank derives the same plan)
            o2 = list(range(len(rows)))
            random.Random(a.seed + 100 * e).shuffle(o2)
            b2 = [[i] for i in o2] if a.no_pack else pack_bins(lengths, o2, a.pack_len)
            random.Random(a.seed + 100 * e + 1).shuffle(b2)
            full += b2[: n_steps_epoch * per_step]
        bins = full
    steps = a.steps or n_steps_epoch * max(1, a.epochs)
    if steps > n_steps_epoch:  # cycle for a probe/dry run
        bins = (bins * math.ceil(steps * per_step / len(bins)))
    if a.synthetic_len:
        bins = [[0]] * (steps * per_step)
    log(f"{len(bins)} micro-batches ({'unpacked' if a.no_pack else 'packed<=' + str(a.pack_len)}); {steps} steps x {per_step} micro-batches; epoch = {n_steps_epoch} steps")

    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(s: int) -> float:
        if s < a.warmup_steps:
            return a.lr * (s + 1) / a.warmup_steps
        t = (s - a.warmup_steps) / max(1, steps - a.warmup_steps)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    pm.train()
    hist: list[dict[str, Any]] = []
    for d in devices:
        torch.cuda.reset_peak_memory_stats(d)
    grad_dev = f"cuda:{devices[0]}"
    for step in range(steps):
        torch.cuda.synchronize(devices[0])
        t0 = time.time()
        step_bins = bins[step * per_step : (step + 1) * per_step]
        step_lab = sum(sum(len(rows[i]["spans"]) and sum(b - a_ for a_, b in rows[i]["spans"]) for i in bn) for bn in step_bins)
        step_tok = sum(sum(lengths[i] for i in bn) for bn in step_bins)
        mine = step_bins[rank * a.packs_per_replica_per_step : (rank + 1) * a.packs_per_replica_per_step]
        loss_sum = 0.0
        for bn in mine:
            batch = make_batch(rows, bn)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls, _ = packed_loss(pm, batch, devices[0])
            (ls / step_lab).backward()  # token-mean over the whole global step
            loss_sum += float(ls)
        if world > 1:
            flat = torch.cat([p.grad.flatten().to(grad_dev) for p in params])
            dist.all_reduce(flat)
            o = 0
            for p in params:
                n = p.numel()
                p.grad.copy_(flat[o : o + n].view_as(p.grad).to(p.grad.device))
                o += n
            del flat
            tot = torch.tensor([loss_sum], device=grad_dev)
            dist.all_reduce(tot)
            loss_sum = float(tot)
        gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in params], a.grad_clip, foreach=False)) if len(devices) == 1 else _clip_multi(params, a.grad_clip)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for d in devices:
            torch.cuda.synchronize(d)
        dt = time.time() - t0
        peak = max(torch.cuda.max_memory_allocated(d) for d in devices) / 2**30
        rec = {"step": step, "loss": loss_sum / step_lab, "gnorm": gnorm, "lr": lr_at(step), "step_s": round(dt, 2),
               "tokens": step_tok, "assistant_tokens": step_lab, "tok_per_s": round(step_tok / dt),
               "peak_gib_rank0": round(peak, 1)}
        hist.append(rec)
        log(json.dumps(rec))
        if rank == 0 and a.save and a.save_every and (step + 1) % a.save_every == 0 and step + 1 < steps:
            pm.save_pretrained(f"{a.out}/step{step + 1}")
            log(f"adapter checkpoint saved to {a.out}/step{step + 1}")
    if rank == 0:
        warm = hist[2:] or hist
        summ = {
            "args": vars(a), "replicas": world, "gpus_per_replica": gpr, "n_examples": len(rows),
            "mean_step_s": sum(h["step_s"] for h in warm) / len(warm),
            "tok_per_s_total": sum(h["tokens"] for h in warm) / sum(h["step_s"] for h in warm),
            "assistant_tok_per_s_total": sum(h["assistant_tokens"] for h in warm) / sum(h["step_s"] for h in warm),
            "peak_gib_per_gpu": max(h["peak_gib_rank0"] for h in hist),
            "first_loss": hist[0]["loss"], "last_loss": hist[-1]["loss"],
            "mean_first5": sum(h["loss"] for h in hist[:5]) / len(hist[:5]), "mean_last5": sum(h["loss"] for h in hist[-5:]) / len(hist[-5:]),
            "history": hist,
        }
        log("SUMMARY " + json.dumps({k: v for k, v in summ.items() if k not in ("history", "args")}))
        if a.log_file:
            Path(a.log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(a.log_file).write_text(json.dumps(summ, indent=1))
        if a.save:
            pm.save_pretrained(a.out)
            log(f"adapter saved to {a.out}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _clip_multi(params: list[torch.nn.Parameter], max_norm: float) -> float:
    norms = torch.stack([p.grad.float().norm().to("cuda:0" if False else params[0].grad.device) for p in params])
    total = float(norms.norm())
    coef = min(1.0, max_norm / (total + 1e-6))
    if coef < 1.0:
        for p in params:
            p.grad.mul_(coef)
    return total


if __name__ == "__main__":
    sys.exit(main())
