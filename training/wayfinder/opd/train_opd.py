"""OPD trainer: LoRA on Muse-Glimmer-30B (optionally on top of an SFT-merged checkpoint) with the support-restricted
reverse KL of wayfinder/opd/loss.py against stored teacher top-k logprobs.

One replica per process (torchrun), data parallel with a hand all-reduce of the fp32 LoRA gradients, exactly like
wayfinder/sft/train.py (whose model loading, LoRA config and chunked lm_head it reuses). One chain per micro-batch (a chain is
a student-sampled sequence of up to ~49k tokens; no packing). Per global step the loss is the domain-normalised mean over
the step's kept assistant positions (domain = task category).

    torchrun --standalone --nproc-per-node=2 /srv/opd/train_opd.py --model /data/sft/v2a_merged --scored /data/opd/smoke/scored \
        --vocab /srv/opd/opd_shared_vocab.json --steps 20 --chains-per-replica 1 --lr 1e-4 --out /data/opd/smoke/adapter

Metrics per step: loss (weighted objective), sva (unweighted token-mean reverse KL on S), fwd_kl, coverage (student mass on the
teacher top-k set under its full softmax), teacher_mass (teacher mass on that set), agree (student argmax == teacher argmax on S),
kept / total assistant positions. `--eval-chains N` holds N chains out and reports the same metrics (no grad) at step 0, every
`--eval-every` steps and at the end.
"""

from __future__ import annotations

import argparse
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
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/srv")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wayfinder.opd.data import K, kept_counts, load_chain, load_manifest, plan_steps  # noqa: E402
from wayfinder.opd.loss import domain_token_weights, sva_from_logits  # noqa: E402
from wayfinder.opd.vocab import allowed_mask, load_allowed  # noqa: E402

STAT_NAMES = ("wloss", "sva", "fwd", "cov", "tmass", "agree", "kept", "total")


def head_logits_fn(inner: Any):
    """The model's exact head (logits * output_multiplier, then tanh softcap), same as sft/train.py `head_logits`."""
    tc = inner.config.text_config

    def f(h: torch.Tensor) -> torch.Tensor:
        logits = inner.lm_head(h).float() * tc.output_multiplier
        return torch.tanh(logits / tc.final_logit_softcapping) * tc.final_logit_softcapping

    return f


def _chunk(head: Any, h: torch.Tensor, ids: torch.Tensor, lp: torch.Tensor, tgt: torch.Tensor, allowed: torch.Tensor, weight: float) -> torch.Tensor:
    out = sva_from_logits(head(h), ids, lp, allowed, tgt)
    f = out.keep.float()
    return torch.stack([(out.loss * weight).sum(), out.loss.detach().sum(), out.fwd_kl.sum(), out.coverage.sum(), out.teacher_mass.sum(),
                        out.agree.float().sum(), f.sum()])


def chain_loss(hidden: torch.Tensor, head: Any, chain: dict[str, Any], allowed: torch.Tensor, weight: float, chunk: int = 2048, use_ckpt: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
    """hidden [T,H] (student last_hidden_state of the chain). Returns (weighted loss to backprop, stats sums)."""
    pos = torch.from_numpy(chain["pos"]).long().to(hidden.device)
    ids = torch.from_numpy(chain["topk_ids"]).long().to(hidden.device)
    lp = torch.from_numpy(chain["topk_lp"]).to(hidden.device)
    tgt = torch.from_numpy(chain["input_ids"]).long().to(hidden.device)[pos]
    allowed = allowed.to(hidden.device)
    acc = hidden.new_zeros(7, dtype=torch.float32)  # [weighted loss (grad), sva, fwd, cov, tmass, agree, kept]
    for s in range(0, pos.numel(), chunk):
        sl = slice(s, s + chunk)
        h = hidden[pos[sl] - 1]
        if use_ckpt and hidden.requires_grad:
            r = checkpoint(_chunk, head, h, ids[sl], lp[sl], tgt[sl], allowed, weight, use_reentrant=False)
        else:
            r = _chunk(head, h, ids[sl], lp[sl], tgt[sl], allowed, weight)
        acc = acc + r
    acc0 = acc[0]
    acc = acc.detach()
    st = {"sva_sum": float(acc[1]), "fwd_sum": float(acc[2]), "cov_sum": float(acc[3]), "tmass_sum": float(acc[4]), "agree_sum": float(acc[5]),
          "kept": float(acc[6]), "total": float(pos.numel())}
    return acc0, st


def merge_stats(parts: list[dict[str, float]]) -> dict[str, float]:
    tot = {k: sum(p[k] for p in parts) for k in parts[0]} if parts else {}
    k = max(tot.get("kept", 0.0), 1.0)
    return {"sva": tot.get("sva_sum", 0) / k, "fwd_kl": tot.get("fwd_sum", 0) / k, "coverage": tot.get("cov_sum", 0) / k,
            "teacher_mass": tot.get("tmass_sum", 0) / k, "agree": tot.get("agree_sum", 0) / k, "kept": tot.get("kept", 0), "positions": tot.get("total", 0)}


def forward_hidden(pm: Any, chain: dict[str, Any]) -> torch.Tensor:
    from train import _unwrap  # /srv/train.py (wayfinder/sft/train.py)

    inner = _unwrap(pm)
    dev = inner.model.language_model.embed_tokens.weight.device
    ids = torch.from_numpy(chain["input_ids"]).long()[None].to(dev)
    pos = torch.arange(ids.shape[1], device=dev)[None]
    return inner.model.language_model(input_ids=ids, position_ids=pos, use_cache=False).last_hidden_state[0]


@torch.no_grad()
def evaluate(pm: Any, chains: list[dict[str, Any]], allowed: torch.Tensor, label: str, log: Any) -> dict[str, Any]:
    from train import _unwrap

    pm.eval()
    head = head_logits_fn(_unwrap(pm))
    parts, per_dom = [], {}
    for c in chains:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = forward_hidden(pm, c)
        _, st = chain_loss(h, head, c, allowed, 0.0, use_ckpt=False)
        parts.append(st)
        per_dom.setdefault(c["meta"].get("domain", "?"), []).append(st)
        del h
    pm.train()
    res = merge_stats(parts)
    res["by_domain"] = {d: merge_stats(v) for d, v in per_dom.items()}
    res["domain_mean_sva"] = sum(v["sva"] for v in res["by_domain"].values()) / max(1, len(res["by_domain"]))
    log(f"EVAL[{label}] " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items() if k != "by_domain"}))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/weights/Muse-Glimmer-30B")
    ap.add_argument("--scored", required=True, help="dir with manifest.jsonl and <id>.npz (wayfinder/opd/data.py)")
    ap.add_argument("--vocab", default="/srv/opd/opd_shared_vocab.json")
    ap.add_argument("--out", default="/data/opd/adapter")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--gpus-per-replica", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--chains-per-replica", type=int, default=1, help="micro-batches (chains) per replica per optimizer step")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=49152)
    ap.add_argument("--eval-chains", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--limit-chains", type=int, default=0)
    ap.add_argument("--max-dropped-mass", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--log-file", default=None)
    a = ap.parse_args()

    from train import _clip_multi, load_model  # /srv/train.py

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    gpr = a.gpus_per_replica
    devices = [rank * gpr + i for i in range(gpr)]
    torch.cuda.set_device(devices[0])
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{devices[0]}"))
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(a.seed)

    def log(msg: str) -> None:
        if rank == 0:
            print(f"[opd {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    spec = load_allowed(a.vocab)
    allowed = allowed_mask(spec["glimmer_vocab_size"], spec)
    man = [m for m in load_manifest(a.scored) if m["n_tokens"] <= a.max_tokens and m.get("n_positions", 1) > 0]
    if a.limit_chains:
        man = man[: a.limit_chains]
    chains = [{**load_chain(Path(a.scored) / f"{m['id']}.npz")} for m in man]
    for c, m in zip(chains, man):
        c["meta"].setdefault("domain", m.get("domain", "?"))
    n_eval = min(a.eval_chains, max(0, len(chains) - 2))
    rng = random.Random(a.seed)
    idx = list(range(len(chains)))
    rng.shuffle(idx)
    eval_set = [chains[i] for i in idx[:n_eval]]
    train = [chains[i] for i in idx[n_eval:]]
    for c in train + eval_set:
        c["kept"] = kept_counts(c, allowed)
    log(f"{len(train)} train chains, {len(eval_set)} eval chains; tokens {sum(len(c['input_ids']) for c in train) / 1e6:.2f}M; "
        f"assistant positions {sum(len(c['pos']) for c in train)}; kept {sum(c['kept'] for c in train)}; "
        f"domains {sorted({c['meta'].get('domain') for c in train})}")

    t_load = time.time()
    pm = load_model(a.model, a.attn, devices, a.lora_r, a.lora_alpha, True)
    params = [p for p in pm.parameters() if p.requires_grad]
    log(f"loaded {a.model} in {time.time() - t_load:.0f}s; lora params {sum(p.numel() for p in params) / 1e6:.0f}M; attn={a.attn}")
    from train import _unwrap

    head = head_logits_fn(_unwrap(pm))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    per_step = a.chains_per_replica * world
    plan = plan_steps(len(train), per_step, a.steps, a.seed)

    def lr_at(s: int) -> float:
        if s < a.warmup_steps:
            return a.lr * (s + 1) / a.warmup_steps
        t = (s - a.warmup_steps) / max(1, a.steps - a.warmup_steps)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    hist: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    if eval_set:
        e0 = evaluate(pm, eval_set, allowed, "step0", log)
        evals.append({"step": 0, **e0})
    pm.train()
    grad_dev = f"cuda:{devices[0]}"
    for step in range(a.steps):
        torch.cuda.synchronize(devices[0])
        t0 = time.time()
        sel = plan[step]
        ws = domain_token_weights([train[i]["meta"].get("domain", "?") for i in sel], [train[i]["kept"] for i in sel])
        mine = list(range(rank * a.chains_per_replica, (rank + 1) * a.chains_per_replica))
        parts: list[dict[str, float]] = []
        wl = 0.0
        for m in mine:
            c = train[sel[m]]
            if c["kept"] == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = forward_hidden(pm, c)
            loss, st = chain_loss(h, head, c, allowed, ws[m])
            loss.backward()
            wl += float(loss)
            parts.append(st)
            del h, loss
        keys = ("sva_sum", "fwd_sum", "cov_sum", "tmass_sum", "agree_sum", "kept", "total")
        loc = torch.tensor([wl] + [sum(p[k] for p in parts) for k in keys], device=grad_dev, dtype=torch.float64)
        if world > 1:
            flat = torch.cat([p.grad.flatten().to(grad_dev) if p.grad is not None else torch.zeros(p.numel(), device=grad_dev) for p in params])
            dist.all_reduce(flat)
            o = 0
            for p in params:
                n = p.numel()
                p.grad = flat[o : o + n].view_as(p).to(p.device).clone()
                o += n
            del flat
            dist.all_reduce(loc)
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip, foreach=False)) if len(devices) == 1 else _clip_multi(params, a.grad_clip)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for d in devices:
            torch.cuda.synchronize(d)
        dt = time.time() - t0
        v = loc.tolist()
        merged = merge_stats([dict(zip(keys, v[1:]))])
        tok = sum(len(train[i]["input_ids"]) for i in sel)
        rec = {"step": step, "loss": v[0], "gnorm": gn, "lr": lr_at(step), "step_s": round(dt, 2), "tokens": tok, "tok_per_s": round(tok / dt),
               "peak_gib": round(max(torch.cuda.max_memory_allocated(d) for d in devices) / 2**30, 1), **{k: round(x, 5) for k, x in merged.items()}}
        hist.append(rec)
        log(json.dumps(rec))
        if eval_set and a.eval_every and (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            evals.append({"step": step + 1, **evaluate(pm, eval_set, allowed, f"step{step + 1}", log)})
        if rank == 0 and a.save_every and (step + 1) % a.save_every == 0 and step + 1 < a.steps:
            pm.save_pretrained(f"{a.out}/step{step + 1}")
    if eval_set:
        evals.append({"step": a.steps, **evaluate(pm, eval_set, allowed, "final", log)})
    if rank == 0:
        warm = hist[1:] or hist
        summ = {"args": vars(a), "replicas": world, "n_train_chains": len(train), "n_eval_chains": len(eval_set),
                "mean_step_s": sum(h["step_s"] for h in warm) / len(warm), "train_tok_per_s_total": sum(h["tokens"] for h in warm) / sum(h["step_s"] for h in warm),
                "assistant_positions_per_s": sum(h["positions"] for h in warm) / sum(h["step_s"] for h in warm),
                "first_sva": hist[0]["sva"], "last_sva": hist[-1]["sva"], "mean_first3_sva": sum(h["sva"] for h in hist[:3]) / len(hist[:3]),
                "mean_last3_sva": sum(h["sva"] for h in hist[-3:]) / len(hist[-3:]), "peak_gib": max(h["peak_gib"] for h in hist),
                "evals": evals, "history": hist}
        log("SUMMARY " + json.dumps({k: v for k, v in summ.items() if k not in ("history", "args", "evals")}))
        if a.log_file:
            Path(a.log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(a.log_file).write_text(json.dumps(summ, indent=1))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        pm.save_pretrained(a.out)
        log(f"adapter saved to {a.out}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
