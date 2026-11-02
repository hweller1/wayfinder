"""GRPO trainer: LoRA r64 on Muse-Glimmer-30B (the SFT checkpoint, merged), one torchrun replica per GPU, data parallel with a hand all-reduce of
the fp32 LoRA gradients (exactly wayfinder/sft/train.py and wayfinder/opd/train_opd.py, whose model loading, LoRA config and softcapped chunked
head it reuses via /srv/train.py).

Control plane is local files under --ctl (the Modal parent process in train_app.py bridges them to a Modal Dict): the driver publishes
`batch_<k>.json` (a manifest of rollouts with their advantages), this process writes `done_<k>.json` after the update and the adapter files.
Per step k: wait for batch_k, split its rollouts over the replicas by token count, forward + backward each chain (loss only on the sampled
assistant tokens: wayfinder/grpo/step.py), all-reduce, clip, AdamW step, save the bf16 adapter `adapters/v<k+1>` for the rollout servers and,
every --save-every steps, the fp32 adapter `ckpt/s<k+1>` for merging and evaluation.

Staleness: batch k was sampled by adapter v_(k-1) (rollouts of k overlap the training of k-1), trained here at v_k. The ratio uses vLLM's
processed logprobs of the sampled tokens as the behaviour policy, so the PPO clip covers the one-step lag.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/srv")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wayfinder.grpo.cached_fork import train_prefix_group  # noqa: E402
from wayfinder.grpo.data import fork_parent, load_rollout, plan_ranks, plan_ranks_grouped, step_normaliser  # noqa: E402
from wayfinder.grpo.optstate import LrSchedule, UpdateGuard, load_state, publish_resume_slot, save_state  # noqa: E402
from wayfinder.grpo.step import STAT_KEYS, merge_stat_sums, rollout_chain_loss  # noqa: E402


def head_fn(inner: Any):
    tc = inner.config.text_config

    def f(h: torch.Tensor) -> torch.Tensor:
        logits = inner.lm_head(h).float() * tc.output_multiplier
        return torch.tanh(logits / tc.final_logit_softcapping) * tc.final_logit_softcapping

    return f


def forward_hidden(pm: Any, ids: torch.Tensor) -> torch.Tensor:
    from train import _unwrap  # /srv/train.py (wayfinder/sft/train.py)

    inner = _unwrap(pm)
    dev = inner.model.language_model.embed_tokens.weight.device
    x = ids[None].to(dev)
    pos = torch.arange(x.shape[1], device=dev)[None]
    return inner.model.language_model(input_ids=x, position_ids=pos, use_cache=False).last_hidden_state[0]


def save_adapter(pm: Any, out: str, dtype: torch.dtype) -> None:
    """Same files peft's save_pretrained writes (adapter_model.safetensors + adapter_config.json), tensors cast to `dtype`."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(dtype).cpu().contiguous() for k, v in get_peft_model_state_dict(pm).items()}
    save_file(sd, str(d / "adapter_model.safetensors"), metadata={"format": "pt"})
    pm.peft_config["default"].save_pretrained(str(d))


def load_adapter_into(pm: Any, path: str) -> int:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(str(Path(path) / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(pm, sd)
    bad = getattr(res, "unexpected_keys", [])
    if bad:
        raise RuntimeError(f"unexpected adapter keys: {bad[:3]}")
    return len(sd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/sft/v3_merged")
    ap.add_argument("--run-name", required=True)  # not --run: torchrun abbreviates it to its own --run-path
    ap.add_argument("--gdata", default="/gdata")
    ap.add_argument("--ctl", default="/tmp/ctl")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--resume-adapter", default="", help="fp32 adapter dir to continue from (ckpt/s<N> or resume/); its optimizer.pt is restored when present, "
                    "otherwise the lr re-warms from 0 over --rewarm-updates updates")
    ap.add_argument("--rewarm-updates", type=int, default=10)
    ap.add_argument("--resume-slot-every", type=int, default=1, help="write the rolling resume/ slot (fp32 adapter + AdamW state) every N steps (0 = only with ckpt/)")
    ap.add_argument("--guard-gn-mult", type=float, default=5.0, help="skip an update whose grad norm exceeds this x its moving average (0 = off)")
    ap.add_argument("--guard-kl", type=float, default=0.1, help="skip an update when the step's KL to the reference exceeds this (0 = off)")
    ap.add_argument("--guard-stuck", type=int, default=3, help="consecutive skipped updates after which the driver stops the run")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.02, help="KL coefficient to the reference (adapter disabled)")
    ap.add_argument("--clip-lo", type=float, default=0.2)
    ap.add_argument("--clip-hi", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--gpus-per-replica", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fork-cached", action="store_true", help="forked-report mode: train the forked completions over a no-grad KV cache of their shared prefix "
                    "(one prefix forward per prefix; gradients skip the prefix K/V: wayfinder/grpo/cached_fork.py). Default off = the full-sequence path, unchanged")
    ap.add_argument("--fork-cached-steps", default="", help="with --fork-cached: comma list of steps that use it (default all); the other steps run the full-sequence path, "
                    "so one run measures cached vs full-sequence seconds per fork (the GPU smoke)")
    ap.add_argument("--fork-cached-verify", type=int, default=0, help="with --fork-cached: for the first N prefix groups of a step also run the full-sequence forward "
                    "(no grad) and log max |logp cached - logp full| (the GPU check of the flash-attention path)")
    ap.add_argument("--wait-s", type=int, default=6 * 3600, help="give up waiting for a batch after this long")
    a = ap.parse_args()

    from train import _clip_multi, _unwrap, load_model  # /srv/train.py

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    gpr = a.gpus_per_replica
    devices = [rank * gpr + i for i in range(gpr)]
    torch.cuda.set_device(devices[0])
    if world > 1:
        # long timeout: non-zero ranks sit in the batch broadcast while rank 0 waits for rollouts (a step can wait an hour+)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{devices[0]}"), timeout=datetime.timedelta(hours=8))
    torch.backends.cuda.matmul.allow_tf32 = True
    ctl = Path(a.ctl)
    ctl.mkdir(parents=True, exist_ok=True)
    gd = Path(a.gdata) / a.run_name

    def log(msg: str) -> None:
        if rank == 0:
            print(f"[grpo-train {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    t_load = time.time()
    pm = load_model(a.model, a.attn, devices, a.lora_r, a.lora_alpha, True)
    params = [p for p in pm.parameters() if p.requires_grad]
    if a.resume_adapter:
        log(f"resumed {load_adapter_into(pm, a.resume_adapter)} adapter tensors from {a.resume_adapter}")
    log(f"loaded {a.model} in {time.time() - t_load:.0f}s; lora params {sum(p.numel() for p in params) / 1e6:.0f}M; replicas {world}")
    head = head_fn(_unwrap(pm))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    guard = UpdateGuard(gn_mult=a.guard_gn_mult if a.guard_gn_mult > 0 else float("inf"), kl_max=a.guard_kl if a.guard_kl > 0 else float("inf"), stuck_after=a.guard_stuck)
    ometa = load_state(opt, Path(a.resume_adapter) / "optimizer.pt") if a.resume_adapter else None
    sched = LrSchedule(a.lr, a.warmup_steps, a.rewarm_updates, resumed=bool(a.resume_adapter), has_opt_state=ometa is not None)
    if ometa is not None:
        sched.applied = int(ometa.get("applied", 0))
        guard.load(ometa.get("guard", {}))
        log(f"restored AdamW state from {a.resume_adapter}/optimizer.pt (applied updates {sched.applied}, guard history {len(guard.hist)})")
    elif a.resume_adapter:
        log(f"WARNING: no optimizer state in {a.resume_adapter}: lr re-warms from 0 over {a.rewarm_updates} updates")
    pm.train()
    grad_dev = f"cuda:{devices[0]}"
    ref_fwd = None
    if a.beta > 0:
        def ref_fwd(ids: torch.Tensor) -> torch.Tensor:  # noqa: F811
            with pm.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16):
                return forward_hidden(pm, ids)
    if rank == 0:
        if a.start_step == 0:  # a fresh run: B = 0, version 0 is exactly the base checkpoint; a new run seeded from another run's adapter: v0 = that adapter (rollouts of steps 0 and 1 load it)
            save_adapter(pm, str(gd / "adapters" / "v0"), torch.bfloat16)
        (ctl / "ready").write_text(json.dumps({"t": time.time(), "load_s": time.time() - t_load}))

    for step in range(a.start_step, a.steps):
        # ---- wait for the batch (rank 0 polls the bridge's files, then broadcasts the manifest)
        man: list[Any] = [None]
        if rank == 0:
            t_wait = time.time()
            bp, sp = ctl / f"batch_{step}.json", ctl / "stop"
            while man[0] is None:
                if bp.exists():
                    man[0] = json.loads(bp.read_text())
                elif sp.exists() or time.time() - t_wait > a.wait_s:
                    man[0] = {"stop": True}
                else:
                    time.sleep(2)
            man[0]["wait_s"] = round(time.time() - t_wait, 1)
        if world > 1:
            dist.broadcast_object_list(man, src=0)
        m = man[0]
        if m.get("stop"):
            log("stop requested")
            break
        for d in devices:
            torch.cuda.synchronize(d)
        t0 = time.time()
        items = m["items"]
        no_update = not items  # every group had zero variance (or failed): publish the unchanged adapter as the next version, no optimizer step
        n_total = step_normaliser(items)
        use_cached = a.fork_cached and (not a.fork_cached_steps or step in {int(x) for x in a.fork_cached_steps.split(",") if x})
        if a.fork_cached:
            for d in devices:
                torch.cuda.reset_peak_memory_stats(d)  # per-step peak, so cached and full-sequence steps of a smoke run are comparable
        if use_cached:  # every forked completion of a prefix on the replica that runs the prefix forward
            plan = plan_ranks_grouped([(it["rid"], it["n_tokens"]) for it in items], world, {it["rid"]: it["n_loss"] for it in items})
        else:
            plan = plan_ranks([(it["rid"], it["n_tokens"]) for it in items], world)
        by_rid = {it["rid"]: it for it in items}
        fk_n = fk_tok = fk_grad_tok = n_verified = 0  # forked-completion timing (only moves when the step has forked items)
        fk_s = fk_abs = fk_lr = fk_ntok = 0.0  # forked-completion trainer seconds and logratio sums (trainer logp - vLLM logp)
        parts: list[dict[str, float]] = []
        n_chains = n_oom = 0
        tok_fwd = 0
        todo = list(plan[rank])
        while todo:
            rid = todo.pop(0)
            it = by_rid[rid]
            if use_cached and fork_parent(rid) is not None:
                par = fork_parent(rid)
                grp = [rid] + [r for r in todo if fork_parent(r) == par]
                todo = [r for r in todo if fork_parent(r) != par]
                torch.cuda.synchronize(devices[0])
                t_f = time.time()
                fchains = []
                for r in grp:
                    with np.load(by_rid[r]["path"], allow_pickle=False) as z:
                        fchains.append({"ids": z["ids0"], "spans": z["sp0"], "lp": z["lp0"]})
                try:
                    sts, rest, P = train_prefix_group(
                        _unwrap(pm).model.language_model, head, fchains, [by_rid[r]["adv"] for r in grp], n_total, ref_ctx=pm.disable_adapter if a.beta > 0 else None,
                        autocast=lambda: torch.autocast("cuda", dtype=torch.bfloat16), temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, clip_lo=a.clip_lo,
                        clip_hi=a.clip_hi, beta=a.beta, device=f"cuda:{devices[0]}")
                    done = [i for i in range(len(grp)) if i not in rest]
                    if n_verified < a.fork_cached_verify and done:
                        n_verified += 1
                        log(f"fork-cached verify (prefix {P} tokens, {len(done)} forks): max|dlogp cached-full| = {verify_group(pm, head, fchains[done[0]], P, a):.3e}")
                    parts += sts
                    fk_abs += sum(x['abs_logratio'] for x in sts)
                    fk_lr += sum(x['logratio'] for x in sts)
                    fk_ntok += sum(x['tokens'] for x in sts)
                    n_chains += len(sts)
                    tok_fwd += P + sum(len(fchains[i]["ids"]) - P for i in done)
                    fk_n += len(done)
                    fk_tok += P + sum(len(fchains[i]["ids"]) - P for i in done)
                    fk_grad_tok += sum(len(fchains[i]["ids"]) - P for i in done)
                except torch.OutOfMemoryError:
                    n_oom += len(grp)
                    rest = []
                    torch.cuda.empty_cache()
                    print(f"[grpo-train rank{rank}] OOM on a cached fork group of {len(grp)} (prefix {par}); skipped", flush=True)
                for i in rest:  # a fork whose prefix differs from its group's: the full-sequence path
                    loss, st = rollout_chain_loss(lambda ids: forward_hidden_autocast(pm, ids), ref_fwd, head, fchains[i], by_rid[grp[i]]["adv"], n_total, temperature=a.temperature,
                                                  top_k=a.top_k, top_p=a.top_p, clip_lo=a.clip_lo, clip_hi=a.clip_hi, beta=a.beta, device=f"cuda:{devices[0]}")
                    if loss is not None:
                        loss.backward()
                    parts.append(st)
                    fk_abs += st["abs_logratio"]
                    fk_lr += st["logratio"]
                    fk_ntok += st["tokens"]
                    n_chains += 1
                    tok_fwd += len(fchains[i]["ids"])
                    del loss
                torch.cuda.synchronize(devices[0])
                fk_s += time.time() - t_f
                continue
            t_f = None
            if fork_parent(rid) is not None:  # flag off: same code below, only timed
                torch.cuda.synchronize(devices[0])
                t_f = time.time()
            n_parts0 = len(parts)
            with np.load(it["path"], allow_pickle=False) as z:  # closed immediately: the bridge reloads the volume between steps
                meta = json.loads(str(z["meta"]))
                chains = [{"ids": z[f"ids{i}"], "spans": z[f"sp{i}"], "lp": z[f"lp{i}"]} for i in range(meta["n_chains"])]
            fk = it.get("fork")  # forked-report mode: the original's report span also carries the intra-prefix advantage
            for ci, ch in enumerate(chains):
                try:
                    loss, st = rollout_chain_loss(
                        lambda ids: forward_hidden_autocast(pm, ids), ref_fwd, head, ch, it["adv"], n_total, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                        clip_lo=a.clip_lo, clip_hi=a.clip_hi, beta=a.beta, device=f"cuda:{devices[0]}",
                        span_adv={int(fk["span"]): float(fk["adv"])} if fk and int(fk["chain"]) == ci else None)
                    if loss is not None:
                        loss.backward()
                    parts.append(st)
                    n_chains += 1
                    tok_fwd += len(ch["ids"])
                    del loss
                except torch.OutOfMemoryError:
                    n_oom += 1
                    torch.cuda.empty_cache()  # grads of the chains finished so far are kept
                    print(f"[grpo-train rank{rank}] OOM on a {len(ch['ids'])}-token chain (rollout {rid}); skipped", flush=True)
            if t_f is not None:
                fk_abs += sum(x["abs_logratio"] for x in parts[n_parts0:])
                fk_lr += sum(x["logratio"] for x in parts[n_parts0:])
                fk_ntok += sum(x["tokens"] for x in parts[n_parts0:])
                torch.cuda.synchronize(devices[0])
                fk_s += time.time() - t_f
                fk_n += 1
                fk_tok += len(chains[0]["ids"])
                fk_grad_tok += len(chains[0]["ids"])
        keys = STAT_KEYS
        loc = torch.tensor([sum(p[k] for p in parts) for k in keys] + [float(n_chains), float(n_oom), float(tok_fwd), float(fk_n), float(fk_s), float(fk_tok), float(fk_grad_tok), fk_abs, fk_lr, fk_ntok], device=grad_dev, dtype=torch.float64)
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
        # ---- the update. Everything the decision needs is identical on every rank (grads and loc are all-reduced), so every rank decides alike.
        v = loc.tolist()
        stats = merge_stat_sums([dict(zip(keys, v[: len(keys)]))])
        fkv = v[len(keys) + 3 : len(keys) + 10]  # forked completions: count, trainer seconds (summed over replicas), forwarded tokens, with-grad tokens
        gn, skip_reason, lr_now = 0.0, None, sched.lr_at()
        if not no_update:
            gn = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip, foreach=False)) if len(devices) == 1 else _clip_multi(params, a.grad_clip)  # pre-clip norm
            skip_reason = guard.check(gn, stats["kl"])
            if skip_reason is None:
                for g in opt.param_groups:
                    g["lr"] = lr_now
                opt.step()
                sched.applied += 1
            else:
                log(f"UPDATE SKIPPED at step {step}: {skip_reason}; republishing the previous adapter as v{step + 1}")
            guard.commit(gn, skipped=skip_reason is not None)
        opt.zero_grad(set_to_none=True)
        for d in devices:
            torch.cuda.synchronize(d)
        dt = time.time() - t0
        if no_update:
            log(f"step {step}: empty batch, no update")
        rec = {"step": step, "n_rollouts": len(items), "n_chains": int(v[len(keys)]), "n_oom": int(v[len(keys) + 1]), "tokens_fwd": int(v[len(keys) + 2]),
               "gnorm": gn, "lr": lr_now, "lr_applied": 0.0 if (no_update or skip_reason) else lr_now,  # lr = the schedule's value for this update; lr_applied is 0 when the update was skipped
               "train_s": round(dt, 1), "wait_s": m.get("wait_s"), "tok_per_s": round(v[len(keys) + 2] / dt),
               "peak_gib": round(max(torch.cuda.max_memory_allocated(d) for d in devices) / 2**30, 1), **{k: round(x, 6) for k, x in stats.items()},
               "updates_applied": sched.applied, "lr_mode": sched.mode, "guard_skipped": skip_reason is not None, "guard_reason": skip_reason, "guard_consecutive": guard.consecutive,
               "guard_skipped_total": guard.skipped_total, "guard_stuck": guard.stuck, "gn_avg": guard.gn_avg}
        if fkv[0] > 0:  # forked-report items in the step: per-fork trainer cost (cached vs full-sequence is the --fork-cached flag)
            rec["fork"] = {"n": int(fkv[0]), "cached": bool(use_cached), "s_per_fork": round(fkv[1] / fkv[0], 2), "tok_fwd": int(fkv[2]), "tok_grad": int(fkv[3]), "loss_tokens": int(fkv[6]), "abs_logratio": round(fkv[4] / max(fkv[6], 1), 5), "logratio": round(fkv[5] / max(fkv[6], 1), 5), "s_total_all_ranks": round(fkv[1], 1)}
        if rank == 0:
            t_save = time.time()
            save_adapter(pm, str(gd / "adapters" / f"v{step + 1}"), torch.bfloat16)
            ck = None
            ometa_now = {"step": step + 1, "applied": sched.applied, "guard": guard.state(), "lr": a.lr, "t": time.time()}
            if a.save_every and (step + 1) % a.save_every == 0:
                ck = f"ckpt/s{step + 1}"
                save_adapter(pm, str(gd / ck), torch.float32)
                save_state(opt, str(gd / ck / "optimizer.pt"), ometa_now)  # with each adapter checkpoint
            if a.resume_slot_every and (step + 1) % a.resume_slot_every == 0:
                publish_resume_slot(gd, lambda d: save_adapter(pm, d, torch.float32), opt, ometa_now)  # rolling slot: a dropped driver resumes from here
            rec.update(save_s=round(time.time() - t_save, 1), adapter=f"adapters/v{step + 1}", ckpt=ck)
            log("STEP " + json.dumps(rec))
            tmp = ctl / f"done_{step}.json.tmp"
            tmp.write_text(json.dumps(rec))
            tmp.rename(ctl / f"done_{step}.json")
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def verify_group(pm: Any, head: Any, ch: dict[str, Any], P: int, a: Any) -> float:
    """--fork-cached-verify: max |logp(cached tail) - logp(full forward)| over one fork's loss tokens (no grad; the GPU check of the attention kernels)."""
    from train import _unwrap  # /srv/train.py

    from wayfinder.grpo.data import label_positions
    from wayfinder.grpo.loss import token_logps

    ids = torch.from_numpy(np.asarray(ch["ids"], np.int64)).to(f"cuda:{torch.cuda.current_device()}")
    pos = torch.from_numpy(label_positions(ch["spans"], len(ids))).to(ids.device)
    lm = _unwrap(pm).model.language_model
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            from wayfinder.grpo.cached_fork import prefix_cache as _pc, tail_hidden as _th

            cache = _pc(lm, ids[:P])
            h_c = _th(lm, cache, ids[P:], P)
            h_f = forward_hidden(pm, ids)
        f_c = token_logps(head, h_c, pos - P, ids[pos], a.temperature, 2048, False, a.top_k, a.top_p)[1]
        f_f = token_logps(head, h_f, pos, ids[pos], a.temperature, 2048, False, a.top_k, a.top_p)[1]
    return float((f_c - f_f).abs().max())


def forward_hidden_autocast(pm: Any, ids: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return forward_hidden(pm, ids)


if __name__ == "__main__":
    sys.exit(main())
