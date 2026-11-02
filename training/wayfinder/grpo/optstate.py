"""Optimizer-state persistence, the lr schedule with its re-warm, and the per-update guard. Pure torch + stdlib (CPU-testable).

Why (the GRPO write-up, not in this release, run f1): arm b diverged after a resume because the adapter came back but AdamW's moments did not. A fresh AdamW's first
steps move every parameter by about lr (bias-corrected m/sqrt(v) is +-1), which on a policy that already sits at the edge of the trust region
is a large unplanned jump (KL 966 in arm a's last logged step). So:

  * `save_state` / `load_state` write and read the AdamW state (+ the trainer counters) next to every adapter checkpoint (`ckpt/s<k>/optimizer.pt`)
    and in a rolling `resume/` slot every step (fp32 adapter + optimizer.pt), written to a temp dir and renamed so a kill mid-write leaves the old slot.
  * `LrSchedule`: a fresh run warms up over `warmup` updates (lr * (u + 1) / warmup, the old behaviour); a resume WITH optimizer state continues at
    full lr; a resume WITHOUT it re-warms from 0 over `rewarm` (10) updates (lr * u / rewarm, so the first update has lr 0 and only fills the moments).
  * `UpdateGuard`: skips an update (no optimizer step; the unchanged adapter is republished as the next version) when the pre-clip grad norm is
    more than `gn_mult` (5) times its moving average over the last `window` accepted updates, or the step's KL to the reference is above `kl_max`
    (0.1), or the grad norm is not finite. Skipped updates do not enter the moving average or advance the lr schedule.
"""

from __future__ import annotations

import math
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import torch


class LrSchedule:
    def __init__(self, lr: float, warmup: int = 2, rewarm: int = 10, resumed: bool = False, has_opt_state: bool = False) -> None:
        self.lr, self.warmup, self.rewarm = lr, max(1, warmup), max(1, rewarm)
        self.mode = "fresh" if not resumed else ("resumed" if has_opt_state else "rewarm")
        self.applied = 0  # optimizer updates applied by this process's schedule (carried across resumes only together with the optimizer state)

    def lr_at(self, applied: int | None = None) -> float:
        u = self.applied if applied is None else applied
        if self.mode == "fresh":
            return self.lr * min(1.0, (u + 1) / self.warmup)
        if self.mode == "rewarm":
            return self.lr * min(1.0, u / self.rewarm)
        return self.lr


class UpdateGuard:
    def __init__(self, gn_mult: float = 5.0, kl_max: float = 0.1, window: int = 10, min_history: int = 3, stuck_after: int = 3) -> None:
        self.gn_mult, self.kl_max, self.min_history, self.stuck_after = gn_mult, kl_max, min_history, stuck_after
        self.hist: deque[float] = deque(maxlen=window)
        self.consecutive = 0
        self.skipped_total = 0

    @property
    def gn_avg(self) -> float | None:
        return sum(self.hist) / len(self.hist) if self.hist else None

    def check(self, gn: float, kl: float) -> str | None:
        """Reason to skip this update, or None. Does not mutate."""
        if not math.isfinite(gn):
            return "nonfinite_grad_norm"
        if not math.isfinite(kl):
            return "nonfinite_kl"
        if kl > self.kl_max:
            return f"kl {kl:.4g} > {self.kl_max}"
        avg = self.gn_avg
        if avg is not None and len(self.hist) >= self.min_history and gn > self.gn_mult * avg:
            return f"grad_norm {gn:.4g} > {self.gn_mult:g} x moving average {avg:.4g}"
        return None

    def commit(self, gn: float, skipped: bool) -> None:
        if skipped:
            self.consecutive += 1
            self.skipped_total += 1
        else:
            self.consecutive = 0
            self.hist.append(gn)

    @property
    def stuck(self) -> bool:
        return self.consecutive >= self.stuck_after

    def state(self) -> dict[str, Any]:
        return {"hist": list(self.hist), "skipped_total": self.skipped_total}

    def load(self, d: dict[str, Any]) -> None:
        self.hist.extend(d.get("hist", []))
        self.skipped_total = int(d.get("skipped_total", 0))


def save_state(opt: torch.optim.Optimizer, path: str | Path, meta: dict[str, Any]) -> None:
    """`optimizer.pt` = {opt: state_dict (tensors moved to CPU), meta: {step, applied, guard, ...}}; written atomically (tmp + rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sd = opt.state_dict()
    cpu = {"state": {k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} for k, v in sd["state"].items()}, "param_groups": sd["param_groups"]}
    tmp = p.with_suffix(".tmp")
    torch.save({"opt": cpu, "meta": meta}, tmp)
    os.replace(tmp, p)


def load_state(opt: torch.optim.Optimizer, path: str | Path) -> dict[str, Any] | None:
    """Restore the optimizer in place (keeps the current lr in the param groups; the schedule sets it each step). None when the file is absent
    or does not match the optimizer's parameters (then the caller re-warms)."""
    p = Path(path)
    if not p.exists():
        return None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    try:
        opt.load_state_dict(blob["opt"])
    except (ValueError, KeyError, RuntimeError):
        return None
    return blob["meta"]


def publish_resume_slot(gd: Path, write_adapter: Any, opt: torch.optim.Optimizer, meta: dict[str, Any]) -> None:
    """Rolling `resume/` slot under the run dir: fp32 adapter + optimizer.pt, built in `resume.tmp` and swapped in so a crash leaves the previous slot."""
    tmp, dst = gd / "resume.tmp", gd / "resume"
    shutil.rmtree(tmp, ignore_errors=True)
    write_adapter(str(tmp))
    save_state(opt, tmp / "optimizer.pt", meta)
    shutil.rmtree(gd / "resume.old", ignore_errors=True)
    if dst.exists():
        os.replace(dst, gd / "resume.old")
    os.replace(tmp, dst)
    shutil.rmtree(gd / "resume.old", ignore_errors=True)
