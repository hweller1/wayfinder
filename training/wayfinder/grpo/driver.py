"""GRPO driver (runs inside Modal as `driver_remote` with --remote-driver, or on the laptop inside `modal run wayfinder/grpo/app.py`): schedules rollouts, turns them into advantages, feeds the
trainer, launches the periodic dev evals, keeps the heartbeat and the budget.

Pipeline with a ONE-STEP LAG. Adapter version v_j is the policy after j optimizer steps (v_0 = the start checkpoint). Rollouts of step j are
sampled with v_(j-1) (v_0 for steps 0 and 1) and are trained at step j, i.e. when the policy is already v_j: one step off-policy, which the
PPO ratio clip (against vLLM's behaviour logprobs) covers. Step j's rollouts are submitted as soon as v_(j-1) exists (the trainer finished
step j-2), so while the trainer works on step j-1 the workers already sample step j, and the trainer never idles waiting for rollouts unless
the workers are the bottleneck. Step 0 is on-policy.

Per step: collect the 32x8 rollouts (drop harness-level errors; a step is cut when >= 90% are in and `straggle_s` passed since), compute the
arm's reward, group advantages with PAPO on failures (advantage.py), drop zero-variance groups, publish the manifest. Everything is logged to
artifacts/grpo/<run>/steps.jsonl (one line per step) and rollout summaries to rollouts.jsonl.
"""

from __future__ import annotations

import json
import os
import queue as pyqueue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from wayfinder.grpo.advantage import LAMBDA, batch_advantages
from wayfinder.grpo import fork
from wayfinder.grpo import g4_link
from wayfinder.grpo import monitor as mon
from wayfinder.grpo.pool import ROOT, load_pool, sample_batch
from wayfinder.grpo.reward import rank5b_tiebreak, total_reward

H100_S, H200_S = 0.001097, 0.001261
RETRY_MAX_S = 600.0  # a transient control-plane error (laptop or Modal network blip) is retried this long before it ends the run


def remote_paths(a: dict, repo: Path, remote_root: str = "/root") -> dict:
    """Config plumbing for the cloud driver: a pool path under the local repo becomes the baked-in path under REMOTE_ROOT; the monitor's screen dir and
    the eval switches are fixed to their cloud values (no local `modal run` subprocess can run inside a container: evals are merged only and run by
    scripts/grpo/eval_ckpt.sh)."""
    a = dict(a)
    pool = a.get("pool") or ""
    if pool:
        pp = Path(pool)
        if pp.is_absolute() and pp.is_relative_to(repo):
            a["pool"] = f"{remote_root}/{pp.relative_to(repo)}"
    ft = a.get("fixed_tasks") or ""
    if ft:
        fp = Path(ft)
        if fp.is_absolute() and fp.is_relative_to(repo):
            a["fixed_tasks"] = f"{remote_root}/{fp.relative_to(repo)}"
    a["screen_dir"] = a.get("screen_dir") or f"{remote_root}/artifacts/rl_pool/screen"
    a["eval_every"] = 0
    return a


class Stopped(RuntimeError):
    """The driver was asked to stop (never retried)."""


def retry_transient(fn: Any, *args: Any, max_s: float = RETRY_MAX_S, base_s: float = 2.0, cap_s: float = 30.0, stop: threading.Event | None = None,
                    sleep: Any = time.sleep, clock: Any = time.time, label: str = "call", **kw: Any) -> Any:
    """Call fn(*args, **kw); on any exception except Stopped retry with exponential backoff (base_s, doubling, capped at cap_s) until max_s seconds
    have passed since the first failure, then re-raise the last error. A set `stop` event ends the retrying with Stopped."""
    t_first: float | None = None
    delay = base_s
    n = 0
    while True:
        try:
            return fn(*args, **kw)
        except Stopped:
            raise
        except Exception as e:  # noqa: BLE001
            now = clock()
            t_first = now if t_first is None else t_first
            n += 1
            if now - t_first >= max_s:
                print(f"[driver] {label}: giving up after {n} attempts over {now - t_first:.0f}s: {e!r}", flush=True)
                raise
            if stop is not None and stop.is_set():
                raise Stopped("stopped") from e
            print(f"[driver] {label}: transient error {e!r}; retry {n} in {delay:.0f}s", flush=True)
            sleep(delay)
            delay = min(delay * 2, cap_s)
GROUP = 8
EVAL_USD_EST = 10.0  # one dev eval (8 H100 containers, ~20 min) cost $9.8 at list price for arm a's step-20 checkpoint


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 5) if xs else None


def step_rollout_stats(rows: list[dict[str, Any]], items: list[dict[str, Any]], adv_stats: dict[str, Any]) -> dict[str, Any]:
    """Curves for one step over every valid rollout (zero-variance groups included)."""
    valid = [r for r in rows if not r.get("infra_error")]
    return {
        "n_rollouts": len(valid), "n_infra_errors": len(rows) - len(valid),
        "reward": mean([r["arm_reward"] for r in valid]), "recall_reported": mean([r["recall_reported"] for r in valid]),
        "all_gold": mean([float(r["all_gold"]) for r in valid]), "report_rate": mean([float(r["valid_report"]) for r in valid]),
        "turns": mean([r["turns"] for r in valid]), "repeats": mean([r["repeats"] for r in valid]),
        "precision": mean([r["precision"] for r in valid if r.get("precision") is not None]), "n_reported": mean([r["n_reported"] for r in valid]),
        "f1": mean([r.get("f1") for r in valid]), "recall_at_5": mean([r.get("recall_at_5") for r in valid]), "recall_at_7": mean([r.get("recall_at_7") for r in valid]),
        "all_gold_window": mean([float(r["all_gold_window"]) for r in valid if r.get("all_gold_window") is not None]), "file_cost": mean([r.get("file_cost") for r in valid]),
        "report_chars": mean([r["report_chars"] for r in valid]), "recall_touched": mean([r.get("recall_touched") for r in valid]),
        "tokens_in": mean([r.get("tokens_in") for r in valid]), "wall_s": mean([r.get("wall_s") for r in valid]),
        "chain_tokens": mean([r.get("n_tokens") for r in valid]), "loss_tokens_per_rollout": mean([r.get("n_loss") for r in valid]),
        "kept_rollouts": sum(1 for it in items if it["keep"]), **adv_stats,
    }


def apply_tiebreak(rows: list[dict[str, Any]]) -> None:
    """rank5b only: add the per-group tie-break (reward.rank5b_tiebreak) to arm_reward and store it in r["tiebreak"]."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if not r.get("infra_error") and r.get("reward_kind") == "rank5b":
            groups.setdefault(r["task_id"], []).append(r)
    for g in groups.values():
        for r, b in zip(g, rank5b_tiebreak(g)):
            r["tiebreak"] = round(b, 6)
            r["arm_reward"] = round(r["arm_reward"] + b, 6)


def reward_terms(r: dict[str, Any]) -> dict[str, float]:
    """Signed per-term reward contributions of one rollout (sum = reward, up to the precision term): rank5b stores them; rank5 is rebuilt from its fields."""
    if r.get("terms"):
        t = dict(r["terms"])
        t["tiebreak"] = r.get("tiebreak", 0.0)
        t["no_report"] = -r.get("no_report_penalty", 0.0)
        return t
    return {"r5": r.get("recall_at_5_capped") or 0.0, "r8": 0.25 * (r.get("recall_at_8") or 0.0), "ag": 0.5 * float(bool(r.get("all_gold_window"))),
            "turn": -(r.get("turn_penalty") or 0.0), "repeat": -(r.get("repeat_penalty") or 0.0), "no_report": -(r.get("no_report_penalty") or 0.0)}


N_BUCKETS = (("0-2", 0, 2), ("3-4", 3, 4), ("5", 5, 5), ("6-7", 6, 7), ("8-9", 8, 9), ("10+", 10, 10**6))


def step_diagnostics(rows: list[dict[str, Any]], std_floor: float = 0.1) -> dict[str, Any]:
    """Early-warning instrumentation of one step (every valid rollout): per-group reward std distribution; each reward term's share of the outcome advantage
    (mean |term advantage| over rollouts of groups with signal, as a fraction of the total over terms; term advantage = (term - group mean) / max(group std, floor), so
    the terms add up to the outcome advantage); recall@5 and counts by n_reported bucket; the n_reported histogram."""
    valid = [r for r in rows if not r.get("infra_error") and r.get("arm_reward") is not None]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in valid:
        groups.setdefault(r["task_id"], []).append(r)
    stds, share, ng = [], {}, 0
    for g in groups.values():
        rs = [x["arm_reward"] for x in g]
        m = sum(rs) / len(rs)
        sd = (sum((x - m) ** 2 for x in rs) / len(rs)) ** 0.5
        stds.append(sd)
        if len(g) < 2 or sd < 1e-3:
            continue
        ng += 1
        T = [reward_terms(x) for x in g]
        for k in {k for t in T for k in t}:
            xs = [t.get(k, 0.0) for t in T]
            mk = sum(xs) / len(xs)
            share[k] = share.get(k, 0.0) + sum(abs(x - mk) for x in xs) / len(xs) / max(sd, std_floor)
    tot = sum(share.values()) or 1.0
    ss = sorted(stds)
    q = lambda p: round(ss[min(len(ss) - 1, int(p * len(ss)))], 4) if ss else None  # noqa: E731
    buckets = {}
    for name, lo, hi in N_BUCKETS:
        sel = [x for x in valid if lo <= int(x["n_reported"]) <= hi]
        buckets[name] = {"n": len(sel), "recall_at_5": mean([x.get("recall_at_5") for x in sel])}
    hist: dict[str, int] = {}
    for x in valid:
        hist[str(int(x["n_reported"]))] = hist.get(str(int(x["n_reported"])), 0) + 1
    return {"group_std": {"n": len(stds), "mean": round(sum(stds) / len(stds), 4) if stds else None, "min": q(0), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "max": q(1.0),
                          "n_below_0.001": sum(1 for x in stds if x < 1e-3), "n_below_floor": sum(1 for x in stds if x < std_floor)},
            "adv_share": {k: round(v / tot, 4) for k, v in sorted(share.items())}, "n_groups_signal": ng, "recall_by_n_reported": buckets,
            "n_reported_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
            "tiebreak_mean": mean([x["tiebreak"] for x in valid if "tiebreak" in x])}


def build_batch(rows: list[dict[str, Any]], precision_coef: float, min_group: int = 4, lam: float = LAMBDA, fork_lambda: float | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """rows: rollout summaries of one step (from the workers). Returns (manifest, stats, adv_stats). Mutates rows (arm_reward, tiebreak)."""
    for r in rows:
        if not r.get("infra_error"):
            r["arm_reward"] = total_reward(r, precision_coef)
    apply_tiebreak(rows)
    usable = [r for r in rows if not r.get("infra_error") and r.get("path")]
    items = [{"rid": r["rid"], "task_id": r["task_id"], "reward": r["arm_reward"], "all_gold": r["all_gold"], "rubric": r.get("rubric"),
              "path": r["path"], "n_tokens": r["n_tokens"], "n_loss": r["n_loss"], "all_gold_window": r.get("all_gold_window"),
              "r5": r.get("recall_at_5_capped"), "r8": r.get("recall_at_8")} for r in usable]
    rb = bool(usable) and all(r.get("reward_kind") == "rank5b" for r in usable)
    adv_stats = batch_advantages(items, min_group=min_group, lam=lam, consistent_clip=rb)
    manifest = {"items": [{"rid": it["rid"], "path": it["path"], "n_tokens": it["n_tokens"], "n_loss": it["n_loss"], "adv": round(it["adv"], 6)} for it in items if it["keep"]]}
    if fork_lambda is not None:  # forked-report mode (--fork-report K > 0): intra-prefix report advantages; absent otherwise, so K=0 output is unchanged
        adv_stats["fork"] = fork.apply_forks(rows, manifest["items"], precision_coef, fork_lambda)
    stats = step_rollout_stats(rows, items, adv_stats)
    stats["diag"] = step_diagnostics(rows)
    if fork_lambda is not None:
        stats["diag"]["fork"] = fork.fork_diagnostics(rows)
    return manifest, stats, adv_stats


def fixed_batch(pool: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    """Calibration: the SAME tasks every step (file {"task_ids": [...]}, in file order); every id must be in the pool."""
    ids = json.loads(Path(path).read_text())["task_ids"]
    by = {t["task_id"]: t for t in pool}
    missing = [i for i in ids if i not in by]
    if missing:
        raise ValueError(f"fixed task list: {len(missing)} ids not in the pool, e.g. {missing[0]}")
    return [by[i] for i in ids]


# padding monitor: wayfinder/grpo/monitor.py (same-task v3 baseline, rebased 2026-10-28)


class Driver:
    def __init__(self, a: Any, ctl: Any, jobs: Any, results: Any, out: Path) -> None:
        self.a, self.ctl, self.jobs, self.results, self.out = a, ctl, jobs, results, out
        self.rows: dict[int, list[dict[str, Any]]] = {}
        self.submitted_at: dict[int, float] = {}
        self.cut: set[int] = set()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.pool_src: str | None = None
        self.pool: list[dict[str, Any]] = []
        self.pool_switches: list[dict[str, Any]] = []
        self.t0 = time.time()
        self.eval_procs: list[tuple[int, subprocess.Popen]] = []
        self.eval_cost = 0.0
        self.mon_flags: list[bool] = []
        self.screen: dict[str, list[dict[str, float | None]]] | None = None
        self.stop_reason: str | None = None
        self.on_step: Any = None  # optional hook called after each step's files are written (the cloud driver commits its volume there)
        self.merge_threads: list[threading.Thread] = []
        # optional G4 lane (--g4-worker, docs/grpo_g4_worker.md): everything below is inert while self.g4 is None
        self.g4: Any = None  # g4_link.G4Link, set by the entrypoint when --g4-worker is on
        self.g4_sub: dict[int, dict[str, list[int]]] = {}  # step -> {task_id: seeds} submitted to the G4
        self.g4_ver: dict[int, int] = {}
        self.g4_rows: dict[int, list[dict[str, Any]]] = {}
        self.g4_cut: set[int] = set()
        self.g4_late = 0  # G4 rows that arrived after their step was closed, since the last steps.jsonl line
        self.g4_skip: dict[int, str] = {}
        sj = out / "steps.jsonl"
        if getattr(a, "start_step", 0) > 0 and sj.exists():  # a resumed run keeps its flag streak (only the same-task rule's flags count)
            for ln in sj.read_text().splitlines():
                m = (json.loads(ln).get("monitor") or {})
                if "flagged_same_task" in m:
                    self.mon_flags.append(bool(m["flagged_same_task"]))

    # ---- pool (switches to rl_pool_v1/v0 the moment it is committed, recorded with the step)
    def refresh_pool(self, step: int) -> None:
        if self.a.pool:
            if not self.pool:
                self.pool, self.pool_src = load_pool(self.a.pool)
            return
        cands = [ROOT / "data/rl_pool_v1.json", ROOT / "data/rl_pool_v0.json"]
        want = next((str(p.relative_to(ROOT)) for p in cands if p.exists()), "prd14_rl_eligible_train")
        if want != self.pool_src and not self.a.no_pool_switch:
            try:
                pool, src = load_pool(None)
            except Exception as e:  # noqa: BLE001  (a half-written or invalid pool file must not kill the run: keep the current one)
                print(f"[driver] pool refresh failed, keeping {self.pool_src}: {e!r}", flush=True)
                if not self.pool:
                    raise
                return
            if src != self.pool_src:
                self.pool_switches.append({"step": step, "from": self.pool_src, "to": src, "n_tasks": len(pool)})
                print(f"[driver] pool switch at step {step}: {self.pool_src} -> {src} ({len(pool)} tasks)", flush=True)
                self.pool, self.pool_src = pool, src
        elif not self.pool:
            self.pool, self.pool_src = load_pool(None)

    def submit(self, step: int, version: int) -> None:
        self.refresh_pool(step)
        tasks = fixed_batch(self.pool, self.a.fixed_tasks) if getattr(self.a, "fixed_tasks", "") else sample_batch(self.pool, step, self.a.tasks_per_step, self.a.seed)
        base_seed = 100000 * (1 + step) + 17 * self.a.seed_salt
        for t in tasks:
            retry_transient(self.jobs.put, {"step": step, "version": version, "task": t, "seeds": [base_seed + s for s in range(self.a.group)]}, stop=self.stop, label="jobs.put")
        self.submitted_at[step] = time.time()
        self.rows.setdefault(step, [])
        (self.out / f"tasks_step{step}.json").write_text(json.dumps([t["task_id"] for t in tasks]))
        if self.g4 is not None:
            self.submit_g4(step, version, tasks)
        print(f"[driver] submitted step {step} ({len(tasks)} tasks x {self.a.group}) with adapter v{version}; pool={self.pool_src}", flush=True)

    def submit_g4(self, step: int, version: int, modal_tasks: list[dict[str, Any]]) -> None:
        """Extra tasks for the G4 pod (never fatal: a failure here costs the extra rollouts, not the step)."""
        a = self.a
        try:
            alive, hb = self.g4.alive()
            if not alive:
                self.g4_skip[step] = "no G4 heartbeat"
                print(f"[driver] step {step}: G4 pod not alive (heartbeat {hb and hb.get('t')}); no G4 tasks", flush=True)
                return
            tasks = g4_link.sample_g4_tasks(self.pool, step, a.g4_tasks, a.seed, {t["task_id"] for t in modal_tasks})
            base_seed = 100000 * (1 + step) + 17 * a.seed_salt
            seeds = [base_seed + s for s in range(a.group)]
            expires = time.time() + a.step_timeout_s
            for t in tasks:
                self.g4.submit(g4_link.make_job(step, version, t, seeds, a.g4_cfg, expires))
            self.g4_sub[step] = {t["task_id"]: seeds for t in tasks}
            self.g4_ver[step] = version
            (self.out / f"tasks_step{step}_g4.json").write_text(json.dumps([t["task_id"] for t in tasks]))
            print(f"[driver] submitted step {step} G4 lane ({len(tasks)} tasks x {a.group}) with adapter v{version}", flush=True)
        except Exception as e:  # noqa: BLE001
            self.g4_skip[step] = f"submit error {type(e).__name__}"
            print(f"[driver] step {step}: G4 submit failed, Modal lane only: {e!r}", flush=True)

    def drain_g4(self) -> None:
        """Background: G4 summaries into per-step lists; a row for a step that is already closed is late and is dropped (never used off-policy)."""
        with (self.out / "g4_late.jsonl").open("a") as late_f:
            while not self.stop.is_set():
                try:
                    got = self.g4.results.get_many(64, block=True, timeout=5)
                except pyqueue.Empty:
                    continue
                except Exception as e:  # noqa: BLE001
                    print(f"[driver] g4 results queue error {e!r}", flush=True)
                    time.sleep(5)
                    continue
                with self.lock:
                    for r in got:
                        if r.get("step") in self.g4_cut:
                            self.g4_late += 1
                            late_f.write(json.dumps({k: v for k, v in r.items() if k != "path"}) + "\n")
                        else:
                            self.g4_rows.setdefault(r["step"], []).append(r)
                late_f.flush()

    def collect_g4(self, step: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Close the G4 lane of `step` (after its Modal rows are in): wait up to g4_grace_s for the outstanding tasks, then take what is complete and current. Late rows
        after this point are dropped by drain_g4."""
        sub = self.g4_sub.get(step)
        if self.g4 is None or not sub:
            return [], ({"skipped": self.g4_skip[step]} if step in self.g4_skip else None)
        n_need = sum(len(v) for v in sub.values())
        deadline = min(time.time() + self.a.g4_grace_s, self.submitted_at[step] + self.a.step_timeout_s)
        while time.time() < deadline:
            with self.lock:
                n = len(self.g4_rows.get(step, []))
            if n >= n_need:
                break
            if self.stop.is_set():
                raise Stopped("stopped")
            time.sleep(2)
        with self.lock:
            self.g4_cut.add(step)
            got = self.g4_rows.pop(step, [])
        try:
            retry_transient(self.g4.mark_cut, step, max_s=60, stop=self.stop, label="g4 cut")
        except Exception as e:  # noqa: BLE001
            print(f"[driver] g4 cut flag failed: {e!r}", flush=True)
        used, info = g4_link.merge_g4(got, sub, self.g4_ver[step])
        with (self.out / "rollouts.jsonl").open("a") as f:
            for r in used:
                f.write(json.dumps({k: v for k, v in r.items() if k != "path"}) + "\n")
        if info["g4_dropped"]:
            print(f"[driver] step {step}: G4 lane dropped {info['g4_dropped']}/{info['tasks_submitted']} tasks {info['drop_reasons']}", flush=True)
        return used, info

    def drain(self) -> None:
        """Background: move worker summaries from the results queue into per-step lists."""
        with (self.out / "rollouts.jsonl").open("a") as f:
            while not self.stop.is_set():
                try:
                    got = self.results.get_many(64, block=True, timeout=5)
                except pyqueue.Empty:
                    continue
                except Exception as e:  # noqa: BLE001
                    print(f"[driver] results queue error {e!r}", flush=True)
                    time.sleep(5)
                    continue
                with self.lock:
                    for r in got:
                        if r["step"] in self.cut:
                            continue
                        self.rows.setdefault(r["step"], []).append(r)
                        slim = {k: v for k, v in r.items() if k not in ("path",)}
                        f.write(json.dumps(slim) + "\n")
                f.flush()

    def heartbeat(self) -> None:
        while not self.stop.is_set():
            try:
                self.ctl.put("heartbeat", time.time())
            except Exception as e:  # noqa: BLE001
                print(f"[driver] heartbeat error {e!r}", flush=True)
            self.stop.wait(60)

    def wait_step(self, step: int) -> list[dict[str, Any]]:
        n_expected = self.a.tasks_per_step * self.a.group
        t90: float | None = None
        while True:
            with self.lock:
                n = len(self.rows.get(step, []))
            if n >= n_expected:
                break
            if n >= 0.9 * n_expected:
                t90 = t90 or time.time()
                if time.time() - t90 > self.a.straggle_s:
                    print(f"[driver] step {step}: cutting stragglers at {n}/{n_expected}", flush=True)
                    break
            if time.time() - self.submitted_at[step] > self.a.step_timeout_s and n >= 0.5 * n_expected:
                print(f"[driver] step {step}: timeout, cutting at {n}/{n_expected}", flush=True)
                break
            if self.stop.is_set():
                raise Stopped("stopped")
            time.sleep(5)
        with self.lock:
            self.cut.add(step)
            rows = list(self.rows[step])
        retry_transient(self.ctl.put, "cut_step", max(self.cut), stop=self.stop, label="ctl.put cut_step")
        return rows

    def train_usd(self) -> float:
        """Rollout + trainer GPU time at list price (what the training budget caps); evals are tracked in eval_cost."""
        return (time.time() - self.t0) * (self.a.workers * H100_S + self.a.train_gpus * H200_S)

    def est_usd(self) -> float:
        return self.train_usd() + self.eval_cost

    def wait_done(self, step: int, timeout: float = 4 * 3600) -> dict[str, Any]:
        t = time.time()
        while time.time() - t < timeout:
            rec = retry_transient(self.ctl.get, f"done_{step}", stop=self.stop, label=f"ctl.get done_{step}")
            if rec is not None:
                return rec
            err = retry_transient(self.ctl.get, "trainer_error", stop=self.stop, label="ctl.get trainer_error")
            if err and time.time() - err["t"] < 120:
                print(f"[driver] trainer reports error: {err}", flush=True)
            if self.stop.is_set():
                raise Stopped("stopped")
            time.sleep(5)
        raise TimeoutError(f"trainer step {step}")

    def launch_eval(self, step: int, merge_fn: Any) -> None:
        def go() -> None:
            try:
                res = merge_fn(self.a.start, self.a.run, step)
                print(f"[driver] merged s{step}: {res}", flush=True)
                self.eval_cost += 0.3
                variant = f"grpo-{self.a.run}-s{step}"
                weights = f"/data/grpo/{self.a.run}/s{step}_merged"
                procs = []
                for sh in range(self.a.eval_shards):
                    od = ROOT / "artifacts" / "grpo" / self.a.run / f"eval_s{step}" / f"shard{sh}"
                    od.mkdir(parents=True, exist_ok=True)
                    env = {**os.environ, "WF_EVAL_VARIANT": variant, "WF_EVAL_WEIGHTS": weights, "WF_EVAL_APP": f"wayfinder-grpo-eval-{self.a.run}-s{step}-{sh}", "WF_EVAL_RETAIN": "1",
                           "WF_EVAL_NUDGE": getattr(self.a, "nudge", "teacher")}
                    cmd = ["modal", "run", str(ROOT / "wayfinder/rollout/sft_eval_app.py"), "--tasks", str(ROOT / f"artifacts/grpo/dev_shard{sh}.json"), "--outdir", str(od),
                           "--seeds", "0,1", "--concurrency", "14", "--max-seconds", "3600", "--arms", "grep,sem"]
                    procs.append(subprocess.Popen(cmd, env=env, stdout=open(od / "run.log", "w"), stderr=subprocess.STDOUT, cwd=str(ROOT)))
                for p in procs:
                    p.wait()
                metas = [json.loads(m.read_text()).get("gpu_cost_estimate_usd") or 0.0 for m in (ROOT / "artifacts" / "grpo" / self.a.run / f"eval_s{step}").glob("shard*/*.meta.json")]
                self.eval_cost += sum(metas) if metas else self.a.eval_shards * 2 * 0.3 * 3600 * H100_S  # container seconds at list price (measured; ~$10 per eval)
                print(f"[driver] eval s{step} finished rc={[p.returncode for p in procs]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[driver] eval s{step} failed: {e!r}", flush=True)

        th = threading.Thread(target=go, daemon=True)
        th.start()
        self.eval_procs.append((step, th))  # type: ignore[arg-type]

    def launch_merge(self, step: int, merge_fn: Any) -> None:
        """Cloud driver: merge ckpt/s<step> into /data/grpo/<run>/s<step>_merged in a background thread and write the marker merged_s<step>.json
        (the dev eval is then run from the laptop: scripts/grpo/eval_ckpt.sh <run> <step>)."""
        def go() -> None:
            try:
                res = retry_transient(merge_fn, self.a.start, self.a.run, step, max_s=120, label=f"merge s{step}")
                self.eval_cost += 0.3
                (self.out / f"merged_s{step}.json").write_text(json.dumps({"step": step, "t": time.time(), "weights": f"/data/grpo/{self.a.run}/s{step}_merged", "merge": res}))
                print(f"[driver] merged s{step}: {res}", flush=True)
                if self.on_step:
                    self.on_step(None)
            except Exception as e:  # noqa: BLE001
                print(f"[driver] merge s{step} failed: {e!r}", flush=True)

        th = threading.Thread(target=go, daemon=True)
        th.start()
        self.merge_threads.append(th)

    def submitter(self) -> None:
        """Step j (j >= 2) samples with v_(j-1), which exists once train(j-2) is done: submit it then, so the workers always have queued work and
        sample step j while the trainer is on step j-1 (the one-step lag)."""
        a = self.a
        for j in range(a.start_step + 2, a.steps):
            while True:
                try:
                    self.wait_done(j - 2)
                    if self.stop.is_set():
                        return
                    self.submit(j, j - 1)
                    break
                except Stopped:
                    return
                except Exception as e:  # noqa: BLE001  (retry_transient already waited out ~10 min: keep trying rather than end the run; the watcher/budget bound the cost)
                    if self.stop.is_set():
                        return
                    print(f"[driver] submitter for step {j} hit {e!r}; retrying in 30s", flush=True)
                    time.sleep(30)
                    if j in self.submitted_at:  # submit() got as far as recording the step: do not double-submit
                        break

    def run(self, merge_fn: Any) -> None:
        a = self.a
        self.out.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self.drain, daemon=True).start()
        threading.Thread(target=self.heartbeat, daemon=True).start()
        if self.g4 is not None:
            threading.Thread(target=self.drain_g4, daemon=True).start()
        first = a.start_step
        v0 = max(first - 1, 0)
        self.submit(first, v0)  # step j samples with v_(j-1); a fresh run (first = 0) samples steps 0 and 1 with the start checkpoint
        if first + 1 < a.steps:
            self.submit(first + 1, first if first > 0 else 0)
        threading.Thread(target=self.submitter, daemon=True).start()
        steps_f = (self.out / "steps.jsonl").open("a")
        for k in range(first, a.steps):
            rows = self.wait_step(k)
            g4_used, g4_info = self.collect_g4(k) if self.g4 is not None else ([], None)
            rows = rows + g4_used  # whole groups only (merge_g4): a task never straddles the two lanes
            manifest, stats, adv_stats = build_batch(rows, a.precision_coef, lam=getattr(a, "papo_lambda", LAMBDA),
                                                   fork_lambda=getattr(a, "lambda_fork", 1.0) if getattr(a, "fork_report", 0) > 0 else None)
            retry_transient(self.ctl.put, f"batch_{k}", manifest, stop=self.stop, label=f"ctl.put batch_{k}")
            print(f"[driver] step {k}: published {len(manifest['items'])} rollouts; reward {stats['reward']} recall {stats['recall_reported']} "
                  f"all_gold {stats['all_gold']} report {stats['report_rate']} turns {stats['turns']} kept_groups {adv_stats['kept_groups']}/{adv_stats['groups']}", flush=True)
            rec = self.wait_done(k)
            mo = self.monitor_step(k, stats, rows)
            line = {"step": k, "version_sampled": rows[0]["version"] if rows else None, "t_elapsed_s": round(time.time() - self.t0), "est_usd": round(self.est_usd(), 2),
                    "train_usd": round(self.train_usd(), 2), "eval_usd": round(self.eval_cost, 2), "pool": self.pool_src, "rollout": stats, "train": rec, "monitor": mo}
            if self.g4 is not None:  # keys exist only with --g4-worker: a flag-off steps.jsonl is unchanged
                with self.lock:
                    late, self.g4_late = self.g4_late, 0
                line["g4"] = {**(g4_info or {}), "late_rows_since_last_step": late}
                line["g4_tasks_used"] = (g4_info or {}).get("g4_tasks_used", 0)
                line["g4_dropped"] = (g4_info or {}).get("g4_dropped", 0)
            steps_f.write(json.dumps(line) + "\n")
            steps_f.flush()
            print(f"[driver] step {k} done: kl {rec.get('kl')} clip {rec.get('clip_frac')} gnorm {rec.get('gnorm')} train_s {rec.get('train_s')} est ${self.est_usd():.0f}", flush=True)
            (self.out / "state.json").write_text(json.dumps({"next_step": k + 1, "pool_switches": self.pool_switches, "est_usd": self.est_usd(), "train_usd": self.train_usd(),
                                                              "eval_usd": self.eval_cost, "stop_reason": self.stop_reason}))
            if self.on_step:
                self.on_step(k)
            if a.eval_every and (k + 1) % a.eval_every == 0:
                cap = getattr(a, "eval_budget_usd", None)
                if cap is not None and self.eval_cost + EVAL_USD_EST > cap:
                    print(f"[driver] eval budget ${cap} would be exceeded (spent ${self.eval_cost:.1f} + ~${EVAL_USD_EST}); skipping the s{k + 1} eval (checkpoint kept)", flush=True)
                else:
                    self.launch_eval(k + 1, merge_fn)
            elif (k + 1) in (getattr(a, "merge_steps", None) or ()):
                self.launch_merge(k + 1, merge_fn)
            if mo and mo.get("stop"):
                self.stop_reason = "padding_monitor"
                print(f"[driver] STOP RULE FIRED after step {k}: {mo}", flush=True)
                break
            if rec.get("guard_stuck"):
                self.stop_reason = "update_guard_stuck"
                print(f"[driver] update guard skipped {rec.get('guard_consecutive')} updates in a row; stopping after step {k}", flush=True)
                break
            if self.train_usd() > a.budget_usd:
                self.stop_reason = "budget"
                print(f"[driver] training budget ${a.budget_usd} reached (est ${self.train_usd():.0f}); stopping after step {k}", flush=True)
                break
        retry_transient(self.ctl.put, "stop", True, label="ctl.put stop")
        sp = self.out / "state.json"
        sp.write_text(json.dumps({**(json.loads(sp.read_text()) if sp.exists() else {}), "stop_reason": self.stop_reason or "completed"}))
        if self.on_step:
            self.on_step(None)

    def monitor_step(self, k: int, stats: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Log the padding monitor for step k (n_reported, precision, recall@5, recall@7 on its training rollouts) and, with --monitor, compare with v3's own
        screen rollouts on the same tasks (monitor.py) and apply the stop rule."""
        a = self.a
        if any(stats.get(m) is None for m in ("n_reported", "precision", "recall_at_5", "recall_at_7")):
            return None
        out: dict[str, Any] = {m: round(stats[m], 4) for m in ("n_reported", "precision", "recall_at_5", "recall_at_7")}
        if not getattr(a, "monitor", True):
            return out
        gold = {t["task_id"]: set(t["gold_files"]) for t in self.pool}
        if self.screen is None:
            self.screen = mon.load_screen(getattr(a, "screen_dir", None) or ROOT / "artifacts/rl_pool/screen", gold)
            print(f"[driver] monitor baseline: v3 screen rollouts for {len(self.screen)} tasks", flush=True)
        cmp = mon.step_vs_screen([r for r in rows if not r.get("infra_error")], self.screen, gold)
        if cmp is None:
            return out | {"same_task": None}
        self.mon_flags.append(cmp["flagged_same_task"])
        d = cmp["delta"]
        out.update(cmp, stop=mon.should_stop(self.mon_flags))
        print(f"[driver] monitor step {k} (vs v3 same {cmp['n_tasks']} tasks): n_reported {cmp['cur']['n_reported']:.2f} ({d['n_reported']:+.2f}) precision {cmp['cur']['precision']:.3f} "
              f"({d['precision']:+.3f}) recall@5 {cmp['cur']['recall_at_5']:.3f} ({d['recall_at_5']:+.3f}) flagged={cmp['flagged_same_task']}", flush=True)
        return out

    def finish(self) -> None:
        for _, th in self.eval_procs:
            th.join(timeout=3 * 3600)  # type: ignore[attr-defined]
        for th in self.merge_threads:
            th.join(timeout=3600)
        self.stop.set()
