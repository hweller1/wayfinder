"""GRPO checkpoint eval on the borrowed GKE G4 (RTX Pro 6000), same protocol as sft_eval_app.py as called by scripts/grpo/eval_ckpt.sh and eval_ood.sh.

vLLM (sibling container, deploy/g4/screen.yaml) serves v3 with --enable-lora; this worker hot-loads the fp32 GRPO adapter /data/adapters/<run>/s<step>
under the name `wayfinder-grpo-<run>-s<step>` (the model name every Modal row carries) and runs the rollouts. Differences from the Modal eval, all
timing-only: one GPU instead of 4 shards x 2 arms on 8 H100s, so both arms run at once with `--concurrency` each (default 30 + 30 <= max-num-seqs 64),
and the policy is base + runtime LoRA instead of the bf16-merged weights. Raw rows (not compacted) are written to the same layout the Modal evals use:
<outroot>/<shard dir>/sft_grpo-<run>-s<step>_{grep,sem}.jsonl, one shard dir per task-shard file (tasks keep their dev_shard/ood40_shard membership).

    python -m wayfinder.rollout.g4_eval --run r2_100_hi --step 10 --kind dev|ood --adapter /data/adapters/r2_100_hi/s10 --tasks-dir /data/out/tasks --outroot /data/out/r2_100_hi
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path

from wayfinder.rollout.nudges import NUDGES

PORT = "http://127.0.0.1:8000"
WEIGHTS_FMT = "/data/grpo/{run}/s{step}_merged"  # the string Modal rows carry in `weights`
LAUNCH_DEADLINE_S = 140 * 60  # as sft_eval_app (v2+)


def _log(m: str) -> None:
    print(f"[g4_eval {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _post(path: str, body: dict, timeout: float = 600.0) -> tuple[int, str]:
    import urllib.error

    req = urllib.request.Request(PORT + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]


def _lp(model: str) -> list[float]:
    c, t = _post("/v1/chat/completions", {"model": model, "messages": [{"role": "system", "content": "You are a code search agent."}, {"role": "user", "content": "Where is request routing implemented in a typical web framework? Think step by step."}],
                                           "max_tokens": 24, "temperature": 0, "logprobs": True}, timeout=300)
    return [x["logprob"] for x in json.loads(t)["choices"][0]["logprobs"]["content"]] if c == 200 and t.startswith("{") else []


def _metrics() -> dict[str, float]:
    import re

    out: dict[str, float] = {}
    try:
        text = urllib.request.urlopen(PORT + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return out
    for name in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:num_preemptions_total"):
        m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)", text, re.M)
        if m:
            out[name] = float(m.group(1))
    return out


async def main_async(a: argparse.Namespace) -> None:
    from pydantic import Field
    from pydantic.fields import FieldInfo
    from scout.core.contracts.types import Budget, ExploreRequest

    from wayfinder.rollout.chat import CompatChat, Prices, Sampling
    from wayfinder.rollout.run import checkout, run_rollout
    from wayfinder.rollout.score import score_row

    name = f"wayfinder-grpo-{a.run}-s{a.step}"
    weights = WEIGHTS_FMT.format(run=a.run, step=a.step)
    nudge_text = NUDGES[a.nudge]
    cfg = {"concurrency": a.concurrency, "temperature": 0.85, "top_p": 0.95, "max_turns": 40, "max_seconds": a.max_seconds, "nudge": True, "prices": [1.25, 4.25, 0.15]}
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = a.arms.split(",")
    shards = sorted(Path(a.tasks_dir).glob(f"{a.kind_prefix}_shard*.json"))
    assert shards, f"no {a.kind_prefix}_shard*.json in {a.tasks_dir}"
    outroot = Path(a.outroot)
    tasks: list[tuple[Path, dict]] = []
    for sf in shards:
        od = outroot / sf.stem.split("_")[-1]  # shard0..3
        od.mkdir(parents=True, exist_ok=True)
        tasks += [(od, t) for t in json.loads(sf.read_text())]
    done: set[tuple[str, int, str]] = set()
    for od, _ in {(od, 0) for od, _ in tasks}:
        for arm in arms:
            fp = od / f"sft_grpo-{a.run}-s{a.step}_{arm}.jsonl"
            if fp.exists():
                for ln in fp.read_text().splitlines():
                    if ln.strip():
                        r = json.loads(ln)
                        if not r.get("error"):
                            done.add((r["task_id"], r["seed"], arm))
    todo = [(od, t, s, arm) for arm in arms for s in seeds for od, t in tasks if (t["task_id"], s, arm) not in done]
    _log(f"{a.kind_prefix} {name}: {len(todo)} rollouts to run, {len(done)} done")
    if not todo:
        return

    for _ in range(600):
        try:
            with urllib.request.urlopen(PORT + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    t_start = time.time()
    meta: dict = {"cfg": cfg, "gpu": "RTX Pro 6000 (GKE G4)", "variant": f"grpo-{a.run}-s{a.step}", "retain_reasoning": True, "weights": weights, "served_via": f"runtime LoRA {a.adapter} on v3_merged",
                  "nudge": nudge_text, "nudge_key": a.nudge, "max_model_len": 65536}
    # load the adapter (retry: first load can race the previous unload), then prove it is live: logprobs differ from the base model
    names = [m["id"] for m in json.loads(urllib.request.urlopen(PORT + "/v1/models", timeout=10).read())["data"]]
    if name not in names:
        for old in [n for n in names if n.startswith("wayfinder-grpo-")]:
            _post("/v1/unload_lora_adapter", {"lora_name": old})
        c, t = _post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": a.adapter})
        if c != 200:
            raise SystemExit(f"load_lora_adapter {name}: {c} {t}")
    base = (names + [""])[0] if "wayfinder-v3" not in names else "wayfinder-v3"
    la, lb = await asyncio.to_thread(_lp, "wayfinder-v3"), await asyncio.to_thread(_lp, name)
    meta["lora_l1_vs_base"] = round(sum(abs(x - y) for x, y in zip(la, lb)), 5) if la and lb else None
    _log(f"adapter {name} loaded, L1 vs base {meta['lora_l1_vs_base']}")
    if not meta["lora_l1_vs_base"]:
        raise SystemExit("adapter check failed (not live)")

    if cfg["max_seconds"] > 900:
        Budget.model_fields["max_seconds"] = FieldInfo.from_annotated_attribute(int, Field(default=120, ge=5, le=100000))
        Budget.model_rebuild(force=True)
        ExploreRequest.model_rebuild(force=True)
    clones: dict[str, asyncio.Future] = {}
    queue: asyncio.Queue = asyncio.Queue()
    sems = {arm: asyncio.Semaphore(cfg["concurrency"]) for arm in arms}

    async def get_root(t: dict) -> tuple[str, float]:
        key = f"{t['repo']}@{t['base_commit']}"
        if key not in clones:
            clones[key] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], a.repos))
        return await clones[key]

    async def one(od: Path, t: dict, seed: int, arm: str) -> None:
        async with sems[arm]:
            if time.time() - t_start > LAUNCH_DEADLINE_S:
                await queue.put((od, {"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "arm": arm, "error": "not launched: function deadline", "infra_error": True}))
                return
            try:
                root, clone_s = await get_root(t)
            except Exception as e:  # noqa: BLE001
                await queue.put((od, {"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "arm": arm, "error": f"checkout: {e}"[:300], "infra_error": True}))
                return
            chat = CompatChat(PORT + "/v1", "x", name, sampling=Sampling(cfg["temperature"], cfg["top_p"]), prices=Prices(*cfg["prices"]), seed=seed, vllm_extensions=False, timeout=600.0, max_retries=4)
            chat.retain_reasoning = True
            try:
                row = await asyncio.wait_for(run_rollout(t, chat, root, seed=seed, max_turns=cfg["max_turns"], max_seconds=cfg["max_seconds"], nudge=nudge_text,
                                                         semantic="off" if arm == "grep" else "auto"), timeout=cfg["max_seconds"] + 600)
                row.update(score_row(row))
                row["arm"] = arm
                row["clone_s"] = round(clone_s, 1)
                row["temperature"], row["top_p"] = cfg["temperature"], cfg["top_p"]
                row["max_model_len"] = 65536
                row["retain_reasoning"] = True
                row["weights"] = weights
                row["g4"] = True
            except Exception as e:  # noqa: BLE001
                row = {"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "arm": arm, "error": f"harness: {type(e).__name__}: {e}"[:300], "infra_error": True}
            await queue.put((od, row))

    async def _tele() -> None:
        while True:
            await asyncio.sleep(60)
            _log(f"telemetry done={n_done}/{len(todo)} elapsed_min={(time.time() - t_start) / 60:.1f} {_metrics()}")

    n_done = 0
    tele = asyncio.ensure_future(_tele())
    jobs = [asyncio.ensure_future(one(od, t, s, arm)) for od, t, s, arm in todo]
    try:
        for _ in jobs:
            od, row = await queue.get()
            with (od / f"sft_grpo-{a.run}-s{a.step}_{row['arm']}.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            n_done += 1
            if n_done % 10 == 0 or row.get("error"):
                _log(f"[{n_done}/{len(todo)}] {row['arm']} {row['task_id'][-30:]} s{row['seed']} turns={row.get('turns')} wall={row.get('wall_s')} recall={row.get('recall')} err={row.get('error')}")
    finally:
        tele.cancel()
        for j in jobs:
            j.cancel()
        meta["vllm_metrics"] = _metrics()
        meta["elapsed_s"] = round(time.time() - t_start, 1)
        for od in {od for od, _ in tasks}:
            for arm in arms:
                (od / f"sft_grpo-{a.run}-s{a.step}_{arm}.meta.json").write_text(json.dumps(meta, indent=2))
        _log(f"finished in {meta['elapsed_s']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--kind", choices=["dev", "ood"], required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tasks-dir", default="/data/out/tasks")
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--nudge", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--arms", default="grep,sem")
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--max-seconds", type=int, default=3600)
    ap.add_argument("--repos", default="/data/repos")
    a = ap.parse_args()
    a.kind_prefix = "dev" if a.kind == "dev" else "ood40"
    a.seeds = a.seeds or ("0,1" if a.kind == "dev" else "0")
    a.nudge = a.nudge or ("teacher" if a.kind == "ood" else os.environ.get("EVAL_NUDGE", "teacher"))
    asyncio.run(main_async(a))
