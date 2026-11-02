"""Student rollout generation for OPD on Modal: vLLM (Glimmer student) and the scout harness in ONE H100/H200 container.

    modal run wayfinder/opd/rollout_app.py --out opd/smoke --n-tasks 20 --seeds 0 --student /data/sft/v1_merged
    # many rollouts: n parallel apps, same --out, --shard i/n each (3 workers x 2 seeds x 340 tasks = 2,040 rollouts)

Rows (with the token chains) land in wayfinder-data:/opd/<out>/rollouts/<id>.json.gz and are committed as they finish; finished
(task, seed) pairs are skipped on restart. Train-split accepted tasks only; dev, test, eval_holdout and the E2E eval tasks
are excluded (data/opd_excluded_tasks.json). App name prefix `wayfinder-opd` (ops/opd_watch.py).
Student: any merged checkpoint on the data volume (the base model is /weights/Muse-Glimmer-30B on wayfinder-weights).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import sys

import modal

APP_NAME = "wayfinder-opd-rollout"
VLLM_VERSION = "0.32.0"
GPU = os.environ.get("WF_OPD_GPU", "H100")  # H100 | H200
USD_PER_S = {"H100": 0.001097, "H200": 0.001261}[GPU]
GPU_TIMEOUT_S = int(os.environ.get("WF_OPD_TIMEOUT_MIN", "120")) * 60

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # local entrypoint imports wayfinder.opd.*
SCOUT_SRC = REPO.parent / "voyage-scout" / "src"

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("wayfinder-weights", create_if_missing=True)
data_vol = modal.Volume.from_name("wayfinder-data", create_if_missing=True)
INDEX_SECRET = modal.Secret.from_name("wayfinder-index-keys")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub", "jsonschema")
    .apt_install("ripgrep", "ca-certificates")
    .uv_pip_install("httpx==0.28.1", "pydantic>=2.9", "pathspec==1.1.1")
    .uv_pip_install("watchfiles==1.3.0", "cryptography==50.0.2", "keyring==25.7.0", "turbopuffer==3.0.0", "voyageai==0.5.0")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_LOGGING_LEVEL": "INFO", "SCOUT_SRC": "/scout_src", "PYTHONPATH": "/root", "GIT_TERMINAL_PROMPT": "0",
          "WF_OPD_GPU": GPU, "WF_OPD_TIMEOUT_MIN": str(GPU_TIMEOUT_S // 60)})
    .add_local_dir(str(SCOUT_SRC), "/scout_src", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)

SERVED = "student"
BASE_FLAGS = [
    "--served-model-name", SERVED, "--dtype", "bfloat16", "--gpu-memory-utilization", "0.92", "--language-model-only",
    "--enable-auto-tool-choice", "--enable-prefix-caching", "--max-num-seqs", "64",
    "--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer", "--port", "8000", "--max-model-len", "65536",
]  # fmt: skip


def _log(msg: str) -> None:
    print(f"[opd_rollout {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _start_vllm(model_dir: str, timeout_s: int = 25 * 60) -> tuple[subprocess.Popen, float, str]:
    import urllib.request

    log_path = "/tmp/vllm_student.log"
    t0 = time.time()
    proc = subprocess.Popen(["vllm", "serve", model_dir, *BASE_FLAGS], stdout=open(log_path, "wb"), stderr=subprocess.STDOUT)
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("vllm exited\n" + "\n".join(Path(log_path).read_text(errors="replace").splitlines()[-40:]))
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                if r.status == 200:
                    return proc, time.time() - t0, log_path
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    proc.kill()
    raise RuntimeError("vllm not healthy in time")


@app.function(image=image, gpu=GPU, volumes={"/weights": weights, "/data": data_vol}, secrets=[INDEX_SECRET], timeout=GPU_TIMEOUT_S, cpu=8, memory=65536)
async def serve_and_rollout(todo: list[tuple[dict, int]], cfg: dict, out_dir: str, student: str) -> dict:
    import asyncio

    from wayfinder.opd.rollout import opd_rollout, write_rollout
    from wayfinder.rollout.run import checkout

    t_start = time.time()
    await asyncio.to_thread(data_vol.reload)
    from wayfinder.opd.rollout import rollout_id

    have = {p.name.split(".")[0] for p in (Path("/data") / out_dir / "rollouts").glob("*.json.gz")}
    todo = [(t, s) for t, s in todo if rollout_id(t["task_id"], s) not in have]  # resumable
    model_dir = student if student.startswith("/") else f"/weights/{student}"
    proc, load_s, log_path = await asyncio.to_thread(_start_vllm, model_dir)
    meta: dict = {"student": student, "gpu": GPU, "load_s": round(load_s, 1), "cfg": cfg, "n_todo": len(todo)}
    _log(f"vllm healthy after {load_s:.0f}s; {len(todo)} rollouts at concurrency {cfg['concurrency']}")
    sem = asyncio.Semaphore(cfg["concurrency"])
    clones: dict[str, asyncio.Future] = {}
    lock = asyncio.Lock()
    stats: list[dict] = []
    n_done = 0

    async def get_root(t: dict) -> tuple[str, float]:
        key = f"{t['repo']}@{t['base_commit']}"
        if key not in clones:
            clones[key] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], "/tmp/repos"))
        return await clones[key]

    async def one(t: dict, seed: int) -> None:
        nonlocal n_done
        async with sem:
            t0 = time.time()
            try:
                root, clone_s = await get_root(t)
                row = await asyncio.wait_for(opd_rollout(t, "http://127.0.0.1:8000/v1", SERVED, root, seed=seed, cfg=cfg), timeout=cfg["max_seconds"] + 900)
                row["clone_s"] = round(clone_s, 1)
            except Exception as e:  # noqa: BLE001
                stats.append({"task_id": t["task_id"], "seed": seed, "error": f"{type(e).__name__}: {e}"[:300]})
                return
            d = row["opd"]["diag"]
            st = {"task_id": t["task_id"], "seed": seed, "turns": row.get("turns"), "wall_s": round(time.time() - t0, 1), "recall": row.get("recall"), "recall_reported": row.get("recall_reported"),
                  "reported": bool((row.get("explore_result") or {}).get("results")) and not (row.get("explore_result") or {}).get("budget_exhausted"),
                  "error": row.get("error"), "n_calls": row["opd"]["n_calls"], **{k: d[k] for k in ("n_chains", "chain_breaks", "chain_tokens", "assistant_tokens", "sampled_tokens", "lenient_mismatch", "no_terminator", "eot_appended", "n_elided_max")}}
            stats.append(st)
            if not row.get("error") and d["n_chains"]:
                async with lock:
                    write_rollout(f"/data/{out_dir}", row)
                    n_done += 1
                    if n_done % 5 == 0:
                        await asyncio.to_thread(data_vol.commit)
            _log(f"[{len(stats)}/{len(todo)}] {json.dumps(st)}")

    try:
        await asyncio.gather(*[one(t, s) for t, s in todo])
    finally:
        try:
            import re
            import urllib.request

            txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
            for n in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total", "vllm:num_preemptions_total"):
                m = re.search(rf"^{re.escape(n)}(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)", txt, re.M)
                if m:
                    meta[n.split(":")[1]] = float(m.group(1))
        except Exception:  # noqa: BLE001
            pass
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        await asyncio.to_thread(data_vol.commit)
    meta["stats"] = stats
    meta["gpu_function_seconds"] = round(time.time() - t_start, 1)
    meta["gpu_cost_estimate_usd"] = round(meta["gpu_function_seconds"] * USD_PER_S, 2)
    return meta


@app.local_entrypoint()
def main(out: str = "opd/smoke", n_tasks: int = 20, seeds: str = "0", student: str = "/data/sft/v1_merged", concurrency: int = 12,
         max_turns: int = 40, max_seconds: int = 900, semantic: str = "off", temperature: float = 0.85, top_p: float = 0.95,
         strict_chains: bool = False, salt: str = "opd-smoke", summary: str = "", tasks: str = "", retain_reasoning: bool = True, shard: str = "", max_chain_tokens: int = 49152):
    from wayfinder.opd.rollout import load_exclude, select_tasks

    splits = json.loads((REPO / "data" / "splits.json").read_text())
    flt = json.loads((REPO / "data" / "task_filter.json").read_text())
    if tasks:  # eval-style A/B: an explicit task list (e.g. artifacts/teacher_test/stageB_tasks.json); NOT for training rollouts
        task_list = json.loads(Path(tasks).read_text())
        tasks = task_list[:n_tasks] if n_tasks else task_list
    else:
        tasks = select_tasks(splits, flt, load_exclude(), n_tasks, salt)
    seed_list = [int(s) for s in seeds.split(",")]
    cfg = {"concurrency": concurrency, "temperature": temperature, "top_p": top_p, "max_turns": max_turns, "max_seconds": max_seconds, "nudge": True,
           "semantic": semantic, "strict_chains": strict_chains, "retain_reasoning": retain_reasoning, "max_chain_tokens": max_chain_tokens}
    todo = [(t, s) for s in seed_list for t in tasks]
    if shard:  # "i/n": run only every n-th rollout starting at i, so n apps (one per GPU container) share one --out dir
        i, n = (int(x) for x in shard.split("/"))
        todo = todo[i::n]
    print(f"{len(todo)} rollouts over {len(tasks)} tasks ({len({t['repo'] for t in tasks})} repos), splits: {sorted({splits['tasks'][t['task_id']]['split'] for t in tasks})}", flush=True)
    res = serve_and_rollout.remote(todo, cfg, out, student)
    od = REPO / "artifacts" / "opd"
    od.mkdir(parents=True, exist_ok=True)
    (od / (summary or f"rollout_{Path(out).name}{'_' + shard.replace('/', 'of') if shard else ''}.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "stats"}), flush=True)
