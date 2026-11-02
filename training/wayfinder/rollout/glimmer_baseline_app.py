"""Prompted-Glimmer baseline ("before training"): vLLM and the rollout harness in ONE Modal H100 container.

    modal run wayfinder/rollout/glimmer_baseline_app.py --tasks artifacts/teacher_test/stageB_tasks.json \
        --out artifacts/baseline/glimmer_dev.jsonl --seeds 0,1 --concurrency 32

Why one container: the laptop resolver fails on *.modal.host, so rollouts run next to the server (127.0.0.1) and rows
stream back through a generator. New entry point; it only imports the shared harness (run.py / chat.py / score.py),
never edits it. Same prompt, tools and convergence nudge as the Spark teacher's Stage B (`run.TEACHER_NUDGE`),
scout product profile, grep-only (`semantic=false`), T=0.85 top_p=0.95, max_turns 40, max_seconds 900.
GPU safety: ephemeral app `wayfinder-glimmer-base*`, function timeout 85 min, local watcher ops/glimmer_base_watch.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "wayfinder-glimmer-base"
MODEL_ID = "meta-models/Muse-Glimmer-30B"
WEIGHTS_DIR = "/weights/Muse-Glimmer-30B"
VLLM_VERSION = "0.32.0"
H100_USD_PER_S = 0.001097  # Modal H100 list price (~$3.95/h)
GPU_TIMEOUT_S = 85 * 60
LAUNCH_DEADLINE_S = 68 * 60  # stop starting new rollouts after this much function time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the container the file sits at /root (REPO unused there)
SCOUT_SRC = REPO.parent / "voyage-scout" / "src"

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("wayfinder-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub", "jsonschema")
    .apt_install("ripgrep", "ca-certificates")
    .uv_pip_install("httpx==0.28.1", "pydantic>=2.9", "pathspec==1.1.1")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_LOGGING_LEVEL": "INFO", "SCOUT_SRC": "/scout_src",
          "PYTHONPATH": "/root", "GIT_TERMINAL_PROMPT": "0"})
    .add_local_dir(str(SCOUT_SRC), "/scout_src", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)

BASE_FLAGS = [
    "--served-model-name", "glimmer", "--dtype", "bfloat16", "--gpu-memory-utilization", "0.92",
    "--language-model-only", "--enable-auto-tool-choice", "--enable-prefix-caching", "--max-num-seqs", "64",
    "--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer", "--port", "8000",
]  # fmt: skip
MAX_LENS = (65536, 32768)  # 65536 first; 32768 if it fails to start (OOM / no KV room)


def _log(msg: str) -> None:
    print(f"[glimmer_base {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _start_vllm(max_len: int, timeout_s: int = 25 * 60) -> tuple[subprocess.Popen[bytes], float, str]:
    import urllib.request

    log_path = f"/tmp/vllm_{max_len}.log"
    cmd = ["vllm", "serve", WEIGHTS_DIR, *BASE_FLAGS, "--max-model-len", str(max_len)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=open(log_path, "wb"), stderr=subprocess.STDOUT)
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited rc={proc.returncode}\n" + "\n".join(Path(log_path).read_text(errors="replace").splitlines()[-40:]))
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                if r.status == 200:
                    return proc, time.time() - t0, log_path
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    proc.kill()
    raise RuntimeError("vllm not healthy in time")


def _metrics() -> dict[str, float]:
    import re
    import urllib.request

    out: dict[str, float] = {}
    try:
        text = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return out
    for name in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:prompt_tokens_total",
                 "vllm:generation_tokens_total", "vllm:num_preemptions_total"):
        m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)", text, re.M)
        if m:
            out[name] = float(m.group(1))
    return out


@app.function(image=image, gpu="H100", volumes={"/weights": weights}, timeout=GPU_TIMEOUT_S, cpu=8, memory=65536)
async def serve_and_rollout(todo: list[tuple[dict, int]], cfg: dict):
    """Async generator: yields one row dict per finished rollout, then a final {"_meta": ...} dict."""
    import asyncio

    t_start = time.time()
    meta: dict = {"max_len_tried": [], "cfg": cfg}
    proc = None
    for ml in MAX_LENS:
        try:
            _log(f"starting vllm max_model_len={ml}")
            proc, load_s, log_path = await asyncio.to_thread(_start_vllm, ml)
            meta.update(max_model_len=ml, load_s=round(load_s, 1))
            break
        except Exception as e:  # noqa: BLE001
            meta["max_len_tried"].append({"max_model_len": ml, "error": repr(e)[:1500]})
            _log(f"vllm failed at {ml}: {repr(e)[:400]}")
            subprocess.run(["pkill", "-f", "vllm serve"], check=False)
            await asyncio.sleep(10)
    if proc is None:
        meta["fatal"] = "vllm never started"
        yield {"_meta": meta}
        return
    lines = Path(log_path).read_text(errors="replace").splitlines()
    meta["log_highlights"] = [ln[:300] for ln in lines if any(k in ln for k in (
        "GPU KV cache size", "Maximum concurrency", "Model loading took", "prefix caching", "Using max model len"))][:12]
    _log(f"healthy after {load_s:.0f}s: {meta['log_highlights']}")

    from wayfinder.rollout.chat import CompatChat, Prices, Sampling
    from wayfinder.rollout.run import TEACHER_NUDGE, checkout, run_rollout
    from wayfinder.rollout.score import score_row

    meta["nudge"] = TEACHER_NUDGE
    sem = asyncio.Semaphore(cfg["concurrency"])
    clones: dict[str, asyncio.Future] = {}
    queue: asyncio.Queue = asyncio.Queue()

    async def get_root(t: dict) -> tuple[str, float]:
        key = f"{t['repo']}@{t['base_commit']}"
        if key not in clones:
            clones[key] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], "/tmp/repos"))
        return await clones[key]

    async def one(t: dict, seed: int) -> None:
        async with sem:
            if time.time() - t_start > LAUNCH_DEADLINE_S:
                await queue.put({"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "error": "not launched: function deadline", "infra_error": True})
                return
            try:
                root, clone_s = await get_root(t)
            except Exception as e:  # noqa: BLE001
                await queue.put({"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "error": f"checkout: {e}"[:300], "infra_error": True})
                return
            chat = CompatChat(
                "http://127.0.0.1:8000/v1", "x", "glimmer",
                sampling=Sampling(cfg["temperature"], cfg["top_p"]), prices=Prices(*cfg["prices"]), seed=seed,
                vllm_extensions=False, timeout=600.0, max_retries=4,
            )
            try:
                row = await asyncio.wait_for(
                    run_rollout(t, chat, root, seed=seed, max_turns=cfg["max_turns"], max_seconds=cfg["max_seconds"],
                                nudge=TEACHER_NUDGE if cfg["nudge"] else None), timeout=cfg["max_seconds"] + 600)
                row.update(score_row(row))
                row["clone_s"] = round(clone_s, 1)
                row["temperature"], row["top_p"] = cfg["temperature"], cfg["top_p"]
                row["max_model_len"] = meta["max_model_len"]
            except Exception as e:  # noqa: BLE001
                row = {"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "error": f"harness: {type(e).__name__}: {e}"[:300], "infra_error": True}
            await queue.put(row)

    tasks = [asyncio.ensure_future(one(t, s)) for t, s in todo]
    try:
        for _ in range(len(tasks)):
            yield await queue.get()
    finally:
        for tk in tasks:
            tk.cancel()
        meta["vllm_metrics"] = _metrics()
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        meta["gpu_function_seconds"] = round(time.time() - t_start, 1)
        meta["gpu_cost_estimate_usd"] = round(meta["gpu_function_seconds"] * H100_USD_PER_S, 2)
    yield {"_meta": meta}


@app.local_entrypoint()
def main(tasks: str, out: str, seeds: str = "0,1", concurrency: int = 32, max_turns: int = 40,
         max_seconds: int = 900, nudge: bool = True, limit: int = 0):
    task_list = json.loads(Path(tasks).read_text())
    if limit:
        task_list = task_list[:limit]
    seed_list = [int(s) for s in seeds.split(",")]
    cfg = {"concurrency": concurrency, "temperature": 0.85, "top_p": 0.95, "max_turns": max_turns,
           "max_seconds": max_seconds, "nudge": nudge, "prices": [1.25, 4.25, 0.15]}  # list prices = teacher's, comparable cost only
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if outp.exists():
        for ln in outp.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if not r.get("error"):
                    done.add((r["task_id"], r["seed"]))
    todo = [(t, s) for s in seed_list for t in task_list if (t["task_id"], s) not in done]
    print(f"{len(todo)} rollouts to run, {len(done)} done", flush=True)
    n = errs = 0
    cost = 0.0
    meta_path = outp.with_suffix(".meta.json")
    with outp.open("a") as f:
        for row in serve_and_rollout.remote_gen(todo, cfg):
            if "_meta" in row:
                meta_path.write_text(json.dumps(row["_meta"], indent=2))
                print("meta:", json.dumps({k: v for k, v in row["_meta"].items() if k != "cfg"}), flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            errs += bool(row.get("error"))
            cost += row.get("cost_usd") or 0
            print(f"[{n}/{len(todo)}] {row['task_id'][-36:]} s{row['seed']} turns={row.get('turns')} wall={row.get('wall_s')} "
                  f"recall={row.get('recall')} err={row.get('error')} | list_cost=${cost:.2f}", flush=True)
