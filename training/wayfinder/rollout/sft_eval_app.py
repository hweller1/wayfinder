"""SFT v1 eval: base Glimmer + LoRA adapter `wayfinder-v1` (Volume wayfinder-data:/sft/v1) in vLLM, rollouts in the same
container. Copy of glimmer_baseline_app.py (not modified) with: vLLM `--enable-lora --max-lora-rank 64
--lora-modules wayfinder-v1=/data/sft/v1`, requests use model `wayfinder-v1`, an adapter-load verification step
(vLLM log, /v1/models, base vs adapter on the same prompt, a tool-call parse check) that aborts the run on failure,
and two arms run sequentially on the same server: `grep` (semantic=off, the baseline protocol) and `sem`
(semantic=auto).

    modal run wayfinder/rollout/sft_eval_app.py --tasks artifacts/teacher_test/stageB_tasks.json \
        --outdir artifacts/sft_v1 --seeds 0,1 --concurrency 32

Original baseline docstring follows.

Prompted-Glimmer baseline ("before training"): vLLM and the rollout harness in ONE Modal H100 container.

Why one container: the laptop resolver fails on *.modal.host, so rollouts run next to the server (127.0.0.1) and rows
stream back through a generator. New entry point; it only imports the shared harness (run.py / chat.py / score.py),
never edits it. Same prompt, tools and convergence nudge as the Spark teacher's Stage B (`run.TEACHER_NUDGE`),
scout product profile, grep-only (`semantic=false`), T=0.85 top_p=0.95, max_turns 40, max_seconds 900.
GPU safety: ephemeral app `wayfinder-sft-eval*`, function timeout 110 min, local watcher ops/sft_eval_watch.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

VARIANT = os.environ.get("WF_EVAL_VARIANT", "v1")  # v1 | v2a | v2b: which merged checkpoint under /data/sft/<variant>_merged
ARM_ONLY = os.environ.get("WF_EVAL_ARMS", "")  # unused here; arms come from --arms
APP_NAME = "wayfinder-sft-eval" if VARIANT == "v1" else f"wayfinder-{'sftv4' if VARIANT == 'v4' else 'sftv3' if VARIANT == 'v3' else 'sftv2'}-eval-{VARIANT}"  # sftv2 prefix = ops/sftv2_watch.py
APP_NAME = os.environ.get("WF_EVAL_APP", APP_NAME)  # OPD round 1: wayfinder-sfteval-opd1-<variant> (ops/opd_eval_watch.py)
NUDGE_KEY = os.environ.get("WF_EVAL_NUDGE", "teacher")  # teacher (default, every earlier eval) | f1 (adds the report-order line; GRPO run f1 and its evals)
RETAIN_REASONING = os.environ.get("WF_EVAL_RETAIN", "1") == "1"  # re-send earlier turns' reasoning; default ON since 2026-10-27 (production scout does, as the SFT template does). WF_EVAL_RETAIN=0 reproduces the pre-2026-10-27 protocol (OPD arm a)
MODEL_ID = "meta-models/Muse-Glimmer-30B"
WEIGHTS_DIR = os.environ.get("WF_EVAL_WEIGHTS", f"/data/sft/{VARIANT}_merged")  # LoRA merged into base by sft_merge_app.py (runtime LoRA was ~10x too slow, see docs)
VLLM_VERSION = "0.32.0"
EVAL_GPU = os.environ.get("WF_EVAL_GPU", "H100")  # same hardware as the base and v1 evals so per-turn latency is comparable
H100_USD_PER_S = 0.001097 if EVAL_GPU == "H100" else 0.001261  # Modal list price: H100 ~$3.95/h, H200 ~$4.54/h (name kept for the existing cost fields)
GPU_TIMEOUT_S = 110 * 60 if VARIANT == "v1" else 170 * 60  # v2: max_seconds 3600 removes the wall cap, so allow long tails
LAUNCH_DEADLINE_S = 95 * 60 if VARIANT == "v1" else 140 * 60  # stop starting new rollouts after this much function time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the container the file sits at /root (REPO unused there)
SCOUT_SRC = REPO.parent / "voyage-scout" / "src"

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("wayfinder-weights", create_if_missing=True)
data_vol = modal.Volume.from_name("wayfinder-data", create_if_missing=True)
INDEX_SECRET = modal.Secret.from_name("wayfinder-index-keys")  # semantic arm: TURBOPUFFER/VOYAGE keys + gateway master key
LORA_NAME = f"wayfinder-{VARIANT}"
LORA_PATH = "/data/sft/v1"  # unused when serving merged weights

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub", "jsonschema")
    .apt_install("ripgrep", "ca-certificates")
    .uv_pip_install("httpx==0.28.1", "pydantic>=2.9", "pathspec==1.1.1")
    .uv_pip_install("watchfiles==1.3.0", "cryptography==50.0.2", "keyring==25.7.0", "turbopuffer==3.0.0", "voyageai==0.5.0")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_LOGGING_LEVEL": "INFO", "SCOUT_SRC": "/scout_src",
          "PYTHONPATH": "/root", "GIT_TERMINAL_PROMPT": "0",
          "WF_EVAL_VARIANT": VARIANT, "WF_EVAL_GPU": EVAL_GPU, "WF_EVAL_APP": APP_NAME, "WF_EVAL_RETAIN": "1" if RETAIN_REASONING else "0", "WF_EVAL_WEIGHTS": WEIGHTS_DIR, "WF_EVAL_NUDGE": NUDGE_KEY})  # the container re-imports this module: it must see the same variant
    .add_local_dir(str(SCOUT_SRC), "/scout_src", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)

BASE_FLAGS = [
    "--served-model-name", LORA_NAME, "--dtype", "bfloat16", "--gpu-memory-utilization", "0.92",
    "--language-model-only", "--enable-auto-tool-choice", "--enable-prefix-caching", "--max-num-seqs", "64",
    "--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer", "--port", "8000",
]  # fmt: skip
MAX_LENS = (65536, 32768)  # 65536 first; 32768 if it fails to start (OOM / no KV room)


def _log(msg: str) -> None:
    print(f"[sft_eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def _post(path: str, body: dict, timeout: float = 300.0) -> dict:
    import urllib.request

    req = urllib.request.Request(f"http://127.0.0.1:8000{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# Recorded earlier, same prompt, T=0, 12 first-token logprobs: base `glimmer` and runtime-LoRA `wayfinder-v1` (lora_attempt/).
BASE_LP12 = [0.0, 0.0, 0.0, -0.04, -0.0001, -0.0045, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]
LORA_LP12 = [-0.0, -0.0004, -0.0, -0.4761, -0.0145, -0.113, -0.5127, -0.0175, -0.0001, -0.0001, -0.0, -0.0182]


def _verify_adapter(log_path: str) -> dict:
    """Merged-weights check: serves under `wayfinder-v1`, tool call parses, and the T=0 logprobs match the runtime-LoRA
    run (close) and differ from the recorded base."""
    import urllib.request

    v: dict = {}
    v["models"] = [m["id"] for m in json.loads(urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10).read())["data"]]
    msgs = [{"role": "system", "content": "You are a code search agent. Find files in a repository. Call tools to search; report when done."},
            {"role": "user", "content": "Find where the HTTP retry backoff maximum delay is configured. Use the grep tool first."}]
    tools = [{"type": "function", "function": {"name": "grep", "description": "Search file contents with a regex.",
              "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
             {"type": "function", "function": {"name": "report", "description": "Report the final files.",
              "parameters": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}}}, "required": ["files"]}}}]
    r = _post("/v1/chat/completions", {"model": LORA_NAME, "messages": msgs, "tools": tools, "tool_choice": "auto",
                                        "temperature": 0, "max_tokens": 300, "logprobs": True, "top_logprobs": 1})
    ch = r["choices"][0]
    lp = [round(t["logprob"], 4) for t in (ch.get("logprobs") or {}).get("content", [])[:12]]
    v["output"] = {"reasoning": (ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or "")[:400],
                   "tool_calls": ch["message"].get("tool_calls"), "finish": ch["finish_reason"], "logprobs12": lp}
    d_base = sum(abs(a - b) for a, b in zip(lp, BASE_LP12))
    d_lora = sum(abs(a - b) for a, b in zip(lp, LORA_LP12))
    v["l1_vs_base"], v["l1_vs_lora_run"] = round(d_base, 4), round(d_lora, 4)
    v["differs"] = d_base > 0.3 and (d_lora < d_base if VARIANT == "v1" else True)  # the recorded LoRA-run logprobs only exist for v1
    v["v1_tool_call_parsed"] = bool(ch["message"].get("tool_calls"))
    return v


@app.function(image=image, gpu=EVAL_GPU, volumes={"/weights": weights, "/data": data_vol}, secrets=[INDEX_SECRET],
              timeout=GPU_TIMEOUT_S, cpu=8, memory=65536)
async def serve_and_rollout(todo: list[tuple[dict, int]], cfg: dict):
    """Async generator: yields one row dict per finished rollout, then a final {"_meta": ...} dict."""
    import asyncio

    t_start = time.time()
    meta: dict = {"max_len_tried": [], "cfg": cfg, "gpu": EVAL_GPU, "variant": VARIANT, "retain_reasoning": RETAIN_REASONING, "weights": WEIGHTS_DIR}
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
    try:
        meta["adapter_check"] = await asyncio.to_thread(_verify_adapter, log_path)
    except Exception as e:  # noqa: BLE001
        meta["adapter_check"] = {"error": repr(e)[:1500]}
    _log(f"adapter_check: {json.dumps(meta['adapter_check'])[:3000]}")
    ck = meta["adapter_check"]
    yield {"_meta": dict(meta)}  # early copy so the adapter evidence reaches the laptop immediately
    if ck.get("error") or LORA_NAME not in ck.get("models", []) or not ck.get("differs") or not ck.get("v1_tool_call_parsed"):
        meta["fatal"] = "adapter check failed (not loaded / identical to base / tool call not parsed); no rollouts run"
        proc.terminate()
        meta["gpu_function_seconds"] = round(time.time() - t_start, 1)
        meta["gpu_cost_estimate_usd"] = round(meta["gpu_function_seconds"] * H100_USD_PER_S, 2)
        yield {"_meta": meta}
        return

    from wayfinder.rollout.chat import CompatChat, Prices, Sampling
    from wayfinder.rollout.run import NUDGES, checkout, run_rollout
    from wayfinder.rollout.score import score_row
    TEACHER_NUDGE = NUDGES[NUDGE_KEY]  # noqa: N806  (the text recorded in meta and in every row's `nudge`)
    if cfg["max_seconds"] > 900:  # scout's Budget caps max_seconds at 900; lift it at runtime (scout source untouched)
        from pydantic import Field
        from pydantic.fields import FieldInfo
        from scout.core.contracts.types import Budget, ExploreRequest

        Budget.model_fields["max_seconds"] = FieldInfo.from_annotated_attribute(int, Field(default=120, ge=5, le=100000))
        Budget.model_rebuild(force=True)
        ExploreRequest.model_rebuild(force=True)
        _log(f"lifted Budget.max_seconds cap; test={Budget(max_turns=40, max_seconds=cfg['max_seconds'])}")

    meta["nudge"] = TEACHER_NUDGE
    sem = asyncio.Semaphore(cfg["concurrency"])
    clones: dict[str, asyncio.Future] = {}
    queue: asyncio.Queue = asyncio.Queue()

    async def get_root(t: dict) -> tuple[str, float]:
        key = f"{t['repo']}@{t['base_commit']}"
        if key not in clones:
            clones[key] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], "/tmp/repos"))
        return await clones[key]

    async def one(t: dict, seed: int, arm: str) -> None:
        async with sem:
            if time.time() - t_start > LAUNCH_DEADLINE_S:
                await queue.put({"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "arm": arm, "error": "not launched: function deadline", "infra_error": True})
                return
            try:
                root, clone_s = await get_root(t)
            except Exception as e:  # noqa: BLE001
                await queue.put({"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "arm": arm, "error": f"checkout: {e}"[:300], "infra_error": True})
                return
            chat = CompatChat(
                "http://127.0.0.1:8000/v1", "x", LORA_NAME,
                sampling=Sampling(cfg["temperature"], cfg["top_p"]), prices=Prices(*cfg["prices"]), seed=seed,
                vllm_extensions=False, timeout=600.0, max_retries=4,
            )
            chat.retain_reasoning = RETAIN_REASONING
            try:
                row = await asyncio.wait_for(
                    run_rollout(t, chat, root, seed=seed, max_turns=cfg["max_turns"], max_seconds=cfg["max_seconds"],
                                nudge=TEACHER_NUDGE if cfg["nudge"] else None,
                                semantic="off" if arm == "grep" else "auto"), timeout=cfg["max_seconds"] + 600)
                row.update(score_row(row))
                row["arm"] = arm
                row["clone_s"] = round(clone_s, 1)
                row["temperature"], row["top_p"] = cfg["temperature"], cfg["top_p"]
                row["max_model_len"] = meta["max_model_len"]
                row["retain_reasoning"] = RETAIN_REASONING
                row["weights"] = WEIGHTS_DIR
            except Exception as e:  # noqa: BLE001
                row = {"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "arm": arm, "error": f"harness: {type(e).__name__}: {e}"[:300], "infra_error": True}
            await queue.put(row)

    async def _telemetry() -> None:
        import re
        import urllib.request
        while True:
            await asyncio.sleep(30)
            try:
                txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
                g = lambda n: re.search(rf"^{re.escape(n)}(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)", txt, re.M)  # noqa: E731
                vals = {n.split(":")[1]: (g(n).group(1) if g(n) else None) for n in (
                    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                    "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")}
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
                tl = [ln[-260:] for ln in Path(log_path).read_text(errors="replace").splitlines()[-400:] if "throughput" in ln][-1:]
                _log(f"telemetry {vals} gpu={gpu} {tl}")
            except Exception as e:  # noqa: BLE001
                _log(f"telemetry error {e!r}")

    tele = asyncio.ensure_future(_telemetry())
    arms = list(dict.fromkeys(arm for _, _, arm in todo))  # arms run one after the other on the same server
    meta["arm_gpu_seconds"] = {}
    tasks: list = []
    try:
        for arm in arms:
            t_arm = time.time()
            m0 = _metrics()
            tasks = [asyncio.ensure_future(one(t, s, a)) for t, s, a in todo if a == arm]
            for _ in range(len(tasks)):
                yield await queue.get()
            m1 = _metrics()
            meta["arm_gpu_seconds"][arm] = round(time.time() - t_arm, 1)
            meta.setdefault("arm_vllm_metrics", {})[arm] = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
            _log(f"arm {arm} done in {meta['arm_gpu_seconds'][arm]}s")
    finally:
        tele.cancel()
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
def main(tasks: str, outdir: str, seeds: str = "0,1", concurrency: int = 32, max_turns: int = 40,
         max_seconds: int = 900, nudge: bool = True, limit: int = 0, arms: str = "grep,sem", only: str = "", shard: str = ""):
    """shard "i/n": run only tasks i, i+n, i+2n, ... of the task list (separate --outdir per shard; concatenate rows afterwards)."""
    task_list = json.loads(Path(tasks).read_text())
    if limit:
        task_list = task_list[:limit]
    if shard:
        si, sn = (int(x) for x in shard.split("/"))
        task_list = task_list[si::sn]
    seed_list = [int(s) for s in seeds.split(",")]
    cfg = {"concurrency": concurrency, "temperature": 0.85, "top_p": 0.95, "max_turns": max_turns,
           "max_seconds": max_seconds, "nudge": nudge, "prices": [1.25, 4.25, 0.15]}  # list prices = teacher's, comparable cost only
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    arm_list = arms.split(",")
    files = {a: od / f"sft_{VARIANT}_{a}.jsonl" for a in arm_list}
    done: set[tuple[str, int, str]] = set()
    for a, fp in files.items():
        if fp.exists():
            for ln in fp.read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    if not r.get("error"):
                        done.add((r["task_id"], r["seed"], a))
    todo = [(t, s, a) for a in arm_list for s in seed_list for t in task_list if (t["task_id"], s, a) not in done]
    if only:  # JSON list of [task_id, seed] pairs: rerun just these (e.g. the wall-capped ones)
        keep = {(a, b) for a, b in json.loads(Path(only).read_text())}
        todo = [x for x in todo if (x[0]["task_id"], x[1]) in keep]
    print(f"{len(todo)} rollouts to run, {len(done)} done", flush=True)
    n = 0
    cost = 0.0
    meta_path = od / f"sft_{VARIANT}.meta.json"
    handles = {a: fp.open("a") for a, fp in files.items()}
    import threading

    lock = threading.Lock()
    counter = {"n": 0, "cost": 0.0}

    def run_arm(arm: str) -> None:  # one H100 container per arm, in parallel
        sub = [x for x in todo if x[2] == arm]
        if not sub:
            return
        for row in serve_and_rollout.remote_gen(sub, cfg):
            with lock:
                if "_meta" in row:
                    (od / f"sft_{VARIANT}_{arm}.meta.json").write_text(json.dumps(row["_meta"], indent=2))
                    print(f"[{arm}] meta:", json.dumps({k: v for k, v in row["_meta"].items() if k != "cfg"})[:1500], flush=True)
                    continue
                f = handles[row["arm"]]
                f.write(json.dumps(row) + "\n")
                f.flush()
                counter["n"] += 1
                counter["cost"] += row.get("cost_usd") or 0
                print(f"[{counter['n']}/{len(todo)}] {row['arm']} {row['task_id'][-36:]} s{row['seed']} turns={row.get('turns')} wall={row.get('wall_s')} "
                      f"recall={row.get('recall')} err={row.get('error')} | list_cost=${counter['cost']:.2f}", flush=True)

    try:
        threads = [threading.Thread(target=run_arm, args=(a,)) for a in arm_list]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        for f in handles.values():
            f.close()
