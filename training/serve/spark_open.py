"""Serve meta-models/Muse-Spark-Open-816B-A42B-NVFP4 on Modal with vLLM (8x B200 only; owner directive).

Ephemeral app only (never `modal deploy`). The GPU function starts vLLM, forwards port 8000 through a
Modal tunnel, publishes the URL and a rolling server-log tail in the Modal Dict `wayfinder-spark-ctl`, and
stays up until the Dict key `stop` is set or the 55 min function timeout. The client side lives in
serve/spark_client.py (reads the URL from the Dict, runs the checks, writes artifacts/serve/spark_check.json).

    modal run serve/spark_open.py            # keep it in the background; arm ops/spark_watch.py first
    python3 serve/spark_client.py            # in another shell once the Dict shows a url
    python3 serve/spark_client.py --stop
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

# Parallel replicas: SPARK_REPLICA (or SPARK_APP_SUFFIX) = "-r1", "-h200", ... gives each its own app and ctl Dict.
SUFFIX = os.environ.get("SPARK_APP_SUFFIX", os.environ.get("SPARK_REPLICA", ""))
APP_NAME = "wayfinder-spark-check" + SUFFIX  # must keep the "wayfinder-spark" prefix (teardown watcher)
CTL_NAME = os.environ.get("SPARK_CTL", "wayfinder-spark-ctl" + SUFFIX)
MODEL_DIR = "/spark/Muse-Spark-Open-816B-A42B-NVFP4"
IMAGE_TAG = os.environ.get("SPARK_IMAGE", "vllm/vllm-openai:muse-spark-open-x86_64-cu130")
GPU = os.environ.get("SPARK_GPU", "B200:8")
# Owner directive 2026-10-26: B200 only (Meta's recommendation; NVFP4 needs Blackwell). Refuse anything else.
if any(g.strip().split(":")[0].upper() != "B200" for g in GPU.split(",")):
    raise SystemExit(f"SPARK_GPU={GPU!r} refused: owner directive is B200 only")
API_KEY = "wf-spark-local"
TIMEOUT_S = int(os.environ.get("SPARK_TIMEOUT_MIN", "55")) * 60  # hard cap; ops/spark_watch.py stops any wayfinder-spark* app older than 70 min (per-app overrides in MAX_AGE_BY_APP, e.g. -r1 = 225 min including the capacity queue; raise it there if you raise this)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("wayfinder-spark")
cache_vol = modal.Volume.from_name("wayfinder-cache", create_if_missing=True)
ctl = modal.Dict.from_name(CTL_NAME, create_if_missing=True)

# Compile caches persist on the Volume so the 8x B200 cold start (torch.compile, Triton, FlashInfer JIT, CUDA graphs
# are rebuilt from these) is paid once, not per launch.
CACHE_ENV = {
    "VLLM_CACHE_ROOT": "/cache/vllm",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
    "TRITON_CACHE_DIR": "/cache/triton",
    "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
}

image = (
    modal.Image.from_registry(IMAGE_TAG, add_python="3.12")
    .entrypoint([])
    # SPARK_CTL is baked into the image so the container writes the SAME Dict the launcher named; without it the
    # container saw no SPARK_APP_SUFFIX and every replica wrote (and clobbered) `wayfinder-spark-ctl`.
    .env({"HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "INFO", "HF_HUB_DISABLE_TELEMETRY": "1", "SPARK_CTL": CTL_NAME, "SPARK_TIMEOUT_MIN": str(TIMEOUT_S // 60),
          **CACHE_ENV})  # SPARK_TIMEOUT_MIN too: the in-container serve loop reads TIMEOUT_S (was 55 min default)
    .add_local_file(HERE / "scout_tool_specs.json", "/srv/scout_tool_specs.json")
    .add_local_file(HERE / "scout_product_prompt.md", "/srv/scout_product_prompt.md")
)

BASE_FLAGS = [
    "--served-model-name", "spark",
    "--tensor-parallel-size", "8",
    "--enable-expert-parallel",
    "--max-model-len", "65536",
    "--reasoning-parser", "muse",
    "--tool-call-parser", "muse",
    "--enable-auto-tool-choice",
    "--enable-prefix-caching",
    "--max-logprobs", "20",
    "--gpu-memory-utilization", "0.90",
    "--max-num-seqs", "128",
    "--api-key", API_KEY,
    "--host", "0.0.0.0",
    "--port", "8000",
]  # fmt: skip
# tried in order; a config that dies in < 150 s (bad flag, frontend bug) falls through to the next
# A config that dies at any point before it is healthy and warm (including a FlashInfer autotune crash during the
# NVFP4 kernel warmup on Blackwell) falls through to the next one; `noautotune` is the documented fallback.
NOAUTOTUNE = ["--kernel-config", '{"enable_flashinfer_autotune": false}']
CONFIGS = [
    {"name": "lm_only_rust", "extra": ["--language-model-only"], "env": {}},
    {"name": "lm_only_rust_noautotune", "extra": ["--language-model-only", *NOAUTOTUNE], "env": {}},
    {"name": "full_rust", "extra": [], "env": {}},
    {"name": "lm_only_pyfrontend", "extra": ["--language-model-only"], "env": {"VLLM_USE_RUST_FRONTEND": "0"}},
    {"name": "lm_only_pyfrontend_noautotune", "extra": ["--language-model-only", *NOAUTOTUNE], "env": {"VLLM_USE_RUST_FRONTEND": "0"}},
]


def _post(path: str, body: dict, timeout: int = 900) -> dict:
    import json
    import urllib.request

    req = urllib.request.Request(
        "http://127.0.0.1:8000" + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def warmup() -> dict:
    """Run real request shapes before the URL is published so the first real rollout does not pay for lazy
    kernel / graph / JIT compilation: (1) a ~20k-token multi-turn tool-calling prompt with scout's system prompt and
    tool schemas, (2) a short tool-call request, (3) top_logprobs=20 on a chat request, (4) prompt_logprobs on a
    completion. Returns timings; raises if (1) or (2) fail."""
    import json
    import random

    tools = json.loads(Path("/srv/scout_tool_specs.json").read_text())
    system = Path("/srv/scout_product_prompt.md").read_text()
    rnd = random.Random(7)
    words = ["buffer", "socket", "parser", "handler", "config", "stream", "token", "cache", "index", "router"]

    def code_text(n_lines: int) -> str:
        return "\n".join(
            f"{i + 1}\tdef {rnd.choice(words)}_{rnd.randint(0, 9999)}(self, {rnd.choice(words)}, {rnd.choice(words)}=None):  # {rnd.choice(words)}"
            for i in range(n_lines)
        )

    t = {}
    calib = _post("/tokenize", {"model": "spark", "prompt": code_text(200)})["count"]
    lines_20k = int(200 * 19000 / max(calib, 1))
    brief = "Find where the HTTP retry backoff is computed and where its maximum delay is configured."
    big = [
        {"role": "system", "content": system},
        {"role": "user", "content": brief},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "src/net/retry.py", "start_line": 1, "end_line": 400})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "src/net/retry.py lines 1-400\n" + code_text(lines_20k)},
    ]
    t0 = time.time()
    r = _post("/v1/chat/completions", {"model": "spark", "messages": big, "tools": tools, "tool_choice": "auto", "max_tokens": 400, "temperature": 0.85, "top_p": 0.95, "reasoning_effort": "high"})
    t["big_prompt_tool_call_s"] = round(time.time() - t0, 1)
    t["big_prompt_tokens"] = r["usage"]["prompt_tokens"]
    t0 = time.time()
    r2 = _post("/v1/chat/completions", {"model": "spark", "messages": big[:2], "tools": tools, "tool_choice": "auto", "max_tokens": 400, "temperature": 0.85, "top_p": 0.95, "reasoning_effort": "high"})
    t["tool_call_s"] = round(time.time() - t0, 1)
    t["tool_call_n"] = len(r2["choices"][0]["message"].get("tool_calls") or [])
    for name, fn in (
        ("logprobs20", lambda: _post("/v1/chat/completions", {"model": "spark", "messages": [{"role": "user", "content": "Name three sorting algorithms."}], "max_tokens": 32, "temperature": 1.0, "logprobs": True, "top_logprobs": 20, "reasoning_effort": "low"})),
        ("prompt_logprobs", lambda: _post("/v1/completions", {"model": "spark", "prompt": "The capital of France is Paris, and the capital of Germany is Berlin.", "max_tokens": 1, "temperature": 0, "prompt_logprobs": 5})),
    ):
        t0 = time.time()
        try:
            fn()
            t[name + "_s"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            t[name + "_error"] = repr(e)[:200]
    return t


def _log(msg: str) -> None:
    print(f"[spark_open {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _publish_log(path: str, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            lines = Path(path).read_text(errors="replace").splitlines()[-60:]
            ctl["log_tail"] = [ln[:300] for ln in lines]
        except Exception:
            pass
        stop.wait(10)


REGION = os.environ.get("SPARK_REGION") or None  # e.g. "us-east" / "us-west" / "us"; widens B200 capacity search


@app.function(image=image, gpu=GPU, volumes={"/spark": vol, "/cache": cache_vol}, timeout=TIMEOUT_S, region=REGION)
def serve() -> dict:
    import urllib.request

    t_fn = time.time()
    ctl["stop"] = False
    ctl.pop("url", None)
    ctl["status"] = "starting"
    info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    ctl["gpu"] = info
    out: dict = {"gpu": info, "gpu_spec": GPU, "image": IMAGE_TAG, "attempts": []}
    for cfg in CONFIGS:
        log_path = f"/tmp/vllm_{cfg['name']}.log"
        cmd = ["vllm", "serve", MODEL_DIR, *BASE_FLAGS, *cfg["extra"]]
        att = {"name": cfg["name"], "cmd": " ".join(cmd), "env": cfg["env"]}
        out["attempts"].append(att)
        _log(f"launching {cfg['name']}: {att['cmd']}")
        t0 = time.time()
        proc = subprocess.Popen(
            cmd, stdout=open(log_path, "wb"), stderr=subprocess.STDOUT, env={**os.environ, **cfg["env"]}
        )
        stop_pub = threading.Event()
        threading.Thread(target=_publish_log, args=(log_path, stop_pub), daemon=True).start()
        healthy = False
        while time.time() - t0 < 40 * 60:
            if proc.poll() is not None:
                break
            try:
                req = urllib.request.Request("http://127.0.0.1:8000/health")
                with urllib.request.urlopen(req, timeout=2) as r:
                    if r.status == 200:
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(3)
        att["seconds_to_health"] = round(time.time() - t0, 1) if healthy else None
        att["exit_code"] = proc.poll()
        log = Path(log_path).read_text(errors="replace").splitlines()
        att["log_highlights"] = [
            ln[:300]
            for ln in log
            if any(
                k in ln
                for k in (
                    "Loading weights took", "Model loading took", "GPU KV cache size", "Maximum concurrency",
                    "torch.compile takes", "Capturing CUDA graph", "Graph capturing finished", "backend",
                    "Loading safetensors checkpoint shards", "init engine", "Application startup complete",
                    "Starting vLLM", "version",
                )
            )
        ][:40]  # fmt: skip
        if not healthy:
            att["log_tail"] = log[-30:]
            att["error_lines"] = [ln[:500] for ln in log if any(k in ln for k in ("ERROR", "Error", "error:", "RuntimeError", "not supported", "NotImplemented", "CUDA out of memory", "OOM"))][:60]
            stop_pub.set()
            ctl["log_tail"] = [ln[:300] for ln in log[-60:]]
            if proc.poll() is None:
                proc.terminate()
            if time.time() - t0 < 150 or proc.poll() is not None:
                _log(f"{cfg['name']} failed after {time.time() - t0:.0f}s; trying next config")
                continue
            break
        _log(f"healthy after {att['seconds_to_health']}s; warming up")
        ctl["status"] = "warming"
        try:
            att["warmup"] = warmup()
        except Exception as e:  # noqa: BLE001
            att["warmup_error"] = repr(e)[:300]
            log = Path(log_path).read_text(errors="replace").splitlines()
            att["log_tail"] = log[-30:]
            att["error_lines"] = [ln[:500] for ln in log if any(k in ln for k in ("ERROR", "Error", "error:", "RuntimeError", "not supported", "NotImplemented", "CUDA out of memory", "OOM"))][:60]
            ctl["log_tail"] = [ln[:300] for ln in log[-60:]]
            stop_pub.set()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
            _log(f"{cfg['name']} warmup failed ({att['warmup_error']}); trying next config")
            continue
        att["seconds_to_warm"] = round(time.time() - t0, 1)
        try:
            cache_vol.commit()  # persist compile caches for the next launch
            att["cache_committed"] = True
        except Exception as e:  # noqa: BLE001
            att["cache_committed"] = repr(e)[:150]
        out["working_config"] = cfg["name"]
        _log(f"warm after {att['seconds_to_warm']}s: {att['warmup']}")
        with modal.forward(8000) as tunnel:
            ctl["url"] = tunnel.url
            ctl["status"] = "ready"
            ctl["cold_start_s"] = att["seconds_to_health"]
            ctl["warm_s"] = att["seconds_to_warm"]
            ctl["config"] = cfg["name"]
            ctl["warmup"] = att["warmup"]
            _log(f"tunnel {tunnel.url}")
            while time.time() - t_fn < TIMEOUT_S - 120 and not ctl.get("stop") and proc.poll() is None:
                time.sleep(5)
        att["server_alive_at_end"] = proc.poll() is None
        stop_pub.set()
        ctl["status"] = "stopping"
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
        break
    out["gpu_function_seconds"] = round(time.time() - t_fn, 1)
    ctl["status"] = "done"
    return out


@app.local_entrypoint()
def main() -> None:
    import json

    t0 = time.time()
    res = serve.remote()
    res["wall_seconds"] = round(time.time() - t0, 1)
    p = Path(__file__).resolve().parent.parent / "artifacts" / "serve" / "spark_server.json"
    p.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "attempts"}, indent=1), flush=True)
