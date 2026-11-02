"""Serve wayfinder SFT v3 (post-trained Muse-Glimmer-30B, merged bf16 at wayfinder-data:/sft/v3_merged) as an
OpenAI-compatible endpoint on Modal, 1x H100, behind the same bearer-key proxy as serve/a1_35b.py (this file is
that pattern with the v3 model, the `muse_glimmer` tool and reasoning parsers, and its own secret).

    modal deploy serve/wayfinder_v3.py            # app wayfinder-v3-serve, model name `wayfinder-v3`
    modal serve serve/wayfinder_v3.py             # ephemeral dev endpoint (validation)
    scripts/v3_warm.sh on|off                     # redeploy with min_containers 1 / 0

Deploy-time knobs (env): V3_MIN_CONTAINERS (0), V3_MAX_CONTAINERS (1), V3_SNAPSHOT (1 default; 0 = plain start with
persistent compile caches, 1 = Modal CPU+GPU memory snapshot with vLLM sleep mode), V3_APP_NAME, V3_URL_LABEL,
V3_SCALEDOWN_S (600), V3_WEIGHTS (/data/sft/v3_merged), V3_SERVED_NAME (wayfinder-v3), V3_SECRET (wayfinder-v3-key).
Another merged checkpoint is served by config alone, same flags, parsers, proxy and scale to zero, for example the GRPO arm a
step-20 checkpoint (do not use a `wayfinder-grpo*` app name: ops/grpo_watch.py stops apps with that prefix):

    V3_APP_NAME=wayfinder-a20-serve V3_URL_LABEL=wayfinder-a20-serve V3_WEIGHTS=/data/grpo/a/s20_merged \\
      V3_SERVED_NAME=wayfinder-grpo-a20 V3_MIN_CONTAINERS=1 modal deploy serve/wayfinder_v3.py

Auth: Modal secret `wayfinder-v3-key` (key V3_API_KEY); client side V3_BASE_URL / V3_API_KEY in .env.

The server scales to zero after `scaledown_window` idle seconds, so a deployed app costs nothing while
idle. Plain mode: vLLM reads the bearer token from VLLM_API_KEY. Snapshot mode: a1_proxy.py enforces it.
Send temperature 0.85 and top_p 0.95 per request (the SFT eval protocol); no top_k or presence_penalty.

Cold-start work: persistent torch.compile / inductor / triton / flashinfer caches under /cache/v3 on Volume `wayfinder-cache`,
FlashInfer kernels pre-installed in the image, phase timings parsed from vLLM's startup log (written to
/cache/startup/*.json on the Volume), and a real-shape warmup (scout system prompt + tool schemas, ~20k-token
prompt, a tool call, a short call) before the server is marked ready. With V3_SNAPSHOT=1 the engine is started,
warmed and put to sleep (vLLM sleep level 1) inside `@modal.enter(snap=True)`; restore just wakes it up.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = os.environ.get("V3_APP_NAME", "wayfinder-v3-serve")
URL_LABEL = os.environ.get("V3_URL_LABEL", "wayfinder-v3-serve")  # keeps the existing public URL stable
WEIGHTS_DIR = os.environ.get("V3_WEIGHTS", "/data/sft/v3_merged")
SERVED_NAME = os.environ.get("V3_SERVED_NAME", "wayfinder-v3")
SECRET_NAME = os.environ.get("V3_SECRET", "wayfinder-v3-key")  # holds V3_API_KEY, the bearer key the proxy enforces
VLLM_VERSION = os.environ.get("V3_VLLM_VERSION", "0.32.0")
PORT = 8000  # public port Modal probes
UPSTREAM_PORT = 8001  # vLLM itself when the auth/allow-list proxy sits in front (snapshot mode)
MAX_CONCURRENCY = 32
MAX_CONTAINERS = int(os.environ.get("V3_MAX_CONTAINERS", "1"))
SCALEDOWN_S = int(os.environ.get("V3_SCALEDOWN_S", "600"))
MIN_CONTAINERS = int(os.environ.get("V3_MIN_CONTAINERS", "0"))  # read at deploy time; see scripts/v3_warm.sh
SNAPSHOT = os.environ.get("V3_SNAPSHOT", "1") == "1"  # default on: Modal GPU memory snapshot (A1 technique)
WARMUP_TOKENS = int(os.environ.get("V3_WARMUP_TOKENS", "20000"))
BURST = 16  # concurrent requests in the warmup burst
CACHE = "/cache"
CACHE_SUB = f"{CACHE}/v3"  # subdir of the shared compile-cache Volume
HERE = Path(__file__).resolve().parent

app = modal.App(APP_NAME)
data = modal.Volume.from_name("wayfinder-data")
cache = modal.Volume.from_name("wayfinder-cache", create_if_missing=True)
# Read-only cache mount (default once the caches are populated): a snapshot is only reused while the mounted Volumes are
# unchanged, and every container that wrote to the writable cache Volume (startup records, new compile artifacts)
# committed a new version on exit, which forced a fresh snapshot build on the next scale-from-zero. Set
# V3_CACHE_RO=0 for the one deploy that has to (re)populate the caches after a vLLM or checkpoint change.
CACHE_RO = os.environ.get("V3_CACHE_RO", "0") == "1"
cache_mount = cache.read_only() if CACHE_RO else cache

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub")
    # vLLM warns that FlashInfer's precompiled kernels are missing and JIT-builds them at startup; fetch them
    # at image build instead (best effort: a failure here only means the JIT cache on the Volume is used).
    .run_commands("vllm download-kernels || echo 'download-kernels unavailable, falling back to JIT'")
    .env({
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        # Persistent compile caches on Volume `wayfinder-cache` (reused across cold starts).
        "VLLM_CACHE_ROOT": f"{CACHE_SUB}/vllm",
        "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_SUB}/inductor",
        "TRITON_CACHE_DIR": f"{CACHE_SUB}/triton",
        "FLASHINFER_WORKSPACE_BASE": f"{CACHE_SUB}/flashinfer" if not CACHE_RO else "/tmp/flashinfer",  # RO mount cannot take its JIT log
        "CUDA_CACHE_PATH": f"{CACHE_SUB}/nv",
        "CUDA_CACHE_MAXSIZE": str(4 << 30),
        # Modal's GPU-snapshot docs: inductor compile threads break snapshot creation.
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        # The container cannot see deploy-time env, so bake the mode in.
        "V3_SNAPSHOT": "1" if SNAPSHOT else "0",
        "V3_CACHE_RO": "1" if CACHE_RO else "0",
        "V3_WARMUP_TOKENS": str(WARMUP_TOKENS),
        "V3_WEIGHTS": WEIGHTS_DIR,
        "V3_SERVED_NAME": SERVED_NAME,
        "V3_SECRET": SECRET_NAME,
    })
    .add_local_file(HERE / "scout_product_prompt.md", "/app/scout_product_prompt.md")
    .add_local_file(HERE / "scout_tool_specs.json", "/app/scout_tool_specs.json")
    .add_local_file(HERE / "a1_proxy.py", "/app/a1_proxy.py")
)

# The checkpoint's own chat template is used (no --chat-template). Server-side sampling defaults cover
# what vLLM can default; presence_penalty must be sent by the client.
VLLM_ARGS = [
    "--served-model-name", SERVED_NAME,
    "--dtype", "bfloat16",
    "--max-model-len", "65536",
    "--gpu-memory-utilization", "0.92",
    "--enable-prefix-caching",
    "--language-model-only",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "muse_glimmer",
    "--reasoning-parser", "muse_glimmer",
    "--max-num-seqs", str(MAX_CONCURRENCY),
    "--host", "0.0.0.0",
]  # fmt: skip

SAMPLING = {"temperature": 0.85, "top_p": 0.95}  # the SFT eval protocol (the SFT v3 write-up, not in this release)

# --------------------------------------------------------------------------------------------------
# vLLM startup-log phase timing
# --------------------------------------------------------------------------------------------------
# First occurrence of each marker, stamped with local arrival time (the log's own timestamps are 1 s).
LOG_EVENTS = [
    ("engine_init", r"Initializing a V1 LLM engine"),
    ("weights_loaded", r"Model loading took"),
    ("compile_start", r"Starting torch\.compile"),
    ("compile_end", r"torch\.compile took"),
    ("kv_mem", r"Available KV cache memory"),
    ("kernel_warmup_start", r"JIT kernel warmup starting"),
    ("engine_ready", r"init engine \(profile, create kv cache, warmup model\) took"),
    ("server_listening", r"Starting vLLM server on"),
]  # fmt: skip
SELF_REPORTED = {
    "loading_weights_s": r"Loading weights took ([\d.]+) seconds",
    "torch_compile_s": r"torch\.compile took ([\d.]+) s",
    "graph_compile_s": r"Compiling a graph for compile range .* takes ([\d.]+) s",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
}  # fmt: skip


def phases_from_events(ev: dict[str, float], t_spawn: float) -> dict[str, float | None]:
    """Phase durations in seconds from event arrival times; None when a marker never appeared."""

    def d(a: str | None, b: str) -> float | None:
        ta = t_spawn if a is None else ev.get(a)
        tb = ev.get(b)
        return round(tb - ta, 2) if ta is not None and tb is not None else None

    after_load = "compile_end" if "compile_end" in ev else "weights_loaded"
    out = {
        "process_boot": d(None, "engine_init"),  # python imports, config resolution, engine process spawn
        "weights_load": d("engine_init", "weights_loaded"),
        "torch_compile": d("compile_start", "compile_end"),  # None on a warm cache (vLLM logs no start marker)
        "memory_profile": d(after_load, "kv_mem"),  # profiling run + cudagraph memory estimate
        "kv_cache_init": d("kv_mem", "kernel_warmup_start"),
        "kernel_warmup_cuda_graphs": d("kernel_warmup_start", "engine_ready"),
        "api_server_start": d("engine_ready", "server_listening"),
    }
    cc = [out["torch_compile"], out["kernel_warmup_cuda_graphs"]]
    out["compile_cuda_graphs"] = round(sum(x for x in cc if x), 2) if any(cc) else None
    return out


class VllmProc:
    """Runs `vllm serve`, mirrors its output to stdout, and records phase timings from the log."""

    def __init__(self, extra_args: list[str], env: dict[str, str], port: int):
        import re
        import threading

        self.port = port
        self.events: dict[str, float] = {}
        self.self_reported: dict[str, float] = {}
        self.t_spawn = time.time()
        self._re = {k: re.compile(p) for k, p in LOG_EVENTS}
        self._re_self = {k: re.compile(p) for k, p in SELF_REPORTED.items()}
        self.proc = subprocess.Popen(
            ["vllm", "serve", WEIGHTS_DIR, *VLLM_ARGS, "--port", str(port), *extra_args],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )  # fmt: skip
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:  # universal newlines: tqdm \r updates become lines
            now = time.time()
            for name, rx in self._re.items():
                if name not in self.events and rx.search(line):
                    self.events[name] = now
            for name, rx in self._re_self.items():
                m = rx.search(line)
                if m and name not in self.self_reported:
                    self.self_reported[name] = float(m.group(1))
            print(line, end="", flush=True)

    def wait_healthy(self, timeout_s: float = 20 * 60) -> float:
        import urllib.request

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return time.time()
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("vllm did not become healthy")

    def terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------------------------------
# Real-shape warmup
# --------------------------------------------------------------------------------------------------
def _post(port: int, path: str, payload: dict | None, key: str | None, timeout: float = 600) -> dict:
    import json
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def _filler(i: int) -> str:
    return (
        f"src/pkg/module_{i % 97}/handler_{i}.py:{(i * 7) % 400 + 1}: "
        f"def handle_{i}(request, ctx, retries={i % 7}): return ctx.dispatch(request, key='k{i}')\n"
    )


def warm_server(port: int, key: str | None, target_tokens: int = WARMUP_TOKENS) -> dict:
    """Issue real-shape requests so JIT kernels, CUDA graphs and the prefix cache are hot before serving.

    Shapes mirror scout's explore loop: its system prompt and tool schemas as the shared prefix, then (1) a
    ~target_tokens prompt with a tool result in context (long chunked prefill + prefix cache), (2) a fresh
    brief that forces a tool call (qwen3_coder parse path, reasoning parser, sampling kernels), (3) a short chat.
    """
    import json

    system = Path("/app/scout_product_prompt.md").read_text()
    tools = json.loads(Path("/app/scout_tool_specs.json").read_text())
    t0 = time.time()
    sample = "".join(_filler(i) for i in range(200))
    ntok = _post(port, "/tokenize", {"model": SERVED_NAME, "prompt": sample}, key)["count"]
    n_lines = int(200 * target_tokens / max(ntok, 1))
    bulk = "".join(_filler(i) for i in range(n_lines))
    out: dict = {}

    def chat(name: str, messages: list, **kw) -> None:
        t = time.time()
        r = _post(port, "/v1/chat/completions", {"model": SERVED_NAME, "messages": messages, **SAMPLING, **kw}, key)
        u = r.get("usage", {})
        m = r["choices"][0]["message"]
        out[name] = {
            "secs": round(time.time() - t, 2), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "tool_calls": len(m.get("tool_calls") or []),
        }  # fmt: skip

    brief = "Brief: Find where the HTTP retry backoff is computed and where its maximum delay is configured."
    call = {"id": "call_warm0", "type": "function", "function": {"name": "grep", "arguments": json.dumps({"pattern": "backoff"})}}
    long_msgs = [
        {"role": "system", "content": system}, {"role": "user", "content": brief},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_warm0", "content": bulk},
    ]  # fmt: skip
    chat("long_prefill", long_msgs, tools=tools, tool_choice="auto", max_tokens=24)
    chat("long_prefill_cached", long_msgs, tools=tools, tool_choice="auto", max_tokens=8)  # prefix-cache hit path
    chat("tool_call", [{"role": "system", "content": system}, {"role": "user", "content": brief}], tools=tools,
         tool_choice="auto", max_tokens=1500)  # fmt: skip
    chat("short", [{"role": "user", "content": "Reply with the single word: ready."}], max_tokens=16)
    # Concurrent burst: batch>1 sampling (top-k/top-p, gumbel) Triton kernels otherwise JIT on the first live
    # concurrent requests (vLLM's jit_monitor flags them), and decode CUDA graphs for mid-size batches get touched.
    from concurrent.futures import ThreadPoolExecutor

    def one(i: int) -> int:
        m = [{"role": "system", "content": system}, {"role": "user", "content": f"Brief {i}: where is handler_{i} defined and who calls it?"}]
        r = _post(port, "/v1/chat/completions", {"model": SERVED_NAME, "messages": m, "tools": tools, "tool_choice": "auto",
                                                 "max_tokens": 64, **SAMPLING}, key)
        return r["usage"]["completion_tokens"]

    tb = time.time()
    with ThreadPoolExecutor(BURST) as ex:
        toks = list(ex.map(one, range(BURST)))
    out["burst"] = {"n": BURST, "secs": round(time.time() - tb, 2), "completion_tokens": sum(toks)}
    out["total_s"] = round(time.time() - t0, 2)
    return out


def _start_proxy(key: str) -> object:
    env = {**os.environ, "A1_API_KEY": key, "A1_UPSTREAM_PORT": str(UPSTREAM_PORT), "A1_PORT": str(PORT)}
    return subprocess.Popen(["python", "/app/a1_proxy.py"], env=env)


def _write_startup_record(rec: dict) -> None:
    """Persist the record on the cache Volume (fetch with scripts/a1_collect_startup.py) and print it."""
    import json

    print("V3_STARTUP " + json.dumps(rec), flush=True)
    if CACHE_RO:
        return  # read-only cache mount: the record is in the container log only
    d = Path(CACHE) / "startup"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rec['utc'].replace(':', '')}_{rec['kind']}.json").write_text(json.dumps(rec, indent=2))


def _commit_cache_sync() -> None:
    try:
        cache.commit()
        print("V3_CACHE committed", flush=True)
    except Exception as e:
        print(f"V3_CACHE commit failed: {e!r}", flush=True)


def _commit_cache_bg() -> None:
    import threading

    def run() -> None:
        try:
            cache.commit()
            print("V3_CACHE committed", flush=True)
        except Exception as e:  # never take the server down over a cache commit
            print(f"V3_CACHE commit failed: {e!r}", flush=True)

    threading.Thread(target=run, daemon=True).start()


_cls_kwargs: dict = {}
if SNAPSHOT:
    _cls_kwargs = {
        "enable_memory_snapshot": True,
        "experimental_options": {"enable_gpu_snapshot": True},
        "memory": (32768, 98304),  # sleep level 1 parks the ~60 GiB of weights in host RAM during the snapshot
    }


@app.cls(
    image=image,
    gpu="H100",
    volumes={"/data": data, CACHE: cache_mount},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    cpu=(4, 8),  # the proxy and vLLM's API-server process share these with the engine's host thread
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_S,
    timeout=60 * 60,
    startup_timeout=20 * 60,
    max_containers=MAX_CONTAINERS,
    **_cls_kwargs,
)
@modal.concurrent(max_inputs=MAX_CONCURRENCY)
class V3:
    @modal.enter(snap=SNAPSHOT)
    def start_engine(self) -> None:
        """Plain mode: start vLLM, warm it, mark ready. Snapshot mode: same, then put the engine to sleep."""
        import datetime

        if not (Path(WEIGHTS_DIR) / "config.json").exists():
            raise RuntimeError(f"weights not found at {WEIGHTS_DIR} (Volume wayfinder-data)")
        self.t_enter = time.time()
        self.rec: dict = {"utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "vllm": VLLM_VERSION, "weights": WEIGHTS_DIR}
        if SNAPSHOT:
            # Dev mode exposes /sleep and /wake_up (and unauthenticated debug routes), so vLLM binds localhost
            # only and carries no key (a key baked into a snapshot would outlive secret rotation); the proxy
            # started after restore enforces the key and forwards only /v1/*, /health, /version, /ping.
            env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
            env.pop("VLLM_API_KEY", None)
            extra = ["--enable-sleep-mode"]
            port, key = UPSTREAM_PORT, None
            extra += ["--host", "127.0.0.1"]
        else:
            env = {**os.environ, "VLLM_API_KEY": os.environ["V3_API_KEY"]}
            extra, port, key = [], PORT, os.environ["V3_API_KEY"]
        self.port = port
        self.v = VllmProc(extra, env, port)
        t_healthy = self.v.wait_healthy()
        t_w0 = time.time()
        warm = warm_server(port, key)
        t_warm = time.time()
        rec = self.rec
        rec["kind"] = "snapshot_create" if SNAPSHOT else "cold_start"
        rec["phases_s"] = phases_from_events(self.v.events, self.v.t_spawn)
        rec["phases_s"]["health_wait_after_engine"] = round(t_healthy - self.v.events.get("server_listening", t_healthy), 2)
        rec["phases_s"]["app_warmup"] = round(t_warm - t_w0, 2)
        rec["phases_s"]["ready"] = round(t_warm - self.v.t_spawn, 2)
        rec["self_reported_s"] = self.v.self_reported
        rec["warmup"] = warm
        if SNAPSHOT:
            # Commit the compile caches only the first time: a Volume commit during snapshot creation changes the
            # mounted Volume's version, and a fresh snapshot was being rebuilt on every scale-from-zero.
            marker = Path(CACHE_SUB) / f".warm_{VLLM_VERSION}"
            if not CACHE_RO and not marker.exists():
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("ok")
                _commit_cache_sync()
            t_s = time.time()
            _post(port, "/sleep?level=1", None, key)
            rec["phases_s"]["sleep_before_snapshot"] = round(time.time() - t_s, 2)
            _write_startup_record(rec)
        else:
            _write_startup_record(rec)
            if not CACHE_RO:
                _commit_cache_bg()

    if SNAPSHOT:

        @modal.enter(snap=False)
        def restore_engine(self) -> None:
            t0 = time.time()
            _post(self.port, "/wake_up", None, None)
            self.v.wait_healthy(120)
            t_wake = time.time()
            key = os.environ["V3_API_KEY"]
            # KV cache (and prefix cache) was dropped by sleep: re-prime on real shapes through the engine port.
            warm = warm_server(self.port, None, target_tokens=min(WARMUP_TOKENS, 12000))
            self.proxy = _start_proxy(key)
            self._wait_proxy()
            _write_startup_record({
                "utc": __import__("datetime").datetime.now(__import__("datetime").UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "kind": "snapshot_restore",
                "phases_s": {"wake_up": round(t_wake - t0, 2), "app_warmup": round(time.time() - t_wake, 2),
                             "ready": round(time.time() - t0, 2)},
                "warmup": warm,
            })  # fmt: skip

        def _wait_proxy(self) -> None:
            import urllib.request

            for _ in range(100):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                    return
                except Exception:
                    time.sleep(0.2)
            raise RuntimeError("proxy did not come up")

    @modal.web_server(port=PORT, startup_timeout=15 * 60, label=URL_LABEL)
    def serve(self) -> None:
        pass  # vLLM (plain mode) or the proxy (snapshot mode) is already listening on PORT

    @modal.method()
    def ping(self) -> str:
        """No-op input: spawning it makes Modal start (or keep) a container without waiting for it."""
        return "ok"

    @modal.exit()
    def stop(self) -> None:
        if getattr(self, "v", None):
            self.v.terminate()


# --------------------------------------------------------------------------------------------------
# /warm: one non-blocking call a SessionStart hook can fire to begin the cold start.
#   curl -s -m 5 -H "Authorization: Bearer $V3_API_KEY" https://<workspace>--<URL_LABEL>-warm.modal.run
# Runs on a tiny CPU function (always up, ~free), so it returns in well under a second while spawning a
# no-op input on the GPU class; Modal boots (or restores) a container for it. Same bearer key as the model API.
# --------------------------------------------------------------------------------------------------
_warm_image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=_warm_image, secrets=[modal.Secret.from_name(SECRET_NAME)], timeout=60)
@modal.asgi_app(label=f"{URL_LABEL}-warm")
def warm():
    """Plain ASGI app (no framework): GET with the bearer key spawns a no-op GPU-class input and returns 200 at once."""
    import hmac
    import json

    async def app_(scope, receive, send):
        if scope["type"] != "http":
            return
        auth = dict(scope["headers"]).get(b"authorization", b"")
        if not hmac.compare_digest(auth, f"Bearer {os.environ['V3_API_KEY']}".encode()):
            status, body = 401, {"error": "Unauthorized"}
        else:
            modal.Cls.from_name(APP_NAME, "V3")().ping.spawn()  # returns at once; the GPU container boots in the background
            status, body = 200, {"warming": True}
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(body).encode()})

    return app_
