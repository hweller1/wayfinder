"""Serve wayfinder-glimmer (hweller1/wayfinder-glimmer-30b) on your own Modal account.

    modal deploy deploy/modal/serve_glimmer.py      # run from the repository root
    modal serve deploy/modal/serve_glimmer.py       # temporary dev endpoint instead

One H100, vLLM 0.32.0, an OpenAI-compatible API under `/v1`, served model name `wayfinder-glimmer`. These are
the version and flags of the app that served every published result (`training/serve/wayfinder_v3.py`), except
that this one serves the Hugging Face weights and always starts vLLM the plain way (no GPU memory snapshot).

The URL is `https://<your-workspace>--wayfinder-glimmer-serve.modal.run` (the deploy prints it). The weights are
public, so no Hugging Face token is needed; they are downloaded once into the Modal Volume
`wayfinder-glimmer-weights`, and vLLM's compile caches go to `wayfinder-glimmer-cache`.

Auth: every `/v1` request needs `Authorization: Bearer <key>`. The key is `WAYFINDER_API_KEY` in the Modal secret
`wayfinder-glimmer-key` (create it with `deploy/modal/make_secrets.py`). vLLM enforces it itself.

Cost: the app scales to zero (`min_containers` 0) after `WAYFINDER_SCALEDOWN_S` idle seconds, and an idle app
costs nothing. While a container is up you pay for one H100 (about $3.95 per hour). The first request after idle
waits several minutes while the container starts and vLLM loads the model; the first ever start also downloads
about 60 GB of weights and compiles kernels, and takes longer.

`GET <url>-warm` with the bearer key is the warm endpoint: it answers at once and starts a container in the
background, so the model is loading while the user types. The gateway calls it for `scout warm`.

Deploy-time knobs (environment of the machine that runs `modal deploy`):
    WAYFINDER_MIN_CONTAINERS (0)   1 keeps the GPU warm and billed; redeploy with 0 to go back
    WAYFINDER_MAX_CONTAINERS (1)   GPUs the app may scale to
    WAYFINDER_SCALEDOWN_S (600)    idle seconds before a container stops
    WAYFINDER_APP_NAME, WAYFINDER_URL_LABEL (wayfinder-glimmer-serve)
    WAYFINDER_SECRET (wayfinder-glimmer-key), WAYFINDER_MODEL (hweller1/wayfinder-glimmer-30b)
    WAYFINDER_REVISION (unset: the latest commit of the model repository)
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = os.environ.get("WAYFINDER_APP_NAME", "wayfinder-glimmer-serve")
URL_LABEL = os.environ.get("WAYFINDER_URL_LABEL", APP_NAME)
MODEL_ID = os.environ.get("WAYFINDER_MODEL", "hweller1/wayfinder-glimmer-30b")
REVISION = os.environ.get("WAYFINDER_REVISION") or None
SERVED_NAME = "wayfinder-glimmer"
SECRET_NAME = os.environ.get("WAYFINDER_SECRET", "wayfinder-glimmer-key")  # holds WAYFINDER_API_KEY
VLLM_VERSION = "0.32.0"
PORT = 8000
MAX_CONCURRENCY = 32
MIN_CONTAINERS = int(os.environ.get("WAYFINDER_MIN_CONTAINERS", "0"))
MAX_CONTAINERS = int(os.environ.get("WAYFINDER_MAX_CONTAINERS", "1"))
SCALEDOWN_S = int(os.environ.get("WAYFINDER_SCALEDOWN_S", "600"))
WARMUP_TOKENS = 20000
BURST = 16  # concurrent requests in the warm-up burst
WEIGHTS_ROOT = "/models"
CACHE = "/cache"
HERE = Path(__file__).resolve()
SERVE_FILES = HERE.parents[2] / "training" / "serve" if len(HERE.parents) > 2 else HERE.parent  # warm-up inputs

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("wayfinder-glimmer-weights", create_if_missing=True)
cache = modal.Volume.from_name("wayfinder-glimmer-cache", create_if_missing=True)

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
        # Persistent compile caches on the cache Volume (reused across cold starts).
        "VLLM_CACHE_ROOT": f"{CACHE}/vllm",
        "TORCHINDUCTOR_CACHE_DIR": f"{CACHE}/inductor",
        "TRITON_CACHE_DIR": f"{CACHE}/triton",
        "FLASHINFER_WORKSPACE_BASE": f"{CACHE}/flashinfer",
        "CUDA_CACHE_PATH": f"{CACHE}/nv",
        "CUDA_CACHE_MAXSIZE": str(4 << 30),
        # The container re-imports this file without the deploy-time environment, so bake the names in.
        "WAYFINDER_APP_NAME": APP_NAME,
        "WAYFINDER_URL_LABEL": URL_LABEL,
        "WAYFINDER_MODEL": MODEL_ID,
        "WAYFINDER_SECRET": SECRET_NAME,
        **({"WAYFINDER_REVISION": REVISION} if REVISION else {}),
    })
    .add_local_file(SERVE_FILES / "scout_product_prompt.md", "/app/scout_product_prompt.md")
    .add_local_file(SERVE_FILES / "scout_tool_specs.json", "/app/scout_tool_specs.json")
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

SAMPLING = {"temperature": 0.85, "top_p": 0.95}  # the protocol the model was evaluated under


def weights_dir() -> str:
    return f"{WEIGHTS_ROOT}/{MODEL_ID.split('/')[-1]}"


def vllm_command(model_path: str, port: int = PORT) -> list[str]:
    return ["vllm", "serve", model_path, *VLLM_ARGS, "--port", str(port)]


def ensure_weights() -> str:
    """Download the public weights into the Volume once. A marker file written last makes an interrupted
    download resume instead of being served half-finished."""
    target = Path(weights_dir())
    marker = target / ".download_complete"
    if not marker.exists():
        from huggingface_hub import snapshot_download

        print(f"downloading {MODEL_ID} to {target} (first start only)", flush=True)
        snapshot_download(MODEL_ID, local_dir=str(target), revision=REVISION)
        marker.write_text("ok")
        weights.commit()
    return str(target)


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
    """Send real-shape requests so JIT kernels, CUDA graphs and the prefix cache are hot before serving.

    The shapes mirror Scout's explore loop: its system prompt and tool schemas as the shared prefix, then a long
    prompt with a tool result in context, a brief that forces a tool call, a short chat, and a concurrent burst.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor

    system = Path("/app/scout_product_prompt.md").read_text()
    tools = json.loads(Path("/app/scout_tool_specs.json").read_text())
    t0 = time.time()
    sample = "".join(_filler(i) for i in range(200))
    ntok = _post(port, "/tokenize", {"model": SERVED_NAME, "prompt": sample}, key)["count"]
    bulk = "".join(_filler(i) for i in range(int(200 * target_tokens / max(ntok, 1))))
    out: dict = {}

    def chat(name: str, messages: list, **kw) -> None:
        t = time.time()
        r = _post(port, "/v1/chat/completions", {"model": SERVED_NAME, "messages": messages, **SAMPLING, **kw}, key)
        u = r.get("usage", {})
        out[name] = {"secs": round(time.time() - t, 2), "prompt_tokens": u.get("prompt_tokens"),
                     "completion_tokens": u.get("completion_tokens")}  # fmt: skip

    brief = "Brief: Find where the HTTP retry backoff is computed and where its maximum delay is configured."
    call = {"id": "call_warm0", "type": "function", "function": {"name": "grep", "arguments": json.dumps({"pattern": "backoff"})}}
    long_msgs = [
        {"role": "system", "content": system}, {"role": "user", "content": brief},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_warm0", "content": bulk},
    ]  # fmt: skip
    chat("long_prefill", long_msgs, tools=tools, tool_choice="auto", max_tokens=24)
    chat("long_prefill_cached", long_msgs, tools=tools, tool_choice="auto", max_tokens=8)  # prefix-cache hit
    chat("tool_call", [{"role": "system", "content": system}, {"role": "user", "content": brief}], tools=tools,
         tool_choice="auto", max_tokens=1500)  # fmt: skip
    chat("short", [{"role": "user", "content": "Reply with the single word: ready."}], max_tokens=16)

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


def wait_healthy(proc: subprocess.Popen, port: int, timeout_s: float = 20 * 60) -> None:
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("vllm did not become healthy")


@app.cls(
    image=image,
    gpu="H100",
    volumes={WEIGHTS_ROOT: weights, CACHE: cache},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    cpu=(4, 8),
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_S,
    timeout=60 * 60,
    startup_timeout=30 * 60,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=MAX_CONCURRENCY)
class Glimmer:
    @modal.enter()
    def start_engine(self) -> None:
        """Fetch the weights if needed, start vLLM with the bearer key, warm it on real shapes."""
        key = os.environ["WAYFINDER_API_KEY"]
        path = ensure_weights()
        env = {**os.environ, "VLLM_API_KEY": key}  # vLLM reads the key from the environment, not the command line
        self.proc = subprocess.Popen(vllm_command(path), env=env)  # vLLM's output goes to the container log
        wait_healthy(self.proc, PORT)
        try:
            warm_server(PORT, key)
        except Exception as e:  # a failed warm-up costs speed, not correctness
            print(f"warm-up failed: {e!r}", flush=True)
        try:
            cache.commit()
        except Exception as e:  # never take the server down over a cache commit
            print(f"cache commit failed: {e!r}", flush=True)

    @modal.web_server(port=PORT, startup_timeout=30 * 60, label=URL_LABEL)
    def serve(self) -> None:
        pass  # vLLM is already listening on PORT

    @modal.method()
    def ping(self) -> str:
        """No-op input: spawning it makes Modal start (or keep) a container without waiting for it."""
        return "ok"

    @modal.exit()
    def stop(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


# /warm: one non-blocking call (a SessionStart hook or the gateway) to begin the cold start.
#   curl -s -m 5 -H "Authorization: Bearer $WAYFINDER_API_KEY" https://<workspace>--<URL_LABEL>-warm.modal.run
# Runs on a tiny CPU function, so it returns in well under a second while spawning a no-op input on the GPU
# class; Modal boots a container for it. Same bearer key as the model API.
_warm_image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=_warm_image, secrets=[modal.Secret.from_name(SECRET_NAME)], timeout=60)
@modal.asgi_app(label=f"{URL_LABEL}-warm")
def warm():
    """Plain ASGI app: GET with the bearer key spawns a no-op GPU-class input and returns 200 at once."""
    import hmac
    import json

    async def app_(scope, receive, send):
        if scope["type"] != "http":
            return
        auth = dict(scope["headers"]).get(b"authorization", b"")
        if not hmac.compare_digest(auth, f"Bearer {os.environ['WAYFINDER_API_KEY']}".encode()):
            status, body = 401, {"error": "Unauthorized"}
        else:
            modal.Cls.from_name(APP_NAME, "Glimmer")().ping.spawn()  # returns at once; the GPU container boots in the background
            status, body = 200, {"warming": True}
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(body).encode()})

    return app_
