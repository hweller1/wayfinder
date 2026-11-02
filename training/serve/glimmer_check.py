"""Day-1 serving check for meta-models/Muse-Glimmer-30B on Modal (one H100, hard cap $15).

Run (ephemeral app, torn down when the local entrypoint exits; never `modal deploy`):

    modal run serve/glimmer_check.py

Steps: CPU function snapshot_downloads the weights into Volume `wayfinder-weights` (skipped on later
runs), then one GPU function serves them with vLLM (BF16, max_model_len 32768, text-only) and measures
cold load time, tool-call parse rate (scout's tool schemas, tool_choice=auto) and throughput. Results
land in artifacts/serve/glimmer_check.json. No HF token is needed: the model is public.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "wayfinder-glimmer-check"
MODEL_ID = "meta-models/Muse-Glimmer-30B"
WEIGHTS_DIR = "/weights/Muse-Glimmer-30B"
VLLM_VERSION = os.environ.get("GLIMMER_VLLM_VERSION", "0.32.0")
H100_USD_PER_S = 0.001097  # Modal H100 list price (~$3.95/h)
GPU_TIMEOUT_S = 40 * 60  # hard cap on the GPU container; the CLI watcher stops the app at 50 min
HERE = Path(__file__).resolve().parent

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("wayfinder-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub", "jsonschema")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_LOGGING_LEVEL": "INFO"})
    .add_local_file(HERE / "scout_tool_specs.json", "/srv/scout_tool_specs.json")
    .add_local_file(HERE / "scout_product_prompt.md", "/srv/scout_product_prompt.md")
)

# Server configurations tried in order. The tool parser is vLLM's `muse_glimmer` (ATEM format); the
# second config adds the reasoning parser to see whether it changes tool-call extraction.
CONFIGS: list[dict[str, Any]] = [
    {"name": "tool_parser_only", "parsers": ["--tool-call-parser", "muse_glimmer"], "throughput": True},
    {
        "name": "tool_and_reasoning_parser",
        "parsers": ["--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer"],
        "throughput": False,
    },
]
BASE_FLAGS = [
    "--served-model-name", "glimmer",
    "--dtype", "bfloat16",
    "--max-model-len", "32768",
    "--gpu-memory-utilization", "0.92",
    "--language-model-only",
    "--enable-auto-tool-choice",
    "--max-num-seqs", "64",
    "--port", "8000",
]  # fmt: skip

BRIEFS = [
    "Find where the HTTP retry backoff is computed and where its maximum delay is configured.",
    "Which module validates the request body before the handler runs, and what does it reject?",
    "Locate the code that serialises a user record to JSON and every caller of it.",
    "Where is the database connection pool created and how is its size determined?",
    "Find the CLI entry point and the function that parses the --config flag.",
    "Which tests cover the cache eviction policy, and which source file do they exercise?",
    "Where does the logging format get configured, and which handlers does it install?",
    "Find the implementation of the rate limiter and the middleware that applies it.",
    "Locate the migration that adds the `archived_at` column and the model that uses it.",
    "Where are environment variables read, and which ones are required at startup?",
    "Find the function that computes the checksum of an uploaded file and what hash it uses.",
    "Which class implements the websocket reconnect logic and what is its backoff schedule?",
    "Locate the code that renders the error page for a 404 and the template it uses.",
    "Where is authentication token expiry checked, and what happens on expiry?",
    "Find the scheduler that runs periodic jobs and where the job list is registered.",
    "Which file defines the public API surface exported by the package, and what is excluded?",
    "Locate the retry decorator, its parameters, and two call sites that use it.",
    "Where does the build script decide which targets to compile, and how is that overridden?",
    "Find the parser for the config file format and the error raised on a malformed file.",
    "Which functions write to the audit log and under what conditions?",
]
FAKE_RESULTS = {
    "grep": "src/app/retry.py:41: def backoff(attempt, base=0.5, cap=30.0):\nsrc/app/http.py:88: delay = backoff(n)",
    "glob": "src/app/retry.py\nsrc/app/http.py\ntests/test_retry.py",
    "read_file": "src/app/retry.py lines 38-46\n38\t\n39\t\n40\t\n41\tdef backoff(attempt, base=0.5, cap=30.0):\n42\t    return min(cap, base * 2**attempt)",
    "semantic_search": "src/app/retry.py:41-46 def backoff(attempt, base=0.5, cap=30.0): ...",
    "report": "ok",
}


def _log(msg: str) -> None:
    print(f"[glimmer_check {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------------
# CPU-side weight download (cached in the volume; no GPU billed)
# --------------------------------------------------------------------------------------------------
@app.function(image=image, volumes={"/weights": weights}, timeout=30 * 60, cpu=4, memory=16384)
def download_weights() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    marker = Path(WEIGHTS_DIR) / ".download_complete"
    if marker.exists():
        return {"cached": True, "seconds": 0.0}
    snapshot_download(MODEL_ID, local_dir=WEIGHTS_DIR, max_workers=16)
    size = sum(f.stat().st_size for f in Path(WEIGHTS_DIR).rglob("*") if f.is_file())
    marker.write_text("ok")
    weights.commit()
    return {"cached": False, "seconds": round(time.time() - t0, 1), "bytes": size}


# --------------------------------------------------------------------------------------------------
# GPU function: serve + measure
# --------------------------------------------------------------------------------------------------
class Server:
    def __init__(self, name: str, parsers: list[str]) -> None:
        self.name = name
        self.log_path = f"/tmp/vllm_{name}.log"
        self.cmd = ["vllm", "serve", WEIGHTS_DIR, *BASE_FLAGS, *parsers]
        self.proc: subprocess.Popen[bytes] | None = None
        self.load_s: float | None = None

    def start(self, timeout_s: int = 20 * 60) -> None:
        import urllib.request

        t0 = time.time()
        logf = open(self.log_path, "wb")
        self.proc = subprocess.Popen(self.cmd, stdout=logf, stderr=subprocess.STDOUT)
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited rc={self.proc.returncode}\n{self.tail(60)}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                    if r.status == 200:
                        self.load_s = round(time.time() - t0, 1)
                        return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError(f"vllm not healthy after {timeout_s}s\n{self.tail(60)}")

    def tail(self, n: int) -> str:
        try:
            return "\n".join(Path(self.log_path).read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def log_grep(self, *needles: str, limit: int = 12) -> list[str]:
        try:
            lines = Path(self.log_path).read_text(errors="replace").splitlines()
        except OSError:
            return []
        return [ln[:300] for ln in lines if any(n in ln for n in needles)][:limit]

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        time.sleep(5)


def _client() -> Any:
    from openai import OpenAI

    return OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x", timeout=600)


SAMPLING = {"temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 64}}


def _tool_correctness(tools: list[dict[str, Any]], system_prompt: str) -> dict[str, Any]:
    import jsonschema

    client = _client()
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    rows: list[dict[str, Any]] = []
    for i, brief in enumerate(BRIEFS):
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Brief: {brief}"},
        ]
        row: dict[str, Any] = {"i": i, "ok": False}
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model="glimmer", messages=msgs, tools=tools, tool_choice="auto", max_tokens=3000, **SAMPLING
            )
            row["secs"] = round(time.time() - t0, 2)
            ch = r.choices[0]
            m = ch.message
            row["finish_reason"] = ch.finish_reason
            row["completion_tokens"] = r.usage.completion_tokens if r.usage else None
            row["n_tool_calls"] = len(m.tool_calls or [])
            content = m.content or ""
            row["leaked_markup"] = any(s in content for s in ("<atem:", "<|message|>", "to=self"))
            row["has_content"] = bool(content.strip())
            row["has_reasoning"] = bool(getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None))
            if m.tool_calls:
                tc = m.tool_calls[0]
                name = tc.function.name
                row["tool"] = name
                try:
                    args = json.loads(tc.function.arguments)
                    row["args_json"] = True
                    if name in schemas:
                        jsonschema.validate(args, schemas[name])
                        row["schema_valid"] = True
                        row["ok"] = True
                    else:
                        row["schema_valid"] = False
                        row["error"] = f"unknown tool {name!r}"
                except json.JSONDecodeError as e:
                    row["args_json"] = False
                    row["error"] = f"bad json: {e}"[:200]
                except jsonschema.ValidationError as e:
                    row["schema_valid"] = False
                    row["error"] = f"schema: {e.message}"[:200]
                # second turn: feed a tool result back, checks the template round trip with history
                if row["ok"]:
                    amsg = {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": name, "arguments": tc.function.arguments},
                            }
                        ],
                    }
                    msgs2 = [*msgs, amsg, {"role": "tool", "tool_call_id": tc.id, "content": FAKE_RESULTS[name]}]
                    try:
                        r2 = client.chat.completions.create(
                            model="glimmer", messages=msgs2, tools=tools, tool_choice="auto", max_tokens=3000, **SAMPLING
                        )
                        m2 = r2.choices[0].message
                        row["turn2_ok"] = True
                        row["turn2_tool_calls"] = len(m2.tool_calls or [])
                        row["turn2_tool"] = m2.tool_calls[0].function.name if m2.tool_calls else None
                    except Exception as e:
                        row["turn2_ok"] = False
                        row["turn2_error"] = repr(e)[:300]
            else:
                row["sample_content"] = content[:300]
        except Exception as e:
            row["error"] = repr(e)[:300]
        rows.append(row)
    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    t2 = [r for r in rows if "turn2_ok" in r]
    return {
        "n_requests": n,
        "parse_success": ok,
        "parse_success_rate": round(ok / n, 3),
        "any_tool_call": sum(1 for r in rows if r.get("n_tool_calls")),
        "leaked_markup": sum(1 for r in rows if r.get("leaked_markup")),
        "turn2_ok": sum(1 for r in t2 if r["turn2_ok"]),
        "turn2_n": len(t2),
        "tools_used": sorted({r["tool"] for r in rows if r.get("tool")}),
        "rows": rows,
    }


def _make_prompt(tok: Any, n_tokens: int, seed: int) -> str:
    import random

    rnd = random.Random(seed)
    words = [
        "alpha", "bridge", "cipher", "delta", "ember", "forest", "granite", "harbor", "island", "jungle",
        "kernel", "lantern", "meadow", "nebula", "orchid", "prairie", "quartz", "river", "summit", "timber",
        "umbra", "valley", "willow", "xenon", "yonder", "zephyr", "function", "buffer", "socket", "parser",
    ]  # fmt: skip
    text = " ".join(rnd.choice(words) + str(rnd.randint(0, 9999)) for _ in range(int(n_tokens * 0.6)))
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return tok.decode(ids)


def _stream_once(client: Any, prompt: str, max_tokens: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    t_first = None
    usage = None
    n_chunks = 0
    stream = client.chat.completions.create(
        model="glimmer",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and (chunk.choices[0].delta.content or getattr(chunk.choices[0].delta, "reasoning", None)):
            n_chunks += 1
            if t_first is None:
                t_first = time.perf_counter()
    t_end = time.perf_counter()
    ct = usage.completion_tokens if usage else n_chunks
    pt = usage.prompt_tokens if usage else None
    first = t_first or t_end
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_s": round(first - t0, 3),
        "decode_tok_s": round((ct - 1) / (t_end - first), 2) if ct > 1 and t_end > first else None,
        "total_s": round(t_end - t0, 3),
    }


def _throughput(tok: Any) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    client = _client()
    out: dict[str, Any] = {}
    _stream_once(client, _make_prompt(tok, 200, 999), 32)  # warmup
    for label, n in (("8k", 8000), ("24k", 24000)):
        reps = []
        for rep in range(3):
            reps.append(_stream_once(client, _make_prompt(tok, n, 1000 * n + rep), 256))
        out[f"single_stream_{label}"] = {
            "runs": reps,
            "ttft_s_median": statistics.median(r["ttft_s"] for r in reps),
            "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in reps if r["decode_tok_s"]),
        }
        _log(f"single_stream_{label}: {out[f'single_stream_{label}']['ttft_s_median']}s ttft")
    reps = [_stream_once(client, _make_prompt(tok, 100, 5000 + r), 256) for r in range(3)]
    out["single_stream_short"] = {
        "runs": reps,
        "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in reps if r["decode_tok_s"]),
    }
    for conc in (8, 32):
        runs = []
        for rep in range(2):
            prompts = [_make_prompt(tok, 100, 70000 + conc * 100 + rep * 1000 + j) for j in range(conc)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(conc) as ex:
                res = list(ex.map(lambda p: _stream_once(client, p, 256), prompts))
            wall = time.perf_counter() - t0
            total = sum(r["completion_tokens"] for r in res)
            runs.append(
                {
                    "wall_s": round(wall, 2),
                    "completion_tokens": total,
                    "aggregate_tok_s": round(total / wall, 1),
                    "ttft_s_median": statistics.median(r["ttft_s"] for r in res),
                    "per_stream_decode_tok_s_median": statistics.median(
                        r["decode_tok_s"] for r in res if r["decode_tok_s"]
                    ),
                }
            )
        out[f"concurrency_{conc}"] = {
            "runs": runs,
            "aggregate_tok_s_best": max(r["aggregate_tok_s"] for r in runs),
            "aggregate_tok_s_last": runs[-1]["aggregate_tok_s"],
        }
        _log(f"concurrency_{conc}: {out[f'concurrency_{conc}']['aggregate_tok_s_last']} tok/s")
    return out


@app.function(image=image, gpu="H100", volumes={"/weights": weights}, timeout=GPU_TIMEOUT_S)
def run_check() -> dict[str, Any]:
    import importlib.metadata as md

    from transformers import AutoTokenizer

    t_start = time.time()
    tools = json.load(open("/srv/scout_tool_specs.json"))
    system_prompt = Path("/srv/scout_product_prompt.md").read_text()
    result: dict[str, Any] = {
        "vllm_version": md.version("vllm"),
        "torch_version": md.version("torch"),
        "transformers_version": md.version("transformers"),
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "base_flags": BASE_FLAGS,
        "configs": {},
        "errors": [],
    }  # fmt: skip
    tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    for cfg in CONFIGS:
        srv = Server(cfg["name"], cfg["parsers"])
        entry: dict[str, Any] = {"cmd": " ".join(srv.cmd), "parsers": cfg["parsers"]}
        result["configs"][cfg["name"]] = entry
        try:
            _log(f"starting {cfg['name']}: {entry['cmd']}")
            srv.start()
            entry["cold_load_s"] = srv.load_s
            entry["log_highlights"] = srv.log_grep(
                "Loading weights took", "Model loading took", "GPU KV cache size", "Maximum concurrency",
                "torch.compile takes", "Capturing CUDA graph", "Using", "Attention backend",
            )  # fmt: skip
            _log(f"healthy after {srv.load_s}s")
            entry["tool_calls"] = _tool_correctness(tools, system_prompt)
            _log(
                f"tool parse {entry['tool_calls']['parse_success']}/{entry['tool_calls']['n_requests']} "
                f"(any tool call {entry['tool_calls']['any_tool_call']})"
            )
            if cfg["throughput"]:
                entry["throughput"] = _throughput(tok)
        except Exception as e:
            entry["error"] = repr(e)[:2000]
            entry["log_tail"] = srv.tail(60)
            result["errors"].append(f"{cfg['name']}: {repr(e)[:500]}")
            _log(f"FAILED {cfg['name']}: {e!r}")
        finally:
            srv.stop()
        if "error" in entry and cfg is CONFIGS[0] and "cold_load_s" not in entry:
            break  # server never came up: a second config will not help, stop paying for the GPU
    result["gpu_function_seconds"] = round(time.time() - t_start, 1)
    return result


@app.local_entrypoint()
def main() -> None:
    t0 = time.time()
    dl = download_weights.remote()
    print("download:", dl, flush=True)
    res = run_check.remote()
    res["download"] = dl
    res["model"] = MODEL_ID
    res["h100_usd_per_s"] = H100_USD_PER_S
    # GPU billed from container start (image pull + volume mount + server) to function end
    res["gpu_cost_estimate_usd"] = round(res["gpu_function_seconds"] * H100_USD_PER_S, 2)
    res["wall_seconds_total"] = round(time.time() - t0, 1)
    out = HERE.parent / "artifacts" / "serve" / "glimmer_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}", flush=True)
    print(json.dumps({k: v for k, v in res.items() if k != "configs"}, indent=2), flush=True)
