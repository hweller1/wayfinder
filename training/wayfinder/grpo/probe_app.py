"""One-H100 probe for the GRPO rollout design (about $3): runtime LoRA hot-load API, logprob mode, and LoRA decode overhead.

    modal run wayfinder/grpo/probe_app.py --out artifacts/grpo/probe.json

Answers before the rollout app is built: (1) does `/v1/load_lora_adapter` + `/v1/unload_lora_adapter` work at runtime with a
peft adapter from the SFT trainer; (2) are returned logprobs temperature-processed (`--logprobs-mode processed_logprobs`)
or raw; (3) do logprobs align with `return_token_ids`; (4) decode throughput at concurrency 32, base vs adapter, same server.
App name wayfinder-grpo-probe (watcher ops/grpo_watch.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
app = modal.App("wayfinder-grpo-probe")
weights = modal.Volume.from_name("wayfinder-weights")
data_vol = modal.Volume.from_name("wayfinder-data")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install("vllm==0.32.0", "huggingface_hub", "jsonschema", "httpx==0.28.1")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"})
)
FLAGS = ["--served-model-name", "policy", "--dtype", "bfloat16", "--gpu-memory-utilization", "0.92", "--language-model-only",
         "--enable-prefix-caching", "--max-num-seqs", "64", "--port", "8000", "--max-model-len", "65536",
         "--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer", "--enable-auto-tool-choice",
         "--enable-lora", "--max-lora-rank", "64", "--max-loras", "2", "--max-cpu-loras", "4", "--logprobs-mode", "processed_logprobs"]


@app.function(image=image, gpu="H100", volumes={"/weights": weights, "/data": data_vol}, timeout=45 * 60, cpu=8, memory=65536)
def probe(model: str, adapter: str) -> dict:
    import shutil
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    out: dict = {"model": model, "adapter": adapter}
    data_vol.reload()
    t0 = time.time()
    log = open("/tmp/vllm.log", "wb")
    proc = subprocess.Popen(["vllm", "serve", model, *FLAGS], stdout=log, stderr=subprocess.STDOUT)
    while True:
        if proc.poll() is not None:
            out["fatal"] = "vllm exited: " + "\n".join(Path("/tmp/vllm.log").read_text(errors="replace").splitlines()[-30:])
            return out
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            break
        except Exception:  # noqa: BLE001
            time.sleep(3)
        if time.time() - t0 > 1500:
            proc.kill()
            out["fatal"] = "not healthy"
            return out
    out["load_s"] = round(time.time() - t0, 1)

    def post(path: str, body: dict, timeout: float = 600) -> tuple[int, dict]:
        req = urllib.request.Request(f"http://127.0.0.1:8000{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                txt = r.read()
                return r.status, (json.loads(txt) if txt.strip().startswith(b"{") or txt.strip().startswith(b"[") else {"raw": txt.decode()[:300]})
        except urllib.error.HTTPError as e:
            return e.code, {"error": e.read().decode()[:500]}

    shutil.copytree(adapter, "/tmp/ad1")
    out["load_adapter"] = post("/v1/load_lora_adapter", {"lora_name": "pol1", "lora_path": "/tmp/ad1"})
    out["models"] = [m["id"] for m in json.loads(urllib.request.urlopen("http://127.0.0.1:8000/v1/models").read())["data"]]
    # same adapter under a second name (stands in for the next step's adapter) and unload of the first
    out["load_adapter2"] = post("/v1/load_lora_adapter", {"lora_name": "pol2", "lora_path": "/tmp/ad1"})
    out["unload_adapter1"] = post("/v1/unload_lora_adapter", {"lora_name": "pol1"})
    out["models_after"] = [m["id"] for m in json.loads(urllib.request.urlopen("http://127.0.0.1:8000/v1/models").read())["data"]]

    msgs = [{"role": "system", "content": "You are a code search agent."}, {"role": "user", "content": "Explain, in about 100 words, how a hash map handles collisions."}]
    base_req = {"messages": msgs, "max_tokens": 48, "temperature": 0.85, "top_p": 0.95, "logprobs": True, "return_token_ids": True, "seed": 3}
    st, r = post("/v1/chat/completions", {**base_req, "model": "policy"})
    ch = r["choices"][0]
    out["lp_alignment"] = {"status": st, "n_token_ids": len(ch.get("token_ids") or []), "n_logprobs": len((ch.get("logprobs") or {}).get("content") or []),
                           "finish": ch.get("finish_reason"), "first_lp": [round(x["logprob"], 4) for x in ((ch.get("logprobs") or {}).get("content") or [])[:6]],
                           "has_prompt_token_ids": bool(r.get("prompt_token_ids"))}
    lp = {}
    for T in (0.85, 1.0):
        st, r = post("/v1/chat/completions", {"model": "policy", "messages": msgs, "max_tokens": 1, "temperature": T, "logprobs": True, "top_logprobs": 5, "seed": 3})
        c = r["choices"][0]["logprobs"]["content"][0]
        lp[str(T)] = {"tok": c["token"], "lp": round(c["logprob"], 5), "top": [(t["token"], round(t["logprob"], 4)) for t in c["top_logprobs"]]}
    out["lp_vs_temperature"] = lp  # equal across T => raw logprobs; different => temperature-processed
    # adapter vs base: sampled logprob on the same prefix differs?
    st, r = post("/v1/chat/completions", {"model": "pol2", "messages": msgs, "max_tokens": 1, "temperature": 0.85, "logprobs": True, "top_logprobs": 5, "seed": 3})
    c = r["choices"][0]["logprobs"]["content"][0]
    out["adapter_first_token"] = {"tok": c["token"], "lp": round(c["logprob"], 5), "status": st}

    filler = ("def handler(request):\n    value = request.args.get('key')\n    return process(value)\n" * 6 + "\n") * 40  # ~7k tokens of prompt
    def one(model: str, i: int) -> tuple[int, float]:
        t = time.time()
        st, r = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": f"[{i}] {filler}\nSummarise the code above."}],
                                              "max_tokens": 256, "min_tokens": 256, "temperature": 0.85, "top_p": 0.95, "logprobs": True}, timeout=900)
        return (r.get("usage") or {}).get("completion_tokens", 0), time.time() - t

    def bench(model: str, conc: int = 32) -> dict:
        t = time.time()
        with ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(lambda i: one(model, i + hash(model) % 1000), range(conc)))
        wall = time.time() - t
        toks = sum(a for a, _ in res)
        return {"conc": conc, "wall_s": round(wall, 1), "gen_tokens": toks, "tok_per_s": round(toks / wall, 1), "p50_req_s": round(sorted(b for _, b in res)[len(res) // 2], 1)}

    bench("policy", 8)  # warmup
    out["bench_base_1"] = bench("policy")
    out["bench_lora"] = bench("pol2")
    out["bench_base_2"] = bench("policy")
    out["bench_lora_2"] = bench("pol2")
    txt = Path("/tmp/vllm.log").read_text(errors="replace")
    out["log_lora"] = [ln[:300] for ln in txt.splitlines() if "ora" in ln][:12]
    proc.terminate()
    return out


@app.local_entrypoint()
def main(out: str = "artifacts/grpo/probe.json", model: str = "/data/sft/v3_merged", adapter: str = "/data/opd/smoke_v2b/adapter"):
    res = probe.remote(model, adapter)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])
