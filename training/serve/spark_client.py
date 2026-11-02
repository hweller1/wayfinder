"""Client-side checks against the Spark server started by serve/spark_open.py (URL read from the Modal Dict).

    python3 serve/spark_client.py              # wait for url, run checks, write artifacts/serve/spark_check.json
    python3 serve/spark_client.py --stop       # tell the GPU function to exit
"""

from __future__ import annotations

import argparse
import os
import json
import os
import random
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "artifacts" / "serve" / "spark_check.json"
API_KEY = "wf-spark-local"
MODEL = "spark"
EFFORT = "high"  # README: high or xhigh for agentic coding; xhigh (default) burns far more tokens
SAMPLING = {"temperature": 1.0, "top_p": 1.0}


def log(msg: str) -> None:
    print(f"[spark_client {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(res: dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))


def http(base: str, path: str, body: dict | None = None, timeout: int = 120) -> Any:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def prefix_metrics(base: str) -> dict[str, float]:
    text = http(base.removesuffix("/v1"), "/metrics")
    out: dict[str, float] = {}
    for ln in text.splitlines():
        m = re.match(r"vllm:(prefix_cache_(?:queries|hits)_total|external_prefix_cache_(?:queries|hits)_total)(?:\{.*\})?\s+([0-9.e+]+)", ln)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def wait_url(timeout_s: int = 45 * 60) -> str:
    import modal

    ctl = modal.Dict.from_name(os.environ.get("SPARK_CTL", "wayfinder-spark-ctl" + os.environ.get("SPARK_APP_SUFFIX", os.environ.get("SPARK_REPLICA", ""))), create_if_missing=True)
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        st = ctl.get("status")
        if st != last:
            log(f"status={st}")
            last = st
        if st == "ready" and ctl.get("url"):
            return ctl["url"]
        if st == "done":
            raise RuntimeError(f"server function finished without becoming ready; log tail:\n" + "\n".join(ctl.get("log_tail") or []))
        time.sleep(10)
    raise RuntimeError("timed out waiting for server")


def pin_dns(host: str, timeout_s: int = 300) -> None:
    """The local resolver SERVFAILs on *.w.modal.host; resolve through 8.8.8.8 and pin the answer."""
    import socket
    import subprocess

    t0 = time.time()
    ip = ""
    while time.time() - t0 < timeout_s and not ip:
        out = subprocess.run(["dig", "+short", "@8.8.8.8", host], capture_output=True, text=True).stdout.split()
        ip = next((x for x in out if x[0].isdigit()), "")
        if not ip:
            time.sleep(5)
    if not ip:
        raise RuntimeError(f"could not resolve {host} via 8.8.8.8")
    orig = socket.getaddrinfo

    def patched(h, *a, **k):  # type: ignore[no-untyped-def]
        return orig(ip if h == host else h, *a, **k)

    socket.getaddrinfo = patched  # type: ignore[assignment]
    log(f"pinned {host} -> {ip}")


def wait_reachable(root: str, timeout_s: int = 600) -> None:
    """New tunnel hostnames can take minutes to resolve; block until /health answers through the tunnel."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(root + "/health", headers={"Authorization": f"Bearer {API_KEY}"})
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    log(f"tunnel reachable after {time.time() - t0:.0f}s")
                    return
        except Exception as e:
            log(f"waiting for tunnel: {repr(e)[:80]}")
        time.sleep(10)
    raise RuntimeError("tunnel never became reachable")


def tool_check(client: OpenAI, tools: list[dict], system_prompt: str) -> dict[str, Any]:
    import jsonschema
    from glimmer_check import BRIEFS, FAKE_RESULTS  # same 20 briefs / fake tool results as the Glimmer check

    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}

    def one(i: int) -> dict[str, Any]:
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Brief: {BRIEFS[i]}"},
        ]
        row: dict[str, Any] = {"i": i, "ok": False}
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=MODEL, messages=msgs, tools=tools, tool_choice="auto", max_tokens=6000,
                extra_body={"reasoning_effort": EFFORT}, **SAMPLING,
            )  # fmt: skip
            row["secs"] = round(time.time() - t0, 2)
            ch = r.choices[0]
            m = ch.message
            row["finish_reason"] = ch.finish_reason
            row["completion_tokens"] = r.usage.completion_tokens
            row["n_tool_calls"] = len(m.tool_calls or [])
            content = m.content or ""
            row["has_reasoning"] = bool(getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None))
            row["leaked_markup"] = bool(re.search(r"<\|?[a-z_:]*(tool|call|channel|invoke|recipient)", content, re.I))
            if m.tool_calls:
                tc = m.tool_calls[0]
                name = tc.function.name
                row["tool"] = name
                try:
                    args = json.loads(tc.function.arguments)
                    if name not in schemas:
                        row["error"] = f"unknown tool {name!r}"
                    else:
                        jsonschema.validate(args, schemas[name])
                        row["ok"] = True
                except json.JSONDecodeError as e:
                    row["error"] = f"bad json: {e}"[:200]
                except jsonschema.ValidationError as e:
                    row["error"] = f"schema: {e.message}"[:200]
                if row["ok"]:
                    amsg: dict[str, Any] = {
                        "role": "assistant", "content": content or None,
                        "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}],
                    }  # fmt: skip
                    msgs2 = [*msgs, amsg, {"role": "tool", "tool_call_id": tc.id, "content": FAKE_RESULTS[name]}]
                    try:
                        r2 = client.chat.completions.create(
                            model=MODEL, messages=msgs2, tools=tools, tool_choice="auto", max_tokens=6000,
                            extra_body={"reasoning_effort": EFFORT}, **SAMPLING,
                        )  # fmt: skip
                        m2 = r2.choices[0].message
                        row["turn2_ok"] = True
                        row["turn2_tool"] = m2.tool_calls[0].function.name if m2.tool_calls else None
                    except Exception as e:
                        row["turn2_ok"] = False
                        row["turn2_error"] = repr(e)[:300]
            else:
                row["sample_content"] = content[:300]
        except Exception as e:
            row["error"] = repr(e)[:300]
        return row

    with ThreadPoolExecutor(10) as ex:
        rows = list(ex.map(one, range(len(BRIEFS))))
    ok = sum(r["ok"] for r in rows)
    t2 = [r for r in rows if "turn2_ok" in r]
    return {
        "n_requests": len(rows), "parse_success": ok, "parse_success_rate": round(ok / len(rows), 3),
        "any_tool_call": sum(1 for r in rows if r.get("n_tool_calls")),
        "leaked_markup": sum(1 for r in rows if r.get("leaked_markup")),
        "with_reasoning": sum(1 for r in rows if r.get("has_reasoning")),
        "turn2_ok": sum(1 for r in t2 if r["turn2_ok"]), "turn2_n": len(t2),
        "tools_used": sorted({r["tool"] for r in rows if r.get("tool")}),
        "median_completion_tokens": statistics.median(r.get("completion_tokens", 0) for r in rows),
        "reasoning_effort": EFFORT, "rows": rows,
    }  # fmt: skip


WORDS = [
    "alpha", "bridge", "cipher", "delta", "ember", "forest", "granite", "harbor", "island", "jungle", "kernel",
    "lantern", "meadow", "nebula", "orchid", "prairie", "quartz", "river", "summit", "timber", "umbra", "valley",
    "willow", "xenon", "yonder", "zephyr", "function", "buffer", "socket", "parser",
]  # fmt: skip


def words_text(n_words: int, seed: int) -> str:
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(n_words))


class Prompts:
    """Builds ~6k-token multi-turn-like prompts: a shared ~2k-token prefix, then unique turns."""

    def __init__(self, base: str) -> None:
        txt = words_text(1000, 1)
        n = http(base, "/tokenize", {"model": MODEL, "prompt": txt})["count"]
        self.tok_per_word = n / 1000
        log(f"tokens/word={self.tok_per_word:.2f}")
        self.prefix = "You are a code-search subagent. Shared repo notes follow.\n" + words_text(int(2000 / self.tok_per_word), 424242)

    def make(self, seed: int) -> list[dict[str, str]]:
        per = int(1300 / self.tok_per_word)
        return [
            {"role": "system", "content": self.prefix},
            {"role": "user", "content": f"Task {seed}: locate the code.\n" + words_text(per, seed * 10 + 1)},
            {"role": "assistant", "content": "Searching.\n" + words_text(per // 4, seed * 10 + 2)},
            {"role": "user", "content": "Tool result:\n" + words_text(per, seed * 10 + 3)},
            {"role": "assistant", "content": "Reading.\n" + words_text(per // 4, seed * 10 + 4)},
            {"role": "user", "content": "Tool result:\n" + words_text(per, seed * 10 + 5) + "\nContinue."},
        ]


def stream_once(client: OpenAI, msgs: list[dict], max_tokens: int, **extra: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    t_first = None
    usage = None
    stream = client.chat.completions.create(
        model=MODEL, messages=msgs, max_tokens=max_tokens, temperature=0.0, stream=True,
        stream_options={"include_usage": True}, extra_body={"ignore_eos": True, "reasoning_effort": "low", **extra},
    )  # fmt: skip
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices:
            d = chunk.choices[0].delta
            if (d.content or getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None)) and t_first is None:
                t_first = time.perf_counter()
    t_end = time.perf_counter()
    ct = usage.completion_tokens if usage else 0
    first = t_first or t_end
    return {
        "prompt_tokens": usage.prompt_tokens if usage else None, "completion_tokens": ct,
        "ttft_s": round(first - t0, 3),
        "decode_tok_s": round((ct - 1) / (t_end - first), 2) if ct > 1 and t_end > first else None,
        "total_s": round(t_end - t0, 3),
    }  # fmt: skip


def throughput(client: OpenAI, base: str, levels: tuple[int, ...] = (32, 64, 128)) -> dict[str, Any]:
    P = Prompts(base)
    out: dict[str, Any] = {}
    stream_once(client, P.make(999), 32)  # warmup (also seeds prefix cache)
    reps = [stream_once(client, P.make(5000 + i), 256) for i in range(3)]
    out["single_stream_6k"] = {"runs": reps, "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in reps if r["decode_tok_s"]),
                               "ttft_s_median": statistics.median(r["ttft_s"] for r in reps)}  # fmt: skip
    log(f"single stream: {out['single_stream_6k']['decode_tok_s_median']} tok/s")
    m0 = prefix_metrics(base)
    for conc in levels:
        n_req = conc * 2
        before = prefix_metrics(base)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(lambda s: stream_once(client, P.make(conc * 1000 + s), 300), range(n_req)))
        wall = time.perf_counter() - t0
        after = prefix_metrics(base)
        tot = sum(r["completion_tokens"] for r in res)
        q = after.get("prefix_cache_queries_total", 0) - before.get("prefix_cache_queries_total", 0)
        h = after.get("prefix_cache_hits_total", 0) - before.get("prefix_cache_hits_total", 0)
        out[f"concurrency_{conc}"] = {
            "n_requests": n_req, "wall_s": round(wall, 1), "completion_tokens": tot,
            "aggregate_tok_s": round(tot / wall, 1),
            "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in res),
            "ttft_s_median": statistics.median(r["ttft_s"] for r in res),
            "per_stream_decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in res if r["decode_tok_s"]),
            "prefix_cache_hit_rate": round(h / q, 3) if q else None, "prefix_queries": q, "prefix_hits": h,
        }  # fmt: skip
        log(f"conc {conc}: {out[f'concurrency_{conc}']}")
    m1 = prefix_metrics(base)
    q = m1.get("prefix_cache_queries_total", 0) - m0.get("prefix_cache_queries_total", 0)
    h = m1.get("prefix_cache_hits_total", 0) - m0.get("prefix_cache_hits_total", 0)
    out["prefix_cache_hit_rate_overall"] = round(h / q, 3) if q else None
    out["metrics_raw_end"] = m1
    return out


def logprob_check(client: OpenAI, base: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # 1. generated-token top-20 logprobs via chat
    try:
        r = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Name three sorting algorithms."}], max_tokens=40,
            temperature=1.0, logprobs=True, top_logprobs=20, extra_body={"reasoning_effort": "low"},
        )  # fmt: skip
        lp = r.choices[0].logprobs.content if r.choices[0].logprobs else None
        out["chat_top20"] = {
            "ok": bool(lp), "n_tokens": len(lp or []),
            "top_logprobs_len_min": min((len(t.top_logprobs) for t in lp), default=0) if lp else 0,
            "top_logprobs_len_max": max((len(t.top_logprobs) for t in lp), default=0) if lp else 0,
            "sample": [{"token": t.token, "logprob": t.logprob, "top3": [(x.token, x.logprob) for x in t.top_logprobs[:3]]} for t in (lp or [])[:3]],
        }  # fmt: skip
    except Exception as e:
        out["chat_top20"] = {"ok": False, "error": repr(e)[:400]}
    # 2. top-20 > max-logprobs must be rejected (sanity that the cap is 20)
    try:
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=2, logprobs=True, top_logprobs=21)
        out["top21_accepted"] = True
    except Exception as e:
        out["top21_accepted"] = False
        out["top21_error"] = repr(e)[:200]
    # 3. prompt_logprobs for a supplied completion, /v1/completions with token ids
    try:
        prompt = "The capital of France is Paris, and the capital of Germany is Berlin."
        ids = http(base, "/tokenize", {"model": MODEL, "prompt": prompt, "add_special_tokens": False})["tokens"]
        r = http(base, "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 5}, timeout=300)
        pl = r["choices"][0].get("prompt_logprobs")
        out["completions_prompt_logprobs"] = {
            "ok": bool(pl), "n_prompt_tokens": len(ids), "n_entries": len(pl or []),
            "first_is_null": (pl or [None])[0] is None,
            "sample": (pl or [])[1:3],
        }  # fmt: skip
    except Exception as e:
        out["completions_prompt_logprobs"] = {"ok": False, "error": repr(e)[:400]}
    # 4. teacher scoring a student completion inside a chat template: chat endpoint with prompt_logprobs
    try:
        msgs = [{"role": "user", "content": "Name a sorting algorithm."}, {"role": "assistant", "content": "Quicksort is a sorting algorithm."}]
        r = http(base, "/v1/chat/completions", {
            "model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
            "add_generation_prompt": False, "continue_final_message": True, "reasoning_effort": "low",
        }, timeout=300)  # fmt: skip
        pl = r.get("prompt_logprobs") or r["choices"][0].get("prompt_logprobs")
        out["chat_prompt_logprobs"] = {"ok": bool(pl), "n_entries": len(pl or []), "usage": r.get("usage"), "sample": (pl or [])[-3:]}
    except Exception as e:
        out["chat_prompt_logprobs"] = {"ok": False, "error": repr(e)[:400]}
    return out


def in_container() -> None:
    """Run the throughput test on the server box (no tunnel). Prints one JSON line prefixed TPJSON."""
    base = "http://127.0.0.1:8000"
    client = OpenAI(base_url=base + "/v1", api_key=API_KEY, timeout=900)
    res = throughput(client, base)
    print("TPJSON " + json.dumps(res), flush=True)


def tp_only(levels: tuple[int, ...]) -> None:
    """Re-run just the throughput levels through the tunnel; merges into spark_check.json."""
    import modal

    ctl = modal.Dict.from_name(os.environ.get("SPARK_CTL", "wayfinder-spark-ctl"), create_if_missing=True)
    root = ctl["url"].rstrip("/")
    pin_dns(root.removeprefix("https://"))
    wait_reachable(root)
    client = OpenAI(base_url=root + "/v1", api_key=API_KEY, timeout=900, max_retries=5)
    res = json.loads(OUT.read_text())
    tp = throughput(client, root, levels)
    res.setdefault("throughput_extra", {}).update(tp)
    save(res)
    log("throughput merged")


def main() -> None:
    import modal

    ap = argparse.ArgumentParser()
    ap.add_argument("--in-container", action="store_true")
    ap.add_argument("--tp-only", type=int, nargs="+")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--skip-throughput", action="store_true")
    args = ap.parse_args()
    if args.tp_only:
        return tp_only(tuple(args.tp_only))
    if args.in_container:
        return in_container()
    ctl = modal.Dict.from_name(os.environ.get("SPARK_CTL", "wayfinder-spark-ctl" + os.environ.get("SPARK_APP_SUFFIX", os.environ.get("SPARK_REPLICA", ""))), create_if_missing=True)
    if args.stop:
        ctl["stop"] = True
        log("stop flag set")
        return
    t0 = time.time()
    url = wait_url()
    base = url.rstrip("/") + "/v1"
    root = url.rstrip("/")
    pin_dns(root.removeprefix("https://"))
    wait_reachable(root)
    res: dict[str, Any] = {"url_wait_s": round(time.time() - t0, 1), "cold_start_s": ctl.get("cold_start_s"), "gpu": ctl.get("gpu")}
    save(res)
    client = OpenAI(base_url=base, api_key=API_KEY, timeout=900, max_retries=5)
    tools = json.loads((HERE / "scout_tool_specs.json").read_text())
    system_prompt = (HERE / "scout_product_prompt.md").read_text()
    steps = [("logprobs", lambda: logprob_check(client, root)), ("tool_calls", lambda: tool_check(client, tools, system_prompt))]
    if not args.skip_throughput:
        steps.append(("throughput", lambda: throughput(client, root)))
    for name, fn in steps:
        try:
            log(f"running {name}")
            res[name] = fn()
        except Exception as e:
            res[name] = {"error": repr(e)[:800]}
            log(f"{name} failed: {e!r}")
        save(res)
    res["log_tail"] = ctl.get("log_tail")
    save(res)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
