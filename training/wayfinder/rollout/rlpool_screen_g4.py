"""RL-pool screening on a plain GPU box (GKE G4) instead of Modal: same protocol as rlpool_screen_app.py.

The vLLM server runs in a sibling container (127.0.0.1:8000, same BASE_FLAGS as the Modal screen, started by the pod
spec in deploy/g4/screen.yaml); this worker is the body of `serve_and_rollout` minus the Modal pieces: adapter check,
rollouts via the same wayfinder.rollout.run/chat/score, rows compacted with the same `_compact`, task-major queue,
resumable jsonl (finished (task, seed) pairs without error are skipped). Helpers below are copied verbatim.

    python -m wayfinder.rollout.rlpool_screen_g4 --tasks tasks.json --out /data/out/g4.jsonl --seeds 0,...,7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

WEIGHTS_DIR = os.environ.get("WF_EVAL_WEIGHTS", "/data/sft/v3_merged")  # same string as the Modal rows' `weights`
VARIANT = "v3"
LORA_NAME = "wayfinder-v3"
RETAIN_REASONING = True
MAX_MODEL_LEN = int(os.environ.get("WF_MAX_LEN", "65536"))
LAUNCH_DEADLINE_S = 40 * 3600


def _log(msg: str) -> None:
    print(f"[g4_screen {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def _compact(row: dict) -> dict:
    """Keep stats, scores and reported paths; drop the transcript (screening needs recall, turns and cost, not the text)."""
    keep = ("task_id", "repo", "base_commit", "model", "seed", "arm", "error", "infra_error", "semantic", "semantic_requested", "semantic_reason",
            "semantic_index", "semantic_query_tokens", "max_turns", "max_seconds", "turns", "model_calls", "wall_s", "wall_budget_s", "model_latency_s",
            "cost_usd", "usage", "recall", "recall_reported", "recall_opened", "recall_seen_k5", "recall_seen_all", "all_gold_found",
            "all_gold_reported", "n_reported", "n_opened", "retain_reasoning", "weights", "temperature", "top_p", "max_model_len", "clone_s",
            "n429", "n_retries", "rg_internal_remapped")
    out = {k: row[k] for k in keep if k in row}
    er = row.get("explore_result") or {}
    out["reported_paths"] = [h.get("path") for h in (er.get("results") or [])]
    out["budget_exhausted"] = bool(er.get("budget_exhausted"))
    out["gaps"] = (er.get("gaps") or [])[:3]
    out["tool_calls"] = (er.get("usage") or {}).get("tool_calls")
    out["n_gold"] = len(row.get("gold_files") or [])
    return out


async def main_async(a: argparse.Namespace) -> None:
    import re
    import urllib.request

    from wayfinder.rollout.chat import CompatChat, Prices, Sampling
    from wayfinder.rollout.run import TEACHER_NUDGE, checkout, run_rollout
    from wayfinder.rollout.score import score_row

    from pydantic import Field
    from pydantic.fields import FieldInfo
    from scout.core.contracts.types import Budget, ExploreRequest

    Budget.model_fields["max_seconds"] = FieldInfo.from_annotated_attribute(int, Field(default=120, ge=5, le=100000))  # as the Modal screen
    Budget.model_rebuild(force=True)
    ExploreRequest.model_rebuild(force=True)
    cfg = {"concurrency": a.concurrency, "temperature": 0.85, "top_p": 0.95, "max_turns": 40, "max_seconds": 1200,
           "nudge": True, "prices": [1.25, 4.25, 0.15]}
    task_list = json.loads(Path(a.tasks).read_text())
    seeds = [int(x) for x in a.seeds.split(",")]
    fp = Path(a.out)
    fp.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if fp.exists():
        for ln in fp.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if not r.get("error"):
                    done.add((r["task_id"], r["seed"]))
    todo = [(t, s) for t in task_list for s in seeds if (t["task_id"], s) not in done]  # task-major
    _log(f"{len(todo)} rollouts to run ({len(task_list)} tasks), {len(done)} done")
    if not todo:
        return
    meta: dict = {"cfg": cfg, "gpu": "RTX Pro 6000 (GKE G4)", "variant": VARIANT, "retain_reasoning": RETAIN_REASONING,
                  "weights": WEIGHTS_DIR, "max_model_len": MAX_MODEL_LEN, "nudge": TEACHER_NUDGE}
    for _ in range(600):  # vllm health (the server is started by the sibling container)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    meta["adapter_check"] = await asyncio.to_thread(_verify_adapter, "")
    _log(f"adapter_check: {json.dumps(meta['adapter_check'])[:1500]}")
    ck = meta["adapter_check"]
    fp.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    if LORA_NAME not in ck.get("models", []) or not ck.get("differs") or not ck.get("v1_tool_call_parsed"):
        raise SystemExit("adapter check failed; no rollouts run")

    t_start = time.time()
    sem = asyncio.Semaphore(cfg["concurrency"])
    clones: dict[str, asyncio.Future] = {}
    queue: asyncio.Queue = asyncio.Queue()

    async def get_root(t: dict) -> tuple[str, float]:
        key = f"{t['repo']}@{t['base_commit']}"
        if key not in clones:
            clones[key] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], a.repos))
        return await clones[key]

    async def one(t: dict, seed: int) -> None:
        arm = "sem"
        async with sem:
            try:
                root, clone_s = await get_root(t)
            except Exception as e:  # noqa: BLE001
                await queue.put({"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "arm": arm, "error": f"checkout: {e}"[:300], "infra_error": True})
                return
            chat = CompatChat("http://127.0.0.1:8000/v1", "x", LORA_NAME, sampling=Sampling(cfg["temperature"], cfg["top_p"]),
                              prices=Prices(*cfg["prices"]), seed=seed, vllm_extensions=False, timeout=600.0, max_retries=4)
            chat.retain_reasoning = RETAIN_REASONING
            try:
                row = await asyncio.wait_for(
                    run_rollout(t, chat, root, seed=seed, max_turns=cfg["max_turns"], max_seconds=cfg["max_seconds"],
                                nudge=TEACHER_NUDGE if cfg["nudge"] else None, semantic="auto"), timeout=cfg["max_seconds"] + 600)
                row.update(score_row(row))
                row["arm"] = arm
                row["clone_s"] = round(clone_s, 1)
                row["temperature"], row["top_p"] = cfg["temperature"], cfg["top_p"]
                row["max_model_len"] = MAX_MODEL_LEN
                row["retain_reasoning"] = RETAIN_REASONING
                row["weights"] = WEIGHTS_DIR
            except Exception as e:  # noqa: BLE001
                row = {"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "arm": arm, "error": f"harness: {type(e).__name__}: {e}"[:300], "infra_error": True}
            await queue.put(_compact(row))

    async def _telemetry() -> None:
        while True:
            await asyncio.sleep(60)
            try:
                txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
                g = lambda n: re.search(rf"^{re.escape(n)}(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)", txt, re.M)  # noqa: E731
                vals = {n.split(":")[1]: (g(n).group(1) if g(n) else None) for n in (
                    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                    "vllm:generation_tokens_total", "vllm:num_preemptions_total")}
                _log(f"telemetry {vals} done_rows={n_done} elapsed_min={(time.time()-t_start)/60:.1f}")
            except Exception as e:  # noqa: BLE001
                _log(f"telemetry error {e!r}")

    n_done = 0
    tele = asyncio.ensure_future(_telemetry())
    tasks = [asyncio.ensure_future(one(t, s)) for t, s in todo]
    try:
        with fp.open("a") as f:
            for _ in range(len(tasks)):
                row = await queue.get()
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
                if n_done % 10 == 0 or row.get("error"):
                    _log(f"[{n_done}/{len(todo)}] {row['task_id'][-30:]} s{row['seed']} turns={row.get('turns')} wall={row.get('wall_s')} "
                         f"rr={row.get('recall_reported')} err={row.get('error')} sem={row.get('semantic')}")
    finally:
        tele.cancel()
        for tk in tasks:
            tk.cancel()
        meta["vllm_metrics"] = _metrics()
        meta["elapsed_s"] = round(time.time() - t_start, 1)
        fp.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--repos", default="/data/repos")
    asyncio.run(main_async(ap.parse_args()))
