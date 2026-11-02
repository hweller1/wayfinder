"""Score student rollout chains with the Spark Open teacher server. CPU Modal job (the laptop cannot resolve *.modal.host).

    modal run wayfinder/opd/score_app.py --rollouts opd/smoke --out opd/smoke --ctl wayfinder-spark-ctl-e,wayfinder-spark-ctl-d

Reads wayfinder-data:/<rollouts>/rollouts/*.json.gz (opd/rollout_app.py), waits for the teacher (`ready` in the ctl Dict, /health
ok), scores every chain with `prompt_logprobs=20` (wayfinder/opd/score.py) and writes wayfinder-data:/<out>/scored/<id>.npz plus
manifest.jsonl and scored_summary.json (throughput). Resumable: chains already in the manifest are skipped. Use
scripts/opd/score_with_teacher.sh, which also starts and stops the teacher.
The Spark tokenizer files are read from wayfinder-data:/opd/spark_tok (copied once from the wayfinder-spark volume).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import sys

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # local entrypoint imports wayfinder.opd.*
app = modal.App("wayfinder-opd-score")  # prefix wayfinder-opd: ops/opd_watch.py
data_vol = modal.Volume.from_name("wayfinder-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==5.21.0", "tokenizers", "jinja2", "numpy", "httpx==0.28.1", "orjson")
    .env({"PYTHONPATH": "/root", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)


def _ready_endpoints(ctl_names: str) -> list[tuple[str, str]]:
    """Every ready, healthy replica among the comma-separated ctl Dict names -> [(root url, ctl name)]."""
    import urllib.request

    out = []
    for n in ctl_names.split(","):
        try:
            d = modal.Dict.from_name(n, create_if_missing=False)
            if d.get("status") != "ready" or not d.get("url"):
                continue
            root = d["url"].rstrip("/")
            req = urllib.request.Request(root + "/health", headers={"Authorization": "Bearer wf-spark-local"})
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    out.append((root, n))
        except Exception:  # noqa: BLE001
            pass
    return out


def _wait_endpoint(ctl_names: str, wait_s: int) -> tuple[str, str]:
    """First ready, healthy replica among the comma-separated ctl Dict names -> (root url, ctl name)."""
    import urllib.request

    names = ctl_names.split(",")
    dicts = {n: modal.Dict.from_name(n, create_if_missing=True) for n in names}  # a replica that is still queued has no Dict yet
    t0 = time.time()
    last: dict[str, object] = {}
    while time.time() - t0 < wait_s:
        for n, d in dicts.items():
            st = d.get("status")
            if st != last.get(n):
                print(f"[score] {n} status={st} after {time.time() - t0:.0f}s", flush=True)
                last[n] = st
            if st == "ready" and d.get("url"):
                root = d["url"].rstrip("/")
                try:
                    req = urllib.request.Request(root + "/health", headers={"Authorization": "Bearer wf-spark-local"})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        if r.status == 200:
                            return root, n
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(10)
    raise RuntimeError(f"no ready teacher among {names} after {wait_s}s")


@app.function(image=image, volumes={"/data": data_vol}, timeout=4 * 3600, cpu=8, memory=32768)
async def score(rollouts: str, out: str, ctl_name: str, concurrency: int, wait_s: int, max_chains: int, tokens_budget: int) -> dict:
    import asyncio

    import httpx
    from transformers import AutoTokenizer

    from wayfinder.opd.rollout import read_rollout
    from wayfinder.opd.score import HeadCache, post_score, store_chain, teacher_ids

    t_start = time.time()
    await asyncio.to_thread(data_vol.reload)
    spark_tok = AutoTokenizer.from_pretrained("/data/opd/spark_tok")
    heads = HeadCache(spark_tok)
    out_dir = Path("/data") / out / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.jsonl"  # next to the npz files: train_opd.py --scored <dir>
    done = {json.loads(ln)["id"] for ln in man_path.read_text().splitlines() if ln.strip()} if man_path.exists() else set()
    work = []
    for p in sorted((Path("/data") / rollouts / "rollouts").glob("*.json.gz")):
        row = read_rollout(p)
        if row["opd"]["diag"].get("spans_v") != 2:  # chains built before spans covered sampled completions only: re-rendered turns would be trained twice
            print(f"[score] skip {p.name}: chains from the old span rule", flush=True)
            continue
        for ci, ch in enumerate(row["opd"]["chains"]):
            cid = f"{row['rollout_id']}-{ci}"
            if cid in done:
                continue
            head = heads.ids(row["opd"]["system"], row["opd"]["tools"])
            tids, shift = teacher_ids(ch["input_ids"], head)
            meta = {"task_id": row["task_id"], "seed": row["seed"], "repo": row["repo"], "domain": row.get("domain"), "rollout_id": row["rollout_id"],
                    "chain": ci, "teacher_tokens": len(tids), "recall": row.get("recall"), "turns": row.get("turns")}
            work.append((cid, ch, tids, shift, meta))
    if max_chains:
        work = work[:max_chains]
    if tokens_budget:
        tot, keep = 0, []
        for w in work:
            if tot + len(w[2]) > tokens_budget:
                break
            tot += len(w[2])
            keep.append(w)
        work = keep
    print(f"[score] {len(work)} chains to score, {sum(len(w[2]) for w in work) / 1e6:.2f}M teacher tokens, {len(done)} already done", flush=True)
    root, used_ctl = await asyncio.to_thread(_wait_endpoint, ctl_name, wait_s)
    eps = await asyncio.to_thread(_ready_endpoints, ctl_name)  # all ready replicas share the load (round-robin per chain; a failed request retries on the next one)
    if not eps:
        eps = [(root, used_ctl)]
    print(f"[score] endpoints: {[n for _, n in eps]}", flush=True)
    used_ctl = ",".join(n for _, n in eps)
    t_ready = time.time()
    print(f"[score] teacher ready after {t_ready - t_start:.0f}s; scoring at concurrency {concurrency}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    results: list[dict] = []
    errors: list[dict] = []
    t_score0 = time.time()

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency + 4)) as client:

        rr = [0]

        async def one(w):  # type: ignore[no-untyped-def]
            cid, ch, tids, shift, meta = w
            async with sem:
                try:
                    rr[0] += 1
                    pl, info = None, None
                    for k in range(len(eps)):
                        try:
                            pl, info = await post_score(client, eps[(rr[0] + k) % len(eps)][0], tids, retries=3 if len(eps) == 1 else 1)
                            break
                        except Exception:  # noqa: BLE001
                            if k == len(eps) - 1:
                                raise
                    rec = await asyncio.to_thread(store_chain, out_dir, cid, ch, shift, pl, {**meta, **info})
                except Exception as e:  # noqa: BLE001
                    errors.append({"id": cid, "error": repr(e)[:300]})
                    print(f"[score] ERROR {cid}: {e!r}"[:300], flush=True)
                    return
            async with lock:
                results.append(rec)
                with man_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if len(results) % 10 == 0:
                    await asyncio.to_thread(data_vol.commit)
                el = time.time() - t_score0
                print(f"[score] {len(results)}/{len(work)} tokens={sum(r['teacher_tokens'] for r in results)} {sum(r['teacher_tokens'] for r in results) / el:.0f} tok/s "
                      f"last={rec['teacher_tokens']}tok req={rec['request_s']}s parse={rec['parse_s']}s {rec['resp_mb']}MB", flush=True)

        await asyncio.gather(*[one(w) for w in work])
    await asyncio.to_thread(data_vol.commit)
    el = time.time() - t_score0
    tok = sum(r["teacher_tokens"] for r in results)
    ast = sum(r["n_positions"] for r in results)
    return {"teacher_ctl": used_ctl, "n_chains": len(results), "n_errors": len(errors), "errors": errors[:20], "teacher_tokens": tok, "assistant_positions": ast,
            "scoring_wall_s": round(el, 1), "scored_tokens_per_s": round(tok / max(el, 1e-9), 1), "assistant_positions_per_s": round(ast / max(el, 1e-9), 2),
            "teacher_ready_after_s": round(t_ready - t_start, 1), "concurrency": concurrency, "request_s": [r["request_s"] for r in results],
            "teacher_tokens_each": [r["teacher_tokens"] for r in results], "resp_mb_each": [r["resp_mb"] for r in results], "parse_s_each": [r["parse_s"] for r in results],
            "n_missing": sum(r["n_missing"] for r in results)}


@app.local_entrypoint()
def main(rollouts: str = "opd/smoke", out: str = "", ctl: str = "wayfinder-spark-ctl-opd", concurrency: int = 8, wait_s: int = 5400,
         max_chains: int = 0, tokens_budget: int = 0, summary: str = ""):
    out = out or rollouts
    res = score.remote(rollouts, out, ctl, concurrency, wait_s, max_chains, tokens_budget)
    od = REPO / "artifacts" / "opd"
    od.mkdir(parents=True, exist_ok=True)
    (od / (summary or f"scored_{Path(out).name}.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_each") and k != "request_s"}), flush=True)
