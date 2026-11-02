"""Run rollouts in Modal CPU containers (no DNS issue, each container has its own repo checkout). Ephemeral app.

Stage B (rows stream back to the laptop and are appended to --out):

    modal run wayfinder/rollout/modal_app.py --tasks artifacts/teacher_test/stageB_tasks.json \
        --out artifacts/teacher_test/stageB_selfhost.jsonl --seeds 0,1 --endpoint-ctl wayfinder-spark-ctl

Bulk teacher collection (orchestrator runs inside Modal, so the laptop can disconnect; rows land incrementally in the
Volume `wayfinder-data` under /teacher/<name>/<task_hash>.jsonl and finished (task, seed) pairs are skipped on restart):

    modal run --detach wayfinder/rollout/modal_app.py::bulk --name spark_v1 --seeds 0,1,2,3

Endpoints are self-hosted Spark Open servers: each container reads the Modal Dict(s) named in `endpoint_ctl`
(`wayfinder-spark-ctl`, `wayfinder-spark-ctl-r1`, ...), takes a replica whose status is `ready` and whose /health
answers (rotated by task hash so replicas share load), and waits up to 45 min for one. Model: `spark`, key
`wf-spark-local` (the server's own local key, not a secret), sampling T=0.85 top_p=0.95, reasoning_effort=high, the
convergence nudge (run.TEACHER_NUDGE), max_turns 40, max_seconds 900 (self-hosted latency under load; recorded per row).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the container the file sits in /root
SCOUT_SRC = REPO.parent / "voyage-scout" / "src"  # local paths only matter at image-build time

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ripgrep", "ca-certificates")
    .pip_install("httpx==0.28.1", "pydantic>=2.9", "pathspec==1.1.1", "watchfiles==1.3.0",
                 # semantic_search (wayfinder/rollout/semantic.py): scout's real gateway/store/embedder path
                 "cryptography==50.0.2", "keyring==25.7.0", "turbopuffer==3.0.0", "voyageai==0.5.0")
    .env({"SCOUT_SRC": "/scout_src", "PYTHONPATH": "/root", "GIT_TERMINAL_PROMPT": "0"})
    .add_local_dir(str(SCOUT_SRC), "/scout_src", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)
app = modal.App("wayfinder-rollouts", image=image)
INDEX_SECRET = modal.Secret.from_name("wayfinder-index-keys")  # TURBOPUFFER_API_KEY, VOYAGE_API_KEY, WF_INDEX_MASTER_KEY_HEX
data_vol = modal.Volume.from_name("wayfinder-data", create_if_missing=True)

SPARK_KEY = "wf-spark-local"
DEFAULT_CTL = ["wayfinder-spark-ctl", "wayfinder-spark-ctl-r1"]
# containers = concurrent tasks; each runs all its seeds concurrently, so server concurrency = containers x seeds
MAX_CONTAINERS = int(os.environ.get("WF_MAX_CONTAINERS", "32"))
DEFAULT_CFG = {
    "model": "spark", "temperature": 0.85, "top_p": 0.95, "max_turns": 40, "max_seconds": 900,
    "reasoning_effort": "high", "nudge": True, "semantic": "auto", "prices": [1.25, 4.25, 0.15],  # prices: Baseten list price, comparable cost only
}


def _pick_endpoint(ctl_names: list[str], key: str, wait_s: int = 45 * 60) -> tuple[str, str]:
    """(base_url_with_v1, ctl_name) of a ready, healthy replica; rotated by `key` so replicas share load."""
    import time
    import urllib.request
    import zlib

    t0 = time.time()
    off = zlib.crc32(key.encode()) % max(len(ctl_names), 1)
    order = ctl_names[off:] + ctl_names[:off]
    while True:
        for name in order:
            try:
                d = modal.Dict.from_name(name)
                if d.get("status") != "ready" or not d.get("url"):
                    continue
                root = d["url"].rstrip("/")
                req = urllib.request.Request(root + "/health", headers={"Authorization": f"Bearer {SPARK_KEY}"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    if r.status == 200:
                        return root + "/v1", name
            except Exception:  # noqa: BLE001
                continue
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"no ready endpoint among {ctl_names} after {wait_s}s")
        time.sleep(15)


@app.function(timeout=3 * 3600, max_containers=MAX_CONTAINERS, cpu=2, memory=4096, volumes={"/data": data_vol},
              secrets=[INDEX_SECRET])
async def rollout_task(task: dict, seeds: list[int], cfg: dict, out_dir: str | None = None) -> list[dict]:
    """One task, all `seeds` concurrently on one shared read-only checkout. With `out_dir`, each finished row without
    an error is appended to /data/<out_dir>/<task_hash>.jsonl and committed (resumable bulk mode) and only a small
    summary is returned; without it the full rows are returned."""
    import asyncio
    import hashlib

    from wayfinder.rollout.chat import CompatChat, Prices, Sampling
    from wayfinder.rollout.run import TEACHER_NUDGE, checkout, run_rollout
    from wayfinder.rollout.score import score_row

    cfg = {**DEFAULT_CFG, **cfg}
    try:
        base_url, ctl_name = await asyncio.to_thread(_pick_endpoint, cfg.get("endpoint_ctl") or DEFAULT_CTL, task["task_id"])
        root, clone_s = await asyncio.to_thread(checkout, task["repo"], task["base_commit"], "/tmp/repos")
    except Exception as e:  # noqa: BLE001
        return [{"task_id": task["task_id"], "repo": task["repo"], "seed": s, "model": cfg["model"],
                 "error": f"setup: {e}"[:300], "infra_error": True} for s in seeds]
    thash = hashlib.sha1(task["task_id"].encode()).hexdigest()[:16]
    lock = asyncio.Lock()

    async def one(seed: int) -> dict:
        chat = CompatChat(
            base_url, SPARK_KEY, cfg["model"],
            sampling=Sampling(cfg["temperature"], cfg["top_p"]),
            prices=Prices(*cfg["prices"]), seed=seed, vllm_extensions=False,
            extra_body={"reasoning_effort": cfg["reasoning_effort"]} if cfg.get("reasoning_effort") else None,
            timeout=600.0, max_retries=6,
            retain_reasoning=False,  # teacher collection (Spark via Baseten): keep the data distribution of the earlier passes
        )
        try:
            row = await run_rollout(task, chat, root, seed=seed, max_turns=cfg["max_turns"],
                                    max_seconds=cfg["max_seconds"], nudge=TEACHER_NUDGE if cfg["nudge"] else None,
                                    semantic=cfg.get("semantic", "auto"))
            row.update(score_row(row))
        except Exception as e:  # noqa: BLE001
            return {"task_id": task["task_id"], "repo": task["repo"], "seed": seed,
                    "error": f"harness: {type(e).__name__}: {e}"[:300], "infra_error": True}
        row["clone_s"] = round(clone_s, 1)
        row["endpoint"] = ctl_name
        row["reasoning_effort"] = cfg.get("reasoning_effort")
        row["temperature"], row["top_p"] = cfg["temperature"], cfg["top_p"]
        if out_dir and not row.get("error"):
            async with lock:  # one file per task; append + commit as each seed finishes
                p = Path("/data") / out_dir
                p.mkdir(parents=True, exist_ok=True)
                with (p / f"{thash}.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                await asyncio.to_thread(data_vol.commit)
        return row

    rows = list(await asyncio.gather(*[one(s) for s in seeds]))
    if out_dir:
        keep = ("task_id", "seed", "turns", "recall", "all_gold_found", "cost_usd", "wall_s", "error", "infra_error", "usage")
        return [{k: r.get(k) for k in keep} for r in rows]
    return rows


def _done_pairs(out_dir: str) -> set[tuple[str, int]]:
    data_vol.reload()
    done: set[tuple[str, int]] = set()
    d = Path("/data") / out_dir
    if d.exists():
        for fp in d.glob("*.jsonl"):
            for ln in fp.read_text().splitlines():
                if ln.strip():
                    try:
                        r = json.loads(ln)
                        done.add((r["task_id"], int(r["seed"])))
                    except (ValueError, KeyError):
                        pass
    return done


@app.function(timeout=24 * 3600, cpu=1, memory=2048, volumes={"/data": data_vol})
def bulk_orchestrator(tasks: list[dict], seeds: list[int], cfg: dict, out_dir: str, passes: int = 8) -> dict:
    """Map rollout_task over every task that still has unfinished seeds; repeat (failed rollouts are not saved, so they
    are retried) up to `passes` times. Progress is appended to /data/<out_dir>/_progress.jsonl."""
    import time

    t0 = time.time()
    n_new = 0
    cost = 0.0
    prog = Path("/data") / out_dir / "_progress.jsonl"
    prog.parent.mkdir(parents=True, exist_ok=True)
    for ps in range(passes):
        done = _done_pairs(out_dir)
        todo = [(t, [s for s in seeds if (t["task_id"], s) not in done], cfg, out_dir) for t in tasks]
        todo = [x for x in todo if x[1]]
        print(f"pass {ps}: {len(todo)} tasks with unfinished seeds, {len(done)} (task,seed) done", flush=True)
        if not todo:
            break
        for rows in rollout_task.starmap(todo, order_outputs=False, return_exceptions=True):
            if isinstance(rows, BaseException):
                print("task failed:", repr(rows)[:200], flush=True)
                continue
            for r in rows:
                if not r.get("error"):
                    n_new += 1
                    cost += r.get("cost_usd") or 0
            with prog.open("a") as f:
                f.write(json.dumps({"t": round(time.time() - t0), "pass": ps, "rows_ok": n_new, "cost_usd_list": round(cost, 2),
                                    "errors": sum(1 for r in rows if r.get("error"))}) + "\n")
            data_vol.commit()
    done = _done_pairs(out_dir)
    return {"new_rows": n_new, "done_pairs": len(done), "wall_s": round(time.time() - t0), "cost_usd_list": round(cost, 2)}


def _load_done_local(outp: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if outp.exists():
        for ln in outp.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if not r.get("infra_error") and not r.get("error"):
                    done.add((r["task_id"], r["seed"]))
    return done


@app.local_entrypoint()
def main(tasks: str, out: str, seeds: str = "0", endpoint_ctl: str = ",".join(DEFAULT_CTL[:1]),
         max_turns: int = 40, max_seconds: int = 900, nudge: bool = True, semantic: str = "auto"):
    """Stage B style: rows stream back to the laptop file `out` (resumable)."""
    task_list = json.loads(Path(tasks).read_text())
    seed_list = [int(s) for s in seeds.split(",")]
    cfg = {"max_turns": max_turns, "max_seconds": max_seconds, "nudge": nudge, "semantic": semantic, "endpoint_ctl": endpoint_ctl.split(",")}
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_local(outp)
    todo = [(t, [s for s in seed_list if (t["task_id"], s) not in done], cfg) for t in task_list]
    todo = [x for x in todo if x[1]]
    print(f"{len(todo)} tasks to run ({len(done)} rows already done)", flush=True)
    n = errs = 0
    cost = 0.0
    with outp.open("a") as f:
        for rows in rollout_task.starmap(todo, order_outputs=False, return_exceptions=True):
            if isinstance(rows, BaseException):
                print("task failed:", repr(rows)[:200], flush=True)
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
                n += 1
                errs += bool(r.get("error"))
                cost += r.get("cost_usd") or 0
            f.flush()
            print(f"rows={n} errors={errs} cost_list=${cost:.3f}", flush=True)


@app.local_entrypoint()
def bulk(name: str = "spark_v1", seeds: str = "0,1,2,3", endpoint_ctl: str = ",".join(DEFAULT_CTL),
         split: str = "train", limit: int = 0, semantic: str = "auto"):
    """Spawn the orchestrator in Modal over accepted `split` tasks of data/task_filter.json; use `modal run --detach`."""
    flt = json.loads((REPO / "data" / "task_filter.json").read_text())["tasks"]
    spl = json.loads((REPO / "data" / "splits.json").read_text())["tasks"]
    tasks = []
    for tid, f in sorted(flt.items()):
        if f.get("accepted") and f["split"] == split:
            t = spl[tid]
            tasks.append({"task_id": tid, "repo": t["repo_full"], "base_commit": t["base_commit"],
                          "question": t["task_text"], "gold_files": list(t["gold_files"]), "category": t.get("category")})
    if limit:
        tasks = tasks[:limit]
    print(f"{len(tasks)} accepted {split} tasks x seeds {seeds}", flush=True)
    cfg = {"endpoint_ctl": endpoint_ctl.split(","), "semantic": semantic}
    res = bulk_orchestrator.remote(tasks, [int(s) for s in seeds.split(",")], cfg, f"teacher/{name}")
    print(json.dumps(res), flush=True)
