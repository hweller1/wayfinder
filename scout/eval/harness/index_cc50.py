"""Index the cc50 tasks' repo@commits through the product gateway (`scout sync --wait` path) and log per-commit cost.

    SCOUT_GATEWAY_URL=... uv run python -m eval.harness.index_cc50 --out artifacts/eval/cc50

Per commit: Voyage tokens embedded (summed from the gateway's EmbedRes), voyage-code-4 USD, rows written,
file count, wall seconds. turbopuffer write/storage is not metered by the API; it is estimated from rows written.
"""
import argparse, asyncio, json, os, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.harness import repos
from eval.harness.run import TASKS, keys, REPO

VOYAGE_USD_PER_M = 0.18  # voyage-code-4
TPUF_WRITE_USD_PER_GB = 2.0   # list price, logical bytes written
TPUF_STORE_USD_PER_GB_MONTH = 0.33
BYTES_PER_ROW = 1024 * 4 + 600  # 1024-dim f32 vector + ciphertext/attrs estimate


async def sync_one(root: Path, key: str, gw: str) -> dict:
    from scout.core.sync.engine import SyncEngine
    from scout.surfaces.cli.commands import build_app
    import argparse as ap
    acc = {"tokens": 0, "written": 0, "batches": 0}
    orig = SyncEngine._send

    async def counting(self, run, batch):  # noqa: ANN001
        res = await orig(self, run, batch)
        acc["tokens"] += res.embedded_tokens
        acc["written"] += res.written
        acc["batches"] += 1
        return res

    SyncEngine._send = counting  # type: ignore[method-assign]
    env = {**os.environ, "SCOUT_KEY": key, "SCOUT_GATEWAY_URL": gw, "SCOUT_KEYSTORE": "file"}
    args = ap.Namespace(profile=None, root=str(root), gateway_url=None)
    app = build_app(args, env)
    t0 = time.time()
    st = await app.sync(True, full=False)
    SyncEngine._send = orig  # type: ignore[method-assign]
    return {"state": st.state, "coverage": st.coverage, "files": st.files, "wall_s": round(time.time() - t0, 1), **acc}


def one(task: dict, key: str, gw: str, out: Path) -> None:
    root = repos.checkout(task["repo"], task["base_commit"])
    err = None
    for attempt in range(3):
        try:
            r = asyncio.run(sync_one(root, key, gw))
            break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}".replace(key, "<key>")
            r = None
    row = {"repo": task["repo"], "base_commit": task["base_commit"], "task_id": task["native_id"], "lang": task["lang"],
           "repo_type": task.get("repo_type"), "tracked_files": len(repos.ls_files(root))}
    if r is None:
        row.update(error=err)
    else:
        usd = r["tokens"] * VOYAGE_USD_PER_M / 1e6
        gb = r["written"] * BYTES_PER_ROW / 2**30
        row.update(r, voyage_usd=round(usd, 5), tpuf_write_usd_est=round(gb * TPUF_WRITE_USD_PER_GB, 6),
                   tpuf_storage_usd_month_est=round(gb * TPUF_STORE_USD_PER_GB_MONTH, 6))
    with (out / "index_costs.jsonl").open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    sys.stderr.write(f"{task['repo']:>40} {row.get('state')} files={row.get('files')} tok={row.get('tokens')} wall={row.get('wall_s')} err={row.get('error')}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=str(REPO / "eval" / "cc50_tasks.jsonl"))
    ap.add_argument("--gateway", default=os.environ.get("SCOUT_GATEWAY_URL", "https://scout-gateway.example.com"))
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    done = set()
    p = out / "index_costs.jsonl"
    if p.exists():
        done = {(r["repo"], r["base_commit"]) for r in map(json.loads, p.read_text().splitlines()) if r.get("state") == "ready"}
    tasks = [json.loads(x) for x in Path(a.tasks).read_text().splitlines() if x.strip()]
    tasks = [t for t in tasks if (t["repo"], t["base_commit"]) not in done]
    if a.limit: tasks = tasks[:a.limit]
    if repos.disk_free_gb() < 8: raise SystemExit("less than 8 GB free")
    ks = keys(); by = defaultdict(list)
    for t in tasks: by[t["repo"]].append(t)
    def lane(i_r):
        i, r = i_r
        for t in by[r]: one(t, ks[i % len(ks)], a.gateway, out)
    with ThreadPoolExecutor(a.width) as ex:
        list(ex.map(lane, enumerate(by)))


if __name__ == "__main__":
    main()
