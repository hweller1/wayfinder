"""Read-only turbopuffer namespace metadata for the cc50 indexes: approx_logical_bytes / approx_row_count.

Reads each checkout's pinned repo_id/branch_id from .git/scout/state.json, addresses `scout_eval_<repo>_<branch>` and
its `__tree`, and merges tpuf_bytes / tpuf_rows (chunks + tree) into index_costs rows. No pricing here. Keys never printed.
"""
import json, os, sys
from pathlib import Path
from eval.harness import repos

def load_key():
    for line in (Path(__file__).resolve().parents[2] / ".env").read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if line.startswith("TURBOPUFFER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return os.environ["TURBOPUFFER_API_KEY"]

def meta(c, ns):
    try:
        m = c.namespace(ns).metadata()
        return {"bytes": m.approx_logical_bytes, "rows": m.approx_row_count}
    except Exception as e:  # noqa: BLE001
        return {"bytes": None, "rows": None, "err": type(e).__name__}

def main(path: str) -> None:
    import turbopuffer
    c = turbopuffer.Turbopuffer(api_key=load_key(), region=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"))
    p = Path(path); rows = [json.loads(l) for l in p.read_text().splitlines()]
    for r in rows:
        root = repos.main_dir(r["repo"])
        try:
            st = json.loads((root / ".git" / "scout" / "state.json").read_text())
        except OSError:
            continue
        base = f"scout_{st['tenant']}_{st['repo_id']}_{st['last_branch']}"
        a, t = meta(c, base), meta(c, base + "__tree")
        if a["bytes"] is None: continue
        r["tpuf_chunk_ns"] = {"bytes": a["bytes"], "rows": a["rows"]}
        r["tpuf_tree_ns"] = {"bytes": t["bytes"], "rows": t["rows"]}
        r["tpuf_bytes"] = a["bytes"] + (t["bytes"] or 0)
        r["tpuf_rows"] = a["rows"] + (t["rows"] or 0)
    shared = {n: meta(c, f"scout_eval_{n}") for n in ("cache", "meta")}
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(json.dumps({"shared_namespaces": shared, "rows_with_tpuf": sum("tpuf_bytes" in r for r in rows), "n": len(rows)}))

if __name__ == "__main__":
    main(sys.argv[1])
