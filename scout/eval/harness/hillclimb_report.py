"""Hill-climb TUNE/CONFIRM table and paired task-bootstrap contrasts.

    python -m eval.harness.hillclimb_report DIR --variants base,h_verbatim,... [--vs x_guid] [--seed 0]
"""
import argparse, json, random, statistics
from pathlib import Path


def load(d: Path, v: str, seed: int) -> dict:
    p = d / f"rows_{v}_s{seed}.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    out = {}
    for r in rows:  # last valid row per task wins
        if not r.get("problem"):
            out[r["task_id"]] = r
    return out


def row_stats(r):
    deleg = bool(r.get("explore_agent_calls") or r.get("scout_agent_calls") or r.get("scout_explore_calls_any"))
    return dict(r5=r["recall5"], r12=r["recall12"], p12=r["precision12"], n=r["n_reported"], deleg=deleg,
                wall=r["wall_s"], cost=r.get("cost_usd") or 0.0)


def mean(xs): return sum(xs) / len(xs) if xs else float("nan")


def boot(a: dict, b: dict, key: str, n=10000, seed=1):
    ids = sorted(set(a) & set(b))
    d = [a[i][key] - b[i][key] for i in ids]
    if not d: return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(seed)
    m = sorted(mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(n))
    return mean(d), m[int(.025 * n)], m[int(.975 * n)], len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--variants", required=True); ap.add_argument("--vs", default="")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ids", default="")
    a = ap.parse_args()
    d = Path(a.dir); vs = a.variants.split(",")
    data = {v: {t: row_stats(r) for t, r in load(d, v, a.seed).items()} for v in vs}
    print("| variant | n | recall@5 | recall@12 | precision@12 | n_reported | delegated | median wall s | $/run |")
    print("|---|---|---|---|---|---|---|---|---|")
    for v in vs:
        s = list(data[v].values())
        if not s: print(f"| {v} | 0 |"); continue
        print(f"| {v} | {len(s)} | {mean([x['r5'] for x in s]):.3f} | {mean([x['r12'] for x in s]):.3f} | {mean([x['p12'] for x in s]):.3f} | "
              f"{mean([x['n'] for x in s]):.1f} | {sum(x['deleg'] for x in s)}/{len(s)} | {statistics.median([x['wall'] for x in s]):.0f} | {mean([x['cost'] for x in s]):.3f} |")
    if a.vs:
        print(f"\nPaired contrasts against {a.vs} (task bootstrap, 10000, 95%)\n")
        print("| variant | d recall@5 | d recall@12 | d precision@12 | n paired |"); print("|---|---|---|---|---|")
        for v in vs:
            if v == a.vs: continue
            cells = []
            for k in ("r5", "r12", "p12"):
                m, lo, hi, n = boot(data[v], data[a.vs], k)
                cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"| {v} | " + " | ".join(cells) + f" | {n} |")


if __name__ == "__main__":
    main()
