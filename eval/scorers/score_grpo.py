"""Score GRPO arms: training curves (steps.jsonl) and every evaluated checkpoint, paired against the arm's start checkpoint (task-bootstrap 95% CIs)
plus PRD s15 against base. Metric definitions are scripts/score_sft_v2.py's, unchanged.

    python3 scripts/score_grpo.py [--runs a,b,c,d]  -> artifacts/grpo/summary.json, artifacts/grpo/curves_<run>.png, artifacts/grpo/curves_all.png, stdout tables

Eval rows: artifacts/grpo/<run>/eval_s<step>/shard<i>/sft_grpo-<run>-s<step>_{grep,sem}.jsonl (4 shards of the 50 dev tasks x 2 seeds, retain_reasoning on, same
config as sft_eval_app.py defaults). A checkpoint is validated before it is scored: 100 rows per arm, 50 tasks, seeds 0 and 1, 0 errors, model
wayfinder-grpo-<run>-s<step> on every row, retain_reasoning true. References (retain-on protocol): v3 grep artifacts/opd/round1/eval_d, v3 sem artifacts/sft_v4/sft_v3r_sem.jsonl,
v4 grep/sem artifacts/sft_v4/sft_v4_{grep,sem}.jsonl.
Agents-A1-35B (the competing model, owner 2026-10-27: every checkpoint must be validated against it) is a fourth reference, "a1":
artifacts/a1_baseline/sft_a1_{grep,sem}.jsonl from wayfinder/rollout/a1_eval_app.py (same tasks, scout product prompt, nudge, tools, retain on,
max_turns 40, max_seconds 3600; A1's own sampling and parser flags; see the Agents-A1 baseline write-up (not in this release)). Every healthy checkpoint arm gets `vs_a1` paired deltas
(checkpoint minus A1, same arm) next to `vs_v3`; the printed table shows reported recall, touched recall, turns and tokens against both.
"""
from __future__ import annotations

from display_names import disp, print  # noqa: E402,A004 - display-only relabel (v3 -> wayfinder-glimmer)
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wayfinder.rollout.nudges import NUDGES  # noqa: E402
from score_glimmer_baseline import boot, paired, per_task  # noqa: E402
from score_sft_v2 import METRICS as BASE_METRICS  # noqa: E402
from score_sft_v2 import ok_rows, prd15, stop_reason  # noqa: E402
from score_sft_v2 import deltas as _deltas  # noqa: E402,F401
from score_sft_v2 import summarize as _summarize  # noqa: E402

# ---- run f1 metrics (first-k = in REPORT ORDER over distinct reported files; the same definitions as scripts/score_test_split.py plus @12 and F1@12) ----
def reported(r: dict) -> list[str]:
    return list(dict.fromkeys(h["path"] for h in ((r.get("explore_result") or {}).get("results") or [])))


def rec_k(r: dict, k: int) -> float:
    gold = set(r["gold_files"])
    return len(gold & set(reported(r)[:k])) / len(gold) if gold else 0.0


def prec_k(r: dict, k: int | None) -> float:
    rep = reported(r)[:k] if k else reported(r)
    return len(set(rep) & set(r["gold_files"])) / len(rep) if rep else 0.0


def f1_k(r: dict, k: int = 12) -> float:
    p, rc = prec_k(r, k), rec_k(r, k)
    return 2 * p * rc / (p + rc) if p + rc > 0 else 0.0


F1_METRICS = {  # the run-f1 table, in this order
    "recall_at_5": lambda r: rec_k(r, 5),
    "recall_at_7": lambda r: rec_k(r, 7),
    "recall_at_12": lambda r: rec_k(r, 12),
    "precision": lambda r: prec_k(r, None),
    "f1_at_12": f1_k,
    "recall_reported": lambda r: r["recall_reported"],
    "n_reported": lambda r: len(reported(r)),
    "turns": lambda r: r["turns"],
    "ended_by_report": lambda r: stop_reason(r) == "report",
}
METRICS = {**BASE_METRICS, **F1_METRICS}
GATE_METRIC = "recall_at_5"


def summarize(rows: list[dict]) -> dict:
    out = _summarize(rows)
    for name, f in F1_METRICS.items():
        if rows and name not in out:
            m, lo, hi = boot(per_task(rows, f))
            out[name] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def deltas(a: list[dict], b: list[dict]) -> dict:
    out = {}
    for name, f in METRICS.items():
        d = paired(per_task(a, f), per_task(b, f))
        out[name] = {"delta": round(d[0], 4), "ci95": [round(d[1], 4), round(d[2], 4)], "n_tasks": d[3]}
    return out


# ---- release gate (pre-registered in the GRPO write-up (not in this release) before the f1 run launched; do not edit the thresholds after results exist) ----
PAD_MAX = 1.0  # run rk: dev n_reported must not exceed v3's (same arm) by more than this (absolute check at every dev eval)
GATE_SEM_FLOOR = -0.03  # "sem not worse": point delta >= 0 and CI lower bound >= this


def padding_check(ck: dict, refs: dict) -> dict:
    """Absolute dev padding check: per arm, checkpoint mean n_reported minus v3's mean n_reported (retain-on dev rows); flagged if > PAD_MAX."""
    out = {}
    for arm in ("grep", "sem"):
        sm = ck.get(arm, {}).get("summary") or {}
        ref = (refs.get("v3", {}).get(arm) or {}).get("summary") or {}
        if "n_reported" in sm and "n_reported" in ref:
            d = sm["n_reported"]["mean"] - ref["n_reported"]["mean"]
            out[arm] = {"n_reported": sm["n_reported"]["mean"], "v3_n_reported": ref["n_reported"]["mean"], "delta": round(d, 3), "flagged": d > PAD_MAX,
                        "recall_at_5": sm["recall_at_5"]["mean"], "v3_recall_at_5": ref["recall_at_5"]["mean"]}
    return out


def gate(ck: dict, pad: dict | None = None) -> dict:
    """A checkpoint qualifies iff, on dev (50 tasks x 2 seeds, both arms healthy): recall@5 grep minus v3 has CI lower bound > 0; recall@5 sem minus v3 has
    point delta >= 0 and CI lower bound >= GATE_SEM_FLOOR; and recall@5 minus A1 has CI lower bound > 0 on grep and on sem."""
    g, s = ck.get("grep", {}), ck.get("sem", {})
    if not (g.get("vs_v3") and s.get("vs_v3") and g.get("vs_a1") and s.get("vs_a1")):
        return {"qualifies": False, "reason": "an arm is unhealthy or a reference is missing"}
    m = GATE_METRIC
    c = {"grep_vs_v3_lb_gt0": g["vs_v3"][m]["ci95"][0] > 0,
         "sem_vs_v3_not_worse": s["vs_v3"][m]["delta"] >= 0 and s["vs_v3"][m]["ci95"][0] >= GATE_SEM_FLOOR,
         "grep_vs_a1_lb_gt0": g["vs_a1"][m]["ci95"][0] > 0, "sem_vs_a1_lb_gt0": s["vs_a1"][m]["ci95"][0] > 0}
    if pad is not None:  # run rk adds the absolute padding check (dev n_reported <= v3 + PAD_MAX on both arms)
        c["dev_n_reported_le_v3_plus_1"] = bool(pad) and not any(x["flagged"] for x in pad.values())
    return {"qualifies": all(c.values()), "checks": c, "grep_recall_at_5_vs_v3": g["vs_v3"][m], "sem_recall_at_5_vs_v3": s["vs_v3"][m],
            "grep_recall_at_5_vs_a1": g["vs_a1"][m], "sem_recall_at_5_vs_a1": s["vs_a1"][m]}


ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "artifacts/grpo"
REFS = {
    "v3": {"grep": ROOT / "artifacts/opd/round1/eval_d/sft_v3_grep.jsonl", "sem": ROOT / "artifacts/sft_v4/sft_v3r_sem.jsonl", "model": "wayfinder-v3"},
    "v4": {"grep": ROOT / "artifacts/sft_v4/sft_v4_grep.jsonl", "sem": ROOT / "artifacts/sft_v4/sft_v4_sem.jsonl", "model": "wayfinder-v4"},
    "a1": {"grep": ROOT / "artifacts/a1_baseline/sft_a1_grep.jsonl", "sem": ROOT / "artifacts/a1_baseline/sft_a1_sem.jsonl", "model": "a1"},  # competing model, matched harness
}


def load_all(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def gather(run: str, step: int, arm: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted((G / run / f"eval_s{step}").glob(f"shard*/sft_grpo-{run}-s{step}_{arm}.jsonl")):
        rows += load_all(f)
    return rows


def validate(rows: list[dict], model: str, n_rows: int = 100, n_tasks: int = 50, nudge: str | None = None) -> dict:
    ok = [r for r in rows if not r.get("error") and not r.get("infra_error")]
    turns = sorted(r["turns"] for r in ok)
    v = {"rows": len(rows), "ok_rows": len(ok), "errors": len(rows) - len(ok), "tasks": len({r["task_id"] for r in ok}), "seeds": sorted({r["seed"] for r in ok}),
         "models": sorted({str(r.get("model")) for r in ok}), "retain_reasoning": sorted({str(r.get("retain_reasoning")) for r in ok}),
         "turns_min_p50_max": [turns[0], turns[len(turns) // 2], turns[-1]] if turns else None}
    v["healthy"] = (len(rows) == n_rows and len(ok) == n_rows and v["tasks"] == n_tasks and v["seeds"] == [0, 1] and v["models"] == [model] and v["retain_reasoning"] == ["True"])
    if nudge is not None:  # run f1: every row must carry the f1 prompt (nudge text recorded by run_rollout)
        v["nudges"] = sorted({str(r.get("nudge")) for r in ok})
        v["healthy"] = v["healthy"] and v["nudges"] == [nudge]
    return v


def cell(s: dict, k: str) -> str:
    x = s[k]
    return f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"


def d_cell(d: dict, k: str) -> str:
    x = d[k]
    return f"{x['delta']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def curves(run: str) -> list[dict]:
    return [json.loads(x) for x in (G / run / "steps.jsonl").read_text().splitlines() if x.strip()] if (G / run / "steps.jsonl").exists() else []


def plot(runs: dict[str, list[dict]], evals: dict[str, list[dict]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("rollout.reward", "mean reward"), ("rollout.recall_reported", "reported recall"), ("rollout.all_gold", "all-gold rate"), ("rollout.n_reported", "n reported"),
              ("rollout.turns", "turns"), ("train.kl", "KL to reference (k3, per token)"), ("rollout.precision", "precision of reported files"), ("rollout.report_chars", "report length (chars)"),
              ("rollout.recall_at_5", "train recall@5"), ("rollout.kept_groups", "groups kept (of 16)"), ("train.gnorm", "grad norm"), ("eval", "dev reported recall (grep, 50 tasks x 2)")]
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    colors = {"a": "#1f77b4", "b": "#d62728", "c": "#2ca02c", "d": "#9467bd", "f1": "#ff7f0e", "rk": "#17becf", "smoke": "#7f7f7f"}
    for ax, (key, label) in zip(axes.flat, panels):
        for run, rows in runs.items():
            if key == "eval":
                pts = [(e["step"], e["grep"]["recall_reported"]["mean"], e["grep"]["recall_reported"]["ci95"]) for e in evals.get(run, []) if e.get("grep")]
                if pts:
                    ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[1] - p[2][0] for p in pts], [p[2][1] - p[1] for p in pts]], marker="o", color=colors.get(run), label=run, capsize=3)
                continue
            top, sub = key.split(".")
            xs, ys = [], []
            for r in rows:
                v = r.get(top, {}).get(sub)
                if v is not None:
                    xs.append(r["step"])
                    ys.append(v)
            ax.plot(xs, ys, marker=".", color=colors.get(run), label=run)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="a,b,c,d")
    ap.add_argument("--starts", default="a=v3,b=v3,c=v3,d=v4,f1=v3,rk=v3,cal_lo=v3,cal_hi=v3", help="run=reference start checkpoint (v3|v4)")
    ap.add_argument("--f1-table", default="", help="run name: print the run-f1 table (every 5 steps, paired against v3 and A1) and the gate, then exit after the usual outputs")
    a = ap.parse_args()
    starts = dict(x.split("=") for x in a.starts.split(","))
    base = ok_rows(ROOT / "artifacts/baseline/glimmer_dev.jsonl")
    out: dict = {"base": summarize(base), "refs": {}, "runs": {}}
    refrows = {}
    for r, p in REFS.items():
        refrows[r] = {arm: ok_rows(p[arm]) for arm in ("grep", "sem")}
        out["refs"][r] = {arm: {"summary": summarize(refrows[r][arm]), "validation": validate(load_all(p[arm]), p["model"])} for arm in ("grep", "sem") if refrows[r][arm]}
    allc, alle = {}, {}
    for run in [x for x in a.runs.split(",") if x]:
        cv = curves(run)
        allc[run] = cv
        ref = starts.get(run, "v3")
        ro: dict = {"start": ref, "steps_logged": len(cv), "checkpoints": {}}
        evs = []
        for ed in sorted((G / run).glob("eval_s*"), key=lambda p: int(p.name[6:])):
            step = int(ed.name[6:])
            ck: dict = {}
            e: dict = {"step": step}
            for arm in ("grep", "sem"):
                raw = gather(run, step, arm)
                if not raw:
                    continue
                v = validate(raw, f"wayfinder-grpo-{run}-s{step}", nudge=NUDGES["f1"] if run.startswith("f1") else NUDGES["teacher"] if run.startswith(("rk", "cal")) else None)
                rows = ok_rows_from(raw)
                ck[arm] = {"validation": v, "summary": summarize(rows) if rows else {}}
                if v["healthy"]:
                    ck[arm]["vs_start"] = deltas(rows, refrows[ref][arm])
                    ck[arm]["vs_v3"] = ck[arm]["vs_start"] if ref == "v3" else deltas(rows, refrows["v3"][arm])
                    if refrows["a1"][arm]:  # checkpoint minus A1-35B, same arm; absent (not an error) until artifacts/a1_baseline has rows
                        ck[arm]["vs_a1"] = deltas(rows, refrows["a1"][arm])
                    ck[arm]["vs_base"] = deltas(rows, base)
                    if arm == "grep":
                        ck[arm]["prd15_vs_base"] = prd15(ck[arm]["summary"], ck[arm]["vs_base"], out["base"])
                    e[arm] = ck[arm]["summary"]
            if ck:
                ro["checkpoints"][f"s{step}"] = ck
                pad = padding_check(ck, out["refs"])
                ro.setdefault("padding", {})[f"s{step}"] = pad
                ro.setdefault("gates", {})[f"s{step}"] = gate(ck, pad if run.startswith(("rk", "cal")) else None)
                evs.append(e)
        alle[run] = evs
        out["runs"][run] = ro
    G.mkdir(parents=True, exist_ok=True)
    (G / "summary.json").write_text(json.dumps(out, indent=2))
    for run, cv in allc.items():
        if cv:
            plot({run: cv}, {run: alle.get(run, [])}, G / f"curves_{run}.png", f"GRPO arm {run}")
    if any(allc.values()):
        plot(allc, alle, G / "curves_all.png", "GRPO arms")
    if a.f1_table:
        print_f1_table(out, a.f1_table)
    for arm in ("grep", "sem"):
        if out["refs"].get("a1", {}).get(arm):
            sm = out["refs"]["a1"][arm]["summary"]
            print(f"A1-35B reference ({arm}): reported recall {cell(sm, 'recall_reported')}, touched {cell(sm, 'recall_touched')}, turns {sm['turns']['mean']:.1f}, healthy={out['refs']['a1'][arm]['validation']['healthy']}")
    for run, ro in out["runs"].items():
        print(f"\n## arm {run} (start {ro['start']}), {ro['steps_logged']} steps logged")
        for ck, d in ro["checkpoints"].items():
            for arm, x in d.items():
                v = x["validation"]
                print(f"{ck} {arm}: healthy={v['healthy']} rows={v['rows']} ok={v['ok_rows']} tasks={v['tasks']} models={v['models']} err={v['errors']}")
                if "vs_start" in x:
                    print("  vs start: " + "; ".join(f"{k} {d_cell(x['vs_start'], k)}" for k in ("recall_reported", "all_gold_reported", "recall_touched", "turns", "tokens")))
                    if "vs_v3" in x:
                        print("  vs v3   : " + "; ".join(f"{k} {d_cell(x['vs_v3'], k)}" for k in ("recall_reported", "recall_at_5", "recall_at_7", "recall_touched", "turns", "tokens")))
                    if "vs_a1" in x:
                        print("  vs A1   : " + "; ".join(f"{k} {d_cell(x['vs_a1'], k)}" for k in ("recall_reported", "all_gold_reported", "recall_touched", "turns", "tokens")) + (
                            "  (CI lower bound > 0: beats A1)" if x["vs_a1"]["recall_reported"]["ci95"][0] > 0 else "  (does NOT clearly beat A1 on reported recall)"))
                    else:
                        print("  vs A1   : n/a (no A1 rows for this arm in artifacts/a1_baseline)")
                    print("  vs base : " + "; ".join(f"{k} {d_cell(x['vs_base'], k)}" for k in ("recall_reported", "recall_touched")))
                    if "prd15_vs_base" in x:
                        print("  PRD15 passes: " + str(sum(1 for c in x["prd15_vs_base"].values() if c["pass"])) + "/5")


def print_f1_table(out: dict, run: str) -> None:
    """One block per evaluated checkpoint and arm: the nine run-f1 metrics, checkpoint mean [CI] with its paired delta against v3 (retain-on rows) and against A1."""
    ro = out["runs"].get(run)
    if not ro:
        print(f"no run {run}")
        return
    names = list(F1_METRICS)
    for ck, d in ro["checkpoints"].items():
        for arm in ("grep", "sem"):
            x = d.get(arm)
            if not x:
                continue
            print(f"\n### {run} {ck} {arm}  healthy={x['validation']['healthy']}  rows={x['validation']['rows']}")
            if not x["validation"]["healthy"] or "vs_v3" not in x:
                print("  (not scored: unhealthy)")
                continue
            sm, ref3, refa = x["summary"], out["refs"]["v3"][arm]["summary"], out["refs"]["a1"][arm]["summary"]
            print("| metric | v3 | A1 | " + ck + " | minus v3 | minus A1 |\n|---|---|---|---|---|---|")
            for k in names:
                print(f"| {k} | {ref3[k]['mean']:.3f} | {refa[k]['mean']:.3f} | {cell(sm, k)} | {d_cell(x['vs_v3'], k)} | {d_cell(x['vs_a1'], k)} |")
        for arm, x in ro.get("padding", {}).get(ck, {}).items():
            print(f"PADDING {run} {ck} {arm}: n_reported {x['n_reported']:.2f} vs v3 {x['v3_n_reported']:.2f} ({x['delta']:+.2f}; flag if > +{PAD_MAX}) flagged={x['flagged']}; recall@5 {x['recall_at_5']:.3f} vs v3 {x['v3_recall_at_5']:.3f}")
        g = ro.get("gates", {}).get(ck)
        if g:
            print(f"GATE {ck}: qualifies={g['qualifies']} {json.dumps(g.get('checks', g.get('reason')))}")


def ok_rows_from(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("error") and not r.get("infra_error")]


if __name__ == "__main__":
    main()
