"""FOUR-WAY scorer (v3, A1, Sonnet 5.5 high, GPT-6.1 Sol): a copy of scripts/score_heldout_hard.py with models sonnet and gpt added, contrasts
v3-A1, v3-Sonnet, v3-GPT, A1-Sonnet, A1-GPT, Sonnet-GPT, per-group breakdowns for every contrast, and a cost section. Writes
artifacts/heldout_hard/summary_4way.json and score_4way.txt; summary.json (the committed v3/A1 result) is not touched.

Original docstring: Score the held-out hard v1 set (data/heldout_hard_v1.json, 200 tasks / 47 repos): a copy of scripts/score_ood_langs.py with the task set,
directory (artifacts/heldout_hard) and groupings changed (language, repo_type, category); models v3 and A1 only. Metrics and CIs unchanged.

    python3 scripts/score_ood_langs.py   -> artifacts/ood_langs/summary.json, rows/*.jsonl.gz, score.txt (stdout)

Rows: artifacts/ood_langs/shards/<model>_s<i>/sft_<variant>_<arm>.jsonl, model in {v3, a1, base}, arm in {grep, sem}
(written by wayfinder/rollout/test_eval_app.py for v3 and base, wayfinder/rollout/a1_eval_app.py for a1). Shard files are merged
and gzipped into artifacts/ood_langs/rows/<model>_<arm>.jsonl.gz (the committed copy; a clean clone scores from the gz).

Unit of analysis: the task (seeds averaged within task). Paired contrasts use the tasks that have at least one valid row in both
arms. CIs (10,000 resamples, seed 11, percentile): (1) task bootstrap, resample tasks; (2) repo-cluster bootstrap, resample the 31
repos with replacement and pool their tasks (mean of the pooled task deltas). Metric definitions are those of scripts/score_sft_v2.py
(reported recall, all-gold reported, ended by report, turns, tokens) plus recall@5, recall@7 (first k distinct reported paths, in
report order; gold recall) and precision (distinct reported gold files / distinct reported files, 0 when nothing is reported).
"""
from __future__ import annotations

from display_names import disp, print  # noqa: E402,A004 - display-only relabel (v3 -> wayfinder-glimmer)
import collections
import glob
import gzip
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sft_v2 import n_asst, n_calls, stop_reason  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TD = ROOT / "artifacts/heldout_hard"
N_BOOT, SEED = 10_000, 11
MODELS = {"v3": {"name": "wayfinder-v3", "weights": "/data/sft/v3_merged"},
          "a1": {"name": "a1", "weights": "/weights/Agents-A1-FP8"},
          "sonnet": {"name": "claude-sonnet-5-5", "weights": "llm_key:claude-sonnet-5-5"},
          "gpt": {"name": "gpt-6.1-sol", "weights": "llm_key:gpt-6.1-sol"},
}
FRONTIER = ("sonnet", "gpt")
PAIRS = [("v3", "a1"), ("v3", "sonnet"), ("v3", "gpt"), ("a1", "sonnet"), ("a1", "gpt"), ("sonnet", "gpt")]
LABEL = {"v3": "wayfinder-glimmer", "a1": "A1", "sonnet": "Sonnet 5.5 high", "gpt": "GPT-6.1 Sol"}
# Open-model API-equivalent price (USD per 1M tokens: fresh input, cached input, output): DeepInfra Qwen3.5-35B-A3B list, Sep 2026, the
# basis scripts/score_value_props.py and the A1 serving-price assumption in docs/e2e_eval_v2.md use. v3 (30B-A3B) is priced the same.
# The v3/A1 rows record cached=0 (vLLM reported no cached tokens), so every input token is priced fresh: an upper bound for the open models.
OPEN_RATE = (0.14, 0.05, 1.00)
H100_USD_PER_S = 0.001097  # Modal list ~$3.95/h, as the eval apps
SEEDS = (0, 1)
ARMS = ("grep", "sem")
NUDGE_TEXT = "Reply with a tool call. When you have enough evidence, call report."
GROUP_KEYS = ("language", "repo_type", "category")


def lang_group(lang: str) -> str:
    return lang


def reported(r: dict) -> list[str]:
    return list(dict.fromkeys(h["path"] for h in ((r.get("explore_result") or {}).get("results") or [])))


def rec_k(r: dict, k: int) -> float:
    gold = set(r["gold_files"])
    return len(gold & set(reported(r)[:k])) / len(gold) if gold else 0.0


def precision(r: dict) -> float:
    rep = reported(r)
    return len(set(rep) & set(r["gold_files"])) / len(rep) if rep else 0.0


METRICS = {
    "recall_reported": lambda r: r["recall_reported"],
    "recall_at_5": lambda r: rec_k(r, 5),
    "recall_at_7": lambda r: rec_k(r, 7),
    "precision": precision,
    "all_gold_reported": lambda r: r["all_gold_reported"],
    "turns": lambda r: r["turns"],
    "tokens": lambda r: r["usage"]["in"] + r["usage"]["out"],
    "ended_by_report": lambda r: stop_reason(r) == "report",
    "n_reported": lambda r: len(reported(r)),
    "recall_touched": lambda r: r["recall_seen_k5"],
}
PRIMARY = ("v3", "a1", "grep", "recall_reported")
OUT_JSON, OUT_TXT = "summary_4way.json", "score_4way.txt"


def ok_of(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("error") and not r.get("infra_error")]


def load_arm(model: str, arm: str) -> list[dict]:
    """Merge shard files (local, untracked) into rows/<model>_<arm>.jsonl.gz; later rows override earlier ones per (task, seed), ok over error."""
    gz = TD / "rows" / f"{model}_{arm}.jsonl.gz"
    files = sorted(glob.glob(str(TD / "shards" / f"{model}_s*" / f"sft_*_{arm}.jsonl")))
    raw: list[dict] = []
    for f in files:
        raw += [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]
    if raw:
        best: dict[tuple, dict] = {}
        for r in raw:  # per (task, seed): an ok row beats an error row; among equals the later one wins
            k = (r["task_id"], r["seed"])
            if k in best and not (best[k].get("error") or best[k].get("infra_error")) and (r.get("error") or r.get("infra_error")):
                continue
            best[k] = r
        rows = list(best.values())
        gz.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz, "wt") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        return rows
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return [json.loads(x) for x in f if x.strip()]
    return []


def per_task(rows: list[dict], f) -> dict[str, float]:
    g: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        v = f(r)
        if v is not None:
            g[r["task_id"]].append(float(v))
    return {k: sum(v) / len(v) for k, v in g.items()}


def pct(xs: list[float]) -> list[float]:
    xs = sorted(xs)
    return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]


def boot_task(vals: list[float], seed: int = SEED) -> list[float]:
    rng = random.Random(seed)
    return pct([sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(N_BOOT)])


def boot_cluster(items: list[tuple[str, float]], seed: int = SEED) -> list[float]:
    by: dict[str, list[float]] = collections.defaultdict(list)
    for repo, d in items:
        by[repo].append(d)
    repos = list(by)
    rng = random.Random(seed)
    out = []
    for _ in range(N_BOOT):
        vals = [d for r in (rng.choice(repos) for _ in repos) for d in by[r]]
        out.append(sum(vals) / len(vals))
    return pct(out)


def mean_ci(pt: dict[str, float]) -> dict:
    v = list(pt.values())
    if not v:
        return {"mean": None, "ci95": None, "n_tasks": 0}
    return {"mean": round(sum(v) / len(v), 4), "ci95": [round(x, 4) for x in boot_task(v)], "n_tasks": len(v)}


def contrast(a: list[dict], b: list[dict], f, repo_of: dict[str, str], cluster: bool = True) -> dict:
    pa, pb = per_task(a, f), per_task(b, f)
    ks = sorted(set(pa) & set(pb))
    d = [pa[k] - pb[k] for k in ks]
    if not d:
        return {"delta": None, "n_tasks": 0}
    out = {"delta": round(sum(d) / len(d), 4), "ci95_task": [round(x, 4) for x in boot_task(d)], "n_tasks": len(d)}
    if cluster:
        out["ci95_repo_cluster"] = [round(x, 4) for x in boot_cluster([(repo_of[k], x) for k, x in zip(ks, d)])]
        out["n_repos"] = len({repo_of[k] for k in ks})
    return out


def parse_health(rows: list[dict]) -> dict:
    out = dict(rollouts=len(rows), assistant_turns=0, tool_calls=0, no_tool_call_turns=0, bad_json_args=0, unknown_tool=0, invalid_args=0,
               rollouts_with_malformed=0, rollouts_stopped_3_malformed=0, rollouts_with_nudge=0, leaked_tool_markup_in_content=0)
    for r in rows:
        bad = 0
        for m in r["messages"]:
            if m["role"] == "assistant":
                out["assistant_turns"] += 1
                calls = m.get("tool_calls") or []
                out["tool_calls"] += len(calls)
                c = m.get("content") or ""
                out["no_tool_call_turns"] += not calls
                out["leaked_tool_markup_in_content"] += ("<tool_call>" in c or "<function=" in c)
            elif m["role"] == "tool":
                t = str(m.get("content", ""))
                if t.startswith("error: the arguments of") and "not valid JSON" in t:
                    out["bad_json_args"] += 1; bad += 1
                elif t.startswith("error: unknown tool"):
                    out["unknown_tool"] += 1; bad += 1
                elif t.startswith("error: invalid arguments"):
                    out["invalid_args"] += 1; bad += 1
        out["rollouts_with_malformed"] += bad > 0
        out["rollouts_with_nudge"] += any(m["role"] == "user" and m.get("content") == NUDGE_TEXT for m in r["messages"])
        out["rollouts_stopped_3_malformed"] += any("consecutive malformed" in g for g in (r.get("explore_result") or {}).get("gaps", []))
    out["malformed_calls_total"] = out["bad_json_args"] + out["unknown_tool"] + out["invalid_args"]
    out["malformed_rate_per_call"] = round(out["malformed_calls_total"] / max(out["tool_calls"], 1), 4)
    return out


def validate(model: str, arm: str, rows: list[dict], tasks: dict[str, dict]) -> dict:
    spec = MODELS[model]
    ok = ok_of(rows)
    v: dict = {"rows": len(rows), "ok_rows": len(ok), "errors": len(rows) - len(ok),
               "error_messages": dict(collections.Counter(str(r.get("error"))[:100] for r in rows if r.get("error") or r.get("infra_error"))),
               "models": sorted({str(r.get("model")) for r in ok}), "weights": sorted({str(r.get("weights")) for r in ok}),
               "retain_reasoning": sorted({str(r.get("retain_reasoning")) for r in ok}), "max_turns": sorted({r.get("max_turns") for r in ok}),
               "max_seconds": sorted({r.get("max_seconds") for r in ok}), "seeds": sorted({r["seed"] for r in ok}),
               "sampling": sorted({f"T{r.get('temperature')} top_p{r.get('top_p')} top_k{r.get('top_k')} pp{r.get('presence_penalty')} think{r.get('enable_thinking')}" for r in ok}),
               "pairs_unique": len({(r["task_id"], r["seed"]) for r in ok}) == len(ok)}
    per_seed = {}
    for s in (0, 1):
        got = {r["task_id"] for r in ok if r["seed"] == s}
        per_seed[s] = {"tasks_ok": len(got), "missing": sorted(set(tasks) - got)[:5], "n_missing": len(set(tasks) - got), "unexpected": len(got - set(tasks))}
    v["per_seed_coverage"] = per_seed
    turns = sorted(r["turns"] for r in ok)
    walls = sorted(r["wall_s"] for r in ok)
    if ok:
        v["turns_min_p50_max"] = [turns[0], turns[len(turns) // 2], turns[-1]]
        v["wall_s_min_p50_max"] = [walls[0], walls[len(walls) // 2], walls[-1]]
    v["n429"], v["n_retries"] = sum(r.get("n429", 0) for r in ok), sum(r.get("n_retries", 0) for r in ok)
    v["with_reasoning_rate"] = round(sum(1 for r in ok if any(m.get("reasoning_content") for m in r["messages"] if m["role"] == "assistant")) / max(len(ok), 1), 3)
    if arm == "sem":
        v["semantic_offered"] = sum(1 for r in ok if r.get("semantic"))
        v["semantic_not_offered_reasons"] = dict(collections.Counter(str(r.get("semantic_reason"))[:60] for r in ok if not r.get("semantic")))
        v["rollouts_calling_semantic_search"] = sum(1 for r in ok if any(t["function"]["name"] == "semantic_search" for m in r["messages"] if m["role"] == "assistant" for t in (m.get("tool_calls") or [])))
    if model in FRONTIER:
        v["meta_settings"] = sorted({json.dumps((r.get("meta") or {}).get("settings"), sort_keys=True)[:200] for r in ok})
        v["meta_model_id_ok"] = all((r.get("meta") or {}).get("model_id") == spec["name"] for r in ok)
        v["billed_cost_present"] = all(isinstance(r.get("cost_usd"), (int, float)) and r["cost_usd"] > 0 for r in ok)
        v["wall_plausible"] = all(3 < r["wall_s"] < 3700 for r in ok)
        v["task_id_in_set"] = all(r["task_id"] in tasks for r in ok)
    v["healthy"] = (v["models"] == [spec["name"]] and v["weights"] == [spec["weights"]] and v["retain_reasoning"] == ["True"] and v["max_turns"] == [40]
                    and v["max_seconds"] == [3600] and v["seeds"] == [0, 1] and v["pairs_unique"]
                    and all(per_seed[s]["n_missing"] == 0 and per_seed[s]["unexpected"] == 0 for s in (0, 1)) and v["errors"] == 0
                    and (model not in FRONTIER or (v["meta_model_id_ok"] and v["billed_cost_present"] and v["wall_plausible"] and v["task_id_in_set"])))
    return v


def summarize(rows: list[dict]) -> dict:
    out = {"n_rollouts": len(rows), "n_tasks": len({r["task_id"] for r in rows})}
    for k, f in METRICS.items():
        out[k] = mean_ci(per_task(rows, f))
    for k in ("turns", "tokens"):
        out[k]["p50"] = round(st.median(METRICS[k](r) for r in rows), 1)
    out["stop_reasons"] = dict(collections.Counter(stop_reason(r) for r in rows))
    out["calls_per_turn"] = round(sum(n_calls(r) for r in rows) / sum(n_asst(r) for r in rows), 3)
    return out


def fmt(x: dict, nd: int = 3, tok: bool = False) -> str:
    if x.get("mean") is None:
        return "n/a"
    if tok:
        return f"{x['mean'] / 1000:.0f}k"
    return f"{x['mean']:.{nd}f} [{x['ci95'][0]:.{nd}f}, {x['ci95'][1]:.{nd}f}]"


def dfmt(c: dict, nd: int = 3, scale: float = 1.0, unit: str = "") -> str:
    if c.get("delta") is None:
        return "n/a"
    s = f"{c['delta'] * scale:+.{nd}f}{unit} [{c['ci95_task'][0] * scale:+.{nd}f}, {c['ci95_task'][1] * scale:+.{nd}f}]"
    if "ci95_repo_cluster" in c:
        s += f" / cluster [{c['ci95_repo_cluster'][0] * scale:+.{nd}f}, {c['ci95_repo_cluster'][1] * scale:+.{nd}f}]"
    return s


def open_price_usd(r: dict) -> float:
    """Open-model API-equivalent price of a v3/A1 row's tokens (cached recorded as 0, so all input is priced fresh)."""
    u = r["usage"]
    fresh = max(u["in"] - u.get("cached", 0), 0)
    return (fresh * OPEN_RATE[0] + u.get("cached", 0) * OPEN_RATE[1] + u["out"] * OPEN_RATE[2]) / 1e6


def gpu_cost(model: str, arm: str) -> dict | None:
    """Modal H100 list-price cost of an arm from the per-shard meta files (function seconds incl. vLLM load, concurrent rollouts shared)."""
    fs = sorted(glob.glob(str(TD / "shards" / f"{model}_s*" / f"sft_*_{arm}.meta.json")))
    if not fs:
        return None
    secs = usd = 0.0
    for f in fs:
        m = json.loads(Path(f).read_text())
        secs += m.get("gpu_function_seconds", 0)
        usd += m.get("gpu_cost_estimate_usd") or m.get("gpu_function_seconds", 0) * H100_USD_PER_S
    return {"shards": len(fs), "gpu_seconds": round(secs, 1), "usd_total": round(usd, 2)}


def cost_section(rows: dict, tasks: dict) -> dict:
    """Three kinds of cost, never mixed. billed: the LLM gateway tokens x rate table (Sonnet, GPT only). open_price_equiv: the v3/A1 tokens at the open
    API-equivalent rate. gpu: Modal H100 list price per arm (v3, A1 only). Dollars per task = mean over tasks of the seed-averaged rollout cost."""
    c: dict = {"definitions": {
        "billed_usd_per_task": "the LLM gateway tokens x rate table (fresh/cache read/cache write/output), Sonnet 5.5 and GPT-6.1 Sol only",
        "open_price_equiv_usd_per_task": f"v3/A1 tokens at ${OPEN_RATE[0]}/M in, ${OPEN_RATE[2]}/M out (cached recorded as 0, so an upper bound): what an API at an open-model price would charge",
        "gpu_usd_per_task": "Modal H100 list price ($3.95/h) of the arm's container seconds incl. vLLM load, divided by 400 rollouts; v3/A1 only"}}
    for (m, ar), rs in rows.items():
        if not rs:
            continue
        e: dict = {}
        if m in FRONTIER:
            e["billed_usd_per_task"] = mean_ci(per_task(rs, lambda r: r["cost_usd"]))
            e["billed_usd_total_this_arm"] = round(sum(r["cost_usd"] for r in rs), 2)
            b = {k: sum(r["usage"]["billed"][k] for r in rs) for k in ("fresh", "cache_read", "cache_write", "out")}
            e["billed_tokens_total"] = b
        else:
            e["open_price_equiv_usd_per_task"] = mean_ci(per_task(rs, open_price_usd))
            g = gpu_cost(m, ar)
            if g:
                g["gpu_usd_per_task"] = round(g["usd_total"] / max(len(rs), 1), 4)
                e["gpu"] = g
        c[f"{m}_{ar}"] = e
    return c


def main() -> None:
    pool = json.loads((ROOT / "data/heldout_hard_v1.json").read_text())["tasks"]
    test = {t["task_id"]: {"repo_full": t["repo"], "language": t["language"], "repo_type": t.get("repo_type"), "category": t.get("category")} for t in pool}
    excl = json.loads((TD / "exclusions.json").read_text()) if (TD / "exclusions.json").exists() else {"excluded_task_ids": []}
    tasks = {t: v for t, v in test.items() if t not in set(excl["excluded_task_ids"])}
    repo_of = {t: v["repo_full"] for t, v in tasks.items()}
    lang_of = {t: lang_group(v["language"]) for t, v in tasks.items()}
    raw = {(m, a): load_arm(m, a) for m in MODELS for a in ARMS}
    rows = {k: [r for r in ok_of(v) if r["task_id"] in tasks] for k, v in raw.items()}
    out: dict = {"protocol": "docs/heldout_hard_v1_eval.md", "n_tasks_preregistered": len(test), "n_tasks_scored_universe": len(tasks), "excluded": excl,
                 "validation": {}, "parse_health": {}, "arms": {}, "contrasts": {}, "per_language": {}, "per_repo": {}, "win_loss_by_repo": {}}
    for (m, a), rs in raw.items():
        if rs:
            out["validation"][f"{m}_{a}"] = validate(m, a, rs, tasks)
            out["parse_health"][f"{m}_{a}"] = parse_health(rows[(m, a)])
            out["arms"][f"{m}_{a}"] = summarize(rows[(m, a)])
    plan = [(x, y, ar) for x, y in PAIRS for ar in ARMS]
    for x, y, a in plan:
        if rows[(x, a)] and rows[(y, a)]:
            out["contrasts"][f"{x}_minus_{y}_{a}"] = {k: contrast(rows[(x, a)], rows[(y, a)], f, repo_of, cluster=True) for k, f in METRICS.items()}
    for m in MODELS:
        if rows[(m, "grep")] and rows[(m, "sem")]:
            out["contrasts"][f"{m}_sem_minus_grep"] = {k: contrast(rows[(m, "sem")], rows[(m, "grep")], f, repo_of, cluster=True) for k, f in METRICS.items() if k in ("recall_reported", "turns", "tokens")}
    # primary and decision rule
    x, y, a, k = PRIMARY
    pc = out["contrasts"].get(f"{x}_minus_{y}_{a}", {}).get(k)
    if pc and pc.get("delta") is not None:
        out["primary"] = {"contrast": "v3 minus A1, grep arm, reported recall", **pc, "task_lower_gt_0": pc["ci95_task"][0] > 0, "cluster_lower_gt_0": pc["ci95_repo_cluster"][0] > 0}
        out["primary"]["generalization_claim_holds"] = out["primary"]["task_lower_gt_0"] and out["primary"]["cluster_lower_gt_0"]
    # per language and per repo (reported recall; contrasts with task CI only, descriptive)
    out["per_group"] = {}
    for gk in GROUP_KEYS:
      for lg in sorted({v[gk] for v in tasks.values()}):
        ts = {t for t in tasks if tasks[t][gk] == lg}
        e: dict = {"n_tasks": len(ts), "n_repos": len({repo_of[t] for t in ts})}
        for m in MODELS:
            for ar in ARMS:
                rs = [r for r in rows[(m, ar)] if r["task_id"] in ts]
                if rs:
                    e[f"{m}_{ar}_recall_reported"] = mean_ci(per_task(rs, METRICS["recall_reported"]))
        for x, y, ar in plan:
            ra, rb = [r for r in rows[(x, ar)] if r["task_id"] in ts], [r for r in rows[(y, ar)] if r["task_id"] in ts]
            if ra and rb:
                for mk in ("recall_reported", "recall_at_5"):
                    e[f"{x}_minus_{y}_{ar}_{mk}"] = contrast(ra, rb, METRICS[mk], repo_of, cluster=False)
        out["per_group"].setdefault(gk, {})[lg] = e
    repos = sorted(set(repo_of.values()))
    for pair_name, (x, y, ar) in {f"{x}_minus_{y}_{ar}": (x, y, ar) for x, y in PAIRS for ar in ARMS}.items():
        pa, pb = per_task(rows[(x, ar)], METRICS["recall_reported"]), per_task(rows[(y, ar)], METRICS["recall_reported"])
        if not pa or not pb:
            continue
        wl = {"win": 0, "loss": 0, "tie": 0}
        per: dict = {}
        for rp in repos:
            ts = [t for t in tasks if repo_of[t] == rp and t in pa and t in pb]
            if not ts:
                continue
            d = sum(pa[t] - pb[t] for t in ts) / len(ts)
            res = "win" if d > 1e-9 else "loss" if d < -1e-9 else "tie"
            wl[res] += 1
            per[rp] = {"n_tasks": len(ts), "language": lang_of[ts[0]], "delta": round(d, 4), "result": res,
                       f"{x}": round(sum(pa[t] for t in ts) / len(ts), 4), f"{y}": round(sum(pb[t] for t in ts) / len(ts), 4)}
        tw = {"win": sum(1 for t in pa if t in pb and pa[t] - pb[t] > 1e-9), "loss": sum(1 for t in pa if t in pb and pa[t] - pb[t] < -1e-9),
              "tie": sum(1 for t in pa if t in pb and abs(pa[t] - pb[t]) <= 1e-9)}
        out["win_loss_by_repo"][pair_name] = {"repos": wl, "tasks": tw}
        out["per_repo"][pair_name] = per
    # semantic arm restricted to the tasks that had an index (the others ran grep-only inside the sem arm)
    idx_tasks = {r["task_id"] for r in rows[("v3", "sem")] if r.get("semantic")}
    out["semantic_indexed_subset"] = {"n_tasks": len(idx_tasks)}
    for m in MODELS:
        rs = [r for r in rows[(m, "sem")] if r["task_id"] in idx_tasks]
        if rs:
            out["semantic_indexed_subset"][m] = {k: mean_ci(per_task(rs, METRICS[k])) for k in ("recall_reported", "recall_at_5", "precision", "turns")}
    for x, y in PAIRS:
        ra, rb = [r for r in rows[(x, "sem")] if r["task_id"] in idx_tasks], [r for r in rows[(y, "sem")] if r["task_id"] in idx_tasks]
        if ra and rb:
            out["semantic_indexed_subset"][f"{x}_minus_{y}_recall_reported"] = contrast(ra, rb, METRICS["recall_reported"], repo_of, cluster=True)
    # reasoning retained by the template? growth of the prompt between consecutive turns over the previous turn's output tokens
    out["reasoning_render_evidence"] = {}
    for m in ("v3", "a1"):
        pairs = []
        for r in rows[(m, "grep")] + rows[(m, "sem")]:
            ms, tl = r["messages"], [t for t in r["turn_log"] if t.get("ok")]
            if any(x["role"] == "user" and x.get("content") == NUDGE_TEXT for x in ms):
                continue
            ai = [i for i, x in enumerate(ms) if x["role"] == "assistant"]
            for k in range(min(len(ai), len(tl)) - 1):
                if sum(len(str(x["content"])) for x in ms[ai[k] + 1: ai[k + 1]] if x["role"] == "tool") < 120:
                    u0, u1 = tl[k]["usage"], tl[k + 1]["usage"]
                    pairs.append((u1["in"] - u0["in"], u0["out"]))
        big = [(d, o) for d, o in pairs if o >= 300]
        out["reasoning_render_evidence"][m] = {"turn_pairs_output_ge_300": len(big),
            "median_growth_over_output": round(st.median(d / o for d, o in big), 2) if big else None,
            "share_growth_ge_0.8x_output": round(sum(d >= 0.8 * o for d, o in big) / len(big), 2) if big else None,
            "rollouts_with_reasoning_content": sum(1 for r in rows[(m, "grep")] + rows[(m, "sem")] if any(x.get("reasoning_content") for x in r["messages"] if x["role"] == "assistant")),
            "rollouts": len(rows[(m, "grep")] + rows[(m, "sem")])}
    out["cost"] = cost_section(rows, tasks)
    (TD / OUT_JSON).write_text(json.dumps(out, indent=2))
    # ---- printed tables
    L = []
    L.append("| arm | n tasks | reported recall | recall@5 | recall@7 | precision | all-gold | turns mean (p50) | tokens | ended by report | n reported |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, s in out["arms"].items():
        L.append(f"| {key} | {s['n_tasks']} | {fmt(s['recall_reported'])} | {fmt(s['recall_at_5'])} | {fmt(s['recall_at_7'])} | {fmt(s['precision'])} | {fmt(s['all_gold_reported'], 2)} | "
                 f"{s['turns']['mean']:.1f} ({s['turns']['p50']:.0f}) | {fmt(s['tokens'], tok=True)} | {fmt(s['ended_by_report'], 2)} | {s['n_reported']['mean']:.1f} |")
    L.append("")
    L.append("contrast (task CI / repo-cluster CI)")
    for name, c in out["contrasts"].items():
        for k, v in c.items():
            if v.get("delta") is None:
                continue
            L.append(f"{name:26s} {k:18s} " + (dfmt(v, 1, 1 / 1000, "k") if k == "tokens" else dfmt(v, 1 if k == "turns" else 3)) + f"  (n={v['n_tasks']})")
    if "primary" in out:
        L.append(f"\nPRIMARY: {json.dumps(out['primary'])}")
    L.append("\ncost per task (never mixed: billed = the LLM gateway tokens x rates; open-equiv = tokens at open-model price; gpu = Modal H100 list)")
    for key, e in out["cost"].items():
        if key == "definitions":
            continue
        parts = []
        if "billed_usd_per_task" in e:
            parts.append(f"billed ${e['billed_usd_per_task']['mean']:.4f}/task (total ${e['billed_usd_total_this_arm']})")
        if "open_price_equiv_usd_per_task" in e:
            parts.append(f"open-price-equiv ${e['open_price_equiv_usd_per_task']['mean']:.4f}/task")
        if "gpu" in e:
            parts.append(f"gpu ${e['gpu']['gpu_usd_per_task']:.4f}/rollout (arm total ${e['gpu']['usd_total']}, {e['gpu']['gpu_seconds']:.0f}s)")
        L.append(f"{key:12s} " + "; ".join(parts))
    L.append("\nwin/loss by repo " + json.dumps(out["win_loss_by_repo"]))
    L.append("\nvalidation")
    for k, v in out["validation"].items():
        L.append(f"{k}: rows={v['rows']} ok={v['ok_rows']} errors={v['errors']} models={v['models']} weights={v['weights']} retain={v['retain_reasoning']} seeds={v['seeds']} "
                 f"missing={ {s: v['per_seed_coverage'][s]['n_missing'] for s in (0, 1)} } healthy={v['healthy']}")
    L.append("\nsemantic indexed subset " + json.dumps(out["semantic_indexed_subset"]))
    L.append("\nreasoning render " + json.dumps(out["reasoning_render_evidence"]))
    L.append("\nparse health " + json.dumps(out["parse_health"]))
    txt = disp("\n".join(L))
    (TD / OUT_TXT).write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
