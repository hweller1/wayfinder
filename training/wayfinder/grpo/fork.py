"""Forked-report mode (optional, --fork-report K; default 0 = everything here is inert). Design and cost: docs/grpo_fork_report.md.

After a search trajectory reaches its report turn, K more report completions are sampled from the SAME prefix (the exact request body of the final call, n=K,
same adapter, same temperature/top-p, behaviour logprobs from vLLM like any rollout). Each is scored with rank5b. The K+1 reports (the original and the forks)
form an intra-prefix group (Tree-GRPO's intra-tree term): their report tokens get the group-relative advantage, with the same std floor, A_CLIP and
correctness-consistent clipping as rank5b. The search tokens of the original keep the normal inter-trajectory advantage (the inter-tree term).

Loss tokens:  original report span = inter adv + lambda_fork * intra adv_0;  forked completion j = lambda_fork * intra adv_j (its own sequence: prefix + completion).
A prefix whose K+1 rewards have zero variance contributes no intra term (and no fork sequences are trained).
Pure Python + numpy; httpx only inside `sample_forks`.
"""

from __future__ import annotations

import json
import math
from typing import Any

from wayfinder.grpo.advantage import A_CLIP, MIN_STD, STD_FLOOR, group_advantages
from wayfinder.grpo.monitor import N_RISE, PREC_FALL
from wayfinder.grpo.reward import compute_reward_rank5b, rank5b_tiebreak, reward_cfg, total_reward
from wayfinder.opd.spans import ASSISTANT, EOM, EOT, START

PART_KEYS = ("reward", "recall_at_5_capped", "recall_at_8", "n_reported", "n_gold5", "n_gold8", "valid_report", "turns", "all_gold_window", "precision",
             "recall_at_5", "recall_reported", "all_gold", "no_report_penalty")
MAX_RESULTS = 12  # the harness's ExploreRequest.max_results


def compact_parts(parts: dict[str, Any]) -> dict[str, Any]:
    """The reward fields the intra group, the tie-break and the padding guard need (JSON-safe)."""
    return {k: parts.get(k) for k in PART_KEYS}


# ---------------------------------------------------------------------------------------------------------------------------------------------
# sampling (rollout worker side)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def make_recording_transport() -> Any:
    """httpx transport that remembers the JSON body of the last request it carried (the report turn's exact request, reasoning and tools included). CompatChat
    opens a fresh AsyncClient per call and a client closes its transport on exit, so the wrapper ignores that and the owner calls `close()` once. httpx is
    imported here so the driver (no httpx in its image) can import this module."""
    import httpx

    class Recording(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.inner = httpx.AsyncHTTPTransport()
            self.last_body: dict[str, Any] | None = None

        async def handle_async_request(self, request: Any) -> Any:
            try:
                self.last_body = json.loads(request.content)
            except (ValueError, TypeError):
                pass
            return await self.inner.handle_async_request(request)

        async def aclose(self) -> None:  # per-call client exit: keep the pool open
            return None

        async def close(self) -> None:
            await self.inner.aclose()

    return Recording()


def report_args(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Arguments of the first `report` tool call of a chat completion message, or None (no report call, unparseable arguments)."""
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        if fn.get("name") != "report":
            continue
        a = fn.get("arguments")
        if isinstance(a, dict):
            return a
        try:
            parsed = json.loads(a) if isinstance(a, str) and a.strip() else None
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


async def sample_forks(url: str, headers: dict[str, str], body: dict[str, Any], k: int, *, timeout: float = 1800.0, transport: Any = None) -> dict[str, Any]:
    """K completions of the recorded final request in ONE call (n=K; vLLM prefills the prefix once and its prefix cache is warm from the rollout itself).
    The seed is dropped (n samples with one seed would be correlated). Returns {prompt_token_ids, samples: [{token_ids, logprobs, finish, tool_calls}], usage}."""
    import httpx

    b = {**body, "n": int(k)}
    b.pop("seed", None)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        res = await client.post(url, headers=headers, json=b)
    res.raise_for_status()
    data = res.json()
    samples = []
    for ch in data["choices"]:
        lp = (ch.get("logprobs") or {}).get("content")
        samples.append({"token_ids": list(ch.get("token_ids") or []), "logprobs": [float(x["logprob"]) for x in lp] if lp else [], "finish": ch.get("finish_reason"),
                        "tool_calls": (ch.get("message") or {}).get("tool_calls") or []})
    return {"prompt_token_ids": list(data.get("prompt_token_ids") or []), "samples": samples, "usage": data.get("usage")}


def fork_chain(prompt_ids: list[int], sample: dict[str, Any]) -> dict[str, Any] | None:
    """Trainable chain {input_ids, assistant_spans, logprobs} = prefix + completion, one loss span, built exactly as opd.spans.build_chains does for a call:
    the server stops on <|eot|> and keeps it out of token_ids (appended here with a NaN logprob); a completion cut by max_tokens has no terminator and cannot be
    trained as a full segment (None). A logprob list that is not aligned with the sampled ids becomes all-NaN (treated as on-policy)."""
    C = list(sample.get("token_ids") or [])
    if len(prompt_ids) < 3 or prompt_ids[-2:] != [START, ASSISTANT] or not C:
        return None
    n_samp = len(C)
    lp = list(sample.get("logprobs") or [])
    lp = lp if len(lp) == n_samp else [math.nan] * n_samp
    if C[-1] not in (EOM, EOT):
        if sample.get("finish") != "stop":
            return None
        C.append(EOT)
        lp.append(math.nan)
    n = len(prompt_ids)
    return {"input_ids": list(prompt_ids) + C, "assistant_spans": [[n, n + len(C)]], "logprobs": [math.nan] * n + lp}


def score_fork_report(paths: list[str] | None, task: dict[str, Any], row: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """rank5b base reward of one forked report. `paths` = the validated reported files in report order (scout's path guard applied by the caller; at most
    MAX_RESULTS), or None when the completion is not a valid `report` call (no report: the no-report penalty applies, as for a rollout that never reports).
    Turns and messages are the original's (the search is shared)."""
    res = [{"path": p} for p in (paths or [])[:MAX_RESULTS]]
    r = {"gold_files": task["gold_files"], "turns": row.get("turns"), "error": None, "messages": row.get("messages") or [],
         "explore_result": {"results": res, "budget_exhausted": False}}
    return compact_parts(compute_reward_rank5b(r, reward_cfg(cfg)))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# advantages (driver side)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def intra_advantages(parts: list[dict[str, Any]], precision_coef: float = 0.0, min_std: float = MIN_STD, clip: float = A_CLIP, std_floor: float = STD_FLOOR) -> dict[str, Any]:
    """Intra-prefix group advantages of the K+1 reports (parts[0] = the original). Reward = rank5b base (+ precision term) + the rank5b tie-break computed
    WITHIN the group (turns are shared, so it ranks by report length inside a recall class: fewer files first, the padding guard). Then exactly the
    outcome advantage of advantage.group_advantages (std floor, A_CLIP, zero-variance drop) with no rubric term and the rank5b correctness-consistent clip:
    an all-gold-in-top-8 report is never negative, a zero-recall report never positive. {"keep", "adv", "std", "rewards"}; adv is all zeros when dropped."""
    rows = [dict(p) for p in parts]
    for r in rows:
        r["arm_reward"] = total_reward(r, precision_coef)
    tb = rank5b_tiebreak(rows)
    rewards = [round(r["arm_reward"] + b, 6) for r, b in zip(rows, tb)]
    n = len(rows)
    out = group_advantages(rewards, [False] * n, [None] * n, lam=0.0, min_std=min_std, clip=clip, std_floor=std_floor,
                           all_gold8=[bool(r.get("all_gold_window")) for r in rows], zero_recall=[(r.get("recall_at_5_capped") or 0) == 0 and (r.get("recall_at_8") or 0) == 0 for r in rows])
    return {"keep": out["keep"], "adv": out["adv"], "std": out["std"], "rewards": rewards}


def pick_train(k: int, m: int, rid: str) -> list[int]:
    """Which M of the K forked completions are trained (uniform, seeded by the rollout id so it is reproducible). The others only enter the baseline."""
    import random

    return sorted(random.Random(rid).sample(range(k), min(max(m, 0), k)))


def apply_forks(rows: list[dict[str, Any]], items: list[dict[str, Any]], precision_coef: float, lam: float) -> dict[str, Any]:
    """Mutates the manifest `items` (list of {rid, path, n_tokens, n_loss, adv}). For every row with a fork record row["fork"] = {parts (K scored forks), ref [chain, span]
    of the original report span, train [{j, path, n_tokens, n_loss}], scale}: the intra group is [original] + forks.
      - the original's item (present only when its inter group was kept) gets item["fork"] = {chain, span, adv: lam * intra adv_0}: that span's advantage becomes
        inter + lam * intra (trainer side, step.py). A dropped inter group does not train the original (a full-trajectory forward for ~2k tokens); its report still
        counts in the intra baseline.
      - each trained forked completion becomes its own item with adv = lam * scale * intra adv_j (scale 1: the M trained forks are a subsample whose total weight
        is M, not K, which keeps per-token advantages bounded by A_CLIP); zero advantages are not trained.
    A prefix whose K+1 rewards have zero variance contributes nothing. Returns counters."""
    by_rid = {it["rid"]: it for it in items}
    st = {"n_prefixes": 0, "kept": 0, "zero_var": 0, "fork_items": 0, "zero_adv": 0, "orig_untrained": 0, "orig_no_ref": 0}
    stds: list[float] = []
    for r in rows:
        f = r.get("fork")
        if not f or not f.get("parts"):
            continue
        st["n_prefixes"] += 1
        g = intra_advantages([compact_parts(r)] + list(f["parts"]), precision_coef)
        if not g["keep"]:
            st["zero_var"] += 1
            continue
        st["kept"] += 1
        stds.append(g["std"])
        it = by_rid.get(r["rid"])
        if it is None:
            st["orig_untrained"] += 1
        elif f.get("ref") is None:
            st["orig_no_ref"] += 1
        else:
            it["fork"] = {"chain": int(f["ref"][0]), "span": int(f["ref"][1]), "adv": round(lam * g["adv"][0], 6)}
        for t in f.get("train") or []:
            a = lam * float(f.get("scale") or 1.0) * g["adv"][1 + int(t["j"])]
            if a == 0.0:
                st["zero_adv"] += 1
                continue
            items.append({"rid": f"{r['rid']}-f{t['j']}", "path": t["path"], "n_tokens": t["n_tokens"], "n_loss": t["n_loss"], "adv": round(a, 6)})
            st["fork_items"] += 1
    st["intra_std_mean"] = round(sum(stds) / len(stds), 4) if stds else None
    return st


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def fork_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-step padding guard for forked reports: the forks of each prefix against its original report (same prefix, so the search and the task mix cancel).
    Same rule as monitor.py: flagged when forks report more files (+N_RISE) AND are less precise (-PREC_FALL) AND recall@5 is not higher. Also the share of forks that are
    valid reports and the n_reported histogram of the forks."""
    fr = [r for r in rows if r.get("fork") and r["fork"].get("parts")]
    if not fr:
        return {"n_prefixes": 0}
    fk = [p for r in fr for p in r["fork"]["parts"]]
    d = {"n_prefixes": len(fr), "n_forks": len(fk),
         "n_reported": {"orig": _mean([float(r["n_reported"]) for r in fr]), "fork": _mean([float(p["n_reported"]) for p in fk])},
         "precision": {"orig": _mean([r.get("precision") for r in fr]), "fork": _mean([p.get("precision") for p in fk])},
         "recall_at_5": {"orig": _mean([r.get("recall_at_5") for r in fr]), "fork": _mean([p.get("recall_at_5") for p in fk])},
         "valid_report": {"orig": _mean([float(bool(r.get("valid_report"))) for r in fr]), "fork": _mean([float(bool(p.get("valid_report"))) for p in fk])},
         "reward": {"orig": _mean([r.get("reward") for r in fr]), "fork": _mean([p.get("reward") for p in fk])}}
    hist: dict[str, int] = {}
    for p in fk:
        hist[str(int(p["n_reported"]))] = hist.get(str(int(p["n_reported"])), 0) + 1
    d["fork_n_reported_hist"] = dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
    delta = {m: (round(d[m]["fork"] - d[m]["orig"], 4) if d[m]["fork"] is not None and d[m]["orig"] is not None else None) for m in ("n_reported", "precision", "recall_at_5")}
    d["delta"] = delta
    d["flagged_padding"] = bool(delta["n_reported"] is not None and delta["precision"] is not None and delta["recall_at_5"] is not None
                                and delta["n_reported"] > N_RISE and delta["precision"] < -PREC_FALL and delta["recall_at_5"] <= 0.0)
    return d
