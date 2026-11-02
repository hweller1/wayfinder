"""One GRPO rollout: the OPD/eval harness (scout product profile, tools, convergence nudge, retain_reasoning on) with token and
behaviour-logprob capture, scored, rewarded and cut into trainable chains (wayfinder.opd.spans.build_chains, which now also carries
the logprobs). Nothing of the harness is modified.

A row that errored at the harness level (upstream timeout, unreachable server) is an infrastructure failure, not a policy outcome:
the caller drops it. A rollout that ends without a report (turn cap, context cap) is a policy outcome and carries the no-report penalty.
"""

from __future__ import annotations

import hashlib
from typing import Any

from wayfinder.grpo.reward import compute_reward, compute_reward_legacy, compute_reward_rank5, compute_reward_rank5b, reward_cfg
from wayfinder.grpo.fork import make_recording_transport
from wayfinder.opd.spans import build_chains

TEMPERATURE = 0.85
TOP_P = 0.95


def rollout_id(run: str, step: int, task_id: str, seed: int) -> str:
    return hashlib.sha1(f"{run}|{step}|{task_id}|{seed}".encode()).hexdigest()[:16]


def lift_budget_cap(max_seconds: int) -> None:
    """scout's Budget caps max_seconds at 900; lift it at runtime exactly as wayfinder/rollout/sft_eval_app.py does (scout untouched)."""
    if max_seconds <= 900:
        return
    from pydantic import Field
    from pydantic.fields import FieldInfo
    from scout.core.contracts.types import Budget, ExploreRequest

    Budget.model_fields["max_seconds"] = FieldInfo.from_annotated_attribute(int, Field(default=120, ge=5, le=100000))
    Budget.model_rebuild(force=True)
    ExploreRequest.model_rebuild(force=True)


async def grpo_rollout(task: dict[str, Any], base_url: str, model: str, root: str, *, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    from wayfinder.rollout.chat import CompatChat, Prices, Sampling
    from wayfinder.rollout.run import NUDGES, run_rollout
    from wayfinder.rollout.score import score_row

    fork_k = int(cfg.get("fork_report", 0) or 0)
    rec = make_recording_transport() if fork_k > 0 else None  # forked-report mode only: remembers the report turn's exact request
    chat = CompatChat(base_url, "x", model, sampling=Sampling(cfg.get("temperature", TEMPERATURE), cfg.get("top_p", TOP_P)),
                      prices=Prices(1.25, 4.25, 0.15), seed=seed, vllm_extensions=False, timeout=1800.0, max_retries=4,
                      extra_body={"return_token_ids": True, "logprobs": True}, **({"transport": rec} if rec is not None else {}))
    calls: list[dict[str, Any]] = []
    chat.capture = calls.append
    chat.retain_reasoning = True
    row = await run_rollout(task, chat, root, seed=seed, max_turns=cfg.get("max_turns", 40), max_seconds=cfg.get("max_seconds", 3600),
                            nudge=NUDGES[cfg.get("nudge", "teacher")], semantic=cfg.get("semantic", "auto"))
    row.update(score_row(row))
    kind = cfg.get("reward_kind", "legacy")  # needs the transcript
    parts = (compute_reward_legacy(row) if kind == "legacy" else compute_reward_rank5(row, reward_cfg(cfg.get("reward"))) if kind == "rank5"
             else compute_reward_rank5b(row, reward_cfg(cfg.get("reward"))) if kind == "rank5b" else compute_reward(row, reward_cfg(cfg.get("reward"))))
    chains, diag = build_chains(calls, max_len=cfg.get("max_chain_tokens", 49152), strict=False)
    er = row.get("explore_result") or {}
    import math

    n_loss = sum(b - a for c in chains for a, b in c["assistant_spans"])
    n_lp = sum(1 for c in chains for a, b in c["assistant_spans"] for x in c.get("logprobs", [])[a:b] if not math.isnan(x))
    summary = {
        "lp_frac": round(n_lp / n_loss, 4) if n_loss else None,  # share of loss tokens that carry a behaviour logprob (rest are treated as on-policy)
        "task_id": task["task_id"], "repo": task["repo"], "seed": seed, "model": model, "category": task.get("category"),
        **parts, "rubric": (row.get("rubric") or {}).get("score"), "recall_touched": row.get("recall_seen_k5"),
        "all_gold_touched": row.get("all_gold_found"), "semantic": row.get("semantic"), "error": row.get("error"),
        "budget_exhausted": bool(er.get("budget_exhausted")), "wall_s": row.get("wall_s"), "model_calls": row.get("model_calls"),
        "tokens_in": (row.get("usage") or {}).get("in"), "tokens_out": (row.get("usage") or {}).get("out"),
        "report_chars": sum(len(h.get("why") or "") for h in er.get("results") or []) + len(er.get("summary") or ""),
        "n_chains": diag["n_chains"], "chain_breaks": diag["chain_breaks"], "chain_tokens": diag["chain_tokens"],
        "assistant_tokens": diag["assistant_tokens"], "sampled_tokens": diag["sampled_tokens"], "lenient_mismatch": diag["lenient_mismatch"],
        "no_terminator": diag["no_terminator"],
    }
    out = {"summary": summary, "chains": chains}
    if fork_k > 0:
        try:
            out["fork"] = await fork_report(task, row, calls, chains, rec, chat, root, fork_k, cfg)
        except Exception as e:  # noqa: BLE001  (a failed fork costs the extra signal, never the rollout)
            out["fork"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        finally:
            await rec.close()
    return out


def validate_report_paths(args: dict[str, Any], root: str, max_results: int = 12) -> list[str] | None:
    """Paths of a forked `report` call the way scout's loop judges a real one (ReportArgs schema, then path guard / clamp / drop via validated_hit), in report
    order; None when the arguments are not a valid report. (The original's extra `changed since seen` drop is not applied to forks: a fork cannot change files.)"""
    from pydantic import ValidationError
    from scout.core.explore.candidates import validated_hit
    from scout.core.explore.tools import ReportArgs
    from scout.core.local.ignore import Matcher

    from wayfinder.rollout.run import HarnessFileSystem

    try:
        rep = ReportArgs.model_validate(args)
    except ValidationError:
        return None
    fs = HarnessFileSystem(root)
    matcher = Matcher.from_fs(fs)
    out = []
    for h in rep.results:
        hit = validated_hit(fs, matcher, path=h.path, start_line=h.start_line, end_line=h.end_line, why=h.why, source="grep")
        if hit is not None:
            out.append(hit.path)
    return out[:max_results]


async def fork_report(task: dict[str, Any], row: dict[str, Any], calls: list[dict[str, Any]], chains: list[dict[str, Any]], rec: Any, chat: Any, root: str, k: int,
                      cfg: dict[str, Any]) -> dict[str, Any]:
    """Sample K more completions of the final (report) request, same adapter/temperature/top-p, score each with rank5b, build the trainable sequences of M of them.
    Returns {"parts": [K compact reward dicts], "ref": [chain, span] of the original report span or None, "train": [(j, chain)], "scale": K/M, "usage": ...}."""
    from wayfinder.grpo.fork import fork_chain, pick_train, report_args, sample_forks, score_fork_report

    er = row.get("explore_result") or {}
    if row.get("error") or er.get("budget_exhausted") or not er.get("results") or not calls or rec is None or rec.last_body is None:
        return {"skipped": "no valid report"}  # no report turn to fork from
    last = calls[-1]
    if not last.get("n_tool_calls") or not last.get("prompt_token_ids"):
        return {"skipped": "final call is not a tool call"}
    body = {**rec.last_body}
    if body.get("model") != chat.model or not body.get("tools"):
        return {"skipped": "unexpected final request"}
    url, headers = chat._url, chat._headers
    got = await sample_forks(url, headers, body, k)
    if got["prompt_token_ids"] != list(last["prompt_token_ids"]):
        return {"error": "fork prefix differs from the original's report prompt"}  # never train sequences that are not the same prefix
    parts, chains_f = [], []
    for smp in got["samples"]:
        args = report_args(smp["tool_calls"])
        paths = validate_report_paths(args, root) if args is not None else None
        paths = paths or None
        parts.append(score_fork_report(paths, task, row, cfg.get("reward")))
        chains_f.append(fork_chain(got["prompt_token_ids"], smp))
    m = int(cfg.get("fork_train", 2))  # trained forks per prefix (each costs a full-prefix trainer forward); all K enter the baseline
    train = [(j, chains_f[j]) for j in pick_train(len(chains_f), m, f"{task['task_id']}|{row.get('seed')}") if chains_f[j] is not None]
    ref = None
    if chains and chains[-1].get("call_end") == len(calls) - 1 and chains[-1]["assistant_spans"]:
        a, b = chains[-1]["assistant_spans"][-1]
        n_last = len(last["token_ids"])
        if b == len(chains[-1]["input_ids"]) and b - a in (n_last, n_last + 1):  # the last span of the last chain is exactly the report completion
            ref = [len(chains) - 1, len(chains[-1]["assistant_spans"]) - 1]
    return {"parts": parts, "ref": ref, "train": train, "scale": 1.0, "usage": got.get("usage"),
            "n_valid": sum(1 for p in parts if p["valid_report"])}
