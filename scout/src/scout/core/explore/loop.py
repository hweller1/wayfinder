"""The explore loop (FR-C4, AC-1.1, AC-1.4; design section 5.5). Host-agnostic: it talks only to the Gateway
(chat + index_status) and FileSystem ports, and time comes from the injected Clock.

Ends on a valid `report`, on budget exhaustion (turns, seconds or context) or after 3 consecutive malformed
tool calls; the last two return the best candidates seen so far, so the caller always gets a schema-valid
`ExploreResult`.
Each assistant turn keeps the model's reasoning (`reasoning_content`) in the history, so the model sees its
earlier thinking as it did in training; the context estimate counts it, and the chat adapter decides whether
it goes on the wire.
Before each model call the prompt is estimated; over `profile.context_tokens`, the oldest tool results are
shrunk to one-line stubs (never the system prompt, the brief, or the latest turn), and if it still does not
fit the loop stops as exhaustion `context`. An upstream context-length error maps to the same stop. Other chat
errors (UPSTREAM_*, RATE_LIMITED) propagate: there is no partial result (design section 15).
"""

import json
from dataclasses import dataclass, field
from typing import Any

from scout.core.contracts.errors import ScoutError
from scout.core.contracts.types import ExploreRequest, ExploreResult, Hit, IndexStatus, ModelTokens, Usage
from scout.core.contracts.wire import StatusReq
from scout.core.explore.backfill import backfill
from scout.core.explore.budget import Budget
from scout.core.explore.candidates import validated_hit
from scout.core.explore.profiles import PROFILES, ExploreProfile
from scout.core.explore.render import approx_tokens
from scout.core.explore.spec import RetrievalSpec
from scout.core.explore.tools import ReportArgs, SemanticBackend, ToolRunner, tool_specs
from scout.core.ports.chat import ChatRequest, ToolCall
from scout.core.ports.clock import Clock
from scout.core.ports.fs import FileSystem
from scout.core.ports.gateway import Gateway

MAX_CONSECUTIVE_MALFORMED = 3
SEMANTIC_MIN_COVERAGE = 0.8  # below this the loop is grep-only and says so (AC-3.8)
SUMMARY_MAX = 1200
DEFAULT_BACKFILL_TO = 12
_ELIDED_PREFIX = "[elided: "
_NUDGE = "Reply with a tool call. When you have enough evidence, call report."


@dataclass(slots=True)
class ExploreContext:
    gateway: Gateway
    fs: FileSystem
    clock: Clock
    status_req: StatusReq
    profile: ExploreProfile = field(default_factory=lambda: PROFILES["product"])
    semantic: SemanticBackend | None = None  # offered only when the index is >= 80% covered
    backfill_to: int = (
        DEFAULT_BACKFILL_TO  # config `explore.backfill_to`: pad the result to this many files; 0 is off
    )
    spec: RetrievalSpec | None = (
        None  # the resolved retrieval spec (instructions, index description, examples)
    )


def _system_prompt(ctx: ExploreContext) -> str:
    extra = ctx.spec.addendum() if ctx.spec is not None else ""
    return f"{ctx.profile.prompt}\n\n## Retrieval spec\n\n{extra}" if extra else ctx.profile.prompt


def _user_message(req: ExploreRequest) -> str:
    parts = [req.brief]
    if req.paths:
        parts.append("Scope: only look at paths matching " + ", ".join(req.paths))
    parts.append(f"Report at most {req.max_results} results.")
    return "\n\n".join(parts)


def _assistant_message(
    content: str | None, calls: tuple[ToolCall, ...], reasoning: str | None = None
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:  # kept in the history (and in the context estimate) so the model sees its earlier thinking
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in calls
        ]
    return msg


def _prompt_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(approx_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)


def _fit_context(messages: list[dict[str, Any]], limit: int) -> bool:
    """Shrink the oldest tool results to stubs until the prompt fits; True if it fits. Only `tool` messages
    before the last assistant message are eligible, so the system prompt, the brief and the latest turn's
    results are never touched."""
    if _prompt_tokens(messages) <= limit:
        return True
    last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    tool_names = {c["id"]: c["function"]["name"] for m in messages for c in m.get("tool_calls", ())}
    for i in range(last_assistant):
        m = messages[i]
        if m["role"] != "tool" or str(m["content"]).startswith(_ELIDED_PREFIX):
            continue
        name = tool_names.get(m.get("tool_call_id"), "tool")
        m["content"] = f"{_ELIDED_PREFIX}{len(str(m['content']).splitlines())} lines from {name}]"
        if _prompt_tokens(messages) <= limit:
            return True
    return False


def _is_context_error(e: ScoutError) -> bool:
    """vLLM answers an oversize prompt with a 400 'maximum context length'; adapters may also set
    `details["reason"] == "context_length"`."""
    return e.details.get("reason") == "context_length" or "maximum context length" in e.message.lower()


def _grep_only_gap(status: IndexStatus) -> str:
    return f"grep-only search (index {status.state}): semantic search was not used"


async def explore(req: ExploreRequest, ctx: ExploreContext) -> ExploreResult:
    started = ctx.clock.monotonic()
    status = await ctx.gateway.index_status(ctx.status_req)
    semantic_on = (
        ctx.semantic is not None
        and status.coverage >= SEMANTIC_MIN_COVERAGE
        and not (ctx.spec is not None and not ctx.spec.index_enabled)
    )
    runner = ToolRunner(ctx.fs, clock=ctx.clock, semantic=ctx.semantic if semantic_on else None)
    budget = Budget.from_limits(ctx.clock, req.budget)
    if (
        ctx.spec is not None and ctx.spec.max_turns
    ):  # `[explore] max_turns` in the retrieval spec: a lower turn cap
        budget.max_turns = min(budget.max_turns, ctx.spec.max_turns)
    tools = tool_specs(semantic_enabled=semantic_on)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(ctx)},
        {"role": "user", "content": _user_message(req)},
    ]
    tokens_in = tokens_out = 0
    strikes = 0
    report: ReportArgs | None = None
    stop_gap: str | None = None
    exhausted = False

    while report is None and stop_gap is None:
        if budget.exhausted:
            exhausted = True
            stop_gap = f"budget exhausted ({budget.reason}) before the explorer reported"
            break
        if not _fit_context(messages, ctx.profile.context_tokens):
            budget.mark_context_exhausted()
            exhausted = True
            stop_gap = f"budget exhausted ({budget.reason}) before the explorer reported"
            break
        try:
            turn = await ctx.gateway.complete(
                ChatRequest(messages=messages, tools=tools, tool_choice="auto", thinking=ctx.profile.thinking)
            )
        except ScoutError as e:
            if not _is_context_error(e):
                raise
            budget.mark_context_exhausted()
            exhausted = True
            stop_gap = f"budget exhausted ({budget.reason}) before the explorer reported"
            break
        budget.tick()
        tokens_in += turn.tokens_in
        tokens_out += turn.tokens_out
        messages.append(_assistant_message(turn.content, turn.tool_calls, turn.reasoning))
        if not turn.tool_calls:
            strikes += 1
            messages.append({"role": "user", "content": _NUDGE})
        for call in turn.tool_calls:
            outcome = await runner.run(call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.text})
            strikes = strikes + 1 if outcome.kind == "malformed" else 0
            if outcome.kind == "report":
                report = outcome.report
                break
            if strikes >= MAX_CONSECUTIVE_MALFORMED:
                break
        if report is None and strikes >= MAX_CONSECUTIVE_MALFORMED:
            stop_gap = f"{strikes} consecutive malformed tool calls; stopped early"

    gaps: list[str] = []
    hits: list[Hit]
    if report is not None:
        hits, summary, gaps = _from_report(req, ctx, runner, report)
    else:
        hits = _best_hits(req, ctx, runner)
        summary = "The explorer did not finish; these are the best candidates found so far."
        gaps.append(stop_gap or "stopped before the explorer reported")
        if not hits:
            gaps.append("no candidate locations were found")
    if not semantic_on:
        gaps.append(
            "grep-only search (the retrieval spec turned the index off)"
            if ctx.spec is not None and not ctx.spec.index_enabled
            else _grep_only_gap(status)
        )
    hits = await backfill(
        hits,
        target=min(ctx.backfill_to, req.max_results),
        brief=req.brief,
        fs=ctx.fs,
        matcher=runner.matcher,
        opened=runner.opened,
        changed=runner.changed_since_seen,
        semantic=ctx.semantic if semantic_on else None,
        clock=ctx.clock,
    )
    return ExploreResult(
        results=hits,
        summary=summary[:SUMMARY_MAX],
        gaps=gaps,
        index=status,
        usage=Usage(
            turns=budget.turns,
            tool_calls=dict(runner.calls),
            model_tokens=ModelTokens(**{"in": tokens_in, "out": tokens_out}),
            wall_ms=int((ctx.clock.monotonic() - started) * 1000),
        ),
        budget_exhausted=exhausted,
    )


def _from_report(
    req: ExploreRequest, ctx: ExploreContext, runner: ToolRunner, report: ReportArgs
) -> tuple[list[Hit], str, list[str]]:
    """Every reported range is re-judged: path guard, clamp to the file, drop denylisted or missing."""
    hits: list[Hit] = []
    dropped = changed = 0
    for h in report.results:
        hit = validated_hit(
            ctx.fs,
            runner.matcher,
            path=h.path,
            start_line=h.start_line,
            end_line=h.end_line,
            why=h.why,
            source=runner.candidates.source_for(h.path, h.start_line, h.end_line),
        )
        if hit is None or (ctx.spec is not None and ctx.spec.filters.excluded(hit.path)):
            dropped += 1
        elif runner.changed_since_seen(hit.path):
            changed += 1
        else:
            hits.append(hit)
    gaps = [g for g in report.gaps if g.strip()]
    if dropped:
        gaps.append(f"{dropped} reported location(s) dropped: refused, missing, or outside the file")
    if changed:
        gaps.append(f"{changed} reported location(s) dropped: the file changed after it was searched or read")
    return hits[: req.max_results], report.summary, gaps


def _best_hits(req: ExploreRequest, ctx: ExploreContext, runner: ToolRunner) -> list[Hit]:
    hits: list[Hit] = []
    for c in runner.candidates.best(req.max_results * 2):  # headroom: files may vanish between read and now
        hit = validated_hit(
            ctx.fs,
            runner.matcher,
            path=c.path,
            start_line=c.start_line,
            end_line=c.end_line,
            why=c.why,
            source=c.source,
            score=c.score,
        )
        if hit is not None and not runner.changed_since_seen(hit.path):
            hits.append(hit)
    return hits[: req.max_results]
