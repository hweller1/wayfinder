"""The five contract operations as data, shared by the CLI, MCP and daemon surfaces.

Not a surface: it imports only contract types, so it keeps the surfaces independent of each other
(import-linter contract 5) while guaranteeing they parse requests, serialise results and render plain text
the same way.
`result_json` is THE wire form (compact JSON, aliases on); conformance compares it across every transport.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import (
    ExploreRequest,
    ExploreResult,
    GrepRequest,
    GrepResult,
    IndexStatus,
    IndexStatusRequest,
    ReadRangeRequest,
    ReadRangeResult,
    RequestModel,
    SemanticSearchRequest,
    SemanticSearchResult,
)

# op name (also the `ScoutApp` method name) -> request model
REQUESTS: dict[str, type[RequestModel]] = {
    "explore": ExploreRequest,
    "semantic_search": SemanticSearchRequest,
    "grep": GrepRequest,
    "read_range": ReadRangeRequest,
    "index_status": IndexStatusRequest,
}
OPS: tuple[str, ...] = tuple(REQUESTS)


def parse_request(op: str, payload: Mapping[str, Any] | None) -> Any:
    """Validate `payload` as the request of `op`; a failure is `INVALID_ARGUMENT` naming fields."""
    model = REQUESTS.get(op)
    if model is None:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"unknown operation {op!r}")
    try:
        return model.model_validate(dict(payload or {}))
    except ValidationError as e:
        parts = [f"{'.'.join(str(p) for p in err['loc']) or 'request'}: {err['msg']}" for err in e.errors()]
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "invalid request: " + "; ".join(parts)) from None


def result_json(result: BaseModel) -> str:
    return result.model_dump_json(by_alias=True)


def error_hint(err: ScoutError) -> str:
    if err.code in (ErrorCode.LOGIN_REQUIRED, ErrorCode.UNAUTHENTICATED) and "scout login" not in err.message:
        return " (run `scout login`)"
    return ""


def _status_line(s: IndexStatus) -> str:
    bits = [f"index {s.state}", f"coverage={s.coverage:.2f}", f"files={s.files}"]
    bits.append("semantic=on" if s.semantic_enabled else "semantic=off")
    if s.syncing:
        bits.append("syncing")
    if s.commit:
        bits.append(f"commit={s.commit[:12]}")
    if s.last_sync_at:
        bits.append(f"last_sync={s.last_sync_at}")
    return " ".join(bits)


def render_primitive(op: str, result: BaseModel) -> str:
    """Plain text of a primitive result (the non-structured MCP content and `--text` of the CLI)."""
    if isinstance(result, IndexStatus):
        return _status_line(result)
    if isinstance(result, GrepResult):
        lines = [f"{f.path}:{m.line}: {m.text}" for f in result.files for m in f.matches]
        if result.truncated:
            lines.append("... truncated: narrow the pattern or path")
        return "\n".join(lines) if lines else "no matches"
    if isinstance(result, ReadRangeResult):
        head = f"{result.path}:{result.start_line}-{result.end_line}" + (
            " (truncated)" if result.truncated else ""
        )
        return "\n".join([head, *(f"{result.start_line + i}\t{t}" for i, t in enumerate(result.lines))])
    if isinstance(result, SemanticSearchResult):
        out: list[str] = []
        for h in result.hits:
            out.append(f"{h.path}:{h.start_line}-{h.end_line}  {' '.join(h.why.split())}")
            out.extend(f"    {ln}" for ln in h.snippet.splitlines())
        if result.note:
            out.append(f"note: {result.note}")
        return "\n".join(out) if out else "no hits"
    raise TypeError(f"no plain-text form for {op}")


def render(op: str, req: Any, result: BaseModel, *, ceiling: int | None = None) -> str:
    """Plain text of any op's result. `explore` is `render_text` under `min(requested, ceiling)` tokens."""
    if isinstance(result, ExploreResult):
        from scout.core.explore.render import render_text

        cap = req.render_max_tokens if ceiling is None else min(req.render_max_tokens, ceiling)
        return render_text(result, cap)
    return render_primitive(op, result)
