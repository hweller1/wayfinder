"""Contract types and error model (design section 4). Covers AC-1.1, AC-1.3, AC-1.6, FR-C1..FR-C3."""

import json

import pytest
from pydantic import ValidationError

from scout.core.contracts.errors import (
    ErrorBody,
    ErrorCode,
    ScoutError,
    cli_exit,
    http_status,
    is_retryable,
)
from scout.core.contracts.types import (
    Budget,
    ExploreRequest,
    ExploreResult,
    GrepMatch,
    GrepRequest,
    GrepResult,
    Hit,
    IndexStatus,
    IndexStatusRequest,
    ReadRangeRequest,
    ReadRangeResult,
    SemanticSearchRequest,
    SemanticSearchResult,
    SnippetHit,
    Usage,
)


def _index() -> IndexStatus:
    return IndexStatus(state="ready", coverage=0.97, commit="abc123", semantic_enabled=True)


def _usage() -> Usage:
    return Usage.model_validate(
        {
            "turns": 7,
            "tool_calls": {"semantic_search": 3, "grep": 4, "read_file": 5},
            "model_tokens": {"in": 41000, "out": 1900},
            "wall_ms": 21400,
            "cost_usd": 0.011,
        }
    )


def _result() -> ExploreResult:
    return ExploreResult(
        results=[
            Hit(
                path="src/auth/session.py",
                start_line=40,
                end_line=88,
                why="session refresh",
                source="semantic",
            )
        ],
        summary="Session refresh lives in session.py [0].",
        gaps=["no tests found"],
        index=_index(),
        usage=_usage(),
    )


# ---- ExploreRequest ----


@pytest.mark.parametrize(
    ("given", "expected"),
    [(1500, 1500), (4000, 4000), (4001, 4000), (10**9, 4000), (200, 200), (199, 200), (0, 200), (-5, 200)],
)
def test_render_max_tokens_clamps_never_rejects(given: int, expected: int) -> None:
    """AC-1.3: render_max_tokens is clamped to [200, 4000], never rejected."""
    assert ExploreRequest(brief="x", render_max_tokens=given).render_max_tokens == expected


def test_explore_request_defaults_match_prd() -> None:
    """AC-1.6: no budget given -> max_turns 40, max_seconds 120; PRD 4.1 defaults."""
    r = ExploreRequest(brief="where is auth")
    assert (r.budget.max_turns, r.budget.max_seconds) == (40, 120)
    assert r.max_results == 12
    assert r.render_max_tokens == 1500
    assert r.paths == []
    assert r.root is None
    assert Budget() == r.budget


def test_budget_default_not_shared_between_requests() -> None:
    a, b = ExploreRequest(brief="a"), ExploreRequest(brief="b")
    a.paths.append("src/**")
    assert b.paths == []
    assert a.budget is not b.budget


@pytest.mark.parametrize(
    "kwargs",
    [
        {"brief": ""},
        {"brief": "x" * 100_001},
        {"brief": "x", "max_results": 0},
        {"brief": "x", "max_results": 41},
        {"brief": "x", "budget": {"max_turns": 0}},
        {"brief": "x", "budget": {"max_turns": 201}},
        {"brief": "x", "budget": {"max_seconds": 4}},
        {"brief": "x", "budget": {"max_seconds": 901}},
    ],
)
def test_explore_request_rejects_out_of_range(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ExploreRequest.model_validate(kwargs)


@pytest.mark.parametrize(
    "model",
    [ExploreRequest, SemanticSearchRequest, GrepRequest, ReadRangeRequest, IndexStatusRequest, Budget],
)
def test_requests_forbid_unknown_fields(model: type) -> None:
    """FR-C2: requests are extra=forbid."""
    payload = {
        ExploreRequest: {"brief": "x"},
        SemanticSearchRequest: {"query": "q"},
        GrepRequest: {"pattern": "p"},
        ReadRangeRequest: {"path": "a.py"},
        IndexStatusRequest: {},
        Budget: {},
    }[model]
    model.model_validate(payload)
    with pytest.raises(ValidationError):
        model.model_validate({**payload, "surprise": 1})


def test_primitive_request_bounds() -> None:
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="q", k=0)
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="q", k=31)
    assert SemanticSearchRequest(query="q").k == 10
    with pytest.raises(ValidationError):
        GrepRequest(pattern="p", context=6)
    assert GrepRequest(pattern="p").regex is True
    with pytest.raises(ValidationError):
        ReadRangeRequest(path="a", start_line=0)
    assert ReadRangeRequest(path="a").start_line == 1


# ---- Hit ----


def test_hit_end_line_must_not_precede_start_line() -> None:
    with pytest.raises(ValidationError, match="end_line"):
        Hit(path="a.py", start_line=10, end_line=9, why="w", source="grep")
    assert Hit(path="a.py", start_line=10, end_line=10, why="w", source="grep").end_line == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_line": 0, "end_line": 1},
        {"start_line": 1, "end_line": 1, "source": "vector"},
        {"start_line": 1, "end_line": 1, "why": "x" * 301},
    ],
)
def test_hit_rejects_invalid_fields(kwargs: dict) -> None:
    base = {"path": "a.py", "start_line": 1, "end_line": 1, "why": "w", "source": "read"}
    with pytest.raises(ValidationError):
        Hit.model_validate({**base, **kwargs})


def test_hit_defaults_and_snippet_hit_inherits_validation() -> None:
    h = Hit(path="a.py", start_line=1, end_line=2, why="w", source="semantic")
    assert h.stale is False
    assert h.score is None
    with pytest.raises(ValidationError):
        SnippetHit(path="a.py", start_line=5, end_line=1, why="w", source="semantic", snippet="s")


# ---- Usage alias ----


def test_usage_in_alias_parses_and_serializes_as_in() -> None:
    """PRD 4.1: model_tokens is {"in": .., "out": ..}."""
    u = _usage()
    assert u.model_tokens.in_ == 41000
    dumped = u.model_dump(mode="json")
    assert dumped["model_tokens"] == {"in": 41000, "out": 1900}
    assert "in_" not in dumped["model_tokens"]
    assert json.loads(u.model_dump_json())["model_tokens"] == {"in": 41000, "out": 1900}


def test_usage_cost_usd_optional() -> None:
    u = Usage.model_validate(
        {"turns": 1, "tool_calls": {}, "model_tokens": {"in": 1, "out": 1}, "wall_ms": 5}
    )
    assert u.cost_usd is None


# ---- ExploreResult serialization ----


def test_explore_result_serializes_exactly_prd_fields() -> None:
    """AC-1.1: schema_version present; field set matches PRD 4.1 (+ budget_exhausted per design)."""
    d = json.loads(_result().model_dump_json())
    assert d["schema_version"] == "1.0"
    assert set(d) == {"schema_version", "results", "summary", "gaps", "index", "usage", "budget_exhausted"}
    assert set(d["results"][0]) == {
        "path",
        "start_line",
        "end_line",
        "why",
        "source",
        "stale",
        "score",
        "verified",
    }
    assert set(d["index"]) >= {"state", "coverage", "commit"}
    assert set(d["usage"]) == {"turns", "tool_calls", "model_tokens", "wall_ms", "cost_usd"}
    assert d["budget_exhausted"] is False


def test_explore_result_json_round_trip() -> None:
    r = _result()
    again = ExploreResult.model_validate_json(r.model_dump_json())
    assert again == r
    assert again.usage.model_tokens.in_ == 41000


def test_responses_ignore_unknown_fields_for_additive_minor_versions() -> None:
    """FR-C2: responses are extra=ignore so a newer server's additive fields do not break old clients."""
    d = json.loads(_result().model_dump_json())
    d["new_in_1_1"] = {"x": 1}
    d["index"]["new_field"] = True
    d["results"][0]["rank"] = 3
    d["usage"]["model_tokens"]["cached"] = 9
    assert ExploreResult.model_validate(d) == _result()


@pytest.mark.parametrize("version", ["1.0", "1.7", "1.12"])
def test_schema_version_accepts_v1_minors(version: str) -> None:
    d = json.loads(_result().model_dump_json())
    d["schema_version"] = version
    assert ExploreResult.model_validate(d).schema_version == version


@pytest.mark.parametrize("version", ["2.0", "1", "1.", "1.x", "v1.0", "0.9", " 1.0"])
def test_schema_version_rejects_other_majors_and_junk(version: str) -> None:
    d = json.loads(_result().model_dump_json())
    d["schema_version"] = version
    with pytest.raises(ValidationError):
        ExploreResult.model_validate(d)


def test_every_response_type_carries_schema_version_default() -> None:
    assert _index().schema_version == "1.0"
    assert SemanticSearchResult(hits=[], index=_index()).schema_version == "1.0"
    assert GrepResult(files=[], truncated=False).schema_version == "1.0"
    rr = ReadRangeResult(path="a", start_line=1, end_line=1, lines=["x"], truncated=False)
    assert rr.schema_version == "1.0"


def test_summary_and_index_bounds() -> None:
    d = json.loads(_result().model_dump_json())
    d["summary"] = "x" * 1201
    with pytest.raises(ValidationError):
        ExploreResult.model_validate(d)
    with pytest.raises(ValidationError):
        IndexStatus(state="ready", coverage=1.01, semantic_enabled=True)
    with pytest.raises(ValidationError):
        IndexStatus(state="stale", coverage=0.5, semantic_enabled=False)  # type: ignore[arg-type]


def test_semantic_result_round_trip_with_snippet_hits() -> None:
    r = SemanticSearchResult(
        hits=[
            SnippetHit(path="a.py", start_line=1, end_line=3, why="w", source="semantic", snippet="1 x\n2 y")
        ],
        index=_index(),
        note="partial: grep-only",
    )
    assert SemanticSearchResult.model_validate_json(r.model_dump_json()) == r


def test_grep_match_text_capped_at_300_chars() -> None:
    assert len(GrepMatch(line=1, text="y" * 1000).text) == 300


def test_read_range_result_caps_lines_at_400() -> None:
    ok = ["l"] * 400
    ReadRangeResult(path="a", start_line=1, end_line=400, lines=ok, truncated=False)
    with pytest.raises(ValidationError):
        ReadRangeResult(path="a", start_line=1, end_line=401, lines=ok + ["l"], truncated=True)


# ---- Errors ----

# (code, http, cli_exit, retryable) transcribed from design section 4 error table
TABLE = [
    (ErrorCode.INVALID_ARGUMENT, 400, 2, False),
    (ErrorCode.UNSUPPORTED_SCHEMA, 422, 2, False),
    (ErrorCode.UNAUTHENTICATED, 401, 3, False),
    (ErrorCode.LOGIN_REQUIRED, 401, 3, False),
    (ErrorCode.PERMISSION_DENIED, 403, 3, False),
    (ErrorCode.NOT_FOUND, 404, 4, False),
    (ErrorCode.NOT_A_REPO, 400, 4, False),
    (ErrorCode.PATH_REFUSED, 403, 4, False),
    (ErrorCode.RATE_LIMITED, 429, 5, True),
    (ErrorCode.UPSTREAM_UNAVAILABLE, 503, 6, True),
    (ErrorCode.UPSTREAM_TIMEOUT, 504, 6, True),
    (ErrorCode.CONFLICT, 409, 7, True),
    (ErrorCode.INTERNAL, 500, 1, False),
]


@pytest.mark.parametrize(("code", "http", "exit_code", "retryable"), TABLE)
def test_error_table_rows(code: ErrorCode, http: int, exit_code: int, retryable: bool) -> None:
    assert http_status(code) == http
    assert cli_exit(code) == exit_code
    assert is_retryable(code) is retryable


def test_error_table_covers_every_code() -> None:
    assert {row[0] for row in TABLE} == set(ErrorCode)
    assert len(ErrorCode) == 13


def test_error_code_values_are_their_names() -> None:
    assert all(c.value == c.name for c in ErrorCode)
    assert ErrorCode("NOT_A_REPO") is ErrorCode.NOT_A_REPO


def test_scout_error_defaults_retryable_from_table_and_allows_override() -> None:
    assert ScoutError(ErrorCode.RATE_LIMITED, "slow down").retryable is True
    assert ScoutError(ErrorCode.NOT_FOUND, "nope").retryable is False
    assert ScoutError(ErrorCode.INTERNAL, "boom", retryable=True).retryable is True
    e = ScoutError(ErrorCode.CONFLICT, "replan", details={"namespace": "n"})
    assert e.details == {"namespace": "n"}
    assert ScoutError(ErrorCode.INTERNAL, "x").details == {}
    assert "CONFLICT" in str(e)
    assert isinstance(e, Exception)


def test_error_body_wire_shape_and_round_trip() -> None:
    body = ScoutError(ErrorCode.LOGIN_REQUIRED, "run scout login").to_body(request_id="req-1")
    d = json.loads(body.model_dump_json())
    assert d == {
        "schema_version": "1.0",
        "error": {
            "code": "LOGIN_REQUIRED",
            "message": "run scout login",
            "retryable": False,
            "details": {},
            "request_id": "req-1",
        },
    }
    again = ErrorBody.model_validate_json(body.model_dump_json())
    assert again == body
    assert again.error.code is ErrorCode.LOGIN_REQUIRED


def test_error_body_rejects_unknown_code_and_bad_schema_version() -> None:
    good = {"schema_version": "1.0", "error": {"code": "INTERNAL", "message": "m", "retryable": False}}
    assert ErrorBody.model_validate(good).error.request_id is None
    bad_code = {**good, "error": {**good["error"], "code": "WAT"}}
    with pytest.raises(ValidationError):
        ErrorBody.model_validate(bad_code)
    with pytest.raises(ValidationError):
        ErrorBody.model_validate({**good, "schema_version": "2.0"})
