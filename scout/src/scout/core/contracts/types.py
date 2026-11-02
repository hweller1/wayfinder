"""Public contract types (design section 4, PRD 4.1). Requests forbid extras, responses ignore them."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
SCHEMA_VERSION_PATTERN = r"^1\.\d+$"

RENDER_MAX_TOKENS_MIN = 200
RENDER_MAX_TOKENS_MAX = 4000
GREP_TEXT_MAX_CHARS = 300


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _schema_version() -> str:
    return Field(default=SCHEMA_VERSION, pattern=SCHEMA_VERSION_PATTERN)


class Budget(RequestModel):
    max_turns: int = Field(default=40, ge=1, le=200)
    max_seconds: int = Field(default=120, ge=5, le=900)


class ExploreRequest(RequestModel):
    brief: str = Field(min_length=1, max_length=100_000)
    paths: list[str] = Field(default_factory=list)
    max_results: int = Field(default=12, ge=1, le=40)
    render_max_tokens: int = 1500
    budget: Budget = Field(default_factory=Budget)
    root: str | None = None

    @field_validator("render_max_tokens")
    @classmethod
    def _clamp_render_max_tokens(cls, v: int) -> int:
        return max(RENDER_MAX_TOKENS_MIN, min(RENDER_MAX_TOKENS_MAX, v))


class Hit(ResponseModel):
    path: str
    start_line: int = Field(ge=1)
    end_line: int
    why: str = Field(max_length=300)
    source: Literal["semantic", "grep", "read"]
    stale: bool = False
    score: float | None = None
    # False for entries appended by the backfill (explore.backfill_to): the model did not rank them.
    verified: bool = True

    @model_validator(mode="after")
    def _end_not_before_start(self) -> "Hit":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class IndexStatus(ResponseModel):
    schema_version: str = _schema_version()
    state: Literal["ready", "partial", "absent"]
    coverage: float = Field(ge=0.0, le=1.0)
    commit: str | None = None
    syncing: bool = False
    files: int = 0
    last_sync_at: str | None = None
    semantic_enabled: bool  # coverage >= 0.8


class ModelTokens(ResponseModel):
    # Wire name is `in` (a Python keyword); serialize by alias by default so dumps match PRD 4.1.
    model_config = ConfigDict(extra="ignore", populate_by_name=True, serialize_by_alias=True)

    in_: int = Field(alias="in")
    out: int


class Usage(ResponseModel):
    turns: int
    tool_calls: dict[str, int]
    model_tokens: ModelTokens
    wall_ms: int
    cost_usd: float | None = None


class ExploreResult(ResponseModel):
    schema_version: str = _schema_version()
    results: list[Hit]
    summary: str = Field(max_length=1200)
    gaps: list[str]
    index: IndexStatus
    usage: Usage
    budget_exhausted: bool = False


class SemanticSearchRequest(RequestModel):
    query: str
    path_glob: str | None = None
    k: int = Field(default=10, ge=1, le=30)
    root: str | None = None


class SnippetHit(Hit):
    snippet: str


class SemanticSearchResult(ResponseModel):
    schema_version: str = _schema_version()
    hits: list[SnippetHit]
    index: IndexStatus
    note: str | None = None


class GrepRequest(RequestModel):
    pattern: str
    path: str | None = None
    regex: bool = True
    context: int = Field(default=0, ge=0, le=5)
    root: str | None = None


class GrepMatch(ResponseModel):
    line: int
    text: str

    @field_validator("text")
    @classmethod
    def _truncate_text(cls, v: str) -> str:
        return v[:GREP_TEXT_MAX_CHARS]


class GrepFile(ResponseModel):
    path: str
    matches: list[GrepMatch]


class GrepResult(ResponseModel):
    schema_version: str = _schema_version()
    files: list[GrepFile]
    truncated: bool


class ReadRangeRequest(RequestModel):
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = None
    root: str | None = None


class ReadRangeResult(ResponseModel):
    schema_version: str = _schema_version()
    path: str
    start_line: int
    end_line: int
    lines: list[str] = Field(max_length=400)
    truncated: bool


class IndexStatusRequest(RequestModel):
    root: str | None = None
