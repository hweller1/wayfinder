"""JSON Schema export (design section 4) and the published `status.json` model (design section 8)."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scout.core.contracts.errors import ErrorBody
from scout.core.contracts.types import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_PATTERN,
    ExploreRequest,
    ExploreResult,
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
from scout.core.contracts.wire import WIRE_MODELS

__all__ = ["SCHEMA_VERSION", "StatusFile", "export_schemas", "render_schemas"]


class StatusFile(BaseModel):
    """`<state_dir>/status.json`: read by the Claude Code hook without importing scout (stdlib only)."""

    # `schema` shadows a BaseModel attribute, so the Python name differs from the wire name.
    model_config = ConfigDict(extra="ignore", populate_by_name=True, serialize_by_alias=True)

    schema_: str = Field(default=SCHEMA_VERSION, pattern=SCHEMA_VERSION_PATTERN, alias="schema")
    state: Literal["ready", "partial", "absent"]
    coverage: float = Field(ge=0.0, le=1.0)
    commit: str | None = None
    syncing: bool = False
    last_sync_at: str | None = None
    gateway_ok_at: str | None = None
    last_error: str | None = (
        None  # the watcher's last failed sync (code + message), cleared when the next sync writes status
    )


_PUBLIC: dict[str, type[BaseModel]] = {
    "ExploreRequest": ExploreRequest,
    "ExploreResult": ExploreResult,
    "Hit": Hit,
    "SnippetHit": SnippetHit,
    "IndexStatus": IndexStatus,
    "IndexStatusRequest": IndexStatusRequest,
    "Usage": Usage,
    "SemanticSearchRequest": SemanticSearchRequest,
    "SemanticSearchResult": SemanticSearchResult,
    "GrepRequest": GrepRequest,
    "GrepResult": GrepResult,
    "ReadRangeRequest": ReadRangeRequest,
    "ReadRangeResult": ReadRangeResult,
    "Error": ErrorBody,
    "status_file": StatusFile,
}


def _models() -> dict[str, type[BaseModel]]:
    models = dict(_PUBLIC)
    for m in WIRE_MODELS:
        models[m.__name__] = m
    return models


def _dump(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def render_schemas() -> dict[str, str]:
    """Filename -> exact file content for every schema file, including `index.json`."""
    out = {
        f"{name}.json": _dump(model.model_json_schema(mode="serialization"))
        for name, model in sorted(_models().items())
    }
    out["index.json"] = _dump({"schema_version": SCHEMA_VERSION, "files": sorted(out)})
    return out


def export_schemas(directory: Path) -> list[Path]:
    """Write all schema files into `directory` (created if absent); returns the written paths."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in render_schemas().items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
