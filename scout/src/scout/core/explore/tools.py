"""Explore tools (FR-C4, FR-C5, FR-C6): OpenAI tool specs generated from the contract models, and dispatch.

`ToolRunner.run` never raises for anything the model can cause. Each call yields a `ToolOutcome`:
`ok` (tool ran), `error` (PATH_REFUSED / NOT_FOUND / INVALID_ARGUMENT from the tool itself, fed back as text),
`malformed` (parse_error, unknown tool, wrong arguments; the loop counts these toward its 3-strike fallback)
or `report` (terminal, arguments validated but paths not yet judged: the loop does that).
Errors that are not the model's doing (rg missing, adapter failures) propagate.

`semantic_search` exists only when the loop passes a `SemanticBackend` (it does so when the index is at least
80% covered): the backend returns decrypted, scope-filtered raw hits; this module reads the CURRENT lines from
disk, drops any hit whose file no longer hashes to the indexed `file_hash`, and numbers the snippet.
`changed_since_seen` tells the loop whether a file differs from what the tools last showed the model, so a
hit on an edited file is never returned as if it still matched.
"""

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import (
    GrepRequest,
    GrepResult,
    ReadRangeRequest,
    ReadRangeResult,
    RequestModel,
    SemanticSearchRequest,
    SnippetHit,
)
from scout.core.explore.candidates import CandidateSet, RawHit, read_checked, verify_hits
from scout.core.local.glob import GlobResult, glob
from scout.core.local.grep import grep
from scout.core.local.ignore import Matcher
from scout.core.local.readfile import read_range
from scout.core.ports.chat import ToolCall
from scout.core.ports.clock import Clock
from scout.core.ports.fs import FileSystem

MAX_TOOL_OUTPUT_CHARS = 24_000  # what one tool result may put in the model's context
_IN_LOOP_ERRORS = frozenset({ErrorCode.INVALID_ARGUMENT, ErrorCode.PATH_REFUSED, ErrorCode.NOT_FOUND})


@dataclass(frozen=True, slots=True)
class SemanticBackend:
    """What the semantic tool needs from the facade: `search(query, path_glob, k)` returns decrypted raw hits
    already narrowed by the glob (best first, up to the overfetch), `leaf_hash(bytes)` the keyed hash the
    sync engine stored as `file_hash`."""

    search: Callable[[str, str | None, int], Awaitable[list[RawHit]]]
    leaf_hash: Callable[[bytes], str]


class GlobArgs(RequestModel):
    pattern: str


class ReportHit(RequestModel):
    path: str
    start_line: int
    end_line: int
    why: str


class ReportArgs(RequestModel):
    results: list[ReportHit]
    summary: str
    gaps: list[str] = Field(default_factory=list[str])


_DESCRIPTIONS = {
    "semantic_search": "Search the indexed repository by meaning. Returns ranked snippets with line numbers.",
    "grep": "Search file contents with ripgrep. Files are ranked by match density. Returns path:line: text.",
    "glob": "List repository files whose path matches a glob pattern (at most 1000).",
    "read_file": "Read a 1-based inclusive line range of a file (at most 400 lines per call).",
    "report": (
        "Finish. Give the most relevant locations first, each as path, start_line, end_line (1-based, "
        "inclusive) and one line saying why, then a summary answering the brief and the gaps you could not "
        "establish. Call exactly once."
    ),
}
_MODELS: dict[str, type[BaseModel]] = {
    "semantic_search": SemanticSearchRequest,
    "grep": GrepRequest,
    "glob": GlobArgs,
    "read_file": ReadRangeRequest,
    "report": ReportArgs,
}
_TOOL_ORDER = ("semantic_search", "grep", "glob", "read_file", "report")


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        d: dict[str, Any] = node  # pyright: ignore[reportUnknownVariableType]
        return {k: _strip_titles(v) for k, v in d.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]  # pyright: ignore[reportUnknownVariableType]
    return node


def _parameters(model: type[BaseModel]) -> dict[str, Any]:
    schema: dict[str, Any] = _strip_titles(model.model_json_schema())
    props: dict[str, Any] = schema.get("properties", {})
    props.pop("root", None)  # the host picks the root, never the model
    return schema


def tool_specs(semantic_enabled: bool) -> list[dict[str, Any]]:
    names = [n for n in _TOOL_ORDER if semantic_enabled or n != "semantic_search"]
    return [
        {
            "type": "function",
            "function": {"name": n, "description": _DESCRIPTIONS[n], "parameters": _parameters(_MODELS[n])},
        }
        for n in names
    ]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    kind: Literal["ok", "error", "malformed", "report"]
    text: str
    report: ReportArgs | None = None


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    cut = text[:MAX_TOOL_OUTPUT_CHARS].rpartition("\n")[0]
    return f"{cut}\n... output truncated ({len(text) - len(cut)} more characters omitted)"


def _grep_text(res: GrepResult) -> str:
    if not res.files:
        return "no matches"
    out = [f"{f.path}:{m.line}: {m.text}" for f in res.files for m in f.matches]
    if res.truncated:
        out.append("... results truncated (per-file or file cap reached); narrow the pattern or path")
    return _clip("\n".join(out))


def _read_text(res: ReadRangeResult) -> str:
    out = [f"{res.path} lines {res.start_line}-{res.end_line}"]
    out += [f"{res.start_line + i}\t{ln}" for i, ln in enumerate(res.lines)]
    if res.truncated:
        out.append("... truncated at 400 lines; call read_file again from the next line")
    return _clip("\n".join(out))


def _glob_text(res: GlobResult) -> str:
    if not res.paths:
        return "no files match"
    out = list(res.paths)
    if res.truncated:
        out.append("... truncated at 1000 paths; narrow the pattern")
    return _clip("\n".join(out))


def semantic_text(hits: list[SnippetHit], dropped: int) -> str:
    if not hits:
        out = ["no semantic matches"]
    else:
        out = [f"{h.path}:{h.start_line}-{h.end_line} (score {h.score or 0:.4f})\n{h.snippet}" for h in hits]
    if dropped:
        out.append(f"... {dropped} hit(s) dropped: the file changed since it was indexed")
    return _clip("\n\n".join(out))


def _arg_errors(e: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in err['loc']) or '(arguments)'}: {err['msg']}" for err in e.errors()]
    return "; ".join(parts[:5])


class ToolRunner:
    """Dispatches model tool calls to `core.local` (and the semantic backend when given) and feeds
    `candidates`. `calls` counts every call to an offered tool, malformed ones included, so usage matches the
    script exactly."""

    def __init__(
        self,
        fs: FileSystem,
        *,
        matcher: Matcher | None = None,
        clock: Clock | None = None,
        semantic: SemanticBackend | None = None,
    ) -> None:
        self._fs = fs
        self._clock = clock
        self._semantic = semantic
        self.matcher = matcher or Matcher.from_fs(fs)
        self.names = tuple(n for n in _TOOL_ORDER if semantic is not None or n != "semantic_search")
        self.candidates = CandidateSet()
        self.calls: dict[str, int] = {}
        self.opened: dict[
            str, tuple[int, int]
        ] = {}  # path -> first range read_file returned (first-seen order)
        self._seen: dict[str, str] = {}  # path -> keyed hash of the bytes the model was first shown

    def changed_since_seen(self, path: str) -> bool:
        """True when `path` was shown to the model (semantic snippet or read_file) and its bytes differ now. A
        range of such a file is dropped from the result rather than reported against lines that moved."""
        if self._semantic is None or path not in self._seen:
            return False
        got = read_checked(self._fs, self.matcher, path)
        return got is None or self._semantic.leaf_hash(got[1]) != self._seen[path]

    def _note_read(self, path: str) -> None:
        if self._semantic is None:
            return
        got = read_checked(self._fs, self.matcher, path)
        if got is not None:
            self._seen.setdefault(got[0], self._semantic.leaf_hash(got[1]))

    async def run(self, call: ToolCall) -> ToolOutcome:
        if call.name in self.names:
            self.calls[call.name] = self.calls.get(call.name, 0) + 1
        if call.parse_error is not None:
            return ToolOutcome(
                "malformed",
                f"error: the arguments of {call.name} were not valid JSON ({call.parse_error}). "
                "Re-issue the call with a valid JSON object.",
            )
        if call.name not in self.names:
            return ToolOutcome(
                "malformed", f"error: unknown tool {call.name!r}. Available tools: {', '.join(self.names)}."
            )
        if "root" in call.arguments:
            return ToolOutcome(
                "malformed", f"error: invalid arguments for {call.name}: root: not an argument"
            )
        try:
            return await self._dispatch(call.name, call.arguments)
        except ValidationError as e:
            return ToolOutcome("malformed", f"error: invalid arguments for {call.name}: {_arg_errors(e)}")
        except ScoutError as e:
            if e.code not in _IN_LOOP_ERRORS:
                raise
            return ToolOutcome("error", f"error: {e.code}: {e.message}")

    async def _dispatch(self, name: str, args: Mapping[str, Any]) -> ToolOutcome:
        if name == "semantic_search":
            assert self._semantic is not None
            sreq = SemanticSearchRequest.model_validate(args)
            if not sreq.query.strip():
                raise ScoutError(ErrorCode.INVALID_ARGUMENT, "query must not be empty")
            raw = await self._semantic.search(sreq.query, sreq.path_glob, sreq.k)
            hits, dropped = verify_hits(
                self._fs, self.matcher, self._semantic.leaf_hash, raw, sreq.k, self._seen
            )
            self.candidates.add_semantic(hits)
            return ToolOutcome("ok", semantic_text(hits, dropped))
        if name == "grep":
            greq = GrepRequest.model_validate(args)
            gres = grep(self._fs, greq, self.matcher, clock=self._clock)
            self.candidates.add_grep(greq.pattern, gres)
            return ToolOutcome("ok", _grep_text(gres))
        if name == "glob":
            return ToolOutcome(
                "ok", _glob_text(glob(self._fs, GlobArgs.model_validate(args).pattern, self.matcher))
            )
        if name == "read_file":
            rres = read_range(self._fs, ReadRangeRequest.model_validate(args), self.matcher)
            self.candidates.add_read(rres)
            if rres.lines:
                self.opened.setdefault(rres.path, (rres.start_line, rres.end_line))
            self._note_read(rres.path)
            return ToolOutcome("ok", _read_text(rres))
        report = ReportArgs.model_validate(args)
        return ToolOutcome("report", json.dumps({"status": "report accepted"}), report)
