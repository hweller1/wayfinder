"""Conformance cases (design section 9): declarative `{id, fixture_repo, chat_script, request, expect}`.

One case = one fake system (the `SCOUT_FAKE_CASE` document built by `Case.doc`) + one request. Every transport
(SDK, CLI, MCP stdio, MCP HTTP, daemon) builds the same system and must return the same `ExploreResult` /
primitive JSON, or the same typed error mapped per design section 4. `expect` states what the case is for, so
a transport that agrees with the others on a wrong answer still fails.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scout.core.contracts.errors import ErrorCode

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
IDENT = "getUserSession"
SESSION = "src/auth/session.py"
BRIEF = "where is getUserSession defined and who calls it?"

Check = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Case:
    id: str
    op: str  # explore | semantic_search | grep | read_range | index_status
    request: dict[str, Any]
    expect_code: ErrorCode | None = None  # None: the call succeeds
    check: Check | None = None  # extra assertions on the parsed success body
    fixture_repo: str = "repo_small"
    chat_script: tuple[dict[str, Any], ...] = ()
    extras: dict[str, Any] = field(
        default_factory=dict
    )  # sync, embed_fail_regex, edit_after_sync, git, gateway_error

    def doc(self) -> dict[str, Any]:
        """The `SCOUT_FAKE_CASE` document (the driver adds `state_dir`)."""
        return {
            "repo": str(FIXTURES / self.fixture_repo),
            "chat_script": list(self.chat_script),
            "backfill_to": 0,  # fake-store ties at the top-k edge differ per process: cases pin the backfill
            **self.extras,
        }


def _call(id_: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_calls": [{"id": id_, "name": name, "arguments": arguments}],
        "tokens_in": 100,
        "tokens_out": 10,
    }


def _hit(path: str, start: int, end: int, why: str = "session lookup") -> dict[str, Any]:
    return {"path": path, "start_line": start, "end_line": end, "why": why}


def _report(*hits: dict[str, Any], gaps: list[str] | None = None) -> dict[str, Any]:
    args = {"results": list(hits), "summary": f"{IDENT} lives in {SESSION}.", "gaps": gaps or []}
    return _call("r", "report", args)


GREP_THEN_REPORT = (
    _call("c1", "grep", {"pattern": IDENT}),
    _call("c2", "read_file", {"path": SESSION, "start_line": 6, "end_line": 12}),
    _report(_hit(SESSION, 6, 12, "defines getUserSession"), _hit("src/handlers/login.py", 4, 6, "calls it")),
)
SEMANTIC_THEN_REPORT = (
    _call("c1", "semantic_search", {"query": f"{IDENT} session lookup"}),
    _call("c2", "read_file", {"path": SESSION, "start_line": 6, "end_line": 12}),
    _report(_hit(SESSION, 6, 12, "defines getUserSession")),
)


def _index(state: str) -> Check:
    def check(body: dict[str, Any]) -> None:
        index = body.get("index", body)
        assert index["state"] == state, index

    return check


def _paths(*expected: str) -> Check:
    def check(body: dict[str, Any]) -> None:
        got = [h["path"] for h in body["results"] if h.get("verified", True)]  # the model's own ranking
        assert got == list(expected), got
        assert all(not h.get("verified", True) for h in body["results"][len(got) :]), body["results"]

    return check


def _backfilled(n: int) -> Check:
    def check(body: dict[str, Any]) -> None:
        extra = [h for h in body["results"] if not h["verified"]]
        assert len(extra) == n, body["results"]
        assert all(h["source"] == "semantic" for h in extra), body["results"]

    return check


def _all(*checks: Check) -> Check:
    def check(body: dict[str, Any]) -> None:
        for c in checks:
            c(body)

    return check


def _flag(key: str, value: object) -> Check:
    def check(body: dict[str, Any]) -> None:
        assert body[key] == value, body

    return check


def _grep_finds(path: str) -> Check:
    def check(body: dict[str, Any]) -> None:
        assert path in {f["path"] for f in body["files"]}, body

    return check


def _gap(fragment: str) -> Check:
    def check(body: dict[str, Any]) -> None:
        assert any(fragment in g for g in body["gaps"]), body["gaps"]

    return check


def _no_hit_in(path: str) -> Check:
    def check(body: dict[str, Any]) -> None:
        assert path not in {h["path"] for h in body["hits"]}, body["hits"]
        assert "dropped" in body["note"], body["note"]

    return check


def _semantic(path: str) -> Check:
    def check(body: dict[str, Any]) -> None:
        assert body["hits"], body
        assert path in {h["path"] for h in body["hits"]}, [h["path"] for h in body["hits"]]
        assert all(h["source"] == "semantic" for h in body["hits"])

    return check


def _note_mentions(word: str) -> Check:
    def check(body: dict[str, Any]) -> None:
        assert body["hits"] == []
        assert word in body["note"], body

    return check


PARTIAL = {"sync": "partial", "embed_fail_regex": "session", "sync_batch_tokens": 60}
READY = {"sync": "ready"}

# Errors with no natural trigger in the fake system are injected at the gateway (`gateway_error`).
_INJECTED = (
    ErrorCode.UNSUPPORTED_SCHEMA,
    ErrorCode.UNAUTHENTICATED,
    ErrorCode.LOGIN_REQUIRED,
    ErrorCode.PERMISSION_DENIED,
    ErrorCode.RATE_LIMITED,
    ErrorCode.CONFLICT,
    ErrorCode.UPSTREAM_UNAVAILABLE,
    ErrorCode.UPSTREAM_TIMEOUT,
    ErrorCode.INTERNAL,
)


def _retry_after(code: ErrorCode) -> int | None:
    return 7 if code is ErrorCode.RATE_LIMITED else None


SUCCESS: list[Case] = [
    Case(
        "explore_ready",
        "explore",
        {"brief": BRIEF},
        check=_all(_index("ready"), _paths(SESSION), _backfilled(1)),
        chat_script=SEMANTIC_THEN_REPORT,
        extras={**READY, "backfill_to": 2},  # the model's file plus the best semantic hit for the brief
    ),
    Case(
        "explore_partial",
        "explore",
        {"brief": BRIEF},
        check=_all(_index("partial"), _paths(SESSION, "src/handlers/login.py"), _gap("grep-only")),
        chat_script=GREP_THEN_REPORT,
        extras=PARTIAL,
    ),
    Case(
        "explore_absent",
        "explore",
        {"brief": BRIEF, "paths": ["src/**"], "max_results": 5},
        check=_all(_index("absent"), _paths(SESSION, "src/handlers/login.py"), _gap("grep-only")),
        chat_script=GREP_THEN_REPORT,
    ),
    Case(
        "explore_budget_exhausted",
        "explore",
        {"brief": BRIEF, "budget": {"max_turns": 1}},
        check=_all(_flag("budget_exhausted", True), _index("absent")),
        chat_script=(_call("c1", "grep", {"pattern": IDENT}), *GREP_THEN_REPORT),
    ),
    Case(
        "explore_stale_hit",
        "explore",
        {"brief": BRIEF},
        check=_all(_index("ready"), _paths(), _gap("dropped")),
        chat_script=SEMANTIC_THEN_REPORT,
        extras={**READY, "edit_after_sync": {SESSION: "def renamed_function():\n    return None\n"}},
    ),
    Case(
        "explore_path_refused_in_loop",
        "explore",
        {"brief": BRIEF},
        check=_paths(SESSION),
        chat_script=(
            _call("c1", "read_file", {"path": "../../etc/passwd"}),
            _call("c2", "read_file", {"path": SESSION, "start_line": 6, "end_line": 12}),
            _report(_hit(SESSION, 6, 12)),
        ),
    ),
    Case("grep", "grep", {"pattern": IDENT, "context": 1}, check=_grep_finds(SESSION)),
    Case(
        "grep_literal_in_dir",
        "grep",
        {"pattern": "getUserSession(", "path": "src", "regex": False},
        check=_grep_finds(SESSION),
    ),
    Case(
        "read_range",
        "read_range",
        {"path": SESSION, "start_line": 6, "end_line": 8},
        check=_all(_flag("start_line", 6), _flag("end_line", 8)),
    ),
    Case("index_status_absent", "index_status", {}, check=_index("absent")),
    Case(
        "index_status_ready",
        "index_status",
        {},
        check=_all(_index("ready"), _flag("semantic_enabled", True)),
        extras=READY,
    ),
    Case(
        "index_status_partial",
        "index_status",
        {},
        check=_all(_index("partial"), _flag("semantic_enabled", False)),
        extras=PARTIAL,
    ),
    Case(
        "semantic_search_ready",
        "semantic_search",
        {"query": f"{IDENT} session lookup", "path_glob": "src/auth/session.py", "k": 3},
        check=_semantic(SESSION),
        extras=READY,
    ),
    Case(
        "semantic_search_absent",
        "semantic_search",
        {"query": "anything"},
        check=_note_mentions("grep"),
    ),
    Case(
        "semantic_search_stale_file",
        "semantic_search",
        {"query": f"{IDENT} session lookup", "path_glob": "src/auth/session.py", "k": 3},
        check=_no_hit_in(SESSION),
        extras={**READY, "edit_after_sync": {SESSION: "def renamed_function():\n    return None\n"}},
    ),
]  # fmt: skip

ERRORS: list[Case] = [
    Case("error_invalid_argument", "grep", {"pattern": "("}, expect_code=ErrorCode.INVALID_ARGUMENT),
    Case("error_not_found", "read_range", {"path": "no/such/file.py"}, expect_code=ErrorCode.NOT_FOUND),
    Case("error_path_refused", "read_range", {"path": "../escape"}, expect_code=ErrorCode.PATH_REFUSED),
    Case("error_not_a_repo", "index_status", {}, expect_code=ErrorCode.NOT_A_REPO, extras={"git": False}),
    *(
        Case(
            f"error_{code.value.lower()}",
            "index_status",
            {},
            expect_code=code,
            extras={"gateway_error": {"code": code.value, "retry_after": _retry_after(code)}},
        )
        for code in _INJECTED
    ),
]  # fmt: skip

CASES: list[Case] = [*SUCCESS, *ERRORS]
BY_ID: dict[str, Case] = {c.id: c for c in CASES}
assert len(BY_ID) == len(CASES)
