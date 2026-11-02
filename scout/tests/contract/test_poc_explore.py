"""POC end to end (task 1.12): `Scout(profile="fake", case=...)` over the fixture repo, fakes only.

US-1, AC-1.1, FR-S1. The fixture carries two traps that name the identifier: `.env` (secret denylist) and
`generated/` (`.scoutignore`); neither may ever surface.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scout import Scout
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ExploreResult

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REPO = FIXTURES / "repo_small"
CASE = FIXTURES / "poc_explore_case.json"
SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "schema/v1/ExploreResult.json").read_text())
IDENT = "getUserSession"
TRAPS = (".env", "generated/session_gen.py")


def _lines(rel: str) -> list[str]:
    return (REPO / rel).read_text(encoding="utf-8").splitlines()


def _case(script: list[dict[str, Any]]) -> dict[str, Any]:
    return {"repo": str(REPO), "chat_script": script}


def test_fixture_has_the_traps() -> None:
    assert IDENT in (REPO / ".env").read_text()
    assert IDENT in (REPO / "generated/session_gen.py").read_text()
    assert "generated/" in (REPO / ".scoutignore").read_text()
    assert len([p for p in REPO.rglob("*") if p.is_file()]) >= 12


def test_explore_end_to_end_validates_and_points_at_real_lines() -> None:
    result = Scout(profile="fake", case=CASE).explore("where is the session lookup implemented?")

    jsonschema.validate(json.loads(result.model_dump_json(by_alias=True)), SCHEMA)
    assert isinstance(result, ExploreResult)
    assert result.index.state == "absent"
    assert not result.index.semantic_enabled
    assert any("grep-only" in g for g in result.gaps)
    assert not result.budget_exhausted

    assert [h.path for h in result.results] == ["src/auth/session.py", "src/handlers/login.py"]
    for h in result.results:
        lines = _lines(h.path)
        assert 1 <= h.start_line <= h.end_line <= len(lines)
    defn, caller = result.results
    assert f"def {IDENT}" in _lines(defn.path)[defn.start_line - 1]
    assert defn.source == "read"
    assert IDENT in "\n".join(_lines(caller.path)[caller.start_line - 1 : caller.end_line])

    assert result.usage.turns == 3
    assert result.usage.tool_calls == {"grep": 1, "read_file": 1, "report": 1}
    assert result.usage.model_tokens.in_ == 720
    assert result.usage.model_tokens.out == 70
    for trap in TRAPS:
        assert trap not in json.dumps(result.model_dump(mode="json"))


def test_traps_never_returned_even_when_the_model_reports_them() -> None:
    script = [
        {"tool_calls": [{"id": "c1", "name": "grep", "arguments": {"pattern": IDENT}}]},
        {
            "tool_calls": [
                {
                    "id": "r",
                    "name": "report",
                    "arguments": {
                        "results": [
                            {"path": ".env", "start_line": 2, "end_line": 2, "why": "secret"},
                            {
                                "path": "generated/session_gen.py",
                                "start_line": 2,
                                "end_line": 3,
                                "why": "gen",
                            },
                            {"path": "web/src/session.ts", "start_line": 6, "end_line": 9, "why": "ts impl"},
                        ],
                        "summary": "s",
                        "gaps": [],
                    },
                }
            ]
        },
    ]
    result = Scout(profile="fake", case=_case(script)).explore("find it")
    jsonschema.validate(json.loads(result.model_dump_json(by_alias=True)), SCHEMA)
    assert [h.path for h in result.results] == ["web/src/session.ts"]
    assert any("dropped" in g for g in result.gaps)


def test_primitives_respect_the_same_filters() -> None:
    scout = Scout(profile="fake", case=_case([]))
    paths = {f.path for f in scout.grep(IDENT).files}
    assert paths == {
        "src/auth/__init__.py",
        "src/auth/session.py",
        "src/handlers/login.py",
        "cmd/server/main.go",
        "internal/session/session.go",
        "web/src/session.ts",
        "web/src/api.ts",
        "tests/session_spec.py",
        "README.md",
    }
    for trap in TRAPS:
        with pytest.raises(ScoutError) as e:
            scout.read_range(trap)
        assert e.value.code is ErrorCode.PATH_REFUSED
    rr = scout.read_range("src/auth/session.py", start_line=6, end_line=7)
    assert rr.lines[0].startswith(f"def {IDENT}")
    assert scout.index_status().state == "absent"
    sem = scout.semantic_search("session lookup")
    assert sem.hits == []
    assert sem.index.state == "absent"
    assert sem.note


def test_foreign_root_is_refused() -> None:
    with pytest.raises(ScoutError) as e:
        Scout(profile="fake", case=_case([])).grep(IDENT, root="/elsewhere")
    assert e.value.code is ErrorCode.INVALID_ARGUMENT


def test_unknown_profile_and_missing_case_are_typed_errors() -> None:
    for kwargs in ({"profile": "nope"}, {"profile": "fake"}):
        with pytest.raises(ScoutError) as e:
            Scout(**kwargs)  # type: ignore[arg-type]
        assert e.value.code is ErrorCode.INVALID_ARGUMENT


async def test_async_variant_matches_sync() -> None:
    result = await Scout(profile="fake", case=CASE).aexplore("where is the session lookup implemented?")
    assert result.results[0].path == "src/auth/session.py"
