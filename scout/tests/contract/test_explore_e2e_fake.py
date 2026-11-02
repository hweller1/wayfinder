"""Task 5.6: semantic explore end to end with every fake: `Scout(profile="fake")` syncs `repo_small`, the
scripted chat calls `semantic_search` -> `read_file` -> `report`, the result validates against the published
schema and points at real lines. A file edited between search and report is never reported with stale lines.

AC-1.1, AC-1.5 (relocation itself is task 5.2: until then a hit on an edited file is dropped).
"""

import json
from pathlib import Path
from typing import Any

import jsonschema

from scout import Scout
from scout.adapters.fake_fs import FakeFileSystem
from scout.core.contracts.types import ExploreResult

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "tests" / "fixtures" / "repo_small"
SCHEMA = json.loads((ROOT / "schema" / "v1" / "ExploreResult.json").read_text())
IDENT = "getUserSession"
SESSION = "src/auth/session.py"


def _lines(rel: str) -> list[str]:
    return (REPO / rel).read_text(encoding="utf-8").splitlines()


def _call(id_: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_calls": [{"id": id_, "name": name, "arguments": arguments}],
        "tokens_in": 100,
        "tokens_out": 10,
    }


def _report(*results: dict[str, Any]) -> dict[str, Any]:
    args = {"results": list(results), "summary": f"{IDENT} lives in {SESSION}.", "gaps": []}
    return _call("r", "report", args)


def _hit(path: str, start: int, end: int) -> dict[str, Any]:
    return {"path": path, "start_line": start, "end_line": end, "why": "session lookup"}


def _fake(script: list[dict[str, Any]], tmp_path: Path) -> Scout:
    case = {"repo": str(REPO), "chat_script": script, "state_dir": str(tmp_path / "state")}
    scout = Scout(profile="fake", case=case)
    assert scout.sync().state == "ready"
    return scout


def _fs(scout: Scout) -> FakeFileSystem:
    fs = scout._app._fs  # pyright: ignore[reportPrivateUsage]
    assert isinstance(fs, FakeFileSystem)
    return fs


def test_semantic_search_read_file_report_end_to_end(tmp_path: Path) -> None:
    script = [
        _call("c1", "semantic_search", {"query": f"{IDENT} session lookup by token", "k": 5}),
        _call("c2", "read_file", {"path": SESSION, "start_line": 6, "end_line": 12}),
        _report(_hit(SESSION, 6, 12)),
    ]
    scout = _fake(script, tmp_path)
    result = scout.explore("where is the session lookup implemented?")

    jsonschema.validate(json.loads(result.model_dump_json(by_alias=True)), SCHEMA)
    assert isinstance(result, ExploreResult)
    assert result.index.state == "ready"
    assert result.index.semantic_enabled
    assert not any("grep-only" in g for g in result.gaps)
    assert not result.budget_exhausted
    assert result.usage.tool_calls == {"semantic_search": 1, "read_file": 1, "report": 1}
    assert result.usage.turns == 3

    assert [(h.path, h.start_line, h.end_line) for h in result.results if h.verified] == [(SESSION, 6, 12)]
    hit = result.results[0]
    assert hit.source == "read"
    disk = _lines(hit.path)
    assert 1 <= hit.start_line <= hit.end_line <= len(disk)
    assert f"def {IDENT}" in disk[hit.start_line - 1]
    assert ".env" not in json.dumps(result.model_dump(mode="json"))


def test_hits_equal_real_line_ranges_of_the_fixture(tmp_path: Path) -> None:
    """Every range the explorer is shown by `semantic_search` and then reports is a real range, and the
    snippet it saw is the numbered disk text of exactly those lines."""
    scout = _fake([], tmp_path)
    sem = scout.semantic_search(f"{IDENT} session lookup", k=8)
    assert sem.hits
    for h in sem.hits:
        disk = _lines(h.path)
        assert 1 <= h.start_line <= h.end_line <= len(disk)
        shown = [ln for ln in h.snippet.splitlines() if ln[:1].isdigit()]
        for ln in shown:
            n, _, text = ln.partition("\t")
            assert disk[int(n) - 1] == text
    assert SESSION in {h.path for h in sem.hits}


def test_semantic_hit_reported_as_is_has_semantic_source(tmp_path: Path) -> None:
    probe = _fake([], tmp_path / "probe").semantic_search(f"{IDENT} session lookup", k=5)
    top = next(h for h in probe.hits if h.path == SESSION)
    script = [
        _call("c1", "semantic_search", {"query": f"{IDENT} session lookup", "k": 5}),
        _report(_hit(SESSION, top.start_line, top.end_line)),
    ]
    result = _fake(script, tmp_path / "run").explore("where is the session lookup?")
    assert [(h.path, h.source) for h in result.results if h.verified] == [(SESSION, "semantic")]
    assert result.usage.tool_calls == {"semantic_search": 1, "report": 1}


def test_file_edited_between_search_and_report_never_returns_stale_lines(tmp_path: Path) -> None:
    script = [
        _call("c1", "semantic_search", {"query": f"{IDENT} session lookup", "k": 5}),
        _call("c2", "glob", {"pattern": "*.md"}),  # the edit lands while this tool runs
        _report(_hit(SESSION, 6, 12), _hit("README.md", 1, 1)),
    ]
    scout = _fake(script, tmp_path)
    fs = _fs(scout)
    original = fs.list_files
    shifted = b"# a new header\n" * 30 + fs.files[SESSION]

    def edit_then_list() -> list[str]:
        fs.files[SESSION] = shifted  # the reported lines 6-12 now hold other text
        fs.list_files = original  # type: ignore[method-assign]
        return original()

    fs.list_files = edit_then_list  # type: ignore[method-assign]
    result = scout.explore("where is the session lookup implemented?")

    paths = [h.path for h in result.results]
    assert SESSION not in paths  # dropped: never the old line numbers against the new text
    for h in result.results:
        current = fs.files[h.path].decode().split("\n")
        assert 1 <= h.start_line <= h.end_line <= len(current)
    assert any("changed after it was searched" in g for g in result.gaps)
    assert result.index.state == "ready"
    assert result.usage.tool_calls["semantic_search"] == 1
    jsonschema.validate(json.loads(result.model_dump_json(by_alias=True)), SCHEMA)
