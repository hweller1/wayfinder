"""Task 6.1: the `scout` CLI over the fake profile, driven through `subprocess` (FR-S2, FR-S7).

Covers: exit code per `ErrorCode` (design section 4), JSON on stdout that validates against `schema/v1`,
`--text` equal to `render_text`, error body on stdout plus one human line on stderr, and lazy imports.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from scout.core.contracts.errors import ErrorBody, ErrorCode, ScoutError, cli_exit
from scout.core.contracts.types import ExploreResult
from scout.core.explore.render import render_text
from scout.surfaces.cli.exit import report_error

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tests" / "fixtures" / "poc_explore_case.json"
SCHEMA = ROOT / "schema" / "v1"


def scout(*argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full = {**os.environ, "SCOUT_PROFILE": "fake", "SCOUT_FAKE_CASE": str(CASE), **(env or {})}
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", *argv], capture_output=True, text=True, env=full, check=False
    )


def validate(name: str, doc: Any) -> None:
    schema = json.loads((SCHEMA / f"{name}.json").read_text())
    Draft202012Validator(schema).validate(doc)


@pytest.mark.parametrize("code", list(ErrorCode))
def test_exit_code_per_error_code(code: ErrorCode, capsys: pytest.CaptureFixture[str]) -> None:
    """Every ErrorCode: exit code from the table, body on stdout, one human line on stderr."""
    rc = report_error(ScoutError(code, "boom"))
    out, err = capsys.readouterr()
    assert rc == cli_exit(code)
    body = ErrorBody.model_validate_json(out)
    assert body.error.code is code
    validate("Error", json.loads(out))
    assert err.count("\n") == 1
    assert f"{code}" in err
    if code in (ErrorCode.LOGIN_REQUIRED, ErrorCode.UNAUTHENTICATED):
        assert "scout login" in err


def test_explore_json_is_schema_valid() -> None:
    p = scout("explore", "where is getUserSession defined?")
    assert p.returncode == 0, p.stderr
    doc = json.loads(p.stdout)
    validate("ExploreResult", doc)
    assert [h["path"] for h in doc["results"] if h["verified"]] == [
        "src/auth/session.py",
        "src/handlers/login.py",
    ]
    assert p.stderr == ""


def test_explore_text_equals_render_text() -> None:
    j = scout("explore", "where is getUserSession defined?")
    t = scout("explore", "where is getUserSession defined?", "--text")
    assert t.returncode == 0, t.stderr
    result = ExploreResult.model_validate_json(j.stdout)
    assert t.stdout == render_text(result, 1500) + "\n"
    small = scout("explore", "where is getUserSession defined?", "--text", "--render-max-tokens", "200")
    assert small.stdout == render_text(result, 200) + "\n"


def test_primitives_emit_schema_valid_json() -> None:
    cases = {
        "IndexStatus": ("status",),
        "GrepResult": ("grep", "getUserSession", "--context", "1"),
        "ReadRangeResult": ("read", "src/auth/session.py", "6", "8"),
        "SemanticSearchResult": ("search", "session lookup"),
    }
    for schema, argv in cases.items():
        p = scout(*argv)
        assert p.returncode == 0, (argv, p.stderr)
        validate(schema, json.loads(p.stdout))
    g = json.loads(scout("grep", "getUserSession").stdout)
    assert "src/auth/session.py" in {f["path"] for f in g["files"]}
    r = json.loads(scout("read", "src/auth/session.py", "6", "7").stdout)
    assert (r["start_line"], r["end_line"]) == (6, 7)
    assert r["lines"][0].startswith("def getUserSession")
    s = json.loads(scout("search", "anything").stdout)  # index absent: no hits, grep-only note
    assert s["hits"] == []
    assert "grep" in s["note"]


def test_text_renderings_of_primitives() -> None:
    assert scout("status", "--text").stdout.startswith("index absent coverage=0.00")
    g = scout("grep", "getUserSession", "--text")
    assert "src/auth/session.py:6:" in g.stdout
    r = scout("read", "src/auth/session.py", "6", "7", "--text")
    assert r.stdout.splitlines()[0] == "src/auth/session.py:6-7"
    assert r.stdout.splitlines()[1].startswith("6\tdef getUserSession")


def test_errors_map_to_exit_codes_with_body_on_stdout() -> None:
    refused = scout("read", "../escape")
    assert refused.returncode == cli_exit(ErrorCode.PATH_REFUSED)
    body = json.loads(refused.stdout)
    validate("Error", body)
    assert body["error"]["code"] == "PATH_REFUSED"
    assert refused.stderr.count("\n") == 1
    assert "PATH_REFUSED" in refused.stderr

    bad = scout("grep", "x", "--context", "99")
    assert bad.returncode == cli_exit(ErrorCode.INVALID_ARGUMENT)
    assert json.loads(bad.stdout)["error"]["code"] == "INVALID_ARGUMENT"

    usage = scout("read")  # a usage error is INVALID_ARGUMENT too
    assert usage.returncode == 2
    assert json.loads(usage.stdout)["error"]["code"] == "INVALID_ARGUMENT"

    missing = scout("status", env={"SCOUT_FAKE_CASE": str(ROOT / "nope.json")})
    assert missing.returncode != 0
    assert json.loads(missing.stdout)["error"]["code"] in {"INVALID_ARGUMENT", "INTERNAL"}


def test_unknown_profile_and_missing_case_are_typed() -> None:
    p = scout("status", "--profile", "nope")
    assert p.returncode == 2
    p2 = scout("status", env={"SCOUT_FAKE_CASE": ""})
    assert p2.returncode == 2
    assert json.loads(p2.stdout)["error"]["code"] == "INVALID_ARGUMENT"


def test_schema_command_lists_and_prints() -> None:
    names = scout("schema").stdout.split()
    assert {"ExploreRequest", "ExploreResult", "Error"} <= set(names)
    doc = json.loads(scout("schema", "ExploreResult").stdout)
    assert doc["title"] == "ExploreResult"
    assert scout("schema", "Nope").returncode == cli_exit(ErrorCode.NOT_FOUND)


def test_parsing_imports_nothing_heavy() -> None:
    """`--help` and a parse error must not import the app, adapters, the MCP SDK or any vendor SDK."""
    code = (
        "import sys\n"
        "from scout.surfaces.cli.main import main\n"
        "from scout.surfaces.cli.main import build_parser\n"
        "build_parser().parse_args(['explore', 'x'])\n"
        "heavy = ('mcp', 'turbopuffer', 'voyageai', 'httpx', 'watchfiles', 'tree_sitter', 'scout.core.app',\n"
        "         'scout.wiring', 'scout.surfaces.cli.commands')\n"
        "bad = [m for m in heavy if m in sys.modules]\n"
        "print(bad)\n"
    )
    p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)  # noqa: S603
    assert p.stdout.strip() == "[]"
