"""Task 8.3: `SdkOverHttpDriver` is a conformance driver variant. FR-G1, AC-6.1, AC-6.2.

Every case runs through the in-process `SdkDriver` and the HTTP-backed one; the parsed JSON results must be
equal (task 6.6 generalises this to the full case list and the other transports).
"""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from drivers import Driver, SdkDriver, SdkOverHttpDriver

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import (
    ExploreRequest,
    GrepRequest,
    IndexStatusRequest,
    ReadRangeRequest,
    SemanticSearchRequest,
)

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "tests" / "fixtures" / "repo_small"
IDENT = "getUserSession"
SESSION = "src/auth/session.py"


def _call(id_: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_calls": [{"id": id_, "name": name, "arguments": arguments}],
        "tokens_in": 100,
        "tokens_out": 10,
    }


def _case(*, semantic: bool) -> dict[str, Any]:
    report = {
        "results": [{"path": SESSION, "start_line": 6, "end_line": 12, "why": "session lookup"}],
        "summary": f"{IDENT} lives in {SESSION}.",
        "gaps": [],
    }
    script = [
        _call(
            "c1",
            "semantic_search" if semantic else "grep",
            {"query": IDENT, "k": 5} if semantic else {"pattern": IDENT},
        ),
        _call("c2", "read_file", {"path": SESSION, "start_line": 6, "end_line": 12}),
        _call("r", "report", report),
    ]
    return {
        "repo": str(REPO),
        "chat_script": script,
        "backfill_to": 0,
    }  # fake-store top-k ties differ per process


def dump(model: Any) -> Any:
    return json.loads(model.model_dump_json(by_alias=True))


async def run_both(case: dict[str, Any], tmp: Path, sync: bool, op: str, req: Any) -> tuple[Any, Any]:
    out: list[Any] = []
    for cls in (SdkDriver, SdkOverHttpDriver):
        d: Driver = cls()
        await d.setup(case, tmp / cls.__name__)
        if sync:
            await d.sync()
        out.append(dump(await getattr(d, op)(req)))
    return out[0], out[1]


@pytest.mark.parametrize("sync", [True, False], ids=["ready", "absent"])
async def test_explore_equal_over_http(tmp_path: Path, sync: bool) -> None:
    schema = json.loads((ROOT / "schema" / "v1" / "ExploreResult.json").read_text())
    a, b = await run_both(
        _case(semantic=sync), tmp_path, sync, "explore", ExploreRequest(brief="where is the session lookup?")
    )
    assert a == b
    jsonschema.validate(b, schema)
    assert b["index"]["state"] == ("ready" if sync else "absent")


async def test_primitives_equal_over_http(tmp_path: Path) -> None:
    case = _case(semantic=True)
    for op, req in (
        ("grep", GrepRequest(pattern=IDENT)),
        ("read_range", ReadRangeRequest(path=SESSION, start_line=1, end_line=12)),
        ("index_status", IndexStatusRequest()),
        ("semantic_search", SemanticSearchRequest(query=f"{IDENT} session lookup", k=5)),
    ):
        a, b = await run_both(case, tmp_path / op, True, op, req)
        assert a == b, op
    a, b = await run_both(case, tmp_path / "status0", False, "index_status", IndexStatusRequest())
    assert a == b
    assert b["state"] == "absent"


async def test_path_refused_maps_the_same_over_http(tmp_path: Path) -> None:
    codes: list[ErrorCode] = []
    for cls in (SdkDriver, SdkOverHttpDriver):
        d = cls()
        await d.setup(_case(semantic=False), tmp_path / cls.__name__)
        with pytest.raises(ScoutError) as ei:
            await d.read_range(ReadRangeRequest(path="../etc/passwd"))
        codes.append(ei.value.code)
    assert codes[0] == codes[1] == ErrorCode.PATH_REFUSED
