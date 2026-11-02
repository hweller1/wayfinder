"""Wire models and JSON Schema export (design sections 4, 6, 8). Covers AC-1.1, FR-C1, FR-G1."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from scout.core.contracts.schema import SCHEMA_VERSION, StatusFile, export_schemas, render_schemas
from scout.core.contracts.wire import (
    ChatReq,
    CommitReq,
    EmbedReq,
    NodesReq,
    PlanReq,
    SearchReq,
    SeedReq,
    StatusReq,
    SyncBeginReq,
    WireRow,
)

REPO = Path(__file__).resolve().parents[2]
GOOD = "a" * 16
GOOD_32 = "0123456789abcdef" * 2


def _row(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "r1",
        "file_id": GOOD,
        "path_enc": "ab/cd",
        "ext_tag": "e1",
        "dir_tags": ["d1"],
        "file_hash": "f" * 64,
        "chunk_key": GOOD,
        "loc_enc": "bG9j",
        "tokens": ["t1"],
    }
    base.update(over)
    return base


@pytest.mark.parametrize("bad", ["", "abc", "A" * 16, "g" * 16, "a" * 33, "../etc/passwd" + "a" * 16])
def test_ids_reject_bad_values(bad: str) -> None:
    """FR-G1: repo_id/branch_id must match ^[0-9a-f]{16,32}$."""
    with pytest.raises(ValidationError):
        StatusReq(repo_id=bad, branch_id=GOOD)
    with pytest.raises(ValidationError):
        StatusReq(repo_id=GOOD, branch_id=bad)


@pytest.mark.parametrize("ok", [GOOD, GOOD_32, "0" * 16])
def test_ids_accept_valid_values(ok: str) -> None:
    assert StatusReq(repo_id=ok, branch_id=ok).repo_id == ok


def test_nodes_list_cap_500() -> None:
    NodesReq(repo_id=GOOD, branch_id=GOOD, ids=[GOOD] * 500)
    with pytest.raises(ValidationError):
        NodesReq(repo_id=GOOD, branch_id=GOOD, ids=[GOOD] * 501)


def test_plan_list_cap_2000_and_id_pattern() -> None:
    PlanReq(repo_id=GOOD, branch_id=GOOD, chunk_keys=[GOOD] * 2000)
    with pytest.raises(ValidationError):
        PlanReq(repo_id=GOOD, branch_id=GOOD, chunk_keys=[GOOD] * 2001)
    with pytest.raises(ValidationError):
        PlanReq(repo_id=GOOD, branch_id=GOOD, chunk_keys=["not-hex"])


def test_embed_rejects_oversize_rows_and_bad_text_keys() -> None:
    ok = EmbedReq.model_validate({"repo_id": GOOD, "branch_id": GOOD, "texts": {GOOD: "x"}, "rows": [_row()]})
    assert ok.rows[0].chunk_key == GOOD
    with pytest.raises(ValidationError):
        EmbedReq.model_validate({"repo_id": GOOD, "branch_id": GOOD, "texts": {}, "rows": [_row()] * 2001})
    with pytest.raises(ValidationError):
        EmbedReq.model_validate({"repo_id": GOOD, "branch_id": GOOD, "texts": {"zz": "x"}, "rows": []})


def test_wire_row_caps_tokens_at_256() -> None:
    WireRow.model_validate(_row(tokens=["t"] * 256))
    with pytest.raises(ValidationError):
        WireRow.model_validate(_row(tokens=["t"] * 257))


def test_requests_forbid_extra_fields() -> None:
    """Requests are extra=forbid so clients cannot smuggle store attribute names."""
    with pytest.raises(ValidationError):
        StatusReq.model_validate({"repo_id": GOOD, "branch_id": GOOD, "namespace": "scout_x"})
    with pytest.raises(ValidationError):
        SearchReq.model_validate(
            {"repo_id": GOOD, "branch_id": GOOD, "query": "q", "scope": {"kind": "node"}}
        )


def test_search_defaults_and_k_bound() -> None:
    req = SearchReq(repo_id=GOOD, branch_id=GOOD, query="q")
    assert (req.k, req.overfetch, req.tokens) == (10, 1, [])
    with pytest.raises(ValidationError):
        SearchReq(repo_id=GOOD, branch_id=GOOD, query="q", k=31)


def test_seed_proofs_cap_and_roundtrip() -> None:
    req = SeedReq(repo_id=GOOD, branch_id=GOOD, minhash_b64="AAAA", proofs=["p"] * 5000)
    assert SeedReq.model_validate_json(req.model_dump_json()) == req
    with pytest.raises(ValidationError):
        SeedReq(repo_id=GOOD, branch_id=GOOD, minhash_b64="AAAA", proofs=["p"] * 5001)


def test_commit_and_sync_begin_validate() -> None:
    CommitReq(repo_id=GOOD, branch_id=GOOD, new_root="r", written_chunks=0)
    with pytest.raises(ValidationError):
        CommitReq(repo_id=GOOD, branch_id=GOOD, new_root="r", written_chunks=-1)
    with pytest.raises(ValidationError):
        SyncBeginReq(repo_id=GOOD, branch_id=GOOD, root="r", expected_chunks=0, base_branch_id="nope")


def test_chat_req_defaults_match_chat_port() -> None:
    req = ChatReq(messages=[{"role": "user", "content": "hi"}])
    assert (req.temperature, req.top_p, req.top_k, req.presence_penalty) == (0.85, 0.95, 20, 1.1)
    assert req.thinking is True
    assert req.tool_choice == "auto"
    with pytest.raises(ValidationError):
        ChatReq(messages=[])


def test_status_file_roundtrips_with_schema_alias() -> None:
    """Design 8: status.json carries `schema`, which a stdlib-only hook reads."""
    sf = StatusFile(state="ready", coverage=0.9, commit="abc", syncing=False)
    dumped = json.loads(sf.model_dump_json())
    assert dumped["schema"] == SCHEMA_VERSION
    assert "schema_" not in dumped
    assert StatusFile.model_validate(dumped) == sf
    with pytest.raises(ValidationError):
        StatusFile(state="ready", coverage=1.5)
    with pytest.raises(ValidationError):
        StatusFile.model_validate({"schema": "2.0", "state": "ready", "coverage": 1.0})


def test_render_schemas_is_deterministic_and_complete() -> None:
    a, b = render_schemas(), render_schemas()
    assert a == b
    index = json.loads(a["index.json"])
    assert index["schema_version"] == SCHEMA_VERSION
    assert index["files"] == sorted(n for n in a if n != "index.json")
    for required in ("ExploreRequest.json", "ExploreResult.json", "Error.json", "status_file.json"):
        assert required in a
    for text in a.values():
        json.loads(text)


def test_export_schemas_writes_files(tmp_path: Path) -> None:
    paths = export_schemas(tmp_path / "out")
    assert {p.name for p in paths} == set(render_schemas())
    assert all(p.read_text(encoding="utf-8") == render_schemas()[p.name] for p in paths)


def _gen(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(REPO / "scripts" / "gen_schema.py"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_committed_schema_has_no_drift() -> None:
    """The committed schema/v1 matches the models (drift gate)."""
    res = _gen("--check")
    assert res.returncode == 0, res.stderr


def test_drift_gate_fails_on_modified_and_missing_files(tmp_path: Path) -> None:
    """The gate must fail when a committed file differs, is missing, or is stale."""
    work = tmp_path / "v1"
    assert _gen("--dir", str(work)).returncode == 0
    assert _gen("--check", "--dir", str(work)).returncode == 0
    target = work / "Hit.json"
    original = target.read_text(encoding="utf-8")
    target.write_text(original.replace('"type": "object"', '"type": "array"', 1), encoding="utf-8")
    res = _gen("--check", "--dir", str(work))
    assert res.returncode != 0
    assert "Hit.json" in res.stderr
    target.write_text(original, encoding="utf-8")
    target.unlink()
    assert _gen("--check", "--dir", str(work)).returncode != 0
    target.write_text(original, encoding="utf-8")
    (work / "Old.json").write_text("{}\n", encoding="utf-8")
    assert _gen("--check", "--dir", str(work)).returncode != 0
