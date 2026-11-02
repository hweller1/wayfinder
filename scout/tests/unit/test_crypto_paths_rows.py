"""Crypto paths and row sealing (design section 7): AC-12.3, FR-X1, FR-X3, FR-X4."""

import base64
import json

import pytest

from scout.core.contracts.wire import WireHit, WireRow
from scout.core.crypto import ids, keys, paths, rows
from scout.core.crypto.keys import CryptoError

SK = keys.derive(bytes(range(64)))
OTHER = keys.derive(bytes(range(1, 65)))
TENANT = "acme"
REPO = ids.repo_id(SK, "a" * 40, "github.com/acme/widgets")
OTHER_REPO = ids.repo_id(SK, "b" * 40, "github.com/acme/other")
CK = "ab" * 16
PATH = "src/auth/session_store.py"


def _row(path: str = PATH, start: int = 17, end: int = 4242, ck: str = CK, tokens: list[str] | None = None):
    return rows.seal_chunk_row(
        SK,
        TENANT,
        REPO,
        path=path,
        start=start,
        end=end,
        chunk_key=ck,
        occurrence=0,
        file_hash="cd" * 32,
        tokens=tokens if tokens is not None else ["ee" * 16],
    )


def _hit(r: WireRow) -> WireHit:
    return WireHit(path_enc=r.path_enc, loc_enc=r.loc_enc, chunk_key=r.chunk_key, file_hash=r.file_hash)


def _flip(b64: str, byte_index: int) -> str:
    raw = bytearray(base64.b64decode(b64))
    raw[byte_index] ^= 0x01
    return base64.b64encode(bytes(raw)).decode()


# ---- paths ----


@pytest.mark.parametrize(
    "path",
    [
        "a",
        "src/auth/s.py",
        "dir with space/f g.txt",
        "ünïcode/日本語/файл.py",
        "a.b.c/.hidden/..x/y.tar.gz",
        "é/😀.md",
    ],
)
def test_path_round_trip(path: str):
    enc = paths.enc_path(SK, TENANT, REPO, path)
    assert paths.dec_path(SK, TENANT, REPO, enc) == path
    assert len(enc.split("/")) == len(path.split("/"))
    assert all(set(s) <= set("0123456789abcdef") for s in enc.split("/"))


def test_path_encryption_is_deterministic():
    assert paths.enc_path(SK, TENANT, REPO, PATH) == paths.enc_path(SK, TENANT, REPO, PATH)


def test_shared_parent_shares_first_segment_by_design():
    """The documented PRD leak: per-segment SIV means `a/b` and `a/c` share enc(a)."""
    ab = paths.enc_path(SK, TENANT, REPO, "a/b").split("/")
    ac = paths.enc_path(SK, TENANT, REPO, "a/c").split("/")
    assert ab[0] == ac[0]
    assert ab[1] != ac[1]


def test_same_segment_name_in_different_dirs_differs():
    x = paths.enc_path(SK, TENANT, REPO, "src/util.py").split("/")[1]
    y = paths.enc_path(SK, TENANT, REPO, "lib/util.py").split("/")[1]
    z = paths.enc_path(SK, TENANT, REPO, "src/deep/util.py").split("/")[2]
    assert len({x, y, z}) == 3


def test_path_aad_binds_tenant_repo_and_key():
    enc = paths.enc_path(SK, TENANT, REPO, PATH)
    assert paths.enc_path(SK, "other", REPO, PATH) != enc
    assert paths.enc_path(SK, TENANT, OTHER_REPO, PATH) != enc
    assert paths.enc_path(OTHER, TENANT, REPO, PATH) != enc
    for args in ((SK, "other", REPO), (SK, TENANT, OTHER_REPO), (OTHER, TENANT, REPO)):
        with pytest.raises(CryptoError):
            paths.dec_path(*args, enc)


def test_path_segment_moved_to_other_parent_fails():
    a = paths.enc_path(SK, TENANT, REPO, "x/f.py").split("/")
    b = paths.enc_path(SK, TENANT, REPO, "y/f.py").split("/")
    with pytest.raises(CryptoError):
        paths.dec_path(SK, TENANT, REPO, f"{a[0]}/{b[1]}")
    with pytest.raises(CryptoError):
        paths.dec_path(SK, TENANT, REPO, b[1])  # leaf presented without its parent


def test_path_tamper_raises():
    seg = paths.enc_path(SK, TENANT, REPO, "only.py")
    raw = bytearray(bytes.fromhex(seg))
    for i in (0, 8, len(raw) - 1):  # SIV tag prefix, tag, ciphertext tail
        bad = bytearray(raw)
        bad[i] ^= 0x01
        with pytest.raises(CryptoError):
            paths.dec_path(SK, TENANT, REPO, bad.hex())


@pytest.mark.parametrize("bad", ["", "zz", "abc", "00" * 4, "ab//cd", "/ab", "AB" * 20, "x" * 40])
def test_dec_path_rejects_malformed(bad: str):
    with pytest.raises(CryptoError):
        paths.dec_path(SK, TENANT, REPO, bad)


@pytest.mark.parametrize("bad", ["", "/a", "a/", "a//b", "a/./b", "../a", "a/..", "a\0b"])
def test_enc_path_rejects_bad_paths(bad: str):
    with pytest.raises(ValueError, match="path"):
        paths.enc_path(SK, TENANT, REPO, bad)


def test_prefix_glob_matches_whole_segment_prefix():
    glob = paths.enc_prefix_glob(SK, TENANT, REPO, "src/auth")
    assert glob == paths.enc_path(SK, TENANT, REPO, "src/auth") + "/**"
    assert paths.enc_path(SK, TENANT, REPO, PATH).startswith(glob[:-2])
    assert "*" not in glob[:-2]
    assert paths.enc_prefix_glob(SK, TENANT, REPO, "") == "**"


def test_ext_and_ancestors():
    assert paths.path_ext("a/b/c.tar.gz") == ".gz"
    assert paths.path_ext("a/.env") == ""
    assert paths.path_ext("Makefile") == ""
    assert paths.ancestors("a/b/c.py") == ["a", "a/b"]
    assert paths.ancestors("c.py") == []


# ---- rows ----


def test_seal_chunk_row_shape_and_round_trip():
    r = _row()
    assert isinstance(r, WireRow)
    assert r.file_id == ids.file_id(SK, PATH)
    assert r.id == ids.row_id(SK, r.file_id, CK, 0)
    assert r.ext_tag == ids.ext_tag(SK, ".py")
    assert r.dir_tags == [ids.dir_tag(SK, "src"), ids.dir_tag(SK, "src/auth")]
    assert r.path_enc == paths.enc_path(SK, TENANT, REPO, PATH)
    assert rows.open_hit(SK, TENANT, REPO, _hit(r)) == (PATH, 17, 4242)


def test_loc_enc_uses_fresh_nonce_each_seal():
    a, b = _row(), _row()
    assert a.loc_enc != b.loc_enc
    assert base64.b64decode(a.loc_enc)[:12] != base64.b64decode(b.loc_enc)[:12]
    assert a.path_enc == b.path_enc  # only path is deterministic
    assert (
        len(base64.b64decode(a.loc_enc))
        == 12 + len(json.dumps({"s": 17, "e": 4242}, separators=(",", ":"))) + 16
    )


def test_tokens_cap_256_enforced_at_seal():
    assert len(_row(tokens=[f"{i:032x}" for i in range(256)]).tokens) == 256
    with pytest.raises(ValueError, match="256"):
        _row(tokens=[f"{i:032x}" for i in range(257)])


@pytest.mark.parametrize(("start", "end"), [(0, 5), (5, 4), (-1, 3)])
def test_seal_rejects_bad_range(start: int, end: int):
    with pytest.raises(ValueError, match="start"):
        _row(start=start, end=end)


def test_ac_12_3_each_aad_component_swap_fails_to_open():
    r = _row()
    good = _hit(r)
    assert rows.open_hit(SK, TENANT, REPO, good)[0] == PATH
    with pytest.raises(CryptoError):  # other tenant
        rows.open_hit(SK, "other", REPO, good)
    with pytest.raises(CryptoError):  # other repo
        rows.open_hit(SK, TENANT, OTHER_REPO, good)
    with pytest.raises(CryptoError):  # other chunk_key
        rows.open_hit(SK, TENANT, REPO, good.model_copy(update={"chunk_key": "cd" * 16}))
    other = _row(path="src/auth/other.py", start=1, end=2)
    with pytest.raises(CryptoError):  # loc moved onto another file's row (path_enc swapped)
        rows.open_hit(SK, TENANT, REPO, good.model_copy(update={"path_enc": other.path_enc}))
    with pytest.raises(CryptoError):  # loc copied from another row onto this row
        rows.open_hit(SK, TENANT, REPO, good.model_copy(update={"loc_enc": other.loc_enc}))
    with pytest.raises(CryptoError):  # wrong key
        rows.open_hit(OTHER, TENANT, REPO, good)


def test_aad_fields_are_length_prefixed_not_concatenated():
    """tenant 'ab' + repo 'c...' must not collide with tenant 'a' + repo 'bc...'."""
    assert rows._aad(b"loc", "ab", "c") != rows._aad(b"loc", "a", "bc")  # pyright: ignore[reportPrivateUsage]
    assert rows._aad(b"loc", "x") != rows._aad(b"node", "x")  # pyright: ignore[reportPrivateUsage]


def test_loc_blob_tamper_raises_in_nonce_ciphertext_and_tag():
    r = _row()
    n = len(base64.b64decode(r.loc_enc))
    for i in (0, 11, 12, n - 17, n - 16, n - 1):  # nonce, ciphertext, tag
        bad = _hit(r).model_copy(update={"loc_enc": _flip(r.loc_enc, i)})
        with pytest.raises(CryptoError):
            rows.open_hit(SK, TENANT, REPO, bad)


@pytest.mark.parametrize("blob", ["", "!!!notb64", "AAAA", base64.b64encode(b"x" * 27).decode()])
def test_loc_malformed_or_truncated_raises(blob: str):
    r = _row()
    with pytest.raises(CryptoError):
        rows.open_hit(SK, TENANT, REPO, _hit(r).model_copy(update={"loc_enc": blob}))
    raw = base64.b64decode(r.loc_enc)
    short = base64.b64encode(raw[:-1]).decode()
    with pytest.raises(CryptoError):
        rows.open_hit(SK, TENANT, REPO, _hit(r).model_copy(update={"loc_enc": short}))


def test_sealed_row_has_no_plaintext_path_line_or_identifier():
    path = "internal_secrets/payroll_service/SalaryCalculator.py"
    ident_tokens = ["ee" * 16, "ff" * 16]  # tokens arrive pre-HMAC'd; plaintext identifiers never enter
    r = _row(path=path, start=3133731, end=4567890, tokens=ident_tokens)
    wire = r.model_dump_json().encode()
    # decode every base64 blob too, so a plaintext JSON range hiding in the blob would also be found
    scan = wire + base64.b64decode(r.loc_enc)
    needles = [
        b"internal_secrets",
        b"payroll_service",
        b"SalaryCalculator",
        b"salarycalculator",
        b"Salary",
        b"3133731",
        b"4567890",
        b'"s":',
        b'"e":',
    ]
    for needle in needles:
        assert needle not in scan, needle
    for seg in path.split("/"):
        assert seg.encode() not in scan
    assert b".py" not in scan


# ---- node blobs ----

NODE = ids.node_id(SK, REPO, "src/auth")
CHILDREN = [("session_store.py", "file", "11" * 32), ("ünï dir", "dir", "22" * 32)]


def test_node_round_trip_and_fresh_nonce():
    a = rows.seal_node(SK, TENANT, REPO, NODE, CHILDREN)
    b = rows.seal_node(SK, TENANT, REPO, NODE, CHILDREN)
    assert a != b
    assert rows.open_node(SK, TENANT, REPO, NODE, a) == CHILDREN
    assert b"session_store" not in base64.b64decode(a)


def test_node_each_aad_component_swap_fails():
    blob = rows.seal_node(SK, TENANT, REPO, NODE, CHILDREN)
    other_node = ids.node_id(SK, REPO, "src/other")
    with pytest.raises(CryptoError):
        rows.open_node(SK, "other", REPO, NODE, blob)
    with pytest.raises(CryptoError):
        rows.open_node(SK, TENANT, OTHER_REPO, NODE, blob)
    with pytest.raises(CryptoError):
        rows.open_node(SK, TENANT, REPO, other_node, blob)
    with pytest.raises(CryptoError):
        rows.open_node(OTHER, TENANT, REPO, NODE, blob)


def test_node_blob_cannot_open_as_loc_and_vice_versa():
    """Domain separation: same key (K_loc), different AAD domain."""
    r = _row()
    with pytest.raises(CryptoError):
        rows.open_node(SK, TENANT, REPO, r.chunk_key, r.loc_enc)
    blob = rows.seal_node(SK, TENANT, REPO, CK, CHILDREN)
    with pytest.raises(CryptoError):
        rows.open_hit(SK, TENANT, REPO, _hit(r).model_copy(update={"loc_enc": blob}))


def test_node_tamper_raises():
    blob = rows.seal_node(SK, TENANT, REPO, NODE, CHILDREN)
    n = len(base64.b64decode(blob))
    for i in (0, 11, 12, n - 17, n - 16, n - 1):
        with pytest.raises(CryptoError):
            rows.open_node(SK, TENANT, REPO, NODE, _flip(blob, i))
    with pytest.raises(CryptoError):
        rows.open_node(SK, TENANT, REPO, NODE, "")
