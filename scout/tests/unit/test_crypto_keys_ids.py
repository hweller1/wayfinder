"""Crypto keys and ids (design section 7): AC-12.2, FR-X2, FR-C15.

Known-answer tests: the pinned hex values below were generated once from this implementation with a fixed
`K_team = bytes(range(64))`. The six HKDF subkeys were additionally cross-checked against an independent
RFC 5869 implementation (stdlib `hmac`) and every id against a direct stdlib HMAC over the documented
message. An accidental change to a salt, info label, message prefix or truncation breaks these tests.
"""

import hashlib
import hmac
import importlib
import inspect
import pkgutil
import re

import pytest

import scout.core.crypto as crypto_pkg
from scout.core.crypto import ids, keys
from scout.core.crypto.keys import CryptoError, Subkeys

K_TEAM = bytes(range(64))
SK = keys.derive(K_TEAM)
ROOT_COMMIT = "a" * 40
REMOTE = "github.com/acme/widgets"
CLONE = "0123456789ab"
REPO = ids.repo_id(SK, ROOT_COMMIT, REMOTE)
BRANCH = ids.branch_id(SK, REPO, CLONE, "main")
CHUNK_KEY = "ab" * 16
NS_RE = re.compile(r"^[A-Za-z0-9\-_.]{1,128}$")  # turbopuffer namespace charset and length

KAT_SUBKEYS = {
    "path": "52793d0e22e9ead9aac1315d8588db7dc9854d00a647047aa8a7da144abfcc9d"
    "062b103b6b96a544e2034e3a0d5ad113e3c4163d115aaf21f8ba284294ff872d",
    "loc": "c874109de533fad89dc13e1afa8d54eac70bd5e3f3a7538975cadb1d0222c37c",
    "leaf": "427538e4bf9f1ccdc25c6aa054704b3b9002175344324169a63c9ff880334bcc",
    "chunk": "5c39f5ae3263f8617acebc4390c9fee649dab883128bb0e195b2bdba2a7f3ded",
    "token": "56cfe1f26860ca707d09c0c280fc2d9d7c150fa4f08f5b732e41f8d6ead5170e",
    "id": "c0e849f7aeaf6e51f3ddfeefebe4328485fd03e2fdc9eb661b2c705221404f5e",
}


def _mac(msg: bytes) -> str:
    return hmac.new(SK.id, msg, hashlib.sha256).hexdigest()


def test_derive_known_answers():
    for name, hexval in KAT_SUBKEYS.items():
        assert getattr(SK, name).hex() == hexval, name


def test_derive_sizes_and_six_distinct_subkeys():
    """AC-12.2: six distinct purpose keys; none equals K_team (or a prefix of it)."""
    vals = [getattr(SK, f) for f in ("path", "loc", "leaf", "chunk", "token", "id")]
    assert len(set(vals)) == 6
    assert [len(v) for v in vals] == [64, 32, 32, 32, 32, 32]
    assert all(v != K_TEAM and v != K_TEAM[: len(v)] for v in vals)


def test_derive_deterministic_and_depends_on_team_key():
    assert keys.derive(K_TEAM) == SK
    other = keys.derive(bytes(reversed(range(64))))
    assert all(getattr(other, f) != getattr(SK, f) for f in KAT_SUBKEYS)


def test_derive_rejects_wrong_size_and_repr_hides_bytes():
    for bad in (b"", b"x" * 63, b"x" * 65, b"x" * 32):
        with pytest.raises(CryptoError):
            keys.derive(bad)
    assert KAT_SUBKEYS["id"] not in repr(SK)
    assert repr(SK) == "Subkeys()"


def test_new_team_key_is_64_random_bytes():
    a, b = keys.new_team_key(), keys.new_team_key()
    assert len(a) == len(b) == 64
    assert a != b


def test_wrap_round_trip_and_fresh_nonce():
    master = bytes(range(100, 132))
    w1 = keys.wrap(master, K_TEAM, b"acme")
    w2 = keys.wrap(master, K_TEAM, b"acme")
    assert len(w1) == 12 + 64 + 16
    assert w1[:12] != w2[:12]
    assert w1 != w2
    assert keys.unwrap(master, w1, b"acme") == K_TEAM
    assert keys.unwrap(master, w2, b"acme") == K_TEAM


def test_unwrap_known_answer():
    """Wire-format pin: a blob wrapped once (nonce 0x00*12 prefix built by hand) still opens."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    master = bytes(range(100, 132))
    nonce = bytes(12)
    blob = nonce + AESGCM(master).encrypt(nonce, K_TEAM, b"acme")
    assert keys.unwrap(master, blob, b"acme") == K_TEAM


def test_wrap_aad_mismatch_fails():
    master = bytes(range(100, 132))
    blob = keys.wrap(master, K_TEAM, b"acme")
    with pytest.raises(CryptoError):
        keys.unwrap(master, blob, b"other-tenant")


def test_unwrap_wrong_master_tamper_and_short_blob_fail():
    master = bytes(range(100, 132))
    blob = keys.wrap(master, K_TEAM, b"acme")
    with pytest.raises(CryptoError):
        keys.unwrap(bytes(32), blob, b"acme")
    for i in (0, 12, len(blob) - 1):
        bad = bytearray(blob)
        bad[i] ^= 1
        with pytest.raises(CryptoError):
            keys.unwrap(master, bytes(bad), b"acme")
    for short in (b"", b"x" * 12):
        with pytest.raises(CryptoError):
            keys.unwrap(master, short, b"acme")


def test_wrap_rejects_bad_key_sizes():
    with pytest.raises(CryptoError):
        keys.wrap(b"x" * 31, K_TEAM, b"acme")
    with pytest.raises(CryptoError):
        keys.wrap(bytes(32), b"x" * 32, b"acme")
    with pytest.raises(CryptoError):
        keys.unwrap(b"x" * 31, b"y" * 100, b"acme")


def _crypto_functions():
    for mod_info in pkgutil.iter_modules(crypto_pkg.__path__, crypto_pkg.__name__ + "."):
        mod = importlib.import_module(mod_info.name)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ == mod.__name__:
                yield mod.__name__, name, inspect.signature(fn)


def test_only_derive_wrap_unwrap_accept_k_team():
    """AC-12.2: K_team never reaches any other primitive. Scans every module under core/crypto."""
    allowed = {
        ("scout.core.crypto.keys", "derive"),
        ("scout.core.crypto.keys", "wrap"),
        ("scout.core.crypto.keys", "unwrap"),
        ("scout.core.crypto.keys", "_hkdf"),  # private helper called only by derive
    }
    seen = set()
    for mod, name, sig in _crypto_functions():
        seen.add((mod, name))
        if (mod, name) in allowed:
            continue
        for pname, p in sig.parameters.items():
            assert not re.search(r"team|master", pname, re.I), f"{mod}.{name}({pname}) takes a root key"
            assert p.annotation is not bytes or pname != "key", f"{mod}.{name} takes a raw key"
    assert allowed <= seen
    # id functions take Subkeys (never raw bytes) as the key argument
    for name in ("file_id", "repo_id", "branch_id", "row_id", "node_id", "ext_tag", "dir_tag"):
        first = next(iter(inspect.signature(getattr(ids, name)).parameters.values()))
        assert first.name == "keys"
        assert first.annotation in (Subkeys, "Subkeys")


def test_id_known_answers():
    fid = ids.file_id(SK, "src/auth/session.py")
    assert REPO == "60a56fd049aa82107a9687ef2248d82b"
    assert fid == "28dd115a823b9cad9d155c85d35f7323"
    assert BRANCH == "7031b1c02a5691e1"
    assert ids.branch_id(SK, REPO, CLONE, ids.detached_name("f" * 40)) == "6be2b65fd5e24b79"
    assert ids.row_id(SK, fid, CHUNK_KEY, 0) == "172f6b4ae31922516911bc95b475f5aa"
    assert ids.node_id(SK, REPO, "src/auth") == "d8a5bd946e5155063c05dcc064d06623"
    assert ids.ext_tag(SK, ".py") == "4e6692ee8c47e471"
    assert ids.dir_tag(SK, "src/auth") == "af6746b9a52d0ed8"
    expected_ns = "scout_acme_60a56fd049aa82107a9687ef2248d82b_7031b1c02a5691e1"
    assert ids.namespace("acme", REPO, BRANCH) == expected_ns


def test_ids_match_documented_messages():
    """Each id is exactly HMAC(K_id, prefix + parts)[:n] per design section 7 (independent of ids.py)."""
    fid = ids.file_id(SK, "src/a.py")
    assert fid == _mac(b"file\0src/a.py")[:32]
    assert _mac(b"repo\0" + ROOT_COMMIT.encode() + b"\0" + REMOTE.encode())[:32] == REPO
    assert _mac(b"branch\0" + REPO.encode() + b"\0" + CLONE.encode() + b"\0main")[:16] == BRANCH
    assert ids.row_id(SK, fid, CHUNK_KEY, 3) == _mac(b"row\0" + fid.encode() + CHUNK_KEY.encode() + b"3")[:32]
    assert ids.node_id(SK, REPO, "src") == _mac(b"node\0" + REPO.encode() + b"src")[:32]
    assert ids.ext_tag(SK, ".go") == _mac(b"ext\0.go")[:16]
    assert ids.dir_tag(SK, "src") == _mac(b"dir\0src")[:16]


def test_id_lengths_and_hex():
    fid = ids.file_id(SK, "x")
    for value, n in (
        (fid, 32),
        (REPO, 32),
        (BRANCH, 16),
        (ids.row_id(SK, fid, CHUNK_KEY, 0), 32),
        (ids.node_id(SK, REPO, "d"), 32),
        (ids.ext_tag(SK, ".py"), 16),
        (ids.dir_tag(SK, "d"), 16),
    ):
        assert re.fullmatch(rf"[0-9a-f]{{{n}}}", value)


def test_domain_separation_same_input_different_functions():
    """Distinct message prefixes: one input string under different id functions never collides."""
    x = "src"
    outs = {
        "file": ids.file_id(SK, x)[:16],
        "ext": ids.ext_tag(SK, x),
        "dir": ids.dir_tag(SK, x),
        "node": ids.node_id(SK, REPO, x)[:16],
    }
    assert len(set(outs.values())) == 4
    assert ids.ext_tag(SK, "x") != ids.dir_tag(SK, "x")
    assert ids.file_id(SK, "x") != ids.node_id(SK, "0" * 32, "x")
    # the full-length MACs differ too (not just the truncations)
    assert len({_mac(b"file\0x"), _mac(b"ext\0x"), _mac(b"dir\0x")}) == 3
    # file_id of a path that spells another prefix is still distinct
    assert ids.file_id(SK, "ext\0x".replace("\0", "/")) != ids.ext_tag(SK, "x")


def test_ids_key_separation():
    other = keys.derive(bytes(reversed(range(64))))
    assert ids.file_id(other, "x") != ids.file_id(SK, "x")
    assert ids.repo_id(other, ROOT_COMMIT, REMOTE) != REPO


def test_determinism():
    assert ids.repo_id(SK, ROOT_COMMIT, REMOTE) == REPO
    assert ids.branch_id(SK, REPO, CLONE, "main") == BRANCH
    assert ids.file_id(SK, "a/b.py") == ids.file_id(SK, "a/b.py")
    assert ids.file_id(SK, "a/b.py") != ids.file_id(SK, "a/c.py")


def test_different_clone_id_different_branch_id():
    """FR-C15: teammates sharing K_team get distinct namespaces for the same branch name."""
    assert ids.branch_id(SK, REPO, "ba9876543210", "main") != BRANCH
    assert ids.branch_id(SK, REPO, CLONE, "dev") != BRANCH


def test_repo_id_depends_on_commit_and_remote():
    assert ids.repo_id(SK, "b" * 40, REMOTE) != REPO
    assert ids.repo_id(SK, ROOT_COMMIT, "github.com/acme/other") != REPO
    # field boundary is unambiguous
    assert ids.repo_id(SK, "ab", "c") != ids.repo_id(SK, "a", "bc")


def test_row_id_occurrence_distinguishes_duplicate_chunks():
    fid = ids.file_id(SK, "x")
    assert ids.row_id(SK, fid, CHUNK_KEY, 0) != ids.row_id(SK, fid, CHUNK_KEY, 1)
    with pytest.raises(ValueError, match="occurrence"):
        ids.row_id(SK, fid, CHUNK_KEY, -1)


def test_detached_head_form():
    name = ids.detached_name("0123456789abcdef0123")
    assert name == "detached-0123456789ab"
    assert ids.branch_id(SK, REPO, CLONE, name) != BRANCH


def test_clone_id_is_12_hex_random():
    a, b = ids.new_clone_id(), ids.new_clone_id()
    assert re.fullmatch(r"[0-9a-f]{12}", a)
    assert a != b


def test_nul_and_bad_shapes_rejected():
    with pytest.raises(ValueError, match="NUL"):
        ids.file_id(SK, "a\0b")
    with pytest.raises(ValueError, match="NUL"):
        ids.branch_id(SK, REPO, CLONE, "m\0ain")
    with pytest.raises(ValueError, match="clone_id"):
        ids.branch_id(SK, REPO, "short", "main")
    with pytest.raises(ValueError, match="repo_id"):
        ids.branch_id(SK, "XYZ", CLONE, "main")
    with pytest.raises(ValueError, match="repo_id"):
        ids.node_id(SK, "nothex", "d")
    with pytest.raises(ValueError, match="chunk_key"):
        ids.row_id(SK, ids.file_id(SK, "x"), "zz", 0)


def test_namespace_regex_and_shape():
    ns = ids.namespace("acme", REPO, BRANCH)
    assert ns == f"scout_acme_{REPO}_{BRANCH}"
    assert NS_RE.match(ns)
    tns = ids.tree_namespace(ns)
    assert tns == ns + "__tree"
    assert NS_RE.match(tns)


def test_namespace_rejects_bad_inputs():
    for bad in ("", "a b", "a/b", "a.b", "a\0", "a_b", "x" * (ids.MAX_TENANT_LEN + 1)):
        with pytest.raises(ValueError, match="tenant"):
            ids.namespace(bad, REPO, BRANCH)
    with pytest.raises(ValueError, match="repo_id"):
        ids.namespace("acme", "A" * 32, BRANCH)
    with pytest.raises(ValueError, match="branch_id"):
        ids.namespace("acme", REPO, BRANCH[:-1])


def test_namespace_boundary_tree_suffix_fits_128():
    """Longest tenant: chunk ns is 122 chars and its `__tree` sibling is exactly 128 (the tpuf maximum)."""
    tenant = "t" * ids.MAX_TENANT_LEN
    ns = ids.namespace(tenant, REPO, BRANCH)
    tns = ids.tree_namespace(ns)
    assert len(ns) == 122
    assert len(tns) == 128
    assert NS_RE.match(ns)
    assert NS_RE.match(tns)
    with pytest.raises(ValueError, match="tenant"):
        ids.namespace(tenant + "t", REPO, BRANCH)
    with pytest.raises(ValueError, match="valid turbopuffer name"):
        ids.tree_namespace("s" * 123)  # would be 129


def test_no_namespace_ends_in_tree_suffix_and_tree_is_never_a_namespace():
    for tenant in ("acme", "a", "a-tree", "x-" * 10, "t" * ids.MAX_TENANT_LEN):
        for branch_name in ("main", "__tree", "x__tree", "detached-abc"):
            br = ids.branch_id(SK, REPO, CLONE, branch_name)
            ns = ids.namespace(tenant, REPO, br)
            assert not ns.endswith("__tree")
            assert re.search(r"_[0-9a-f]{16}$", ns)
            assert ids.tree_namespace(ns) != ns
    with pytest.raises(ValueError, match="already"):
        ids.tree_namespace(ids.tree_namespace(ids.namespace("acme", REPO, BRANCH)))
    # a tree namespace cannot be produced as a chunk namespace
    with pytest.raises(ValueError, match="branch_id"):
        ids.namespace("acme", REPO, "__tree")


def test_tenant_prefix_scan_cannot_reach_another_tenant():
    """Tenant names exclude "_", so `scout_{a}_` never prefixes a namespace of a different tenant."""
    for a, b in (("acme", "acme-x"), ("acme", "acmex"), ("a", "a-b")):
        ns_b = ids.namespace(b, REPO, BRANCH)
        assert not ns_b.startswith(f"scout_{a}_")
        assert not ids.tree_namespace(ns_b).startswith(f"scout_{a}_")


def test_wrap_and_unwrap_refuse_empty_aad():
    master = bytes(32)
    with pytest.raises(keys.CryptoError, match="aad"):
        keys.wrap(master, bytes(64), b"")
    blob = keys.wrap(master, bytes(64), b"acme")
    with pytest.raises(keys.CryptoError, match="aad"):
        keys.unwrap(master, blob, b"")
