"""HMAC ids and turbopuffer namespace names (design section 7, Ids).

Every id is `HMAC-SHA256(K_id, <prefix> + ...)` truncated to hex. Each id kind has its own message prefix
(`file\\0`, `repo\\0`, ...), so one input under two id functions gives unrelated outputs. Free-form inputs
are rejected when they contain NUL, which keeps the `\\0`-joined messages injective. No function here takes
`K_team`; they take `Subkeys` (AC-12.2).
"""

import hashlib
import hmac
import os
import re

from scout.core.crypto.keys import Subkeys

TREE_SUFFIX = "__tree"
NAMESPACE_RE = re.compile(r"^[A-Za-z0-9\-_.]{1,128}$")  # turbopuffer namespace charset and length
_HEX_RE = re.compile(r"^[0-9a-f]+$")

REPO_ID_LEN = 32
BRANCH_ID_LEN = 16
CLONE_ID_LEN = 12
# scout_{tenant}_{repo_id}_{branch_id}__tree must fit 128 chars: 6 + tenant + 1 + 32 + 1 + 16 + 6.
MAX_TENANT_LEN = 128 - (len("scout_") + 1 + REPO_ID_LEN + 1 + BRANCH_ID_LEN + len(TREE_SUFFIX))
# No "_" in tenants: "_" joins namespace fields, so tenant "acme_x" would match
# tenant "acme"'s prefix scan.
_TENANT_RE = re.compile(rf"^[A-Za-z0-9\-]{{1,{MAX_TENANT_LEN}}}$")


def _mac(keys: Subkeys, prefix: bytes, *parts: bytes) -> str:
    return hmac.new(keys.id, prefix + b"".join(parts), hashlib.sha256).hexdigest()


_SEP = b"\0"


def _text(value: str) -> bytes:
    if "\0" in value:
        raise ValueError("id input must not contain NUL")
    return value.encode("utf-8")


def _hex(value: str, length: int, what: str) -> str:
    if len(value) != length or not _HEX_RE.match(value):
        raise ValueError(f"{what} must be {length} lowercase hex chars")
    return value


def new_clone_id() -> str:
    """12 hex random chars, written once to the state dir's `state.json` by the caller."""
    return os.urandom(CLONE_ID_LEN // 2).hex()


def detached_name(commit: str) -> str:
    """The `branch_name` fed to `branch_id` for a detached HEAD."""
    return f"detached-{commit[:12]}"


def file_id(keys: Subkeys, path: str) -> str:
    return _mac(keys, b"file\0", _text(path))[:32]


def repo_id(keys: Subkeys, root_commit: str, remote_norm: str) -> str:
    return _mac(keys, b"repo\0", _text(root_commit), _SEP, _text(remote_norm))[:REPO_ID_LEN]


def branch_id(keys: Subkeys, repo_id: str, clone_id: str, branch_name: str) -> str:
    """Clone-scoped: teammates sharing `K_team` get different ids for the same branch name."""
    _hex(repo_id, REPO_ID_LEN, "repo_id")
    _hex(clone_id, CLONE_ID_LEN, "clone_id")
    msg = (_text(repo_id), _SEP, _text(clone_id), _SEP, _text(branch_name))
    return _mac(keys, b"branch\0", *msg)[:BRANCH_ID_LEN]


def row_id(keys: Subkeys, file_id: str, chunk_key: str, occurrence: int) -> str:
    if occurrence < 0:
        raise ValueError("occurrence must be >= 0")
    _hex(file_id, 32, "file_id")
    _hex(chunk_key, 32, "chunk_key")
    return _mac(keys, b"row\0", _text(file_id), _text(chunk_key), str(occurrence).encode())[:32]


def node_id(keys: Subkeys, repo_id: str, dirpath: str) -> str:
    _hex(repo_id, REPO_ID_LEN, "repo_id")
    return _mac(keys, b"node\0", _text(repo_id), _text(dirpath))[:32]


def ext_tag(keys: Subkeys, ext: str) -> str:
    return _mac(keys, b"ext\0", _text(ext))[:16]


def dir_tag(keys: Subkeys, ancestor: str) -> str:
    return _mac(keys, b"dir\0", _text(ancestor))[:16]


def namespace(tenant: str, repo_id: str, branch_id: str) -> str:
    """`scout_{tenant}_{repo_id}_{branch_id}`; always ends in 16 hex chars, never in `__tree`."""
    if not _TENANT_RE.match(tenant):
        raise ValueError(f"tenant must match [A-Za-z0-9-]{{1,{MAX_TENANT_LEN}}}")
    _hex(repo_id, REPO_ID_LEN, "repo_id")
    _hex(branch_id, BRANCH_ID_LEN, "branch_id")
    return f"scout_{tenant}_{repo_id}_{branch_id}"


def tree_namespace(ns: str) -> str:
    """The vector-less Merkle sibling `{ns}__tree`."""
    if ns.endswith(TREE_SUFFIX):
        raise ValueError("already a tree namespace")
    out = ns + TREE_SUFFIX
    if not NAMESPACE_RE.match(out):
        raise ValueError("tree namespace is not a valid turbopuffer name")
    return out
