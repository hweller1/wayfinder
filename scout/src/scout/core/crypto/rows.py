"""Row sealing (design section 7): sealed chunk rows and Merkle node blobs.

Everything here except paths is AES-256-GCM under `K_loc` with a fresh 12-byte `os.urandom` nonce, stored as
`b64(nonce || ciphertext || tag)`. The AAD is a domain tag plus length-prefixed fields (so `("ab", "c")` and
`("a", "bc")` differ): `loc` binds tenant, repo_id, chunk_key and the row's `path_enc`; `node` binds tenant,
repo_id and node_id. A blob moved to another tenant, repo, chunk or file fails to open (AC-12.3). `path_enc`
is in the loc AAD because `WireHit` carries it, so equal chunk bodies in two files (same `chunk_key`) cannot
swap line ranges; two identical chunks inside one file share an AAD (the hit has no occurrence).
"""

import base64
import binascii
import json
import os
import struct
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from scout.core.contracts.wire import MAX_TOKENS_PER_ROW, WireHit, WireRow
from scout.core.crypto import ids, paths
from scout.core.crypto.keys import NONCE_LEN, CryptoError, Subkeys

_TAG_LEN = 16
_MAX_DIR_TAGS = 64


def _aad(domain: bytes, *fields: str) -> bytes:
    out = bytearray(domain)
    for f in fields:
        raw = f.encode("utf-8")
        out += struct.pack(">I", len(raw)) + raw
    return bytes(out)


def _seal(keys: Subkeys, aad: bytes, plaintext: bytes) -> str:
    nonce = os.urandom(NONCE_LEN)
    return base64.b64encode(nonce + AESGCM(keys.loc).encrypt(nonce, plaintext, aad)).decode("ascii")


def _open(keys: Subkeys, aad: bytes, blob_b64: str) -> bytes:
    try:
        blob = base64.b64decode(blob_b64, validate=True)
    except (binascii.Error, ValueError):
        raise CryptoError("malformed sealed blob") from None
    if len(blob) < NONCE_LEN + _TAG_LEN:
        raise CryptoError("malformed sealed blob")
    try:
        return AESGCM(keys.loc).decrypt(blob[:NONCE_LEN], blob[NONCE_LEN:], aad)
    except InvalidTag:
        raise CryptoError("sealed blob failed authentication") from None


def _loc_aad(tenant: str, repo_id: str, chunk_key: str, path_enc: str) -> bytes:
    return _aad(b"loc", tenant, repo_id, chunk_key, path_enc)


def seal_chunk_row(
    keys: Subkeys,
    tenant: str,
    repo_id: str,
    *,
    path: str,
    start: int,
    end: int,
    chunk_key: str,
    occurrence: int,
    file_hash: str,
    tokens: list[str],
) -> WireRow:
    """One chunk's wire row. `tokens` are already HMAC'd (cap 256); no vector (the gateway adds it)."""
    if not 1 <= start <= end:
        raise ValueError("need 1 <= start <= end")
    if len(tokens) > MAX_TOKENS_PER_ROW:
        raise ValueError(f"at most {MAX_TOKENS_PER_ROW} tokens per row")
    dirs = paths.ancestors(path)
    if len(dirs) > _MAX_DIR_TAGS:
        raise ValueError(f"path deeper than {_MAX_DIR_TAGS} directories")
    fid = ids.file_id(keys, path)
    path_enc = paths.enc_path(keys, tenant, repo_id, path)
    plain = json.dumps({"s": start, "e": end}, separators=(",", ":")).encode("ascii")
    return WireRow(
        id=ids.row_id(keys, fid, chunk_key, occurrence),
        file_id=fid,
        path_enc=path_enc,
        ext_tag=ids.ext_tag(keys, paths.path_ext(path)),
        dir_tags=[ids.dir_tag(keys, d) for d in dirs],
        file_hash=file_hash,
        chunk_key=chunk_key,
        loc_enc=_seal(keys, _loc_aad(tenant, repo_id, chunk_key, path_enc), plain),
        tokens=list(tokens),
    )


def open_hit(keys: Subkeys, tenant: str, repo_id: str, hit: WireHit) -> tuple[str, int, int]:
    """Decrypt a search hit to `(path, start_line, end_line)`; raises `CryptoError` on any mismatch."""
    path = paths.dec_path(keys, tenant, repo_id, hit.path_enc)
    plain = _open(keys, _loc_aad(tenant, repo_id, hit.chunk_key, hit.path_enc), hit.loc_enc)
    try:
        obj: Any = json.loads(plain)
        start, end = obj["s"], obj["e"]
    except (ValueError, KeyError, TypeError):
        raise CryptoError("malformed line range") from None
    ok = all(type(v) is int for v in (start, end)) and 1 <= start <= end
    if not ok:
        raise CryptoError("malformed line range")
    return path, start, end


def seal_node(
    keys: Subkeys, tenant: str, repo_id: str, node_id: str, children: list[tuple[str, str, str]]
) -> str:
    """`children_enc` for a Merkle node: sealed JSON list of `(name, kind, hash)`."""
    plain = json.dumps(children, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _seal(keys, _aad(b"node", tenant, repo_id, node_id), plain)


def open_node(
    keys: Subkeys, tenant: str, repo_id: str, node_id: str, blob_b64: str
) -> list[tuple[str, str, str]]:
    plain = _open(keys, _aad(b"node", tenant, repo_id, node_id), blob_b64)
    try:
        raw: Any = json.loads(plain)
        return [(str(n), str(k), str(h)) for n, k, h in raw]
    except (ValueError, TypeError):
        raise CryptoError("malformed node blob") from None
