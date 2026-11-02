"""Path encryption (design section 7): AES-256-SIV per segment, deterministic by design.

Each segment is encrypted with `AAD = [b"path", tenant, repo_id, parent_plaintext_prefix]` and hex-encoded;
segments are joined with `/`. Determinism is what lets a client encrypt a path filter and have the store
match it. The parent prefix in the AAD means equal names under different directories differ, while two
paths sharing a parent (`a/b`, `a/c`) share the first encrypted segment: that leak is documented in the PRD
(section 5) and pinned by a test. Hex output contains no glob metacharacters, so a prefix glob is safe.
"""

import re
from pathlib import PurePosixPath

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESSIV

from scout.core.crypto.keys import CryptoError, Subkeys

_HEX_SEGMENT_RE = re.compile(r"^(?:[0-9a-f]{2})+$")
_TAG_LEN = 16  # SIV tag; a valid segment ciphertext is at least tag + 1 byte


def _segments(path: str) -> list[str]:
    segs = path.split("/")
    for seg in segs:
        if seg in ("", ".", "..") or "\0" in seg:
            raise ValueError("path must be relative with non-empty segments, no '.', '..' or NUL")
    return segs


def _aad(tenant: str, repo_id: str, parent: str) -> list[bytes]:
    return [b"path", tenant.encode("utf-8"), repo_id.encode("utf-8"), parent.encode("utf-8")]


def enc_path(keys: Subkeys, tenant: str, repo_id: str, path: str) -> str:
    siv = AESSIV(keys.path)
    out: list[str] = []
    parent = ""
    for seg in _segments(path):
        out.append(siv.encrypt(seg.encode("utf-8"), _aad(tenant, repo_id, parent)).hex())
        parent = f"{parent}/{seg}" if parent else seg
    return "/".join(out)


def dec_path(keys: Subkeys, tenant: str, repo_id: str, path_enc: str) -> str:
    siv = AESSIV(keys.path)
    out: list[str] = []
    parent = ""
    for enc in path_enc.split("/"):
        if not _HEX_SEGMENT_RE.match(enc) or len(enc) // 2 <= _TAG_LEN:
            raise CryptoError("malformed encrypted path")
        try:
            seg = siv.decrypt(bytes.fromhex(enc), _aad(tenant, repo_id, parent)).decode("utf-8")
        except (InvalidTag, UnicodeDecodeError):
            raise CryptoError("path decryption failed") from None
        if "/" in seg or seg in ("", ".", "..") or "\0" in seg:
            raise CryptoError("path decryption failed")
        out.append(seg)
        parent = f"{parent}/{seg}" if parent else seg
    return "/".join(out)


def enc_prefix_glob(keys: Subkeys, tenant: str, repo_id: str, dir_path: str) -> str:
    """Whole-segment prefix glob `enc(dir)/**`; an empty dir (repo root) is plain `**`."""
    if dir_path == "":
        return "**"
    return enc_path(keys, tenant, repo_id, dir_path) + "/**"


def path_ext(path: str) -> str:
    """Extension of the last segment with its dot (`.py`); `""` for none (and for dotfiles like `.env`)."""
    return PurePosixPath(path).suffix


def ancestors(path: str) -> list[str]:
    """Every ancestor directory, shallowest first: `a/b/c.py` -> `["a", "a/b"]`."""
    segs = _segments(path)
    return ["/".join(segs[: i + 1]) for i in range(len(segs) - 1)]
