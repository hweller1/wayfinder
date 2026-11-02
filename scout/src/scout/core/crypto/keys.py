"""Key hierarchy (design section 7): `K_team` root, HKDF-SHA256 subkeys, AES-256-GCM wrap.

`K_team` is never used as a key for any primitive other than HKDF (AC-12.2). Only `derive`, `wrap` and
`unwrap` touch it; every other crypto function takes a `Subkeys` value. Only `cryptography` primitives are
used, and every wrap draws a fresh 96-bit nonce from `os.urandom`.
"""

import os
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

HKDF_SALT = b"scout-hkdf-v1"
TEAM_KEY_LEN = 64
MASTER_KEY_LEN = 32
NONCE_LEN = 12

INFO_PATH = b"scout/v1/path"
INFO_LOC = b"scout/v1/loc"
INFO_LEAF = b"scout/v1/leaf"
INFO_CHUNK = b"scout/v1/chunk"
INFO_TOKEN = b"scout/v1/token"
INFO_ID = b"scout/v1/id"


class CryptoError(ValueError):
    """Bad key size or a failed authenticated decryption (wrong key, wrong AAD, tampered blob)."""


@dataclass(frozen=True, slots=True)
class Subkeys:
    """The six purpose keys. `path` is 64 B (AES-256-SIV); the rest are 32 B. `repr` never shows bytes."""

    path: bytes = field(repr=False)
    loc: bytes = field(repr=False)
    leaf: bytes = field(repr=False)
    chunk: bytes = field(repr=False)
    token: bytes = field(repr=False)
    id: bytes = field(repr=False)


def new_team_key() -> bytes:
    """A fresh 512-bit `K_team` from the OS CSPRNG."""
    return os.urandom(TEAM_KEY_LEN)


def _hkdf(k_team: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=HKDF_SALT, info=info).derive(k_team)


def derive(k_team: bytes) -> Subkeys:
    if len(k_team) != TEAM_KEY_LEN:
        raise CryptoError(f"K_team must be {TEAM_KEY_LEN} bytes")
    return Subkeys(
        path=_hkdf(k_team, INFO_PATH, 64),
        loc=_hkdf(k_team, INFO_LOC, 32),
        leaf=_hkdf(k_team, INFO_LEAF, 32),
        chunk=_hkdf(k_team, INFO_CHUNK, 32),
        token=_hkdf(k_team, INFO_TOKEN, 32),
        id=_hkdf(k_team, INFO_ID, 32),
    )


def wrap(master: bytes, k_team: bytes, aad: bytes) -> bytes:
    """`nonce(12) || AES-256-GCM(master, nonce, k_team, aad)`; the gateway uses `aad = tenant`."""
    if len(master) != MASTER_KEY_LEN:
        raise CryptoError(f"master key must be {MASTER_KEY_LEN} bytes")
    if len(k_team) != TEAM_KEY_LEN:
        raise CryptoError(f"K_team must be {TEAM_KEY_LEN} bytes")
    if not aad:
        raise CryptoError("wrap needs a non-empty aad (the tenant), so keys cannot move between tenants")
    nonce = os.urandom(NONCE_LEN)
    return nonce + AESGCM(master).encrypt(nonce, k_team, aad)


def unwrap(master: bytes, blob: bytes, aad: bytes) -> bytes:
    if len(master) != MASTER_KEY_LEN:
        raise CryptoError(f"master key must be {MASTER_KEY_LEN} bytes")
    if not aad:
        raise CryptoError("unwrap needs a non-empty aad (the tenant)")
    if len(blob) <= NONCE_LEN:
        raise CryptoError("wrapped key too short")
    try:
        k_team = AESGCM(master).decrypt(blob[:NONCE_LEN], blob[NONCE_LEN:], aad)
    except InvalidTag:
        raise CryptoError("unwrap failed") from None
    if len(k_team) != TEAM_KEY_LEN:
        raise CryptoError("unwrapped key has the wrong size")
    return k_team
