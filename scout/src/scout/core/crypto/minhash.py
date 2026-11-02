"""MinHash-128 sketch over a repo's keyed leaf hashes (design section 8; PRD 4.4 teammate seed).

Input is the set of keyed leaf hashes (HMAC under K_leaf), never plaintext paths or content. An item is the
first 8 bytes of a leaf hash as a big-endian uint64 `x`; slot `i` is `min((a_i*x + b_i) mod (2^61-1)) &
0xFFFFFFFF` over the set. 128 slots x 32 bit, little-endian, give the 512 B wire sketch.

Keying (deviation from the design text, which derived `(a_i, b_i)` from a public SHA256 label): the pair is
`HMAC-SHA256(K_leaf, "scout-minhash-v1" || i || half)` for two halves, reduced mod the prime. Teammates share
`K_team` so their sketches agree; a different team has different leaf hashes *and* different permutations, so
two teams' sketches of the same repo are unrelated and reveal nothing without the key (an observer cannot even
re-derive the permutations). The sketch is a set function: order and duplicates in the input do not matter.

Estimate = fraction of equal slots, unbiased for Jaccard with standard error `sqrt(J(1-J)/128)` (<= 0.044).
"""

import hashlib
import hmac
from collections.abc import Iterable

from scout.core.crypto.keys import CryptoError, Subkeys

SLOTS = 128
SLOT_BYTES = 4
SKETCH_BYTES = SLOTS * SLOT_BYTES  # 512
MASK32 = 0xFFFFFFFF
PRIME = (1 << 61) - 1
LABEL = b"scout-minhash-v1"
EMPTY_SLOT = MASK32  # slot value of an empty set

_params_cache: dict[bytes, tuple[tuple[int, int], ...]] = {}


def _params(keys: Subkeys) -> tuple[tuple[int, int], ...]:
    """Per-team `(a_i, b_i)` with `a_i` in [1, p-1] and `b_i` in [0, p-1]."""
    cached = _params_cache.get(keys.leaf)
    if cached is not None:
        return cached
    out: list[tuple[int, int]] = []
    for i in range(SLOTS):
        msg = LABEL + i.to_bytes(4, "big")
        a = int.from_bytes(hmac.new(keys.leaf, msg + b"a", hashlib.sha256).digest(), "big")
        b = int.from_bytes(hmac.new(keys.leaf, msg + b"b", hashlib.sha256).digest(), "big")
        out.append((a % (PRIME - 1) + 1, b % PRIME))
    result = tuple(out)
    _params_cache[keys.leaf] = result
    return result


def sketch(keys: Subkeys, leaf_hashes: Iterable[bytes]) -> bytes:
    """The 512 B sketch of a set of leaf hashes (each >= 8 bytes)."""
    items: set[int] = set()
    for leaf in leaf_hashes:
        if len(leaf) < 8:
            raise CryptoError("leaf hash must be at least 8 bytes")
        items.add(int.from_bytes(leaf[:8], "big"))
    params = _params(keys)
    out = bytearray()
    for a, b in params:
        slot = min(((a * x + b) % PRIME for x in items), default=EMPTY_SLOT) & MASK32
        out += slot.to_bytes(SLOT_BYTES, "little")
    return bytes(out)


def jaccard(a: bytes, b: bytes) -> float:
    """Estimated Jaccard similarity: the fraction of equal slots. Needs two 512 B sketches."""
    if len(a) != SKETCH_BYTES or len(b) != SKETCH_BYTES:
        raise CryptoError(f"sketch must be {SKETCH_BYTES} bytes")
    equal = sum(a[i : i + SLOT_BYTES] == b[i : i + SLOT_BYTES] for i in range(0, SKETCH_BYTES, SLOT_BYTES))
    return equal / SLOTS
