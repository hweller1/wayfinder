"""MinHash-128 (design section 8, FR-C16, AC-4.1)."""

# ruff: noqa: S311  (seeded, non-cryptographic randomness is the point of these statistical tests)

import random
import time

import pytest

from scout.core.crypto import keys
from scout.core.crypto.keys import CryptoError
from scout.core.crypto.minhash import SKETCH_BYTES, SLOTS, jaccard, sketch

SK = keys.derive(bytes(range(64)))
OTHER = keys.derive(bytes(range(1, 65)))
TRIALS = 50
# (shared, only-A, only-B) with union 100, so true Jaccard = shared / 100.
LEVELS = {0.9: (90, 5, 5), 0.6: (60, 20, 20), 0.3: (30, 35, 35), 0.0: (0, 50, 50)}


def _hashes(rng: random.Random, n: int) -> list[bytes]:
    return [rng.randbytes(32) for _ in range(n)]


def _trial(rng: random.Random, level: float) -> tuple[list[bytes], list[bytes]]:
    shared, only_a, only_b = LEVELS[level]
    s, a, b = _hashes(rng, shared), _hashes(rng, only_a), _hashes(rng, only_b)
    return s + a, s + b


def test_accuracy_over_seeded_trials() -> None:
    """AC-4.1 per parent ruling Q2: bias and mean abs error over 50 seeded trials, never a single draw.

    One draw has std error sqrt(J(1-J)/128) up to 0.044 and E|err| ~ 0.8 sigma (<= 0.035), so a 0.02
    bound can only apply to the *bias* (mean signed error), whose sd over 50 trials is ~0.006; the
    mean absolute error is held to 0.05 (AC-4.1's per-estimate target).
    """
    start = time.perf_counter()
    for level in LEVELS:
        rng = random.Random(f"minhash-{level}")
        errs = []
        for _ in range(TRIALS):
            a, b = _trial(rng, level)
            errs.append(jaccard(sketch(SK, a), sketch(SK, b)) - level)
        bias = abs(sum(errs) / TRIALS)
        mae = sum(abs(e) for e in errs) / TRIALS
        assert bias <= 0.02, (level, bias)
        assert mae <= 0.05, (level, mae)
    assert time.perf_counter() - start < 2.0


def test_identical_and_disjoint_extremes() -> None:
    items = _hashes(random.Random(1), 40)
    assert jaccard(sketch(SK, items), sketch(SK, items)) == 1.0
    assert jaccard(sketch(SK, items), sketch(SK, _hashes(random.Random(2), 40))) == 0.0


def test_order_independent_and_duplicate_insensitive() -> None:
    items = _hashes(random.Random(3), 60)
    shuffled = items[:]
    random.Random(4).shuffle(shuffled)
    assert sketch(SK, items) == sketch(SK, shuffled) == sketch(SK, iter(shuffled + items[:10]))


def test_deterministic_across_key_derivations() -> None:
    items = _hashes(random.Random(5), 30)
    assert sketch(SK, items) == sketch(keys.derive(bytes(range(64))), items)


def test_size_and_roundtrip() -> None:
    sk = sketch(SK, _hashes(random.Random(6), 25))
    assert SKETCH_BYTES == 512
    assert SLOTS == 128
    assert len(sk) == 512
    assert bytes(bytearray(sk)) == sk
    assert jaccard(sk, bytes(sk)) == 1.0
    assert len(sketch(SK, [])) == 512


def test_teams_are_not_comparable() -> None:
    """The same repo (same leaf hashes) sketched under two team keys is unrelated; same key is identical."""
    items = _hashes(random.Random(7), 80)
    assert jaccard(sketch(SK, items), sketch(OTHER, items)) < 0.05
    assert jaccard(sketch(SK, items), sketch(SK, items)) == 1.0


def test_sketch_reveals_no_leaf_item() -> None:
    items = _hashes(random.Random(8), 50)
    blob = sketch(SK, items)
    assert all(it[:8] not in blob and it[:4] not in blob for it in items)


def test_empty_set_and_short_input() -> None:
    assert jaccard(sketch(SK, []), sketch(SK, _hashes(random.Random(9), 5))) == 0.0
    with pytest.raises(CryptoError):
        sketch(SK, [b"short"])


def test_jaccard_rejects_wrong_size() -> None:
    with pytest.raises(CryptoError):
        jaccard(b"\0" * 511, b"\0" * 512)
