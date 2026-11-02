"""Token buckets and a concurrency cap, per key, on the injected `Clock` (design section 6).

Limits are per container: Modal runs `max_containers=3`, so the effective limit is at most 3x. A bucket may go
negative after `reconcile` (real usage above the pre-charge); later takes then wait for the debt to clear.
"""

import math
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.clock import Clock

CHARS_PER_TOKEN = 3.5


def estimate_tokens(chars: int) -> int:
    """Embed pre-charge: `ceil(chars / 3.5)`."""
    return math.ceil(chars / CHARS_PER_TOKEN)


def _limited(what: str, retry_after: float, limit: float) -> ScoutError:
    """`details.retry_after` is seconds (the surface turns it into `Retry-After`)."""
    return ScoutError(
        ErrorCode.RATE_LIMITED,
        f"{what} rate limit exceeded",
        details={"retry_after": round(retry_after, 3), "limit": limit},
    )


class Bucket:
    def __init__(self, clock: Clock, *, capacity: float, refill_per_s: float, name: str = "request") -> None:
        if capacity <= 0 or refill_per_s <= 0:
            raise ValueError("capacity and refill_per_s must be positive")
        self._clock = clock
        self._name = name
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self._tokens = float(capacity)
        self._last = clock.monotonic()

    def _refill(self) -> None:
        now = self._clock.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_per_s)
        self._last = now

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens

    def take(self, cost: float = 1.0) -> None:
        """Spend `cost` or raise RATE_LIMITED with the wait until it fits. A cost above capacity can never
        fit, so it is INVALID_ARGUMENT rather than a retry loop."""
        if cost > self.capacity:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"{self._name} cost exceeds the limit's capacity")
        self._refill()
        if self._tokens < cost:
            raise _limited(self._name, (cost - self._tokens) / self.refill_per_s, self.capacity)
        self._tokens -= cost

    def reconcile(self, charged: float, actual: float) -> None:
        """Refund an over-charge (up to capacity) or bill an under-charge (may go negative)."""
        self._refill()
        self._tokens = min(self.capacity, self._tokens + (charged - actual))


class Concurrency:
    """At most `limit` holders at once (chat: 2 per key)."""

    def __init__(self, limit: int, *, name: str = "concurrency", retry_after: float = 1.0) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._name = name
        self._retry_after = retry_after
        self._held = 0

    @contextmanager
    def slot(self) -> Generator[None]:
        if self._held >= self._limit:
            raise _limited(self._name, self._retry_after, self._limit)
        self._held += 1
        try:
            yield
        finally:
            self._held -= 1


@dataclass(frozen=True, slots=True)
class Limits:
    search_per_s: float = 10.0
    search_burst: float = 20.0
    general_per_s: float = 10.0
    embed_tokens_per_min: float = 2_000_000.0
    chat_concurrent: int = 2


class RateLimiter:
    """Per-key buckets, created on first use. Call only after authentication, so memory is bounded by the
    number of valid keys."""

    def __init__(self, clock: Clock, limits: Limits | None = None) -> None:
        self._clock = clock
        self._limits = limits or Limits()
        self._search: dict[str, Bucket] = {}
        self._general: dict[str, Bucket] = {}
        self._embed: dict[str, Bucket] = {}
        self._chat: dict[str, Concurrency] = {}

    def _bucket(
        self, pool: dict[str, Bucket], key_id: str, name: str, capacity: float, per_s: float
    ) -> Bucket:
        b = pool.get(key_id)
        if b is None:
            b = pool[key_id] = Bucket(self._clock, capacity=capacity, refill_per_s=per_s, name=name)
        return b

    def search(self, key_id: str) -> None:
        lim = self._limits
        self._bucket(self._search, key_id, "search", lim.search_burst, lim.search_per_s).take()

    def general(self, key_id: str) -> None:
        lim = self._limits
        self._bucket(self._general, key_id, "request", lim.general_per_s, lim.general_per_s).take()

    def _embed_bucket(self, key_id: str) -> Bucket:
        per_min = self._limits.embed_tokens_per_min
        return self._bucket(self._embed, key_id, "embed token", per_min, per_min / 60.0)

    def embed(self, key_id: str, chars: int) -> int:
        """Pre-charge `ceil(chars/3.5)` tokens; returns the amount charged, to pass to `reconcile_embed`."""
        charged = estimate_tokens(chars)
        self._embed_bucket(key_id).take(charged)
        return charged

    def reconcile_embed(self, key_id: str, charged: int, real_tokens: int) -> None:
        self._embed_bucket(key_id).reconcile(charged, real_tokens)

    def chat_slot(self, key_id: str) -> AbstractContextManager[None]:
        c = self._chat.get(key_id)
        if c is None:
            c = self._chat[key_id] = Concurrency(self._limits.chat_concurrent, name="chat")
        return c.slot()
