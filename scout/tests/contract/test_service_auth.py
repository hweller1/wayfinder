"""Task 3.1: service auth, tenants, rate limits.

AC-9.1, AC-9.2, FR-X11, FR-G2 (HMAC key lookup), FR-G3 (per-key limits); AC-9.3/9.4 limit behaviour.
"""

import base64
import hmac
import json
import re
from collections.abc import Mapping
from typing import Any

import pytest

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError, http_status
from scout.core.service.auth import AuthService
from scout.core.service.ratelimit import Bucket, Concurrency, Limits, RateLimiter, estimate_tokens
from scout.core.service.tenants import KEY_LEN, KEY_PREFIX, KEYS_NAMESPACE, TenantDirectory

PEPPER = b"test-pepper-0123456789abcdef"


class DictCache:
    """Stands in for the Modal Dict."""

    def __init__(self) -> None:
        self.d: dict[str, Mapping[str, Any]] = {}
        self.gets = 0

    async def get(self, key_id: str) -> Mapping[str, Any] | None:
        self.gets += 1
        return self.d.get(key_id)

    async def set(self, key_id: str, record: Mapping[str, Any]) -> None:
        self.d[key_id] = record

    async def delete(self, key_id: str) -> None:
        self.d.pop(key_id, None)


class CountingStore(FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def get_by_ids(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self.reads += 1
        return await super().get_by_ids(*args, **kwargs)


def make(cache: DictCache | None = None) -> tuple[CountingStore, FakeClock, TenantDirectory, AuthService]:
    store, clock = CountingStore(), FakeClock()
    directory = TenantDirectory(store, clock, PEPPER, cache=cache)
    return store, clock, directory, AuthService(directory, clock)


async def _unauth(auth: AuthService, bearer: str | None) -> ScoutError:
    with pytest.raises(ScoutError) as ei:
        await auth.resolve(bearer)
    assert ei.value.code == ErrorCode.UNAUTHENTICATED
    return ei.value


# --- key format, storage, FR-X11 ------------------------------------------------------------------
async def test_issue_key_format_and_entropy() -> None:
    """FR-X11: `sk_scout_` + 43 base64url chars = 256 random bits."""
    _, _, d, _ = make()
    keys = [await d.issue_key("acme") for _ in range(50)]
    assert len(set(keys)) == 50
    for k in keys:
        assert k.startswith(KEY_PREFIX)
        assert len(k) == KEY_LEN == 52
        assert re.fullmatch(r"sk_scout_[A-Za-z0-9_-]{43}", k)
        assert len(base64.urlsafe_b64decode(k[len(KEY_PREFIX) :] + "=")) == 32


async def test_store_holds_tag_only_never_key_bytes() -> None:
    """AC-9.1: the durable record has the HMAC lookup id and no key material."""
    store, _, d, _ = make()
    key = await d.issue_key("acme")
    body = key[len(KEY_PREFIX) :]
    raw = base64.urlsafe_b64decode(body + "=")
    ns = store._ns[KEYS_NAMESPACE]  # pyright: ignore[reportPrivateUsage]
    assert not ns.schema.dims  # vector-less
    assert len(ns.rows) == 1
    ((rid, stored),) = ns.rows.items()
    assert stored.vector is None
    assert set(stored.attrs) == {"key_tag", "tenant", "wrapped_k_team", "created_at", "revoked"}
    blob = json.dumps({"id": rid, "attrs": stored.attrs}, sort_keys=True).encode()
    for needle in (key.encode(), body.encode(), raw, raw.hex().encode(), base64.b64encode(raw)):
        assert needle not in blob
    tag = d.tag(key)
    assert rid == tag[:32]
    assert stored.attrs["key_tag"] == tag
    assert stored.attrs["key_tag"] == hmac.new(PEPPER, key.encode(), "sha256").hexdigest()
    assert stored.attrs["tenant"] == "acme"
    assert stored.attrs["revoked"] is False


async def test_pepper_changes_lookup_id() -> None:
    _, clock, d, _ = make()
    key = await d.issue_key("acme")
    other = TenantDirectory(FakeStore(), clock, b"another-pepper-0123456789")
    assert other.tag(key) != d.tag(key)


def test_short_pepper_refused() -> None:
    with pytest.raises(ValueError, match="pepper"):
        TenantDirectory(FakeStore(), FakeClock(), b"short")


@pytest.mark.parametrize("bad", ["", "acme_x", "a" * 200, "a b", "../x", "acme/x"])
async def test_issue_key_rejects_bad_tenant_names(bad: str) -> None:
    store, _, d, _ = make()
    with pytest.raises(ScoutError) as ei:
        await d.issue_key(bad)
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert not await store.namespace_exists(KEYS_NAMESPACE)  # nothing written for a refused tenant


# --- resolve --------------------------------------------------------------------------------------
async def test_resolve_valid_key() -> None:
    _, _, d, auth = make()
    key = await d.issue_key("acme")
    t = await auth.resolve(key)
    assert t.tenant == "acme"
    assert t.key_id == d.tag(key)[:32]


async def test_constant_time_compare_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-X11 / AC-9.1: the tag comparison goes through `hmac.compare_digest`, on every outcome."""
    _, _, d, auth = make()
    key = await d.issue_key("acme")
    real = hmac.compare_digest
    calls: list[tuple[int, int]] = []

    def spy(a: Any, b: Any) -> bool:
        calls.append((len(a), len(b)))
        return real(a, b)

    monkeypatch.setattr("scout.core.service.auth.hmac.compare_digest", spy)
    await auth.resolve(key)
    assert calls == [(64, 64)]
    calls.clear()
    for bad in (key[:-1] + ("A" if key[-1] != "A" else "B"), "", "garbage", KEY_PREFIX + "x" * 43):
        await _unauth(auth, bad)
    assert calls == [(64, 64)] * 4  # equal-length compare even for unknown keys


async def test_unknown_malformed_revoked_empty_are_indistinguishable() -> None:
    """No oracle: same code, message, details, wire body and the same work (store reads, compares)."""
    store, _, d, auth = make()
    revoked = await d.issue_key("acme")
    await d.issue_key("acme")  # keep the namespace populated
    assert await d.revoke(d.tag(revoked)[:32])
    unknown = KEY_PREFIX + base64.urlsafe_b64encode(b"\x01" * 32).rstrip(b"=").decode()
    cases = [
        None,
        "",
        " ",
        "sk_scout_",
        "garbage",
        unknown,
        revoked,
        revoked + "x",
        "x" * 10_000,
        "sk_scout_é" * 5,
    ]
    bodies: list[str] = []
    reads: list[int] = []
    for bearer in cases:
        before = store.reads
        err = await _unauth(auth, bearer)
        reads.append(store.reads - before)
        bodies.append(err.to_body().model_dump_json())
        assert http_status(err.code) == 401
        assert err.details == {}
    assert len(set(bodies)) == 1
    assert set(reads) == {1}


async def test_no_namespace_yet_is_unauthenticated_not_internal() -> None:
    _, _, _, auth = make()
    await _unauth(auth, KEY_PREFIX + "A" * 43)


async def test_revoked_key_is_unauthenticated_even_if_cached() -> None:
    cache = DictCache()
    _, _, d, auth = make(cache)
    key = await d.issue_key("acme")
    await auth.resolve(key)
    assert cache.d  # warmed
    await d.revoke(d.tag(key)[:32])
    await _unauth(auth, key)


async def test_cross_container_revocation_bounded_by_ttl() -> None:
    """A stale cache entry (another container's revoke) stops working after the TTL."""
    cache = DictCache()
    store, clock, d, auth = make(cache)
    key = await d.issue_key("acme")
    await auth.resolve(key)
    rec = await d.get_record(d.tag(key)[:32])
    assert rec is not None
    from scout.core.ports.store import Row

    await store.upsert(
        KEYS_NAMESPACE, [Row(id=d.tag(key)[:32], attrs={**rec, "revoked": True})]
    )  # no cache delete
    await auth.resolve(key)  # still inside the TTL
    clock.advance(61)
    await _unauth(auth, key)


async def test_cache_miss_still_resolves_and_populates() -> None:
    """AC-9.2: the durable store answers when the cache is empty; the cache is only read-through."""
    cache = DictCache()
    store, _, d, auth = make(cache)
    key = await d.issue_key("acme")
    assert cache.d == {}
    assert (await auth.resolve(key)).tenant == "acme"
    assert store.reads == 1
    assert len(cache.d) == 1
    assert (await auth.resolve(key)).tenant == "acme"
    assert store.reads == 1  # served from cache
    cache.d.clear()  # Dict expiry
    assert (await auth.resolve(key)).tenant == "acme"
    assert store.reads == 2


async def test_durable_namespace_is_outside_tenant_prefixes() -> None:
    """AC-9.2: `scoutsys_keys` can never be a tenant namespace (`scout_{tenant}_...`)."""
    assert not KEYS_NAMESPACE.startswith("scout_")
    store, _, d, _ = make()
    await d.issue_key("acme")
    assert await store.list_namespaces("scout_") == []


async def test_poisoned_cache_entry_cannot_authenticate_another_key() -> None:
    """A cache record under the right id but another key's tag fails the constant-time compare."""
    cache = DictCache()
    _, clock, d, auth = make(cache)
    k1, k2 = await d.issue_key("acme"), await d.issue_key("beta")
    rec1 = await d.get_record(d.tag(k1)[:32])
    await cache.set(d.tag(k2)[:32], {"cached_at": clock.now(), "record": dict(rec1 or {})})
    await _unauth(auth, k2)


# --- tenant comes only from the server-side record ------------------------------------------------
async def test_tenant_from_record_only_request_body_has_no_effect() -> None:
    _, _, d, auth = make()
    ka, kb = await d.issue_key("alpha"), await d.issue_key("beta")
    body = {"tenant": "beta", "tenant_id": "beta", "repo_id": "a" * 32}  # hostile client input
    # resolve's only input is the bearer: the body cannot reach it
    assert (await auth.resolve(ka)).tenant == "alpha"
    assert (await auth.resolve(kb)).tenant == "beta"
    assert body["tenant"] == "beta"
    import inspect

    assert list(inspect.signature(auth.resolve).parameters) == ["bearer"]


async def test_one_key_maps_to_exactly_one_tenant() -> None:
    store, _, d, auth = make()
    k1, k2 = await d.issue_key("alpha"), await d.issue_key("alpha")
    assert k1 != k2
    assert (await auth.resolve(k1)).key_id != (await auth.resolve(k2)).key_id
    ns = store._ns[KEYS_NAMESPACE]  # pyright: ignore[reportPrivateUsage]
    assert {r.attrs["tenant"] for r in ns.rows.values()} == {"alpha"}


# --- rate limits (FakeClock) ----------------------------------------------------------------------
def _limited(fn: Any) -> ScoutError:
    with pytest.raises(ScoutError) as ei:
        fn()
    assert ei.value.code == ErrorCode.RATE_LIMITED
    assert ei.value.retryable
    assert http_status(ei.value.code) == 429
    return ei.value


def test_search_burst_then_429_with_retry_after() -> None:
    clock = FakeClock()
    rl = RateLimiter(clock)
    for _ in range(20):
        rl.search("k1")
    err = _limited(lambda: rl.search("k1"))
    assert err.details["retry_after"] == pytest.approx(0.1)


def test_search_refill_is_10_per_second() -> None:
    clock = FakeClock()
    rl = RateLimiter(clock)
    for _ in range(20):
        rl.search("k1")
    clock.advance(0.5)
    for _ in range(5):
        rl.search("k1")
    _limited(lambda: rl.search("k1"))
    clock.advance(1000)  # refill caps at the burst
    for _ in range(20):
        rl.search("k1")
    _limited(lambda: rl.search("k1"))


def test_limits_are_per_key() -> None:
    rl = RateLimiter(FakeClock())
    for _ in range(20):
        rl.search("k1")
    _limited(lambda: rl.search("k1"))
    rl.search("k2")  # unaffected
    for _ in range(10):
        rl.general("k1")  # separate bucket from search
    _limited(lambda: rl.general("k1"))
    rl.general("k2")


def test_embed_token_bucket_precharge_and_reconcile() -> None:
    clock = FakeClock()
    rl = RateLimiter(clock)
    assert estimate_tokens(7) == 2
    assert estimate_tokens(0) == 0
    charged = rl.embed("k1", int(1_500_000 * 3.5))
    assert charged == 1_500_000
    err = _limited(lambda: rl.embed("k1", int(1_000_000 * 3.5)))
    assert err.details["retry_after"] == pytest.approx(15.0)  # 500k short at 2M/min = 33,333/s
    rl.reconcile_embed("k1", charged, 100_000)  # real usage was far lower: refund
    rl.embed("k1", int(1_000_000 * 3.5))
    clock.advance(60)
    rl.embed("k1", int(2_000_000 * 3.5))  # full refill after a minute
    rl.embed("k2", int(2_000_000 * 3.5))  # other key independent


def test_embed_under_estimate_bills_the_difference() -> None:
    rl = RateLimiter(FakeClock())
    charged = rl.embed("k1", int(1_000_000 * 3.5))
    rl.reconcile_embed("k1", charged, 2_500_000)  # real tokens exceeded the estimate: debt
    _limited(lambda: rl.embed("k1", 7))


def test_embed_request_larger_than_a_minute_is_invalid_not_retryable() -> None:
    rl = RateLimiter(FakeClock())
    with pytest.raises(ScoutError) as ei:
        rl.embed("k1", int(2_000_001 * 3.5))
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT


def test_chat_two_concurrent_per_key() -> None:
    rl = RateLimiter(FakeClock())
    with rl.chat_slot("k1"), rl.chat_slot("k1"):
        _limited(lambda: rl.chat_slot("k1").__enter__())
        with rl.chat_slot("k2"):
            pass
    with rl.chat_slot("k1"):  # released
        pass


def test_bucket_and_concurrency_validate_config() -> None:
    with pytest.raises(ValueError, match="positive"):
        Bucket(FakeClock(), capacity=0, refill_per_s=1)
    with pytest.raises(ValueError, match="limit"):
        Concurrency(0)
    assert Limits().embed_tokens_per_min == 2_000_000
