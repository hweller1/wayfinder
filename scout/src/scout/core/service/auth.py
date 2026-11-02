"""`AuthService.resolve`: bearer key -> `Tenant` (design section 6, PRD section 6).

Every failure (empty, malformed, unknown, revoked, record without a tag, tenant name rejected) takes the same
steps (one HMAC, one record lookup, one constant-time compare against a tag of equal length) and raises the
same UNAUTHENTICATED error, so neither the body nor the work done tells a caller which case it hit.
"""

import hmac
from collections.abc import Mapping
from typing import Any

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.clock import Clock
from scout.core.service.tenants import ID_LEN, Tenant, TenantDirectory, key_shape_ok

CACHE_TTL_S = 60.0  # cross-container revocation lag is bounded by this, whatever the cache's own expiry
_MISS_TAG = "0" * 64  # compared when there is no record, so the compare always runs on equal-length input


def unauthenticated() -> ScoutError:
    return ScoutError(ErrorCode.UNAUTHENTICATED, "missing or invalid API key")


class AuthService:
    def __init__(self, directory: TenantDirectory, clock: Clock, *, cache_ttl_s: float = CACHE_TTL_S) -> None:
        self._dir = directory
        self._clock = clock
        self._ttl = cache_ttl_s

    async def _record(self, key_id: str) -> Mapping[str, Any] | None:
        cached = await self._dir.cache.get(key_id)
        if cached is not None:
            at = cached.get("cached_at")
            record = cached.get("record")
            if (
                isinstance(at, int | float)
                and isinstance(record, dict)
                and self._clock.now() - at < self._ttl
            ):
                return record  # pyright: ignore[reportUnknownVariableType]
        record = await self._dir.get_record(key_id)  # miss, stale or no cache: the store decides (AC-9.2)
        if record is not None and not record.get("revoked"):
            await self._dir.cache.set(key_id, {"cached_at": self._clock.now(), "record": dict(record)})
        return record

    async def resolve(self, bearer: str | None) -> Tenant:
        """The tenant comes only from the stored record; nothing else is an input."""
        presented = bearer or ""
        tag = self._dir.tag(presented)
        key_id = tag[:ID_LEN]
        record = await self._record(key_id)
        stored = record.get("key_tag") if record is not None else None
        stored_tag = stored if isinstance(stored, str) and len(stored) == len(tag) else _MISS_TAG
        tag_ok = hmac.compare_digest(tag.encode(), stored_tag.encode())
        tenant = record.get("tenant") if record is not None else None
        ok = (
            key_shape_ok(presented)
            and record is not None
            and tag_ok
            and record.get("revoked") is False
            and isinstance(tenant, str)
            and bool(tenant)
        )
        if not ok or not isinstance(tenant, str):
            raise unauthenticated()
        return Tenant(tenant=tenant, key_id=key_id)
