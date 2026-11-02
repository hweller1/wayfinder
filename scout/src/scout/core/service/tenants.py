"""Key records (design sections 7 and 9): the durable key -> tenant map in a vector-less global namespace.

A Scout key is `sk_scout_` + base64url(32 random bytes) (256 bits). The server never stores it: the record id
is `tag[:32]` and `key_tag` is the full `tag = HMAC-SHA256(pepper, key)` (hex). Keys are high entropy, so a
plain keyed hash is enough (no password hashing on the request path). `Tenant` comes only from the record,
never from a request. Issuing a key is an admin function (no HTTP route): the plaintext is returned once.
"""

import base64
import hashlib
import hmac
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.crypto import ids
from scout.core.ports.clock import Clock
from scout.core.ports.store import AttrSpec, NamespaceSchema, Row, VectorStore

KEY_PREFIX = "sk_scout_"
KEY_BYTES = 32
KEY_BODY_LEN = 43  # base64url of 32 bytes, unpadded
KEY_LEN = len(KEY_PREFIX) + KEY_BODY_LEN
MAX_PRESENTED_LEN = 256  # longer credentials are never hashed in full
MIN_PEPPER_LEN = 16
KEYS_NAMESPACE = "scoutsys_keys"
ID_LEN = 32

KEYS_SCHEMA = NamespaceSchema(
    dims=None,
    attrs={
        "key_tag": AttrSpec("string", filterable=False),
        "tenant": AttrSpec("string"),
        "wrapped_k_team": AttrSpec("string", filterable=False),
        "created_at": AttrSpec("int", filterable=False),
        "revoked": AttrSpec("bool"),
    },
)


@dataclass(frozen=True, slots=True)
class Tenant:
    tenant: str
    key_id: str


class Cache(Protocol):
    """Read-through cache for key records (Modal Dict plugs in here). A cache is never authoritative."""

    async def get(self, key_id: str) -> Mapping[str, Any] | None: ...

    async def set(self, key_id: str, record: Mapping[str, Any]) -> None: ...

    async def delete(self, key_id: str) -> None: ...


class NoCache:
    async def get(self, key_id: str) -> Mapping[str, Any] | None:
        return None

    async def set(self, key_id: str, record: Mapping[str, Any]) -> None:
        return None

    async def delete(self, key_id: str) -> None:
        return None


def lookup_tag(pepper: bytes, presented: str) -> str:
    """`HMAC-SHA256(pepper, key)` as hex. Total over any string (malformed keys hash like any other)."""
    data = presented.encode("utf-8", "surrogatepass")[:MAX_PRESENTED_LEN]
    return hmac.new(pepper, data, hashlib.sha256).hexdigest()


def key_shape_ok(presented: str) -> bool:
    if len(presented) != KEY_LEN or not presented.startswith(KEY_PREFIX):
        return False
    body = presented[len(KEY_PREFIX) :]
    try:
        return len(base64.urlsafe_b64decode(body + "=")) == KEY_BYTES and body.isascii()
    except ValueError:
        return False


def _check_tenant(tenant: str) -> str:
    try:
        ids.namespace(tenant, "0" * ids.REPO_ID_LEN, "0" * ids.BRANCH_ID_LEN)  # the one tenant-name rule
    except ValueError as exc:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, str(exc)) from None
    return tenant


class TenantDirectory:
    def __init__(
        self,
        store: VectorStore,
        clock: Clock,
        pepper: bytes,
        *,
        namespace: str = KEYS_NAMESPACE,
        cache: Cache | None = None,
    ) -> None:
        if len(pepper) < MIN_PEPPER_LEN:
            raise ValueError(f"pepper must be at least {MIN_PEPPER_LEN} bytes")
        self._store = store
        self._clock = clock
        self._pepper = pepper
        self.namespace = namespace
        self.cache: Cache = cache or NoCache()

    def tag(self, presented: str) -> str:
        return lookup_tag(self._pepper, presented)

    async def issue_key(self, tenant: str) -> str:
        """Admin only. Returns the plaintext key once; the store gets the tag, never the key."""
        _check_tenant(tenant)
        key = KEY_PREFIX + base64.urlsafe_b64encode(secrets.token_bytes(KEY_BYTES)).rstrip(b"=").decode()
        tag = self.tag(key)
        await self._store.ensure_namespace(self.namespace, KEYS_SCHEMA)
        await self._store.upsert(
            self.namespace,
            [
                Row(
                    id=tag[:ID_LEN],
                    attrs={
                        "key_tag": tag,
                        "tenant": tenant,
                        "wrapped_k_team": "",  # unused on key rows; the team key lives in the "team-" row
                        "created_at": int(self._clock.now()),
                        "revoked": False,
                    },
                )
            ],
        )
        return key

    async def get_record(self, key_id: str) -> Mapping[str, Any] | None:
        """Authoritative lookup. A missing namespace (no key ever issued) is a miss, not an error."""
        if not await self._store.namespace_exists(self.namespace):
            return None
        rows = await self._store.get_by_ids(self.namespace, [key_id])
        return dict(rows[0].attrs) if rows else None

    async def revoke(self, key_id: str) -> bool:
        """Admin only. Marks the record revoked and drops it from this container's cache."""
        record = await self.get_record(key_id)
        if record is None:
            return False
        await self._store.upsert(self.namespace, [Row(id=key_id, attrs={**record, "revoked": True})])
        await self.cache.delete(key_id)
        return True
