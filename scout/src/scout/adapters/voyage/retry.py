"""Bounded retry with exponential backoff for Voyage calls (design section 13, AC-5.5).

`retries` counts retries AFTER the first call, so `retries=6` makes up to 7 calls and sleeps
`min(2**n, 60)` for n = 0..5, i.e. 1, 2, 4, 8, 16, 32 s (the cap only bites for larger `retries`).
Sleeping goes through an injected `sleep` (the `Clock` port's), so tests never wait.
"""

from collections.abc import Awaitable, Callable

from scout.core.contracts.errors import ErrorCode, ScoutError

__all__ = ["backoff_delay", "call_with_retry", "is_retryable", "to_scout_error"]

BACKOFF_CAP_S = 60.0

type Sleep = Callable[[float], Awaitable[None]]

_RETRYABLE_NAMES = frozenset(
    {
        "RateLimitError",
        "Timeout",
        "APIConnectionError",
        "ServerError",
        "ServiceUnavailableError",
        "TryAgain",
        "TimeoutException",  # httpx
        "TransportError",  # httpx
    }
)


def backoff_delay(n: int) -> float:
    """Seconds to wait before retry number n+1 (n starts at 0): `min(2**n, 60)`."""
    return min(float(2**n), BACKOFF_CAP_S)


def _names(exc: BaseException) -> set[str]:
    return {c.__name__ for c in type(exc).__mro__}


def is_retryable(exc: BaseException) -> bool:
    """429, timeouts, connection faults and 5xx are retried; auth, bad requests and our own errors are not."""
    if isinstance(exc, ScoutError):
        return False
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    names = _names(exc)
    if names & _RETRYABLE_NAMES:
        return True
    status = getattr(exc, "http_status", None)
    return "APIError" in names and (status is None or (isinstance(status, int) and status >= 500))


def to_scout_error(exc: BaseException) -> ScoutError:
    """Fixed-text typed error; the vendor message and body are never echoed."""
    if isinstance(exc, ScoutError):
        return exc
    names = _names(exc)
    if "AuthenticationError" in names:
        return ScoutError(ErrorCode.INTERNAL, "the embedding service rejected the configured credentials")
    if names & {"InvalidRequestError", "MalformedRequestError"}:
        return ScoutError(ErrorCode.INVALID_ARGUMENT, "the embedding request was rejected")
    if is_retryable(exc):
        return ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the embedding service is unavailable")
    return ScoutError(ErrorCode.INTERNAL, "the embedding adapter failed")


async def call_with_retry[T](op: Callable[[], Awaitable[T]], *, retries: int, sleep: Sleep) -> T:
    """Run `op`; on a retryable failure sleep `backoff_delay(n)` and try again, up to `retries` times.
    Terminal failures (and exhausted retries) raise a typed `ScoutError`."""
    n = 0
    while True:
        try:
            return await op()
        except Exception as exc:
            if not is_retryable(exc) or n >= retries:
                raise to_scout_error(exc) from None
        await sleep(backoff_delay(n))
        n += 1
