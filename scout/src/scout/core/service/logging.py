"""Body-free request log (design section 6, PRD section 5): ids, sizes, status, latency and token counts only.

Neither the gateway nor scout-llm logs prompts, tool results, queries, chunk text or paths. The only way the
service logs a request is `request_log`, whose fields are numbers, a fixed operation name, an id and a status
code; there is no parameter that accepts text from a request. Failures log the status code and the exception
CLASS name, never its message (messages may quote a prompt).
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.clock import Clock

LOGGER = logging.getLogger("scout.service.request")


@dataclass(slots=True)
class Usage:
    """Metering the caller fills in as it learns it (all integers; never content)."""

    messages: int = 0
    tools: int = 0
    chars: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None


@contextmanager
def request_log(op: str, *, key_id: str, clock: Clock) -> Generator[Usage]:
    """Time one request and emit a single INFO record on exit (also on failure, with the status code)."""
    usage = Usage()
    started = clock.monotonic()
    status = "ok"
    error_type = ""
    try:
        yield usage
    except ScoutError as exc:
        status, error_type = str(exc.code), type(exc).__name__
        raise
    except Exception as exc:
        status, error_type = str(ErrorCode.INTERNAL), type(exc).__name__
        raise
    finally:
        LOGGER.info(
            "op=%s key=%s status=%s latency_ms=%.1f messages=%d tools=%d chars=%d"
            " tokens_in=%s tokens_out=%s err=%s",
            op,
            key_id,
            status,
            (clock.monotonic() - started) * 1000.0,
            usage.messages,
            usage.tools,
            usage.chars,
            usage.tokens_in,
            usage.tokens_out,
            error_type or "-",
        )
