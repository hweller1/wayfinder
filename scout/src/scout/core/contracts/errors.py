"""Error model (design section 4): codes, the raised exception, the wire body, and the mapping tables."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scout.core.contracts.types import SCHEMA_VERSION, SCHEMA_VERSION_PATTERN


class ErrorCode(StrEnum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    NOT_A_REPO = "NOT_A_REPO"
    PATH_REFUSED = "PATH_REFUSED"
    RATE_LIMITED = "RATE_LIMITED"
    CONFLICT = "CONFLICT"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    INTERNAL = "INTERNAL"


_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_ARGUMENT: 400,
    ErrorCode.UNSUPPORTED_SCHEMA: 422,
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.LOGIN_REQUIRED: 401,
    ErrorCode.PERMISSION_DENIED: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.NOT_A_REPO: 400,
    ErrorCode.PATH_REFUSED: 403,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.UPSTREAM_UNAVAILABLE: 503,
    ErrorCode.UPSTREAM_TIMEOUT: 504,
    ErrorCode.CONFLICT: 409,
    ErrorCode.INTERNAL: 500,
}

_CLI_EXIT: dict[ErrorCode, int] = {
    ErrorCode.INVALID_ARGUMENT: 2,
    ErrorCode.UNSUPPORTED_SCHEMA: 2,
    ErrorCode.UNAUTHENTICATED: 3,
    ErrorCode.LOGIN_REQUIRED: 3,
    ErrorCode.PERMISSION_DENIED: 3,
    ErrorCode.NOT_FOUND: 4,
    ErrorCode.NOT_A_REPO: 4,
    ErrorCode.PATH_REFUSED: 4,
    ErrorCode.RATE_LIMITED: 5,
    ErrorCode.UPSTREAM_UNAVAILABLE: 6,
    ErrorCode.UPSTREAM_TIMEOUT: 6,
    ErrorCode.CONFLICT: 7,
    ErrorCode.INTERNAL: 1,
}

_RETRYABLE: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.RATE_LIMITED,
        ErrorCode.UPSTREAM_UNAVAILABLE,
        ErrorCode.UPSTREAM_TIMEOUT,
        ErrorCode.CONFLICT,
    }
)


def http_status(code: ErrorCode) -> int:
    return _HTTP_STATUS[code]


def cli_exit(code: ErrorCode) -> int:
    return _CLI_EXIT[code]


def is_retryable(code: ErrorCode) -> bool:
    return code in _RETRYABLE


class ErrorInfo(BaseModel):
    """The `error` object of the wire body."""

    model_config = ConfigDict(extra="ignore")

    code: ErrorCode
    message: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorBody(BaseModel):
    """Wire body (HTTP, CLI stdout on failure): `{"schema_version","error":{...}}`."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default=SCHEMA_VERSION, pattern=SCHEMA_VERSION_PATTERN)
    error: ErrorInfo


class ScoutError(Exception):
    """Raised for failures crossing a port or surface boundary. `details` never carries code or paths."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = is_retryable(code) if retryable is None else retryable
        self.details: dict[str, Any] = dict(details) if details else {}

    def to_body(self, request_id: str | None = None) -> ErrorBody:
        return ErrorBody(
            error=ErrorInfo(
                code=self.code,
                message=self.message,
                retryable=self.retryable,
                details=self.details,
                request_id=request_id,
            )
        )
