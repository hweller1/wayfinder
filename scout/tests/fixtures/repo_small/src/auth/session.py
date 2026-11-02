"""Session lookup for the auth package."""

SESSIONS: dict[str, dict[str, str]] = {}


def getUserSession(token: str) -> dict[str, str] | None:
    """Return the session for a bearer token, or None when it is unknown."""
    session = SESSIONS.get(token)
    if session is None:
        return None
    return session


def clear_sessions() -> None:
    SESSIONS.clear()
