from src.auth.session import SESSIONS


def handle_logout(token: str) -> int:
    SESSIONS.pop(token, None)
    return 204
