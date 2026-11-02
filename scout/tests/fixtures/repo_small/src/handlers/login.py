from src.auth import getUserSession


def handle_login(token: str) -> int:
    session = getUserSession(token)
    if session is None:
        return 401
    return 200
