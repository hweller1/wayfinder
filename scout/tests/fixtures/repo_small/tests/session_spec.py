from src.auth import getUserSession
from src.auth.session import SESSIONS


def test_unknown_token_has_no_session():
    assert getUserSession("nope") is None


def test_known_token_returns_session():
    SESSIONS["t"] = {"user": "ada"}
    assert getUserSession("t") == {"user": "ada"}
