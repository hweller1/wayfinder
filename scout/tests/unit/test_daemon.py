"""Task 6.3: the local HTTP daemon over the fake profile, via `TestClient` (FR-S5).

Token required (401), non-loopback bind refused, each route returns schema-valid JSON equal to the CLI's,
error codes map to the HTTP status of design section 4, token file is 0600.
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from scout.core.contracts.errors import ErrorCode, ScoutError, http_status
from scout.surfaces.daemon.server import (
    ROUTES,
    check_loopback,
    create_daemon,
    default_token_path,
    load_or_create_token,
)
from scout.wiring import Config, build_app

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tests" / "fixtures" / "poc_explore_case.json"
SCHEMA = ROOT / "schema" / "v1"
TOKEN = "t0ken-for-tests"  # noqa: S105
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def validate(name: str, doc: Any) -> None:
    Draft202012Validator(json.loads((SCHEMA / f"{name}.json").read_text())).validate(doc)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_daemon(lambda: build_app(Config(profile="fake", case=CASE)), TOKEN))


def cli(*argv: str) -> str:
    p = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", *argv],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "SCOUT_PROFILE": "fake", "SCOUT_FAKE_CASE": str(CASE)},
    )
    return p.stdout.strip()


@pytest.mark.parametrize("path", sorted(ROUTES))
def test_token_is_required_on_every_route(client: TestClient, path: str) -> None:
    for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": f"Basic {TOKEN}"}):
        r = client.post(path, json={}, headers=headers)
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "UNAUTHENTICATED"
        validate("Error", r.json())
    assert client.get(path, headers=AUTH).status_code == 405  # POST only


def test_each_route_returns_schema_valid_json_equal_to_the_cli(client: TestClient) -> None:
    brief = "where is getUserSession defined?"
    read = {"path": "src/auth/session.py", "start_line": 6, "end_line": 8}
    cases = [
        ("/explore", {"brief": brief}, "ExploreResult", ["explore", brief]),
        ("/status", {}, "IndexStatus", ["status"]),
        ("/grep", {"pattern": "getUserSession"}, "GrepResult", ["grep", "getUserSession"]),
        ("/read", read, "ReadRangeResult", ["read", "src/auth/session.py", "6", "8"]),
        ("/search", {"query": "session lookup"}, "SemanticSearchResult", ["search", "session lookup"]),
    ]
    for path, body, schema, argv in cases:
        r = client.post(path, json=body, headers=AUTH)
        assert r.status_code == 200, (path, r.text)
        assert r.headers["content-type"].startswith("application/json")
        validate(schema, r.json())
        assert r.text == cli(*argv), path  # byte-identical to the CLI


def test_empty_body_is_allowed_for_status(client: TestClient) -> None:
    r = client.post("/status", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["state"] == "absent"


@pytest.mark.parametrize("code", [c for c in ErrorCode if c is not ErrorCode.UNSUPPORTED_SCHEMA])
def test_every_error_code_maps_to_its_http_status(code: ErrorCode) -> None:
    class Failing:
        async def index_status(self, req: Any) -> Any:
            raise ScoutError(code, "boom", details={"retry_after": 7})

    r = TestClient(create_daemon(lambda: Failing(), TOKEN)).post("/status", json={}, headers=AUTH)  # type: ignore[arg-type, return-value]
    assert r.status_code == http_status(code)
    assert r.json()["error"]["code"] == code
    validate("Error", r.json())
    assert ("retry-after" in r.headers) == (code is ErrorCode.RATE_LIMITED)


def test_real_errors_map_end_to_end(client: TestClient) -> None:
    refused = client.post("/read", json={"path": "../etc/passwd"}, headers=AUTH)
    assert (refused.status_code, refused.json()["error"]["code"]) == (403, "PATH_REFUSED")
    bad = client.post("/grep", json={"pattern": "x", "context": 99}, headers=AUTH)
    assert (bad.status_code, bad.json()["error"]["code"]) == (400, "INVALID_ARGUMENT")
    extra = client.post("/status", json={"bogus": 1}, headers=AUTH)  # requests forbid extras
    assert extra.status_code == 400
    notjson = client.post("/grep", content=b"{nope", headers=AUTH)
    assert notjson.status_code == 400
    assert client.post("/grep", json=[1], headers=AUTH).status_code == 400
    big = client.post("/grep", content=b" " * 1_100_000, headers=AUTH)
    assert big.status_code == 400


def test_unexpected_exceptions_leak_nothing() -> None:
    class Boom:
        async def index_status(self, req: Any) -> Any:
            raise RuntimeError("secret /home/x/.env")

    r = TestClient(create_daemon(lambda: Boom(), TOKEN)).post("/status", json={}, headers=AUTH)  # type: ignore[arg-type, return-value]
    assert r.status_code == 500
    assert "secret" not in r.text
    assert r.json()["error"]["message"] == "internal error: RuntimeError"


def test_the_app_is_built_lazily_and_once() -> None:
    built: list[int] = []

    def factory() -> Any:
        built.append(1)
        return build_app(Config(profile="fake", case=CASE))

    c = TestClient(create_daemon(factory, TOKEN))
    assert built == []
    assert c.post("/status", json={}).status_code == 401
    assert built == []  # an unauthenticated call never builds the app
    c.post("/status", json={}, headers=AUTH)
    c.post("/status", json={}, headers=AUTH)
    assert built == [1]


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.5.5.5"])
def test_loopback_hosts_are_accepted(host: str) -> None:
    assert check_loopback(host) == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::", "example.com", "", "10.0.0.1"])  # noqa: S104
def test_non_loopback_bind_is_refused(host: str) -> None:
    with pytest.raises(ScoutError) as ei:
        check_loopback(host)
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


def test_serve_command_refuses_a_non_loopback_host(tmp_path: Path) -> None:
    token = tmp_path / "never-created"
    p = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", "serve", "--host", "0.0.0.0", "--token-file", str(token)],  # noqa: S104
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 2
    assert json.loads(p.stdout)["error"]["code"] == "INVALID_ARGUMENT"
    assert not token.exists()


def test_token_file_is_created_0600_and_reused(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "daemon.token"
    token = load_or_create_token(path)
    assert len(token) == 64
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_or_create_token(path) == token


def test_a_group_readable_token_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "daemon.token"
    path.write_text("secret\n")
    path.chmod(0o644)
    with pytest.raises(ScoutError) as ei:
        load_or_create_token(path)
    assert ei.value.code is ErrorCode.PERMISSION_DENIED
    path.chmod(0o600)
    path.write_text("\n")
    with pytest.raises(ScoutError):
        load_or_create_token(path)


def test_default_token_path_honours_the_env(tmp_path: Path) -> None:
    assert default_token_path({"SCOUT_DAEMON_TOKEN_FILE": str(tmp_path / "t")}) == tmp_path / "t"
    assert default_token_path({}).name == "daemon.token"


def test_an_empty_token_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="token"):
        create_daemon(lambda: None, "")  # type: ignore[arg-type, return-value]
