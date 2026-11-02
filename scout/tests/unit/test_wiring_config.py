"""Task 7.7: config precedence and the real composition root. FR-S8, FR-P1.

Precedence is CLI flag > env `SCOUT_*` > config.toml > default, per field. The real profile builds with no
network and without importing the vendor SDKs; `build_gateway_service` fails closed on missing env.
"""

import base64
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.wiring import DEFAULT_GATEWAY_URL, Config, build_app, build_gateway_service, resolve_config

B64_MASTER = base64.b64encode(b"m" * 32).decode()
B64_PEPPER = base64.b64encode(b"p" * 32).decode()


def _toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(body))
    return p


def test_default_gateway_url_is_the_modal_app() -> None:
    assert DEFAULT_GATEWAY_URL == "https://scout-gateway.example.com"
    cfg = resolve_config(env={}, config_file=Path("/nonexistent/config.toml"))
    assert (cfg.profile, cfg.gateway_url, cfg.key, cfg.root) == ("default", None, None, None)


@pytest.mark.parametrize(
    ("flags", "env", "file", "want"),
    [
        ({"gateway_url": "flag"}, {"SCOUT_GATEWAY_URL": "env"}, 'gateway_url = "file"', "flag"),
        ({}, {"SCOUT_GATEWAY_URL": "env"}, 'gateway_url = "file"', "env"),
        ({}, {}, 'gateway_url = "file"', "file"),
        ({}, {}, "", None),
        ({"gateway_url": ""}, {"SCOUT_GATEWAY_URL": "  "}, 'gateway_url = "file"', "file"),
        ({"gateway_url": None}, {}, "gateway_url = 7", None),
    ],
)
def test_precedence_matrix(
    tmp_path: Path, flags: dict[str, str | None], env: dict[str, str], file: str, want: str | None
) -> None:
    cfg = resolve_config(flags, env, _toml(tmp_path, file))
    assert cfg.gateway_url == want


def test_precedence_is_per_field(tmp_path: Path) -> None:
    cfg = resolve_config(
        {"key": "sk_flag"},
        {"SCOUT_GATEWAY_URL": "https://env.test", "SCOUT_ROOT": "/env/root"},
        _toml(tmp_path, 'key = "sk_file"\nroot = "/file/root"\nprofile = "embedded"\n'),
    )
    assert (cfg.key, cfg.gateway_url, cfg.root, cfg.profile) == (
        "sk_flag",
        "https://env.test",
        "/env/root",
        "embedded",
    )


def test_unknown_profile_and_bad_toml_are_typed(tmp_path: Path) -> None:
    with pytest.raises(ScoutError) as ei:
        resolve_config({"profile": "nope"}, {}, tmp_path / "none.toml")
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError) as ei2:
        resolve_config({}, {}, _toml(tmp_path, "not = = toml"))
    assert ei2.value.code is ErrorCode.INVALID_ARGUMENT


def test_scout_config_env_selects_the_file(tmp_path: Path) -> None:
    p = _toml(tmp_path, 'gateway_url = "https://via-env-file.test"')
    assert resolve_config(env={"SCOUT_CONFIG": str(p)}).gateway_url == "https://via-env-file.test"


def test_build_app_real_profile_needs_no_network_and_no_vendor_sdk(tmp_path: Path) -> None:
    """In a fresh interpreter: constructing the real app imports neither turbopuffer nor voyageai."""
    code = textwrap.dedent(
        f"""
        import sys
        from scout.wiring import Config, build_app
        env = {{"SCOUT_KEYSTORE": "file", "HOME": {str(tmp_path)!r}}}
        app = build_app(Config(root={str(tmp_path)!r}), env)
        bad = [m for m in ("turbopuffer", "voyageai") if m in sys.modules]
        assert not bad, bad
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "ok"


async def test_real_profile_without_a_key_is_login_required(tmp_path: Path) -> None:
    app = build_app(
        Config(root=str(tmp_path)),
        {"SCOUT_KEYSTORE": "file", "HOME": str(tmp_path)},
    )
    # an unreachable key store (tmp HOME has no credentials) -> the gateway client refuses before any request
    with pytest.raises(ScoutError) as ei:
        await app.sync()
    assert ei.value.code in (ErrorCode.LOGIN_REQUIRED, ErrorCode.NOT_A_REPO)


def test_build_app_unknown_profile() -> None:
    with pytest.raises(ScoutError) as ei:
        build_app(Config(profile="nope"))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


def test_fake_http_needs_a_transport() -> None:
    with pytest.raises(ScoutError) as ei:
        build_app(Config(profile="fake-http", case={"repo": "."}))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


FULL_ENV = {
    "TURBOPUFFER_API_KEY": "t",
    "VOYAGE_API_KEY": "v",
    "SCOUT_MASTER_KEY": B64_MASTER,
    "SCOUT_PEPPER": B64_PEPPER,
    "SCOUT_LLM_URL": "https://llm.test/v1",
}


@pytest.mark.parametrize("missing", sorted(FULL_ENV))
def test_build_gateway_service_fails_closed_on_missing_env(missing: str) -> None:
    env = {k: v for k, v in FULL_ENV.items() if k != missing}
    with pytest.raises(ScoutError) as ei:
        build_gateway_service(env)
    assert ei.value.code is ErrorCode.INTERNAL
    assert missing in ei.value.message
    assert FULL_ENV[missing] not in ei.value.message or len(FULL_ENV[missing]) < 3


def test_build_gateway_service_rejects_bad_master_key() -> None:
    for bad in ("not base64!!", base64.b64encode(b"short").decode()):
        with pytest.raises(ScoutError) as ei:
            build_gateway_service({**FULL_ENV, "SCOUT_MASTER_KEY": bad})
        assert ei.value.code is ErrorCode.INTERNAL
        assert bad not in ei.value.message


def test_build_gateway_service_with_full_env_and_keys_namespace_default() -> None:
    svc = build_gateway_service(FULL_ENV)
    assert svc.keys_namespace == "scoutsys_keys"
    svc2 = build_gateway_service({**FULL_ENV, "SCOUT_KEYS_NS": "scout_dev_keys"})
    assert svc2.keys_namespace == "scout_dev_keys"
