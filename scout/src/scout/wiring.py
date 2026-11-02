"""Composition root: the only place that picks concrete adapters (design section 1, `wiring`).

Profile `fake` runs the whole system in one process with no network: a fixture repo loaded into
`FakeFileSystem`, a case file of scripted chat turns, `FakeClock`, and a `Gateway` that is the real
`GatewayService` over `FakeStore`/`FakeEmbedder`/`ScriptedChat` (the index is `absent` until `sync` runs).
Profile `default` (alias `real`) is the product: `LocalFileSystem`, `HttpGateway` (the hosted gateway, key
from the key store), `KeychainStore`, `SystemClock`. Profile `embedded` runs the real `GatewayService`
in-process (live tests; needs the gateway env). Profile `fake-http` is the fake system behind an injected
httpx transport (`Config.transport`, an ASGI transport over the gateway app in tests). `Config`
precedence, highest first: CLI flag, env `SCOUT_*`, `~/.config/scout/config.toml`, defaults. Vendor
adapters (turbopuffer, voyage) are imported inside `build_gateway_service` only, so the client process
never loads them. Profile `demo` is the demo lane's composition (`scout.demo.runtime.DemoRuntime`: real
turbopuffer, Voyage and an OpenAI-compatible chat endpoint, repos indexed with `scout-demo index`) behind
the same surfaces, so `scout mcp --profile demo` is a drop-in for `scout-demo mcp`. `scout.demo` sits above
`wiring` in the import layers: wiring reaches it only through the function-level imports of the demo
builders below, and `pyproject.toml` lists exactly those edges under `ignore_imports` of contract 1. Every
heavy import (adapters, the app, crypto, httpx) is inside the function that needs it,
so `import scout.wiring` and `resolve_config` cost almost nothing: `scout mcp` imports this module at startup
and builds the app on its first call (design section 2, startup guard).
"""

import asyncio
import base64
import binascii
import json
import os
import re
import tempfile
import threading
import tomllib
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.chat import ChatTurn, ToolCall

if TYPE_CHECKING:
    import httpx

    from scout.adapters.fake_fs import FakeFileSystem
    from scout.core.app import ScoutApp
    from scout.core.ports.embedder import EmbedResult
    from scout.core.ports.gateway import Gateway
    from scout.core.ports.keystore import KeyStore
    from scout.core.service.auth import AuthService
    from scout.core.service.gateway import GatewayService
    from scout.core.service.syncsvc import ReplayStore
    from scout.core.service.tenants import Cache, TenantDirectory

DEFAULT_GATEWAY_URL = "https://scout-gateway.example.com"
FAKE_ROOT = "/repo"
PROFILES = ("default", "real", "fake", "fake-http", "embedded", "demo")

type CaseSource = str | os.PathLike[str] | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Config:
    """`case` is a path to a JSON case file or the parsed mapping: `{"repo": <dir, relative to the case file>,
    "chat_script": [{"content", "tool_calls": [{"id", "name", "arguments"}], "tokens_in", "tokens_out",
    "finish"}], "state_dir": <optional directory for the sync state; a fresh temp dir by default>}`.
    Optional keys make the other index states and failures reproducible (conformance cases): `sync`
    ("ready" | "partial": index the repo before serving; partial fails the embed of every text matching
    `embed_fail_regex`, in batches of `sync_batch_tokens`), `edit_after_sync` ({path: text}: edited after the
    sync, so a semantic hit on it is stale), `git` (false: not a checkout, `NOT_A_REPO`) and `gateway_error`
    ({"code", "retry_after"}: every gateway call fails with that typed error).

    `gateway_url` None means `DEFAULT_GATEWAY_URL`; `key` None reads the key store (`scout login`); `root`
    None is the working directory; `transport` is only used by profile `fake-http`."""

    profile: str = "default"
    case: CaseSource | None = None
    gateway_url: str | None = None
    key: str | None = None
    root: str | None = None
    transport: "httpx.AsyncBaseTransport | None" = None
    backfill_to: int = (
        # `explore.backfill_to` (config table `[explore]`, env SCOUT_EXPLORE_BACKFILL_TO).
        # Off by default: Addendum E showed no recall@12 gain and lower precision.
        0
    )


def backfill_setting(env: Mapping[str, str], file_vals: Mapping[str, Any] | None = None) -> int:
    """`explore.backfill_to`: env `SCOUT_EXPLORE_BACKFILL_TO` > config file `[explore] backfill_to` > 0 (off).
    0 turns the backfill off; anything that is not a non-negative integer is INVALID_ARGUMENT."""
    raw: Any = env.get("SCOUT_EXPLORE_BACKFILL_TO")
    if raw is None or not str(raw).strip():
        table = (file_vals or {}).get("explore")
        raw = table.get("backfill_to") if isinstance(table, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if raw is None or isinstance(raw, str) and not raw.strip():
        return 0
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = -1
    if isinstance(raw, bool) or not 0 <= n <= 40:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "explore.backfill_to must be an integer from 0 to 40")
    return n


def default_config_path(env: Mapping[str, str] | None = None) -> Path:
    src = os.environ if env is None else env
    return Path(src.get("SCOUT_CONFIG") or Path.home() / ".config" / "scout" / "config.toml")


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError):
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "the scout config file is not readable TOML") from None


def resolve_config(
    flags: Mapping[str, str | None] | None = None,
    env: Mapping[str, str] | None = None,
    config_file: Path | None = None,
    *,
    case: CaseSource | None = None,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> Config:
    """CLI flag > env `SCOUT_<NAME>` > config file > default, per field (`profile`, `gateway_url`, `key`,
    `root`). An empty value counts as unset. Anything but a string in the file is ignored."""
    flags = flags or {}
    environ = os.environ if env is None else env
    file_vals = _read_toml(config_file if config_file is not None else default_config_path(environ))

    def pick(name: str, default: str | None) -> str | None:
        for value in (flags.get(name), environ.get(f"SCOUT_{name.upper()}"), file_vals.get(name)):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    profile = pick("profile", "default") or "default"
    if profile not in PROFILES:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"unknown profile {profile!r}")
    return Config(
        profile=profile,
        case=case,
        gateway_url=pick("gateway_url", None),
        key=pick("key", None),
        root=pick("root", None),
        transport=transport,
        backfill_to=backfill_setting(environ, file_vals),
    )


def _load_case(case: CaseSource) -> tuple[dict[str, Any], Path]:
    if isinstance(case, Mapping):
        return dict(case), Path.cwd()
    path = Path(case)
    try:
        return json.loads(path.read_text(encoding="utf-8")), path.parent
    except (OSError, ValueError):
        raise ScoutError(
            ErrorCode.INVALID_ARGUMENT, "fake case: the case file is not readable JSON"
        ) from None


def _load_repo(repo: Path) -> dict[str, bytes]:
    if not repo.is_dir():
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "fake case: repo is not a directory")
    return {
        p.relative_to(repo).as_posix(): p.read_bytes()
        for p in sorted(repo.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(repo).parts
    }


def _turn(d: Mapping[str, Any]) -> ChatTurn:
    calls = tuple(
        ToolCall(id=str(c["id"]), name=str(c["name"]), arguments=dict(c.get("arguments", {})))
        for c in d.get("tool_calls", ())
    )
    return ChatTurn(
        content=d.get("content"),
        tool_calls=calls,
        tokens_in=int(d.get("tokens_in", 0)),
        tokens_out=int(d.get("tokens_out", 0)),
        finish=str(d.get("finish", "tool_calls" if calls else "stop")),
    )


def _fake_parts(cfg: Config, profile: str) -> "tuple[FakeFileSystem, Any, Any]":
    from scout.adapters.fake_chat import ScriptedChat
    from scout.adapters.fake_clock import FakeClock
    from scout.adapters.fake_fs import FakeFileSystem

    if cfg.case is None:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"profile {profile!r} needs a case")
    case, base = _load_case(cfg.case)
    state_dir = str(case.get("state_dir") or tempfile.mkdtemp(prefix="scout-fake-state-"))
    extra: dict[str, Any] = {"git": None} if case.get("git") is False else {}
    fs = FakeFileSystem(_load_repo(base / case["repo"]), root=FAKE_ROOT, state_dir=state_dir, **extra)
    return fs, ScriptedChat([_turn(t) for t in case.get("chat_script", ())]), FakeClock()


def _doomed_embedder(pattern: str) -> Any:
    """A `FakeEmbedder` failing the embed of any batch with a text matching `pattern` (a partial index)."""
    from scout.adapters.fake_embedder import FakeEmbedder

    rx = re.compile(pattern)

    class Doomed(FakeEmbedder):
        async def embed(
            self, texts: Sequence[str], *, input_type: Literal["document", "query"]
        ) -> "EmbedResult":
            if any(rx.search(t) for t in texts):
                raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "embedding upstream 503")
            return await super().embed(texts, input_type=input_type)

    return Doomed()


class _ErroringGateway:
    """Delegates to a gateway, but every identity call (all that explore, search and status start with) fails
    with `code`: the typed-error cases of the conformance suite."""

    def __init__(self, inner: Any, code: ErrorCode, retry_after: int | None) -> None:
        self._inner = inner
        self._err = (code, {"retry_after": retry_after} if retry_after else None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def whoami(self) -> Any:
        raise ScoutError(self._err[0], "scripted gateway failure", details=self._err[1])


def _run_blocking[T](coro: Awaitable[T]) -> T:
    """Run `coro` to completion from sync code, also when the caller already sits in a running event loop."""

    async def run() -> T:
        return await coro

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run())
    box: list[Any] = []

    def work() -> None:
        try:
            box.append((True, asyncio.run(run())))
        except BaseException as e:  # noqa: BLE001 - re-raised below on the caller's thread
            box.append((False, e))

    t = threading.Thread(target=work)
    t.start()
    t.join()
    ok, value = box[0]
    if not ok:
        raise value
    return value  # type: ignore[no-any-return]


def _fake_system(cfg: Config) -> "tuple[Any, FakeFileSystem, Any, Any, dict[str, Any]]":
    """`(gateway, fs, clock, sync_config, case)` of the fake profile: the system `_build_fake` serves and
    `build_login` verifies a key against."""
    from scout.adapters.fake_gateway import make_fake_gateway
    from scout.core.sync.engine import SyncConfig

    fs, chat, clock = _fake_parts(cfg, "fake")
    case, _ = _load_case(cfg.case)  # type: ignore[arg-type]  # _fake_parts rejected a missing case
    embedder = _doomed_embedder(case["embed_fail_regex"]) if case.get("embed_fail_regex") else None
    gateway: Any = make_fake_gateway(chat=chat, clock=clock, embedder=embedder)
    sync_config = (
        SyncConfig(batch_tokens=int(case["sync_batch_tokens"]), workers=2, retries=1)
        if case.get("sync_batch_tokens")
        else None
    )
    if case.get("gateway_error"):
        err = case["gateway_error"]
        gateway = _ErroringGateway(gateway, ErrorCode(err["code"]), err.get("retry_after"))
    return gateway, fs, clock, sync_config, case


def _build_fake(cfg: Config) -> "ScoutApp":
    from scout.adapters.fake_keystore import FakeKeyStore
    from scout.core.app import ScoutApp

    gateway, fs, clock, sync_config, case = _fake_system(cfg)
    app = ScoutApp(
        gateway,
        fs,
        clock,
        FakeKeyStore(),
        sync_config=sync_config,
        backfill_to=int(
            case.get("backfill_to", cfg.backfill_to)
        ),  # a case may pin it (fake ties are per process)
    )
    if case.get("sync"):
        status = _run_blocking(app.sync())
        if status.state != case["sync"]:
            raise ScoutError(
                ErrorCode.INVALID_ARGUMENT, f"fake case: sync ended {status.state}, not {case['sync']}"
            )
        for path, text in (case.get("edit_after_sync") or {}).items():
            fs.files[path] = text.encode()
    return app


_REQUIRED_GATEWAY_ENV = (
    "TURBOPUFFER_API_KEY",
    "VOYAGE_API_KEY",
    "SCOUT_MASTER_KEY",
    "SCOUT_PEPPER",
    "SCOUT_LLM_URL",
)


def _b64_secret(env: Mapping[str, str], name: str, *, exact: int | None = None, minimum: int = 16) -> bytes:
    try:
        raw = base64.b64decode(env[name], validate=True)
    except (KeyError, ValueError, binascii.Error):
        raise ScoutError(
            ErrorCode.INTERNAL, f"gateway service is not configured: {name} is not base64"
        ) from None
    if (exact is not None and len(raw) != exact) or len(raw) < minimum:
        raise ScoutError(
            ErrorCode.INTERNAL, f"gateway service is not configured: {name} has the wrong length"
        )
    return raw


def keys_namespace(env: Mapping[str, str]) -> str:
    """`SCOUT_KEYS_NS` (live deployments use `scout_dev_keys` so the janitor prefix covers it)."""
    from scout.core.service.tenants import KEYS_NAMESPACE

    return env.get("SCOUT_KEYS_NS") or KEYS_NAMESPACE


def build_gateway_service(
    env: Mapping[str, str], *, replays: "ReplayStore | None" = None
) -> "GatewayService":
    """The gateway process's composition: real turbopuffer store, Voyage embedder and OpenAI-compatible chat
    from `env` (the Modal secrets). Fails closed with a typed error naming the missing variables (never their
    values). Vendor adapters are imported here, not at module level."""
    missing = [n for n in _REQUIRED_GATEWAY_ENV if not env.get(n)]
    if missing:
        raise ScoutError(
            ErrorCode.INTERNAL, "gateway service is not configured: missing " + ", ".join(missing)
        )
    from scout.core.crypto.keys import MASTER_KEY_LEN
    from scout.core.service.gateway import GatewayService

    master = _b64_secret(env, "SCOUT_MASTER_KEY", exact=MASTER_KEY_LEN)
    from scout.adapters.clock_system import SystemClock
    from scout.adapters.openai_chat import OpenAIChat
    from scout.adapters.turbopuffer import TurbopufferStore
    from scout.adapters.voyage import VoyageEmbedder

    clock = SystemClock()
    return GatewayService(
        TurbopufferStore.from_env(env),
        clock,
        master,
        keys_namespace=keys_namespace(env),
        embedder=VoyageEmbedder.from_env(env, clock=clock),
        chat=OpenAIChat.from_env(env),
        replays=replays,
    )


def build_auth(
    service: "GatewayService", env: Mapping[str, str], cache: "Cache | None" = None
) -> "tuple[TenantDirectory, AuthService]":
    """Key directory and `AuthService` over the service's own store and clock (the key records live in the
    keys namespace beside the tenants' data)."""
    from scout.core.service.auth import AuthService
    from scout.core.service.tenants import TenantDirectory

    pepper = _b64_secret(env, "SCOUT_PEPPER")
    directory = TenantDirectory(
        service.store, service.clock, pepper, namespace=service.keys_namespace, cache=cache
    )
    return directory, AuthService(directory, service.clock)


def _client_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("scout")
    except metadata.PackageNotFoundError:
        return "0"


def _stored_key(keystore: "KeyStore", explicit: str | None) -> str | None:
    from scout.core.app import KEY_NAME

    if explicit:
        return explicit
    raw = keystore.get(KEY_NAME)
    return raw.decode("utf-8").strip() if raw else None


def _build_real(cfg: Config, env: Mapping[str, str]) -> "ScoutApp":
    """No network at construction: the gateway client connects on first call."""
    from scout.adapters.clock_system import SystemClock
    from scout.adapters.fs_local import LocalFileSystem
    from scout.adapters.gateway_http import HttpGateway
    from scout.adapters.keychain import KeychainStore
    from scout.core.app import ScoutApp

    keystore = KeychainStore(env=env)
    gateway = HttpGateway(
        cfg.gateway_url or DEFAULT_GATEWAY_URL,
        lambda: _stored_key(keystore, cfg.key),
        _client_version(),
    )
    return ScoutApp(
        gateway,
        LocalFileSystem(cfg.root or os.getcwd()),
        SystemClock(),
        keystore,
        backfill_to=cfg.backfill_to,
    )


def _build_embedded(cfg: Config, env: Mapping[str, str]) -> "ScoutApp":
    """The real `GatewayService` in this process (live tests). Tenant from `SCOUT_TENANT` (default `dev`)."""
    from scout.adapters.clock_system import SystemClock
    from scout.adapters.fs_local import LocalFileSystem
    from scout.adapters.keychain import KeychainStore
    from scout.core.app import ScoutApp

    gateway: Gateway = build_gateway_service(env).for_tenant(env.get("SCOUT_TENANT") or "dev")
    return ScoutApp(gateway, LocalFileSystem(cfg.root or os.getcwd()), SystemClock(), KeychainStore(env=env),
        backfill_to=cfg.backfill_to,
    )  # fmt: skip


def _build_fake_http(cfg: Config) -> "ScoutApp":
    """The fake profile's repo, but the gateway is an `HttpGateway` over `cfg.transport` (an ASGI transport
    over the gateway app, whose service owns the chat script)."""
    from scout.adapters.fake_keystore import FakeKeyStore
    from scout.adapters.gateway_http import HttpGateway
    from scout.core.app import ScoutApp

    if cfg.transport is None:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "profile 'fake-http' needs an injected transport")
    fs, _, clock = _fake_parts(cfg, "fake-http")
    gateway = HttpGateway(
        cfg.gateway_url or "http://gateway.test", cfg.key or "sk_scout_fake", "0", transport=cfg.transport
    )
    case, _ = _load_case(cfg.case)  # type: ignore[arg-type]  # _fake_parts rejected a missing case
    return ScoutApp(
        gateway, fs, clock, FakeKeyStore(), backfill_to=int(case.get("backfill_to", cfg.backfill_to))
    )


# --- profile demo ---------------------------------------------------------------------------------------
class _DemoApp:
    """The demo's `ScoutApp` for one indexed repo. Delegates everything, except that `explore` first reports a
    missing chat model with the demo's own full message (the gateway would replace it with fixed text), so
    the primitives keep working on a host with no model configured."""

    def __init__(self, app: "ScoutApp", model_error: ScoutError | None) -> None:
        self._app = app
        self._model_error = model_error

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)

    async def explore(self, req: Any) -> Any:
        if self._model_error is not None:
            raise self._model_error
        return await self._app.explore(req)


def build_demo_runtime() -> Any:
    """A `DemoRuntime` from the process environment and the demo `.env` files; no network at construction."""
    from scout.demo.runtime import DemoRuntime

    return DemoRuntime()


def build_demo_app(cfg: Config, runtime: Any = None) -> Any:
    """The app of the demo repo checked out at `cfg.root` (default: the working directory), which must have
    been indexed with `scout-demo index` (or the `index_repo` tool). `runtime` is a seam for tests."""
    from scout.demo.runtime import MeteredEmbedder, resolve_model

    rt = runtime if runtime is not None else build_demo_runtime()
    root = os.path.realpath(cfg.root or os.getcwd())
    repos = rt.list_repos()
    rec = next((r for r in repos if os.path.realpath(r.root) == root), None)
    if rec is None:
        known = ", ".join(r.name for r in repos) or "none"
        raise ScoutError(
            ErrorCode.NOT_FOUND,
            f"{root} is not an indexed demo repo: run `scout-demo index {root}` (indexed: {known})",
        )
    model_error: ScoutError | None = None
    chat = None
    try:
        chat = rt.chat_for(resolve_model(rt.env))
    except ScoutError as e:
        model_error = e
    embedder = MeteredEmbedder(rt._real_embedder(), lambda _msg: None)  # noqa: SLF001
    return _DemoApp(rt._app(rec.root, rec.name, chat, embedder), model_error)  # noqa: SLF001


def build_demo_mcp(runtime: Any = None) -> Any:
    """The demo lane's MCP server (`explore(brief, repo)`, `index_repo`, `list_repos`) over its runtime.
    Importing it here keeps the `mcp` package out of wiring and `scout.demo` out of the surfaces."""
    from scout.demo.mcp_server import build_server

    return build_server(runtime if runtime is not None else build_demo_runtime())


def build_login(cfg: Config, env: Mapping[str, str] | None = None) -> "tuple[Gateway, KeyStore]":
    """What `scout login` needs: the gateway to verify `cfg.key` against and the key store to save it in.

    The key store is the real one (`KeychainStore`, so `HOME` and `SCOUT_KEYSTORE=file` scope it) in every
    profile; profile `fake` verifies against the fake gateway, which accepts any key."""
    from scout.adapters.keychain import KeychainStore

    environ = os.environ if env is None else env
    keystore = KeychainStore(env=environ)
    if cfg.profile == "fake":
        return _fake_system(cfg)[0], keystore
    if cfg.profile in ("default", "real"):
        from scout.adapters.gateway_http import HttpGateway

        if not cfg.key:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "login needs a key")
        return HttpGateway(cfg.gateway_url or DEFAULT_GATEWAY_URL, cfg.key, _client_version()), keystore
    if cfg.profile == "embedded":
        return build_gateway_service(environ).for_tenant(environ.get("SCOUT_TENANT") or "dev"), keystore
    raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"profile {cfg.profile!r} cannot log in")


def build_app(cfg: Config, env: Mapping[str, str] | None = None) -> "ScoutApp":
    environ = os.environ if env is None else env
    if cfg.profile == "fake":
        return _build_fake(cfg)
    if cfg.profile in ("default", "real"):
        return _build_real(cfg, environ)
    if cfg.profile == "embedded":
        return _build_embedded(cfg, environ)
    if cfg.profile == "fake-http":
        return _build_fake_http(cfg)
    if cfg.profile == "demo":
        return build_demo_app(cfg)  # type: ignore[no-any-return]
    raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"profile {cfg.profile!r} is not available")
