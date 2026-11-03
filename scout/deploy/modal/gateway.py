"""Modal deployment of the Scout gateway: one CPU ASGI app, the single auth point. Runs on your own Modal account.

Set up first (repository root): the secrets, then the model app, see the README section "Run on Modal (your own
account)". `deploy/modal/make_secrets.py` creates the one secret this app needs, `scout-gateway`.

    cd scout
    modal deploy deploy/modal/gateway.py                        # prints https://<workspace>--scout-gateway-api.modal.run
    modal run deploy/modal/gateway.py::issue_key --tenant dev   # prints a scout key once; stores its tag only

The key is printed once and cannot be recovered; give it to one person or machine (`scout login --key <key>`).
Run `issue_key` again with another `--tenant` name for each team member; tenants cannot see each other's index.

The Modal secret `scout-gateway` holds every vendor credential: TURBOPUFFER_API_KEY, VOYAGE_API_KEY,
SCOUT_MASTER_KEY, SCOUT_PEPPER, SCOUT_KEYS_NS and the chat upstream SCOUT_LLM_URL / SCOUT_LLM_API_KEY /
SCOUT_LLM_MODEL / SCOUT_LLM_SAMPLING (your `deploy/modal/serve_glimmer.py` endpoint). A client needs only a scout
key. To point the gateway at another OpenAI-compatible model, recreate the secret with other SCOUT_LLM_* values.
`SCOUT_GATEWAY_APP=<name>` deploys under another app name (default `scout-gateway`), for a second gateway.

The app scales to zero (`min_containers=0`): the first request after idle pays a container start (a few
seconds), and the idle app costs nothing. The gateway's `/v1/warm` calls the model's `-warm` endpoint, derived
from SCOUT_LLM_URL.

Key records and idempotency replays live in a Modal Dict (a cache only: the keys namespace in turbopuffer is
authoritative, row ids and the root CAS keep a lost replay entry correct). The deploy builds a wheel of this
repo with `uv build` (install uv first) and installs it in the image, so the deployed code is the code in this
working tree.
"""

import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent  # in the container the file sits in /root
WHEEL_DIR = REPO / ".cache" / "gateway-wheel"
REMOTE_DIR = "/opt/scout-gateway"
CACHE_DICT = "scout-gateway-cache"
KEY_PREFIX = "key:"
IDEM_PREFIX = "idem:"


APP_NAME = os.environ.get("SCOUT_GATEWAY_APP", "scout-gateway")


def _build_wheel() -> Path:
    """`uv build --wheel` of this repo into a clean directory; exactly one wheel results."""
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    for old in WHEEL_DIR.glob("*.whl"):
        old.unlink()
    subprocess.run(  # noqa: S603
        ["uv", "build", "--wheel", "--out-dir", str(WHEEL_DIR), str(REPO)],  # noqa: S607
        check=True,
        capture_output=True,
        cwd=REPO,
    )
    wheels = sorted(WHEEL_DIR.glob("scout-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one scout wheel in {WHEEL_DIR}, found {len(wheels)}")
    return wheels[0]


def _image() -> modal.Image:
    # The app name is baked into the image env: the container re-imports this file without the deploy-time
    # environment and must declare the same app as the deploy did.
    base = modal.Image.debian_slim(python_version="3.12").env({"SCOUT_GATEWAY_APP": APP_NAME})
    if not modal.is_local():
        return base  # the container's image is already built; nothing to add at runtime
    wheel = _build_wheel()
    remote = f"{REMOTE_DIR}/{wheel.name}"  # a wheel file keeps its real name: pip parses the version from it
    return base.add_local_file(wheel, remote, copy=True).uv_pip_install(f"scout[gateway] @ file://{remote}")


app = modal.App(APP_NAME)
image = _image()
secrets = [modal.Secret.from_name("scout-gateway")]


class DictKeyCache:
    """`core.service.tenants.Cache` (key records) over a Modal Dict. A Dict failure is a miss or a skipped
    write, never an error: the keys namespace in turbopuffer is authoritative."""

    def __init__(self, store: Any) -> None:
        self._d = store

    async def get(self, key_id: str) -> Mapping[str, Any] | None:
        with contextlib.suppress(Exception):
            item = await self._d.get.aio(KEY_PREFIX + key_id)
            return item if isinstance(item, dict) else None
        return None

    async def set(self, key_id: str, record: Mapping[str, Any]) -> None:
        with contextlib.suppress(Exception):
            await self._d.put.aio(KEY_PREFIX + key_id, dict(record))

    async def delete(self, key_id: str) -> None:
        with contextlib.suppress(Exception):
            await self._d.pop.aio(KEY_PREFIX + key_id, None)


class DictIdempotency:
    """`surfaces.gateway_http.app.IdempotencyCache` over a Modal Dict; entries carry their own expiry."""

    def __init__(self, store: Any, now: Callable[[], float] = time.time) -> None:
        self._d = store
        self._now = now

    async def get(self, key: str) -> str | None:
        with contextlib.suppress(Exception):
            item = await self._d.get.aio(IDEM_PREFIX + key)
            if isinstance(item, tuple) and len(item) == 2 and item[0] > self._now():  # pyright: ignore[reportUnknownArgumentType]
                return str(item[1])  # pyright: ignore[reportUnknownArgumentType]
        return None

    async def set(self, key: str, value: str, ttl_s: float) -> None:
        with contextlib.suppress(Exception):
            await self._d.put.aio(IDEM_PREFIX + key, (self._now() + ttl_s, value))


def _version() -> str:
    from importlib import metadata

    try:
        return metadata.version("scout")
    except metadata.PackageNotFoundError:
        return "0"


@app.function(
    image=image,
    secrets=secrets,
    min_containers=0,
    max_containers=3,
    cpu=1,
    memory=1024,
    timeout=900,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api() -> Any:
    from scout.surfaces.gateway_http.app import create_gateway_app
    from scout.wiring import build_auth, build_gateway_service

    env = dict(os.environ)
    store = modal.Dict.from_name(CACHE_DICT, create_if_missing=True)
    service = build_gateway_service(env)
    idempotency = DictIdempotency(store)
    _, auth = build_auth(service, env, cache=DictKeyCache(store))
    return create_gateway_app(service, auth, cache=idempotency, version=_version())


@app.function(image=image, secrets=secrets, timeout=120)
async def _issue(tenant: str) -> str:
    from scout.wiring import build_auth, build_gateway_service

    env = dict(os.environ)
    service = build_gateway_service(env)
    directory, _ = build_auth(service, env)
    return await directory.issue_key(tenant)


@app.local_entrypoint()
def issue_key(tenant: str) -> None:
    """Issue a key for `tenant` (`[A-Za-z0-9-]`, at most 24 chars). The plaintext is printed once and is not
    recoverable: the store holds only its HMAC tag."""
    key = _issue.remote(tenant)
    sys.stdout.write(f"tenant={tenant}\nkey={key}\n")
