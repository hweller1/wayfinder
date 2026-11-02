"""Offline checks of the GCP kit: no network, no GPU, no cloud calls. Run with the scout environment:

    cd scout && uv run --extra gateway python -m pytest ../deploy/gcp/tests -q -p no:cacheprovider
"""

from __future__ import annotations

import ast
import asyncio
import base64
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

GCP = Path(__file__).resolve().parents[1]
REPO = GCP.parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, GCP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gateway_vm = _load("gateway_vm")
vllm_args = _load("vllm_args")


def _b64(n: int = 32) -> str:
    return base64.b64encode(bytes(range(n))).decode()


GOOD_ENV = {
    "TURBOPUFFER_API_KEY": "tpuf-test",
    "VOYAGE_API_KEY": "voyage-test",
    "SCOUT_MASTER_KEY": _b64(),
    "SCOUT_PEPPER": _b64(),
    "SCOUT_LLM_URL": "http://127.0.0.1:8000/v1",
    "SCOUT_LLM_API_KEY": "local-test",
    "SCOUT_LLM_MODEL": "wayfinder-glimmer",
    "SCOUT_LLM_SAMPLING": "sft",
}


def _modal_vllm_args() -> list[str]:
    tree = ast.parse((REPO / "training/serve/wayfinder_v3.py").read_text())
    consts = {
        t.id: n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "VLLM_ARGS" for t in node.targets):
            out = []
            for e in node.value.elts:
                if isinstance(e, ast.Constant):
                    out.append(e.value)
                elif isinstance(e, ast.Call) and ast.unparse(e.args[0]) in consts:  # str(MAX_CONCURRENCY)
                    out.append(str(consts[ast.unparse(e.args[0])]))
                else:
                    out.append(ast.unparse(e))
            return out
    raise AssertionError("VLLM_ARGS not found in the Modal serving app")


def _pairs(args: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            nxt = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else None
            out[args[i]] = nxt
            i += 2 if nxt is not None else 1
        else:
            i += 1
    return out


def test_vllm_flags_match_the_modal_serving_app():
    modal = _pairs([a for a in _modal_vllm_args() if isinstance(a, str)])
    vm = _pairs(vllm_args.render({"VLLM_API_KEY": "k"}))
    # The VM serves under another name, on localhost, with its own port.
    for flag in ("--served-model-name", "--host", "--port"):
        modal.pop(flag, None)
        vm.pop(flag, None)
    assert vm == modal


def test_vllm_command_defaults_and_fp8():
    cmd = vllm_args.render({"VLLM_API_KEY": "k"})
    assert cmd[:3] == ["vllm", "serve", "hweller1/wayfinder-glimmer-30b"]
    assert "--quantization" not in cmd and cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--tool-call-parser") + 1] == cmd[cmd.index("--reasoning-parser") + 1] == "muse_glimmer"
    assert "--chat-template" not in cmd
    fp8 = vllm_args.render({"VLLM_API_KEY": "k", "WAYFINDER_QUANT": "fp8"})
    assert fp8[fp8.index("--quantization") + 1] == "fp8"
    assert "k" not in cmd  # the key never appears on the command line


def test_vllm_args_refuse_bad_config():
    with pytest.raises(SystemExit):
        vllm_args.render({})
    with pytest.raises(SystemExit):
        vllm_args.render({"VLLM_API_KEY": "k", "WAYFINDER_QUANT": "int4"})


def test_missing_env_names_variables_not_values():
    env = {**GOOD_ENV, "VOYAGE_API_KEY": ""}
    del env["SCOUT_PEPPER"]
    assert gateway_vm.missing_env(env) == ["VOYAGE_API_KEY", "SCOUT_PEPPER"]
    from scout.core.contracts.errors import ScoutError

    with pytest.raises(ScoutError) as e:
        gateway_vm.build_app(env)
    assert "VOYAGE_API_KEY" in str(e.value) and "tpuf-test" not in str(e.value)


def test_gateway_app_builds_and_serves_healthz():
    from fastapi.testclient import TestClient

    app = gateway_vm.build_app(GOOD_ENV)
    with TestClient(app) as c:
        r = c.get("/v1/healthz")
        assert r.status_code == 200, r.text
        # Authenticated routes are not exercised here: resolving a key reads the keys namespace in turbopuffer,
        # which is a network call. The live check is `healthcheck.sh remote`.


def test_bind_address_defaults_to_loopback():
    assert gateway_vm.bind_address({}) == ("127.0.0.1", 8080)
    assert gateway_vm.bind_address({"SCOUT_BIND_PORT": "9000"}) == ("127.0.0.1", 9000)


def test_memory_key_cache_expires():
    now = [0.0]
    cache = gateway_vm.MemoryKeyCache(ttl_s=10, now=lambda: now[0])

    async def run():
        await cache.set("k", {"tenant": "t"})
        assert await cache.get("k") == {"tenant": "t"}
        now[0] = 11
        assert await cache.get("k") is None
        await cache.set("k", {"tenant": "t"})
        await cache.delete("k")
        assert await cache.get("k") is None

    asyncio.run(run())


def _env_names(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_env_examples_name_every_required_variable_and_hold_no_secret():
    g = _env_names(GCP / "env/gateway.env.example")
    for name in gateway_vm.REQUIRED:
        assert name in g, name
    for name in ("TURBOPUFFER_API_KEY", "VOYAGE_API_KEY", "SCOUT_MASTER_KEY", "SCOUT_PEPPER", "SCOUT_LLM_API_KEY"):
        assert g[name] == "", f"{name} must be empty in the example"
    v = _env_names(GCP / "env/vllm.env.example")
    assert v["VLLM_API_KEY"] == "" and v["HF_TOKEN"] == ""


def test_vllm_stays_off_the_network_and_only_the_gateway_is_proxied():
    caddy = (GCP / "Caddyfile.template").read_text()
    assert "reverse_proxy 127.0.0.1:8080" in caddy and "8000" not in caddy.replace("127.0.0.1:8000 and is not proxied", "")
    assert '"--host", "127.0.0.1"' in (GCP / "vllm_args.py").read_text()
    create = (GCP / "create_vm.sh").read_text()
    assert "tcp:80,tcp:443" in create and "35.235.240.0/20" in create and "8000" not in create.split("--allow")[1][:40]


def test_units_read_secrets_from_root_owned_env_files():
    for unit, env in (("wayfinder-vllm", "vllm"), ("scout-gateway", "gateway")):
        text = (GCP / f"systemd/{unit}.service").read_text()
        assert f"EnvironmentFile=/etc/scout/{env}.env" in text and "User=scout" in text


@pytest.mark.parametrize("script", ["install.sh", "startup.sh", "create_vm.sh", "fetch_secrets.sh", "healthcheck.sh"])
def test_shell_scripts_parse(script: str):
    assert subprocess.run(["bash", "-n", str(GCP / script)], capture_output=True).returncode == 0


def test_create_vm_prints_without_creating_anything():
    r = subprocess.run(
        ["bash", str(GCP / "create_vm.sh")],
        env={"PROJECT": "demo", "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "gcloud compute instances create" in r.stdout and "a2-ultragpu-1g" in r.stdout
