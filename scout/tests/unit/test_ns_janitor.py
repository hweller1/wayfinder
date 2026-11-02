"""Namespace janitor: scout_dev_/scout_eval_ only, dry-run default, verify-empty. Covers AC-14.6 / NFR-13."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[2] / "ops" / "ns_janitor.py"
_spec = importlib.util.spec_from_file_location("ns_janitor", OPS)
assert _spec
assert _spec.loader
nj = importlib.util.module_from_spec(_spec)
sys.modules["ns_janitor"] = nj
_spec.loader.exec_module(nj)

SEED = [
    "scout_dev_a",
    "scout_dev_b",
    "scout_dev_keys",
    "scout_eval_x",
    "scout_prod_keys",
    "scout_devops",
    "production",
    "other_scout_dev_z",
]


class SpyClient(nj.FakeClient):
    """Records every list prefix and every delete so tests can assert nothing out of scope is touched."""

    def __init__(self, namespaces: list[str]) -> None:
        super().__init__(namespaces)
        self.list_calls: list[str] = []

    def list_namespaces(self, prefix: str) -> list[str]:
        self.list_calls.append(prefix)
        return super().list_namespaces(prefix)


class IgnoresPrefixClient(SpyClient):
    """Hostile/buggy client: returns every namespace regardless of the prefix asked for."""

    def list_namespaces(self, prefix: str) -> list[str]:
        self.list_calls.append(prefix)
        return sorted(self.namespaces)


def go(client: Any, prefix: str, **kw: Any) -> tuple[int, list[str]]:
    lines: list[str] = []
    return nj.run(client, prefix, out=lines.append, **kw), lines


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "scout_",
        "scout_dev",
        "scout_eval",
        "scout",
        "prod_",
        "scout_prod_",
        "SCOUT_DEV_",
        " scout_dev_",
        "scout_dev_ x",
        "scout_dev_;rm",
        "xscout_dev_",
        "scout_devx_",
        "*",
        "scout_dev_\\n",
    ],
)
def test_bad_prefix_refused_before_any_client_call(prefix: str) -> None:
    client = SpyClient(SEED)
    code, lines = go(client, prefix, delete=True, verify_empty=True)
    assert code == 2
    assert "REFUSED" in lines[0]
    assert client.list_calls == []
    assert client.deleted == []
    assert client.namespaces == set(SEED)


@pytest.mark.parametrize("prefix", ["", "scout_", "scout_dev", "scout_prod_"])
def test_cli_refuses_without_client_or_key(prefix: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TURBOPUFFER_API_KEY", raising=False)
    assert nj.main(["--prefix", prefix, "--delete", "--backend", "tpuf"]) == 2


def test_dry_run_is_default_and_deletes_nothing() -> None:
    client = SpyClient(SEED)
    code, lines = go(client, "scout_dev_")
    assert code == 0
    assert client.deleted == []
    assert client.namespaces == set(SEED)
    assert "would delete scout_dev_a" in lines
    assert any("dry-run" in x for x in lines)


def test_delete_removes_only_prefix_matches() -> None:
    client = SpyClient(SEED)
    code, _ = go(client, "scout_dev_", delete=True)
    assert code == 0
    assert sorted(client.deleted) == ["scout_dev_a", "scout_dev_b", "scout_dev_keys"]
    assert {"scout_eval_x", "scout_prod_keys", "scout_devops", "production", "other_scout_dev_z"} <= (
        client.namespaces
    )


def test_eval_prefix_allowed() -> None:
    client = SpyClient(SEED)
    assert go(client, "scout_eval_", delete=True)[0] == 0
    assert client.deleted == ["scout_eval_x"]


def test_narrower_prefix_allowed() -> None:
    client = SpyClient(SEED)
    go(client, "scout_dev_a", delete=True)
    assert client.deleted == ["scout_dev_a"]


def test_keep_list_honored() -> None:
    client = SpyClient(SEED)
    code, lines = go(client, "scout_dev_", keep=["scout_dev_keys"], delete=True, verify_empty=True)
    assert code == 0
    assert "scout_dev_keys" in client.namespaces
    assert "scout_dev_keys" not in client.deleted
    assert "keep scout_dev_keys" in lines


def test_client_ignoring_prefix_filter_cannot_cause_out_of_scope_delete() -> None:
    client = IgnoresPrefixClient(SEED)
    code, lines = go(client, "scout_dev_", delete=True, verify_empty=True)
    assert code == 0
    assert sorted(client.deleted) == ["scout_dev_a", "scout_dev_b", "scout_dev_keys"]
    assert "scout_prod_keys" in client.namespaces
    assert "production" in client.namespaces
    assert any(x.startswith("SKIP out-of-scope") for x in lines)


def test_verify_empty_exits_1_with_leftovers() -> None:
    client = SpyClient(SEED)
    code, lines = go(client, "scout_dev_", verify_empty=True)  # dry-run: nothing deleted
    assert code == 1
    leftover = next(x for x in lines if x.startswith("LEFTOVER"))
    assert "scout_dev_a" in leftover
    assert "scout_dev_keys" in leftover
    assert "scout_eval_x" not in leftover


def test_verify_empty_exits_0_when_clean() -> None:
    client = SpyClient(["scout_eval_x", "production"])
    code, lines = go(client, "scout_dev_", verify_empty=True)
    assert code == 0
    assert any("verified empty" in x for x in lines)


def test_verify_empty_after_delete_detects_undeletable() -> None:
    class Sticky(SpyClient):
        def delete_namespace(self, name: str) -> None:
            if name == "scout_dev_b":
                return  # silently fails to delete
            super().delete_namespace(name)

    code, lines = go(Sticky(SEED), "scout_dev_", delete=True, verify_empty=True)
    assert code == 1
    assert any(x.startswith("LEFTOVER") and "scout_dev_b" in x for x in lines)


def test_delete_error_continues_and_exits_1() -> None:
    class Boom(SpyClient):
        def delete_namespace(self, name: str) -> None:
            if name == "scout_dev_a":
                raise RuntimeError("secret-looking detail")
            super().delete_namespace(name)

    client = Boom(SEED)
    code, lines = go(client, "scout_dev_", delete=True)
    assert code == 1
    assert "scout_dev_b" in client.deleted
    assert "scout_dev_a" in client.namespaces
    assert not any("secret-looking" in x for x in lines)  # exception text is never echoed


def test_main_with_injected_fake_client_and_flags() -> None:
    client = SpyClient(SEED)
    assert nj.main(["--prefix", "scout_dev_", "--verify-empty"], client=client) == 1
    assert nj.main(["--prefix", "scout_dev_", "--delete", "--verify-empty"], client=client) == 0
    assert nj.main(["--prefix", "scout_dev_", "--verify-empty"], client=client) == 0


def test_main_fake_backend_seeded() -> None:
    argv = ["--prefix", "scout_eval_", "--backend", "fake", "--fake-ns", "scout_eval_1", "--verify-empty"]
    assert nj.main(argv) == 1


def test_real_client_requires_key_and_never_called_here(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TURBOPUFFER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TURBOPUFFER_API_KEY"):
        nj.TurbopufferClient()
