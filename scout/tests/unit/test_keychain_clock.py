"""Task 7.4: keychain store with file fallback, system clock. FR-P5, AC-5.4. Tmp HOME; the real keychain is
never touched (a stub backend stands in for `keyring`)."""

import asyncio
import base64
import json
import stat
import time
from pathlib import Path
from typing import Any

import pytest

from scout.adapters.clock_system import SystemClock
from scout.adapters.keychain import KeychainStore


class StubKeyring:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self.items.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.items[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        self.items.pop((service, name), None)


class BrokenKeyring:
    def get_password(self, *a: Any) -> str | None:
        raise RuntimeError("no backend")

    set_password = delete_password = get_password


def _store(tmp_path: Path, backend: Any, env: dict[str, str] | None = None) -> KeychainStore:
    return KeychainStore(backend=backend, fallback_path=tmp_path / "cfg" / "credentials", env=env or {})


def test_round_trip_in_keychain_and_delete(tmp_path: Path) -> None:
    kr = StubKeyring()
    s = _store(tmp_path, kr)
    s.set("scout_key", b"sk_scout_abc")
    s.set("k_team:dev", bytes(range(64)))
    assert s.get("scout_key") == b"sk_scout_abc"
    assert s.get("k_team:dev") == bytes(range(64))
    assert not (tmp_path / "cfg" / "credentials").exists()
    s.delete("scout_key")
    assert s.get("scout_key") is None
    assert s.get("missing") is None


def test_backend_failure_falls_back_to_file_with_0600(tmp_path: Path) -> None:
    """AC-5.4."""
    s = _store(tmp_path, BrokenKeyring())
    s.set("scout_key", b"sk_scout_abc")
    path = tmp_path / "cfg" / "credentials"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert json.loads(path.read_text())["scout_key"] == base64.b64encode(b"sk_scout_abc").decode()
    assert s.get("scout_key") == b"sk_scout_abc"
    s.set("k_team:dev", b"\x00\x01")
    assert s.get("k_team:dev") == b"\x00\x01"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    s.delete("scout_key")
    assert s.get("scout_key") is None
    assert s.get("k_team:dev") == b"\x00\x01"


def test_file_mode_env_skips_the_keychain(tmp_path: Path) -> None:
    kr = StubKeyring()
    s = _store(tmp_path, kr, {"SCOUT_KEYSTORE": "file"})
    s.set("scout_key", b"v")
    assert kr.items == {}
    assert s.get("scout_key") == b"v"


def test_keychain_write_clears_a_stale_fallback_copy(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "credentials"
    _store(tmp_path, BrokenKeyring()).set("scout_key", b"old")
    s = _store(tmp_path, StubKeyring())
    assert s.get("scout_key") == b"old"  # found in the fallback while the keychain is empty
    s.set("scout_key", b"new")
    assert s.get("scout_key") == b"new"
    assert "scout_key" not in json.loads(path.read_text())


def test_corrupt_fallback_file_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "credentials"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert _store(tmp_path, BrokenKeyring()).get("scout_key") is None


async def test_system_clock() -> None:
    c = SystemClock()
    assert abs(c.now() - time.time()) < 5
    a = c.monotonic()
    await c.sleep(0.01)
    assert c.monotonic() > a
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(c.sleep(5), 0.01)
