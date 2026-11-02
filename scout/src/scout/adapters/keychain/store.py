"""`KeychainStore`: the `KeyStore` port over the OS keychain (`keyring`) with a 0600 file fallback (AC-5.4).

Values are stored base64-encoded (keyring stores text). When the keyring backend is missing, locked or
fails in any way, the value goes to `~/.config/scout/credentials` instead (directory 0700, file 0600, written
atomically). `get` looks in the keychain first, then the file, so a value written to the fallback is still
found when the keychain later works again. `SCOUT_KEYSTORE=file` skips the keychain (headless hosts,
tests). Values are never logged and never appear in an exception.
"""

import base64
import contextlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["KeychainStore", "default_credentials_path"]

SERVICE = "scout"


def default_credentials_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".config" / "scout" / "credentials"


class KeychainStore:
    def __init__(
        self,
        service: str = SERVICE,
        *,
        backend: Any = None,
        fallback_path: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """`backend` is a `keyring`-shaped object (`get_password`/`set_password`/`delete_password`); None
        imports `keyring` lazily on first use."""
        self._service = service
        self._backend = backend
        environ = os.environ if env is None else env
        home = Path(environ["HOME"]) if environ.get("HOME") else None
        self._path = fallback_path or default_credentials_path(home)
        self._file_only = environ.get("SCOUT_KEYSTORE") == "file"

    def _keyring(self) -> Any:
        if self._file_only:
            return None
        if self._backend is None:
            try:
                import keyring

                self._backend = keyring
            except Exception:  # noqa: BLE001
                return None
        return self._backend

    # --- fallback file -------------------------------------------------------------------------
    def _read_file(self) -> dict[str, str]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}

    def _write_file(self, items: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".credentials-")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(items, f)
            os.replace(tmp, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # --- KeyStore ------------------------------------------------------------------------------
    def get(self, name: str) -> bytes | None:
        kr = self._keyring()
        if kr is not None:
            try:
                value = kr.get_password(self._service, name)
            except Exception:  # noqa: BLE001
                value = None
            if value:
                return base64.b64decode(value)
        stored = self._read_file().get(name)
        return base64.b64decode(stored) if stored else None

    def set(self, name: str, value: bytes) -> None:
        encoded = base64.b64encode(value).decode("ascii")
        kr = self._keyring()
        if kr is not None:
            try:
                kr.set_password(self._service, name, encoded)
                stored = True
            except Exception:  # noqa: BLE001
                stored = False
            if stored:
                items = self._read_file()
                if name in items:  # a stale fallback copy must not shadow the keychain later
                    del items[name]
                    self._write_file(items)
                return
        items = self._read_file()
        items[name] = encoded
        self._write_file(items)

    def delete(self, name: str) -> None:
        kr = self._keyring()
        if kr is not None:
            with contextlib.suppress(Exception):
                kr.delete_password(self._service, name)
        items = self._read_file()
        if name in items:
            del items[name]
            self._write_file(items)
