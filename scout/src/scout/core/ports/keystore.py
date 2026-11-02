"""KeyStore port (design section 3). CLIENT side; sync."""

from typing import Protocol


class KeyStore(Protocol):
    def get(self, name: str) -> bytes | None:
        """Names: "scout_key", "k_team:<tenant>"."""
        ...

    def set(self, name: str, value: bytes) -> None:
        """Keychain; fallback file 0600 (AC-5.4)."""
        ...

    def delete(self, name: str) -> None: ...
