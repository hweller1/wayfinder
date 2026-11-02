"""In-memory `KeyStore` for tests. Never touches the OS keychain or the disk."""


class FakeKeyStore:
    def __init__(self) -> None:
        self._items: dict[str, bytes] = {}

    def get(self, name: str) -> bytes | None:
        return self._items.get(name)

    def set(self, name: str, value: bytes) -> None:
        self._items[name] = bytes(value)

    def delete(self, name: str) -> None:
        self._items.pop(name, None)
