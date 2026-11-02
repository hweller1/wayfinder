"""Clock port (design section 3)."""

from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Epoch seconds."""
        ...

    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...
