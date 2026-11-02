"""Manual clock for tests: time moves only when advanced; `sleep` advances instead of waiting."""

import asyncio


class FakeClock:
    def __init__(self, *, start_epoch: float = 1_700_000_000.0, start_monotonic: float = 1_000.0) -> None:
        self._epoch = start_epoch
        self._mono = start_monotonic
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self._epoch

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("clock cannot go backwards")
        self._epoch += seconds
        self._mono += seconds

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(max(seconds, 0.0))
        await asyncio.sleep(0)  # yield to the loop like a real sleep, without waiting
