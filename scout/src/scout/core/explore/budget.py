"""Turn, clock and context budget for one explore call (FR-C4, AC-1.4). Time is the injected `Clock`."""

from scout.core.contracts.types import Budget as BudgetLimits
from scout.core.ports.clock import Clock


class Budget:
    def __init__(self, clock: Clock, max_turns: int, max_seconds: float) -> None:
        self._clock = clock
        self.max_turns = max_turns
        self.max_seconds = max_seconds
        self.turns = 0
        self._context_exceeded = False
        self._started = clock.monotonic()

    @classmethod
    def from_limits(cls, clock: Clock, limits: BudgetLimits) -> "Budget":
        return cls(clock, limits.max_turns, limits.max_seconds)

    def elapsed(self) -> float:
        return self._clock.monotonic() - self._started

    def tick(self) -> None:
        """Count one model turn."""
        self.turns += 1

    def mark_context_exhausted(self) -> None:
        """The prompt no longer fits the model's context window (local estimate or upstream error)."""
        self._context_exceeded = True

    @property
    def reason(self) -> str | None:
        if self.turns >= self.max_turns:
            return "turns"
        if self.elapsed() >= self.max_seconds:
            return "seconds"
        if self._context_exceeded:
            return "context"
        return None

    @property
    def exhausted(self) -> bool:
        return self.reason is not None
