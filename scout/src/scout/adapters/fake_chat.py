"""`ScriptedChat`: deterministic `ChatModel` that replays a fixed script and records every request."""

import copy
from collections.abc import AsyncIterator, Sequence

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.chat import ChatDelta, ChatRequest, ChatTurn


class ScriptExhaustedError(ScoutError):
    """The code under test asked for more turns than the script holds (a test bug, surfaced loudly)."""

    def __init__(self, asked: int, scripted: int) -> None:
        super().__init__(
            ErrorCode.INTERNAL,
            f"ScriptedChat exhausted: call {asked} but only {scripted} turns were scripted",
        )


class ScriptedChat:
    """Each `complete` returns the next scripted `ChatTurn`; a scripted `ScoutError` is raised instead
    (to script RATE_LIMITED / UPSTREAM_* paths). Requests are snapshotted at call time, so a loop that
    mutates its message list afterwards cannot rewrite history."""

    def __init__(self, script: Sequence[ChatTurn | ScoutError]) -> None:
        self._script = list(script)
        self._next = 0
        self.requests: list[ChatRequest] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def remaining(self) -> int:
        return len(self._script) - self._next

    async def complete(self, req: ChatRequest) -> ChatTurn:
        self.requests.append(
            ChatRequest(
                messages=copy.deepcopy(list(req.messages)),
                tools=copy.deepcopy(list(req.tools)),
                tool_choice=req.tool_choice,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                presence_penalty=req.presence_penalty,
                max_tokens=req.max_tokens,
                thinking=req.thinking,
            )
        )
        if self._next >= len(self._script):
            raise ScriptExhaustedError(len(self.requests), len(self._script))
        item = self._script[self._next]
        self._next += 1
        if isinstance(item, ScoutError):
            raise item
        return item

    def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        raise NotImplementedError("ScriptedChat does not stream")
