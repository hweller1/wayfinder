"""ChatModel port (design section 3)."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: Sequence[
        Mapping[str, Any]
    ]  # OpenAI message dicts; an assistant one may carry `reasoning_content`
    tools: Sequence[Mapping[str, Any]] = ()  # OpenAI function specs
    tool_choice: Literal["auto", "required", "none"] = "auto"
    temperature: float = 0.85
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.1
    max_tokens: int = 4096
    thinking: bool = True  # -> chat_template_kwargs.enable_thinking


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class ChatTurn:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    tokens_in: int
    tokens_out: int
    finish: str
    # The model's reasoning for this turn, when the endpoint returns it (`reasoning_content` / `reasoning`);
    # None for non-reasoning models. The explore loop keeps it in the history as the assistant message's
    # `reasoning_content`, exactly where the Glimmer chat template renders it; an adapter decides whether it
    # is re-sent (`OpenAIChat(retain_reasoning=...)`).
    reasoning: str | None = None


@dataclass(frozen=True, slots=True)
class ChatDelta:
    """One streamed increment (design leaves the shape open; streaming is optional)."""

    content: str | None = None
    tool_call: ToolCall | None = None
    finish: str | None = None


class ChatModel(Protocol):
    async def complete(self, req: ChatRequest) -> ChatTurn:
        """OpenAI-compatible tool calling. Returns one assistant turn. Tool-call arguments are parsed
        JSON dicts; a call whose arguments do not parse is returned with `parse_error` set (the loop
        feeds the error back to the model, counted in usage). Errors: UPSTREAM_UNAVAILABLE,
        UPSTREAM_TIMEOUT, RATE_LIMITED."""
        ...

    def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        """OPTIONAL. Adapters without streaming raise NotImplementedError; callers must not rely on it."""
        ...
