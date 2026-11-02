"""`OpenAIChat`: `ChatModel` over any OpenAI-compatible `/chat/completions` endpoint (design sections 3, 14).

Built on `httpx` (no `openai` SDK). Target: vLLM's OpenAI server serving Agents-A1-35B with
`--enable-auto-tool-choice --tool-call-parser qwen3_coder` (tool calls arrive as structured `tool_calls`),
but any compatible endpoint works (Baseten, Meta API, ...). Sampling comes from `ChatRequest`
(A1 defaults: temperature 0.85, top_p 0.95, top_k 20, presence_penalty 1.1). `top_k` and
`chat_template_kwargs` are vLLM extensions; pass `vllm_extensions=False` for a strict endpoint that
rejects unknown fields.

Reasoning: the response's `reasoning_content` (or `reasoning`) is returned as `ChatTurn.reasoning`, and by
default (`retain_reasoning=True`) assistant messages carrying `reasoning_content` are sent back as they are,
because the model was trained to see its earlier reasoning (the chat template renders it on every assistant
turn). `retain_reasoning=False` (env `SCOUT_LLM_RETAIN_REASONING=0`) neither keeps nor sends it, for an
endpoint that rejects it or a model that was not trained that way. Non-reasoning models never produce the
field, so nothing changes for them.

Failures map to typed `ScoutError`s with fixed messages (the vendor body is never echoed): 429 ->
RATE_LIMITED (`details.retry_after` when the header is a number), timeouts and 504 -> UPSTREAM_TIMEOUT,
connection errors, 5xx and malformed bodies -> UPSTREAM_UNAVAILABLE, a context-length 400 ->
INVALID_ARGUMENT with `details.reason == "context_length"`, other 4xx -> INVALID_ARGUMENT, 401/403 -> INTERNAL
(a credentials problem is operator-side, never the caller's scout key).
"""

import contextlib
import dataclasses
import json
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.chat import ChatDelta, ChatRequest, ChatTurn, ToolCall

__all__ = ["SAMPLING_PROFILES", "OpenAIChat", "retain_reasoning_from_env", "sampling_profile_from_env"]

DEFAULT_MODEL = "scout-llm"
DEFAULT_TIMEOUT_S = 300.0
WARM_TIMEOUT_S = 10.0


def warm_url_for(url: str) -> str | None:
    """The Modal `/warm` endpoint of a Modal-served model URL: `https://<ws>--<label>[.<region>].modal.run/v1`
    -> `https://<ws>--<label>-warm[.<region>].modal.run/` (wayfinder `serve/wayfinder_v3.py`). None for any
    other host."""
    m = re.match(r"^(https://[^/.]+)((?:\.[a-z0-9-]+)?\.modal\.run)(?:/.*)?$", url)
    return f"{m.group(1)}-warm{m.group(2)}/" if m else None


_FALSE = frozenset({"0", "false", "no", "off"})


SAMPLING_PROFILES = ("a1", "sft")


def sampling_profile_from_env(env: Mapping[str, str]) -> str:
    """`SCOUT_LLM_SAMPLING`: `a1` (default: top_k, presence_penalty and `enable_thinking` are sent, as A1
    wants) or `sft` (wayfinder-glimmer, served as wayfinder-v3, and other Muse-Glimmer post-trains: only
    temperature and top_p, the protocol the model was evaluated under). An unknown value is an error, never a
    silent default."""
    value = env.get("SCOUT_LLM_SAMPLING", "a1").strip().lower() or "a1"
    if value not in SAMPLING_PROFILES:
        raise ScoutError(
            ErrorCode.INTERNAL, f"SCOUT_LLM_SAMPLING must be one of {', '.join(SAMPLING_PROFILES)}"
        )
    return value


def retain_reasoning_from_env(env: Mapping[str, str]) -> bool:
    """`SCOUT_LLM_RETAIN_REASONING`: on unless set to 0/false/no/off."""
    return env.get("SCOUT_LLM_RETAIN_REASONING", "1").strip().lower() not in _FALSE


class OpenAIChat:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        extra_headers: Mapping[str, str] | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        vllm_extensions: bool = True,
        retain_reasoning: bool = True,
        sampling: str = "a1",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """`base_url` includes the version prefix (`https://host/v1`). `api_key` None sends no
        Authorization header. `extra_headers` is added to every request (e.g. Modal proxy auth)."""
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = timeout
        self._sft = sampling == "sft"
        self._vllm = vllm_extensions and not self._sft
        self.retain_reasoning = retain_reasoning
        self._transport = transport
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._headers.update(extra_headers or {})
        self.warm_url: str | None = warm_url_for(base_url)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "OpenAIChat":
        """`SCOUT_LLM_URL` (required, with `/v1`), `SCOUT_LLM_API_KEY`, `SCOUT_LLM_MODEL` (default
        `scout-llm`), `SCOUT_LLM_SAMPLING` (`a1` or `sft`, see `sampling_profile_from_env`), and Modal
        proxy auth `MODAL_PROXY_TOKEN_ID` / `MODAL_PROXY_TOKEN_SECRET` if both set."""
        url = env.get("SCOUT_LLM_URL")
        if not url:
            raise ScoutError(ErrorCode.INTERNAL, "SCOUT_LLM_URL is not set")
        headers: dict[str, str] = {}
        if env.get("MODAL_PROXY_TOKEN_ID") and env.get("MODAL_PROXY_TOKEN_SECRET"):
            headers = {
                "Modal-Key": env["MODAL_PROXY_TOKEN_ID"],
                "Modal-Secret": env["MODAL_PROXY_TOKEN_SECRET"],
            }
        return cls(
            url,
            env.get("SCOUT_LLM_API_KEY"),
            env.get("SCOUT_LLM_MODEL", DEFAULT_MODEL),
            headers,
            retain_reasoning=retain_reasoning_from_env(env),
            sampling=sampling_profile_from_env(env),
        )._with_warm(env.get("SCOUT_LLM_WARM_URL"))

    def _with_warm(self, url: str | None) -> "OpenAIChat":
        if url:
            self.warm_url = url
        return self

    async def warm(self) -> bool:
        """Start the model's cold start without waiting for it: one GET to its `/warm` endpoint (same bearer
        key), which answers in about a second. False when the upstream has no warm endpoint or it failed."""
        if not self.warm_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=WARM_TIMEOUT_S, transport=self._transport) as client:
                res = await client.get(self.warm_url, headers=self._headers)
        except httpx.HTTPError:
            return False
        return res.status_code == 200

    def _messages(self, req: ChatRequest) -> list[dict[str, Any]]:
        if self.retain_reasoning:
            return [dict(m) for m in req.messages]
        return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in req.messages]

    def _body(self, req: ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages(req),
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "stream": False,
        }
        if req.presence_penalty and not self._sft:
            body["presence_penalty"] = req.presence_penalty
        if req.tools:
            body["tools"] = [dict(t) for t in req.tools]
            body["tool_choice"] = req.tool_choice
        if self._vllm:
            if req.top_k > 0:
                body["top_k"] = req.top_k
            body["chat_template_kwargs"] = {"enable_thinking": req.thinking}
        return body

    async def complete(self, req: ChatRequest) -> ChatTurn:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                res = await client.post(self._url, headers=self._headers, json=self._body(req))
        except httpx.TimeoutException:
            raise ScoutError(ErrorCode.UPSTREAM_TIMEOUT, "the chat model timed out") from None
        except httpx.HTTPError:
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the chat model is unreachable") from None
        if res.status_code != 200:
            raise _status_error(res)
        turn = _parse_turn(res)
        # Off: the loop never holds the reasoning, so the context estimate matches what is sent.
        return turn if self.retain_reasoning else dataclasses.replace(turn, reasoning=None)

    def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        raise NotImplementedError("OpenAIChat does not stream")


def _status_error(res: httpx.Response) -> ScoutError:
    code = res.status_code
    if code == 429:
        details: dict[str, Any] = {}
        with contextlib.suppress(KeyError, ValueError):
            details["retry_after"] = float(res.headers["retry-after"])
        return ScoutError(ErrorCode.RATE_LIMITED, "the chat model is rate limited", details=details)
    if code in (408, 504):
        return ScoutError(ErrorCode.UPSTREAM_TIMEOUT, "the chat model timed out")
    if code >= 500:
        return ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the chat model is unavailable")
    if code in (401, 403):
        return ScoutError(ErrorCode.INTERNAL, "the chat endpoint rejected the configured credentials")
    if code == 400 and _is_context_overflow(res.text):
        return ScoutError(
            ErrorCode.INVALID_ARGUMENT,
            "the prompt exceeds the model's maximum context length",
            details={"reason": "context_length"},
        )
    if code == 404:
        return ScoutError(ErrorCode.INTERNAL, "the chat endpoint or model was not found")
    return ScoutError(ErrorCode.INVALID_ARGUMENT, "the chat request was rejected by the model")


def _is_context_overflow(text: str) -> bool:
    low = text.lower()
    return "maximum context length" in low or "context_length_exceeded" in low or "context length" in low


def _parse_turn(res: httpx.Response) -> ChatTurn:
    try:
        data = res.json()
        choice = data["choices"][0]
        msg = choice["message"]
        calls = tuple(_parse_call(i, c) for i, c in enumerate(msg.get("tool_calls") or ()))
        usage = data.get("usage") or {}
        content = msg.get("content")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        return ChatTurn(
            content=content if isinstance(content, str) else None,
            tool_calls=calls,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            finish=str(choice.get("finish_reason") or ("tool_calls" if calls else "stop")),
            reasoning=reasoning if isinstance(reasoning, str) and reasoning else None,
        )
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise ScoutError(
            ErrorCode.UPSTREAM_UNAVAILABLE, "the chat model returned a malformed response"
        ) from None


def _parse_call(index: int, raw: Mapping[str, Any]) -> ToolCall:
    fn = raw["function"]
    call_id = str(raw.get("id") or f"call_{index}")
    name = str(fn["name"])
    args = fn.get("arguments")
    if isinstance(args, dict):  # some servers return the object, not a JSON string
        return ToolCall(id=call_id, name=name, arguments=dict(args))  # pyright: ignore[reportUnknownArgumentType]
    if args is None or (isinstance(args, str) and not args.strip()):
        return ToolCall(id=call_id, name=name, arguments={})
    try:
        parsed = json.loads(args)
    except (TypeError, ValueError) as e:
        return ToolCall(id=call_id, name=name, arguments={}, parse_error=f"{type(e).__name__}: {e}"[:200])
    if not isinstance(parsed, dict):
        return ToolCall(id=call_id, name=name, arguments={}, parse_error="arguments must be a JSON object")
    return ToolCall(id=call_id, name=name, arguments=dict(parsed))  # pyright: ignore[reportUnknownArgumentType]
