"""OpenAI-compatible chat adapter for the `ChatModel` port (design section 3)."""

from scout.adapters.openai_chat.client import (
    SAMPLING_PROFILES,
    OpenAIChat,
    retain_reasoning_from_env,
    sampling_profile_from_env,
)

__all__ = ["SAMPLING_PROFILES", "OpenAIChat", "retain_reasoning_from_env", "sampling_profile_from_env"]
