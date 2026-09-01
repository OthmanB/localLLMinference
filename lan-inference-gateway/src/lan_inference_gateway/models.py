"""OpenAI-compatible request models accepted by the gateway."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[dict[str, Any]]
    stream: bool = False
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
