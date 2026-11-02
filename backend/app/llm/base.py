from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel, Field

class MessageRole(str):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None

class LLMToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str
    provider: str

class BaseLLMProvider(ABC):
    """Abstract interface for LLM Reasoning Engines."""
    
    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format_json: bool = False
    ) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
