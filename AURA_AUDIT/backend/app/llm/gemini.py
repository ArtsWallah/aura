import json
import httpx
from typing import Any
from app.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, LLMToolCall
from app.core.config import settings
from app.core.logging import logger

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 5)

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format_json: bool = False
    ) -> LLMResponse:
        if not self.is_available():
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY in environment or settings.")

        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        # Format contents for Gemini API
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            elif msg.role in ["user", "assistant"]:
                role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == "tool":
                contents.append({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": msg.name or "tool_result",
                            "response": {"result": msg.content}
                        }
                    }]
                })

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        
        if response_format_json:
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config
        }
        
        if system_instruction:
            payload["systemInstruction"] = system_instruction
            
        if tools:
            # Map tools to Gemini functionDeclarations format
            function_declarations = []
            for t in tools:
                function_declarations.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}})
                })
            payload["tools"] = [{"functionDeclarations": function_declarations}]

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            try:
                response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                if response.status_code != 200:
                    logger.error(f"Gemini API returned error code {response.status_code}: {response.text}")
                    raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text}")

                data = response.json()
                candidate = data.get("candidates", [{}])[0]
                content_obj = candidate.get("content", {})
                parts = content_obj.get("parts", [])
                
                text_content = ""
                tool_calls = []
                
                for idx, part in enumerate(parts):
                    if "text" in part:
                        text_content += part["text"]
                    elif "functionCall" in part:
                        fn = part["functionCall"]
                        tool_calls.append(LLMToolCall(
                            id=f"call_{idx}",
                            name=fn.get("name", ""),
                            arguments=fn.get("args", {})
                        ))

                usage_metadata = data.get("usageMetadata", {})
                
                return LLMResponse(
                    content=text_content or None,
                    tool_calls=tool_calls,
                    raw_response=data,
                    prompt_tokens=usage_metadata.get("promptTokenCount", 0),
                    completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
                    model=self.model,
                    provider="gemini"
                )
            except httpx.TimeoutException:
                raise TimeoutError(f"Gemini API request timed out after {settings.LLM_TIMEOUT_SECONDS}s.")
            except Exception as e:
                logger.error(f"Gemini execution exception: {str(e)}")
                raise

