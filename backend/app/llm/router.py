from typing import Literal, Any
from app.llm.base import BaseLLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.mock_provider import TestReasoningProvider
from app.core.config import settings
from app.core.logging import logger

class LLMRouter:
    def __init__(self):
        self._providers: dict[str, BaseLLMProvider] = {
            "gemini": GeminiProvider(),
            "test_reasoner": TestReasoningProvider()
        }

    def get_provider(
        self,
        requested_provider: str | None = None,
        privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] = "hybrid"
    ) -> BaseLLMProvider:
        if privacy_mode not in {"local_only", "hybrid", "cloud_enabled"}:
            raise ValueError("Unknown privacy mode; refusing cloud routing.")
        # STRICT PRIVACY ENFORCEMENT:
        # If privacy_mode is "local_only", cloud LLMs MUST NOT receive data.
        if privacy_mode == "local_only":
            logger.info("Privacy Mode is 'local_only': Routing exclusively to local/offline deterministic engine.")
            return self._providers["test_reasoner"]

        target = requested_provider or settings.DEFAULT_LLM_PROVIDER
        
        provider = self._providers.get(target)
        if provider and provider.is_available():
            return provider
            
        # Fallback to test_reasoner if requested provider lacks credentials
        logger.warning(f"Provider '{target}' is unavailable (e.g. missing API key). Falling back to TestReasoningProvider.")
        return self._providers["test_reasoner"]

    def list_providers(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "gemini",
                "name": "Google Gemini API (Cloud LLM)",
                "available": self._providers["gemini"].is_available(),
                "model": settings.GEMINI_MODEL,
                "is_cloud": True,
                "requires_api_key": True
            },
            {
                "id": "test_reasoner",
                "name": "Deterministic Test Reasoner (Local / Offline)",
                "available": True,
                "model": "aura-test-reasoner-v1",
                "is_cloud": False,
                "requires_api_key": False
            }
        ]

llm_router = LLMRouter()
