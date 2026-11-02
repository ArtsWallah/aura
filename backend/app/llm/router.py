from typing import Literal, Any
from app.llm.base import BaseLLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.mock_provider import TestReasoningProvider
from app.core.config import settings


class ProviderSelectionError(RuntimeError):
    """Safe failure raised when the explicitly configured provider is unavailable."""

    def __init__(self, provider_id: str, code: str):
        self.provider_id = provider_id
        self.code = code
        super().__init__("The configured provider is unavailable; no replacement was selected.")


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
        # Local-only mode selects the offline test implementation. Its output
        # identifies itself as test_reasoner rather than as Gemini.
        if privacy_mode == "local_only":
            # This is an explicit local policy route. The response remains
            # labeled test_mock_provider by TestReasoningProvider.
            return self._providers["test_reasoner"]

        target = requested_provider or settings.DEFAULT_LLM_PROVIDER
        
        provider = self._providers.get(target)
        if provider is None:
            raise ProviderSelectionError(str(target), "unknown_provider")
        if not provider.is_available():
            code = "missing_api_key" if target == "gemini" else "provider_unavailable"
            raise ProviderSelectionError(str(target), code)
        return provider

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
