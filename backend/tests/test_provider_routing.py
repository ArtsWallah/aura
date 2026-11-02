import httpx
import pytest

from app.agent.intelligence_contracts import ProviderIdentity
from app.core.config import settings
from app.llm.base import LLMMessage, ProviderRequestError
from app.llm.gemini import GeminiProvider
from app.llm.mock_provider import TestReasoningProvider as MockProvider
from app.llm.router import LLMRouter, ProviderSelectionError
from app.llm.base import LLMResponse
from app.agent.planner import ProviderPlanError, TaskPlanner
from app.privacy.outbound import PrivacyAwareAsyncClient, privacy_scope


def test_local_only_selects_explicitly_identifiable_local_mock_without_cloud(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "valid-test-key")
    provider = LLMRouter().get_provider(privacy_mode="local_only")
    assert isinstance(provider, MockProvider)
    assert provider.is_available()


@pytest.mark.asyncio
async def test_local_only_mock_response_keeps_mock_identity(monkeypatch):
    provider = LLMRouter().get_provider(privacy_mode="local_only")
    response = await provider.chat([LLMMessage(role="user", content="synthetic request")])
    assert response.provider == "test_reasoner"
    assert response.provider != ProviderIdentity.GEMINI.value


@pytest.mark.parametrize("mode", ["hybrid", "cloud_enabled"])
def test_configured_cloud_provider_is_selected_without_fallback(monkeypatch, mode):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "valid-test-key")
    monkeypatch.setattr(settings, "DEFAULT_LLM_PROVIDER", "gemini")
    provider = LLMRouter().get_provider(privacy_mode=mode)
    assert isinstance(provider, GeminiProvider)


@pytest.mark.parametrize("mode", ["hybrid", "cloud_enabled"])
def test_unavailable_configured_provider_fails_closed_without_mock_substitution(monkeypatch, mode):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(settings, "DEFAULT_LLM_PROVIDER", "gemini")
    with pytest.raises(ProviderSelectionError) as exc_info:
        LLMRouter().get_provider(privacy_mode=mode)
    assert exc_info.value.provider_id == "gemini"
    assert exc_info.value.code == "missing_api_key"
    assert "test_reasoner" not in str(exc_info.value)


def test_mock_provider_requires_explicit_selection_outside_local_only(monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    with pytest.raises(ProviderSelectionError):
        LLMRouter().get_provider(privacy_mode="hybrid")
    provider = LLMRouter().get_provider(requested_provider="test_reasoner", privacy_mode="hybrid")
    assert isinstance(provider, MockProvider)


def test_test_configuration_does_not_silently_replace_unavailable_provider(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "test")
    monkeypatch.setattr(settings, "DEFAULT_LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    with pytest.raises(ProviderSelectionError):
        LLMRouter().get_provider(privacy_mode="cloud_enabled")


def test_provider_marked_unavailable_fails_without_selecting_mock(monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_LLM_PROVIDER", "gemini")
    router = LLMRouter()
    router._providers["gemini"] = GeminiProvider(api_key=None)
    with pytest.raises(ProviderSelectionError) as exc_info:
        router.get_provider(privacy_mode="cloud_enabled")
    assert exc_info.value.code == "missing_api_key"


@pytest.mark.asyncio
async def test_gemini_direct_missing_key_error_is_typed_and_safe():
    with pytest.raises(ProviderRequestError) as exc_info:
        await GeminiProvider(api_key=None).chat([LLMMessage(role="user", content="synthetic")])
    assert exc_info.value.code == "missing_api_key"
    assert "API key" not in str(exc_info.value)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class _FakeClient:
    response = None
    error = None

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error", "expected_code"),
    [
        (None, httpx.ReadTimeout("sensitive-key-in-url"), "timeout"),
        (_FakeResponse(503, {"error": "private response body"}), None, "http_failure"),
        (_FakeResponse(200, ValueError("private response body")), None, "invalid_response"),
    ],
)
async def test_gemini_failure_is_sanitized(response, error, expected_code, monkeypatch, caplog):
    _FakeClient.response = response
    _FakeClient.error = error
    monkeypatch.setattr("app.llm.gemini.PrivacyAwareAsyncClient", _FakeClient)
    provider = GeminiProvider(api_key="synthetic-test-key")
    with pytest.raises(ProviderRequestError) as exc_info:
        await provider.chat([LLMMessage(role="user", content="synthetic objective")])
    assert exc_info.value.code == expected_code
    assert "sensitive-key-in-url" not in str(exc_info.value)
    assert "private response body" not in caplog.text


@pytest.mark.asyncio
async def test_successful_gemini_response_does_not_retain_raw_provider_body(monkeypatch):
    _FakeClient.response = _FakeResponse(200, {
        "candidates": [{"content": {"parts": [{"text": "safe synthetic response"}]}}],
        "secret": "raw-provider-payload",
    })
    _FakeClient.error = None
    monkeypatch.setattr("app.llm.gemini.PrivacyAwareAsyncClient", _FakeClient)
    response = await GeminiProvider(api_key="synthetic-test-key").chat(
        [LLMMessage(role="user", content="synthetic objective")]
    )
    assert response.provider == "gemini"
    assert response.raw_response == {}
    assert response.provider == ProviderIdentity.GEMINI.value


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local_only", "hybrid"])
async def test_local_and_hybrid_modes_block_outbound_requests(mode):
    with privacy_scope(mode), pytest.raises(PermissionError):
        async with PrivacyAwareAsyncClient() as client:
            await client.get("https://example.invalid")


class _MalformedPlanProvider:
    async def chat(self, *args, **kwargs):
        return LLMResponse(content="private provider response body", model="synthetic", provider="gemini")


@pytest.mark.asyncio
async def test_task_planner_does_not_silently_replace_malformed_provider_plan(caplog, monkeypatch):
    planner = TaskPlanner(_MalformedPlanProvider())
    monkeypatch.setattr(
        planner, "_create_fallback_plan",
        lambda *args, **kwargs: pytest.fail("invalid provider output must not trigger local fallback"),
    )
    with pytest.raises(ProviderPlanError) as exc_info:
        await planner.create_plan("sensitive synthetic objective", privacy_mode="cloud_enabled")
    assert "private provider response body" not in str(exc_info.value)
    assert "sensitive synthetic objective" not in str(exc_info.value)
    assert "private provider response body" not in caplog.text
    assert "sensitive synthetic objective" not in caplog.text
