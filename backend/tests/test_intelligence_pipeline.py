import json

import pytest

from app.agent.dynamic_planner import PlanningInputMetadata
from app.agent.intelligence_contracts import (
    EvaluationResult,
    EvaluationStatus,
    ProviderIdentity,
    RequiredInputKind,
)
from app.agent.intelligence_pipeline import (
    IntelligencePipeline,
    PipelineDisposition,
    PipelineFailureCode,
)
from app.core.config import settings
from app.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from app.llm.gemini import GeminiProvider
from app.llm.router import ProviderSelectionError
from app.tools.registry import tool_registry


class StubProvider(BaseLLMProvider):
    def __init__(self, content=None, error=None, provider="gemini"):
        self.content = content
        self.error = error
        self.identity = provider
        self.calls = []

    def is_available(self):
        return True

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return LLMResponse(content=self.content, model="synthetic", provider=self.identity)


class RaisingRouter:
    def get_provider(self, **kwargs):
        raise ProviderSelectionError("gemini", "missing_api_key")


class FixedRouter:
    def __init__(self, provider):
        self.provider = provider
        self.calls = []

    def get_provider(self, **kwargs):
        self.calls.append(kwargs)
        return self.provider


class FixedEvaluator:
    def evaluate(self, request):
        return EvaluationResult(
            execution_valid=False,
            objective_coverage=0.0,
            evaluation_status=EvaluationStatus.INCONCLUSIVE,
            issues=["missing_evidence"],
        )


def _provider_json(steps):
    return json.dumps({
        "response_type": "PLAN",
        "plan": {"plan_id": "untrusted", "goal": "untrusted", "steps": steps},
        "explanation": "untrusted",
    })


def _step(step_id, name, params, dependencies=None):
    return {
        "step_id": step_id,
        "tool_name": name,
        "parameters": params,
        "dependencies": dependencies or [],
    }


def _managed_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOADS_DIR", tmp_path)
    path = tmp_path / "synthetic.csv"
    path.write_bytes(b"unused synthetic dataset")
    return path


@pytest.mark.asyncio
async def test_supported_single_tool_request_returns_ready_without_execution(monkeypatch):
    calls = []

    async def forbidden_execute(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("pipeline must not execute tools")

    monkeypatch.setattr(tool_registry, "execute_tool", forbidden_execute)
    result = await IntelligencePipeline().run("Check system health", privacy_mode="local_only")
    assert result.disposition is PipelineDisposition.READY
    assert result.intent.category.value == "system_health"
    assert result.planning_response.plan.steps[0].tool_name == "system_health"
    assert result.validation_outcome.value == "accepted"
    assert result.evaluation.evaluation_status is EvaluationStatus.PLAN_VALID
    assert result.provider_identity is ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
    assert calls == []


@pytest.mark.asyncio
async def test_supported_multi_step_request_is_evaluated_against_assessed_tool_groups(tmp_path, monkeypatch):
    path = _managed_dataset(tmp_path, monkeypatch)
    result = await IntelligencePipeline().run(
        "Inspect the dataset and then calculate descriptive statistics",
        PlanningInputMetadata(authorized_file_path=str(path)),
        privacy_mode="hybrid",
    )
    assert result.disposition is PipelineDisposition.READY
    assert result.intent.is_multi_step is True
    assert len(result.intent.required_tool_groups) == 2
    steps = result.planning_response.plan.steps
    assert [step.tool_name for step in steps] == ["inspect_dataset", "descriptive_statistics"]
    assert steps[1].dependencies == [steps[0].step_id]
    assert result.provider_identity is ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("objective", "required_kind"),
    [
        ("Inspect this dataset", RequiredInputKind.DATASET),
        ("Analyze this document", RequiredInputKind.DOCUMENT),
    ],
)
async def test_missing_required_dataset_or_document_returns_clarification(objective, required_kind):
    result = await IntelligencePipeline().run(objective, privacy_mode="local_only")
    assert result.disposition is PipelineDisposition.CLARIFICATION_REQUIRED
    assert any(item.kind == required_kind for item in result.intent.missing_inputs)
    assert result.planning_response.plan is None


@pytest.mark.asyncio
async def test_ambiguous_request_clarifies_and_unsupported_request_is_rejected():
    pipeline = IntelligencePipeline()
    ambiguous = await pipeline.run("Analyze this", privacy_mode="local_only")
    unsupported = await pipeline.run("Teleport the database to Mars", privacy_mode="local_only")
    assert ambiguous.disposition is PipelineDisposition.CLARIFICATION_REQUIRED
    assert unsupported.disposition is PipelineDisposition.REJECTED
    assert unsupported.failure_code is PipelineFailureCode.UNSUPPORTED_INTENT


@pytest.mark.asyncio
async def test_provider_unavailable_returns_safe_rejection():
    result = await IntelligencePipeline(router=RaisingRouter()).run(
        "Check system health", privacy_mode="cloud_enabled"
    )
    assert result.disposition is PipelineDisposition.REJECTED
    assert result.failure_code is PipelineFailureCode.PROVIDER_UNAVAILABLE
    assert result.provider_identity is None


@pytest.mark.asyncio
async def test_provider_timeout_is_rejected_without_provider_error_details():
    marker = "PRIVATE_PROVIDER_BODY_AND_KEY"
    provider = StubProvider(error=TimeoutError(marker))
    result = await IntelligencePipeline(provider=provider, provider_identity=ProviderIdentity.GEMINI).run(
        "Check system health", privacy_mode="cloud_enabled"
    )
    assert result.disposition is PipelineDisposition.REJECTED
    assert result.failure_code is PipelineFailureCode.PROVIDER_TIMEOUT
    assert marker not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "failure"),
    [
        ("not-json PRIVATE-RESPONSE", PipelineFailureCode.INVALID_PROVIDER_RESPONSE),
        (_provider_json([_step("step_health", "system_health", {"check_storage": "wrong"})]), PipelineFailureCode.INVALID_PLAN),
        (_provider_json([
            _step("step_a", "system_health", {"check_storage": True}, ["step_b"]),
            _step("step_b", "system_health", {"check_storage": True}, ["step_a"]),
        ]), PipelineFailureCode.INVALID_PLAN),
    ],
)
async def test_malformed_provider_plan_and_invalid_arguments_or_dependencies_are_rejected(content, failure):
    provider = StubProvider(content=content)
    result = await IntelligencePipeline(provider=provider, provider_identity=ProviderIdentity.GEMINI).run(
        "Check system health", privacy_mode="cloud_enabled"
    )
    assert result.disposition is PipelineDisposition.REJECTED
    assert result.failure_code is failure


@pytest.mark.asyncio
async def test_incomplete_evaluation_cannot_become_ready():
    result = await IntelligencePipeline(evaluator=FixedEvaluator()).run(
        "Check system health", privacy_mode="local_only"
    )
    assert result.disposition is PipelineDisposition.REJECTED
    assert result.failure_code is PipelineFailureCode.INCOMPLETE_EVALUATION
    assert result.validation_outcome.value == "accepted"
    assert result.evaluation.evaluation_status is EvaluationStatus.INCONCLUSIVE


@pytest.mark.asyncio
async def test_cloud_enabled_uses_selected_configured_provider_and_identity(monkeypatch):
    content = _provider_json([_step("step_health", "system_health", {"check_storage": True})])
    stub = StubProvider(content=content)
    provider = GeminiProvider(api_key="synthetic-key")
    provider.chat = stub.chat
    router = FixedRouter(provider)
    result = await IntelligencePipeline(router=router).run(
        "Check system health", privacy_mode="cloud_enabled", requested_provider="gemini"
    )
    assert router.calls == [{"requested_provider": "gemini", "privacy_mode": "cloud_enabled"}]
    assert result.disposition is PipelineDisposition.READY
    assert result.provider_identity is ProviderIdentity.GEMINI
    assert result.planning_response.provider_identity is ProviderIdentity.GEMINI


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local_only", "hybrid"])
async def test_local_and_hybrid_do_not_select_or_call_injected_cloud_provider(mode):
    provider = StubProvider(content=_provider_json([_step("step_health", "system_health", {"check_storage": True})]))
    router = RaisingRouter()
    result = await IntelligencePipeline(
        router=router, provider=provider, provider_identity=ProviderIdentity.GEMINI,
    ).run("Check system health", privacy_mode=mode)
    assert result.disposition is PipelineDisposition.READY
    assert result.provider_identity is ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
    assert provider.calls == []


@pytest.mark.asyncio
async def test_sensitive_paths_objectives_and_provider_bodies_do_not_leak_to_failure_or_logs(tmp_path, monkeypatch, caplog):
    path = _managed_dataset(tmp_path, monkeypatch)
    objective_marker = "PRIVATE_OBJECTIVE_MARKER"
    response_marker = "PRIVATE_PROVIDER_RESPONSE_MARKER"
    provider = StubProvider(error=RuntimeError(response_marker))
    result = await IntelligencePipeline(provider=provider, provider_identity=ProviderIdentity.GEMINI).run(
        f"Check system health {objective_marker}",
        PlanningInputMetadata(authorized_file_path=str(path)),
        privacy_mode="cloud_enabled",
    )
    rendered = result.model_dump_json()
    assert result.disposition is PipelineDisposition.REJECTED
    assert str(path) not in rendered
    assert objective_marker not in rendered
    assert response_marker not in rendered
    assert objective_marker not in caplog.text and response_marker not in caplog.text


@pytest.mark.asyncio
async def test_invalid_input_metadata_returns_typed_rejection_without_echoing_values():
    marker = "PRIVATE-SENSITIVE-PATH"
    result = await IntelligencePipeline().run(
        "Check system health",
        {"authorized_file_path": marker, "unexpected": marker},
        privacy_mode="local_only",
    )
    assert result.disposition is PipelineDisposition.REJECTED
    assert result.failure_code is PipelineFailureCode.INVALID_INPUT
    assert marker not in result.model_dump_json()


@pytest.mark.asyncio
async def test_invalid_privacy_mode_is_rejected_instead_of_defaulted():
    result = await IntelligencePipeline().run("Check system health", privacy_mode="")
    assert result.disposition is PipelineDisposition.REJECTED
    assert result.failure_code is PipelineFailureCode.INVALID_INPUT
