import json
from typing import Any

import pytest

from app.agent.dynamic_planner import ColumnKind, DynamicPlanner, PlanningInputMetadata
from app.agent.intent import assess_intent
from app.agent.intelligence_contracts import (
    InputRequirement,
    IntentAssessment,
    IntentCategory,
    PlanningResponseType,
    ProviderIdentity,
    RequiredInputKind,
)
from app.core.config import settings
from app.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from app.tools.registry import tool_registry


def _intent(category: IntentCategory, missing=(), required=(), multi: bool = False) -> IntentAssessment:
    return IntentAssessment(
        category=category,
        confidence=0.9,
        required_inputs=[InputRequirement(kind=item) for item in required],
        missing_inputs=[InputRequirement(kind=item) for item in missing],
        is_multi_step=multi,
    )


def _managed_file(tmp_path, monkeypatch, name="dataset.csv"):
    monkeypatch.setattr(settings, "UPLOADS_DIR", tmp_path)
    path = tmp_path / name
    path.write_bytes(b"synthetic only")
    return path


class StubProvider(BaseLLMProvider):
    def __init__(self, content: str | None = None, provider: str = "gemini", error: Exception | None = None, available: bool = True):
        self.content = content
        self.response_provider = provider
        self.error = error
        self.available = available
        self.calls: list[list[LLMMessage]] = []

    def is_available(self) -> bool:
        return self.available

    async def chat(self, messages, tools=None, temperature=0.2, max_tokens=2048, response_format_json=False):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return LLMResponse(
            content=self.content,
            provider=self.response_provider,
            model="stub",
        )


def _provider_json(steps: list[dict[str, Any]]) -> str:
    return json.dumps({
        "response_type": "PLAN",
        "plan": {"plan_id": "untrusted", "goal": "untrusted", "steps": steps},
        "explanation": "untrusted explanation",
    })


def _step(step_id: str, tool_name: str, parameters: dict[str, Any], dependencies=None):
    return {
        "step_id": step_id,
        "tool_name": tool_name,
        "parameters": parameters,
        "dependencies": dependencies or [],
    }


@pytest.mark.asyncio
async def test_simple_system_health_request_returns_valid_registered_single_step():
    planner = DynamicPlanner()
    result = await planner.create_plan("Check system health", _intent(IntentCategory.SYSTEM_HEALTH))
    assert result.response_type is PlanningResponseType.PLAN
    assert result.plan.steps[0].tool_name == "system_health"
    assert result.plan.steps[0].parameters == {"check_storage": True}
    assert planner.last_provider_identity is ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER


@pytest.mark.asyncio
async def test_dataset_inspection_plan_uses_only_server_authorized_managed_path(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    intent = _intent(IntentCategory.DATA_ANALYSIS, missing=[RequiredInputKind.DATASET], required=[RequiredInputKind.DATASET])
    planner = DynamicPlanner()
    result = await planner.create_plan(
        "Inspect the uploaded dataset", intent,
        PlanningInputMetadata(authorized_file_path=str(path)),
    )
    assert result.response_type is PlanningResponseType.PLAN
    assert result.plan.steps[0].tool_name == "inspect_dataset"
    assert result.plan.steps[0].parameters["file_path"] == str(path.resolve())


@pytest.mark.asyncio
async def test_missing_dataset_returns_clarification_without_plan():
    result = await DynamicPlanner().create_plan(
        "Inspect the dataset",
        _intent(IntentCategory.DATA_ANALYSIS, missing=[RequiredInputKind.DATASET], required=[RequiredInputKind.DATASET]),
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION
    assert result.plan is None
    assert "dataset" in result.clarification_question.lower()


@pytest.mark.asyncio
async def test_descriptive_statistics_uses_registered_schema_without_inventing_columns(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    result = await DynamicPlanner().create_plan(
        "Calculate descriptive statistics for the uploaded dataset",
        _intent(IntentCategory.STATISTICAL_ANALYSIS, required=[RequiredInputKind.DATASET]),
        PlanningInputMetadata(authorized_file_path=str(path)),
    )
    assert result.response_type is PlanningResponseType.PLAN
    assert [step.tool_name for step in result.plan.steps] == ["descriptive_statistics"]
    assert "columns" not in result.plan.steps[0].parameters


@pytest.mark.asyncio
async def test_phase_312_assessment_is_accepted_without_changing_its_contract(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    assessment = assess_intent(
        "Inspect the uploaded dataset and show its columns",
        available_inputs=[RequiredInputKind.DATASET],
    )
    result = await DynamicPlanner().create_plan(
        "Inspect the uploaded dataset and show its columns",
        assessment,
        PlanningInputMetadata(authorized_file_path=str(path)),
    )
    assert result.response_type is PlanningResponseType.PLAN
    assert result.plan.steps[0].tool_name == "inspect_dataset"


@pytest.mark.asyncio
async def test_group_comparison_clarifies_when_required_column_metadata_is_missing(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    result = await DynamicPlanner().create_plan(
        "Compare average sales across regions",
        _intent(IntentCategory.STATISTICAL_ANALYSIS, required=[RequiredInputKind.DATASET]),
        PlanningInputMetadata(authorized_file_path=str(path)),
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION
    assert "column" in result.explanation.lower()


@pytest.mark.asyncio
async def test_group_comparison_uses_exact_supplied_column_names_and_types(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    result = await DynamicPlanner().create_plan(
        "Compare average sales across regions",
        _intent(IntentCategory.STATISTICAL_ANALYSIS, required=[RequiredInputKind.DATASET]),
        PlanningInputMetadata(
            authorized_file_path=str(path),
            column_types={"Region": ColumnKind.CATEGORICAL, "Sales": ColumnKind.NUMERIC},
        ),
    )
    assert result.response_type is PlanningResponseType.PLAN
    params = result.plan.steps[0].parameters
    assert params["group_by"] == "Region"
    assert params["metric_column"] == "Sales"
    assert params["aggregations"] == ["mean"]


@pytest.mark.asyncio
async def test_visualization_uses_requested_chart_type_and_registered_columns(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    result = await DynamicPlanner().create_plan(
        "Create a bar chart of sales by region",
        _intent(IntentCategory.VISUALIZATION, required=[RequiredInputKind.DATASET]),
        PlanningInputMetadata(
            authorized_file_path=str(path),
            column_types={"Region": ColumnKind.CATEGORICAL, "Sales": ColumnKind.NUMERIC},
        ),
    )
    assert result.response_type is PlanningResponseType.PLAN
    assert result.plan.steps[0].parameters["chart_type"] == "bar"
    assert result.plan.steps[0].parameters["x_column"] == "Region"
    assert result.plan.steps[0].parameters["y_column"] == "Sales"


@pytest.mark.asyncio
async def test_outlier_plan_requires_and_uses_a_named_numeric_column(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    intent = _intent(IntentCategory.STATISTICAL_ANALYSIS, required=[RequiredInputKind.DATASET])
    missing = await DynamicPlanner().create_plan(
        "Detect outliers in the dataset", intent, PlanningInputMetadata(authorized_file_path=str(path))
    )
    assert missing.response_type is PlanningResponseType.CLARIFICATION
    resolved = await DynamicPlanner().create_plan(
        "Detect outliers in Sales", intent,
        PlanningInputMetadata(authorized_file_path=str(path), column_types={"Sales": ColumnKind.NUMERIC}),
    )
    assert resolved.response_type is PlanningResponseType.PLAN
    assert resolved.plan.steps[0].parameters["column"] == "Sales"


@pytest.mark.asyncio
async def test_document_read_plan_is_local_and_semantic_analysis_requests_clarification(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch, "document.txt")
    metadata = PlanningInputMetadata(authorized_file_path=str(path))
    planner = DynamicPlanner()
    read = await planner.create_plan(
        "Read this document",
        _intent(IntentCategory.DOCUMENT_ANALYSIS, required=[RequiredInputKind.DOCUMENT]),
        metadata,
    )
    assert read.response_type is PlanningResponseType.PLAN
    assert read.plan.steps[0].tool_name == "read_document"
    analyze = await planner.create_plan(
        "Analyze this document",
        _intent(IntentCategory.DOCUMENT_ANALYSIS, required=[RequiredInputKind.DOCUMENT]),
        metadata,
    )
    assert analyze.response_type is PlanningResponseType.CLARIFICATION


@pytest.mark.asyncio
async def test_sequential_inspection_and_statistics_plan_has_valid_dependency(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    result = await DynamicPlanner().create_plan(
        "Inspect the dataset and then calculate descriptive statistics",
        _intent(IntentCategory.MULTI_STEP_ANALYSIS, required=[RequiredInputKind.DATASET], multi=True),
        PlanningInputMetadata(authorized_file_path=str(path)),
    )
    assert result.response_type is PlanningResponseType.PLAN
    assert [step.tool_name for step in result.plan.steps] == ["inspect_dataset", "descriptive_statistics"]
    assert result.plan.steps[1].dependencies == [result.plan.steps[0].step_id]


@pytest.mark.asyncio
async def test_correlation_and_visualization_plan_uses_schema_and_dependencies(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    result = await DynamicPlanner().create_plan(
        "Find correlations and visualize them",
        _intent(IntentCategory.MULTI_STEP_ANALYSIS, required=[RequiredInputKind.DATASET], multi=True),
        PlanningInputMetadata(
            authorized_file_path=str(path),
            column_types={"Sales": ColumnKind.NUMERIC, "Profit": ColumnKind.NUMERIC},
        ),
    )
    assert result.response_type is PlanningResponseType.PLAN
    assert [step.tool_name for step in result.plan.steps] == ["correlation_analysis", "generate_chart"]
    assert result.plan.steps[1].dependencies == [result.plan.steps[0].step_id]
    assert result.plan.steps[1].parameters["chart_type"] == "heatmap"


@pytest.mark.asyncio
async def test_research_plan_requires_cloud_mode_and_uses_registered_tool():
    planner = DynamicPlanner()
    intent = _intent(IntentCategory.WEB_RESEARCH, required=[RequiredInputKind.RESEARCH_TOPIC])
    hybrid = await planner.create_plan("Research current solar market trends", intent, privacy_mode="hybrid")
    assert hybrid.response_type is PlanningResponseType.CLARIFICATION
    cloud = await planner.create_plan("Research current solar market trends", intent, privacy_mode="cloud_enabled")
    assert cloud.response_type is PlanningResponseType.PLAN
    assert cloud.plan.steps[0].tool_name == "web_research"


@pytest.mark.asyncio
async def test_local_only_and_hybrid_keep_dataset_planning_local(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    intent = _intent(IntentCategory.DATA_ANALYSIS, required=[RequiredInputKind.DATASET])
    for mode in ("local_only", "hybrid"):
        result = await DynamicPlanner().create_plan(
            "Inspect the uploaded dataset", intent,
            PlanningInputMetadata(authorized_file_path=str(path)), privacy_mode=mode,
        )
        assert result.response_type is PlanningResponseType.PLAN
        assert result.plan.steps[0].tool_name == "inspect_dataset"


@pytest.mark.asyncio
async def test_research_then_report_clarifies_instead_of_fabricating_future_findings():
    result = await DynamicPlanner().create_plan(
        "Research current solar market trends and prepare a report",
        _intent(IntentCategory.MULTI_STEP_ANALYSIS, required=[RequiredInputKind.RESEARCH_TOPIC], multi=True),
        privacy_mode="cloud_enabled",
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION
    assert "findings" in result.explanation.lower()


@pytest.mark.asyncio
async def test_unsupported_or_ambiguous_intent_returns_clarification():
    for intent in (
        _intent(IntentCategory.UNKNOWN),
        IntentAssessment(category="data_analysis", confidence=0.2, is_multi_step=False),
    ):
        result = await DynamicPlanner().create_plan("Analyze this", intent)
        assert result.response_type is PlanningResponseType.CLARIFICATION
        assert result.plan is None


@pytest.mark.asyncio
async def test_unavailable_tool_metadata_is_not_used():
    tools = [item for item in tool_registry.list_tools() if item.name != "system_health"]
    result = await DynamicPlanner(available_tools=tools).create_plan(
        "Check system health", _intent(IntentCategory.SYSTEM_HEALTH)
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION


@pytest.mark.asyncio
async def test_invalid_authorized_path_is_rejected(tmp_path):
    result = await DynamicPlanner().create_plan(
        "Inspect the dataset",
        _intent(IntentCategory.DATA_ANALYSIS, required=[RequiredInputKind.DATASET]),
        PlanningInputMetadata(authorized_file_path=str(tmp_path / "outside.csv")),
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION
    assert result.plan is None


@pytest.mark.asyncio
async def test_gemini_provider_is_called_only_in_cloud_enabled_and_identity_is_explicit():
    content = _provider_json([_step("step_health", "system_health", {"check_storage": True})])
    provider = StubProvider(content)
    planner = DynamicPlanner(provider, ProviderIdentity.GEMINI)
    result = await planner.create_plan("Check system health", _intent(IntentCategory.SYSTEM_HEALTH), privacy_mode="cloud_enabled")
    assert result.response_type is PlanningResponseType.PLAN
    assert planner.last_provider_identity is ProviderIdentity.GEMINI
    assert result.provider_identity is ProviderIdentity.GEMINI
    assert len(provider.calls) == 1
    assert "Check system health" in provider.calls[0][-1].content


@pytest.mark.asyncio
async def test_provider_never_receives_file_path_or_column_metadata(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    content = _provider_json([_step("step_inspect", "inspect_dataset", {"file_path": "C:/fabricated.csv", "sample_rows": 5})])
    provider = StubProvider(content)
    planner = DynamicPlanner(provider, ProviderIdentity.GEMINI)
    result = await planner.create_plan(
        "Inspect the uploaded dataset",
        _intent(IntentCategory.DATA_ANALYSIS, required=[RequiredInputKind.DATASET]),
        PlanningInputMetadata(authorized_file_path=str(path), column_types={"Private Customer Field": ColumnKind.NUMERIC}),
        privacy_mode="cloud_enabled",
    )
    assert result.response_type is PlanningResponseType.PLAN
    request = provider.calls[0][-1].content
    assert str(path) not in request
    assert "Private Customer Field" not in request
    assert result.plan.steps[0].parameters["file_path"] == str(path.resolve())
    assert "C:/fabricated.csv" not in json.dumps(result.plan.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_gemini_is_not_called_or_silently_replaced_in_local_or_hybrid():
    provider = StubProvider(_provider_json([_step("step_health", "system_health", {"check_storage": True})]))
    planner = DynamicPlanner(provider, ProviderIdentity.GEMINI)
    for mode in ("local_only", "hybrid"):
        result = await planner.create_plan("Check system health", _intent(IntentCategory.SYSTEM_HEALTH), privacy_mode=mode)
        assert result.response_type is PlanningResponseType.CLARIFICATION
        assert not provider.calls
        assert planner.last_provider_identity is ProviderIdentity.GEMINI


@pytest.mark.asyncio
async def test_provider_failure_is_safe_and_does_not_fall_back():
    provider = StubProvider(error=RuntimeError("sensitive provider response body"))
    planner = DynamicPlanner(provider, ProviderIdentity.GEMINI)
    result = await planner.create_plan("Check system health", _intent(IntentCategory.SYSTEM_HEALTH), privacy_mode="cloud_enabled")
    assert result.response_type is PlanningResponseType.CLARIFICATION
    assert result.provider_identity is ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
    assert "sensitive provider response body" not in result.model_dump_json()
    assert len(provider.calls) == 1
    assert planner.last_provider_identity is ProviderIdentity.GEMINI


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_steps", [
    [_step("step_unknown", "not_registered", {})],
    [_step("step_health", "system_health", {"check_storage": "not-a-bool"})],
    [_step("step_dup", "system_health", {"check_storage": True}), _step("step_dup", "system_health", {"check_storage": True})],
    [_step("step_health", "system_health", {"check_storage": True}, ["step_missing"])],
    [_step("step_a", "system_health", {"check_storage": True}, ["step_b"]), _step("step_b", "system_health", {"check_storage": True}, ["step_a"])],
])
async def test_malicious_or_invalid_provider_plans_are_rejected(bad_steps):
    planner = DynamicPlanner(StubProvider(_provider_json(bad_steps)), ProviderIdentity.GEMINI)
    result = await planner.create_plan("Check system health", _intent(IntentCategory.SYSTEM_HEALTH), privacy_mode="cloud_enabled")
    assert result.response_type is PlanningResponseType.CLARIFICATION
    assert result.plan is None


@pytest.mark.asyncio
async def test_provider_cannot_inject_unregistered_column_or_extra_plan_fields(tmp_path, monkeypatch):
    path = _managed_file(tmp_path, monkeypatch)
    bad = _step("step_chart", "generate_chart", {
        "file_path": "unused.csv", "chart_type": "histogram", "title": "private", "x_column": "invented_column"
    })
    provider = StubProvider(_provider_json([bad]))
    result = await DynamicPlanner(provider, ProviderIdentity.GEMINI).create_plan(
        "Make a histogram",
        _intent(IntentCategory.VISUALIZATION, required=[RequiredInputKind.DATASET]),
        PlanningInputMetadata(authorized_file_path=str(path), column_types={"Revenue": ColumnKind.NUMERIC}),
        privacy_mode="cloud_enabled",
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION

    bad_stats = _provider_json([_step(
        "step_stats", "descriptive_statistics",
        {"file_path": "unused.csv", "columns": ["invented_column"]},
    )])
    result = await DynamicPlanner(StubProvider(bad_stats), ProviderIdentity.GEMINI).create_plan(
        "Calculate descriptive statistics",
        _intent(IntentCategory.STATISTICAL_ANALYSIS, required=[RequiredInputKind.DATASET]),
        PlanningInputMetadata(authorized_file_path=str(path), column_types={"Revenue": ColumnKind.NUMERIC}),
        privacy_mode="cloud_enabled",
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION

    extra = _provider_json([{**_step("step_health", "system_health", {"check_storage": True}), "output": "sensitive"}])
    result = await DynamicPlanner(StubProvider(extra), ProviderIdentity.GEMINI).create_plan(
        "Check system health", _intent(IntentCategory.SYSTEM_HEALTH), privacy_mode="cloud_enabled"
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION


@pytest.mark.asyncio
async def test_provider_step_limit_and_intent_consistency_are_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MAX_PLAN_STEPS", 1)
    two_steps = [
        _step("step_a", "system_health", {"check_storage": True}),
        _step("step_b", "system_health", {"check_storage": True}),
    ]
    result = await DynamicPlanner(StubProvider(_provider_json(two_steps)), ProviderIdentity.GEMINI).create_plan(
        "Check system health", _intent(IntentCategory.SYSTEM_HEALTH), privacy_mode="cloud_enabled"
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION

    path = _managed_file(tmp_path, monkeypatch)
    mismatched = _provider_json([_step("step_inspect", "inspect_dataset", {"file_path": "ignore.csv", "sample_rows": 5})])
    result = await DynamicPlanner(StubProvider(mismatched), ProviderIdentity.GEMINI).create_plan(
        "Check system health", _intent(IntentCategory.SYSTEM_HEALTH),
        PlanningInputMetadata(authorized_file_path=str(path)), privacy_mode="cloud_enabled"
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION


@pytest.mark.asyncio
async def test_provider_identity_mismatch_and_mock_identity_are_never_mislabeled(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    mismatch = DynamicPlanner(StubProvider("{}", provider="test_reasoner"), ProviderIdentity.GEMINI)
    response = await mismatch.create_plan("Check system health", _intent(IntentCategory.SYSTEM_HEALTH), privacy_mode="cloud_enabled")
    assert response.response_type is PlanningResponseType.CLARIFICATION
    assert mismatch.last_provider_identity is ProviderIdentity.GEMINI

    mock = DynamicPlanner(StubProvider("{}", provider="test_reasoner"), ProviderIdentity.TEST_MOCK_PROVIDER)
    response = await mock.create_plan("Check system health", _intent(IntentCategory.SYSTEM_HEALTH))
    assert response.response_type is PlanningResponseType.CLARIFICATION
    assert mock.last_provider_identity is ProviderIdentity.TEST_MOCK_PROVIDER


@pytest.mark.asyncio
async def test_test_mock_provider_identity_is_preserved_in_test_configuration(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "test")
    content = _provider_json([_step("step_health", "system_health", {"check_storage": True})])
    planner = DynamicPlanner(StubProvider(content, provider="test_reasoner"), ProviderIdentity.TEST_MOCK_PROVIDER)
    result = await planner.create_plan(
        "Check system health", _intent(IntentCategory.SYSTEM_HEALTH), privacy_mode="local_only"
    )
    assert result.response_type is PlanningResponseType.PLAN
    assert result.provider_identity is ProviderIdentity.TEST_MOCK_PROVIDER


@pytest.mark.asyncio
async def test_mock_provider_is_rejected_outside_test_configuration(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    provider = StubProvider(_provider_json([_step("step_health", "system_health", {"check_storage": True})]), provider="test_reasoner")
    planner = DynamicPlanner(provider, ProviderIdentity.TEST_MOCK_PROVIDER)
    result = await planner.create_plan("Check system health", _intent(IntentCategory.SYSTEM_HEALTH))
    assert result.response_type is PlanningResponseType.CLARIFICATION
    assert result.provider_identity is ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
    assert not provider.calls


@pytest.mark.asyncio
async def test_cloud_provider_does_not_receive_objectives_containing_file_paths():
    provider = StubProvider(_provider_json([_step("step_health", "system_health", {"check_storage": True})]))
    planner = DynamicPlanner(provider, ProviderIdentity.GEMINI)
    result = await planner.create_plan(
        "Inspect C:\\Users\\private\\data.csv and check the status",
        _intent(IntentCategory.SYSTEM_HEALTH),
        privacy_mode="cloud_enabled",
    )
    assert result.response_type is PlanningResponseType.CLARIFICATION
    assert not provider.calls
