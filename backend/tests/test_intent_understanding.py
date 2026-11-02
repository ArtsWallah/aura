import socket

import httpx
from app.agent.intelligence_contracts import (
    IntentCategory,
    RequiredInputKind,
)
from app.agent.intent import IntentUnderstander, assess_intent
from app.tools.registry import tool_registry


def _assess(objective: str, *inputs: RequiredInputKind):
    return assess_intent(objective, available_inputs=inputs)


def test_clear_dataset_inspection_uses_registered_dataset_tool():
    result = _assess("Inspect the uploaded dataset and show its columns", RequiredInputKind.DATASET)
    assert result.category is IntentCategory.DATA_ANALYSIS
    assert result.confidence >= 0.7
    assert [item.kind for item in result.required_inputs] == [RequiredInputKind.DATASET]
    assert result.missing_inputs == []
    assert not result.is_multi_step


def test_descriptive_statistics_request_is_statistical_analysis():
    result = _assess("Calculate descriptive statistics for my dataset", RequiredInputKind.DATASET)
    assert result.category is IntentCategory.STATISTICAL_ANALYSIS
    assert result.confidence >= 0.7
    assert result.missing_inputs == []


def test_group_comparison_request_is_detected_and_requires_dataset():
    result = _assess("Compare groups by region and revenue", RequiredInputKind.DATASET)
    assert result.category is IntentCategory.STATISTICAL_ANALYSIS
    assert RequiredInputKind.DATASET in {item.kind for item in result.required_inputs}
    assert result.missing_inputs == []


def test_correlation_request_is_detected():
    result = _assess("Calculate correlation between revenue and profit", RequiredInputKind.DATASET)
    assert result.category is IntentCategory.STATISTICAL_ANALYSIS
    assert result.confidence >= 0.7


def test_visualization_request_is_detected_and_missing_dataset_is_reported():
    result = _assess("Create a bar chart of sales data")
    assert result.category is IntentCategory.VISUALIZATION
    assert [item.kind for item in result.missing_inputs] == [RequiredInputKind.DATASET]


def test_document_analysis_requires_safe_document_presence_metadata():
    result = _assess("Summarize this PDF contract")
    assert result.category is IntentCategory.DOCUMENT_ANALYSIS
    assert [item.kind for item in result.missing_inputs] == [RequiredInputKind.DOCUMENT]
    supplied = _assess("Summarize this PDF contract", RequiredInputKind.DOCUMENT)
    assert supplied.missing_inputs == []


def test_web_research_request_is_detected_without_external_call():
    result = _assess("Research current solar market trends and competitors")
    assert result.category is IntentCategory.WEB_RESEARCH
    assert result.missing_inputs == []


def test_bare_research_request_reports_missing_topic():
    result = _assess("Research")
    assert result.category is IntentCategory.WEB_RESEARCH
    assert [item.kind for item in result.missing_inputs] == [RequiredInputKind.RESEARCH_TOPIC]


def test_report_generation_request_is_detected():
    result = _assess("Create a report about the sales results")
    assert result.category is IntentCategory.REPORT_GENERATION
    assert result.confidence >= 0.7
    assert result.missing_inputs == []


def test_combined_dataset_and_chart_request_is_marked_multi_step():
    result = _assess(
        "Analyze the dataset and then create a chart of the results",
        RequiredInputKind.DATASET,
    )
    assert result.category is IntentCategory.MULTI_STEP_ANALYSIS
    assert result.is_multi_step


def test_vague_request_is_ambiguous_and_unsupported_request_is_not_analytics():
    ambiguous = _assess("Analyze this")
    assert ambiguous.category is IntentCategory.UNKNOWN
    assert ambiguous.confidence < 0.5
    assert [item.kind for item in ambiguous.missing_inputs] == [RequiredInputKind.CLARIFICATION]

    unsupported = _assess("Send an email to the customer")
    assert unsupported.category is IntentCategory.UNKNOWN
    assert unsupported.confidence < ambiguous.confidence
    assert unsupported.missing_inputs == []


def test_sql_request_is_unsupported_without_registered_sql_capability():
    result = _assess("Run a SQL query against the customer database")
    assert result.category is IntentCategory.UNKNOWN


def test_available_tool_metadata_controls_capability_classification():
    no_dataset_tool = [item for item in tool_registry.list_tools() if item.name != "inspect_dataset"]
    result = assess_intent("Inspect the dataset and show its columns", available_tools=no_dataset_tool)
    assert result.category is IntentCategory.UNKNOWN


def test_input_metadata_accepts_only_known_safe_values():
    missing = _assess("Inspect the dataset and show its columns")
    present = _assess("Inspect the dataset and show its columns", RequiredInputKind.DATASET)
    ignored = assess_intent(
        "Inspect the dataset and show its columns",
        available_inputs=["C:\\private\\customer.csv"],
    )
    assert missing.missing_inputs[0].kind is RequiredInputKind.DATASET
    assert present.missing_inputs == []
    assert ignored.missing_inputs[0].kind is RequiredInputKind.DATASET


def test_assessment_never_copies_objective_paths_or_sensitive_values():
    marker = "CUSTOMER-SSN-938-22-1047"
    objective = f"Inspect my dataset at C:\\uploads\\secret.csv for {marker}"
    result = assess_intent(objective, available_inputs=[RequiredInputKind.DATASET])
    serialized = result.model_dump_json()
    assert marker not in serialized
    assert "secret.csv" not in serialized
    assert objective not in serialized


def test_classifier_makes_no_network_calls(monkeypatch):
    calls = []

    async def blocked_http(*args, **kwargs):
        calls.append("httpx")
        raise AssertionError("Intent assessment must not use the network")

    def blocked_socket(*args, **kwargs):
        calls.append("socket")
        raise AssertionError("Intent assessment must not use the network")

    monkeypatch.setattr(httpx.AsyncClient, "request", blocked_http)
    monkeypatch.setattr(socket, "create_connection", blocked_socket)
    assessment = _assess("Calculate descriptive statistics for the dataset", RequiredInputKind.DATASET)
    assert assessment.category is IntentCategory.STATISTICAL_ANALYSIS
    assert calls == []


def test_understander_accepts_an_injected_registry_and_returns_contract_type():
    understander = IntentUnderstander(tool_registry)
    result = understander.assess("Check system health")
    assert result.category is IntentCategory.SYSTEM_HEALTH
    assert isinstance(result, type(assess_intent("unknown request")))
