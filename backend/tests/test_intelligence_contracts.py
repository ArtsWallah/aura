import pytest
from pydantic import ValidationError

from app.agent.intelligence_contracts import (
    AlternativeReason,
    EvaluationResult,
    ExecutionTiming,
    ExecutionTrace,
    InputRequirement,
    IntentAssessment,
    IntentCategory,
    IssueCode,
    PlanningResponse,
    PlanningResponseType,
    ProviderIdentity,
    RequiredInputKind,
    RetryRecommendation,
    SafeAlternativeRecommendation,
    ValidationOutcome,
)
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan-1",
        goal="analyze a dataset",
        steps=[ExecutionStep(
            step_id="step_inspect",
            title="Inspect",
            description="Inspect dataset",
            capability="dataset",
            tool_name="inspect_dataset",
            parameters={"file_path": "authorized.csv"},
        )],
    )


def test_intent_contract_validates_category_confidence_and_input_shapes():
    assessment = IntentAssessment(
        category="statistical_analysis",
        confidence=0.75,
        required_inputs=[{"kind": "dataset"}],
        missing_inputs=[{"kind": "dataset", "required": True}],
        is_multi_step=False,
    )
    assert assessment.category is IntentCategory.STATISTICAL_ANALYSIS
    assert assessment.model_dump(mode="json")["missing_inputs"] == [
        {"kind": "dataset", "required": True}
    ]

    with pytest.raises(ValidationError):
        IntentAssessment(category="made_up", confidence=0.5, is_multi_step=False)
    for confidence in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            IntentAssessment(category="unknown", confidence=confidence, is_multi_step=False)
    with pytest.raises(ValidationError):
        InputRequirement(kind="dataset", required=True, path="private.csv")


def test_planning_response_enforces_plan_and_clarification_variants():
    plan_response = PlanningResponse(
        response_type="PLAN", plan=_plan(), explanation="Inspect, then summarize the dataset."
    )
    assert plan_response.plan is not None
    clarification = PlanningResponse(
        response_type="CLARIFICATION",
        clarification_question="Which date column should be used?",
        explanation="The request needs a date field.",
    )
    assert clarification.plan is None

    invalid = [
        {"response_type": "PLAN", "explanation": "missing plan"},
        {"response_type": "PLAN", "plan": _plan(), "clarification_question": "Which?", "explanation": "x"},
        {"response_type": "CLARIFICATION", "explanation": "missing question"},
        {"response_type": "CLARIFICATION", "plan": _plan(), "clarification_question": "Which?", "explanation": "x"},
    ]
    for payload in invalid:
        with pytest.raises(ValidationError):
            PlanningResponse.model_validate(payload)


def test_evaluation_contract_rejects_invalid_bounds_and_unsafe_combinations():
    assert EvaluationResult(execution_valid=True, objective_coverage=1.0).execution_valid
    with pytest.raises(ValidationError):
        EvaluationResult(execution_valid=False, objective_coverage=1.2)
    with pytest.raises(ValidationError):
        EvaluationResult(execution_valid=True, objective_coverage=1.0, issues=[IssueCode.TOOL_FAILED])
    with pytest.raises(ValidationError):
        EvaluationResult(execution_valid=False, objective_coverage=0.2, retry_recommendation="replan")
    with pytest.raises(ValidationError):
        SafeAlternativeRecommendation(tool_name="../../secret", reason=AlternativeReason.OUTPUT_INVALID)

    replan = EvaluationResult(
        execution_valid=False,
        objective_coverage=0.2,
        issues=[IssueCode.OUTPUT_INVALID],
        retry_recommendation=RetryRecommendation.REPLAN,
        alternative=SafeAlternativeRecommendation(
            tool_name="read_document", reason=AlternativeReason.OUTPUT_INVALID
        ),
    )
    assert replan.alternative.requires_plan_revalidation is True


def test_trace_is_allowlisted_rejects_sensitive_or_unknown_fields_and_serializes():
    trace = ExecutionTrace(
        intent_category=IntentCategory.DATA_ANALYSIS,
        provider_identity=ProviderIdentity.GEMINI,
        selected_tools=["inspect_dataset"],
        step_statuses=[{
            "step_id": "step_inspect",
            "tool_name": "inspect_dataset",
            "status": StepStatus.COMPLETED,
            "duration_ms": 12.5,
        }],
        validation_outcome=ValidationOutcome.ACCEPTED,
        timing=ExecutionTiming(planning_ms=10, validation_ms=1, execution_ms=12.5),
    )
    payload = trace.model_dump(mode="json")
    assert payload["provider_identity"] == "gemini"
    assert payload["step_statuses"][0]["status"] == "completed"
    assert "objective" not in payload and "prompt" not in payload and "raw_output" not in payload

    for extra in ("objective", "prompt", "file_contents", "dataset_rows", "tool_arguments", "raw_output"):
        with pytest.raises(ValidationError):
            ExecutionTrace.model_validate({**payload, extra: "SENSITIVE-MARKER"})
    with pytest.raises(ValidationError):
        ExecutionTrace.model_validate({**payload, "provider_identity": "gemini-mock-fallback"})
    with pytest.raises(ValidationError):
        ExecutionTrace.model_validate({**payload, "selected_tools": ["../../private"]})


def test_provider_identity_keeps_mock_and_local_distinct_from_gemini():
    assert ProviderIdentity("gemini") is ProviderIdentity.GEMINI
    assert ProviderIdentity("local_deterministic_planner") is ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
    assert ProviderIdentity("test_mock_provider") is ProviderIdentity.TEST_MOCK_PROVIDER
    with pytest.raises(ValueError):
        ProviderIdentity("gemini-mock-fallback")


def test_contracts_round_trip_without_changing_existing_task_models():
    plan = _plan()
    decoded = ExecutionPlan.model_validate(plan.model_dump())
    assert decoded == plan
    response = PlanningResponse(
        response_type=PlanningResponseType.PLAN,
        plan=decoded,
        explanation="Plan is structurally valid.",
    )
    round_trip = PlanningResponse.model_validate_json(response.model_dump_json())
    assert round_trip.plan == plan
    assert IntentAssessment(
        category=IntentCategory.UNKNOWN,
        confidence=0,
        required_inputs=[InputRequirement(kind=RequiredInputKind.ANALYSIS_GOAL)],
        is_multi_step=False,
    ).confidence == 0
