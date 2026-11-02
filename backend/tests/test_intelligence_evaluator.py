import pytest
from pydantic import ValidationError

from app.agent.intelligence_contracts import (
    EvaluationStatus,
    InputRequirement,
    IntentAssessment,
    IntentCategory,
    IssueCode,
    ProviderIdentity,
    RequiredInputKind,
    RetryRecommendation,
    ValidationOutcome,
)
from app.agent.intelligence_evaluator import (
    EvaluationInput,
    EvidenceResultKind,
    IntelligenceEvaluator,
    PlanStepSummary,
    PlanValidationEvidence,
    StepOutcomeEvidence,
)
from app.agent.state import StepStatus
from app.tools.registry import tool_registry


def _intent(category=IntentCategory.SYSTEM_HEALTH, missing=(), multi=False):
    return IntentAssessment(
        category=category,
        confidence=0.9,
        required_inputs=[InputRequirement(kind=RequiredInputKind.DATASET)] if category == IntentCategory.DATA_ANALYSIS else [],
        missing_inputs=[InputRequirement(kind=item) for item in missing],
        is_multi_step=multi,
    )


def _request(
    *, category=IntentCategory.SYSTEM_HEALTH, steps=None, outcomes=None,
    target="plan", validation=ValidationOutcome.ACCEPTED, missing=(),
):
    return EvaluationInput(
        intent=_intent(category, missing),
        steps=steps or [PlanStepSummary(step_id="step_health", tool_name="system_health")],
        validation=PlanValidationEvidence(outcome=validation),
        evaluation_target=target,
        outcomes=outcomes or [],
    )


def _health_outcome(**overrides):
    values = {
        "step_id": "step_health",
        "status": StepStatus.COMPLETED,
        "result_kind": EvidenceResultKind.SYSTEM_STATUS,
        "result_count": 1,
    }
    values.update(overrides)
    return StepOutcomeEvidence(**values)


def test_valid_plan_with_complete_intent_coverage_is_plan_valid():
    result = IntelligenceEvaluator().evaluate(_request())
    assert result.evaluation_status is EvaluationStatus.PLAN_VALID
    assert result.execution_valid is True
    assert result.objective_coverage == 1.0
    assert not result.issues


def test_valid_plan_with_incomplete_coverage_returns_registered_alternative():
    request = _request(
        category=IntentCategory.DATA_ANALYSIS,
        steps=[PlanStepSummary(step_id="step_health", tool_name="system_health")],
    )
    result = IntelligenceEvaluator().evaluate(request)
    assert result.evaluation_status is EvaluationStatus.INCOMPLETE_COVERAGE
    assert result.execution_valid is True
    assert result.objective_coverage == 0.0
    assert IssueCode.OBJECTIVE_UNCOVERED in result.issues
    assert result.retry_recommendation is RetryRecommendation.REPLAN
    assert result.alternative.tool_name == "inspect_dataset"
    assert result.alternative.requires_plan_revalidation is True


def test_unregistered_canonical_alternative_is_never_recommended():
    class RegistryWithoutDatasetInspector:
        def list_tools(self):
            return [tool for tool in tool_registry.list_tools() if tool.name != "inspect_dataset"]

    request = _request(
        category=IntentCategory.DATA_ANALYSIS,
        steps=[PlanStepSummary(step_id="step_health", tool_name="system_health")],
    )
    result = IntelligenceEvaluator(RegistryWithoutDatasetInspector()).evaluate(request)
    assert result.evaluation_status is EvaluationStatus.INCOMPLETE_COVERAGE
    assert result.alternative is None
    assert result.retry_recommendation is RetryRecommendation.NONE


@pytest.mark.parametrize("steps", [
    [PlanStepSummary(step_id="step_unregistered", tool_name="not_registered")],
    [PlanStepSummary(step_id="step_health", tool_name="system_health", dependencies=["step_missing"])],
    [
        PlanStepSummary(step_id="step_a", tool_name="system_health", dependencies=["step_b"]),
        PlanStepSummary(step_id="step_b", tool_name="system_health", dependencies=["step_a"]),
    ],
])
def test_invalid_plan_or_dependencies_are_rejected(steps):
    result = IntelligenceEvaluator().evaluate(_request(steps=steps))
    assert result.evaluation_status is EvaluationStatus.INVALID_PLAN
    assert result.execution_valid is False


def test_rejected_plan_validation_is_invalid_plan():
    result = IntelligenceEvaluator().evaluate(_request(validation=ValidationOutcome.REJECTED))
    assert result.evaluation_status is EvaluationStatus.INVALID_PLAN
    assert IssueCode.POLICY_REJECTED in result.issues


def test_execution_success_requires_completed_steps_and_expected_typed_results():
    result = IntelligenceEvaluator().evaluate(
        _request(target="execution", outcomes=[_health_outcome()])
    )
    assert result.evaluation_status is EvaluationStatus.SUCCEEDED
    assert result.execution_valid is True
    assert result.objective_coverage == 1.0


def test_failed_retry_safe_step_returns_bounded_retry_recommendation():
    result = IntelligenceEvaluator().evaluate(_request(
        target="execution",
        outcomes=[_health_outcome(status=StepStatus.FAILED, result_kind=None, result_count=None)],
    ))
    assert result.evaluation_status is EvaluationStatus.EXECUTION_FAILED
    assert result.retry_recommendation is RetryRecommendation.RETRY


def test_failed_non_retry_safe_step_does_not_recommend_retry():
    request = _request(
        category=IntentCategory.REPORT_GENERATION,
        steps=[PlanStepSummary(step_id="step_report", tool_name="generate_report")],
        target="execution",
        outcomes=[StepOutcomeEvidence(step_id="step_report", status=StepStatus.FAILED)],
    )
    result = IntelligenceEvaluator().evaluate(request)
    assert result.evaluation_status is EvaluationStatus.EXECUTION_FAILED
    assert result.retry_recommendation is RetryRecommendation.NONE


def test_empty_or_wrong_typed_result_is_an_execution_failure():
    evaluator = IntelligenceEvaluator()
    empty = evaluator.evaluate(_request(target="execution", outcomes=[_health_outcome(result_count=0)]))
    wrong_type = evaluator.evaluate(_request(
        target="execution", outcomes=[_health_outcome(result_kind=EvidenceResultKind.DATASET_PROFILE)]
    ))
    assert empty.evaluation_status is EvaluationStatus.EXECUTION_FAILED
    assert IssueCode.OUTPUT_EMPTY in empty.issues
    assert IssueCode.OUTPUT_INVALID in wrong_type.issues


def test_missing_evidence_and_unsupported_multi_step_assessment_are_inconclusive():
    evaluator = IntelligenceEvaluator()
    absent = evaluator.evaluate(_request(target="execution"))
    multi = evaluator.evaluate(_request(category=IntentCategory.MULTI_STEP_ANALYSIS))
    assert absent.evaluation_status is EvaluationStatus.INCONCLUSIVE
    assert IssueCode.MISSING_EVIDENCE in absent.issues
    assert multi.evaluation_status is EvaluationStatus.INCONCLUSIVE


def test_missing_required_input_is_inconclusive():
    result = IntelligenceEvaluator().evaluate(_request(missing=[RequiredInputKind.DATASET]))
    assert result.evaluation_status is EvaluationStatus.INCONCLUSIVE
    assert IssueCode.MISSING_INPUT in result.issues


def test_oversized_or_malformed_evidence_fails_closed():
    evaluator = IntelligenceEvaluator()
    oversized = evaluator.evaluate({
        "intent": _intent().model_dump(mode="json"),
        "steps": [{"step_id": "step_health", "tool_name": "system_health"}],
        "validation": {"outcome": "accepted"},
        "evaluation_target": "execution",
        "outcomes": [{
            "step_id": "step_health", "status": "completed",
            "result_kind": "system_status", "result_count": 10**20,
        }],
    })
    malformed = evaluator.evaluate({
        **_request().model_dump(mode="json"),
        "sensitive_raw_output": "SHOULD_NOT_ESCAPE",
    })
    assert oversized.evaluation_status is EvaluationStatus.INCONCLUSIVE
    assert malformed.evaluation_status is EvaluationStatus.INCONCLUSIVE
    assert "SHOULD_NOT_ESCAPE" not in oversized.model_dump_json() + malformed.model_dump_json()


def test_evaluation_result_never_echoes_raw_or_sensitive_evidence():
    marker = "PRIVATE-DOCUMENT-CONTENT"
    payload = _request().model_dump(mode="json")
    payload["sensitive_path"] = marker
    result = IntelligenceEvaluator().evaluate(payload)
    assert marker not in result.model_dump_json()


def test_evaluation_contract_addition_keeps_legacy_payloads_compatible():
    from app.agent.intelligence_contracts import EvaluationResult as ContractEvaluationResult

    legacy = ContractEvaluationResult(execution_valid=True, objective_coverage=1.0)
    assert legacy.evaluation_status is None
    round_trip = ContractEvaluationResult.model_validate_json(legacy.model_dump_json())
    assert round_trip == legacy
    assert ProviderIdentity.GEMINI.value == "gemini"


def test_evidence_models_forbid_unbounded_free_form_fields():
    with pytest.raises(ValidationError):
        StepOutcomeEvidence(
            step_id="step_health", status=StepStatus.COMPLETED,
            result_kind=EvidenceResultKind.SYSTEM_STATUS, result_count=1,
            raw_output={"marker": "private"},
        )
