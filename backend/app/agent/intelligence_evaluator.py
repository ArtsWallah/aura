"""Local, evidence-bounded evaluation for Phase 3 intelligence contracts.

This module intentionally accepts plan summaries and typed outcome metadata,
never objectives, execution arguments, paths, or raw tool output. It does not
call providers or trigger retries/replanning.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from app.agent.intelligence_contracts import (
    AlternativeReason,
    EvaluationResult,
    EvaluationStatus,
    IntentAssessment,
    IntentCategory,
    IssueCode,
    RetryRecommendation,
    SafeAlternativeRecommendation,
    StrictContract,
    ValidationOutcome,
)
from app.agent.state import StepStatus
from app.core.config import settings
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.schemas import ExecutionLocation, PermissionLevel, ToolCategory, ToolDefinition

import app.tools  # noqa: F401,E402  Register built-in tool metadata locally.


class EvidenceResultKind(str, Enum):
    DATASET_PROFILE = "dataset_profile"
    STATISTICAL_SUMMARY = "statistical_summary"
    GROUP_COMPARISON = "group_comparison"
    CORRELATION = "correlation"
    OUTLIER_ANALYSIS = "outlier_analysis"
    VISUALIZATION = "visualization"
    DOCUMENT_EXTRACTION = "document_extraction"
    RESEARCH_RESULTS = "research_results"
    REPORT = "report"
    SYSTEM_STATUS = "system_status"


_STEP_ID = r"^step_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
_TOOL_ID = r"^[a-z][a-z0-9_]{0,63}$"


class PlanStepSummary(StrictContract):
    step_id: str = Field(pattern=_STEP_ID, max_length=70)
    tool_name: str = Field(pattern=_TOOL_ID, max_length=64)
    dependencies: list[str] = Field(default_factory=list, max_length=32)


class PlanValidationEvidence(StrictContract):
    outcome: ValidationOutcome
    issues: list[IssueCode] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def require_consistent_validation(self) -> "PlanValidationEvidence":
        if self.outcome == ValidationOutcome.ACCEPTED and self.issues:
            raise ValueError("Accepted plan validation cannot contain issues.")
        return self


class StepOutcomeEvidence(StrictContract):
    """Minimized result metadata; deliberately has no free-form payload field."""

    step_id: str = Field(pattern=_STEP_ID, max_length=70)
    status: StepStatus
    result_kind: EvidenceResultKind | None = None
    result_count: int | None = Field(default=None, ge=0, le=1_000_000_000)
    artifact_count: int | None = Field(default=None, ge=0, le=10_000)


class EvaluationInput(StrictContract):
    intent: IntentAssessment
    steps: list[PlanStepSummary] = Field(default_factory=list, max_length=100)
    validation: PlanValidationEvidence | None = None
    evaluation_target: Literal["plan", "execution"] = "plan"
    outcomes: list[StepOutcomeEvidence] = Field(default_factory=list, max_length=100)
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] = "hybrid"


_INTENT_CATEGORY: dict[IntentCategory, ToolCategory] = {
    IntentCategory.DATA_ANALYSIS: ToolCategory.DATASET,
    IntentCategory.STATISTICAL_ANALYSIS: ToolCategory.ANALYTICS,
    IntentCategory.VISUALIZATION: ToolCategory.VISUALIZATION,
    IntentCategory.DOCUMENT_ANALYSIS: ToolCategory.DOCUMENT,
    IntentCategory.WEB_RESEARCH: ToolCategory.RESEARCH,
    IntentCategory.REPORT_GENERATION: ToolCategory.REPORT,
    IntentCategory.SYSTEM_HEALTH: ToolCategory.SYSTEM,
}

# These are canonical, registry-backed capabilities for recommendations. The
# evaluator never guesses substitute tools from their names or descriptions.
_CANONICAL_TOOL: dict[IntentCategory, str] = {
    IntentCategory.DATA_ANALYSIS: "inspect_dataset",
    IntentCategory.VISUALIZATION: "generate_chart",
    IntentCategory.DOCUMENT_ANALYSIS: "read_document",
    IntentCategory.WEB_RESEARCH: "web_research",
    IntentCategory.REPORT_GENERATION: "generate_report",
    IntentCategory.SYSTEM_HEALTH: "system_health",
}

_RESULT_KIND: dict[str, EvidenceResultKind] = {
    "inspect_dataset": EvidenceResultKind.DATASET_PROFILE,
    "descriptive_statistics": EvidenceResultKind.STATISTICAL_SUMMARY,
    "group_comparison": EvidenceResultKind.GROUP_COMPARISON,
    "correlation_analysis": EvidenceResultKind.CORRELATION,
    "detect_outliers": EvidenceResultKind.OUTLIER_ANALYSIS,
    "generate_chart": EvidenceResultKind.VISUALIZATION,
    "read_document": EvidenceResultKind.DOCUMENT_EXTRACTION,
    "search_document_chunks": EvidenceResultKind.DOCUMENT_EXTRACTION,
    "web_research": EvidenceResultKind.RESEARCH_RESULTS,
    "generate_report": EvidenceResultKind.REPORT,
    "system_health": EvidenceResultKind.SYSTEM_STATUS,
}


class IntelligenceEvaluator:
    """Evaluate only validated summaries and bounded, typed evidence."""

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or tool_registry

    def evaluate(self, request: EvaluationInput | dict) -> EvaluationResult:
        """Return a typed assessment without retaining or echoing input details."""
        try:
            data = EvaluationInput.model_validate(request)
        except (ValidationError, TypeError, ValueError):
            return self._result(
                False, 0.0, EvaluationStatus.INCONCLUSIVE,
                [IssueCode.MISSING_EVIDENCE],
            )

        if data.validation is None:
            return self._result(False, 0.0, EvaluationStatus.INCONCLUSIVE, [IssueCode.MISSING_EVIDENCE])
        if data.validation.outcome == ValidationOutcome.REJECTED:
            issues = data.validation.issues or [IssueCode.POLICY_REJECTED]
            return self._result(False, 0.0, EvaluationStatus.INVALID_PLAN, issues)

        definitions = {definition.name: definition for definition in self.registry.list_tools()}
        plan_issue = self._validate_summary(data, definitions)
        if plan_issue is not None:
            return self._result(False, 0.0, EvaluationStatus.INVALID_PLAN, [plan_issue])

        if data.intent.missing_inputs:
            return self._result(False, 0.0, EvaluationStatus.INCONCLUSIVE, [IssueCode.MISSING_INPUT])
        if data.intent.category == IntentCategory.MULTI_STEP_ANALYSIS or data.intent.is_multi_step:
            # The Phase 3.1.2 assessment identifies multi-step intent but does
            # not carry its required sub-goals unless assessed tool groups exist.
            if not data.intent.required_tool_groups:
                return self._result(False, 0.0, EvaluationStatus.INCONCLUSIVE, [IssueCode.MISSING_EVIDENCE])
        expected_category = _INTENT_CATEGORY.get(data.intent.category)
        if (expected_category is None and not data.intent.required_tool_groups) or data.intent.category == IntentCategory.UNKNOWN:
            return self._result(False, 0.0, EvaluationStatus.INCONCLUSIVE, [IssueCode.UNKNOWN])

        selected_tools = {step.tool_name for step in data.steps}
        if data.intent.required_tool_groups:
            matched_groups = sum(bool(selected_tools.intersection(group)) for group in data.intent.required_tool_groups)
            coverage = matched_groups / len(data.intent.required_tool_groups)
            covered = matched_groups == len(data.intent.required_tool_groups)
        else:
            covered = any(definitions[step.tool_name].category == expected_category for step in data.steps)
            coverage = 1.0 if covered else 0.0
        if not covered:
            alternative = self._canonical_alternative(data.intent.category, definitions, data)
            return self._result(
                True,
                coverage,
                EvaluationStatus.INCOMPLETE_COVERAGE,
                [IssueCode.OBJECTIVE_UNCOVERED],
                RetryRecommendation.REPLAN if alternative else RetryRecommendation.NONE,
                alternative,
            )

        if data.evaluation_target == "execution" and not data.outcomes:
            return self._result(False, 1.0, EvaluationStatus.INCONCLUSIVE, [IssueCode.MISSING_EVIDENCE])
        if not data.outcomes:
            return self._result(True, 1.0, EvaluationStatus.PLAN_VALID, [])

        outcome_issue, retry = self._evaluate_outcomes(data, definitions)
        if outcome_issue == IssueCode.MISSING_EVIDENCE:
            return self._result(False, 1.0, EvaluationStatus.INCONCLUSIVE, [outcome_issue])
        if outcome_issue is not None:
            return self._result(
                False,
                1.0,
                EvaluationStatus.EXECUTION_FAILED,
                [outcome_issue],
                retry,
            )
        return self._result(True, 1.0, EvaluationStatus.SUCCEEDED, [])

    @staticmethod
    def _result(
        execution_valid: bool,
        coverage: float,
        status: EvaluationStatus,
        issues: list[IssueCode],
        retry: RetryRecommendation = RetryRecommendation.NONE,
        alternative: SafeAlternativeRecommendation | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            execution_valid=execution_valid,
            objective_coverage=coverage,
            evaluation_status=status,
            issues=issues,
            retry_recommendation=retry,
            alternative=alternative,
        )

    def _validate_summary(
        self,
        data: EvaluationInput,
        definitions: dict[str, ToolDefinition],
    ) -> IssueCode | None:
        if not data.steps:
            return IssueCode.POLICY_REJECTED
        if len(data.steps) > settings.MAX_PLAN_STEPS:
            return IssueCode.POLICY_REJECTED
        by_id = {step.step_id: step for step in data.steps}
        if len(by_id) != len(data.steps):
            return IssueCode.POLICY_REJECTED

        indegree = {step_id: 0 for step_id in by_id}
        dependents: dict[str, list[str]] = {step_id: [] for step_id in by_id}
        for step in data.steps:
            if len(set(step.dependencies)) != len(step.dependencies):
                return IssueCode.POLICY_REJECTED
            definition = definitions.get(step.tool_name)
            if definition is None:
                return IssueCode.POLICY_REJECTED
            if definition.execution_location == ExecutionLocation.BROWSER:
                return IssueCode.POLICY_REJECTED
            if definition.execution_location == ExecutionLocation.CLOUD and data.privacy_mode != "cloud_enabled":
                return IssueCode.POLICY_REJECTED
            if definition.permission_level == PermissionLevel.PRIVILEGED or definition.requires_approval:
                return IssueCode.POLICY_REJECTED
            for dependency in step.dependencies:
                if dependency not in by_id or dependency == step.step_id:
                    return IssueCode.POLICY_REJECTED
                indegree[step.step_id] += 1
                dependents[dependency].append(step.step_id)

        ready = deque(step_id for step_id, degree in indegree.items() if degree == 0)
        visited = 0
        while ready:
            step_id = ready.popleft()
            visited += 1
            for child in dependents[step_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        return None if visited == len(by_id) else IssueCode.DEPENDENCY_FAILED

    def _evaluate_outcomes(
        self,
        data: EvaluationInput,
        definitions: dict[str, ToolDefinition],
    ) -> tuple[IssueCode | None, RetryRecommendation]:
        planned = {step.step_id: step for step in data.steps}
        outcomes = {outcome.step_id: outcome for outcome in data.outcomes}
        if len(outcomes) != len(data.outcomes) or set(outcomes) != set(planned):
            return IssueCode.MISSING_EVIDENCE, RetryRecommendation.NONE

        failed_steps = [
            step for step_id, step in planned.items()
            if outcomes[step_id].status == StepStatus.FAILED
        ]
        if failed_steps:
            # RETRY is only useful as a bounded suggestion when one failed
            # retry-safe tool can be identified without exposing step data.
            can_retry = len(failed_steps) == 1 and definitions[failed_steps[0].tool_name].retry_safe
            return IssueCode.TOOL_FAILED, RetryRecommendation.RETRY if can_retry else RetryRecommendation.NONE

        for step_id, step in planned.items():
            evidence = outcomes[step_id]
            if evidence.status != StepStatus.COMPLETED:
                return IssueCode.MISSING_EVIDENCE, RetryRecommendation.NONE
            expected = _RESULT_KIND.get(step.tool_name)
            if expected is None or evidence.result_kind is None or evidence.result_count is None:
                return IssueCode.MISSING_EVIDENCE, RetryRecommendation.NONE
            if evidence.result_kind != expected:
                return IssueCode.OUTPUT_INVALID, RetryRecommendation.NONE
            if evidence.result_count == 0:
                return IssueCode.OUTPUT_EMPTY, RetryRecommendation.NONE
            if step.tool_name == "generate_chart" and evidence.artifact_count is None:
                return IssueCode.MISSING_EVIDENCE, RetryRecommendation.NONE
            if step.tool_name == "generate_chart" and evidence.artifact_count == 0:
                return IssueCode.OUTPUT_EMPTY, RetryRecommendation.NONE

        return None, RetryRecommendation.NONE

    @staticmethod
    def _canonical_alternative(
        intent: IntentCategory,
        definitions: dict[str, ToolDefinition],
        data: EvaluationInput,
    ) -> SafeAlternativeRecommendation | None:
        name = _CANONICAL_TOOL.get(intent)
        definition = definitions.get(name) if name else None
        existing = {step.tool_name for step in data.steps}
        expected_category = _INTENT_CATEGORY.get(intent)
        if (
            definition is None
            or name in existing
            or definition.category != expected_category
            or definition.execution_location == ExecutionLocation.BROWSER
            or definition.execution_location == ExecutionLocation.CLOUD and data.privacy_mode != "cloud_enabled"
            or definition.permission_level == PermissionLevel.PRIVILEGED
            or definition.requires_approval
        ):
            return None
        return SafeAlternativeRecommendation(
            tool_name=name,
            reason=AlternativeReason.INSUFFICIENT_COVERAGE,
            requires_plan_revalidation=True,
        )


intelligence_evaluator = IntelligenceEvaluator()
