"""Typed contracts for the intelligence layer.

These models are deliberately not wired into task execution yet. Trace models
are allowlisted schemas: they have no free-form objective, prompt, parameter,
file, dataset, or output fields, and reject unknown fields.
"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.state import ExecutionPlan, StepStatus
from app.tools.schemas import ToolCategory


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IntentCategory(str, Enum):
    DATA_ANALYSIS = "data_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    VISUALIZATION = "visualization"
    SQL_QUERY = "sql_query"
    DOCUMENT_ANALYSIS = "document_analysis"
    WEB_RESEARCH = "web_research"
    REPORT_GENERATION = "report_generation"
    MULTI_STEP_ANALYSIS = "multi_step_analysis"
    SYSTEM_HEALTH = "system_health"
    UNKNOWN = "unknown"


class RequiredInputKind(str, Enum):
    DATASET = "dataset"
    DOCUMENT = "document"
    DATABASE = "database"
    RESEARCH_TOPIC = "research_topic"
    ANALYSIS_GOAL = "analysis_goal"
    CLARIFICATION = "clarification"


class InputRequirement(StrictContract):
    kind: RequiredInputKind
    required: bool = True


_ToolId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
_ToolGroup = Annotated[list[_ToolId], Field(min_length=1, max_length=16)]


class IntentAssessment(StrictContract):
    category: IntentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    required_inputs: list[InputRequirement] = Field(default_factory=list, max_length=32)
    missing_inputs: list[InputRequirement] = Field(default_factory=list, max_length=32)
    # Each inner group is an OR of tools for one assessed capability;
    # separate groups are all required. Empty defaults preserve old payloads.
    required_tool_groups: list[_ToolGroup] = Field(default_factory=list, max_length=16)
    is_multi_step: bool

    @model_validator(mode="after")
    def validate_required_tool_groups(self) -> "IntentAssessment":
        if any(len(group) != len(set(group)) for group in self.required_tool_groups):
            raise ValueError("Required tool groups cannot contain duplicate tools.")
        if len({tuple(group) for group in self.required_tool_groups}) != len(self.required_tool_groups):
            raise ValueError("Required tool groups cannot be duplicated.")
        return self


class PlanningResponseType(str, Enum):
    PLAN = "PLAN"
    CLARIFICATION = "CLARIFICATION"


class ProviderIdentity(str, Enum):
    GEMINI = "gemini"
    LOCAL_DETERMINISTIC_PLANNER = "local_deterministic_planner"
    TEST_MOCK_PROVIDER = "test_mock_provider"


class PlanningResponse(StrictContract):
    response_type: PlanningResponseType
    provider_identity: ProviderIdentity = ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
    plan: ExecutionPlan | None = None
    clarification_question: str | None = Field(default=None, min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_variant(self) -> "PlanningResponse":
        if self.response_type == PlanningResponseType.PLAN:
            if self.plan is None or self.clarification_question is not None:
                raise ValueError("PLAN responses require a plan and cannot include a clarification question.")
        elif self.plan is not None or self.clarification_question is None:
            raise ValueError("CLARIFICATION responses require a question and cannot include a plan.")
        return self


class RetryRecommendation(str, Enum):
    NONE = "none"
    RETRY = "retry"
    REPLAN = "replan"


class IssueCode(str, Enum):
    TOOL_FAILED = "tool_failed"
    OUTPUT_INVALID = "output_invalid"
    OUTPUT_EMPTY = "output_empty"
    OBJECTIVE_UNCOVERED = "objective_uncovered"
    DEPENDENCY_FAILED = "dependency_failed"
    POLICY_REJECTED = "policy_rejected"
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_INPUT = "missing_input"
    UNKNOWN = "unknown"


class StepFailureCategory(str, Enum):
    NONE = "none"
    TOOL_FAILURE = "tool_failure"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class AlternativeReason(str, Enum):
    TOOL_UNAVAILABLE = "tool_unavailable"
    OUTPUT_INVALID = "output_invalid"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"


class EvaluationStatus(str, Enum):
    PLAN_VALID = "plan_valid"
    INCOMPLETE_COVERAGE = "incomplete_coverage"
    INVALID_PLAN = "invalid_plan"
    EXECUTION_FAILED = "execution_failed"
    PARTIAL_SUCCESS = "partial_success"
    SUCCEEDED = "succeeded"
    INCONCLUSIVE = "inconclusive"


class SafeAlternativeRecommendation(StrictContract):
    tool_name: _ToolId
    reason: AlternativeReason
    requires_plan_revalidation: bool = True


class EvaluationResult(StrictContract):
    execution_valid: bool
    objective_coverage: float = Field(ge=0.0, le=1.0)
    evaluation_status: EvaluationStatus | None = None
    issues: list[IssueCode] = Field(default_factory=list, max_length=32)
    retry_recommendation: RetryRecommendation = RetryRecommendation.NONE
    alternative: SafeAlternativeRecommendation | None = None

    @model_validator(mode="after")
    def constrain_recommendations(self) -> "EvaluationResult":
        if self.evaluation_status is None:
            # Preserve Phase 3.1.1 semantics for callers that have not adopted
            # the additive status field yet.
            if self.execution_valid and self.issues:
                raise ValueError("A valid execution cannot contain unresolved issues.")
        else:
            if self.execution_valid and self.issues and not (
                self.evaluation_status == EvaluationStatus.INCOMPLETE_COVERAGE
                and set(self.issues) == {IssueCode.OBJECTIVE_UNCOVERED}
            ):
                raise ValueError("A valid plan or execution cannot contain unrelated unresolved issues.")
            if self.evaluation_status in {
                EvaluationStatus.INVALID_PLAN,
                EvaluationStatus.EXECUTION_FAILED,
                EvaluationStatus.PARTIAL_SUCCESS,
                EvaluationStatus.INCONCLUSIVE,
            } and self.execution_valid:
                raise ValueError("Invalid, failed, or inconclusive evaluations cannot be marked valid.")
            if self.evaluation_status == EvaluationStatus.PARTIAL_SUCCESS and not self.issues:
                raise ValueError("Partial execution success must identify an unresolved issue.")
            if self.evaluation_status == EvaluationStatus.PARTIAL_SUCCESS and self.objective_coverage <= 0.0:
                raise ValueError("Partial execution success requires evidence of useful objective coverage.")
            if self.evaluation_status == EvaluationStatus.SUCCEEDED and (
                not self.execution_valid or self.objective_coverage != 1.0 or self.issues
            ):
                raise ValueError("Successful evaluations require valid execution and complete coverage.")
            if self.evaluation_status == EvaluationStatus.INCOMPLETE_COVERAGE and (
                not self.execution_valid or self.objective_coverage >= 1.0
            ):
                raise ValueError("Incomplete coverage requires a valid plan or execution with coverage below 1.")
            if self.evaluation_status == EvaluationStatus.PLAN_VALID and (
                not self.execution_valid or self.objective_coverage != 1.0 or self.issues
            ):
                raise ValueError("A valid plan requires structural validity and complete capability coverage.")
            if self.evaluation_status in {EvaluationStatus.INVALID_PLAN, EvaluationStatus.EXECUTION_FAILED} and not self.issues:
                raise ValueError("Invalid plans and failed executions must identify at least one issue.")
            if self.evaluation_status == EvaluationStatus.INCONCLUSIVE and not self.issues:
                raise ValueError("Inconclusive evaluations must identify why the evidence was insufficient.")
            if (
                self.evaluation_status == EvaluationStatus.INCOMPLETE_COVERAGE
                and self.issues
                and set(self.issues) != {IssueCode.OBJECTIVE_UNCOVERED}
            ):
                raise ValueError("Incomplete coverage may only report the uncovered-objective issue.")
        if self.retry_recommendation == RetryRecommendation.REPLAN and self.alternative is None:
            raise ValueError("REPLAN requires a safe alternative recommendation.")
        if self.retry_recommendation != RetryRecommendation.REPLAN and self.alternative is not None:
            raise ValueError("Alternative metadata is only valid with REPLAN.")
        return self


class ValidationOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class TraceArtifactType(str, Enum):
    CHART_PNG = "chart_png"
    REPORT_HTML = "report_html"
    REPORT_MARKDOWN = "report_markdown"


class TraceArtifact(StrictContract):
    artifact_type: TraceArtifactType
    available: bool
    count: int = Field(ge=0, le=10_000)


class TraceStep(StrictContract):
    step_id: Annotated[str, Field(pattern=r"^step_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
    tool_name: _ToolId
    status: StepStatus
    tool_label: str | None = Field(default=None, min_length=1, max_length=100)
    tool_category: ToolCategory | None = None
    retry_count: int = Field(default=0, ge=0, le=10_000)
    retry_limit: int = Field(default=0, ge=0, le=10_000)
    failure_category: StepFailureCategory | None = None
    artifacts: list[TraceArtifact] = Field(default_factory=list, max_length=4)
    duration_ms: float = Field(default=0.0, ge=0.0, le=86_400_000)


class ExecutionTiming(StrictContract):
    planning_ms: float = Field(default=0.0, ge=0.0, le=86_400_000)
    validation_ms: float = Field(default=0.0, ge=0.0, le=86_400_000)
    execution_ms: float = Field(default=0.0, ge=0.0, le=86_400_000)


class ExecutionTrace(StrictContract):
    intent_category: IntentCategory
    intent_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provider_identity: ProviderIdentity
    selected_tools: list[_ToolId] = Field(default_factory=list, max_length=128)
    step_statuses: list[TraceStep] = Field(default_factory=list, max_length=128)
    validation_outcome: ValidationOutcome
    evaluation_status: EvaluationStatus | None = None
    timing: ExecutionTiming = Field(default_factory=ExecutionTiming)

    @model_validator(mode="after")
    def selected_tools_match_steps(self) -> "ExecutionTrace":
        expected = list(dict.fromkeys(step.tool_name for step in self.step_statuses))
        if self.selected_tools != expected:
            raise ValueError("Selected tools must match the ordered trace steps.")
        return self
