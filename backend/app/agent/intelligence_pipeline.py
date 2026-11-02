"""Independent, non-executing intent-to-evaluated-plan pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from app.agent.dynamic_planner import DynamicPlanner, PlanningInputMetadata
from app.agent.intelligence_contracts import (
    EvaluationResult,
    EvaluationStatus,
    IntentAssessment,
    IntentCategory,
    PlanningResponse,
    PlanningResponseType,
    ProviderIdentity,
    RequiredInputKind,
    StrictContract,
    ValidationOutcome,
)
from app.agent.intelligence_evaluator import (
    EvaluationInput,
    IntelligenceEvaluator,
    PlanStepSummary,
    PlanValidationEvidence,
)
from app.agent.intent import assess_intent
from app.agent.plan_validation import PlanValidationError, validate_execution_plan
from app.core.config import settings
from app.core.security import managed_file_suffix, validate_safe_path
from app.llm.base import BaseLLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.mock_provider import TestReasoningProvider
from app.llm.router import LLMRouter, ProviderSelectionError, llm_router
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.schemas import ToolDefinition

import app.tools  # noqa: F401,E402  Register built-in tools locally.


class PipelineDisposition(str, Enum):
    READY = "READY"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    REJECTED = "REJECTED"


class PipelineFailureCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_INTENT = "unsupported_intent"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_FAILURE = "provider_failure"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    INVALID_PLAN = "invalid_plan"
    INCOMPLETE_EVALUATION = "incomplete_evaluation"
    INTERNAL_ERROR = "internal_error"


class IntelligencePipelineResult(StrictContract):
    disposition: PipelineDisposition
    intent: IntentAssessment
    provider_identity: ProviderIdentity | None = None
    planning_response: PlanningResponse | None = None
    validation_outcome: ValidationOutcome | None = None
    evaluation: EvaluationResult | None = None
    failure_code: PipelineFailureCode | None = None
    validation_registry_id: str | None = None
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] | None = None

    @model_validator(mode="after")
    def validate_disposition(self) -> "IntelligencePipelineResult":
        if self.disposition == PipelineDisposition.READY:
            if (
                self.failure_code is not None
                or self.planning_response is None
                or self.planning_response.response_type != PlanningResponseType.PLAN
                or self.validation_outcome != ValidationOutcome.ACCEPTED
                or self.evaluation is None
                or self.evaluation.evaluation_status != EvaluationStatus.PLAN_VALID
                or not self.evaluation.execution_valid
                or self.provider_identity != self.planning_response.provider_identity
                or not self.validation_registry_id
                or self.privacy_mode not in {"local_only", "hybrid", "cloud_enabled"}
            ):
                raise ValueError("READY requires a provider-identified plan that passed validation and evaluation.")
        elif self.disposition == PipelineDisposition.CLARIFICATION_REQUIRED:
            if (
                self.planning_response is None
                or self.planning_response.response_type != PlanningResponseType.CLARIFICATION
                or self.failure_code is not None
            ):
                raise ValueError("Clarification disposition requires a clarification response.")
        elif self.failure_code is None:
            raise ValueError("Rejected results require a safe failure code.")
        return self


class IntelligencePipeline:
    """Assess, plan, validate, and evaluate a request without executing tools."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        router: LLMRouter | None = None,
        evaluator: IntelligenceEvaluator | None = None,
        provider: BaseLLMProvider | None = None,
        provider_identity: ProviderIdentity | str | None = None,
    ):
        if provider is not None and provider_identity is None:
            raise ValueError("Injected providers require an explicit identity.")
        if provider is None and provider_identity is not None:
            raise ValueError("A provider identity requires an injected provider.")
        self.registry = registry or tool_registry
        self.router = router or llm_router
        self.evaluator = evaluator or IntelligenceEvaluator(self.registry)
        self.provider = provider
        self.provider_identity = ProviderIdentity(provider_identity) if provider_identity is not None else None

    async def run(
        self,
        objective: str,
        available_inputs: PlanningInputMetadata | dict[str, Any] | None = None,
        privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] | str | None = None,
        requested_provider: str | None = None,
    ) -> IntelligencePipelineResult:
        """Return READY, clarification, or safe rejection; never calls a tool."""
        mode = settings.DEFAULT_PRIVACY_MODE if privacy_mode is None else privacy_mode
        try:
            definitions = self.registry.list_tools()
        except Exception:
            return self._rejected(
                IntentAssessment(category=IntentCategory.UNKNOWN, confidence=0.0, is_multi_step=False),
                PipelineFailureCode.INTERNAL_ERROR,
            )

        try:
            metadata = PlanningInputMetadata.model_validate(available_inputs or {})
        except (ValidationError, TypeError, ValueError):
            assessment = assess_intent(objective, available_tools=definitions)
            return self._rejected(assessment, PipelineFailureCode.INVALID_INPUT)

        available_kinds = set(metadata.available_inputs)
        path_kinds = self._authorized_input_kinds(metadata)
        available_kinds.update(path_kinds)
        try:
            assessment = assess_intent(
                objective,
                available_tools=definitions,
                available_inputs=available_kinds,
            )
        except Exception:
            assessment = IntentAssessment(
                category=IntentCategory.UNKNOWN,
                confidence=0.0,
                is_multi_step=False,
            )
            return self._rejected(assessment, PipelineFailureCode.INTERNAL_ERROR)

        if mode not in {"local_only", "hybrid", "cloud_enabled"}:
            return self._rejected(assessment, PipelineFailureCode.INVALID_INPUT)
        if not isinstance(objective, str) or not objective.strip():
            return self._clarification(assessment, "What would you like AURA to plan?")

        if assessment.category == IntentCategory.UNKNOWN or assessment.confidence < 0.5:
            if any(item.kind == RequiredInputKind.CLARIFICATION for item in assessment.missing_inputs):
                return self._clarification(assessment, "Could you clarify the task and the result you want?")
            return self._rejected(assessment, PipelineFailureCode.UNSUPPORTED_INTENT)

        if any(item.required for item in assessment.missing_inputs):
            missing = next(item.kind for item in assessment.missing_inputs if item.required)
            return self._clarification(assessment, self._missing_input_question(missing))

        provider: BaseLLMProvider | None = None
        identity: ProviderIdentity | None = None
        if mode == "cloud_enabled":
            try:
                provider, identity = self._select_provider(requested_provider)
            except ProviderSelectionError:
                return self._rejected(assessment, PipelineFailureCode.PROVIDER_UNAVAILABLE)
            except Exception:
                return self._rejected(assessment, PipelineFailureCode.PROVIDER_FAILURE)
            if identity == ProviderIdentity.TEST_MOCK_PROVIDER and settings.ENVIRONMENT != "test":
                return self._rejected(assessment, PipelineFailureCode.PROVIDER_UNAVAILABLE)

        try:
            planner = DynamicPlanner(
                provider=provider,
                provider_identity=identity,
                registry=self.registry,
                available_tools=definitions,
            )
            planning = await planner.create_plan(
                objective,
                assessment,
                metadata,
                privacy_mode=mode,
            )
            if planner.last_failure_code:
                failure = self._planner_failure(planner.last_failure_code)
                return self._rejected(assessment, failure)
            if planning.response_type == PlanningResponseType.CLARIFICATION:
                return IntelligencePipelineResult(
                    disposition=PipelineDisposition.CLARIFICATION_REQUIRED,
                    intent=assessment,
                    provider_identity=planning.provider_identity,
                    planning_response=planning,
                )
            if planning.plan is None:
                return self._rejected(assessment, PipelineFailureCode.INVALID_PLAN)

            # The legacy ExecutionPlan.goal is free-form and planners commonly
            # copy the full user objective into it. Keep returned READY plans
            # and their later handoffs limited to an allowlisted intent label.
            planning.plan.goal = assessment.category.value

            for step in planning.plan.steps:
                definition = self.registry.get_tool_definition(step.tool_name)
                if definition is None:
                    return self._rejected(assessment, PipelineFailureCode.INVALID_PLAN)
                # Keep non-executable display text registry-derived. Provider
                # prose and user objectives do not belong in execution plans.
                step.title = step.tool_name.replace("_", " ").title()
                step.description = definition.description
                step.capability = definition.category.value
            validate_execution_plan(planning.plan, privacy_mode=mode, registry=self.registry)
            validation_outcome = ValidationOutcome.ACCEPTED
            evaluation = self.evaluator.evaluate(EvaluationInput(
                intent=assessment,
                steps=[PlanStepSummary(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    dependencies=step.dependencies,
                ) for step in planning.plan.steps],
                validation=PlanValidationEvidence(outcome=validation_outcome),
                evaluation_target="plan",
                privacy_mode=mode,
            ))
        except (PlanValidationError, ValidationError, ValueError, KeyError):
            return self._rejected(assessment, PipelineFailureCode.INVALID_PLAN, ValidationOutcome.REJECTED)
        except Exception:
            return self._rejected(assessment, PipelineFailureCode.INTERNAL_ERROR)

        if evaluation.evaluation_status == EvaluationStatus.PLAN_VALID and evaluation.execution_valid:
            return IntelligencePipelineResult(
                disposition=PipelineDisposition.READY,
                intent=assessment,
                provider_identity=planning.provider_identity,
                planning_response=planning,
                validation_outcome=validation_outcome,
                evaluation=evaluation,
                validation_registry_id=self.registry.registry_id,
                privacy_mode=mode,
            )
        if evaluation.evaluation_status == EvaluationStatus.INCOMPLETE_COVERAGE:
            return self._clarification(
                assessment,
                "Could you clarify which supported operations should be included in the plan?",
                validation_outcome=validation_outcome,
                evaluation=evaluation,
            )
        return self._rejected(
            assessment,
            PipelineFailureCode.INCOMPLETE_EVALUATION,
            validation_outcome,
            evaluation,
        )

    def _select_provider(self, requested_provider: str | None) -> tuple[BaseLLMProvider, ProviderIdentity]:
        if self.provider is not None:
            assert self.provider_identity is not None
            return self.provider, self.provider_identity
        provider = self.router.get_provider(requested_provider=requested_provider, privacy_mode="cloud_enabled")
        if isinstance(provider, GeminiProvider):
            identity = ProviderIdentity.GEMINI
        elif isinstance(provider, TestReasoningProvider):
            identity = ProviderIdentity.TEST_MOCK_PROVIDER
        else:
            raise ProviderSelectionError("unknown", "provider_unavailable")
        return provider, identity

    @staticmethod
    def _authorized_input_kinds(metadata: PlanningInputMetadata) -> set[RequiredInputKind]:
        if metadata.authorized_file_path is None:
            return set()
        try:
            path = validate_safe_path(metadata.authorized_file_path)
            suffix = managed_file_suffix(path)
        except (OSError, PermissionError, ValueError):
            return set()
        if suffix in {".csv", ".xlsx", ".xls"}:
            return {RequiredInputKind.DATASET}
        if suffix in {".txt", ".md", ".json"}:
            return {RequiredInputKind.DOCUMENT}
        return set()

    @staticmethod
    def _planner_failure(code: str) -> PipelineFailureCode:
        return {
            "invalid_plan": PipelineFailureCode.INVALID_PLAN,
            "provider_unavailable": PipelineFailureCode.PROVIDER_UNAVAILABLE,
            "provider_timeout": PipelineFailureCode.PROVIDER_TIMEOUT,
            "provider_failure": PipelineFailureCode.PROVIDER_FAILURE,
            "invalid_provider_response": PipelineFailureCode.INVALID_PROVIDER_RESPONSE,
        }.get(code, PipelineFailureCode.INTERNAL_ERROR)

    @staticmethod
    def _missing_input_question(kind: RequiredInputKind) -> str:
        return {
            RequiredInputKind.DATASET: "Please select an authorized dataset upload.",
            RequiredInputKind.DOCUMENT: "Please select an authorized document upload.",
            RequiredInputKind.DATABASE: "Please provide an authorized database connection.",
            RequiredInputKind.RESEARCH_TOPIC: "What specific topic should be researched?",
            RequiredInputKind.ANALYSIS_GOAL: "What result should the analysis provide?",
            RequiredInputKind.CLARIFICATION: "Could you clarify the task and the result you want?",
        }.get(kind, "Could you provide the missing information?")

    @staticmethod
    def _clarification(
        intent: IntentAssessment,
        question: str,
        validation_outcome: ValidationOutcome | None = None,
        evaluation: EvaluationResult | None = None,
    ) -> IntelligencePipelineResult:
        response = PlanningResponse(
            response_type=PlanningResponseType.CLARIFICATION,
            provider_identity=ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER,
            clarification_question=question,
            explanation="Additional information is required before a validated plan can be returned.",
        )
        return IntelligencePipelineResult(
            disposition=PipelineDisposition.CLARIFICATION_REQUIRED,
            intent=intent,
            provider_identity=ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER,
            planning_response=response,
            validation_outcome=validation_outcome,
            evaluation=evaluation,
        )

    @staticmethod
    def _rejected(
        intent: IntentAssessment,
        code: PipelineFailureCode,
        validation: ValidationOutcome | None = None,
        evaluation: EvaluationResult | None = None,
    ) -> IntelligencePipelineResult:
        return IntelligencePipelineResult(
            disposition=PipelineDisposition.REJECTED,
            intent=intent,
            validation_outcome=validation,
            evaluation=evaluation,
            failure_code=code,
        )


intelligence_pipeline = IntelligencePipeline()
