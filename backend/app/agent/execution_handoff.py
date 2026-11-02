"""Process-local, integrity-checked handoff from planning to execution."""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.agent.intelligence_contracts import (
    EvaluationStatus,
    IntentCategory,
    ProviderIdentity,
    ValidationOutcome,
)
from app.agent.intelligence_pipeline import (
    IntelligencePipelineResult,
    PipelineDisposition,
)
from app.agent.plan_validation import PlanValidationError, validate_execution_plan
from app.agent.state import ExecutionPlan
from app.tools.registry import ToolRegistry, tool_registry


class HandoffFailureCode(str, Enum):
    INVALID_SOURCE = "invalid_source"
    INVALID_HANDOFF = "invalid_handoff"
    PLAN_TAMPERED = "plan_tampered"
    REGISTRY_MISMATCH = "registry_mismatch"
    REGISTRY_UNAVAILABLE = "registry_unavailable"
    PRIVACY_MISMATCH = "privacy_mismatch"
    PLAN_REVALIDATION_FAILED = "plan_revalidation_failed"


class ExecutionHandoffError(ValueError):
    """A safe handoff failure containing only a stable code."""

    def __init__(self, code: HandoffFailureCode):
        self.code = code
        super().__init__(code.value)


def _plan_digest(plan: ExecutionPlan) -> str:
    payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_HANDOFF_FACTORY_TOKEN = object()


class ExecutionHandoff(BaseModel):
    """Validated plan plus minimal safe provenance; never accepts client payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Explicit annotations keep Pydantic's strict schema finite and allowlisted.
    handoff_id: str
    plan: ExecutionPlan
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"]
    provider_identity: ProviderIdentity
    intent_category: IntentCategory
    evaluation_status: EvaluationStatus
    objective_coverage: float
    validation_registry_id: str
    plan_digest: str

    def __init__(self, **data):  # type: ignore[no-untyped-def]
        if data.get("_factory_token") is not _HANDOFF_FACTORY_TOKEN:
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
        super().__init__(**data)

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not isinstance(obj, dict) or obj.get("_factory_token") is not _HANDOFF_FACTORY_TOKEN:
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
        return super().model_validate(obj, *args, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data, *args, **kwargs):  # type: ignore[no-untyped-def]
        # A process-local object token cannot be represented in JSON.
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)

    @classmethod
    def model_validate_strings(cls, obj, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)

    @model_validator(mode="before")
    @classmethod
    def require_factory_token(cls, data):  # type: ignore[no-untyped-def]
        # model_validate() is also gated, so a client-shaped mapping cannot
        # become executable merely by bypassing the convenience constructor.
        if not isinstance(data, dict) or data.pop("_factory_token", None) is not _HANDOFF_FACTORY_TOKEN:
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
        return data

    @classmethod
    def from_pipeline_result(
        cls,
        result: IntelligencePipelineResult,
        *,
        privacy_mode: str,
        registry: ToolRegistry = tool_registry,
    ) -> "ExecutionHandoff":
        if not isinstance(result, IntelligencePipelineResult):
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_SOURCE)
        if registry is not tool_registry:
            raise ExecutionHandoffError(HandoffFailureCode.REGISTRY_MISMATCH)
        if privacy_mode not in {"local_only", "hybrid", "cloud_enabled"}:
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_SOURCE)
        try:
            source = result.model_copy(deep=True)
            planning = source.planning_response
            evaluation = source.evaluation
            if (
                source.disposition != PipelineDisposition.READY
                or planning is None
                or planning.plan is None
                or source.provider_identity is None
                or planning.provider_identity != source.provider_identity
                or source.validation_outcome != ValidationOutcome.ACCEPTED
                or evaluation is None
                or evaluation.evaluation_status != EvaluationStatus.PLAN_VALID
                or not evaluation.execution_valid
                or evaluation.objective_coverage != 1.0
                or evaluation.issues
                or source.failure_code is not None
                or source.validation_registry_id != registry.registry_id
                or source.privacy_mode != privacy_mode
                or source.intent.category == IntentCategory.UNKNOWN
            ):
                raise ExecutionHandoffError(HandoffFailureCode.INVALID_SOURCE)
            plan = planning.plan.model_copy(deep=True)
            try:
                validate_execution_plan(plan, privacy_mode=privacy_mode, registry=registry)
            except PlanValidationError:
                raise ExecutionHandoffError(HandoffFailureCode.PLAN_REVALIDATION_FAILED) from None
            except Exception:
                raise ExecutionHandoffError(HandoffFailureCode.REGISTRY_UNAVAILABLE) from None
            return cls(
                _factory_token=_HANDOFF_FACTORY_TOKEN,
                handoff_id=f"handoff_{uuid.uuid4().hex}",
                plan=plan,
                privacy_mode=privacy_mode,
                provider_identity=source.provider_identity,
                intent_category=source.intent.category,
                evaluation_status=evaluation.evaluation_status,
                objective_coverage=evaluation.objective_coverage,
                validation_registry_id=registry.registry_id,
                plan_digest=_plan_digest(plan),
            )
        except ExecutionHandoffError:
            raise
        except Exception:
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_SOURCE) from None

    def validate_integrity(self, *, privacy_mode: str, registry: ToolRegistry) -> ExecutionPlan:
        try:
            if (
                not isinstance(self.plan, ExecutionPlan)
                or self.evaluation_status != EvaluationStatus.PLAN_VALID
                or self.objective_coverage != 1.0
                or self.intent_category == IntentCategory.UNKNOWN
                or not isinstance(self.provider_identity, ProviderIdentity)
            ):
                raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
            if privacy_mode != self.privacy_mode:
                raise ExecutionHandoffError(HandoffFailureCode.PRIVACY_MISMATCH)
            if registry is not tool_registry or registry.registry_id != self.validation_registry_id:
                raise ExecutionHandoffError(HandoffFailureCode.REGISTRY_MISMATCH)
            if _plan_digest(self.plan) != self.plan_digest:
                raise ExecutionHandoffError(HandoffFailureCode.PLAN_TAMPERED)
            plan = self.plan.model_copy(deep=True)
            validate_execution_plan(plan, privacy_mode=privacy_mode, registry=registry)
            return plan
        except ExecutionHandoffError:
            raise
        except PlanValidationError:
            raise ExecutionHandoffError(HandoffFailureCode.PLAN_REVALIDATION_FAILED) from None
        except Exception:
            raise ExecutionHandoffError(HandoffFailureCode.REGISTRY_UNAVAILABLE) from None
