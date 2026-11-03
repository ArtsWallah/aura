"""Process-local, integrity-checked handoff from planning to execution."""

from __future__ import annotations

import hashlib
import json
import asyncio
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Iterator, Literal, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.core.config import test_harness_enabled

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

if TYPE_CHECKING:
    from app.agent.state import ExecutionStep, TaskState


_EXECUTION_SCOPE_TOKEN = object()
_active_handoff_scope: ContextVar[tuple[object, "ExecutionHandoff", str] | None] = ContextVar(
    "aura_handoff_execution_scope", default=None
)
_active_step_scope: ContextVar[tuple[object, str, str, str] | None] = ContextVar(
    "aura_handoff_step_scope", default=None
)
_active_tool_invocation: ContextVar[tuple[object, str | None, str | None, str, str, str, object, list[bool]] | None] = ContextVar(
    "aura_tool_invocation_scope", default=None
)


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


def _intent_digest(
    confidence: float,
    is_multi_step: bool,
    required_tool_groups: list[list[str]],
) -> str:
    payload = json.dumps(
        {
            "confidence": confidence,
            "is_multi_step": is_multi_step,
            "required_tool_groups": required_tool_groups,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
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
    intent_confidence: float = Field(ge=0.0, le=1.0)
    intent_is_multi_step: bool = False
    required_tool_groups: list[list[str]] = Field(default_factory=list, max_length=16)
    evaluation_status: EvaluationStatus
    objective_coverage: float
    validation_registry_id: str
    plan_digest: str
    intent_digest: str

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
                intent_confidence=source.intent.confidence,
                intent_is_multi_step=source.intent.is_multi_step,
                required_tool_groups=[list(group) for group in source.intent.required_tool_groups],
                evaluation_status=evaluation.evaluation_status,
                objective_coverage=evaluation.objective_coverage,
                validation_registry_id=registry.registry_id,
                plan_digest=_plan_digest(plan),
                intent_digest=_intent_digest(
                    source.intent.confidence,
                    source.intent.is_multi_step,
                    [list(group) for group in source.intent.required_tool_groups],
                ),
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
                or _intent_digest(
                    self.intent_confidence,
                    self.intent_is_multi_step,
                    self.required_tool_groups,
                ) != self.intent_digest
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


@contextmanager
def handoff_execution_scope(
    handoff: ExecutionHandoff,
    *,
    task_id: str,
    privacy_mode: str,
    registry: ToolRegistry = tool_registry,
) -> Iterator[None]:
    """Authorize one task's executor calls from a freshly validated handoff."""
    if not isinstance(handoff, ExecutionHandoff) or not task_id:
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
    handoff.validate_integrity(privacy_mode=privacy_mode, registry=registry)
    token = _active_handoff_scope.set((_EXECUTION_SCOPE_TOKEN, handoff, task_id))
    try:
        yield
    finally:
        _active_handoff_scope.reset(token)


@contextmanager
def authorized_step_scope(step: "ExecutionStep", task_state: "TaskState") -> Iterator[None]:
    """Bind a registry invocation to a step in the active validated handoff."""
    scope = _active_handoff_scope.get()
    if scope is None and test_harness_enabled():
        # Legacy direct executor calls remain available only to the test harness.
        token = _active_step_scope.set((
            _EXECUTION_SCOPE_TOKEN, task_state.task_id, step.step_id, step.tool_name
        ))
        try:
            yield
        finally:
            _active_step_scope.reset(token)
        return
    if scope is None or scope[0] is not _EXECUTION_SCOPE_TOKEN:
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
    _, handoff, task_id = scope
    if task_state.task_id != task_id or task_state.plan is None:
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
    planned = next((item for item in handoff.plan.steps if item.step_id == step.step_id), None)
    current = next((item for item in task_state.plan.steps if item.step_id == step.step_id), None)
    if (
        planned is None
        or current is not step
        or planned.tool_name != step.tool_name
        or planned.parameters != step.parameters
        or planned.dependencies != step.dependencies
        or planned.max_retries != step.max_retries
    ):
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
    token = _active_step_scope.set((_EXECUTION_SCOPE_TOKEN, task_id, step.step_id, step.tool_name))
    try:
        yield
    finally:
        _active_step_scope.reset(token)


def require_authorized_tool_call(tool_name: str) -> None:
    """Reject low-level registry execution without a one-call capability."""
    invocation = _active_tool_invocation.get()
    if invocation is not None and invocation[0] is _EXECUTION_SCOPE_TOKEN and invocation[3] == tool_name:
        return
    raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)


@contextmanager
def authorized_tool_invocation(tool_name: str, arguments: dict) -> Iterator[None]:
    """Grant one exact argument payload to the already-authorized active step."""
    step_scope = _active_step_scope.get()
    if (
        step_scope is None
        or step_scope[0] is not _EXECUTION_SCOPE_TOKEN
        or step_scope[3] != tool_name
    ):
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
    digest = hashlib.sha256(
        json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    consumed = [False]
    token = _active_tool_invocation.set((
        _EXECUTION_SCOPE_TOKEN, step_scope[1], step_scope[2], tool_name, digest, "handoff", asyncio.current_task(), consumed
    ))
    try:
        yield
    finally:
        _active_tool_invocation.reset(token)


@contextmanager
def direct_read_only_tool_invocation(tool_name: str, arguments: dict) -> Iterator[None]:
    """Separate one-call capability for the fixed direct local read-only tools."""
    allowed = {
        "inspect_dataset", "descriptive_statistics", "group_comparison",
        "correlation_analysis", "detect_outliers", "read_document",
        "search_document_chunks", "system_health",
    }
    if tool_name not in allowed:
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
    digest = hashlib.sha256(
        json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    consumed = [False]
    token = _active_tool_invocation.set((
        _EXECUTION_SCOPE_TOKEN, None, None, tool_name, digest, "direct_read_only", asyncio.current_task(), consumed
    ))
    try:
        yield
    finally:
        _active_tool_invocation.reset(token)


@contextmanager
def test_legacy_tool_invocation(tool_name: str, arguments: dict) -> Iterator[None]:
    """Keep legacy registry calls available only to the isolated pytest process."""
    if not test_harness_enabled():
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
    digest = hashlib.sha256(
        json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    consumed = [False]
    token = _active_tool_invocation.set((
        _EXECUTION_SCOPE_TOKEN, None, None, tool_name, digest, "test_legacy", asyncio.current_task(), consumed
    ))
    try:
        yield
    finally:
        _active_tool_invocation.reset(token)


def require_authorized_tool_arguments(tool_name: str, arguments: dict, *, consume: bool = False) -> None:
    invocation = _active_tool_invocation.get()
    digest = hashlib.sha256(
        json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    if (
        invocation is None
        or invocation[0] is not _EXECUTION_SCOPE_TOKEN
        or invocation[3] != tool_name
        or invocation[4] != digest
        or invocation[6] is not asyncio.current_task()
        or invocation[7][0] is True
    ):
        raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
    if invocation[5] == "handoff":
        step_scope = _active_step_scope.get()
        if (
            step_scope is None
            or step_scope[0] is not _EXECUTION_SCOPE_TOKEN
            or step_scope[1] != invocation[1]
            or step_scope[2] != invocation[2]
            or step_scope[3] != invocation[3]
        ):
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
    if consume:
        invocation[7][0] = True
