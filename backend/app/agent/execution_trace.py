"""Registry-validated projections for pipeline execution traces."""

from __future__ import annotations

import json
from typing import Any

from app.agent.intelligence_contracts import (
    EvaluationStatus,
    ExecutionTiming,
    ExecutionTrace,
    IntentCategory,
    ProviderIdentity,
    StepFailureCategory,
    TraceArtifact,
    TraceArtifactType,
    TraceStep,
    ValidationOutcome,
)
from app.agent.state import ExecutionStep, StepStatus, TaskState, TaskStatus
from app.tools.registry import ToolRegistry, tool_registry


MAX_TRACE_BYTES = 32_768


def _artifact_summaries(tool_name: str, artifacts: Any) -> list[TraceArtifact]:
    if not isinstance(artifacts, list):
        return []
    paths = [path.lower() for path in artifacts if isinstance(path, str)]
    if tool_name == "generate_chart":
        count = sum(path.endswith(".png.enc") for path in paths)
        return [TraceArtifact(artifact_type=TraceArtifactType.CHART_PNG, available=count > 0, count=min(count, 10_000))]
    if tool_name == "generate_report":
        markdown = sum(path.endswith(".md.enc") for path in paths)
        html = sum(path.endswith(".html.enc") for path in paths)
        return [
            TraceArtifact(artifact_type=TraceArtifactType.REPORT_MARKDOWN, available=markdown > 0, count=min(markdown, 10_000)),
            TraceArtifact(artifact_type=TraceArtifactType.REPORT_HTML, available=html > 0, count=min(html, 10_000)),
        ]
    return []


def _failure_category(step: ExecutionStep, task: TaskState, status: StepStatus) -> StepFailureCategory | None:
    if status == StepStatus.FAILED:
        return StepFailureCategory.TOOL_FAILURE
    if status == StepStatus.INTERRUPTED or status in {StepStatus.PENDING, StepStatus.IN_PROGRESS} and task.recovery_required:
        return StepFailureCategory.INTERRUPTED
    if status == StepStatus.SKIPPED:
        if task.status == TaskStatus.CANCELLED:
            return StepFailureCategory.CANCELLED
        if task.plan and any(
            next((dependency.status for dependency in task.plan.steps if dependency.step_id == dependency_id), None)
            != StepStatus.COMPLETED
            for dependency_id in step.dependencies
        ):
            return StepFailureCategory.DEPENDENCY_BLOCKED
        return StepFailureCategory.UNKNOWN
    return None


def _trace_step(
    step: ExecutionStep,
    task: TaskState,
    registry: ToolRegistry,
    *,
    status: StepStatus | None = None,
    retry_count: int | None = None,
) -> TraceStep | None:
    definition = registry.get_tool_definition(step.tool_name)
    if definition is None:
        return None
    result = task.step_results.get(step.step_id)
    artifacts = result.get("artifacts") if isinstance(result, dict) else None
    trace_status = status or step.status
    if task.status in {TaskStatus.CANCELLED, TaskStatus.CANCEL_REQUESTED} and trace_status == StepStatus.PENDING:
        trace_status = StepStatus.SKIPPED
    return TraceStep(
        step_id=step.step_id,
        tool_name=definition.name,
        tool_label=definition.name.replace("_", " ").title(),
        tool_category=definition.category,
        status=trace_status,
        retry_count=min(step.retry_count if retry_count is None else retry_count, 10_000),
        retry_limit=min(step.max_retries, 10_000),
        failure_category=_failure_category(step, task, trace_status),
        artifacts=_artifact_summaries(definition.name, artifacts) if trace_status == StepStatus.COMPLETED else [],
        duration_ms=max(0.0, min(step.duration_ms, 86_400_000)),
    )


def build_handoff_trace(
    handoff: Any,
    task: TaskState,
    registry: ToolRegistry = tool_registry,
) -> ExecutionTrace | None:
    """Project handoff provenance and current step states without free-form data."""
    if task.plan is None:
        return None
    try:
        steps = [_trace_step(step, task, registry) for step in task.plan.steps]
        if any(step is None for step in steps):
            return None
        trace = ExecutionTrace(
            intent_category=IntentCategory(handoff.intent_category),
            intent_confidence=handoff.intent_confidence,
            provider_identity=ProviderIdentity(handoff.provider_identity),
            selected_tools=list(dict.fromkeys(step.tool_name for step in steps if step is not None)),
            step_statuses=[step for step in steps if step is not None],
            validation_outcome=ValidationOutcome.ACCEPTED,
            evaluation_status=(
                task.execution_evaluation.evaluation_status
                if task.execution_evaluation is not None
                else None
            ),
            timing=task.execution_trace.timing if isinstance(task.execution_trace, ExecutionTrace) else ExecutionTiming(),
        )
        return sanitize_execution_trace(trace, registry)
    except Exception:
        return None


def update_trace_step(
    trace: ExecutionTrace,
    step: ExecutionStep,
    task: TaskState,
    event_type: str,
    retry_count: int | None = None,
    registry: ToolRegistry = tool_registry,
) -> ExecutionTrace | None:
    event_status = {
        "step_started": StepStatus.IN_PROGRESS,
        "step_retry": StepStatus.IN_PROGRESS,
        "step_completed": StepStatus.COMPLETED,
        "step_failed": StepStatus.FAILED,
    }.get(event_type)
    if event_status is None:
        return trace
    replacement = _trace_step(step, task, registry, status=event_status, retry_count=retry_count)
    if replacement is None:
        return None
    if event_type == "step_retry":
        replacement = replacement.model_copy(update={"failure_category": StepFailureCategory.TOOL_FAILURE})
    steps = [replacement if item.step_id == replacement.step_id else item for item in trace.step_statuses]
    updated = trace.model_copy(update={"step_statuses": steps})
    return sanitize_execution_trace(updated, registry)


def sanitize_execution_trace(
    value: ExecutionTrace | dict[str, Any],
    registry: ToolRegistry = tool_registry,
) -> ExecutionTrace | None:
    """Rebuild an allowlisted trace with labels and categories from the live registry."""
    try:
        source = value if isinstance(value, ExecutionTrace) else ExecutionTrace.model_validate(value)
        steps: list[TraceStep] = []
        for item in source.step_statuses:
            definition = registry.get_tool_definition(item.tool_name)
            if definition is None:
                continue
            allowed_artifacts = {"generate_chart": {TraceArtifactType.CHART_PNG}}
            report_artifacts = {TraceArtifactType.REPORT_HTML, TraceArtifactType.REPORT_MARKDOWN}
            artifact_types = allowed_artifacts.get(definition.name, report_artifacts if definition.name == "generate_report" else set())
            artifacts = [artifact for artifact in item.artifacts if artifact.artifact_type in artifact_types]
            steps.append(TraceStep(
                step_id=item.step_id,
                tool_name=definition.name,
                tool_label=definition.name.replace("_", " ").title(),
                tool_category=definition.category,
                status=StepStatus(item.status),
                retry_count=min(item.retry_count, 10_000),
                retry_limit=min(item.retry_limit, 10_000),
                failure_category=StepFailureCategory(item.failure_category) if item.failure_category else None,
                artifacts=artifacts,
                duration_ms=max(0.0, min(item.duration_ms, 86_400_000)),
            ))
        trace = ExecutionTrace(
            intent_category=IntentCategory(source.intent_category),
            intent_confidence=source.intent_confidence,
            provider_identity=ProviderIdentity(source.provider_identity),
            selected_tools=list(dict.fromkeys(step.tool_name for step in steps)),
            step_statuses=steps,
            validation_outcome=ValidationOutcome(source.validation_outcome),
            evaluation_status=EvaluationStatus(source.evaluation_status) if source.evaluation_status else None,
            timing=source.timing,
        )
        if len(trace.model_dump_json().encode("utf-8")) > MAX_TRACE_BYTES:
            return None
        return trace
    except Exception:
        return None


def sanitize_trace_step(
    value: TraceStep | dict[str, Any],
    registry: ToolRegistry = tool_registry,
) -> TraceStep | None:
    """Validate a single SSE delta and replace labels/categories with registry data."""
    try:
        source = value if isinstance(value, TraceStep) else TraceStep.model_validate(value)
        definition = registry.get_tool_definition(source.tool_name)
        if definition is None:
            return None
        allowed = {TraceArtifactType.CHART_PNG} if definition.name == "generate_chart" else (
            {TraceArtifactType.REPORT_HTML, TraceArtifactType.REPORT_MARKDOWN}
            if definition.name == "generate_report" else set()
        )
        step = TraceStep(
            step_id=source.step_id,
            tool_name=definition.name,
            tool_label=definition.name.replace("_", " ").title(),
            tool_category=definition.category,
            status=StepStatus(source.status),
            retry_count=min(source.retry_count, 10_000),
            retry_limit=min(source.retry_limit, 10_000),
            failure_category=StepFailureCategory(source.failure_category) if source.failure_category else None,
            artifacts=[artifact for artifact in source.artifacts if artifact.artifact_type in allowed],
            duration_ms=max(0.0, min(source.duration_ms, 86_400_000)),
        )
        return step if len(step.model_dump_json().encode("utf-8")) <= 2_048 else None
    except Exception:
        return None


def apply_trace_step(
    trace: ExecutionTrace,
    event_data: dict[str, Any],
    registry: ToolRegistry = tool_registry,
) -> ExecutionTrace | None:
    """Apply one validated delta from the existing SSE event projection."""
    raw = event_data.get("trace_step")
    if not isinstance(raw, dict):
        return sanitize_execution_trace(trace, registry)
    try:
        replacement = sanitize_trace_step(raw, registry)
        if replacement is None:
            return None
        item = replacement
        if item.step_id not in {step.step_id for step in trace.step_statuses}:
            return None
        steps = [item if step.step_id == item.step_id else step for step in trace.step_statuses]
        updated = sanitize_execution_trace(trace.model_copy(update={"step_statuses": steps}), registry)
        if updated is None or len(updated.model_dump_json().encode("utf-8")) > MAX_TRACE_BYTES:
            return None
        return updated
    except Exception:
        return None
