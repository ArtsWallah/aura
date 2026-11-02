"""Structural and policy validation for LLM and fallback execution plans."""
import re
from collections import deque

from pydantic import ValidationError

from app.agent.state import ExecutionPlan
from app.core.config import settings
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.schemas import ExecutionLocation, PermissionLevel


STEP_ID_PATTERN = re.compile(r"^step_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PlanValidationError(ValueError):
    """A safe, non-sensitive plan validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_execution_plan(
    plan: ExecutionPlan,
    privacy_mode: str = "hybrid",
    max_steps: int | None = None,
    registry: ToolRegistry | None = None,
) -> ExecutionPlan:
    """Validate plan structure, tool schemas, dependencies, and runtime policy."""
    registry = registry or tool_registry
    if privacy_mode not in {"local_only", "hybrid", "cloud_enabled"}:
        raise PlanValidationError("invalid_privacy_mode", "Plan privacy mode is invalid.")

    step_limit = settings.MAX_PLAN_STEPS if max_steps is None else max_steps
    if not plan.steps:
        raise PlanValidationError("empty_plan", "Execution plan must contain at least one step.")
    if len(plan.steps) > step_limit:
        raise PlanValidationError("step_limit", "Execution plan exceeds the configured step limit.")

    steps_by_id = {}
    for step in plan.steps:
        if not STEP_ID_PATTERN.fullmatch(step.step_id):
            raise PlanValidationError("invalid_step_id", "Execution plan contains an invalid step identifier.")
        if step.step_id in steps_by_id:
            raise PlanValidationError("duplicate_step_id", "Execution plan contains duplicate step identifiers.")
        steps_by_id[step.step_id] = step

    indegree = {step_id: 0 for step_id in steps_by_id}
    dependents = {step_id: [] for step_id in steps_by_id}
    for step in plan.steps:
        if len(set(step.dependencies)) != len(step.dependencies):
            raise PlanValidationError("duplicate_dependency", "Execution plan contains duplicate dependencies.")
        for dependency in step.dependencies:
            if dependency == step.step_id:
                raise PlanValidationError("self_dependency", "A step cannot depend on itself.")
            if dependency not in steps_by_id:
                raise PlanValidationError("missing_dependency", "Execution plan references an unknown dependency.")
            indegree[step.step_id] += 1
            dependents[dependency].append(step.step_id)

        definition = registry.get_tool_definition(step.tool_name)
        if definition is None:
            raise PlanValidationError("unknown_tool", "Execution plan references an unregistered tool.")
        if definition.execution_location == ExecutionLocation.BROWSER:
            raise PlanValidationError("unsupported_location", "Browser-only tools cannot run in the backend executor.")
        if definition.execution_location == ExecutionLocation.CLOUD and privacy_mode != "cloud_enabled":
            raise PlanValidationError("privacy_policy", "This plan requires explicit cloud-enabled privacy mode.")
        if definition.permission_level == PermissionLevel.PRIVILEGED or definition.requires_approval:
            raise PlanValidationError("approval_required", "Execution plan contains a tool requiring explicit approval.")
        if step.max_retries < 0:
            raise PlanValidationError("invalid_retry_limit", "Execution plan contains an invalid retry limit.")
        step.max_retries = min(step.max_retries, settings.MAX_STEP_RETRIES)

        try:
            registry.validate_arguments(step.tool_name, step.parameters)
        except ValidationError as exc:
            # Never return Pydantic's error payload because it includes submitted values.
            raise PlanValidationError("invalid_tool_parameters", "Execution plan contains invalid tool parameters.") from exc

    ready = deque(step_id for step_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        step_id = ready.popleft()
        visited += 1
        for dependent in dependents[step_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if visited != len(steps_by_id):
        raise PlanValidationError("dependency_cycle", "Execution plan contains a dependency cycle.")

    return plan
