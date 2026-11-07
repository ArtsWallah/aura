"""Structural and policy validation for LLM and fallback execution plans."""
import re
from collections import deque

from pydantic import ValidationError

from app.agent.state import ExecutionPlan
from app.agent.step_references import (
    ReferenceError,
    ReferenceSyntaxError,
    ReferenceTypeError,
    check_type_compatibility,
    create_type_placeholder,
    find_all_references,
    infer_output_path_type,
    resolve_target_parameter_type,
    validate_reference_syntax,
)
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


def _build_placeholder_parameters(params: Any, in_model: Any, prefix: str = "") -> Any:
    if isinstance(params, dict):
        if "$ref" in params and len(params) == 1:
            tgt_type = resolve_target_parameter_type(in_model, prefix)
            return create_type_placeholder(tgt_type)
        new_d = {}
        for k, v in params.items():
            new_k = "$ref" if k == "$$ref" else k
            sub_prefix = f"{prefix}.{k}" if prefix else k
            new_d[new_k] = _build_placeholder_parameters(v, in_model, sub_prefix)
        return new_d
    elif isinstance(params, list):
        return [
            _build_placeholder_parameters(item, in_model, f"{prefix}[{idx}]")
            for idx, item in enumerate(params)
        ]
    return params


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
    step_order = {}
    for idx, step in enumerate(plan.steps):
        if not STEP_ID_PATTERN.fullmatch(step.step_id):
            raise PlanValidationError("invalid_step_id", "Execution plan contains an invalid step identifier.")
        if step.step_id in steps_by_id:
            raise PlanValidationError("duplicate_step_id", "Execution plan contains duplicate step identifiers.")
        steps_by_id[step.step_id] = step
        step_order[step.step_id] = idx

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

        # Validate reference syntax and recursion limits
        try:
            validate_reference_syntax(step.parameters)
        except ReferenceSyntaxError as exc:
            raise PlanValidationError("malformed_reference", exc.message) from exc

        refs = find_all_references(step.parameters)
        dest_in_model = registry.get_input_model(step.tool_name)

        if refs:
            for arg_path, ref_dict in refs:
                src_id = ref_dict["step_id"]
                ref_path = ref_dict["path"]

                if src_id not in steps_by_id:
                    raise PlanValidationError(
                        "missing_reference_step",
                        f"Step '{step.step_id}' references unknown step '{src_id}'.",
                    )
                if src_id == step.step_id:
                    raise PlanValidationError(
                        "self_reference",
                        "A step cannot reference its own output.",
                    )
                if src_id not in step.dependencies:
                    raise PlanValidationError(
                        "undeclared_reference_dependency",
                        f"Step '{step.step_id}' references '{src_id}' but does not declare it as a dependency.",
                    )
                if step_order[src_id] >= step_order[step.step_id]:
                    raise PlanValidationError(
                        "forward_reference",
                        f"Step '{step.step_id}' references downstream step '{src_id}'.",
                    )

                src_tool_name = steps_by_id[src_id].tool_name
                src_out_model = registry.get_output_model(src_tool_name)
                try:
                    src_type = infer_output_path_type(src_out_model, ref_path)
                except ReferenceTypeError as exc:
                    raise PlanValidationError("invalid_reference_path", exc.message) from exc

                tgt_type = resolve_target_parameter_type(dest_in_model, arg_path)

                if arg_path == "file_path" or arg_path.endswith(".file_path"):
                    if not ref_path.startswith("artifacts"):
                        raise PlanValidationError(
                            "type_incompatibility",
                            f"Parameter '{arg_path}' can only reference artifacts from an upstream transform_dataset step.",
                        )
                    if src_tool_name != "transform_dataset":
                        raise PlanValidationError(
                            "type_incompatibility",
                            f"Tool '{src_tool_name}' produces non-dataset artifacts incompatible with '{arg_path}'.",
                        )
                    # When targeting file_path from transform_dataset artifacts, single-item references resolve to str (canonical file path).
                    if check_type_compatibility(src_type, dict[str, str])[0] or src_type is str:
                        src_type = str

                ok, reason = check_type_compatibility(src_type, tgt_type)
                if not ok:
                    raise PlanValidationError(
                        "type_incompatibility",
                        f"Reference in '{arg_path}' is incompatible: {reason}",
                    )

            placeholder_args = _build_placeholder_parameters(step.parameters, dest_in_model)
            try:
                registry.validate_arguments(step.tool_name, placeholder_args)
            except ValidationError as exc:
                raise PlanValidationError("invalid_tool_parameters", "Execution plan contains invalid tool parameters.") from exc
        else:
            unescaped_args = _build_placeholder_parameters(step.parameters, dest_in_model)
            try:
                registry.validate_arguments(step.tool_name, unescaped_args)
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
