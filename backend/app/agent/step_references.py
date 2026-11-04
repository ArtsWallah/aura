"""
Safe, typed step output references for AURA execution plans.
Enforces strict grammar, static type compatibility, and safe runtime resolution.
"""

from __future__ import annotations

import collections.abc
import copy
import re
import typing
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from app.tools.schemas import StrictJsonValue

# Supported path syntax:
# Starts with 'data' or 'artifacts', followed by dotted identifiers or bracketed non-negative integer indices.
# Prohibits dunders, slices, negative indices, spaces, parentheses, and quotes.
PATH_REGEX = re.compile(r"^(data|artifacts)(\.[a-zA-Z_][a-zA-Z0-9_]*|\[[0-9]+\])*$")
SEGMENT_REGEX = re.compile(r"\.([a-zA-Z_][a-zA-Z0-9_]*)|\[([0-9]+)\]")
MAX_REFERENCE_DEPTH = 10

FORBIDDEN_IDENTIFIERS = frozenset({
    "__class__",
    "__mro__",
    "__dict__",
    "__proto__",
    "__subclasses__",
    "__init__",
    "__globals__",
    "__code__",
})


class ReferenceError(ValueError):
    """Base exception for step output references."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class ReferenceSyntaxError(ReferenceError):
    """Syntax or structural violation in a reference expression."""


class ReferenceTypeError(ReferenceError):
    """Static or runtime type incompatibility."""


class ReferenceResolutionError(ReferenceError):
    """Failure during runtime reference resolution."""


def validate_reference_syntax(value: Any, *, depth: int = 0) -> None:
    """
    Recursively validates reference grammar, sibling key constraints, and recursion depth.
    Rejects malformed references, sibling collisions, and invalid path syntax.
    """
    if depth > MAX_REFERENCE_DEPTH:
        raise ReferenceSyntaxError("NESTING_DEPTH_EXCEEDED", "Parameter nesting exceeds limit of 10.")

    if isinstance(value, dict):
        has_ref = "$ref" in value
        has_escaped = "$$ref" in value

        if has_ref and has_escaped:
            raise ReferenceSyntaxError(
                "RESERVED_KEY_CONFLICT", "Dict cannot contain both '$ref' and '$$ref'."
            )

        if has_ref:
            if len(value) != 1:
                raise ReferenceSyntaxError(
                    "SIBLING_KEYS_FORBIDDEN", "Reference object must contain only '$ref'."
                )
            body = value["$ref"]
            if not isinstance(body, dict):
                raise ReferenceSyntaxError(
                    "MALFORMED_REFERENCE_BODY", "'$ref' value must be a JSON object."
                )
            if set(body.keys()) != {"step_id", "path"}:
                raise ReferenceSyntaxError(
                    "INVALID_REFERENCE_FIELDS", "Body must contain exactly 'step_id' and 'path'."
                )
            step_id = body["step_id"]
            if not isinstance(step_id, str) or not step_id.strip():
                raise ReferenceSyntaxError(
                    "INVALID_STEP_ID", "'step_id' must be a non-empty string."
                )
            path = body["path"]
            if not isinstance(path, str) or not PATH_REGEX.fullmatch(path):
                raise ReferenceSyntaxError(
                    "INVALID_PATH_SYNTAX", f"Path '{path}' has invalid syntax."
                )
            if "__" in path or any(forbidden in path for forbidden in FORBIDDEN_IDENTIFIERS):
                raise ReferenceSyntaxError(
                    "FORBIDDEN_PATH_SEGMENT", f"Path '{path}' contains prohibited attribute access."
                )
            return

        # Validate non-reference dictionary items
        for k, v in value.items():
            if not isinstance(k, str):
                raise ReferenceSyntaxError("NON_STRING_KEY", "Dictionary keys must be strings.")
            validate_reference_syntax(v, depth=depth + 1)

    elif isinstance(value, list):
        for item in value:
            validate_reference_syntax(item, depth=depth + 1)


def find_all_references(value: Any, *, prefix: str = "", depth: int = 0) -> list[tuple[str, dict[str, str]]]:
    """
    Recursively collects all valid $ref specifications with their parameter location prefix.
    """
    if depth > MAX_REFERENCE_DEPTH:
        raise ReferenceSyntaxError("NESTING_DEPTH_EXCEEDED", "Parameter nesting exceeds limit of 10.")

    references: list[tuple[str, dict[str, str]]] = []
    if isinstance(value, dict):
        if "$ref" in value and len(value) == 1:
            references.append((prefix, value["$ref"]))
            return references
        for k, v in value.items():
            sub_prefix = f"{prefix}.{k}" if prefix else k
            references.extend(find_all_references(v, prefix=sub_prefix, depth=depth + 1))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            sub_prefix = f"{prefix}[{idx}]"
            references.extend(find_all_references(item, prefix=sub_prefix, depth=depth + 1))
    return references


def is_nullable(tp: Any) -> bool:
    """Returns True if the type accepts None."""
    if tp is type(None):
        return True
    origin = get_origin(tp)
    if origin is typing.Union:
        return type(None) in get_args(tp)
    return False


def strip_none(tp: Any) -> Any:
    """Removes NoneType from a Union type."""
    origin = get_origin(tp)
    if origin is typing.Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
        return typing.Union[tuple(args)]
    return tp


def check_type_compatibility(source_type: Any, target_type: Any) -> tuple[bool, str]:
    """
    Evaluates conservative type compatibility between upstream source and downstream target.
    
    Rules enforced:
    - Nullable -> non-nullable is strictly REJECTED.
    - bool -> int is strictly REJECTED.
    - int -> float is ALLOWED.
    - float -> int is strictly REJECTED.
    - str -> numeric/bool is strictly REJECTED.
    - list[A] -> list[B] allowed iff A -> B.
    - dict[str, A] -> dict[str, B] allowed iff A -> B.
    - Literal -> str is ALLOWED.
    - str -> Literal is REJECTED.
    """
    if target_type in (Any, StrictJsonValue):
        return True, ""

    # 1. Nullability check
    src_is_nullable = is_nullable(source_type)
    tgt_is_nullable = is_nullable(target_type)
    if src_is_nullable and not tgt_is_nullable:
        return False, f"Nullable source '{source_type}' cannot be assigned to non-nullable target '{target_type}'."

    non_null_src = strip_none(source_type)
    non_null_tgt = strip_none(target_type)

    if non_null_src is non_null_tgt:
        return True, ""

    # 2. Target Union handling (source matches if compatible with ANY target variant)
    tgt_origin = get_origin(non_null_tgt)
    if tgt_origin is typing.Union:
        variants = get_args(non_null_tgt)
        for variant in variants:
            ok, _ = check_type_compatibility(non_null_src, variant)
            if ok:
                return True, ""
        return False, f"Source '{source_type}' is not compatible with any variant of target '{target_type}'."

    # 3. Source Union handling (EVERY variant in source must be compatible with target)
    src_origin = get_origin(non_null_src)
    if src_origin is typing.Union:
        variants = get_args(non_null_src)
        for variant in variants:
            ok, reason = check_type_compatibility(variant, non_null_tgt)
            if not ok:
                return False, f"Source variant '{variant}' in '{source_type}' is incompatible: {reason}"
        return True, ""

    # 4. Booleans (strictly distinct from int)
    if non_null_src is bool:
        if non_null_tgt is bool:
            return True, ""
        if non_null_tgt is int:
            return False, "Boolean cannot be converted to integer."
        return False, f"Boolean is incompatible with target '{non_null_tgt}'."

    # 5. Integers
    if non_null_src is int:
        if non_null_tgt is int:
            return True, ""
        if non_null_tgt is float:
            return True, ""  # Lossless widening
        return False, f"Integer is incompatible with target '{non_null_tgt}'."

    # 6. Floats
    if non_null_src is float:
        if non_null_tgt is float:
            return True, ""
        if non_null_tgt is int:
            return False, "Float cannot be truncated to integer."
        return False, f"Float is incompatible with target '{non_null_tgt}'."

    # 7. Strings
    if non_null_src is str:
        if non_null_tgt is str:
            return True, ""
        if tgt_origin is Literal:
            return False, f"Unconstrained string cannot satisfy specific Literal target '{non_null_tgt}'."
        return False, f"String is incompatible with target '{non_null_tgt}'."

    # 8. Literals
    if src_origin is Literal:
        if non_null_tgt is str:
            return True, ""
        if tgt_origin is Literal:
            src_vals = set(get_args(non_null_src))
            tgt_vals = set(get_args(non_null_tgt))
            if src_vals.issubset(tgt_vals):
                return True, ""
            return False, f"Literal source values {src_vals} not contained in target {tgt_vals}."
        return False, f"Literal is incompatible with target '{non_null_tgt}'."

    # 9. Lists / Collections
    if src_origin in (list, typing.List, collections.abc.Sequence):
        if tgt_origin in (list, typing.List, collections.abc.Sequence):
            src_elem = get_args(non_null_src)[0] if get_args(non_null_src) else Any
            tgt_elem = get_args(non_null_tgt)[0] if get_args(non_null_tgt) else Any
            return check_type_compatibility(src_elem, tgt_elem)
        return False, f"List source is incompatible with non-list target '{non_null_tgt}'."

    # 10. Dictionaries / Mappings
    if src_origin in (dict, typing.Dict, collections.abc.Mapping):
        if tgt_origin in (dict, typing.Dict, collections.abc.Mapping):
            src_args = get_args(non_null_src)
            tgt_args = get_args(non_null_tgt)
            src_val = src_args[1] if len(src_args) > 1 else Any
            tgt_val = tgt_args[1] if len(tgt_args) > 1 else Any
            return check_type_compatibility(src_val, tgt_val)
        return False, f"Dict source is incompatible with non-dict target '{non_null_tgt}'."

    # 11. Subclass checks for Pydantic models / custom classes
    if isinstance(non_null_src, type) and isinstance(non_null_tgt, type):
        if issubclass(non_null_src, non_null_tgt) and not (non_null_src is bool and issubclass(non_null_tgt, int)):
            return True, ""

    return False, f"Source type '{source_type}' is incompatible with target '{target_type}'."


def infer_output_path_type(output_model: type[BaseModel] | None, path: str) -> Any:
    """
    Statically analyzes an output model schema to determine the resulting type of a path.
    Raises ReferenceTypeError if the path is invalid or accesses nonexistent fields.
    """
    if not path.startswith("data") and not path.startswith("artifacts"):
        raise ReferenceTypeError("INVALID_PATH_ROOT", f"Path must start with 'data' or 'artifacts', got '{path}'.")

    if path.startswith("artifacts"):
        curr_type: Any = list[str]
        remainder = path[len("artifacts"):]
    else:
        if output_model is None:
            raise ReferenceTypeError("MISSING_OUTPUT_MODEL", "Source tool has no registered output model.")
        curr_type = output_model
        remainder = path[len("data"):]

    for match in SEGMENT_REGEX.finditer(remainder):
        attr_name, index_str = match.groups()

        if attr_name is not None:
            if attr_name.startswith("__") or attr_name in FORBIDDEN_IDENTIFIERS:
                raise ReferenceTypeError("FORBIDDEN_IDENTIFIER", f"Prohibited access to '{attr_name}'.")

            # Handle optional wrapping
            curr_is_nullable = is_nullable(curr_type)
            actual_type = strip_none(curr_type)

            if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
                if attr_name not in actual_type.model_fields:
                    raise ReferenceTypeError(
                        "UNKNOWN_FIELD", f"Field '{attr_name}' does not exist on model '{actual_type.__name__}'."
                    )
                field_annotation = actual_type.model_fields[attr_name].annotation
                curr_type = field_annotation if not curr_is_nullable else (field_annotation | None)

            elif get_origin(actual_type) in (dict, typing.Dict, collections.abc.Mapping):
                args = get_args(actual_type)
                val_type = args[1] if len(args) > 1 else Any
                curr_type = val_type if not curr_is_nullable else (val_type | None)

            else:
                raise ReferenceTypeError(
                    "CANNOT_NAVIGATE_ATTRIBUTE",
                    f"Cannot navigate attribute '{attr_name}' on non-object type '{curr_type}'.",
                )

        elif index_str is not None:
            actual_type = strip_none(curr_type)
            origin = get_origin(actual_type)
            if origin in (list, typing.List, collections.abc.Sequence):
                args = get_args(actual_type)
                curr_type = args[0] if args else Any
            elif origin is tuple:
                args = get_args(actual_type)
                idx = int(index_str)
                if len(args) == 2 and args[1] is Ellipsis:
                    curr_type = args[0]
                elif idx < len(args):
                    curr_type = args[idx]
                else:
                    raise ReferenceTypeError("TUPLE_INDEX_OUT_OF_BOUNDS", f"Index [{idx}] out of range for tuple.")
            else:
                raise ReferenceTypeError(
                    "CANNOT_INDEX_NON_LIST", f"Cannot apply index '[{index_str}]' to non-list type '{curr_type}'."
                )

    return curr_type


def resolve_target_parameter_type(input_model: type[BaseModel] | None, arg_path: str) -> Any:
    """
    Inspects destination tool input model to determine expected type for an argument slot.
    """
    if input_model is None:
        return Any

    tokens = []
    # Split by dotted names and bracketed indexes: e.g. "columns[0]" -> ["columns", "[0]"]
    first = arg_path.split(".")[0].split("[")[0]
    tokens.append(first)
    rem = arg_path[len(first):]
    for match in SEGMENT_REGEX.finditer(rem):
        attr, idx = match.groups()
        if attr:
            tokens.append(attr)
        if idx:
            tokens.append(f"[{idx}]")

    curr_type: Any = input_model
    for token in tokens:
        if token.startswith("["):
            origin = get_origin(strip_none(curr_type))
            if origin in (list, typing.List, collections.abc.Sequence):
                args = get_args(strip_none(curr_type))
                curr_type = args[0] if args else Any
            else:
                return Any
        else:
            actual_type = strip_none(curr_type)
            if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
                field_info = actual_type.model_fields.get(token)
                if field_info is None:
                    return Any
                curr_type = field_info.annotation
            elif get_origin(actual_type) in (dict, typing.Dict):
                args = get_args(actual_type)
                curr_type = args[1] if len(args) > 1 else Any
            else:
                return Any

    return curr_type


def create_type_placeholder(tp: Any) -> Any:
    """Creates a benign placeholder value matching tp for static Pydantic argument validation."""
    if is_nullable(tp):
        return None
    actual = strip_none(tp)
    origin = get_origin(actual)
    if origin is Literal:
        args = get_args(actual)
        return args[0] if args else "placeholder"
    if origin in (list, typing.List, collections.abc.Sequence):
        return []
    if origin in (dict, typing.Dict, collections.abc.Mapping):
        return {}
    if actual is bool:
        return True
    if actual is int:
        return 0
    if actual is float:
        return 0.0
    if actual is str:
        return "placeholder"
    return "placeholder"


def navigate_runtime_path(
    root: Any,
    path: str,
    *,
    active_task_id: str | None = None,
    task_manifest: list[dict[str, str]] | None = None,
) -> Any:
    """
    Traverses concrete runtime dictionary/list output according to a path expression.
    Enforces task isolation and manifest boundary checks on artifacts.
    """
    if not path.startswith("data") and not path.startswith("artifacts"):
        raise ReferenceResolutionError("INVALID_PATH_ROOT", f"Path must start with 'data' or 'artifacts', got '{path}'.")

    if path.startswith("artifacts"):
        # Artifacts root navigation
        if isinstance(root, dict) and "artifacts" in root:
            current = root["artifacts"]
        elif hasattr(root, "artifacts"):
            current = getattr(root, "artifacts")
        else:
            current = root
        remainder = path[len("artifacts"):]
    else:
        # Data root navigation
        if isinstance(root, dict) and "data" in root:
            current = root["data"]
        else:
            current = root
        remainder = path[len("data"):]

    for match in SEGMENT_REGEX.finditer(remainder):
        attr_name, index_str = match.groups()

        if current is None:
            raise ReferenceResolutionError("NULL_DEREFERENCE", f"Cannot navigate '{path}' through null value.")

        if attr_name is not None:
            if attr_name.startswith("__") or attr_name in FORBIDDEN_IDENTIFIERS:
                raise ReferenceResolutionError("FORBIDDEN_ATTRIBUTE", f"Access to '{attr_name}' is forbidden.")

            if isinstance(current, dict):
                if attr_name not in current:
                    raise ReferenceResolutionError("KEY_NOT_FOUND", f"Key '{attr_name}' not found in output.")
                current = current[attr_name]
            elif hasattr(current, attr_name):
                current = getattr(current, attr_name)
            else:
                raise ReferenceResolutionError(
                    "FIELD_NOT_FOUND", f"Field '{attr_name}' not found on {type(current).__name__}."
                )

        elif index_str is not None:
            idx = int(index_str)
            if not isinstance(current, (list, tuple)):
                raise ReferenceResolutionError(
                    "TYPE_MISMATCH", f"Expected list at '[{idx}]', got {type(current).__name__}."
                )
            if idx < 0 or idx >= len(current):
                raise ReferenceResolutionError(
                    "INDEX_OUT_OF_BOUNDS", f"Index [{idx}] out of range (length {len(current)})."
                )
            current = current[idx]

    # Post-traversal security verification for artifact results
    if path.startswith("artifacts") and current is not None:
        # If an artifact item or list was resolved, enforce task manifest ownership
        _verify_artifact_isolation(current, active_task_id=active_task_id, task_manifest=task_manifest)

    return current


def _verify_artifact_isolation(
    artifact_val: Any,
    *,
    active_task_id: str | None,
    task_manifest: list[dict[str, str]] | None,
) -> None:
    """Verifies that resolved artifact references belong exclusively to active_task_id."""
    items = artifact_val if isinstance(artifact_val, list) else [artifact_val]
    for item in items:
        if isinstance(item, dict):
            item_task_id = item.get("task_id")
            if item_task_id and active_task_id and item_task_id != active_task_id:
                raise ReferenceResolutionError(
                    "CROSS_TASK_ARTIFACT_ACCESS",
                    f"Artifact belongs to task '{item_task_id}', not active task '{active_task_id}'.",
                )
        elif isinstance(item, str):
            # Check string paths or identifiers against authorized task manifest
            if task_manifest is not None and active_task_id:
                allowed_identifiers = {m["artifact_id"] for m in task_manifest} | {
                    m["relative_path"] for m in task_manifest
                }
                filename = item.replace("\\", "/").split("/")[-1]
                if item not in allowed_identifiers and filename not in allowed_identifiers:
                    # If it's a cross-task or uncontained path, reject
                    if "task_" in item and active_task_id not in item:
                        raise ReferenceResolutionError(
                            "CROSS_TASK_ARTIFACT_ACCESS",
                            f"Artifact string '{item}' not found in authorized task manifest.",
                        )


def resolve_step_references(
    parameters: dict[str, Any],
    step_results: dict[str, Any],
    *,
    dependencies: list[str],
    active_task_id: str | None = None,
    task_manifest: list[dict[str, str]] | None = None,
    depth: int = 0,
) -> dict[str, Any]:
    """
    Recursively resolves $ref expressions against step_results and unescapes $$ref.
    Never mutates the input parameters.
    """
    if depth > MAX_REFERENCE_DEPTH:
        raise ReferenceResolutionError("NESTING_DEPTH_EXCEEDED", "Reference nesting depth limit (10) exceeded.")

    def _resolve_node(node: Any, current_depth: int) -> Any:
        if current_depth > MAX_REFERENCE_DEPTH:
            raise ReferenceResolutionError("NESTING_DEPTH_EXCEEDED", "Reference nesting depth limit (10) exceeded.")

        if isinstance(node, dict):
            # 1. Reserved Key Conflicts
            if "$ref" in node and "$$ref" in node:
                raise ReferenceSyntaxError(
                    "RESERVED_KEY_CONFLICT", "Dict cannot contain both '$ref' and '$$ref'."
                )

            # 2. Reference Object Resolution
            if "$ref" in node:
                if len(node) != 1:
                    raise ReferenceSyntaxError(
                        "SIBLING_KEYS_FORBIDDEN", "Reference object must contain only '$ref'."
                    )
                ref_body = node["$ref"]
                if not isinstance(ref_body, dict) or set(ref_body.keys()) != {"step_id", "path"}:
                    raise ReferenceSyntaxError("MALFORMED_REFERENCE_BODY", "'$ref' body is invalid.")

                step_id = ref_body["step_id"]
                path = ref_body["path"]

                if step_id not in dependencies:
                    raise ReferenceResolutionError(
                        "UNDECLARED_DEPENDENCY",
                        f"Referenced step '{step_id}' is not in declared dependencies {dependencies}.",
                    )

                if step_id not in step_results:
                    raise ReferenceResolutionError(
                        "MISSING_STEP_RESULT", f"Output for prerequisite step '{step_id}' not found."
                    )

                source_result = step_results[step_id]

                # Ensure source step did not fail
                if isinstance(source_result, dict):
                    if source_result.get("success") is False:
                        raise ReferenceResolutionError(
                            "FAILED_PREREQUISITE", f"Prerequisite step '{step_id}' failed execution."
                        )
                    # Cross-task output isolation check
                    result_task_id = source_result.get("task_id")
                    if result_task_id and active_task_id and result_task_id != active_task_id:
                        raise ReferenceResolutionError(
                            "CROSS_TASK_ACCESS",
                            f"Step result belongs to task '{result_task_id}', not active task '{active_task_id}'.",
                        )

                return navigate_runtime_path(
                    source_result,
                    path,
                    active_task_id=active_task_id,
                    task_manifest=task_manifest,
                )

            # 3. Literal Escaped Reference Unescaping ($$ref -> $ref)
            new_dict: dict[str, Any] = {}
            for k, v in node.items():
                resolved_key = "$ref" if k == "$$ref" else k
                new_dict[resolved_key] = _resolve_node(v, current_depth + 1)
            return new_dict

        elif isinstance(node, list):
            return [_resolve_node(item, current_depth + 1) for item in node]

        return node

    # Construct clean resolved dictionary without mutating input
    resolved = _resolve_node(parameters, depth)
    return resolved if isinstance(resolved, dict) else {}

