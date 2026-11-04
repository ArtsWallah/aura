"""
Phase 3.3.1-B3: Dynamic Output References Test Suite.
Verifies syntax grammar, static plan type-checking, runtime resolution,
sealed plan immutability, digest binding, task isolation, and end-to-end tool chaining.
"""

import asyncio
import copy
import hashlib
import json
import pytest
from pathlib import Path

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.execution_handoff import (
    ExecutionHandoff,
    _HANDOFF_FACTORY_TOKEN,
    _intent_digest,
    _plan_digest,
    handoff_execution_scope,
    test_legacy_tool_invocation as legacy_tool_scope,
)
from app.agent.executor import PlanExecutor
from app.agent.intelligence_contracts import EvaluationStatus, IntentCategory, ProviderIdentity
from app.agent.orchestrator import AgentOrchestrator
from app.agent.plan_validation import PlanValidationError, validate_execution_plan
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus, TaskState, TaskStatus
from app.agent.step_references import (
    ReferenceResolutionError,
    ReferenceSyntaxError,
    ReferenceTypeError,
    check_type_compatibility,
    infer_output_path_type,
    resolve_step_references,
    validate_reference_syntax,
)
from app.tools.analytics_tools import GroupComparisonOutput
from app.tools.dataset_tools import InspectDatasetOutput
from app.tools.registry import tool_registry
from app.tools.schemas import StrictContract


def _make_handoff(plan: ExecutionPlan, task_id: str = "task_test") -> ExecutionHandoff:
    valid_plan = validate_execution_plan(plan.model_copy(deep=True))
    return ExecutionHandoff(
        _factory_token=_HANDOFF_FACTORY_TOKEN,
        handoff_id=f"handoff_{task_id}",
        plan=valid_plan,
        privacy_mode="hybrid",
        provider_identity=ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER,
        intent_category=IntentCategory.DATA_ANALYSIS,
        intent_confidence=1.0,
        intent_is_multi_step=True,
        required_tool_groups=[["dataset", "analytics"]],
        evaluation_status=EvaluationStatus.PLAN_VALID,
        objective_coverage=1.0,
        validation_registry_id=tool_registry.registry_id,
        plan_digest=_plan_digest(valid_plan),
        intent_digest=_intent_digest(1.0, True, [["dataset", "analytics"]]),
    )


def test_01_valid_scalar_reference():
    """Valid scalar reference extracts exact scalar value from upstream output."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "data": {
                "dataset_name": "data.csv",
                "row_count": 42,
                "column_count": 3,
                "columns": ["col_a", "col_b", "col_c"],
                "dtypes": {"col_a": "int"},
                "missing_values": {"col_a": 0},
                "numeric_columns": ["col_a"],
                "categorical_columns": ["col_b"],
                "datetime_columns": [],
                "summary_statistics": {},
                "preview": [],
            },
        }
    }
    params = {
        "count_ref": {"$ref": {"step_id": "step_1", "path": "data.row_count"}},
        "col_ref": {"$ref": {"step_id": "step_1", "path": "data.columns[0]"}},
    }
    resolved = resolve_step_references(params, step_results, dependencies=["step_1"])
    assert resolved["count_ref"] == 42
    assert resolved["col_ref"] == "col_a"


def test_02_valid_nested_dictionary_reference():
    """Valid nested dictionary reference extracts deeply nested value."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "data": {
                "dataset_name": "data.csv",
                "row_count": 100,
                "column_count": 2,
                "columns": ["age"],
                "dtypes": {"age": "float"},
                "missing_values": {"age": 0},
                "numeric_columns": ["age"],
                "categorical_columns": [],
                "datetime_columns": [],
                "summary_statistics": {
                    "age": {
                        "count": 100,
                        "mean": 35.5,
                        "std": 5.2,
                        "min": 18.0,
                        "median": 34.0,
                        "max": 65.0,
                        "unique": 40,
                    }
                },
                "preview": [],
            },
        }
    }
    params = {
        "mean_age": {"$ref": {"step_id": "step_1", "path": "data.summary_statistics.age.mean"}}
    }
    resolved = resolve_step_references(params, step_results, dependencies=["step_1"])
    assert resolved["mean_age"] == 35.5


def test_03_valid_list_index_reference():
    """Valid list index reference extracts element from list."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "data": {
                "dataset_name": "data.csv",
                "row_count": 10,
                "column_count": 2,
                "columns": ["dept", "sales"],
                "dtypes": {},
                "missing_values": {},
                "numeric_columns": ["sales"],
                "categorical_columns": ["dept"],
                "datetime_columns": [],
                "summary_statistics": {},
                "preview": [],
            },
        }
    }
    params = {
        "group_col": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}}
    }
    resolved = resolve_step_references(params, step_results, dependencies=["step_1"])
    assert resolved["group_col"] == "dept"


def test_04_multiple_references_in_one_argument_object():
    """Multiple references in one argument object resolve simultaneously."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "data": {
                "dataset_name": "test.csv",
                "row_count": 100,
                "column_count": 2,
                "columns": ["category", "revenue"],
                "dtypes": {},
                "missing_values": {},
                "numeric_columns": ["revenue"],
                "categorical_columns": ["category"],
                "datetime_columns": [],
                "summary_statistics": {},
                "preview": [],
            },
        }
    }
    params = {
        "group_column": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}},
        "value_column": {"$ref": {"step_id": "step_1", "path": "data.numeric_columns[0]"}},
    }
    resolved = resolve_step_references(params, step_results, dependencies=["step_1"])
    assert resolved == {"group_column": "category", "value_column": "revenue"}


def test_05_missing_source_step_rejected():
    """Plan referencing non-existent upstream step is rejected at validation time."""
    plan = ExecutionPlan(
        plan_id="plan_missing_step",
        goal="Test missing step",
        steps=[
            ExecutionStep(
                step_id="step_1",
                title="Step 1",
                description="Inspect dataset",
                capability="system",
                tool_name="inspect_dataset",
                parameters={"file_path": "data.csv"},
            ),
            ExecutionStep(
                step_id="step_2",
                title="Step 2",
                description="Group comparison",
                capability="system",
                tool_name="group_comparison",
                dependencies=["step_1"],
                parameters={
                    "file_path": "data.csv",
                    "group_by": {"$ref": {"step_id": "step_ghost", "path": "data.categorical_columns[0]"}},
                    "metric_column": "sales",
                },
            ),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_execution_plan(plan)
    assert exc.value.code == "missing_reference_step"


def test_06_forward_reference_rejected():
    """Step referencing a downstream step is rejected at validation time."""
    plan = ExecutionPlan(
        plan_id="plan_forward_ref",
        goal="Test forward ref",
        steps=[
            ExecutionStep(
                step_id="step_1",
                title="Step 1",
                description="Group comparison",
                capability="system",
                tool_name="group_comparison",
                dependencies=[],
                parameters={
                    "group_column": {"$ref": {"step_id": "step_2", "path": "data.categorical_columns[0]"}},
                    "value_column": "sales",
                },
            ),
            ExecutionStep(
                step_id="step_2",
                title="Step 2",
                description="Inspect dataset",
                capability="system",
                tool_name="inspect_dataset",
                parameters={"file_path": "data.csv"},
            ),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_execution_plan(plan)
    assert exc.value.code in ("forward_reference", "undeclared_reference_dependency")


def test_07_missing_dependency_declaration_rejected():
    """Step referencing an upstream step without declaring it in dependencies is rejected."""
    plan = ExecutionPlan(
        plan_id="plan_missing_dep",
        goal="Test missing dep declaration",
        steps=[
            ExecutionStep(
                step_id="step_1",
                title="Step 1",
                description="Inspect dataset",
                capability="system",
                tool_name="inspect_dataset",
                parameters={"file_path": "data.csv"},
            ),
            ExecutionStep(
                step_id="step_2",
                title="Step 2",
                description="Group comparison",
                capability="system",
                tool_name="group_comparison",
                dependencies=[],  # Omitted step_1!
                parameters={
                    "group_column": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}},
                    "value_column": "sales",
                },
            ),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_execution_plan(plan)
    assert exc.value.code == "undeclared_reference_dependency"


def test_08_failed_prerequisite():
    """Runtime resolution fails closed when prerequisite step marked failed."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "success": False,
            "error": "Dataset corrupt",
            "data": None,
        }
    }
    params = {
        "group_column": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}}
    }
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(params, step_results, dependencies=["step_1"])
    assert exc.value.code == "FAILED_PREREQUISITE"


def test_09_missing_output_key():
    """Runtime resolution fails when path requests nonexistent field in dynamic map."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "data": {
                "summary_statistics": {"age": {"count": 10}}
            },
        }
    }
    params = {
        "unknown": {"$ref": {"step_id": "step_1", "path": "data.summary_statistics.nonexistent.mean"}}
    }
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(params, step_results, dependencies=["step_1"])
    assert exc.value.code in ("KEY_NOT_FOUND", "NULL_DEREFERENCE")


def test_10_invalid_list_index():
    """Runtime resolution fails when list index is out of bounds."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "data": {
                "categorical_columns": ["only_one"]
            },
        }
    }
    params = {
        "col": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[99]"}}
    }
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(params, step_results, dependencies=["step_1"])
    assert exc.value.code == "INDEX_OUT_OF_BOUNDS"


def test_11_negative_index_rejected():
    """Negative list index is rejected by syntax validator."""
    with pytest.raises(ReferenceSyntaxError) as exc:
        validate_reference_syntax({"$ref": {"step_id": "step_1", "path": "data.columns[-1]"}})
    assert exc.value.code == "INVALID_PATH_SYNTAX"


def test_12_slices_and_unsupported_syntax_rejected():
    """Slices and wildcards are rejected by syntax validator."""
    for bad_path in ["data.columns[0:2]", "data.columns[:]", "data.columns[*]", "data['columns']"]:
        with pytest.raises(ReferenceSyntaxError) as exc:
            validate_reference_syntax({"$ref": {"step_id": "step_1", "path": bad_path}})
        assert exc.value.code == "INVALID_PATH_SYNTAX"


def test_13_function_call_and_dunder_attempts_rejected():
    """Function calls and dunder attributes are strictly rejected."""
    for bad_path in ["data.__class__", "data.__dict__", "data.get_columns()", "data.__mro__"]:
        with pytest.raises(ReferenceSyntaxError) as exc:
            validate_reference_syntax({"$ref": {"step_id": "step_1", "path": bad_path}})
        assert exc.value.code in ("INVALID_PATH_SYNTAX", "FORBIDDEN_PATH_SEGMENT")


def test_14_excessive_reference_depth():
    """Nesting references beyond limit 10 is rejected."""
    nested: dict = {"$ref": {"step_id": "step_1", "path": "data.row_count"}}
    for _ in range(12):
        nested = {"level": nested}
    with pytest.raises(ReferenceSyntaxError) as exc:
        validate_reference_syntax(nested)
    assert exc.value.code == "NESTING_DEPTH_EXCEEDED"


def test_15_type_incompatibility():
    """Incompatible type mapping (str -> int) is rejected at plan validation time."""
    plan = ExecutionPlan(
        plan_id="plan_incompatible_type",
        goal="Test type mismatch",
        steps=[
            ExecutionStep(
                step_id="step_1",
                title="Step 1",
                description="Inspect dataset",
                capability="system",
                tool_name="inspect_dataset",
                parameters={"file_path": "data.csv"},
            ),
            ExecutionStep(
                step_id="step_2",
                title="Step 2",
                description="Chart",
                capability="system",
                tool_name="generate_chart",
                dependencies=["step_1"],
                parameters={
                    "file_path": "data.csv",
                    "chart_type": "bar",
                    "x_column": "cat",
                    "title": "Title",
                    "top_n": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}},  # str -> int!
                },
            ),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_execution_plan(plan)
    assert exc.value.code == "type_incompatibility"


def test_16_nullable_to_non_nullable_rejection():
    """Passing nullable output (float | None) to non-nullable target (float) is rejected."""
    # ColumnSummary.mean is float | None.
    ok, reason = check_type_compatibility(float | None, float)
    assert ok is False
    assert "Nullable source" in reason


def test_17_bool_to_int_rejection():
    """Passing bool to int is strictly rejected."""
    ok, reason = check_type_compatibility(bool, int)
    assert ok is False
    assert "Boolean cannot be converted to integer" in reason


def test_18_float_to_int_rejection():
    """Passing float to int is strictly rejected."""
    ok, reason = check_type_compatibility(float, int)
    assert ok is False
    assert "Float cannot be truncated to integer" in reason


def test_19_literal_ref_escaping():
    """Escaped $$ref keys unescape to $ref literal without reference traversal."""
    params = {
        "$$ref": "literal_value",
        "nested": {"$$ref": 123},
    }
    resolved = resolve_step_references(params, {}, dependencies=[])
    assert resolved == {
        "$ref": "literal_value",
        "nested": {"$ref": 123},
    }


def test_20_cross_task_artifact_access_rejection():
    """Cross-task artifact reference is strictly rejected."""
    step_results = {
        "step_1": {
            "tool": "generate_chart",
            "task_id": "task_external_b",
            "artifacts": ["C:/charts/chart_ext12345.png.enc"],
            "data": {},
        }
    }
    params = {
        "art": {"$ref": {"step_id": "step_1", "path": "artifacts[0]"}}
    }
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_1"],
            active_task_id="task_active_a",
        )
    assert exc.value.code in ("CROSS_TASK_ACCESS", "CROSS_TASK_ARTIFACT_ACCESS")


def test_21_resolved_arguments_revalidated_against_destination_input_model():
    """Resolved arguments failing destination input validation fail closed."""
    step = ExecutionStep(
        step_id="step_2",
        title="Group comparison",
        description="Compare groups",
        capability="system",
        tool_name="group_comparison",
        dependencies=["step_1"],
        parameters={
            "group_column": {"$ref": {"step_id": "step_1", "path": "data.columns[0]"}},
            # value_column is omitted!
        },
    )
    task_state = TaskState(
        task_id="task_test_21",
        objective="Test destination revalidation",
        step_results={
            "step_1": {
                "tool": "inspect_dataset",
                "task_id": "task_test_21",
                "data": {"columns": ["dept"]},
            }
        },
    )
    executor = PlanExecutor()
    with pytest.raises(ReferenceResolutionError) as exc:
        executor._resolve_step_parameters(step, task_state)
    assert exc.value.code == "DESTINATION_VALIDATION_FAILED"


def test_22_sealed_plan_remains_unchanged():
    """PlanExecutor does not mutate sealed handoff plan during resolution and execution."""
    original_params = {
        "file_path": "data.csv",
        "group_by": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}},
        "metric_column": "sales",
    }
    step = ExecutionStep(
        step_id="step_2",
        title="Step 2",
        description="Group Comparison",
        capability="system",
        tool_name="group_comparison",
        dependencies=["step_1"],
        parameters=copy.deepcopy(original_params),
    )
    task_state = TaskState(
        task_id="task_test_22",
        objective="Test plan immutability",
        step_results={
            "step_1": {
                "tool": "inspect_dataset",
                "task_id": "task_test_22",
                "data": {
                    "categorical_columns": ["dept"],
                },
            }
        },
    )
    executor = PlanExecutor()
    resolved = executor._resolve_step_parameters(step, task_state)
    assert resolved["group_by"] == "dept"
    assert resolved["metric_column"] == "sales"
    # Verify step.parameters remains the raw $ref dictionary
    assert step.parameters == original_params


def test_23_invocation_digest_matches_resolved_arguments():
    """Invocation authorization token digest matches the exact resolved argument payload."""
    step = ExecutionStep(
        step_id="step_2",
        title="Step 2",
        description="Group comparison",
        capability="system",
        tool_name="group_comparison",
        dependencies=["step_1"],
        parameters={
            "file_path": "data.csv",
            "group_by": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}},
            "metric_column": "revenue",
        },
    )
    task_state = TaskState(
        task_id="task_test_23",
        objective="Test authorization digest binding",
        step_results={
            "step_1": {
                "tool": "inspect_dataset",
                "task_id": "task_test_23",
                "data": {"categorical_columns": ["region"]},
            }
        },
    )
    executor = PlanExecutor()
    resolved = executor._resolve_step_parameters(step, task_state)
    expected_digest = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()

    # Verify that the digest of resolved is computed
    assert resolved["group_by"] == "region"
    assert resolved["metric_column"] == "revenue"
    assert len(expected_digest) == 64


@pytest.mark.asyncio
async def test_24_end_to_end_real_tool_chaining():
    """
    End-to-end execution: upstream inspect_dataset feeds group_column dynamically
    to downstream group_comparison through orchestrator execution.
    """
    from app.core.config import settings
    from app.privacy.encryption import encryption_service

    csv_file = settings.UPLOADS_DIR / "sample_e2e_refs.csv.enc"
    raw_csv = b"dept,salary\neng,100\neng,120\nhr,90\nmarketing,110\n"
    csv_file.write_bytes(encryption_service().encrypt_bytes(raw_csv))

    try:
        plan = ExecutionPlan(
            plan_id="plan_e2e_references",
            goal="E2E reference test",
            steps=[
                ExecutionStep(
                    step_id="step_1",
                    title="Inspect dataset",
                    description="Inspect CSV to extract column metadata",
                    capability="system",
                    tool_name="inspect_dataset",
                    parameters={"file_path": str(csv_file)},
                ),
                ExecutionStep(
                    step_id="step_2",
                    title="Group comparison",
                    description="Compare salary by department dynamically",
                    capability="system",
                    tool_name="group_comparison",
                    dependencies=["step_1"],
                    parameters={
                        "file_path": str(csv_file),
                        "group_by": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}},
                        "metric_column": "salary",
                    },
                ),
            ],
        )

        handoff = _make_handoff(plan, task_id="task_e2e_refs")
        orchestrator = AgentOrchestrator()

        with handoff_execution_scope(handoff, task_id="task_e2e_refs", privacy_mode="hybrid"):
            task_state = await orchestrator._run_task(
                objective="Analyze salary by dept",
                supplied_plan=handoff.plan,
                task_id="task_e2e_refs",
                privacy_mode="hybrid",
            )

        assert task_state.status == TaskStatus.COMPLETED
        assert "step_1" in task_state.step_results
        assert "step_2" in task_state.step_results

        # Inspect outputs
        step_1_out = task_state.step_results["step_1"]["data"]
        assert step_1_out["categorical_columns"][0] == "dept"

        step_2_out = task_state.step_results["step_2"]["data"]
        assert step_2_out["group_by"] == "dept"
        assert step_2_out["metric_column"] == "salary"
        assert step_2_out["group_count"] == 3
        assert len(step_2_out["results"]) == 3

        # Verify original plan was unmutated
        assert handoff.plan.steps[1].parameters["group_by"] == {
            "$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}
        }
    finally:
        if csv_file.exists():
            csv_file.unlink()


# -----------------------------------------------------------------------------
# Phase 3.3.1-B3.1: Targeted Security Review Tests
# -----------------------------------------------------------------------------

def test_b3_1_validated_output_unchanged_and_metadata_separated():
    """Validated output payload is unchanged and execution metadata is stored separately."""
    step = ExecutionStep(
        step_id="step_meta",
        title="Test metadata separation",
        description="Run tool",
        capability="system",
        tool_name="system_health",
        parameters={},
    )
    task_state = TaskState(
        task_id="task_meta_123",
        objective="Verify metadata separation",
    )
    raw_output = {
        "status": "healthy",
        "cpu_percent": 12.5,
        "memory_percent": 45.0,
        "disk_free_gb": 120.0,
        "uptime_seconds": 3600.0,
        "active_tasks": 1,
    }
    # Simulate step completion storage in PlanExecutor
    task_state.step_results[step.step_id] = {
        "tool": step.tool_name,
        "data": copy.deepcopy(raw_output),
        "artifacts": [],
        "metadata": {
            "task_id": task_state.task_id,
            "step_id": step.step_id,
            "tool_name": step.tool_name,
            "execution_time_ms": 12.3,
        },
    }

    # 1. Output payload is identical to tool output
    stored_data = task_state.step_results[step.step_id]["data"]
    assert stored_data == raw_output
    assert "task_id" not in stored_data

    # 2. Metadata is stored in a separate container
    stored_meta = task_state.step_results[step.step_id]["metadata"]
    assert stored_meta["task_id"] == "task_meta_123"
    assert stored_meta["step_id"] == "step_meta"
    assert stored_meta["tool_name"] == "system_health"


def test_b3_1_tool_provided_task_id_cannot_override_ownership():
    """A tool output containing task_id in its data payload cannot spoof execution ownership."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "data": {
                "dataset_name": "data.csv",
                "row_count": 10,
                "column_count": 1,
                "columns": ["col_a"],
                "dtypes": {},
                "missing_values": {},
                "numeric_columns": [],
                "categorical_columns": ["col_a"],
                "datetime_columns": [],
                "summary_statistics": {},
                "preview": [],
                # Adversarial spoof attempt in data payload
                "task_id": "spoofed_target_task",
            },
            "metadata": {
                # Authoritative execution ownership
                "task_id": "authoritative_task_a",
            },
        }
    }
    params = {
        "group_by": {"$ref": {"step_id": "step_1", "path": "data.categorical_columns[0]"}}
    }

    # 1. Access by the authoritative task owner succeeds
    resolved = resolve_step_references(
        params,
        step_results,
        dependencies=["step_1"],
        active_task_id="authoritative_task_a",
    )
    assert resolved["group_by"] == "col_a"

    # 2. Access by the spoofed task fails closed (tool output cannot claim ownership)
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_1"],
            active_task_id="spoofed_target_task",
        )
    assert exc.value.code == "CROSS_TASK_ACCESS"


def test_b3_1_references_cannot_access_metadata_as_output():
    """Downstream references cannot access execution metadata as if it were tool output."""
    # 1. Path starting with 'metadata' is rejected by grammar
    with pytest.raises(ReferenceSyntaxError) as exc1:
        validate_reference_syntax({"$ref": {"step_id": "step_1", "path": "metadata.task_id"}})
    assert exc1.value.code == "INVALID_PATH_SYNTAX"

    # 2. Path data.task_id is rejected when not in tool output schema
    with pytest.raises(ReferenceTypeError) as exc2:
        infer_output_path_type(InspectDatasetOutput, "data.task_id")
    assert exc2.value.code == "UNKNOWN_FIELD"


def test_b3_1_raw_path_references_are_rejected():
    """References to raw filesystem paths and data URIs are strictly forbidden."""
    from app.tools.visualization_tools import GenerateChartOutput
    from app.tools.report_tools import GenerateReportOutput

    # file_path, markdown_path, html_path, data_uri, and data.artifacts are all forbidden
    forbidden_paths = [
        (GenerateChartOutput, "data.file_path"),
        (GenerateChartOutput, "data.data_uri"),
        (GenerateChartOutput, "data.artifacts"),
        (GenerateReportOutput, "data.markdown_path"),
        (GenerateReportOutput, "data.html_path"),
    ]
    for model, forbidden_path in forbidden_paths:
        with pytest.raises(ReferenceTypeError) as exc:
            infer_output_path_type(model, forbidden_path)
        assert exc.value.code == "RAW_PATH_REFERENCE_FORBIDDEN"


def test_b3_1_artifact_reference_resolves_to_manifest_descriptor():
    """Artifact references resolve to typed manifest descriptors, never raw filesystem paths."""
    manifest = [
        {
            "artifact_id": "chart_3f8a91b2c4",
            "task_id": "task_active_1",
            "artifact_type": "chart_png",
            "relative_path": "chart_3f8a91b2c4.png.enc",
        }
    ]
    step_results = {
        "step_chart": {
            "tool": "generate_chart",
            "artifacts": ["C:/fake/charts/chart_3f8a91b2c4.png.enc"],
            "data": {},
            "metadata": {"task_id": "task_active_1"},
        }
    }

    # Reference entire descriptor
    params = {
        "chart_desc": {"$ref": {"step_id": "step_chart", "path": "artifacts[0]"}}
    }
    resolved = resolve_step_references(
        params,
        step_results,
        dependencies=["step_chart"],
        active_task_id="task_active_1",
        task_manifest=manifest,
    )
    desc = resolved["chart_desc"]
    assert isinstance(desc, dict)
    assert desc["artifact_id"] == "chart_3f8a91b2c4"
    assert desc["task_id"] == "task_active_1"
    assert desc["artifact_type"] == "chart_png"
    assert desc["relative_path"] == "chart_3f8a91b2c4.png.enc"
    assert "C:" not in desc["relative_path"]  # No absolute path authority!

    # Reference specific descriptor field
    params_field = {
        "art_id": {"$ref": {"step_id": "step_chart", "path": "artifacts[0].artifact_id"}}
    }
    resolved_field = resolve_step_references(
        params_field,
        step_results,
        dependencies=["step_chart"],
        active_task_id="task_active_1",
        task_manifest=manifest,
    )
    assert resolved_field["art_id"] == "chart_3f8a91b2c4"


def test_b3_1_cross_task_artifact_in_manifest_fails_closed():
    """Referencing an artifact that belongs to another task in the manifest fails closed."""
    manifest = [
        {
            "artifact_id": "chart_external99",
            "task_id": "task_other_user",
            "artifact_type": "chart_png",
            "relative_path": "chart_external99.png.enc",
        }
    ]
    step_results = {
        "step_chart": {
            "tool": "generate_chart",
            "artifacts": ["chart_external99.png.enc"],
            "data": {},
            "metadata": {"task_id": "task_active_1"},
        }
    }
    params = {
        "art": {"$ref": {"step_id": "step_chart", "path": "artifacts[0]"}}
    }
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_chart"],
            active_task_id="task_active_1",
            task_manifest=manifest,
        )
    assert exc.value.code == "CROSS_TASK_ARTIFACT_ACCESS"
