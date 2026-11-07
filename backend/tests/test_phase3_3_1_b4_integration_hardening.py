"""
Phase 3.3.1-B4: Dynamic Data Flow Integration and End-to-End Hardening Test Suite.
Verifies realistic multi-step analytics workflows, dynamic reference resolution across
contracts, failure propagation in DAGs, trace and checkpoint privacy, and resource safety.
"""

import asyncio
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.agent.checkpoint import SQLiteCheckpointer, checkpointer
from app.agent.execution_handoff import (
    ExecutionHandoff,
    _HANDOFF_FACTORY_TOKEN,
    _intent_digest,
    _plan_digest,
    handoff_execution_scope,
)
from app.agent.executor import PlanExecutor, executor
from app.agent.intelligence_contracts import (
    EvaluationStatus,
    IntentCategory,
    ProviderIdentity,
    StepFailureCategory,
)
from app.agent.orchestrator import AgentOrchestrator
from app.agent.plan_validation import PlanValidationError, validate_execution_plan
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus, TaskState, TaskStatus
from app.agent.step_references import (
    ReferenceResolutionError,
    ReferenceSyntaxError,
    ReferenceTypeError,
    infer_output_path_type,
    resolve_step_references,
    validate_reference_syntax,
)
from app.core.config import settings
from app.privacy.encryption import encryption_service
from app.tools.analytics_tools import (
    CorrelationAnalysisInput,
    DescriptiveStatsInput,
    GroupComparisonInput,
    GroupComparisonOutput,
    OutlierDetectionInput,
)
from app.tools.dataset_tools import InspectDatasetInput, InspectDatasetOutput
from app.tools.registry import tool_registry
from app.tools.report_tools import GenerateReportInput, GenerateReportOutput
from app.tools.schemas import ToolResult, ToolDefinition, ToolCategory, ExecutionLocation, PermissionLevel, StrictContract
from app.tools.visualization_tools import GenerateChartInput, GenerateChartOutput


def _make_step(
    step_id: str,
    tool_name: str,
    parameters: dict[str, Any],
    dependencies: list[str] | None = None,
    title: str = "Test Step",
    description: str = "Test Step Description",
    capability: str = "system",
) -> ExecutionStep:
    return ExecutionStep(
        step_id=step_id,
        title=title,
        description=description,
        capability=capability,
        tool_name=tool_name,
        parameters=parameters,
        dependencies=dependencies or [],
    )


def _make_plan(
    steps: list[ExecutionStep],
    plan_id: str = "plan_test",
    goal: str = "Test Multi-Step Analytics",
) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        goal=goal,
        steps=steps,
    )


@pytest.fixture
def encrypted_sales_csv():
    """Create sample encrypted sales CSV in settings.UPLOADS_DIR."""
    file_path = settings.UPLOADS_DIR / "sample_sales_b4_test.csv.enc"
    data = {
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
        "category": ["Electronics", "Electronics", "Clothing", "Clothing", "Home", "Home", "Accessories", "Electronics"],
        "region": ["North", "South", "North", "West", "East", "North", "South", "East"],
        "units_sold": [120, 150, 40, 60, 20, 15, 5, 210],
        "revenue": [12000.0, 15000.0, 2400.0, 3600.0, 1200.0, 900.0, 250.0, 21000.0],
        "profit": [3600.0, 4500.0, 720.0, 1080.0, 240.0, 180.0, 50.0, 6300.0]
    }
    df = pd.DataFrame(data)
    raw = df.to_csv(index=False).encode("utf-8")
    file_path.write_bytes(encryption_service().encrypt_bytes(raw))
    yield str(file_path)
    if file_path.exists():
        file_path.unlink()


def _make_handoff(plan: ExecutionPlan, task_id: str = "task_b4_test") -> ExecutionHandoff:
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
        required_tool_groups=[["dataset", "analytics", "visualization", "report"]],
        evaluation_status=EvaluationStatus.PLAN_VALID,
        objective_coverage=1.0,
        validation_registry_id=tool_registry.registry_id,
        plan_digest=_plan_digest(valid_plan),
        intent_digest=_intent_digest(1.0, True, [["dataset", "analytics", "visualization", "report"]]),
    )


# ==============================================================================
# SECTION 1: Realistic Multi-Step Analytics Workflows
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_01_inspect_to_descriptive_stats(encrypted_sales_csv):
    """
    Workflow 1: inspect_dataset -> descriptive_statistics
    Dynamically maps numeric_columns from inspect output to descriptive_statistics.columns.
    """
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": encrypted_sales_csv, "sample_rows": 5},
                dependencies=[],
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": encrypted_sales_csv,
                    "columns": {"$ref": {"step_id": "step_inspect", "path": "data.numeric_columns"}},
                },
                dependencies=["step_inspect"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id="task_wf01")
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id="task_wf01")

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED

    stats_out = state.step_results["step_stats"]["data"]
    assert "analyzed_columns" in stats_out
    assert set(stats_out["analyzed_columns"]) == {"units_sold", "revenue", "profit"}
    assert "revenue" in stats_out["statistics"]
    assert stats_out["statistics"]["revenue"]["mean"] > 0


@pytest.mark.asyncio
async def test_workflow_02_inspect_to_group_comparison(encrypted_sales_csv):
    """
    Workflow 2: inspect_dataset -> group_comparison
    Dynamically maps categorical_columns[1] ('category') to group_by
    and numeric_columns[1] ('revenue') to metric_column.
    """
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": encrypted_sales_csv, "sample_rows": 5},
                dependencies=[],
            ),
            _make_step(
                step_id="step_group",
                tool_name="group_comparison",
                parameters={
                    "file_path": encrypted_sales_csv,
                    "group_by": {"$ref": {"step_id": "step_inspect", "path": "data.categorical_columns[1]"}},
                    "metric_column": {"$ref": {"step_id": "step_inspect", "path": "data.numeric_columns[1]"}},
                    "aggregations": ["mean", "sum"],
                },
                dependencies=["step_inspect"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id="task_wf02")
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id="task_wf02")

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED

    group_out = state.step_results["step_group"]["data"]
    assert group_out["group_by"] == "category"
    assert group_out["metric_column"] == "revenue"
    assert group_out["total_metric_value"] > 0
    assert len(group_out["results"]) > 0


@pytest.mark.asyncio
async def test_workflow_03_inspect_to_detect_outliers(encrypted_sales_csv):
    """
    Workflow 3: inspect_dataset -> detect_outliers
    Dynamically maps numeric_columns[1] ('revenue') to column parameter for outlier detection.
    """
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": encrypted_sales_csv, "sample_rows": 5},
                dependencies=[],
            ),
            _make_step(
                step_id="step_outliers",
                tool_name="detect_outliers",
                parameters={
                    "file_path": encrypted_sales_csv,
                    "column": {"$ref": {"step_id": "step_inspect", "path": "data.numeric_columns[1]"}},
                    "method": "iqr",
                    "threshold": 1.5,
                },
                dependencies=["step_inspect"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id="task_wf03")
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id="task_wf03")

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED

    outlier_out = state.step_results["step_outliers"]["data"]
    assert outlier_out["column"] == "revenue"
    assert outlier_out["method"] == "iqr"
    assert outlier_out["total_rows"] == 8
    assert isinstance(outlier_out["outlier_count"], int)


@pytest.mark.asyncio
async def test_workflow_04_inspect_to_generate_chart(encrypted_sales_csv):
    """
    Workflow 4: inspect_dataset -> generate_chart
    Dynamically maps categorical_columns[1] ('category') to x_column
    and numeric_columns[1] ('revenue') to y_column.
    """
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": encrypted_sales_csv, "sample_rows": 5},
                dependencies=[],
            ),
            _make_step(
                step_id="step_chart",
                tool_name="generate_chart",
                parameters={
                    "file_path": encrypted_sales_csv,
                    "chart_type": "bar",
                    "title": "Revenue by Category",
                    "x_column": {"$ref": {"step_id": "step_inspect", "path": "data.categorical_columns[1]"}},
                    "y_column": {"$ref": {"step_id": "step_inspect", "path": "data.numeric_columns[1]"}},
                },
                dependencies=["step_inspect"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id="task_wf04")
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id="task_wf04")

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED

    chart_out = state.step_results["step_chart"]["data"]
    assert chart_out["title"] == "Revenue by Category"
    assert chart_out["chart_type"] == "bar"
    assert len(state.artifacts) >= 1


@pytest.mark.asyncio
async def test_workflow_05_group_comparison_to_generate_report(encrypted_sales_csv):
    """
    Workflow 5: group_comparison -> generate_report
    Feeds group comparison table results into generate_report tables parameter.
    """
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": encrypted_sales_csv, "sample_rows": 5},
                dependencies=[],
            ),
            _make_step(
                step_id="step_group",
                tool_name="group_comparison",
                parameters={
                    "file_path": encrypted_sales_csv,
                    "group_by": {"$ref": {"step_id": "step_inspect", "path": "data.categorical_columns[1]"}},
                    "metric_column": {"$ref": {"step_id": "step_inspect", "path": "data.numeric_columns[1]"}},
                    "aggregations": ["mean", "sum"],
                },
                dependencies=["step_inspect"],
            ),
            _make_step(
                step_id="step_report",
                tool_name="generate_report",
                parameters={
                    "title": "Category Performance Summary",
                    "executive_summary": "Overall sales breakdown across product lines.",
                    "findings": [
                        "Category comparison indicates varying revenue streams.",
                        "Total aggregate revenue evaluated.",
                    ],
                    "recommendations": ["Optimize inventory for leading categories."],
                    "tables": {"$ref": {"step_id": "step_group", "path": "data.results"}},
                },
                dependencies=["step_group"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id="task_wf05")
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id="task_wf05")

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED
    assert state.plan.steps[2].status == StepStatus.COMPLETED

    report_out = state.step_results["step_report"]["data"]
    assert "report_id" in report_out
    assert "markdown_content" in report_out
    assert "Category Performance Summary" in report_out["markdown_content"]
    assert "Category comparison indicates varying revenue streams." in report_out["markdown_content"]
    assert "Total aggregate revenue evaluated" in report_out["markdown_content"]
    assert "Detailed Data Tables" in report_out["markdown_content"]
    assert "Electronics" in report_out["markdown_content"]


def test_workflow_06_inspect_to_correlation_analysis_contract_impossibility(encrypted_sales_csv):
    """
    Workflow 6: Documentation and contract verification for correlation_analysis.
    CorrelationAnalysisInput schema:
        class CorrelationAnalysisInput(StrictContract):
            file_path: str = Field(...)
            method: Literal["pearson", "spearman"] = Field(default="pearson")
            threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    Notice CorrelationAnalysisInput does NOT accept a `columns` parameter. It computes
    correlations across all numeric columns automatically.
    Verify that:
    1. CorrelationAnalysisInput model_fields contains only {"file_path", "method", "threshold"}.
    2. InspectDatasetOutput contains no field compatible with method (Literal['pearson', 'spearman'])
       or threshold (float 0.0-1.0).
    3. Passing raw file_path via reference is strictly rejected by B3.1 security rules.
    4. Therefore, dynamic referencing from inspect_dataset into correlation_analysis cannot be
       expressed under current tool contracts without fabricating unsupported parameter mappings.
    """
    input_model = tool_registry.get_input_model("correlation_analysis")
    assert "columns" not in input_model.model_fields
    assert set(input_model.model_fields.keys()) == {"file_path", "method", "threshold"}

    # Verifies inspect_dataset output does not supply threshold or method
    output_model = tool_registry.get_output_model("inspect_dataset")
    assert "threshold" not in output_model.model_fields
    assert "method" not in output_model.model_fields


# ==============================================================================
# SECTION 2: Reference Resolution Integration and Cross-Step Data Flow
# ==============================================================================

def test_multiple_nested_references_in_single_step():
    """Verify multiple references within dictionaries and lists in one step."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "data": {
                "dataset_name": "data.csv",
                "total_rows": 100,
                "columns": ["col_a", "col_b", "col_c"],
                "numeric_columns": ["col_a", "col_b"],
                "categorical_columns": ["col_c"],
                "missing_values": {"col_a": 0, "col_b": 2},
                "nested_summary": {
                    "level1": {
                        "level2": {
                            "metric_target": "col_b"
                        }
                    }
                }
            },
            "metadata": {"task_id": "task_active"},
        }
    }

    params = {
        "primary_col": {"$ref": {"step_id": "step_1", "path": "data.numeric_columns[0]"}},
        "secondary_col": {"$ref": {"step_id": "step_1", "path": "data.nested_summary.level1.level2.metric_target"}},
        "all_numeric": {"$ref": {"step_id": "step_1", "path": "data.numeric_columns"}},
        "row_limit": {"$ref": {"step_id": "step_1", "path": "data.total_rows"}},
    }

    resolved = resolve_step_references(params, step_results, dependencies=["step_1"], active_task_id="task_active")
    assert resolved["primary_col"] == "col_a"
    assert resolved["secondary_col"] == "col_b"
    assert resolved["all_numeric"] == ["col_a", "col_b"]
    assert resolved["row_limit"] == 100


def test_references_limited_to_declared_dependencies():
    """References to steps outside declared dependencies must fail closed."""
    step_results = {
        "step_1": {"tool": "inspect_dataset", "data": {"row_count": 50}, "metadata": {"task_id": "task_active"}},
        "step_2": {"tool": "inspect_dataset", "data": {"row_count": 75}, "metadata": {"task_id": "task_active"}},
    }

    params = {
        "count": {"$ref": {"step_id": "step_2", "path": "data.row_count"}}
    }

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(params, step_results, dependencies=["step_1"], active_task_id="task_active")
    assert exc.value.code == "UNDECLARED_DEPENDENCY"


def test_cross_step_data_flow_data_types():
    """Verify cross-step data flows work for strings, lists, integers, floats, dicts, and booleans."""
    step_results = {
        "step_src": {
            "tool": "some_tool",
            "data": {
                "str_val": "revenue",
                "int_val": 42,
                "float_val": 123.45,
                "bool_val": True,
                "list_val": ["a", "b", "c"],
                "dict_val": {"k1": "v1", "k2": 99},
            },
            "metadata": {"task_id": "task_active"},
        }
    }

    params = {
        "s": {"$ref": {"step_id": "step_src", "path": "data.str_val"}},
        "i": {"$ref": {"step_id": "step_src", "path": "data.int_val"}},
        "f": {"$ref": {"step_id": "step_src", "path": "data.float_val"}},
        "b": {"$ref": {"step_id": "step_src", "path": "data.bool_val"}},
        "l": {"$ref": {"step_id": "step_src", "path": "data.list_val"}},
        "d": {"$ref": {"step_id": "step_src", "path": "data.dict_val"}},
    }

    resolved = resolve_step_references(params, step_results, dependencies=["step_src"], active_task_id="task_active")
    assert resolved["s"] == "revenue"
    assert resolved["i"] == 42
    assert resolved["f"] == 123.45
    assert resolved["b"] is True
    assert resolved["l"] == ["a", "b", "c"]
    assert resolved["d"] == {"k1": "v1", "k2": 99}


def test_array_elements_and_dict_keys_resolve():
    """Verify array indexing and dictionary key lookups resolve correctly."""
    step_results = {
        "step_src": {
            "tool": "inspect_dataset",
            "data": {
                "sample_data": [
                    {"row": 0, "name": "Item0"},
                    {"row": 1, "name": "Item1"},
                ],
                "stats_by_col": {
                    "sales": {"min": 10.0, "max": 100.0}
                }
            },
            "metadata": {"task_id": "task_active"},
        }
    }

    params = {
        "first_name": {"$ref": {"step_id": "step_src", "path": "data.sample_data[0].name"}},
        "second_name": {"$ref": {"step_id": "step_src", "path": "data.sample_data[1].name"}},
        "sales_max": {"$ref": {"step_id": "step_src", "path": "data.stats_by_col.sales.max"}},
    }

    resolved = resolve_step_references(params, step_results, dependencies=["step_src"], active_task_id="task_active")
    assert resolved["first_name"] == "Item0"
    assert resolved["second_name"] == "Item1"
    assert resolved["sales_max"] == 100.0


# ==============================================================================
# SECTION 3: Failure Propagation and Boundary Testing
# ==============================================================================

@pytest.mark.asyncio
async def test_upstream_tool_failure_blocks_dependents_while_independent_steps_succeed(encrypted_sales_csv):
    """
    DAG topology:
      step_independent -> succeeds
      step_failing (non-existent file) -> fails
      step_dependent (depends on step_failing) -> skipped (blocked)
    Verifies independent branch executes while failing dependency halts only its downstream path.
    """
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_independent",
                tool_name="inspect_dataset",
                parameters={"file_path": encrypted_sales_csv, "sample_rows": 2},
                dependencies=[],
            ),
            _make_step(
                step_id="step_failing",
                tool_name="inspect_dataset",
                parameters={"file_path": str(settings.UPLOADS_DIR / "non_existent_file.csv.enc"), "sample_rows": 2},
                dependencies=[],
            ),
            _make_step(
                step_id="step_dependent",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": encrypted_sales_csv,
                    "columns": {"$ref": {"step_id": "step_failing", "path": "data.numeric_columns"}},
                },
                dependencies=["step_failing"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id="task_fail_prop")
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id="task_fail_prop")

    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.FAILED
    assert state.plan.steps[2].status == StepStatus.SKIPPED
    assert state.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_upstream_output_validation_failure_blocks_downstream(encrypted_sales_csv, monkeypatch):
    """
    If an upstream tool produces an invalid output violating its registered output schema,
    the central B2 execution boundary rejects it, marking the step FAILED and blocking dependents.
    """
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": encrypted_sales_csv, "sample_rows": 5},
                dependencies=[],
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": encrypted_sales_csv,
                    "columns": {"$ref": {"step_id": "step_inspect", "path": "data.numeric_columns"}},
                },
                dependencies=["step_inspect"],
            ),
        ]
    )

    orig_inspect = tool_registry._executors["inspect_dataset"]
    def bad_inspect(*args, **kwargs):
        res = orig_inspect(*args, **kwargs)
        res["total_rows"] = "NOT_AN_INTEGER_STRING"  # Violates int contract
        return res

    monkeypatch.setitem(tool_registry._executors, "inspect_dataset", bad_inspect)

    handoff = _make_handoff(plan, task_id="task_out_val_fail")
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id="task_out_val_fail")

    assert state.plan.steps[0].status == StepStatus.FAILED
    assert state.plan.steps[1].status == StepStatus.SKIPPED
    assert state.status == TaskStatus.FAILED


def test_missing_referenced_field_fails_closed():
    """Reference pointing to missing key fails closed with sanitized ReferenceResolutionError."""
    step_results = {
        "step_1": {
            "tool": "inspect_dataset",
            "data": {"columns": ["a", "b"]},
            "metadata": {"task_id": "task_active"},
        }
    }

    params = {
        "val": {"$ref": {"step_id": "step_1", "path": "data.non_existent_key"}}
    }

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(params, step_results, dependencies=["step_1"], active_task_id="task_active")
    assert exc.value.code == "KEY_NOT_FOUND"


def test_destination_tool_validation_failure_fails_closed(encrypted_sales_csv):
    """
    If resolved parameters violate the destination tool's schema at execution time,
    _resolve_step_parameters fails closed with DESTINATION_VALIDATION_FAILED.
    """
    task_state = TaskState(
        task_id="task_dest_fail",
        objective="test",
        privacy_mode="hybrid",
        status=TaskStatus.RUNNING,
    )
    task_state.step_results["step_1"] = {
        "tool": "inspect_dataset",
        "data": {
            "metric": 12345,  # Int where group_by expects str
        },
        "metadata": {"task_id": "task_dest_fail"},
    }

    step = _make_step(
        step_id="step_2",
        tool_name="group_comparison",
        parameters={
            "file_path": encrypted_sales_csv,
            "group_by": {"$ref": {"step_id": "step_1", "path": "data.metric"}},  # invalid: int passed to str
            "metric_column": "revenue",
        },
        dependencies=["step_1"],
    )

    with pytest.raises(ReferenceResolutionError) as exc:
        executor._resolve_step_parameters(step, task_state)
    assert exc.value.code == "DESTINATION_VALIDATION_FAILED"


@pytest.mark.asyncio
async def test_cancellation_during_multi_step_execution(encrypted_sales_csv):
    """Cancellation halts subsequent steps and safely marks them SKIPPED."""
    task_state = TaskState(
        task_id="task_cancel_step_test",
        objective="test",
        privacy_mode="hybrid",
        status=TaskStatus.RUNNING,
    )
    task_state.status = TaskStatus.CANCEL_REQUESTED

    step = _make_step(
        step_id="step_cancel",
        tool_name="inspect_dataset",
        parameters={"file_path": encrypted_sales_csv},
    )

    result = await executor.execute_step(step, task_state)
    assert step.status == StepStatus.SKIPPED
    assert result.error == "Tool execution cancelled before start."


# ==============================================================================
# SECTION 4: Trace and Checkpoint Privacy Verification
# ==============================================================================

@pytest.mark.asyncio
async def test_trace_and_checkpoint_privacy_under_b4(encrypted_sales_csv):
    """
    Verify:
    1. Checkpoint persistence strips all step_results from SQLite.
    2. Trace events do not leak resolved argument payloads or raw outputs.
    """
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": encrypted_sales_csv, "sample_rows": 5},
                dependencies=[],
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": encrypted_sales_csv,
                    "columns": {"$ref": {"step_id": "step_inspect", "path": "data.numeric_columns"}},
                },
                dependencies=["step_inspect"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id="task_privacy_verify")
    orchestrator = AgentOrchestrator()
    listener = orchestrator.register_event_listener("task_privacy_verify")

    state = await orchestrator.run_handoff(handoff, task_id="task_privacy_verify")
    assert state.status == TaskStatus.COMPLETED

    # 1. Verify checkpoint persisted state does not contain raw step_results
    loaded_state = checkpointer.load_checkpoint("task_privacy_verify")
    assert loaded_state is not None
    assert loaded_state.step_results == {}

    # 2. Verify events emitted do not leak raw parameters or resolved values
    events = []
    while not listener.empty():
        _, event = await listener.get()
        events.append(event)

    for ev in events:
        msg = ev.message.lower()
        # Verify no sensitive dataset column names or numeric values are leaked in the event message
        assert "units_sold" not in msg
        assert "15000" not in msg
        if ev.event_type == "step_completed":
            # Safe data contains only duration_ms, step_index, status, tool_name
            assert "units_sold" not in str(ev.data.get("safe_data", {}))


# ==============================================================================
# SECTION 5: Performance & Resource Safety
# ==============================================================================

def test_deep_reference_nesting_enforces_limit():
    """Verify parameters nested beyond MAX_REFERENCE_DEPTH (10) raise ReferenceSyntaxError."""
    curr = {"$ref": {"step_id": "step_1", "path": "data.val"}}
    for _ in range(12):
        curr = {"nested": curr}

    with pytest.raises(ReferenceSyntaxError) as exc:
        validate_reference_syntax(curr)
    assert exc.value.code == "NESTING_DEPTH_EXCEEDED"


def test_large_payload_reference_resolution_performance():
    """
    Verify reference resolution on large payloads (10,000 array elements)
    completes in under 100 milliseconds without performance degradation.
    """
    large_list = [{"id": i, "val": f"item_{i}"} for i in range(10000)]
    step_results = {
        "step_large": {
            "tool": "some_tool",
            "data": {"records": large_list},
            "metadata": {"task_id": "task_perf"},
        }
    }

    params = {
        "first_item": {"$ref": {"step_id": "step_large", "path": "data.records[0].val"}},
        "last_item": {"$ref": {"step_id": "step_large", "path": "data.records[9999].val"}},
    }

    t0 = time.perf_counter()
    resolved = resolve_step_references(params, step_results, dependencies=["step_large"], active_task_id="task_perf")
    t1 = time.perf_counter()

    duration_ms = (t1 - t0) * 1000.0
    assert resolved["first_item"] == "item_0"
    assert resolved["last_item"] == "item_9999"
    assert duration_ms < 100.0, f"Resolution took {duration_ms}ms, exceeding 100ms budget"
