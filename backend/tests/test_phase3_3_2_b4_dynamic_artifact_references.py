"""
Phase 3.3.2-B4: Dynamic Artifact Reference Integration Test Suite.

Verifies end-to-end integration between transform_dataset and downstream analytics tools
via validated dynamic $ref references, strict task isolation, manifest boundaries,
in-memory encryption/decryption, and fail-closed security invariants.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.execution_handoff import (
    ExecutionHandoff,
    _HANDOFF_FACTORY_TOKEN,
    _intent_digest,
    _plan_digest,
)
from app.agent.executor import PlanExecutor, executor
from app.agent.intelligence_contracts import (
    EvaluationStatus,
    IntentCategory,
    ProviderIdentity,
)
from app.agent.orchestrator import AgentOrchestrator
from app.agent.plan_validation import PlanValidationError, validate_execution_plan
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus, TaskState, TaskStatus
from app.agent.step_references import (
    ReferenceResolutionError,
    ReferenceTypeError,
    resolve_step_references,
)
from app.core.config import settings
from app.privacy.encryption import encryption_service
from app.tools.registry import tool_registry


def _make_step(
    step_id: str,
    tool_name: str,
    parameters: dict[str, Any],
    dependencies: list[str] | None = None,
    title: str = "Test Step",
    description: str = "Test Step Description",
) -> ExecutionStep:
    return ExecutionStep(
        step_id=step_id,
        title=title,
        description=description,
        capability="system",
        tool_name=tool_name,
        parameters=parameters,
        dependencies=dependencies or [],
    )


def _make_plan(
    steps: list[ExecutionStep],
    plan_id: str = "plan_b4_test",
    goal: str = "Test Dynamic Artifact References",
) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        goal=goal,
        steps=steps,
    )


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


@pytest.fixture
def sample_sales_encrypted_csv():
    """Create a sample encrypted CSV in settings.UPLOADS_DIR."""
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    settings.TRANSFORMS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = settings.UPLOADS_DIR / "sample_sales_b4_source.csv.enc"
    df = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "category": ["Electronics", "Electronics", "Clothing", "Clothing", "Home", "Home", "Accessories", "Electronics"],
        "region": ["North", "South", "North", "West", "East", "North", "South", "East"],
        "units_sold": [120, 150, 40, 60, 20, 15, 5, 210],
        "revenue": [12000.0, 15000.0, 2400.0, 3600.0, 1200.0, 900.0, 250.0, 21000.0],
        "profit": [3600.0, 4500.0, 720.0, 1080.0, 240.0, 180.0, 50.0, 6300.0],
    })
    raw_csv = df.to_csv(index=False).encode("utf-8")
    enc_bytes = encryption_service().encrypt_bytes(raw_csv)
    file_path.write_bytes(enc_bytes)

    yield str(file_path)

    if file_path.exists():
        file_path.unlink()
    # Cleanup any transformed artifacts produced during tests
    for f in settings.TRANSFORMS_DIR.glob("transformed_*.csv.enc"):
        try:
            f.unlink()
        except OSError:
            pass


# ==============================================================================
# SECTION 1: End-to-End Multi-Step Workflows
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_01_transform_to_descriptive_statistics(sample_sales_encrypted_csv):
    """
    transform_dataset -> descriptive_statistics
    Filters revenue > 5000; downstream computes statistics on the filtered artifact.
    """
    task_id = "task_b4_e2e_01"
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_transform",
                tool_name="transform_dataset",
                parameters={
                    "file_path": sample_sales_encrypted_csv,
                    "operations": [
                        {
                            "op": "filter_rows",
                            "column": "revenue",
                            "operator": "gt",
                            "value": 5000.0,
                        }
                    ],
                },
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_transform", "path": "artifacts[0]"}},
                    "columns": ["revenue"],
                },
                dependencies=["step_transform"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id=task_id)
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id=task_id)

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED

    transform_out = state.step_results["step_transform"]["data"]
    assert transform_out["initial_row_count"] == 8
    assert transform_out["final_row_count"] == 3  # Only rows with revenue > 5000: 12000, 15000, 21000

    stats_out = state.step_results["step_stats"]["data"]
    rev_stats = stats_out["statistics"]["revenue"]
    assert rev_stats["count"] == 3
    assert rev_stats["min"] == 12000.0
    assert rev_stats["max"] == 21000.0
    assert rev_stats["mean"] == 16000.0

    # Verify the artifact was written encrypted to disk
    art_path = Path(state.artifacts[0])
    assert art_path.is_file()
    assert art_path.name.endswith(".csv.enc")


@pytest.mark.asyncio
async def test_workflow_02_transform_to_correlation_analysis(sample_sales_encrypted_csv):
    """
    transform_dataset -> correlation_analysis
    References transformed artifact via artifacts[0].relative_path.
    """
    task_id = "task_b4_e2e_02"
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_transform",
                tool_name="transform_dataset",
                parameters={
                    "file_path": sample_sales_encrypted_csv,
                    "operations": [
                        {
                            "op": "select_columns",
                            "columns": ["units_sold", "revenue", "profit"],
                        }
                    ],
                },
            ),
            _make_step(
                step_id="step_corr",
                tool_name="correlation_analysis",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_transform", "path": "artifacts[0].relative_path"}},
                    "columns": ["units_sold", "revenue", "profit"],
                },
                dependencies=["step_transform"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id=task_id)
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id=task_id)

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED

    corr_out = state.step_results["step_corr"]["data"]
    assert "correlation_matrix" in corr_out
    assert "units_sold" in corr_out["analyzed_columns"]
    assert "revenue" in corr_out["analyzed_columns"]


@pytest.mark.asyncio
async def test_workflow_03_transform_to_detect_outliers(sample_sales_encrypted_csv):
    """
    transform_dataset -> detect_outliers
    References transformed artifact via artifacts[0].artifact_id.
    """
    task_id = "task_b4_e2e_03"
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_transform",
                tool_name="transform_dataset",
                parameters={
                    "file_path": sample_sales_encrypted_csv,
                    "operations": [
                        {
                            "op": "filter_rows",
                            "column": "category",
                            "operator": "eq",
                            "value": "Electronics",
                        }
                    ],
                },
            ),
            _make_step(
                step_id="step_outliers",
                tool_name="detect_outliers",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_transform", "path": "artifacts[0].artifact_id"}},
                    "column": "units_sold",
                    "method": "iqr",
                },
                dependencies=["step_transform"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id=task_id)
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id=task_id)

    assert state.status == TaskStatus.COMPLETED
    outlier_out = state.step_results["step_outliers"]["data"]
    assert outlier_out["column"] == "units_sold"
    assert outlier_out["total_rows"] == 3


@pytest.mark.asyncio
async def test_workflow_04_transform_to_generate_chart(sample_sales_encrypted_csv):
    """
    transform_dataset -> generate_chart
    Generates a chart directly from the transformed dataset.
    """
    task_id = "task_b4_e2e_04"
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_transform",
                tool_name="transform_dataset",
                parameters={
                    "file_path": sample_sales_encrypted_csv,
                    "operations": [
                        {
                            "op": "filter_rows",
                            "column": "profit",
                            "operator": "gt",
                            "value": 500.0,
                        }
                    ],
                },
            ),
            _make_step(
                step_id="step_chart",
                tool_name="generate_chart",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_transform", "path": "artifacts[0]"}},
                    "chart_type": "bar",
                    "x_column": "category",
                    "y_column": "profit",
                    "title": "Profit by Category for High Margin Items",
                },
                dependencies=["step_transform"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id=task_id)
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id=task_id)

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED
    assert len(state.artifacts) >= 2  # transformed dataset + chart


@pytest.mark.asyncio
async def test_workflow_05_transform_to_group_comparison(sample_sales_encrypted_csv):
    """
    transform_dataset -> group_comparison
    Aggregates metrics by category on the transformed dataset.
    """
    task_id = "task_b4_e2e_05"
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_transform",
                tool_name="transform_dataset",
                parameters={
                    "file_path": sample_sales_encrypted_csv,
                    "operations": [
                        {
                            "op": "filter_rows",
                            "column": "region",
                            "operator": "in",
                            "value": ["North", "South"],
                        }
                    ],
                },
            ),
            _make_step(
                step_id="step_group",
                tool_name="group_comparison",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_transform", "path": "artifacts[0]"}},
                    "group_by": "category",
                    "metric_column": "revenue",
                },
                dependencies=["step_transform"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id=task_id)
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id=task_id)

    assert state.status == TaskStatus.COMPLETED
    group_out = state.step_results["step_group"]["data"]
    assert group_out["group_by"] == "category"
    assert group_out["metric_column"] == "revenue"
    assert group_out["group_count"] > 0


@pytest.mark.asyncio
async def test_workflow_06_chained_transformations(sample_sales_encrypted_csv):
    """
    transform_dataset -> transform_dataset -> descriptive_statistics
    Chains two sequential transformation steps before downstream analytics.
    """
    task_id = "task_b4_e2e_06"
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_filter",
                tool_name="transform_dataset",
                parameters={
                    "file_path": sample_sales_encrypted_csv,
                    "operations": [
                        {
                            "op": "filter_rows",
                            "column": "units_sold",
                            "operator": "gte",
                            "value": 50,
                        }
                    ],
                },
            ),
            _make_step(
                step_id="step_rename",
                tool_name="transform_dataset",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_filter", "path": "artifacts[0]"}},
                    "operations": [
                        {
                            "op": "rename_columns",
                            "mapping": {"revenue": "gross_revenue"},
                        }
                    ],
                },
                dependencies=["step_filter"],
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_rename", "path": "artifacts[0]"}},
                    "columns": ["gross_revenue"],
                },
                dependencies=["step_rename"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id=task_id)
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id=task_id)

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED
    assert state.plan.steps[2].status == StepStatus.COMPLETED

    step1_out = state.step_results["step_filter"]["data"]
    step2_out = state.step_results["step_rename"]["data"]
    step3_out = state.step_results["step_stats"]["data"]

    assert step1_out["final_row_count"] == 4  # units >= 50: 120, 150, 60, 210
    assert "gross_revenue" in step2_out["columns"]
    assert "revenue" not in step2_out["columns"]
    assert "gross_revenue" in step3_out["statistics"]


@pytest.mark.asyncio
async def test_workflow_07_transform_with_column_data_reference(sample_sales_encrypted_csv):
    """
    transform_dataset -> descriptive_statistics
    Downstream references both file_path from artifacts[0] and columns from data.columns.
    """
    task_id = "task_b4_e2e_07"
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_select",
                tool_name="transform_dataset",
                parameters={
                    "file_path": sample_sales_encrypted_csv,
                    "operations": [
                        {
                            "op": "select_columns",
                            "columns": ["revenue", "profit"],
                        }
                    ],
                },
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_select", "path": "artifacts[0]"}},
                    "columns": {"$ref": {"step_id": "step_select", "path": "data.columns"}},
                },
                dependencies=["step_select"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id=task_id)
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id=task_id)

    assert state.status == TaskStatus.COMPLETED
    stats_out = state.step_results["step_stats"]["data"]
    assert sorted(stats_out["analyzed_columns"]) == ["profit", "revenue"]


# ==============================================================================
# SECTION 2: Security & Isolation Boundaries
# ==============================================================================

def test_cross_task_artifact_access_fails_closed():
    """Referencing an artifact belonging to another task is strictly rejected."""
    foreign_task_id = "task_foreign_999"
    active_task_id = "task_current_111"
    foreign_hash = hashlib.sha256(foreign_task_id.encode("utf-8")).hexdigest()[:10]

    manifest = [
        {
            "artifact_id": "transformed_abcdef1234",
            "task_id": foreign_task_id,
            "artifact_type": "transformed_dataset",
            "relative_path": f"transformed_{foreign_hash}_abcdef1234.csv.enc",
        }
    ]
    step_results = {
        "step_tx": {
            "tool": "transform_dataset",
            "artifacts": [f"transformed_{foreign_hash}_abcdef1234.csv.enc"],
            "data": {},
            "metadata": {"task_id": active_task_id},
        }
    }
    params = {
        "file_path": {"$ref": {"step_id": "step_tx", "path": "artifacts[0]"}}
    }

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_tx"],
            active_task_id=active_task_id,
            task_manifest=manifest,
        )
    assert exc.value.code == "CROSS_TASK_ARTIFACT_ACCESS"


def test_cross_task_filename_hash_mismatch_fails_closed():
    """An artifact claiming active_task_id but with a mismatched filename hash fails closed."""
    active_task_id = "task_current_222"
    mismatched_hash = "0000000000"

    manifest = [
        {
            "artifact_id": "transformed_1122334455",
            "task_id": active_task_id,
            "artifact_type": "transformed_dataset",
            "relative_path": f"transformed_{mismatched_hash}_1122334455.csv.enc",
        }
    ]
    step_results = {
        "step_tx": {
            "tool": "transform_dataset",
            "artifacts": [f"transformed_{mismatched_hash}_1122334455.csv.enc"],
            "data": {},
            "metadata": {"task_id": active_task_id},
        }
    }
    params = {
        "file_path": {"$ref": {"step_id": "step_tx", "path": "artifacts[0]"}}
    }

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_tx"],
            active_task_id=active_task_id,
            task_manifest=manifest,
        )
    assert exc.value.code == "CROSS_TASK_ARTIFACT_ACCESS"


def test_unregistered_artifact_fails_closed():
    """Referencing an artifact that is not registered in the task manifest fails closed."""
    active_task_id = "task_current_333"
    task_hash = hashlib.sha256(active_task_id.encode("utf-8")).hexdigest()[:10]

    manifest: list[dict[str, str]] = []  # empty manifest
    step_results = {
        "step_tx": {
            "tool": "transform_dataset",
            "artifacts": [f"transformed_{task_hash}_abcdef1111.csv.enc"],
            "data": {},
            "metadata": {"task_id": active_task_id},
        }
    }
    params = {
        "file_path": {"$ref": {"step_id": "step_tx", "path": "artifacts[0]"}}
    }

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_tx"],
            active_task_id=active_task_id,
            task_manifest=manifest,
        )
    # Fails either in descriptor generation or in canonical path resolution
    assert exc.value.code in {"CROSS_TASK_ARTIFACT_ACCESS", "UNREGISTERED_ARTIFACT"}


def test_missing_manifest_fails_closed():
    """Attempting to resolve file_path with task_manifest=None fails closed."""
    active_task_id = "task_current_444"
    task_hash = hashlib.sha256(active_task_id.encode("utf-8")).hexdigest()[:10]

    step_results = {
        "step_tx": {
            "tool": "transform_dataset",
            "artifacts": [f"transformed_{task_hash}_abcdef2222.csv.enc"],
            "data": {},
            "metadata": {"task_id": active_task_id},
        }
    }
    params = {
        "file_path": {"$ref": {"step_id": "step_tx", "path": "artifacts[0]"}}
    }

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_tx"],
            active_task_id=active_task_id,
            task_manifest=None,
        )
    assert exc.value.code == "UNREGISTERED_ARTIFACT"


def test_wrong_artifact_type_rejected_at_plan_validation():
    """Plan validation rejects file_path referencing non-dataset artifacts (e.g. from generate_chart)."""
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_chart",
                tool_name="generate_chart",
                parameters={
                    "file_path": "uploads/sales.csv",
                    "chart_type": "bar",
                    "x_column": "cat",
                    "y_column": "rev",
                    "title": "Chart",
                },
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_chart", "path": "artifacts[0]"}},
                },
                dependencies=["step_chart"],
            ),
        ]
    )

    with pytest.raises(PlanValidationError) as exc:
        validate_execution_plan(plan)
    assert exc.value.code == "type_incompatibility"
    assert "Tool 'generate_chart' produces non-dataset artifacts" in exc.value.message


def test_wrong_artifact_type_rejected_at_runtime():
    """Runtime resolution rejects a manifest entry whose artifact_type is not transformed_dataset."""
    active_task_id = "task_current_555"

    manifest = [
        {
            "artifact_id": "chart_aabbccdd11",
            "task_id": active_task_id,
            "artifact_type": "chart_png",
            "relative_path": "chart_aabbccdd11.png.enc",
        }
    ]
    step_results = {
        "step_tx": {
            "tool": "transform_dataset",
            "artifacts": ["chart_aabbccdd11.png.enc"],
            "data": {},
            "metadata": {"task_id": active_task_id},
        }
    }
    params = {
        "file_path": {"$ref": {"step_id": "step_tx", "path": "artifacts[0]"}}
    }

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_tx"],
            active_task_id=active_task_id,
            task_manifest=manifest,
        )
    assert exc.value.code == "INVALID_ARTIFACT_TYPE"


def test_deleted_artifact_file_fails_closed():
    """An artifact present in the manifest but deleted from disk fails closed."""
    active_task_id = "task_current_666"
    task_hash = hashlib.sha256(active_task_id.encode("utf-8")).hexdigest()[:10]
    filename = f"transformed_{task_hash}_9988776655.csv.enc"

    manifest = [
        {
            "artifact_id": "transformed_9988776655",
            "task_id": active_task_id,
            "artifact_type": "transformed_dataset",
            "relative_path": filename,
        }
    ]
    step_results = {
        "step_tx": {
            "tool": "transform_dataset",
            "artifacts": [filename],
            "data": {},
            "metadata": {"task_id": active_task_id},
        }
    }
    params = {
        "file_path": {"$ref": {"step_id": "step_tx", "path": "artifacts[0]"}}
    }

    # Ensure file definitely does NOT exist on disk
    target_file = settings.TRANSFORMS_DIR / filename
    if target_file.exists():
        target_file.unlink()

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_tx"],
            active_task_id=active_task_id,
            task_manifest=manifest,
        )
    assert exc.value.code == "ARTIFACT_FILE_NOT_FOUND"


def test_path_containment_violation_fails_closed():
    """Manifest entry containing directory traversal fails closed."""
    active_task_id = "task_current_777"

    manifest = [
        {
            "artifact_id": "transformed_traversal1",
            "task_id": active_task_id,
            "artifact_type": "transformed_dataset",
            "relative_path": "../uploads/secret.csv.enc",
        }
    ]
    step_results = {
        "step_tx": {
            "tool": "transform_dataset",
            "artifacts": ["../uploads/secret.csv.enc"],
            "data": {},
            "metadata": {"task_id": active_task_id},
        }
    }
    params = {
        "file_path": {"$ref": {"step_id": "step_tx", "path": "artifacts[0]"}}
    }

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_step_references(
            params,
            step_results,
            dependencies=["step_tx"],
            active_task_id=active_task_id,
            task_manifest=manifest,
        )
    assert exc.value.code in {"CROSS_TASK_ARTIFACT_ACCESS", "PATH_CONTAINMENT_VIOLATION"}


def test_raw_path_reference_in_file_path_rejected():
    """Plans attempting to reference data.file_path directly are rejected."""
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": "uploads/sales.csv"},
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_inspect", "path": "data.file_path"}},
                },
                dependencies=["step_inspect"],
            ),
        ]
    )

    with pytest.raises(PlanValidationError) as exc:
        validate_execution_plan(plan)
    assert exc.value.code in {"invalid_reference_path", "type_incompatibility"}


def test_data_field_reference_in_file_path_rejected():
    """Plans attempting to reference non-artifact data fields for file_path are rejected."""
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": "uploads/sales.csv"},
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_inspect", "path": "data.columns[0]"}},
                },
                dependencies=["step_inspect"],
            ),
        ]
    )

    with pytest.raises(PlanValidationError) as exc:
        validate_execution_plan(plan)
    assert exc.value.code == "type_incompatibility"
    assert "Parameter 'file_path' can only reference artifacts" in exc.value.message


def test_whole_list_artifact_reference_in_file_path_rejected():
    """Plans attempting to reference the entire artifacts list for scalar file_path are rejected."""
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_tx",
                tool_name="transform_dataset",
                parameters={
                    "file_path": "uploads/sales.csv",
                    "operations": [{"op": "select_columns", "columns": ["revenue"]}],
                },
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_tx", "path": "artifacts"}},
                },
                dependencies=["step_tx"],
            ),
        ]
    )

    with pytest.raises(PlanValidationError) as exc:
        validate_execution_plan(plan)
    assert exc.value.code == "type_incompatibility"


@pytest.mark.asyncio
async def test_failed_prerequisite_aborts_downstream_step(sample_sales_encrypted_csv):
    """When transform_dataset fails, downstream tool reference fails closed and step is not executed."""
    task_id = "task_b4_prereq_fail"
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_transform",
                tool_name="transform_dataset",
                parameters={
                    "file_path": sample_sales_encrypted_csv,
                    "operations": [
                        {
                            "op": "filter_rows",
                            "column": "nonexistent_column",
                            "operator": "eq",
                            "value": 123,
                        }
                    ],
                },
            ),
            _make_step(
                step_id="step_stats",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": {"$ref": {"step_id": "step_transform", "path": "artifacts[0]"}},
                },
                dependencies=["step_transform"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id=task_id)
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id=task_id)

    assert state.status == TaskStatus.FAILED
    assert state.plan.steps[0].status == StepStatus.FAILED
    assert state.plan.steps[1].status in {StepStatus.SKIPPED, StepStatus.PENDING}
    assert "step_stats" not in state.step_results


def test_plan_immutability_preserved():
    """Verifies that PlanExecutor does not mutate the plan step parameters during resolution."""
    raw_params = {
        "file_path": {"$ref": {"step_id": "step_1", "path": "artifacts[0]"}},
        "columns": ["revenue"],
    }
    step = _make_step(
        step_id="step_2",
        tool_name="descriptive_statistics",
        parameters=copy.deepcopy(raw_params),
        dependencies=["step_1"],
    )

    active_task_id = "task_immutability_1"
    task_hash = hashlib.sha256(active_task_id.encode("utf-8")).hexdigest()[:10]
    filename = f"transformed_{task_hash}_abcdef9999.csv.enc"
    file_path = settings.TRANSFORMS_DIR / filename
    file_path.write_bytes(encryption_service().encrypt_bytes(b"col1\nval1\n"))

    try:
        manifest = [
            {
                "artifact_id": "transformed_abcdef9999",
                "task_id": active_task_id,
                "artifact_type": "transformed_dataset",
                "relative_path": filename,
            }
        ]
        step_results = {
            "step_1": {
                "tool": "transform_dataset",
                "artifacts": [str(file_path)],
                "data": {},
                "metadata": {"task_id": active_task_id},
            }
        }
        task_state = TaskState(
            task_id=active_task_id,
            objective="Test immutability",
            artifacts=[str(file_path)],
            step_results=step_results,
        )

        executor_instance = PlanExecutor()
        resolved = executor_instance._resolve_step_parameters(step, task_state)

        # Resolved dict contains the actual canonical path
        assert resolved["file_path"] == str(file_path.resolve())

        # Original step parameters in the plan remain the raw $ref dict unmodified
        assert step.parameters == raw_params
    finally:
        if file_path.exists():
            file_path.unlink()


@pytest.mark.asyncio
async def test_normal_uploaded_file_unaffected(sample_sales_encrypted_csv):
    """Direct uploaded file paths continue to work without $ref or regression."""
    task_id = "task_b4_direct_upload"
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_direct",
                tool_name="descriptive_statistics",
                parameters={
                    "file_path": sample_sales_encrypted_csv,
                    "columns": ["revenue"],
                },
            )
        ]
    )

    handoff = _make_handoff(plan, task_id=task_id)
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id=task_id)

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert "revenue" in state.step_results["step_direct"]["data"]["statistics"]
