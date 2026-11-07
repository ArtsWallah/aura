"""Focused test suite for Phase 3.3.2-B3: Transformation Engine Implementation."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.core.config import settings
from app.privacy.encryption import encryption_service
from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.state import TaskState
from app.tools.registry import tool_registry
from app.tools.schemas import PermissionLevel, ToolCategory
from app.tools.transform_schemas import (
    CastColumns,
    Deduplicate,
    FilterOperator,
    FilterRows,
    HandleMissing,
    MissingStrategy,
    RenameColumns,
    SelectColumns,
    TargetDataType,
    TransformDatasetInput,
    TransformDatasetOutput,
)
from app.tools.transforms.engine import (
    apply_transformations,
    check_dataset_bounds,
    execute_cast_columns,
    execute_deduplicate,
    execute_filter_rows,
    execute_handle_missing,
    execute_rename_columns,
    execute_select_columns,
)
from app.tools.transforms.tool import transform_dataset


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture providing diverse data types, nulls, and duplicates for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 5],
        "name": ["Alice", "Bob", "Charlie", "David", None, None],
        "score": [85.5, 92.0, np.nan, 78.5, 92.0, 92.0],
        "active": ["true", "false", "TRUE", "no", None, None],
        "category": ["A", "B", "A", "C", "B", "B"],
        "joined": ["2023-01-15", "2023-02-20", "2023-03-25", "2023-04-30", "2023-05-01", "2023-05-01"],
    })


@pytest.fixture
def managed_transforms_dir(tmp_path, monkeypatch):
    """Fixture configuring dedicated managed transforms and uploads directories."""
    uploads = tmp_path / "uploads"
    transforms = tmp_path / "transforms"
    uploads.mkdir(parents=True, exist_ok=True)
    transforms.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(settings, "TRANSFORMS_DIR", transforms)
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    return transforms


# ==============================================================================
# 1. Operation-Level Unit Tests
# ==============================================================================

def test_filter_rows_equality_and_nulls(sample_df):
    """Verify FilterRows handles eq, neq, is_null, and not_null with 3VL semantics."""
    # EQ excludes nulls
    op_eq = FilterRows(column="name", operator=FilterOperator.EQ, value="Alice")
    df_eq, details = execute_filter_rows(sample_df, op_eq)
    assert len(df_eq) == 1
    assert df_eq.iloc[0]["name"] == "Alice"
    assert details["rows_removed"] == 5

    # NEQ excludes nulls (SQL 3VL behavior)
    op_neq = FilterRows(column="category", operator=FilterOperator.NEQ, value="A")
    df_neq, _ = execute_filter_rows(sample_df, op_neq)
    assert set(df_neq["category"].unique()) == {"B", "C"}

    # IS_NULL
    op_null = FilterRows(column="score", operator=FilterOperator.IS_NULL)
    df_null, _ = execute_filter_rows(sample_df, op_null)
    assert len(df_null) == 1
    assert df_null.iloc[0]["id"] == 3

    # NOT_NULL
    op_not_null = FilterRows(column="score", operator=FilterOperator.NOT_NULL)
    df_not_null, _ = execute_filter_rows(sample_df, op_not_null)
    assert len(df_not_null) == 5


def test_filter_rows_ordered_and_datetime():
    """Verify FilterRows handles numeric and ISO datetime comparisons."""
    df = pd.DataFrame({
        "age": [20, 25, 30, np.nan],
        "created_at": ["2023-01-01", "2023-06-01", "2023-12-01", None],
    })

    # Numeric GT
    op_gt = FilterRows(column="age", operator=FilterOperator.GT, value=22)
    res_gt, _ = execute_filter_rows(df, op_gt)
    assert len(res_gt) == 2
    assert set(res_gt["age"].tolist()) == {25.0, 30.0}

    # Datetime LTE
    op_dt = FilterRows(column="created_at", operator=FilterOperator.LTE, value="2023-06-01")
    res_dt, _ = execute_filter_rows(df, op_dt)
    assert len(res_dt) == 2
    assert "2023-12-01" not in res_dt["created_at"].values


def test_filter_rows_membership_and_contains():
    """Verify in, not_in, and contains operators."""
    df = pd.DataFrame({
        "fruit": ["apple", "banana", "apricot", "cherry", None],
    })

    # IN
    op_in = FilterRows(column="fruit", operator=FilterOperator.IN, value=["apple", "cherry"])
    res_in, _ = execute_filter_rows(df, op_in)
    assert set(res_in["fruit"].tolist()) == {"apple", "cherry"}

    # NOT_IN
    op_notin = FilterRows(column="fruit", operator=FilterOperator.NOT_IN, value=["apple", "banana"])
    res_notin, _ = execute_filter_rows(df, op_notin)
    assert set(res_notin["fruit"].tolist()) == {"apricot", "cherry"}

    # CONTAINS (case-insensitive)
    op_contains = FilterRows(column="fruit", operator=FilterOperator.CONTAINS, value="AP", case_sensitive=False)
    res_contains, _ = execute_filter_rows(df, op_contains)
    assert set(res_contains["fruit"].tolist()) == {"apple", "apricot"}


def test_filter_rows_missing_column_fails(sample_df):
    """Verify FilterRows fails safely when column does not exist."""
    op = FilterRows(column="nonexistent", operator=FilterOperator.EQ, value=1)
    with pytest.raises(ValueError, match="does not exist in dataset"):
        execute_filter_rows(sample_df, op)


def test_select_columns_keep_and_drop(sample_df):
    """Verify SelectColumns handles keep and drop modes."""
    # Keep mode
    op_keep = SelectColumns(columns=["id", "name"], mode="keep")
    df_keep, details = execute_select_columns(sample_df, op_keep)
    assert list(df_keep.columns) == ["id", "name"]
    assert details["cols_affected"] == 2

    # Drop mode
    op_drop = SelectColumns(columns=["id", "score"], mode="drop")
    df_drop, _ = execute_select_columns(sample_df, op_drop)
    assert "id" not in df_drop.columns
    assert "score" not in df_drop.columns
    assert "name" in df_drop.columns


def test_select_columns_drop_all_fails(sample_df):
    """Verify dropping all columns raises ValueError."""
    op = SelectColumns(columns=list(sample_df.columns), mode="drop")
    with pytest.raises(ValueError, match="Cannot drop all columns"):
        execute_select_columns(sample_df, op)


def test_handle_missing_imputation_semantics():
    """Verify mean, median, mode, and fill_value imputation rules."""
    df = pd.DataFrame({
        "ints": [10, 20, np.nan, 40],
        "floats": [1.5, 2.5, np.nan, 4.0],
        "category": ["apple", "banana", "apple", np.nan],
    })

    # Integer column imputed with mean becomes float64
    op_mean = HandleMissing(columns=["ints"], strategy=MissingStrategy.FILL_MEAN)
    df_mean, _ = execute_handle_missing(df, op_mean)
    assert df_mean["ints"].dtype == np.float64
    assert df_mean["ints"].iloc[2] == pytest.approx(23.333333)

    # Median imputation on floats
    op_median = HandleMissing(columns=["floats"], strategy=MissingStrategy.FILL_MEDIAN)
    df_median, _ = execute_handle_missing(df, op_median)
    assert df_median["floats"].iloc[2] == 2.5

    # Mode imputation
    op_mode = HandleMissing(columns=["category"], strategy=MissingStrategy.FILL_MODE)
    df_mode, _ = execute_handle_missing(df, op_mode)
    assert df_mode["category"].iloc[3] == "apple"

    # Fill value
    op_val = HandleMissing(columns=["category"], strategy=MissingStrategy.FILL_VALUE, fill_value="pear")
    df_val, _ = execute_handle_missing(df, op_val)
    assert df_val["category"].iloc[3] == "pear"


def test_handle_missing_rejects_infinity():
    """Verify mean/median imputation rejects infinite values."""
    df_inf = pd.DataFrame({"nums": [1.0, np.inf, np.nan]})
    op = HandleMissing(columns=["nums"], strategy=MissingStrategy.FILL_MEAN)
    with pytest.raises(ValueError, match="infinite values"):
        execute_handle_missing(df_inf, op)


def test_handle_missing_mode_deterministic_tie_breaker():
    """Verify mode imputation uses deterministic JSON tie-breaking on frequency ties."""
    # Both "Zebra" and "Apple" have frequency 1
    df = pd.DataFrame({"animals": ["Zebra", "Apple", np.nan]})
    op = HandleMissing(columns=["animals"], strategy=MissingStrategy.FILL_MODE)
    res_df, _ = execute_handle_missing(df, op)
    # Sorted canonically, "Apple" < "Zebra", so "Apple" must win deterministically
    assert res_df["animals"].iloc[2] == "Apple"


def test_deduplicate_first_and_last():
    """Verify Deduplicate supports subset, first, and last."""
    df = pd.DataFrame({
        "key": ["k1", "k1", "k2"],
        "val": [10, 20, 30],
    })

    op_first = Deduplicate(subset=["key"], keep="first")
    res_first, details_first = execute_deduplicate(df, op_first)
    assert len(res_first) == 2
    assert res_first.iloc[0]["val"] == 10
    assert details_first["duplicates_removed"] == 1

    op_last = Deduplicate(subset=["key"], keep="last")
    res_last, _ = execute_deduplicate(df, op_last)
    assert len(res_last) == 2
    assert res_last.iloc[0]["val"] == 20


def test_rename_columns_and_collision_check():
    """Verify RenameColumns succeeds and rejects collisions."""
    df = pd.DataFrame({"a": [1], "b": [2]})

    op_valid = RenameColumns(mapping={"a": "alpha"})
    res_valid, _ = execute_rename_columns(df, op_valid)
    assert "alpha" in res_valid.columns
    assert "a" not in res_valid.columns

    # Collision with existing column "b"
    op_collision = RenameColumns(mapping={"a": "b"})
    with pytest.raises(ValueError, match="Rename collision"):
        execute_rename_columns(df, op_collision)


def test_cast_columns_semantics():
    """Verify strict type casting and lossy float rejection."""
    df = pd.DataFrame({
        "int_str": ["10", "20", np.nan],
        "float_lossy": [1.0, 2.5, np.nan],
        "float_clean": [1.0, 2.0, np.nan],
        "bool_str": ["true", "False", "1"],
        "dt_str": ["2023-01-01", "2023-02-01", np.nan],
    })

    # Lossy float to int must fail
    op_lossy = CastColumns(casts={"float_lossy": TargetDataType.INT})
    with pytest.raises(ValueError, match="Lossy cast to int"):
        execute_cast_columns(df, op_lossy)

    # Clean float to int succeeds with nullable Int64
    op_clean = CastColumns(casts={"float_clean": TargetDataType.INT})
    res_clean, _ = execute_cast_columns(df, op_clean)
    assert res_clean["float_clean"].dtype == "Int64"
    assert res_clean["float_clean"].iloc[0] == 1

    # Boolean cast
    op_bool = CastColumns(casts={"bool_str": TargetDataType.BOOL})
    res_bool, _ = execute_cast_columns(df, op_bool)
    assert res_bool["bool_str"].dtype == "boolean"
    assert res_bool["bool_str"].iloc[0] is True or res_bool["bool_str"].iloc[0] == 1
    assert res_bool["bool_str"].iloc[1] is False or res_bool["bool_str"].iloc[1] == 0

    # Datetime cast
    op_dt = CastColumns(casts={"dt_str": TargetDataType.DATETIME})
    res_dt, _ = execute_cast_columns(df, op_dt)
    assert pd.api.types.is_datetime64_any_dtype(res_dt["dt_str"])


# ==============================================================================
# 2. Sequential Execution & Resource Bounds Tests
# ==============================================================================

def test_sequential_pipeline_execution(sample_df):
    """Verify operations execute in exact sequential order."""
    operations = [
        Deduplicate(subset=["id"], keep="first"),
        HandleMissing(columns=["name"], strategy=MissingStrategy.FILL_VALUE, fill_value="Unknown"),
        SelectColumns(columns=["id", "name", "score"], mode="keep"),
        FilterRows(column="score", operator=FilterOperator.GT, value=80),
        RenameColumns(mapping={"name": "full_name"}),
    ]

    out_df, summaries = apply_transformations(sample_df, operations)
    assert len(summaries) == 5
    assert list(out_df.columns) == ["id", "full_name", "score"]
    assert "Alice" in out_df["full_name"].values
    assert len(out_df) == 3  # Alice (85.5), Bob (92.0), and Unknown (92.0)


def test_resource_bounds_row_limit():
    """Verify check_dataset_bounds enforces 100,000 row maximum."""
    big_df = pd.DataFrame({"x": range(100_001)})
    with pytest.raises(ValueError, match="maximum row limit"):
        check_dataset_bounds(big_df)


def test_resource_bounds_column_limit():
    """Verify check_dataset_bounds enforces 200 column maximum."""
    cols = {f"c{i}": [1] for i in range(201)}
    wide_df = pd.DataFrame(cols)
    with pytest.raises(ValueError, match="maximum column limit"):
        check_dataset_bounds(wide_df)


def test_cooperative_cancellation_during_pipeline(sample_df):
    """Verify cooperative cancellation check aborts pipeline execution."""
    cancel_flag = False

    def check_cancel() -> bool:
        return cancel_flag

    operations = [
        Deduplicate(subset=["id"], keep="first"),
        RenameColumns(mapping={"id": "identifier"}),
    ]

    cancel_flag = True
    with pytest.raises(asyncio.CancelledError, match="cancelled"):
        apply_transformations(sample_df, operations, cancellation_check=check_cancel)


# ==============================================================================
# 3. Tool Registration & End-to-End Execution Tests
# ==============================================================================

def test_tool_registry_definition():
    """Verify transform_dataset is registered with strict schemas and permissions."""
    tool_def = tool_registry.get_tool_definition("transform_dataset")
    assert tool_def is not None
    assert tool_def.name == "transform_dataset"
    assert tool_def.category == ToolCategory.DATASET
    assert tool_def.permission_level == PermissionLevel.EXECUTE
    assert tool_def.retry_safe is False
    assert tool_def.timeout_seconds == 60


def test_execute_direct_read_only_tool_rejects_transform_dataset():
    """Verify direct API execution rejects transform_dataset (it requires EXECUTE permission)."""
    with pytest.raises(PermissionError, match="not available for direct execution"):
        asyncio.run(
            tool_registry.execute_direct_read_only_tool(
                "transform_dataset",
                {"file_path": "data/sample_sales.csv", "operations": []},
            )
        )


def test_transform_dataset_tool_execution(managed_transforms_dir, monkeypatch):
    """Verify transform_dataset produces encrypted, atomic, task-scoped CSV artifact."""
    # Write a test encrypted CSV in uploads
    source_file = settings.UPLOADS_DIR / "input.csv.enc"
    sample_data = "id,name,score\n1,Alice,90\n2,Bob,80\n3,Charlie,70\n"
    source_file.write_bytes(encryption_service().encrypt_bytes(sample_data.encode("utf-8")))

    task_id = "task_b3_test_12345"
    payload = TransformDatasetInput(
        file_path=str(source_file),
        operations=[
            FilterRows(column="score", operator=FilterOperator.GTE, value=80),
            RenameColumns(mapping={"name": "student_name"}),
        ],
        task_id=task_id,
    )

    res = transform_dataset(payload)
    assert res["status"] == "success"
    assert res["final_row_count"] == 2
    assert "student_name" in res["columns"]
    assert len(res["artifacts"]) == 1

    artifact_path = Path(res["artifacts"][0])
    assert artifact_path.exists()
    assert artifact_path.parent == managed_transforms_dir.resolve()

    # Verify filename format: transformed_<task_id_hash10>_<file_id10>.csv.enc
    expected_task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:10]
    assert artifact_path.name.startswith(f"transformed_{expected_task_hash}_")
    assert artifact_path.name.endswith(".csv.enc")

    # Decrypt and verify contents
    decrypted_bytes = encryption_service().decrypt_bytes(artifact_path.read_bytes())
    df_recovered = pd.read_csv(pd.io.common.BytesIO(decrypted_bytes))
    assert len(df_recovered) == 2
    assert set(df_recovered["student_name"].tolist()) == {"Alice", "Bob"}


def test_transform_dataset_task_artifact_quota(managed_transforms_dir, monkeypatch):
    """Verify task limit of maximum 5 transformed artifacts is enforced."""
    source_file = settings.UPLOADS_DIR / "small.csv.enc"
    source_file.write_bytes(encryption_service().encrypt_bytes(b"x\n1\n"))

    task_id = "task_quota_test"
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:10]

    # Pre-create 5 dummy artifacts for this task
    for i in range(5):
        dummy = managed_transforms_dir / f"transformed_{task_hash}_012345678{i}.csv.enc"
        dummy.write_bytes(b"dummy")

    payload = TransformDatasetInput(
        file_path=str(source_file),
        operations=[SelectColumns(columns=["x"], mode="keep")],
        task_id=task_id,
    )

    with pytest.raises(ValueError, match="Maximum transformed artifact limit"):
        transform_dataset(payload)


def test_manifest_integration_and_task_hash_validation(managed_transforms_dir, monkeypatch):
    """Verify generated artifact seamlessly integrates into SQLiteCheckpointer manifest."""
    source_file = settings.UPLOADS_DIR / "data.csv.enc"
    source_file.write_bytes(encryption_service().encrypt_bytes(b"val\n100\n"))

    task_id = "task_manifest_e2e"
    payload = TransformDatasetInput(
        file_path=str(source_file),
        operations=[SelectColumns(columns=["val"], mode="keep")],
        task_id=task_id,
    )

    res = transform_dataset(payload)
    artifact_str = res["artifacts"][0]

    task_state = TaskState(
        task_id=task_id,
        objective="E2E Manifest test",
        artifacts=[artifact_str],
    )

    manifest = SQLiteCheckpointer._build_artifact_manifest(task_state)
    assert len(manifest) == 1
    assert manifest[0]["artifact_type"] == "transformed_dataset"
    assert manifest[0]["task_id"] == task_id

    # Verify cross-task adoption of this artifact is rejected
    foreign_state = TaskState(
        task_id="task_FOREIGN_999",
        objective="Foreign adoption attempt",
        artifacts=[artifact_str],
    )
    foreign_manifest = SQLiteCheckpointer._build_artifact_manifest(foreign_state)
    assert len(foreign_manifest) == 0


# ==============================================================================
# 4. Phase 3.3.2-B3.1 Remediation Regression Tests
# ==============================================================================

def test_remediation_b3_01_report_with_analytics_receives_findings():
    """Verify generate_report retains upstream analytics findings enrichment (FINDING-B3-01)."""
    from app.agent.executor import PlanExecutor
    from app.agent.state import ExecutionStep, TaskState

    task_state = TaskState(
        task_id="task_report_findings_enrich",
        objective="test",
        step_results={
            "step_analytics": {
                "tool": "group_comparison",
                "data": {
                    "underperforming_groups": ["LowRevenue_A", "LowRevenue_B"],
                    "total_metric_value": 125000.50,
                    "metric_column": "revenue",
                },
                "metadata": {"task_id": "task_report_findings_enrich"},
            }
        },
    )

    step = ExecutionStep(
        step_id="step_rep",
        title="Report Step",
        description="Generates report",
        capability="system",
        tool_name="generate_report",
        parameters={
            "title": "Quarterly Financial Analysis",
            "executive_summary": "Summary of quarterly financial breakdown.",
            "findings": ["Initial core observation."],
            "recommendations": ["Expand high-performing categories."],
        },
        dependencies=["step_analytics"],
    )

    executor = PlanExecutor()
    resolved = executor._resolve_step_parameters(step, task_state)
    assert "findings" in resolved
    findings_text = " ".join(resolved["findings"])
    assert "Underperforming segments identified below 5% contribution threshold" in findings_text
    assert "Total aggregate revenue evaluated: 125,000.50" in findings_text


def test_remediation_b3_01_transform_dataset_step_with_dependencies_no_findings():
    """Verify transform_dataset does not receive unexpected findings from dependencies (FINDING-B3-01)."""
    from app.agent.executor import PlanExecutor
    from app.agent.state import ExecutionStep, TaskState

    task_state = TaskState(
        task_id="task_transform_dep_clean",
        objective="test",
        step_results={
            "step_analytics": {
                "tool": "group_comparison",
                "data": {
                    "underperforming_groups": ["LowRevenue_A"],
                    "total_metric_value": 50000.0,
                },
                "metadata": {"task_id": "task_transform_dep_clean"},
            }
        },
    )

    step = ExecutionStep(
        step_id="step_tr",
        title="Transform Step",
        description="Transforms data",
        capability="system",
        tool_name="transform_dataset",
        parameters={
            "file_path": "data/sample_sales.csv",
            "operations": [{"op": "select_columns", "columns": ["date", "category"], "mode": "keep"}],
        },
        dependencies=["step_analytics"],
    )

    executor = PlanExecutor()
    resolved = executor._resolve_step_parameters(step, task_state)
    # Findings must NOT be injected into transform_dataset parameters
    assert "findings" not in resolved
    assert resolved["task_id"] == "task_transform_dep_clean"


def test_remediation_b3_01_transform_dataset_input_rejects_extra_fields():
    """Verify TransformDatasetInput strictly forbids unexpected client fields (FINDING-B3-01)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TransformDatasetInput.model_validate({
            "file_path": "data/sample_sales.csv",
            "operations": [{"op": "select_columns", "columns": ["category"], "mode": "keep"}],
            "findings": ["Injected malicious findings"],
        })


@pytest.mark.asyncio
async def test_remediation_b3_01_both_workflows_work_through_authorized_pipeline(managed_transforms_dir):
    """Verify transform_dataset with dependencies executes successfully through PlanExecutor."""
    from app.agent.execution_handoff import test_legacy_tool_invocation
    from app.agent.executor import executor
    from app.agent.state import ExecutionStep, StepStatus, TaskState

    source_file = settings.UPLOADS_DIR / "pipeline_in.csv.enc"
    source_file.write_bytes(encryption_service().encrypt_bytes(b"colA,colB\n1,2\n3,4\n"))

    task_state = TaskState(
        task_id="task_pipeline_b3_1",
        objective="Verify authorized execution of transform_dataset with dependencies",
        step_results={
            "dep_step": {
                "tool": "some_prior_tool",
                "data": {"underperforming_groups": ["GroupX"], "total_metric_value": 999.0},
                "metadata": {"task_id": "task_pipeline_b3_1"},
            }
        },
    )

    step = ExecutionStep(
        step_id="step_pipeline_tr",
        title="Transform Data",
        description="Select colA",
        capability="system",
        tool_name="transform_dataset",
        parameters={
            "file_path": str(source_file),
            "operations": [{"op": "select_columns", "columns": ["colA"], "mode": "keep"}],
        },
        dependencies=["dep_step"],
    )

    result = await executor.execute_step(step, task_state)
    assert result.success is True
    assert step.status == StepStatus.COMPLETED
    assert result.data["final_column_count"] == 1


def test_remediation_b3_02_boolean_casting_all_literals():
    """Verify CastColumns accepts all approved B1.2 boolean representations and rejects invalid values (FINDING-B3-02)."""
    # Test True representations
    true_literals = ["true", "TRUE", "True", " 1 ", 1, 1.0, "t", "T", "yes", "YES", "Yes", "y", "Y"]
    for lit in true_literals:
        df = pd.DataFrame({"flag": [lit, np.nan]})
        res, _ = execute_cast_columns(df, CastColumns(casts={"flag": TargetDataType.BOOL}))
        assert res["flag"].iloc[0] is True or res["flag"].iloc[0] == 1, f"Failed for True literal: {lit}"
        assert pd.isna(res["flag"].iloc[1])

    # Test False representations
    false_literals = ["false", "FALSE", "False", " 0 ", 0, 0.0, "f", "F", "no", "NO", "No", "n", "N"]
    for lit in false_literals:
        df = pd.DataFrame({"flag": [lit, np.nan]})
        res, _ = execute_cast_columns(df, CastColumns(casts={"flag": TargetDataType.BOOL}))
        assert res["flag"].iloc[0] is False or res["flag"].iloc[0] == 0, f"Failed for False literal: {lit}"
        assert pd.isna(res["flag"].iloc[1])

    # Test invalid values fail closed
    invalid_literals = ["maybe", "yep", "nope", 2, -1, "2", "foo"]
    for bad in invalid_literals:
        df = pd.DataFrame({"flag": [bad]})
        with pytest.raises(ValueError, match="Lossy/invalid cast to bool"):
            execute_cast_columns(df, CastColumns(casts={"flag": TargetDataType.BOOL}))


def test_remediation_b3_03_datetime_timezone_comparisons():
    """Verify datetime comparisons under B1.2 timezone semantics across all combinations (FINDING-B3-03)."""
    # 1. Aware vs Aware with different offsets
    # 10:00 UTC+2 is 08:00 UTC; 09:00 UTC-2 is 11:00 UTC
    df_aware = pd.DataFrame({"dt": ["2026-01-01 10:00:00+02:00", "2026-01-01 15:00:00+02:00"]})
    op_aware = FilterRows(column="dt", operator=FilterOperator.LT, value="2026-01-01T09:00:00-02:00")
    res_aware, _ = execute_filter_rows(df_aware, op_aware)
    # 08:00 UTC is < 11:00 UTC (10:00+02:00 matches); 13:00 UTC is not < 11:00 UTC (15:00+02:00 excluded)
    assert len(res_aware) == 1
    assert res_aware.iloc[0]["dt"] == "2026-01-01 10:00:00+02:00"

    # 2. Naive vs Naive
    df_naive = pd.DataFrame({"dt": ["2026-01-01 10:00:00", "2026-01-01 14:00:00"]})
    op_naive = FilterRows(column="dt", operator=FilterOperator.GT, value="2026-01-01T12:00:00")
    res_naive, _ = execute_filter_rows(df_naive, op_naive)
    assert len(res_naive) == 1
    assert res_naive.iloc[0]["dt"] == "2026-01-01 14:00:00"

    # 3. Aware column vs Naive operand (naive interpreted as UTC)
    # Operand: 11:00 naive -> 11:00 UTC. Column: 10:00+02:00 (08:00 UTC), 14:00+02:00 (12:00 UTC)
    op_aware_vs_naive = FilterRows(column="dt", operator=FilterOperator.GT, value="2026-01-01T11:00:00")
    res_aware_vs_naive, _ = execute_filter_rows(df_aware, op_aware_vs_naive)
    assert len(res_aware_vs_naive) == 1
    assert res_aware_vs_naive.iloc[0]["dt"] == "2026-01-01 15:00:00+02:00"

    # 4. Naive column vs Aware operand (naive column interpreted as UTC)
    # Column: 10:00 naive -> 10:00 UTC, 14:00 naive -> 14:00 UTC. Operand: 13:00+02:00 -> 11:00 UTC
    op_naive_vs_aware = FilterRows(column="dt", operator=FilterOperator.GT, value="2026-01-01T13:00:00+02:00")
    res_naive_vs_aware, _ = execute_filter_rows(df_naive, op_naive_vs_aware)
    assert len(res_naive_vs_aware) == 1
    assert res_naive_vs_aware.iloc[0]["dt"] == "2026-01-01 14:00:00"

    # 5. Mixed values across a single column (aware, naive, nulls)
    df_mixed = pd.DataFrame({
        "dt": [
            "2026-01-01 10:00:00+02:00",  # 08:00 UTC
            "2026-01-01 12:00:00",        # 12:00 UTC (naive as UTC)
            None,                         # Null
            "2026-01-01 07:00:00Z",       # 07:00 UTC
        ]
    })
    # Filter GT 07:30 UTC
    op_mixed = FilterRows(column="dt", operator=FilterOperator.GT, value="2026-01-01T07:30:00Z")
    res_mixed, _ = execute_filter_rows(df_mixed, op_mixed)
    # 08:00 UTC and 12:00 UTC match; 07:00 UTC and None excluded
    assert len(res_mixed) == 2
    assert "2026-01-01 10:00:00+02:00" in res_mixed["dt"].tolist()
    assert "2026-01-01 12:00:00" in res_mixed["dt"].tolist()

    # 6. Invalid datetime operand fails with sanitized typed ValueError
    with pytest.raises(ValueError, match="Invalid ISO datetime operand"):
        execute_filter_rows(df_naive, FilterRows(column="dt", operator=FilterOperator.GT, value="not-a-datetime"))

    # 7. Unparseable column fails with sanitized typed ValueError
    df_unparseable = pd.DataFrame({"dt": ["not-a-date", "also-not-a-date"]})
    with pytest.raises(ValueError, match="Cannot perform datetime comparison"):
        execute_filter_rows(df_unparseable, FilterRows(column="dt", operator=FilterOperator.GT, value="2026-01-01T00:00:00"))


def test_remediation_b3_04_pre_finalization_cancellation_boundaries(managed_transforms_dir):
    """Verify cooperative cancellation checks at safe boundaries prevent file finalization (FINDING-B3-04)."""
    from app.tools.registry import _active_cancellation_check

    source_file = settings.UPLOADS_DIR / "cancel_test.csv.enc"
    source_file.write_bytes(encryption_service().encrypt_bytes(b"a,b\n1,2\n3,4\n"))

    task_id = "task_cancel_boundaries"
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:10]
    payload = TransformDatasetInput(
        file_path=str(source_file),
        operations=[SelectColumns(columns=["a"], mode="keep")],
        task_id=task_id,
    )

    # 1. Cancellation before load/execution
    token = _active_cancellation_check.set(lambda: True)
    try:
        with pytest.raises(asyncio.CancelledError, match="pre_load"):
            transform_dataset(payload)
    finally:
        _active_cancellation_check.reset(token)

    # Confirm no temp files or final files were created
    assert len(list(managed_transforms_dir.glob(f"*{task_hash}*"))) == 0

    # 2. Dynamic cancellation flag triggered during finalization
    call_count = 0

    def trigger_during_finalization() -> bool:
        nonlocal call_count
        call_count += 1
        # Trigger cancellation after transformation execution completes
        return call_count >= 3

    token2 = _active_cancellation_check.set(trigger_during_finalization)
    try:
        with pytest.raises(asyncio.CancelledError):
            transform_dataset(payload)
    finally:
        _active_cancellation_check.reset(token2)

    # Confirm any .tmp files were unlinked and no finalized file exists
    assert len(list(managed_transforms_dir.glob("*.tmp"))) == 0
    assert len(list(managed_transforms_dir.glob(f"transformed_{task_hash}_*.csv.enc"))) == 0
