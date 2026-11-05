"""Tests for Phase 3.3.2-B2: Data Transformation Foundation (Schemas, Storage, and Manifests)."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.execution_trace import TraceArtifactType
from app.agent.state import TaskState
from app.core.config import settings
from app.core.security import validate_safe_path
from app.privacy.encryption import encryption_service
from app.tools.transform_schemas import (
    CastColumns,
    Deduplicate,
    FilterOperator,
    FilterRows,
    HandleMissing,
    MissingStrategy,
    OperationSummary,
    RenameColumns,
    SelectColumns,
    TargetDataType,
    TransformDatasetInput,
    TransformDatasetOutput,
    generate_transformed_filename,
)


@pytest.fixture
def managed_test_dirs(tmp_path):
    """Fixture providing dedicated isolated directories for testing."""
    uploads = tmp_path / "uploads"
    transforms = tmp_path / "transforms"
    reports = tmp_path / "reports"
    charts = tmp_path / "charts"
    for d in (uploads, transforms, reports, charts):
        d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        uploads=uploads,
        transforms=transforms,
        reports=reports,
        charts=charts,
        tmp_path=tmp_path,
    )


# ==============================================================================
# 1. TRANSFORMS_DIR & Filename Generation Tests
# ==============================================================================

def test_transforms_dir_configuration():
    """Verify TRANSFORMS_DIR configuration and safe initialization."""
    assert settings.TRANSFORMS_DIR == settings.DATA_DIR / "transforms"
    assert settings.TRANSFORMS_DIR.exists()
    assert settings.TRANSFORMS_DIR.is_dir()


def test_generate_transformed_filename_format():
    """Verify generated filename structure matches strict naming standard."""
    task_id = "task_test_12345"
    filename = generate_transformed_filename(task_id)

    # Must match regex: transformed_<task_id_hash10>_<file_id10>.csv.enc
    match = re.fullmatch(r"^transformed_([a-f0-9]{10})_([a-f0-9]{10})\.csv\.enc$", filename)
    assert match is not None

    expected_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:10]
    assert match.group(1) == expected_hash

    # Repeated calls should produce distinct random file_id10
    second = generate_transformed_filename(task_id)
    assert second != filename
    assert second.startswith(f"transformed_{expected_hash}_")


# ==============================================================================
# 2. Path Validation & Symlink Security Tests
# ==============================================================================

def test_validate_safe_path_accepts_transforms_dir(managed_test_dirs, monkeypatch):
    """Verify validate_safe_path accepts files inside TRANSFORMS_DIR."""
    monkeypatch.setattr(settings, "TRANSFORMS_DIR", managed_test_dirs.transforms)
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dirs.uploads)

    safe_file = managed_test_dirs.transforms / "transformed_0123456789_abcdef0123.csv.enc"
    safe_file.write_bytes(b"encrypted_data")

    resolved = validate_safe_path(safe_file)
    assert resolved == safe_file.resolve()


def test_validate_safe_path_rejects_null_bytes():
    """Verify null bytes in target path are strictly rejected."""
    with pytest.raises(PermissionError, match="null byte"):
        validate_safe_path("uploads/safe.csv\x00malicious")


def test_validate_safe_path_rejects_outside_files(managed_test_dirs, monkeypatch):
    """Verify files outside approved roots are rejected."""
    monkeypatch.setattr(settings, "TRANSFORMS_DIR", managed_test_dirs.transforms)
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dirs.uploads)

    outside = managed_test_dirs.tmp_path / "outside.csv"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(PermissionError, match="outside managed uploads or transforms"):
        validate_safe_path(outside)


def test_validate_safe_path_rejects_unsupported_extensions(managed_test_dirs, monkeypatch):
    """Verify unsupported extensions are rejected even if in approved root."""
    monkeypatch.setattr(settings, "TRANSFORMS_DIR", managed_test_dirs.transforms)
    bad_ext = managed_test_dirs.transforms / "script.py"
    bad_ext.write_text("print(1)", encoding="utf-8")

    with pytest.raises(PermissionError, match="unsupported managed file type"):
        validate_safe_path(bad_ext)


def test_validate_safe_path_rejects_symlink_leaf(managed_test_dirs, monkeypatch):
    """Verify symlink file pointing to a legitimate file is rejected before resolution."""
    monkeypatch.setattr(settings, "TRANSFORMS_DIR", managed_test_dirs.transforms)
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dirs.uploads)

    target_file = managed_test_dirs.transforms / "real.csv"
    target_file.write_text("a,b\n1,2\n", encoding="utf-8")

    symlink_file = managed_test_dirs.transforms / "symlink.csv"
    try:
        symlink_file.symlink_to(target_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not permitted in this OS environment.")

    with pytest.raises(PermissionError, match="symlinks are not permitted"):
        validate_safe_path(symlink_file)


def test_validate_safe_path_rejects_symlink_parent_directory(managed_test_dirs, monkeypatch):
    """Verify path containing an intermediate symlinked directory is rejected."""
    monkeypatch.setattr(settings, "TRANSFORMS_DIR", managed_test_dirs.transforms)
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dirs.uploads)

    real_dir = managed_test_dirs.transforms / "real_dir"
    real_dir.mkdir()
    real_file = real_dir / "data.csv"
    real_file.write_text("x,y\n1,2\n", encoding="utf-8")

    symlink_dir = managed_test_dirs.transforms / "link_dir"
    try:
        symlink_dir.symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlink creation not permitted in this OS environment.")

    linked_file = symlink_dir / "data.csv"
    with pytest.raises(PermissionError, match="symlinks are not permitted"):
        validate_safe_path(linked_file)


# ==============================================================================
# 3. Checkpoint Manifest & Artifact Restoration Tests
# ==============================================================================

def test_checkpoint_artifact_type_and_id_for_transformed_dataset():
    """Verify _artifact_type_and_id parses transformed dataset filenames correctly."""
    valid_name = "transformed_1a2b3c4d5e_6f7a8b9c0d.csv.enc"
    res = SQLiteCheckpointer._artifact_type_and_id(valid_name)
    assert res == ("transformed_dataset", "transformed_6f7a8b9c0d")

    # Invalid names must return None
    assert SQLiteCheckpointer._artifact_type_and_id("transformed_short.csv.enc") is None
    assert SQLiteCheckpointer._artifact_type_and_id("transformed_123_456.csv") is None
    assert SQLiteCheckpointer._artifact_type_and_id("transformed_1a2b3c4d5e_6f7a8b9c0d.csv") is None


def test_checkpoint_build_manifest_includes_transformed_dataset(managed_test_dirs, monkeypatch):
    """Verify _build_artifact_manifest identifies transformed datasets in TRANSFORMS_DIR."""
    monkeypatch.setattr(settings, "TRANSFORMS_DIR", managed_test_dirs.transforms)
    monkeypatch.setattr(settings, "CHARTS_DIR", managed_test_dirs.charts)
    monkeypatch.setattr(settings, "REPORTS_DIR", managed_test_dirs.reports)

    task_id = "task_manifest_test_01"
    valid_filename = "transformed_1a2b3c4d5e_6f7a8b9c0d.csv.enc"
    artifact_file = managed_test_dirs.transforms / valid_filename
    artifact_file.write_bytes(b"encrypted_content")

    task_state = TaskState(
        task_id=task_id,
        objective="Transform test",
        artifacts=[str(artifact_file)],
    )

    manifest = SQLiteCheckpointer._build_artifact_manifest(task_state)
    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["artifact_id"] == "transformed_6f7a8b9c0d"
    assert entry["task_id"] == task_id
    assert entry["artifact_type"] == "transformed_dataset"
    assert entry["relative_path"] == valid_filename


def test_checkpoint_restore_manifest_for_transformed_dataset(managed_test_dirs, monkeypatch):
    """Verify _restore_artifact_manifest restores transformed dataset artifacts safely."""
    monkeypatch.setattr(settings, "TRANSFORMS_DIR", managed_test_dirs.transforms)
    monkeypatch.setattr(settings, "CHARTS_DIR", managed_test_dirs.charts)
    monkeypatch.setattr(settings, "REPORTS_DIR", managed_test_dirs.reports)

    task_id = "task_restore_test_02"
    valid_filename = "transformed_0123456789_abcdef0123.csv.enc"
    artifact_file = managed_test_dirs.transforms / valid_filename
    artifact_file.write_bytes(b"encrypted_content")

    manifest_entries = [
        {
            "artifact_id": "transformed_abcdef0123",
            "task_id": task_id,
            "artifact_type": "transformed_dataset",
            "relative_path": valid_filename,
        }
    ]
    manifest_json = json.dumps(manifest_entries)
    encrypted_manifest = encryption_service().encrypt_bytes(manifest_json.encode("utf-8")).decode("ascii")

    checkpointer = SQLiteCheckpointer(managed_test_dirs.tmp_path / "test.db")
    state = TaskState(task_id=task_id, objective="Restore test")
    checkpointer._restore_artifact_manifest(state, encrypted_manifest)

    assert len(state.artifacts) == 1
    assert Path(state.artifacts[0]).resolve() == artifact_file.resolve()


def test_checkpoint_restore_rejects_cross_task_manifest(managed_test_dirs, monkeypatch):
    """Verify _restore_artifact_manifest fails closed on task ID mismatch."""
    monkeypatch.setattr(settings, "TRANSFORMS_DIR", managed_test_dirs.transforms)

    manifest_entries = [
        {
            "artifact_id": "transformed_abcdef0123",
            "task_id": "task_DIFFERENT",
            "artifact_type": "transformed_dataset",
            "relative_path": "transformed_0123456789_abcdef0123.csv.enc",
        }
    ]
    encrypted = encryption_service().encrypt_bytes(json.dumps(manifest_entries).encode("utf-8")).decode("ascii")
    checkpointer = SQLiteCheckpointer(managed_test_dirs.tmp_path / "test.db")
    state = TaskState(task_id="task_CURRENT", objective="Mismatch test")

    with pytest.raises(ValueError, match="Artifact manifest task mismatch"):
        checkpointer._restore_artifact_manifest(state, encrypted)


# ==============================================================================
# 4. Pydantic Transformation Schemas Tests
# ==============================================================================

def test_filter_rows_valid_variants():
    """Verify valid FilterRows models."""
    f1 = FilterRows(column="age", operator=FilterOperator.GT, value=21)
    assert f1.op == "filter_rows"
    assert f1.value == 21

    f2 = FilterRows(column="name", operator=FilterOperator.CONTAINS, value="smith", case_sensitive=False)
    assert f2.case_sensitive is False

    f3 = FilterRows(column="status", operator=FilterOperator.IN, value=["active", "pending"])
    assert len(f3.value) == 2

    f4 = FilterRows(column="score", operator=FilterOperator.IS_NULL)
    assert f4.value is None

    f5 = FilterRows(column="score", operator=FilterOperator.NOT_NULL)
    assert f5.value is None


def test_filter_rows_validation_failures():
    """Verify invalid FilterRows configurations are rejected."""
    # is_null / not_null must not provide value
    with pytest.raises(ValidationError, match="must not have a comparison value"):
        FilterRows(column="age", operator=FilterOperator.IS_NULL, value=10)

    # Missing value for equality/ordered operators
    with pytest.raises(ValidationError, match="requires a non-null comparison value"):
        FilterRows(column="age", operator=FilterOperator.EQ, value=None)

    # Boolean rejected for ordered operators
    with pytest.raises(ValidationError, match="does not accept boolean operands"):
        FilterRows(column="age", operator=FilterOperator.GT, value=True)

    # Non-string for contains
    with pytest.raises(ValidationError, match="requires a non-empty string operand"):
        FilterRows(column="age", operator=FilterOperator.CONTAINS, value=123)

    # Empty string for contains
    with pytest.raises(ValidationError, match="requires a non-empty string operand"):
        FilterRows(column="age", operator=FilterOperator.CONTAINS, value="")

    # Empty list for in
    with pytest.raises(ValidationError, match="requires a non-empty list of items"):
        FilterRows(column="status", operator=FilterOperator.IN, value=[])

    # In with nested dictionary
    with pytest.raises(ValidationError, match="must be a non-null scalar"):
        FilterRows(column="status", operator=FilterOperator.IN, value=[{"bad": 1}])

    # Equality with collection
    with pytest.raises(ValidationError, match="does not accept collection operands"):
        FilterRows(column="status", operator=FilterOperator.EQ, value=["a", "b"])

    # Invalid column name pattern (spaces or special chars)
    with pytest.raises(ValidationError):
        FilterRows(column="invalid column!", operator=FilterOperator.EQ, value=1)


def test_select_columns_validation():
    """Verify SelectColumns validation and duplicate detection."""
    valid = SelectColumns(columns=["col_a", "col_b"], mode="keep")
    assert valid.columns == ["col_a", "col_b"]

    # Reject duplicate columns
    with pytest.raises(ValidationError, match="contains duplicate column names"):
        SelectColumns(columns=["col_a", "col_b", "col_a"])

    # Reject empty list
    with pytest.raises(ValidationError):
        SelectColumns(columns=[])


def test_handle_missing_validation():
    """Verify HandleMissing validation for strategies and fill_value."""
    # drop_row without fill_value is valid
    hm1 = HandleMissing(columns=["age"], strategy=MissingStrategy.DROP_ROW)
    assert hm1.fill_value is None

    # fill_value requires fill_value
    hm2 = HandleMissing(columns=["score"], strategy=MissingStrategy.FILL_VALUE, fill_value=0.0)
    assert hm2.fill_value == 0.0

    # fill_value cannot be missing
    with pytest.raises(ValidationError, match="requires 'fill_value' to be set"):
        HandleMissing(columns=["score"], strategy=MissingStrategy.FILL_VALUE)

    # fill_value cannot be a collection
    with pytest.raises(ValidationError, match="requires a scalar replacement value"):
        HandleMissing(columns=["score"], strategy=MissingStrategy.FILL_VALUE, fill_value=[1, 2])

    # Other strategies must not have fill_value
    with pytest.raises(ValidationError, match="'fill_value' is forbidden when strategy is 'fill_mean'"):
        HandleMissing(columns=["score"], strategy=MissingStrategy.FILL_MEAN, fill_value=5.0)

    # Reject duplicate columns
    with pytest.raises(ValidationError, match="contains duplicate column names"):
        HandleMissing(columns=["col_a", "col_a"], strategy=MissingStrategy.DROP_ROW)


def test_deduplicate_validation():
    """Verify Deduplicate validation."""
    d1 = Deduplicate(subset=["user_id"], keep="last")
    assert d1.keep == "last"

    d2 = Deduplicate(subset=None, keep="first")
    assert d2.subset is None

    # Duplicate in subset rejected
    with pytest.raises(ValidationError, match="contains duplicate column names"):
        Deduplicate(subset=["id", "id"])


def test_rename_columns_validation():
    """Verify RenameColumns validation."""
    valid = RenameColumns(mapping={"old_name": "new_name", "a": "b"})
    assert len(valid.mapping) == 2

    # Reject self-mapping
    with pytest.raises(ValidationError, match="Self-mapping detected"):
        RenameColumns(mapping={"col_a": "col_a"})

    # Reject duplicate target names
    with pytest.raises(ValidationError, match="Rename targets contain duplicate column names"):
        RenameColumns(mapping={"col_a": "target", "col_b": "target"})


def test_cast_columns_validation():
    """Verify CastColumns validation."""
    valid = CastColumns(casts={"age": TargetDataType.INT, "salary": TargetDataType.FLOAT})
    assert valid.casts["age"] == TargetDataType.INT


def test_transform_dataset_input_validation():
    """Verify TransformDatasetInput operation bounds and constraints."""
    ops = [
        FilterRows(column="age", operator=FilterOperator.GT, value=18),
        SelectColumns(columns=["age", "name"]),
    ]
    inp = TransformDatasetInput(file_path="uploads/data.csv.enc", operations=ops)
    assert len(inp.operations) == 2

    # Empty operations rejected
    with pytest.raises(ValidationError):
        TransformDatasetInput(file_path="uploads/data.csv.enc", operations=[])

    # More than 20 operations rejected
    too_many = [FilterRows(column="c", operator=FilterOperator.EQ, value=i) for i in range(21)]
    with pytest.raises(ValidationError):
        TransformDatasetInput(file_path="uploads/data.csv.enc", operations=too_many)


def test_transform_dataset_output_validation():
    """Verify TransformDatasetOutput schema and artifact constraints."""
    out = TransformDatasetOutput(
        status="success",
        summary="Transformed 100 rows",
        initial_row_count=100,
        final_row_count=80,
        initial_column_count=5,
        final_column_count=4,
        columns=["a", "b", "c", "d"],
        operations_applied=[
            OperationSummary(op="filter_rows", rows_before=100, rows_after=80, cols_before=5, cols_after=5)
        ],
        artifacts=["/abs/path/to/transformed_1234567890_abcdef1234.csv.enc"],
    )
    assert out.final_row_count == 80
    assert len(out.artifacts) == 1

    # artifacts cannot be empty
    with pytest.raises(ValidationError):
        TransformDatasetOutput(
            status="success",
            summary="test",
            initial_row_count=10,
            final_row_count=10,
            initial_column_count=2,
            final_column_count=2,
            columns=["a"],
            operations_applied=[],
            artifacts=[],
        )

    # Extra fields forbidden (StrictContract)
    with pytest.raises(ValidationError):
        TransformDatasetOutput(
            status="success",
            summary="test",
            initial_row_count=10,
            final_row_count=10,
            initial_column_count=2,
            final_column_count=2,
            columns=["a"],
            operations_applied=[],
            artifacts=["/path/to/file.csv.enc"],
            unexpected_field="disallowed",
        )

