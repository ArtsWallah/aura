"""Transform dataset tool registration and execution handler."""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.security import read_managed_dataset, validate_safe_path
from app.privacy.encryption import encryption_service
from app.tools.registry import tool_registry
from app.tools.schemas import ExecutionLocation, PermissionLevel, ToolCategory
from app.tools.transform_schemas import (
    TransformDatasetInput,
    TransformDatasetOutput,
    generate_transformed_filename,
)
from app.tools.transforms.engine import apply_transformations

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_ARTIFACTS_PER_TASK = 5


@tool_registry.register(
    name="transform_dataset",
    category=ToolCategory.DATASET,
    description="Applies sequential deterministic data transformations (filtering, selection, missing value imputation, deduplication, renaming, type casting) to tabular datasets, producing an encrypted transformed dataset artifact.",
    input_model=TransformDatasetInput,
    output_model=TransformDatasetOutput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.EXECUTE,
    retry_safe=False,
    timeout_seconds=60,
)
def transform_dataset(params: TransformDatasetInput) -> dict[str, Any]:
    """Execute validated dataset transformation pipeline and persist encrypted artifact."""
    safe_path = validate_safe_path(params.file_path)
    if not safe_path.exists():
        raise FileNotFoundError(f"File not found: {safe_path.name}")

    df = read_managed_dataset(safe_path)
    initial_rows = len(df)
    initial_cols = len(df.columns)

    task_id = params.task_id or "task_standalone"
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:10]

    # Verify task artifact quota
    existing = list(settings.TRANSFORMS_DIR.glob(f"transformed_{task_hash}_*.csv.enc"))
    if len(existing) >= MAX_ARTIFACTS_PER_TASK:
        raise ValueError(
            f"Maximum transformed artifact limit ({MAX_ARTIFACTS_PER_TASK}) reached for task."
        )

    # Execute transformations
    transformed_df, summaries = apply_transformations(df, params.operations)

    # CSV serialization and size check
    csv_bytes = transformed_df.to_csv(index=False).encode("utf-8")
    if len(csv_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Transformed dataset exceeds serialized size limit of {MAX_FILE_SIZE_BYTES} bytes."
        )

    # Encryption and encrypted size check
    encrypted_bytes = encryption_service().encrypt_bytes(csv_bytes)
    if len(encrypted_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Transformed dataset exceeds encrypted size limit of {MAX_FILE_SIZE_BYTES} bytes."
        )

    # Exclusive temporary file creation and atomic finalization
    temp_filename = f".tmp_{task_hash}_{secrets.token_hex(8)}.tmp"
    temp_path = settings.TRANSFORMS_DIR / temp_filename

    fd = os.open(temp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(encrypted_bytes)
            f.flush()
            os.fsync(f.fileno())

        final_filename = generate_transformed_filename(task_id)
        final_path = settings.TRANSFORMS_DIR / final_filename
        os.replace(temp_path, final_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return {
        "status": "success",
        "summary": f"Applied {len(summaries)} transformation operations successfully.",
        "initial_row_count": initial_rows,
        "final_row_count": len(transformed_df),
        "initial_column_count": initial_cols,
        "final_column_count": len(transformed_df.columns),
        "columns": transformed_df.columns.tolist(),
        "operations_applied": [s.model_dump(mode="json") for s in summaries],
        "artifacts": [str(final_path)],
    }

