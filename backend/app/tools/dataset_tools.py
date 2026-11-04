import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, JsonValue

from app.tools.schemas import ToolCategory, ExecutionLocation, PermissionLevel, ToolResult, StrictContract
from app.tools.registry import tool_registry
from app.core.security import validate_safe_path, read_managed_dataset, managed_file_suffix

class InspectDatasetInput(BaseModel):
    file_path: str = Field(description="Relative or absolute path to CSV or Excel file.")
    sample_rows: int = Field(default=5, description="Number of sample rows to return for preview.")

class InspectDatasetOutput(StrictContract):
    file_name: str
    format: str
    total_rows: int
    total_columns: int
    columns: list[str]
    column_types: dict[str, str]
    missing_values: dict[str, int]
    missing_percentages: dict[str, float]
    duplicate_rows_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    date_columns: list[str]
    sample_data: list[dict[str, JsonValue]]
    memory_usage_kb: float

@tool_registry.register(
    name="inspect_dataset",
    category=ToolCategory.DATASET,
    description="Inspects tabular files (CSV, Excel) to report dimensions, schema, column data types, missing values, duplicates, and data samples.",
    input_model=InspectDatasetInput,
    output_model=InspectDatasetOutput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY,
    retry_safe=True,
    timeout_seconds=30
)
def inspect_dataset(params: InspectDatasetInput) -> dict[str, Any]:
    safe_path = validate_safe_path(params.file_path)
    if not safe_path.exists():
        raise FileNotFoundError(f"File not found: {safe_path.name}")
        
    ext = managed_file_suffix(safe_path)
    df = read_managed_dataset(safe_path)

    total_rows, total_cols = df.shape
    columns = df.columns.tolist()
    col_types = {col: str(df[col].dtype) for col in columns}
    
    null_counts = df.isnull().sum().to_dict()
    null_pcts = {col: round(float((cnt / total_rows) * 100), 2) if total_rows > 0 else 0.0 for col, cnt in null_counts.items()}
    dup_count = int(df.duplicated().sum())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat_cols = [c for c in columns if c not in numeric_cols and c not in date_cols]
    
    # Clean preview data for JSON serialization (replace NaN/Inf with None)
    sample_df = df.head(params.sample_rows).replace({np.nan: None, np.inf: None, -np.inf: None})
    sample_data = sample_df.to_dict(orient="records")
    
    mem_kb = round(float(df.memory_usage(deep=True).sum() / 1024), 2)

    return {
        "file_name": safe_path.name,
        "format": ext[1:].upper(),
        "total_rows": total_rows,
        "total_columns": total_cols,
        "columns": columns,
        "column_types": col_types,
        "missing_values": null_counts,
        "missing_percentages": null_pcts,
        "duplicate_rows_count": dup_count,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "date_columns": date_cols,
        "sample_data": sample_data,
        "memory_usage_kb": mem_kb
    }

