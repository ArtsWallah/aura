"""Pydantic contracts and schemas for dataset transformations (Phase 3.3.2-B2)."""

from __future__ import annotations

import hashlib
import re
import secrets
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from app.tools.schemas import StrictContract, StrictJsonValue

COLUMN_NAME_REGEX = r"^[a-zA-Z0-9_.-]{1,128}$"
ColumnName = Annotated[str, Field(pattern=COLUMN_NAME_REGEX)]


def generate_transformed_filename(task_id: str) -> str:
    """Generate a consistently task-scoped transformed dataset filename.

    Format: transformed_<task_id_hash10>_<file_id10>.csv.enc
    - task_id_hash10: First 10 hex characters of SHA-256(task_id).
    - file_id10: Cryptographically secure random 10 hex characters.
    """
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:10]
    file_id = secrets.token_hex(5)  # 5 bytes = 10 hex characters
    return f"transformed_{task_hash}_{file_id}.csv.enc"


class FilterOperator(str, Enum):
    EQ = "eq"              # Equal to
    NEQ = "neq"            # Not equal to
    GT = "gt"              # Greater than
    GTE = "gte"            # Greater than or equal
    LT = "lt"              # Less than
    LTE = "lte"            # Less than or equal
    IN = "in"              # In sequence
    NOT_IN = "not_in"      # Not in sequence
    CONTAINS = "contains"  # Substring match
    IS_NULL = "is_null"    # Null/NaN check
    NOT_NULL = "not_null"  # Non-null check


class FilterRows(StrictContract):
    """Filter rows based on strict column condition."""

    op: Literal["filter_rows"] = "filter_rows"
    column: ColumnName
    operator: FilterOperator
    value: StrictJsonValue | None = Field(
        default=None,
        description="Comparison operand. Forbidden for is_null/not_null.",
    )
    case_sensitive: bool = Field(
        default=True,
        description="Applies to 'contains' operator. Defaults to True.",
    )

    @model_validator(mode="after")
    def validate_operator_constraints(self) -> "FilterRows":
        op = self.operator
        val = self.value

        # 1. Null test operators: value must be None
        if op in (FilterOperator.IS_NULL, FilterOperator.NOT_NULL):
            if val is not None:
                raise ValueError(f"Operator '{op.value}' must not have a comparison value.")
            return self

        # All other operators require a non-null operand
        if val is None:
            raise ValueError(f"Operator '{op.value}' requires a non-null comparison value.")

        # 2. Substring match: non-empty string operand only
        if op == FilterOperator.CONTAINS:
            if not isinstance(val, str) or not val:
                raise ValueError("Operator 'contains' requires a non-empty string operand.")
            return self

        # 3. Collection membership: non-empty list of scalars, max 1000 items
        if op in (FilterOperator.IN, FilterOperator.NOT_IN):
            if not isinstance(val, list) or len(val) == 0:
                raise ValueError(f"Operator '{op.value}' requires a non-empty list of items.")
            if len(val) > 1000:
                raise ValueError(f"Operator '{op.value}' list exceeds maximum limit of 1000 items.")
            for idx, item in enumerate(val):
                if isinstance(item, (dict, list)) or item is None:
                    raise ValueError(f"Item at index {idx} in '{op.value}' must be a non-null scalar.")
            return self

        # 4. Ordered numeric/datetime comparisons
        if op in (FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE):
            if isinstance(val, bool):
                raise ValueError(f"Operator '{op.value}' does not accept boolean operands.")
            if not isinstance(val, (int, float, str)):
                raise ValueError(f"Operator '{op.value}' requires a numeric or ISO datetime operand.")
            return self

        # 5. Equality: scalar only (dict/list rejected)
        if op in (FilterOperator.EQ, FilterOperator.NEQ):
            if isinstance(val, (dict, list)):
                raise ValueError(f"Operator '{op.value}' does not accept collection operands.")
            return self

        return self


class SelectColumns(StrictContract):
    """Select or drop specific columns."""

    op: Literal["select_columns"] = "select_columns"
    columns: list[ColumnName] = Field(..., min_length=1, max_length=100)
    mode: Literal["keep", "drop"] = Field(default="keep")

    @field_validator("columns")
    @classmethod
    def validate_unique_columns(cls, cols: list[str]) -> list[str]:
        if len(cols) != len(set(cols)):
            raise ValueError("Columns list contains duplicate column names.")
        return cols


class MissingStrategy(str, Enum):
    DROP_ROW = "drop_row"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_VALUE = "fill_value"


class HandleMissing(StrictContract):
    """Handle missing/null values in target columns."""

    op: Literal["handle_missing"] = "handle_missing"
    columns: list[ColumnName] = Field(..., min_length=1, max_length=100)
    strategy: MissingStrategy
    fill_value: StrictJsonValue | None = Field(
        default=None,
        description="Required for fill_value; forbidden for all other strategies.",
    )

    @field_validator("columns")
    @classmethod
    def validate_unique_columns(cls, cols: list[str]) -> list[str]:
        if len(cols) != len(set(cols)):
            raise ValueError("Columns list contains duplicate column names.")
        return cols

    @model_validator(mode="after")
    def validate_strategy_parameters(self) -> "HandleMissing":
        if self.strategy == MissingStrategy.FILL_VALUE:
            if self.fill_value is None:
                raise ValueError("Strategy 'fill_value' requires 'fill_value' to be set.")
            if isinstance(self.fill_value, (dict, list)):
                raise ValueError("Strategy 'fill_value' requires a scalar replacement value.")
        else:
            if self.fill_value is not None:
                raise ValueError(
                    f"'fill_value' is forbidden when strategy is '{self.strategy.value}'."
                )
        return self


class Deduplicate(StrictContract):
    """Remove duplicate rows based on subset or all columns."""

    op: Literal["deduplicate"] = "deduplicate"
    subset: list[ColumnName] | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Subset of columns to evaluate. If None, evaluates all columns.",
    )
    keep: Literal["first", "last"] = Field(default="first")

    @field_validator("subset")
    @classmethod
    def validate_unique_subset(cls, cols: list[str] | None) -> list[str] | None:
        if cols is not None and len(cols) != len(set(cols)):
            raise ValueError("Deduplicate subset contains duplicate column names.")
        return cols


class RenameColumns(StrictContract):
    """Rename columns via explicit mapping dictionary."""

    op: Literal["rename_columns"] = "rename_columns"
    mapping: dict[ColumnName, ColumnName] = Field(..., min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_rename_mapping(self) -> "RenameColumns":
        for src, dst in self.mapping.items():
            if src == dst:
                raise ValueError(f"Self-mapping detected: column '{src}' is mapped to itself.")
        targets = list(self.mapping.values())
        if len(targets) != len(set(targets)):
            raise ValueError("Rename targets contain duplicate column names.")
        return self


class TargetDataType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    DATETIME = "datetime"


class CastColumns(StrictContract):
    """Explicitly cast columns to validated target data types."""

    op: Literal["cast_columns"] = "cast_columns"
    casts: dict[ColumnName, TargetDataType] = Field(..., min_length=1, max_length=100)


TransformationOperation = Annotated[
    FilterRows | SelectColumns | HandleMissing | Deduplicate | RenameColumns | CastColumns,
    Field(discriminator="op"),
]


class TransformDatasetInput(StrictContract):
    """Input contract for transform_dataset."""

    file_path: str = Field(..., min_length=1, description="Source dataset path.")
    operations: list[TransformationOperation] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Ordered sequence of transformations (max 20).",
    )


class OperationSummary(StrictContract):
    """Execution telemetry for a single applied operation."""

    op: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    details: dict[str, StrictJsonValue] = Field(default_factory=dict)


class TransformDatasetOutput(StrictContract):
    """Output contract for transform_dataset."""

    status: Literal["success"] = "success"
    summary: str
    initial_row_count: int
    final_row_count: int
    initial_column_count: int
    final_column_count: int
    columns: list[str]
    operations_applied: list[OperationSummary]
    artifacts: list[str] = Field(
        ...,
        min_length=1,
        max_length=1,
        description="List containing the absolute path string of the encrypted transformed dataset.",
    )

