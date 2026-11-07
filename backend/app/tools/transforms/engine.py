"""Transformation execution engine implementing sequential dataset operations."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.tools.registry import current_cancellation_check
from app.tools.schemas import StrictJsonValue
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
    TransformationOperation,
)

MAX_ROWS = 100_000
MAX_COLS = 200
MAX_CELLS = 10_000_000


def check_dataset_bounds(df: pd.DataFrame) -> None:
    """Verify DataFrame conforms to strict memory and dimension bounds."""
    rows = len(df)
    cols = len(df.columns)
    cells = rows * cols
    if rows > MAX_ROWS:
        raise ValueError(f"Dataset exceeds maximum row limit: {rows} > {MAX_ROWS}.")
    if cols > MAX_COLS:
        raise ValueError(f"Dataset exceeds maximum column limit: {cols} > {MAX_COLS}.")
    if cells > MAX_CELLS:
        raise ValueError(f"Dataset exceeds maximum cell limit: {cells} > {MAX_CELLS}.")


def _normalize_datetime_operands(
    series: pd.Series, target_val: str, col_name: str, op_value: str
) -> tuple[pd.Series, pd.Timestamp]:
    """Normalize datetime series and comparison target per B1.2 timezone semantics."""
    try:
        target_dt = pd.to_datetime(target_val)
    except Exception as exc:
        raise ValueError(
            f"Invalid ISO datetime operand for operator '{op_value}'."
        ) from exc

    def _parse_item(item: Any) -> Any:
        if pd.isna(item) or item is None:
            return pd.NaT
        if isinstance(item, pd.Timestamp):
            return item
        try:
            return pd.to_datetime(item)
        except Exception:
            return pd.NaT

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed_series = series.copy()
    else:
        parsed_series = series.apply(_parse_item)

    if parsed_series.notna().sum() == 0 and series.notna().sum() > 0:
        raise ValueError(
            f"Cannot perform datetime comparison on column '{col_name}'."
        )

    target_aware = getattr(target_dt, "tzinfo", None) is not None
    has_aware = False
    has_naive = False

    if pd.api.types.is_datetime64tz_dtype(parsed_series):
        has_aware = True
    elif pd.api.types.is_datetime64_dtype(parsed_series):
        has_naive = True
    else:
        for item in parsed_series.dropna():
            if getattr(item, "tzinfo", None) is not None:
                has_aware = True
            else:
                has_naive = True
            if has_aware and has_naive:
                break

    if target_aware or has_aware:
        norm_target = target_dt.tz_convert("UTC") if target_aware else target_dt.tz_localize("UTC")

        def _to_utc(item: Any) -> Any:
            if pd.isna(item):
                return pd.NaT
            if not isinstance(item, pd.Timestamp):
                try:
                    item = pd.to_datetime(item)
                except Exception:
                    return pd.NaT
            if item.tzinfo is None:
                return item.tz_localize("UTC")
            return item.tz_convert("UTC")

        converted = parsed_series.apply(_to_utc)
        norm_series = pd.to_datetime(converted, utc=True)
    else:
        norm_target = target_dt
        norm_series = pd.to_datetime(parsed_series)

    return norm_series, norm_target


def execute_filter_rows(
    df: pd.DataFrame, op: FilterRows
) -> tuple[pd.DataFrame, dict[str, StrictJsonValue]]:
    """Execute FilterRows operation."""
    if op.column not in df.columns:
        raise ValueError(f"Column '{op.column}' does not exist in dataset.")

    series = df[op.column]
    operator = op.operator
    val = op.value

    if operator == FilterOperator.IS_NULL:
        mask = series.isna()
    elif operator == FilterOperator.NOT_NULL:
        mask = series.notna()
    elif operator == FilterOperator.CONTAINS:
        target_str = str(val)
        mask = (
            series.astype(str).str.contains(
                target_str, case=op.case_sensitive, regex=False, na=False
            )
            & series.notna()
        )
    elif operator == FilterOperator.IN:
        items = list(val) if isinstance(val, (list, tuple, set)) else [val]
        mask = series.isin(items) & series.notna()
    elif operator == FilterOperator.NOT_IN:
        items = list(val) if isinstance(val, (list, tuple, set)) else [val]
        mask = (~series.isin(items)) & series.notna()
    elif operator == FilterOperator.EQ:
        mask = (series == val) & series.notna()
    elif operator == FilterOperator.NEQ:
        mask = (series != val) & series.notna()
    elif operator in (
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    ):
        if isinstance(val, (int, float)):
            if not pd.api.types.is_numeric_dtype(series):
                num_s = pd.to_numeric(series, errors="coerce")
                if num_s.notna().sum() == 0 and series.notna().sum() > 0:
                    raise ValueError(
                        f"Cannot perform numeric comparison on non-numeric column '{op.column}'."
                    )
            else:
                num_s = series
            if operator == FilterOperator.GT:
                mask = (num_s > val) & num_s.notna()
            elif operator == FilterOperator.GTE:
                mask = (num_s >= val) & num_s.notna()
            elif operator == FilterOperator.LT:
                mask = (num_s < val) & num_s.notna()
            else:
                mask = (num_s <= val) & num_s.notna()
        elif isinstance(val, str):
            dt_s, target_dt = _normalize_datetime_operands(
                series, val, op.column, operator.value
            )
            if operator == FilterOperator.GT:
                mask = (dt_s > target_dt) & dt_s.notna()
            elif operator == FilterOperator.GTE:
                mask = (dt_s >= target_dt) & dt_s.notna()
            elif operator == FilterOperator.LT:
                mask = (dt_s < target_dt) & dt_s.notna()
            else:
                mask = (dt_s <= target_dt) & dt_s.notna()
        else:
            raise ValueError(
                f"Unsupported operand type '{type(val).__name__}' for operator '{operator.value}'."
            )
    else:
        raise ValueError(f"Unsupported filter operator: '{operator.value}'.")

    filtered_df = df[mask].copy()
    details: dict[str, StrictJsonValue] = {
        "column": op.column,
        "operator": operator.value,
        "rows_removed": len(df) - len(filtered_df),
    }
    return filtered_df, details


def execute_select_columns(
    df: pd.DataFrame, op: SelectColumns
) -> tuple[pd.DataFrame, dict[str, StrictJsonValue]]:
    """Execute SelectColumns operation."""
    missing = [c for c in op.columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in dataset: {missing}.")

    if op.mode == "keep":
        new_df = df[op.columns].copy()
    elif op.mode == "drop":
        remaining = [c for c in df.columns if c not in op.columns]
        if not remaining:
            raise ValueError(
                "Cannot drop all columns; at least one column must remain in dataset."
            )
        new_df = df.drop(columns=op.columns).copy()
    else:
        raise ValueError(f"Invalid select mode '{op.mode}'.")

    details: dict[str, StrictJsonValue] = {
        "columns": op.columns,
        "mode": op.mode,
        "cols_affected": len(op.columns),
    }
    return new_df, details


def execute_handle_missing(
    df: pd.DataFrame, op: HandleMissing
) -> tuple[pd.DataFrame, dict[str, StrictJsonValue]]:
    """Execute HandleMissing operation."""
    missing = [c for c in op.columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in dataset: {missing}.")

    new_df = df.copy()

    if op.strategy == MissingStrategy.DROP_ROW:
        new_df = new_df.dropna(subset=op.columns).copy()
    elif op.strategy in (MissingStrategy.FILL_MEAN, MissingStrategy.FILL_MEDIAN):
        for col in op.columns:
            if not pd.api.types.is_numeric_dtype(new_df[col]):
                raise ValueError(
                    f"Strategy '{op.strategy.value}' requires numeric column, but '{col}' has dtype '{new_df[col].dtype}'."
                )
            if np.isinf(new_df[col]).any():
                raise ValueError(
                    f"Column '{col}' contains infinite values and cannot be imputed with {op.strategy.value}."
                )
            if op.strategy == MissingStrategy.FILL_MEAN:
                stat_val = float(new_df[col].mean())
                if pd.isna(stat_val):
                    raise ValueError(
                        f"Cannot compute mean for column '{col}': all values are missing."
                    )
            else:
                stat_val = float(new_df[col].median())
                if pd.isna(stat_val):
                    raise ValueError(
                        f"Cannot compute median for column '{col}': all values are missing."
                    )
            # Integer columns imputed using mean or median become float64
            if pd.api.types.is_integer_dtype(new_df[col]):
                new_df[col] = new_df[col].astype("float64")
            new_df[col] = new_df[col].fillna(stat_val)
    elif op.strategy == MissingStrategy.FILL_MODE:
        for col in op.columns:
            non_null = new_df[col].dropna()
            if len(non_null) == 0:
                raise ValueError(
                    f"Cannot compute mode for column '{col}': all values are missing."
                )
            counts = non_null.value_counts()
            max_freq = counts.max()
            candidates = counts[counts == max_freq].index.tolist()

            def _canonical_key(item: Any) -> str:
                if isinstance(item, (dict, list)):
                    return json.dumps(item, sort_keys=True, default=str)
                return str(item)

            candidates.sort(key=_canonical_key)
            chosen_mode = candidates[0]
            new_df[col] = new_df[col].fillna(chosen_mode)
    elif op.strategy == MissingStrategy.FILL_VALUE:
        if op.fill_value is None:
            raise ValueError("Strategy 'fill_value' requires a non-null fill_value.")
        for col in op.columns:
            new_df[col] = new_df[col].fillna(op.fill_value)
    else:
        raise ValueError(f"Unsupported missing strategy: '{op.strategy.value}'.")

    details: dict[str, StrictJsonValue] = {
        "columns": op.columns,
        "strategy": op.strategy.value,
    }
    return new_df, details


def execute_deduplicate(
    df: pd.DataFrame, op: Deduplicate
) -> tuple[pd.DataFrame, dict[str, StrictJsonValue]]:
    """Execute Deduplicate operation."""
    if op.subset is not None:
        missing = [c for c in op.subset if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) not found in dataset: {missing}.")

    new_df = df.drop_duplicates(subset=op.subset, keep=op.keep).copy()
    details: dict[str, StrictJsonValue] = {
        "subset": op.subset if op.subset is not None else [],
        "keep": op.keep,
        "duplicates_removed": len(df) - len(new_df),
    }
    return new_df, details


def execute_rename_columns(
    df: pd.DataFrame, op: RenameColumns
) -> tuple[pd.DataFrame, dict[str, StrictJsonValue]]:
    """Execute RenameColumns operation."""
    missing = [c for c in op.mapping.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in dataset: {missing}.")

    for src, dst in op.mapping.items():
        if dst in df.columns and dst not in op.mapping:
            raise ValueError(
                f"Rename collision: target column '{dst}' already exists in dataset."
            )

    new_df = df.rename(columns=op.mapping).copy()
    details: dict[str, StrictJsonValue] = {
        "mapping": {k: str(v) for k, v in op.mapping.items()}
    }
    return new_df, details


def execute_cast_columns(
    df: pd.DataFrame, op: CastColumns
) -> tuple[pd.DataFrame, dict[str, StrictJsonValue]]:
    """Execute CastColumns operation."""
    missing = [c for c in op.casts.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in dataset: {missing}.")

    new_df = df.copy()

    for col, target_type in op.casts.items():
        if target_type == TargetDataType.INT:
            non_null = new_df[col].dropna()
            if len(non_null) > 0:
                try:
                    s_num = pd.to_numeric(new_df[col], errors="raise")
                except Exception as exc:
                    raise ValueError(
                        f"Cannot cast column '{col}' to int: unparseable numeric values."
                    ) from exc
                num_non_null = s_num.dropna()
                if np.isinf(num_non_null).any():
                    raise ValueError(
                        f"Cannot cast column '{col}' to int: column contains infinite values."
                    )
                if not (num_non_null == num_non_null.round()).all():
                    raise ValueError(
                        f"Lossy cast to int: column '{col}' contains non-integer float values."
                    )
                new_df[col] = s_num.astype("Int64")
            else:
                new_df[col] = new_df[col].astype("Int64")

        elif target_type == TargetDataType.FLOAT:
            try:
                new_df[col] = pd.to_numeric(new_df[col], errors="raise").astype("float64")
            except Exception as exc:
                raise ValueError(
                    f"Cannot cast column '{col}' to float: unparseable numeric values."
                ) from exc

        elif target_type == TargetDataType.STR:
            new_df[col] = new_df[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)

        elif target_type == TargetDataType.BOOL:
            bool_map = {
                True: True,
                1: True,
                1.0: True,
                "true": True,
                "t": True,
                "yes": True,
                "y": True,
                "1": True,
                False: False,
                0: False,
                0.0: False,
                "false": False,
                "f": False,
                "no": False,
                "n": False,
                "0": False,
            }

            def _to_bool(val: Any) -> Any:
                if pd.isna(val):
                    return np.nan
                if isinstance(val, (int, float)):
                    if val in (1, 1.0):
                        return True
                    if val in (0, 0.0):
                        return False
                    raise ValueError(
                        f"Lossy/invalid cast to bool: column '{col}' contains numeric value '{val}'."
                    )
                if isinstance(val, str):
                    key = val.strip().lower()
                    if key in bool_map:
                        return bool_map[key]
                    raise ValueError(
                        f"Lossy/invalid cast to bool: column '{col}' contains string value '{val}'."
                    )
                if val in bool_map:
                    return bool_map[val]
                raise ValueError(
                    f"Lossy/invalid cast to bool: column '{col}' contains unrecognized value '{val}'."
                )

            new_df[col] = new_df[col].apply(_to_bool).astype("boolean")

        elif target_type == TargetDataType.DATETIME:
            try:
                new_df[col] = pd.to_datetime(new_df[col], errors="raise")
            except Exception as exc:
                raise ValueError(
                    f"Cannot cast column '{col}' to datetime: unparseable datetime values."
                ) from exc

        else:
            raise ValueError(f"Unsupported target data type: '{target_type.value}'.")

    details: dict[str, StrictJsonValue] = {
        "casts": {k: v.value for k, v in op.casts.items()}
    }
    return new_df, details


def apply_transformations(
    df: pd.DataFrame,
    operations: list[TransformationOperation],
    cancellation_check: Callable[[], bool] | None = None,
) -> tuple[pd.DataFrame, list[OperationSummary]]:
    """Sequentially apply validated transformation operations with bounds and cancellation enforcement."""
    check_dataset_bounds(df)
    summaries: list[OperationSummary] = []
    current_df = df.copy()

    check_cancelled = cancellation_check or current_cancellation_check()

    for idx, op in enumerate(operations):
        if check_cancelled is not None and check_cancelled():
            raise asyncio.CancelledError(
                f"Transformation sequence cancelled before operation {idx + 1}."
            )

        rows_before = len(current_df)
        cols_before = len(current_df.columns)

        if op.op == "filter_rows":
            current_df, details = execute_filter_rows(current_df, op)
        elif op.op == "select_columns":
            current_df, details = execute_select_columns(current_df, op)
        elif op.op == "handle_missing":
            current_df, details = execute_handle_missing(current_df, op)
        elif op.op == "deduplicate":
            current_df, details = execute_deduplicate(current_df, op)
        elif op.op == "rename_columns":
            current_df, details = execute_rename_columns(current_df, op)
        elif op.op == "cast_columns":
            current_df, details = execute_cast_columns(current_df, op)
        else:
            raise ValueError(f"Unsupported transformation operation: '{op.op}'.")

        check_dataset_bounds(current_df)

        summaries.append(
            OperationSummary(
                op=op.op,
                rows_before=rows_before,
                rows_after=len(current_df),
                cols_before=cols_before,
                cols_after=len(current_df.columns),
                details=details,
            )
        )

    return current_df, summaries

