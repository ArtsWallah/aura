import pandas as pd
import numpy as np
from scipy import stats
from typing import Any, Literal
from pydantic import BaseModel, Field

from app.tools.schemas import ToolCategory, ExecutionLocation, PermissionLevel
from app.tools.registry import tool_registry
from app.core.security import validate_safe_path

# 1. Descriptive Statistics Tool
class DescriptiveStatsInput(BaseModel):
    file_path: str = Field(description="Path to CSV or Excel file.")
    columns: list[str] | None = Field(default=None, description="Optional specific columns to compute stats for. Defaults to all numeric.")

@tool_registry.register(
    name="descriptive_statistics",
    category=ToolCategory.ANALYTICS,
    description="Calculates comprehensive descriptive statistics: mean, median, std, IQR, skewness, kurtosis, min, max, and quartiles for numeric fields.",
    input_model=DescriptiveStatsInput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY
)
def descriptive_statistics(params: DescriptiveStatsInput) -> dict[str, Any]:
    safe_path = validate_safe_path(params.file_path)
    df = pd.read_csv(safe_path) if safe_path.suffix.lower() == ".csv" else pd.read_excel(safe_path)
    
    num_df = df.select_dtypes(include=[np.number])
    if params.columns:
        valid_cols = [c for c in params.columns if c in num_df.columns]
        if not valid_cols:
            raise ValueError(f"None of specified columns {params.columns} are numeric in the dataset.")
        num_df = num_df[valid_cols]

    result = {}
    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) == 0:
            continue
        
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        
        result[col] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "q25": round(q25, 4),
            "median": round(float(series.median()), 4),
            "q75": round(q75, 4),
            "max": round(float(series.max()), 4),
            "iqr": round(float(q75 - q25), 4),
            "skewness": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
            "kurtosis": round(float(series.kurtosis()), 4) if len(series) > 3 else 0.0
        }
        
    return {"statistics": result, "analyzed_columns": list(result.keys())}


# 2. Group Comparison Tool
class GroupComparisonInput(BaseModel):
    file_path: str = Field(description="Path to CSV or Excel file.")
    group_by: str = Field(description="Categorical column to group by.")
    metric_column: str = Field(description="Numerical column to aggregate.")
    aggregations: list[Literal["mean", "sum", "count", "min", "max", "std"]] = Field(
        default=["sum", "mean", "count"],
        description="Aggregations to compute for each group."
    )
    sort_by: Literal["sum", "mean", "count", "min", "max", "std"] = Field(
        default="sum",
        description="Metric to sort results by."
    )
    ascending: bool = Field(default=False, description="Sort order.")
    limit: int = Field(default=20, description="Maximum number of groups to return.")

@tool_registry.register(
    name="group_comparison",
    category=ToolCategory.ANALYTICS,
    description="Groups data by a categorical feature and computes aggregated metrics (sum, mean, count, min, max, std) for comparative segment analysis.",
    input_model=GroupComparisonInput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY
)
def group_comparison(params: GroupComparisonInput) -> dict[str, Any]:
    safe_path = validate_safe_path(params.file_path)
    df = pd.read_csv(safe_path) if safe_path.suffix.lower() == ".csv" else pd.read_excel(safe_path)
    
    if params.group_by not in df.columns:
        raise ValueError(f"Group column '{params.group_by}' does not exist.")
    if params.metric_column not in df.columns:
        raise ValueError(f"Metric column '{params.metric_column}' does not exist.")
        
    grouped = df.groupby(params.group_by)[params.metric_column].agg(params.aggregations).reset_index()
    
    if params.sort_by in grouped.columns:
        grouped = grouped.sort_values(by=params.sort_by, ascending=params.ascending)
    elif len(params.aggregations) > 0 and params.aggregations[0] in grouped.columns:
        grouped = grouped.sort_values(by=params.aggregations[0], ascending=params.ascending)
        
    top_groups = grouped.head(params.limit).replace({np.nan: None})
    records = top_groups.to_dict(orient="records")
    
    # Calculate group totals and percentages
    total_val = float(df[params.metric_column].sum())
    underperforming = []
    
    for r in records:
        for agg in params.aggregations:
            if agg in r and r[agg] is not None:
                r[agg] = round(float(r[agg]), 2)
        if total_val > 0 and "sum" in r and r["sum"] is not None:
            pct = round((r["sum"] / total_val) * 100, 2)
            r["percentage_of_total"] = pct
            if pct < 5.0: # under 5% contribution
                underperforming.append(r[params.group_by])

    return {
        "group_by": params.group_by,
        "metric_column": params.metric_column,
        "total_metric_value": round(total_val, 2),
        "group_count": len(grouped),
        "results": records,
        "underperforming_groups": underperforming
    }


# 3. Correlation Analysis Tool
class CorrelationAnalysisInput(BaseModel):
    file_path: str = Field(description="Path to CSV or Excel file.")
    method: Literal["pearson", "spearman"] = Field(default="pearson", description="Correlation method.")
    threshold: float = Field(default=0.3, description="Minimum absolute correlation to report.")

@tool_registry.register(
    name="correlation_analysis",
    category=ToolCategory.ANALYTICS,
    description="Computes Pearson or Spearman correlation coefficients across numeric columns and identifies prominent positive and negative relationships.",
    input_model=CorrelationAnalysisInput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY
)
def correlation_analysis(params: CorrelationAnalysisInput) -> dict[str, Any]:
    safe_path = validate_safe_path(params.file_path)
    df = pd.read_csv(safe_path) if safe_path.suffix.lower() == ".csv" else pd.read_excel(safe_path)
    
    num_df = df.select_dtypes(include=[np.number]).dropna(how="all")
    if num_df.shape[1] < 2:
        return {"error": "Dataset requires at least 2 numeric columns for correlation analysis."}
        
    corr_matrix = num_df.corr(method=params.method)
    
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            val = float(corr_matrix.loc[col1, col2])
            if not np.isnan(val) and abs(val) >= params.threshold:
                pairs.append({
                    "column_1": col1,
                    "column_2": col2,
                    "correlation": round(val, 4),
                    "relationship": "strong_positive" if val >= 0.7 else (
                        "moderate_positive" if val >= 0.4 else (
                            "strong_negative" if val <= -0.7 else (
                                "moderate_negative" if val <= -0.4 else "weak"
                            )
                        )
                    )
                })
                
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    # Format matrix dict for serialization
    matrix_dict = {}
    for col in cols:
        matrix_dict[col] = {c: round(float(corr_matrix.loc[col, c]), 4) if not np.isnan(corr_matrix.loc[col, c]) else 0.0 for c in cols}

    return {
        "method": params.method,
        "analyzed_columns": cols,
        "prominent_relationships": pairs,
        "correlation_matrix": matrix_dict
    }


# 4. Outlier Detection Tool
class OutlierDetectionInput(BaseModel):
    file_path: str = Field(description="Path to CSV or Excel file.")
    column: str = Field(description="Numeric column to detect outliers in.")
    method: Literal["iqr", "zscore"] = Field(default="iqr", description="Method: IQR (interquartile range) or zscore.")
    threshold: float = Field(default=1.5, description="IQR multiplier (typically 1.5) or z-score threshold (typically 3.0).")

@tool_registry.register(
    name="detect_outliers",
    category=ToolCategory.ANALYTICS,
    description="Identifies statistical anomalies and extreme outlier records using IQR or Z-score boundaries.",
    input_model=OutlierDetectionInput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY
)
def detect_outliers(params: OutlierDetectionInput) -> dict[str, Any]:
    safe_path = validate_safe_path(params.file_path)
    df = pd.read_csv(safe_path) if safe_path.suffix.lower() == ".csv" else pd.read_excel(safe_path)
    
    if params.column not in df.columns:
        raise ValueError(f"Column '{params.column}' does not exist.")
        
    series = df[params.column].dropna()
    if not np.issubdtype(series.dtype, np.number):
        raise ValueError(f"Column '{params.column}' is not numeric.")
        
    outliers = []
    if params.method == "iqr":
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - (params.threshold * iqr)
        upper_bound = q75 + (params.threshold * iqr)
        outlier_mask = (df[params.column] < lower_bound) | (df[params.column] > upper_bound)
    else: # zscore
        z_scores = np.abs(stats.zscore(series))
        outlier_indices = series.index[z_scores > params.threshold]
        outlier_mask = df.index.isin(outlier_indices)
        mean_val = float(series.mean())
        std_val = float(series.std())
        lower_bound = mean_val - (params.threshold * std_val)
        upper_bound = mean_val + (params.threshold * std_val)

    outlier_df = df[outlier_mask]
    outlier_count = int(len(outlier_df))
    outlier_percentage = round(float((outlier_count / len(df)) * 100), 2)
    
    sample_outliers = outlier_df.head(10).replace({np.nan: None}).to_dict(orient="records")

    return {
        "column": params.column,
        "method": params.method,
        "total_rows": len(df),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage,
        "lower_bound": round(float(lower_bound), 4),
        "upper_bound": round(float(upper_bound), 4),
        "sample_outliers": sample_outliers
    }

