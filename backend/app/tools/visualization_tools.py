import os
import uuid
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
from typing import Any, Literal
from pydantic import BaseModel, Field

from app.tools.schemas import ToolCategory, ExecutionLocation, PermissionLevel
from app.tools.registry import tool_registry
from app.core.config import settings
from app.core.security import validate_safe_path

class GenerateChartInput(BaseModel):
    file_path: str = Field(description="Path to CSV or Excel dataset.")
    chart_type: Literal["bar", "line", "scatter", "histogram", "box", "heatmap"] = Field(
        description="Type of chart to generate."
    )
    title: str = Field(description="Descriptive chart title.")
    x_column: str = Field(description="Column for X-axis (or category for bar/box, primary variable for histogram).")
    y_column: str | None = Field(default=None, description="Column for Y-axis (metric/value).")
    group_column: str | None = Field(default=None, description="Optional categorical column for color grouping/hue.")
    aggregation: Literal["sum", "mean", "count"] | None = Field(default="sum", description="Aggregation for bar charts.")
    top_n: int = Field(default=15, description="Max categories to plot.")

@tool_registry.register(
    name="generate_chart",
    category=ToolCategory.VISUALIZATION,
    description="Renders high-resolution statistical data visualizations (bar, line, scatter, histogram, box, correlation heatmap) and persists PNG charts.",
    input_model=GenerateChartInput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY
)
def generate_chart(params: GenerateChartInput) -> dict[str, Any]:
    safe_path = validate_safe_path(params.file_path)
    df = pd.read_csv(safe_path) if safe_path.suffix.lower() == ".csv" else pd.read_excel(safe_path)
    
    # Modern aesthetics
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    
    # Custom color palette
    primary_color = "#3b82f6" # Indigo / Blue
    accent_palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"]

    chart_type = params.chart_type
    
    if chart_type == "bar":
        if params.y_column:
            agg_df = df.groupby(params.x_column)[params.y_column].agg(params.aggregation or "sum").reset_index()
            agg_df = agg_df.sort_values(by=params.y_column, ascending=False).head(params.top_n)
            bars = ax.bar(agg_df[params.x_column].astype(str), agg_df[params.y_column], color=primary_color, edgecolor="#1e40af", alpha=0.85)
            ax.set_ylabel(f"{params.aggregation.capitalize() if params.aggregation else ''} of {params.y_column}")
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:,.0f}" if height > 100 else f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color="#1e293b")
        else:
            val_counts = df[params.x_column].value_counts().head(params.top_n)
            ax.bar(val_counts.index.astype(str), val_counts.values, color=primary_color, alpha=0.85)
            ax.set_ylabel("Count")
            
        ax.set_xlabel(params.x_column)
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "line":
        if not params.y_column:
            raise ValueError("Line chart requires y_column parameter.")
        if params.group_column and params.group_column in df.columns:
            for i, (group_name, group_data) in enumerate(df.groupby(params.group_column)):
                color = accent_palette[i % len(accent_palette)]
                ax.plot(group_data[params.x_column], group_data[params.y_column], marker="o", label=str(group_name), color=color, linewidth=2)
            ax.legend(title=params.group_column)
        else:
            # Aggregate if multiple entries per x
            agg_df = df.groupby(params.x_column)[params.y_column].mean().reset_index()
            ax.plot(agg_df[params.x_column], agg_df[params.y_column], marker="o", color=primary_color, linewidth=2.5)
            
        ax.set_xlabel(params.x_column)
        ax.set_ylabel(params.y_column)
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "scatter":
        if not params.y_column:
            raise ValueError("Scatter plot requires y_column parameter.")
        if params.group_column and params.group_column in df.columns:
            for i, (group_name, group_data) in enumerate(df.groupby(params.group_column)):
                color = accent_palette[i % len(accent_palette)]
                ax.scatter(group_data[params.x_column], group_data[params.y_column], alpha=0.7, label=str(group_name), color=color, edgecolors="none")
            ax.legend(title=params.group_column)
        else:
            ax.scatter(df[params.x_column], df[params.y_column], color=primary_color, alpha=0.6, edgecolors="none")
            
        ax.set_xlabel(params.x_column)
        ax.set_ylabel(params.y_column)

    elif chart_type == "histogram":
        series = df[params.x_column].dropna()
        n, bins, patches = ax.hist(series, bins=20, color=primary_color, edgecolor="#1e40af", alpha=0.75)
        ax.set_xlabel(params.x_column)
        ax.set_ylabel("Frequency")

    elif chart_type == "box":
        if params.y_column:
            categories = df[params.x_column].dropna().unique()[:params.top_n]
            data_to_plot = [df[df[params.x_column] == cat][params.y_column].dropna() for cat in categories]
            ax.boxplot(data_to_plot, labels=[str(c) for c in categories], patch_artist=True,
                       boxprops=dict(facecolor=primary_color, color="#1e40af", alpha=0.6))
            ax.set_xlabel(params.x_column)
            ax.set_ylabel(params.y_column)
            plt.xticks(rotation=45, ha="right")
        else:
            ax.boxplot(df[params.x_column].dropna(), patch_artist=True,
                       boxprops=dict(facecolor=primary_color, color="#1e40af", alpha=0.6))
            ax.set_xlabel(params.x_column)

    elif chart_type == "heatmap":
        num_df = df.select_dtypes(include=[np.number]).dropna(how="all")
        corr = num_df.corr().values
        cols = num_df.columns.tolist()
        cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="left")
        ax.set_yticklabels(cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr[i, j]) < 0.5 else "white", fontsize=8)

    ax.set_title(params.title, fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    plt.tight_layout()

    # Save to disk
    chart_id = f"chart_{uuid.uuid4().hex[:10]}"
    filename = f"{chart_id}.png"
    out_path = settings.CHARTS_DIR / filename
    plt.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)

    # Convert thumbnail to base64
    with open(out_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{img_b64}"

    return {
        "chart_id": chart_id,
        "title": params.title,
        "chart_type": params.chart_type,
        "relative_path": f"/data/charts/{filename}",
        "file_path": str(out_path),
        "data_uri": data_uri,
        "artifacts": [str(out_path)]
    }
