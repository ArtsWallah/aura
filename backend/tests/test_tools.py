import pytest
import pandas as pd
from pathlib import Path

from app.core.config import settings
from app.tools.registry import tool_registry
from app.tools.dataset_tools import InspectDatasetInput
from app.tools.analytics_tools import DescriptiveStatsInput, GroupComparisonInput, CorrelationAnalysisInput, OutlierDetectionInput
from app.tools.visualization_tools import GenerateChartInput
from app.tools.report_tools import GenerateReportInput
from app.privacy.encryption import encryption_service

@pytest.fixture
def sample_sales_csv():
    # Create sample sales CSV
    file_path = settings.UPLOADS_DIR / "sample_sales_test.csv.enc"
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

@pytest.mark.asyncio
async def test_tool_registry_registration():
    tools = tool_registry.list_tools()
    tool_names = [t.name for t in tools]
    assert "inspect_dataset" in tool_names
    assert "descriptive_statistics" in tool_names
    assert "group_comparison" in tool_names
    assert "correlation_analysis" in tool_names
    assert "detect_outliers" in tool_names
    assert "generate_chart" in tool_names
    assert "generate_report" in tool_names

@pytest.mark.asyncio
async def test_inspect_dataset(sample_sales_csv):
    result = await tool_registry.execute_tool(
        "inspect_dataset",
        {"file_path": sample_sales_csv, "sample_rows": 3}
    )
    assert result.success is True
    assert result.data["total_rows"] == 8
    assert result.data["total_columns"] == 6
    assert "revenue" in result.data["numeric_columns"]
    assert "category" in result.data["categorical_columns"]

@pytest.mark.asyncio
async def test_analytics_and_group_comparison(sample_sales_csv):
    # Test descriptive stats
    stats_res = await tool_registry.execute_tool(
        "descriptive_statistics",
        {"file_path": sample_sales_csv, "columns": ["revenue", "profit"]}
    )
    assert stats_res.success is True
    assert "revenue" in stats_res.data["statistics"]
    assert stats_res.data["statistics"]["revenue"]["mean"] > 0

    # Test group comparison
    group_res = await tool_registry.execute_tool(
        "group_comparison",
        {
            "file_path": sample_sales_csv,
            "group_by": "category",
            "metric_column": "revenue",
            "aggregations": ["sum", "mean", "count"]
        }
    )
    assert group_res.success is True
    assert group_res.data["group_by"] == "category"
    assert len(group_res.data["results"]) >= 3
    # Check that Accessories is detected as underperforming or low contributor
    underperforming = group_res.data["underperforming_groups"]
    assert "Accessories" in underperforming

@pytest.mark.asyncio
async def test_visualization_generation(sample_sales_csv):
    chart_res = await tool_registry.execute_tool(
        "generate_chart",
        {
            "file_path": sample_sales_csv,
            "chart_type": "bar",
            "title": "Revenue by Product Category",
            "x_column": "category",
            "y_column": "revenue"
        }
    )
    assert chart_res.success is True
    assert Path(chart_res.data["file_path"]).exists()
    assert chart_res.data["data_uri"].startswith("data:image/png;base64,")

@pytest.mark.asyncio
async def test_report_generation(sample_sales_csv):
    report_res = await tool_registry.execute_tool(
        "generate_report",
        {
            "title": "Quarterly Sales Performance Analysis",
            "executive_summary": "Electronics is the dominant revenue driver while Accessories underperforms.",
            "findings": [
                "Electronics accounts for over 70% of total revenue.",
                "Accessories contributed less than 5% of overall sales."
            ],
            "recommendations": [
                "Scale investment in high-margin Electronics line.",
                "Review marketing strategy for underperforming categories."
            ]
        },
        user_approval_granted=True,
    )
    assert report_res.success is True
    assert Path(report_res.data["markdown_path"]).exists()
    assert Path(report_res.data["html_path"]).exists()
    assert "# Quarterly Sales Performance Analysis" in report_res.data["markdown_content"]
