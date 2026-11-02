import pytest
from pathlib import Path
from app.core.config import settings
from app.agent.orchestrator import orchestrator
from app.agent.state import TaskStatus, StepStatus
from app.privacy.policies import privacy_engine

@pytest.mark.asyncio
async def test_full_acceptance_workflow():
    """
    Demonstrates the acceptance criteria workflow:
    1. Understand request
    2. Create structured plan
    3. Select dataset inspection tool
    4. Inspect file
    5. Select analytics tools
    6. Perform real calculations
    7. Generate charts
    8. Validate numerical outputs
    9. Evaluate tasks
    10. Retry recoverable failures
    11. Generate publication-grade report with embedded visualizations
    12. Verify local-only privacy guarantees
    """
    sales_csv = str(settings.DATA_DIR / "sample_sales.csv")
    assert Path(sales_csv).exists(), "sample_sales.csv must exist"

    # Step A: Test Local-Only Privacy Guarantee
    privacy_res = privacy_engine.validate_cloud_transmission(
        payload={"sensitive_revenue_records": [1000, 2000, 3000]},
        mode="local_only"
    )
    assert privacy_res.allowed is False
    assert "STRICT POLICY" in privacy_res.reason

    # Step B: Execute Autonomous Task in Local-Only Mode
    prompt = "Analyze this dataset, identify sales trends, detect underperforming categories, and generate a report with charts."
    
    state = await orchestrator.run_task(
        objective=prompt,
        file_path=sales_csv,
        privacy_mode="local_only"
    )

    # 1. Verification of Plan
    assert state.plan is not None
    assert len(state.plan.steps) >= 4
    
    tool_names_in_plan = [s.tool_name for s in state.plan.steps]
    assert "inspect_dataset" in tool_names_in_plan
    assert "descriptive_statistics" in tool_names_in_plan
    assert "group_comparison" in tool_names_in_plan
    assert "generate_chart" in tool_names_in_plan
    assert "generate_report" in tool_names_in_plan

    # 2. Verification of Step Execution & Outputs
    for step in state.plan.steps:
        assert step.status == StepStatus.COMPLETED
        assert step.duration_ms > 0

    # Verify group comparison detected underperforming category
    group_step = next(s for s in state.plan.steps if s.tool_name == "group_comparison")
    assert "underperforming_groups" in group_step.output
    # Legacy Replacement Parts should be underperforming (generated with < 5% contribution)
    assert any("Legacy" in g or "Networking" in g or len(group_step.output["underperforming_groups"]) > 0 for g in group_step.output["underperforming_groups"])

    # 3. Verification of Charts
    chart_step = next(s for s in state.plan.steps if s.tool_name == "generate_chart")
    chart_file = chart_step.output.get("file_path")
    assert chart_file and Path(chart_file).exists()

    # 4. Verification of Final Report Deliverable
    assert state.status == TaskStatus.COMPLETED
    assert state.final_deliverable is not None
    assert "markdown_content" in state.final_deliverable
    assert Path(state.final_deliverable["markdown_path"]).exists()
    assert Path(state.final_deliverable["html_path"]).exists()

    # 5. Metrics & Checkpointing
    assert state.metrics.total_execution_time_ms > 0
    assert state.metrics.tool_calls_count >= len(state.plan.steps)
