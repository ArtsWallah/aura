import pytest
import pandas as pd
from pathlib import Path
from app.core.config import settings
from app.agent.orchestrator import orchestrator
from app.agent.state import TaskStatus, StepStatus
from app.agent.checkpoint import checkpointer
from app.privacy.encryption import encryption_service

@pytest.fixture
def sales_data_file():
    path = settings.UPLOADS_DIR / "agent_test_sales.csv.enc"
    df = pd.DataFrame({
        "category": ["Electronics", "Electronics", "Furniture", "Supplies", "Electronics", "Furniture"],
        "region": ["East", "West", "East", "West", "North", "South"],
        "revenue": [15000, 22000, 4000, 350, 18000, 3100],
        "cost": [9000, 14000, 2500, 300, 11000, 2100]
    })
    path.write_bytes(encryption_service().encrypt_bytes(df.to_csv(index=False).encode("utf-8")))
    yield str(path)
    if path.exists():
        path.unlink()

@pytest.mark.asyncio
async def test_end_to_end_agent_orchestration(sales_data_file):
    objective = "Analyze this dataset, identify sales trends, detect underperforming categories, and generate a report with charts."
    
    state = await orchestrator.run_task(
        objective=objective,
        file_path=sales_data_file,
        privacy_mode="local_only"
    )

    assert state.status == TaskStatus.COMPLETED
    assert state.plan is not None
    assert len(state.plan.steps) >= 3
    
    # Verify all planned steps completed
    for step in state.plan.steps:
        assert step.status == StepStatus.COMPLETED, f"Step {step.step_id} status is {step.status}"

    # Verify report was generated as final deliverable
    assert state.final_deliverable is not None
    assert "report_id" in state.final_deliverable or "markdown_content" in state.final_deliverable
    
    # Verify checkpointer saved state
    loaded = checkpointer.load_checkpoint(state.task_id)
    assert loaded is not None
    assert loaded.status == TaskStatus.COMPLETED
    assert loaded.task_id == state.task_id

