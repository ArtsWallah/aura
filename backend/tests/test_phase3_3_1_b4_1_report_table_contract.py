"""
Phase 3.3.1-B4.1: Report Table Contract and Rendering Review Test Suite.
Verifies:
1. GenerateReportInput strict contract enforcement with StrictJsonValue.
2. Dynamic column flexibility while rejecting unsupported Python objects.
3. Accurate and safe rendering into Markdown and HTML.
4. Safe escaping of special characters, pipes, and HTML injection vectors.
5. End-to-end analytics workflow: inspect_dataset -> group_comparison -> generate_report.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from app.agent.execution_handoff import (
    ExecutionHandoff,
    _HANDOFF_FACTORY_TOKEN,
    _intent_digest,
    _plan_digest,
)
from app.agent.intelligence_contracts import (
    EvaluationStatus,
    IntentCategory,
    ProviderIdentity,
)
from app.agent.orchestrator import AgentOrchestrator
from app.agent.plan_validation import validate_execution_plan
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus, TaskStatus
from app.core.config import settings
from app.privacy.encryption import encryption_service
from app.tools.registry import tool_registry
from app.tools.report_tools import GenerateReportInput, GenerateReportOutput


def _make_step(
    step_id: str,
    tool_name: str,
    parameters: dict[str, Any],
    dependencies: list[str] | None = None,
    title: str = "Test Step",
    description: str = "Test Step Description",
    capability: str = "system",
) -> ExecutionStep:
    return ExecutionStep(
        step_id=step_id,
        title=title,
        description=description,
        capability=capability,
        tool_name=tool_name,
        parameters=parameters,
        dependencies=dependencies or [],
    )


def _make_plan(
    steps: list[ExecutionStep],
    plan_id: str = "plan_b4_1",
    goal: str = "Test Report Table Rendering",
) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        goal=goal,
        steps=steps,
    )


@pytest.fixture
def encrypted_sales_csv():
    """Create sample encrypted sales CSV in settings.UPLOADS_DIR."""
    file_path = settings.UPLOADS_DIR / "sample_sales_b4_1_test.csv.enc"
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


def _make_handoff(plan: ExecutionPlan, task_id: str = "task_b4_1_test") -> ExecutionHandoff:
    valid_plan = validate_execution_plan(plan.model_copy(deep=True))
    return ExecutionHandoff(
        _factory_token=_HANDOFF_FACTORY_TOKEN,
        handoff_id=f"handoff_{task_id}",
        plan=valid_plan,
        privacy_mode="hybrid",
        provider_identity=ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER,
        intent_category=IntentCategory.DATA_ANALYSIS,
        intent_confidence=1.0,
        intent_is_multi_step=True,
        required_tool_groups=[["dataset", "analytics", "visualization", "report"]],
        evaluation_status=EvaluationStatus.PLAN_VALID,
        objective_coverage=1.0,
        validation_registry_id=tool_registry.registry_id,
        plan_digest=_plan_digest(valid_plan),
        intent_digest=_intent_digest(1.0, True, [["dataset", "analytics", "visualization", "report"]]),
    )


# ==============================================================================
# SECTION 1: Input Validation & Contract Enforcement
# ==============================================================================

def test_generate_report_input_contract_rejects_unsupported_python_objects():
    """Verify GenerateReportInput rejects non-JSON Python objects in tables."""
    class CustomObject:
        pass

    # 1. Custom class instance rejected
    with pytest.raises(ValidationError):
        GenerateReportInput(
            title="Report",
            executive_summary="Summary",
            findings=["Finding 1"],
            tables=[{"col1": CustomObject()}],
        )

    # 2. Function/lambda rejected
    with pytest.raises(ValidationError):
        GenerateReportInput(
            title="Report",
            executive_summary="Summary",
            findings=["Finding 1"],
            tables=[{"col1": lambda x: x}],
        )

    # 3. Python set rejected (not valid JSON)
    with pytest.raises(ValidationError):
        GenerateReportInput(
            title="Report",
            executive_summary="Summary",
            findings=["Finding 1"],
            tables=[{"col1": {1, 2, 3}}],
        )


def test_generate_report_input_contract_rejects_unknown_fields():
    """Verify GenerateReportInput enforces extra='forbid' from StrictContract."""
    with pytest.raises(ValidationError):
        GenerateReportInput(
            title="Report",
            executive_summary="Summary",
            findings=["Finding 1"],
            unexpected_field="disallowed",
        )


def test_generate_report_input_accepts_valid_dynamic_analytics_tables():
    """
    Verify GenerateReportInput accepts arbitrary column names and valid JSON types:
    strings, ints, floats, booleans, nulls (None), nested lists, and nested dicts.
    """
    valid_tables = [
        {
            "category": "Electronics",
            "mean_revenue": 16000.0,
            "count": 3,
            "is_active": True,
            "notes": None,
            "tags": ["core", "high-margin"],
            "nested_meta": {"tier": 1, "priority": "high"},
        },
        {
            "category": "Clothing",
            "mean_revenue": 3000.0,
            "count": 2,
            "is_active": False,
            "notes": "seasonal adjustment",
            "tags": ["retail"],
            "nested_meta": {"tier": 2, "priority": "standard"},
        },
    ]

    inp = GenerateReportInput(
        title="Valid Report",
        executive_summary="Summary",
        findings=["Finding 1"],
        tables=valid_tables,
    )
    assert len(inp.tables) == 2
    assert inp.tables[0]["category"] == "Electronics"
    assert inp.tables[0]["mean_revenue"] == 16000.0
    assert inp.tables[0]["notes"] is None


# ==============================================================================
# SECTION 2: Table Rendering Correctness (Markdown and HTML)
# ==============================================================================

@pytest.mark.asyncio
async def test_rendering_column_names_and_multiple_rows():
    """Verify column headers and rows are rendered accurately into both Markdown and HTML."""
    tables_data = [
        {"segment": "Enterprise", "q1_revenue": 50000.0, "active_clients": 12},
        {"segment": "SMB", "q1_revenue": 25000.0, "active_clients": 45},
    ]

    res = await tool_registry.execute_tool(
        "generate_report",
        {
            "title": "Revenue by Segment",
            "executive_summary": "Quarterly breakdown.",
            "findings": ["Enterprise generates 66% of revenue."],
            "tables": tables_data,
        },
        user_approval_granted=True,
    )
    assert res.success is True

    md = res.data["markdown_content"]
    assert "## Detailed Data Tables" in md
    assert "| segment | q1_revenue | active_clients |" in md
    assert "| --- | --- | --- |" in md
    assert "| Enterprise | 50000.0 | 12 |" in md
    assert "| SMB | 25000.0 | 45 |" in md

    # Verify HTML
    html_bytes = Path(res.data["html_path"]).read_bytes()
    html = encryption_service().decrypt_bytes(html_bytes).decode("utf-8")
    assert "<h2>Detailed Data Tables</h2>" in html
    assert "<th>segment</th><th>q1_revenue</th><th>active_clients</th>" in html
    assert "<td>Enterprise</td><td>50000.0</td><td>12</td>" in html
    assert "<td>SMB</td><td>25000.0</td><td>45</td>" in html


@pytest.mark.asyncio
async def test_rendering_null_values_and_empty_tables():
    """
    Verify:
    1. Null (None) values render gracefully without crashing or throwing TypeError.
    2. Empty tables (tables=[]) cleanly omit the table section.
    3. Empty rows (tables=[{}]) cleanly omit the table section.
    """
    # 1. Null values render as empty cells
    res_null = await tool_registry.execute_tool(
        "generate_report",
        {
            "title": "Null Handling",
            "executive_summary": "Testing nulls.",
            "findings": ["Check null cells."],
            "tables": [{"col_a": "present", "col_b": None}],
        },
        user_approval_granted=True,
    )
    assert res_null.success is True
    md_null = res_null.data["markdown_content"]
    assert "| col_a | col_b |" in md_null
    assert "| present |  |" in md_null

    html_null = encryption_service().decrypt_bytes(Path(res_null.data["html_path"]).read_bytes()).decode("utf-8")
    assert "<td>present</td><td></td>" in html_null

    # 2. Empty tables (tables=[]) omits table section
    res_empty = await tool_registry.execute_tool(
        "generate_report",
        {
            "title": "Empty Table",
            "executive_summary": "No tables supplied.",
            "findings": ["Standard findings."],
            "tables": [],
        },
        user_approval_granted=True,
    )
    assert res_empty.success is True
    assert "## Detailed Data Tables" not in res_empty.data["markdown_content"]
    html_empty = encryption_service().decrypt_bytes(Path(res_empty.data["html_path"]).read_bytes()).decode("utf-8")
    assert "Detailed Data Tables" not in html_empty

    # 3. Empty row dicts (tables=[{}]) omits table section
    res_empty_row = await tool_registry.execute_tool(
        "generate_report",
        {
            "title": "Empty Row Dict",
            "executive_summary": "Empty dict supplied.",
            "findings": ["Findings."],
            "tables": [{}],
        },
        user_approval_granted=True,
    )
    assert res_empty_row.success is True
    assert "## Detailed Data Tables" not in res_empty_row.data["markdown_content"]


@pytest.mark.asyncio
async def test_rendering_nested_collections():
    """Verify nested dictionaries and lists in table cells are JSON-serialized and escaped."""
    tables_data = [
        {
            "metric": "summary_stats",
            "nested_dict": {"p50": 100, "p95": 250},
            "nested_list": ["alpha", "beta"],
        }
    ]

    res = await tool_registry.execute_tool(
        "generate_report",
        {
            "title": "Nested Data Report",
            "executive_summary": "Nested values.",
            "findings": ["Complex structures present."],
            "tables": tables_data,
        },
        user_approval_granted=True,
    )
    assert res.success is True
    md = res.data["markdown_content"]
    assert '{"p50": 100, "p95": 250}' in md
    assert '["alpha", "beta"]' in md

    html = encryption_service().decrypt_bytes(Path(res.data["html_path"]).read_bytes()).decode("utf-8")
    # Quotes escaped as &quot; in HTML
    assert "{&quot;p50&quot;: 100, &quot;p95&quot;: 250}" in html
    assert "[&quot;alpha&quot;, &quot;beta&quot;]" in html


@pytest.mark.asyncio
async def test_rendering_special_characters_and_html_injection_prevention():
    """
    Verify:
    1. HTML injection vectors (<script>, <img onerror>, etc.) are strictly escaped in HTML.
    2. Markdown pipe characters ('|') in headers and cell values are escaped with '\\|'.
    """
    malicious_data = [
        {
            "col | with | pipe": "<script>alert('xss')</script>",
            "exploit_vector": "<img src=x onerror=alert(1)>",
            "nested_attack": {"payload": "<svg/onload=alert(2)>"},
        }
    ]

    res = await tool_registry.execute_tool(
        "generate_report",
        {
            "title": "Security Escaping Test",
            "executive_summary": "Sanitization test.",
            "findings": ["Verifying escaping."],
            "tables": malicious_data,
        },
        user_approval_granted=True,
    )
    assert res.success is True

    # 1. Markdown escaping: pipes must be escaped as \|
    md = res.data["markdown_content"]
    assert "col \\| with \\| pipe" in md

    # 2. HTML escaping: raw scripts and tags MUST NOT appear unescaped
    html = encryption_service().decrypt_bytes(Path(res.data["html_path"]).read_bytes()).decode("utf-8")
    assert "<script>" not in html
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html or "&lt;script&gt;alert('xss')&lt;/script&gt;" in html
    assert "<img src=x" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<svg" not in html


# ==============================================================================
# SECTION 3: End-to-End Analytics Workflow with Dynamic Table Propagation
# ==============================================================================

@pytest.mark.asyncio
async def test_e2e_inspect_to_group_comparison_to_generate_report(encrypted_sales_csv):
    """
    Realistic multi-step pipeline:
      Step 1: inspect_dataset (extracts categorical and numeric columns)
      Step 2: group_comparison (groups by category, sums and averages revenue)
      Step 3: generate_report (dynamically references step_group's data.results into tables)
    Verifies that:
    1. Pipeline completes with TaskStatus.COMPLETED.
    2. Group comparison results are dynamically resolved and passed into generate_report.tables.
    3. The generated Markdown content renders the data table containing category names
       ('Electronics', 'Clothing', 'Home', 'Accessories') and aggregated metrics.
    4. The decrypted HTML report contains the table with escaped cell values and headers.
    """
    plan = _make_plan(
        steps=[
            _make_step(
                step_id="step_inspect",
                tool_name="inspect_dataset",
                parameters={"file_path": encrypted_sales_csv, "sample_rows": 5},
                dependencies=[],
            ),
            _make_step(
                step_id="step_group",
                tool_name="group_comparison",
                parameters={
                    "file_path": encrypted_sales_csv,
                    "group_by": {"$ref": {"step_id": "step_inspect", "path": "data.categorical_columns[1]"}},
                    "metric_column": {"$ref": {"step_id": "step_inspect", "path": "data.numeric_columns[1]"}},
                    "aggregations": ["mean", "sum"],
                },
                dependencies=["step_inspect"],
            ),
            _make_step(
                step_id="step_report",
                tool_name="generate_report",
                parameters={
                    "title": "Category Sales Report",
                    "executive_summary": "Comprehensive analysis of revenue streams across product categories.",
                    "findings": ["Electronics is the top grossing segment."],
                    "recommendations": ["Expand high-demand Electronics inventory."],
                    "tables": {"$ref": {"step_id": "step_group", "path": "data.results"}},
                },
                dependencies=["step_group"],
            ),
        ]
    )

    handoff = _make_handoff(plan, task_id="task_e2e_table_report")
    orchestrator = AgentOrchestrator()
    state = await orchestrator.run_handoff(handoff, task_id="task_e2e_table_report")

    assert state.status == TaskStatus.COMPLETED
    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED
    assert state.plan.steps[2].status == StepStatus.COMPLETED

    report_out = state.step_results["step_report"]["data"]
    md = report_out["markdown_content"]

    # Verify Markdown table contains the group comparison headers and data rows
    assert "## Detailed Data Tables" in md
    assert "category" in md
    assert "mean" in md
    assert "sum" in md
    assert "Electronics" in md
    assert "Clothing" in md
    assert "Home" in md
    assert "Accessories" in md
    assert "48000.0" in md or "16000.0" in md

    # Verify decrypted HTML contains the data-table and rows
    html_path = Path(report_out["html_path"])
    assert html_path.exists()
    html_bytes = html_path.read_bytes()
    html = encryption_service().decrypt_bytes(html_bytes).decode("utf-8")

    assert '<table class="data-table">' in html
    assert "<th>category</th>" in html
    assert "<td>Electronics</td>" in html
    assert "<td>Clothing</td>" in html
    assert "<td>Home</td>" in html
    assert "<td>Accessories</td>" in html
