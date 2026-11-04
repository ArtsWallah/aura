"""Phase 3.3.1-B2: Tests for runtime tool output contract validation and fail-closed behavior."""

import asyncio
import time
import pytest
from pydantic import BaseModel, Field, ValidationError

from app.agent.execution_handoff import (
    ExecutionHandoffError,
    authorized_step_scope,
    authorized_tool_invocation,
    direct_read_only_tool_invocation,
    test_legacy_tool_invocation as legacy_tool_scope,
)
from app.agent.executor import PlanExecutor
from app.agent.orchestrator import AgentOrchestrator
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus, TaskEvent, TaskState, TaskStatus
from app.core.config import settings
from app.privacy.encryption import encryption_service
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.schemas import (
    ExecutionLocation,
    PermissionLevel,
    StrictContract,
    StrictJsonValue,
    ToolCategory,
    ToolResult,
)
from app.tools import (
    InspectDatasetOutput,
    DescriptiveStatisticsOutput,
    GroupComparisonOutput,
    CorrelationAnalysisOutput,
    DetectOutliersOutput,
    GenerateChartOutput,
    ReadDocumentOutput,
    SearchDocumentChunksOutput,
    GenerateReportOutput,
    WebResearchOutput,
    SystemHealthOutput,
)


class DummyProbeInput(BaseModel):
    mode: str = "valid"


class DummyChildContract(StrictContract):
    score: float
    tag: str


class DummyProbeOutput(StrictContract):
    status: str
    code: int
    details: DummyChildContract
    metadata: list[dict[str, StrictJsonValue]] = Field(default_factory=list)


@pytest.fixture
def isolated_registry():
    """Returns an isolated ToolRegistry for controlled failure injections."""
    registry = ToolRegistry()

    def dummy_executor(params: DummyProbeInput):
        if params.mode == "valid":
            return {
                "status": "success",
                "code": 200,
                "details": {"score": 99.5, "tag": "prod"},
                "metadata": [{"env": "test", "active": True}],
            }
        elif params.mode == "missing_field":
            # Missing 'code' and 'details'
            return {"status": "success"}
        elif params.mode == "unexpected_field":
            # Extra field injected into StrictContract
            return {
                "status": "success",
                "code": 200,
                "details": {"score": 99.5, "tag": "prod"},
                "metadata": [],
                "unauthorized_extra": "leak",
            }
        elif params.mode == "invalid_type":
            # 'code' should be int, given string
            return {
                "status": "success",
                "code": "NOT_AN_INT",
                "details": {"score": 99.5, "tag": "prod"},
                "metadata": [],
            }
        elif params.mode == "unsupported_python_object":
            # Pass a Python function into JsonValue
            return {
                "status": "success",
                "code": 200,
                "details": {"score": 99.5, "tag": "prod"},
                "metadata": [{"fn": lambda x: x}],
            }
        elif params.mode == "none_return":
            return None
        return {}

    registry.register(
        name="probe_tool",
        category=ToolCategory.SYSTEM,
        description="Test probe tool for runtime validation.",
        input_model=DummyProbeInput,
        output_model=DummyProbeOutput,
        execution_location=ExecutionLocation.LOCAL,
        permission_level=PermissionLevel.READ_ONLY,
        retry_safe=False,
    )(dummy_executor)

    return registry


# 1. Valid output passes validation
@pytest.mark.asyncio
async def test_valid_output_passes_runtime_validation(isolated_registry):
    with legacy_tool_scope("probe_tool", {"mode": "valid"}):
        result = await isolated_registry.execute_tool(
            "probe_tool", {"mode": "valid"}, user_approval_granted=True
        )
    assert result.success is True
    assert result.error is None
    assert result.data["status"] == "success"
    assert result.data["code"] == 200
    assert result.data["details"]["tag"] == "prod"


# 2. Missing required output fields fail
@pytest.mark.asyncio
async def test_missing_required_output_fields_fail(isolated_registry):
    with legacy_tool_scope("probe_tool", {"mode": "missing_field"}):
        result = await isolated_registry.execute_tool(
            "probe_tool", {"mode": "missing_field"}, user_approval_granted=True
        )
    assert result.success is False
    assert result.error == "Output validation failed for tool 'probe_tool'."
    assert result.data is None


# 3. Unexpected fields fail
@pytest.mark.asyncio
async def test_unexpected_fields_fail_runtime_validation(isolated_registry):
    with legacy_tool_scope("probe_tool", {"mode": "unexpected_field"}):
        result = await isolated_registry.execute_tool(
            "probe_tool", {"mode": "unexpected_field"}, user_approval_granted=True
        )
    assert result.success is False
    assert result.error == "Output validation failed for tool 'probe_tool'."
    assert result.data is None


# 4. Invalid field types fail
@pytest.mark.asyncio
async def test_invalid_field_types_fail_runtime_validation(isolated_registry):
    with legacy_tool_scope("probe_tool", {"mode": "invalid_type"}):
        result = await isolated_registry.execute_tool(
            "probe_tool", {"mode": "invalid_type"}, user_approval_granted=True
        )
    assert result.success is False
    assert result.error == "Output validation failed for tool 'probe_tool'."
    assert result.data is None


# 5. Unsupported Python objects fail
@pytest.mark.asyncio
async def test_unsupported_python_objects_fail_runtime_validation(isolated_registry):
    with legacy_tool_scope("probe_tool", {"mode": "unsupported_python_object"}):
        result = await isolated_registry.execute_tool(
            "probe_tool", {"mode": "unsupported_python_object"}, user_approval_granted=True
        )
    assert result.success is False
    assert result.error == "Output validation failed for tool 'probe_tool'."
    assert result.data is None


# 5b. None return fails runtime validation
@pytest.mark.asyncio
async def test_none_return_fails_runtime_validation(isolated_registry):
    with legacy_tool_scope("probe_tool", {"mode": "none_return"}):
        result = await isolated_registry.execute_tool(
            "probe_tool", {"mode": "none_return"}, user_approval_granted=True
        )
    assert result.success is False
    assert result.error == "Output validation failed for tool 'probe_tool'."
    assert result.data is None


# 6. Invalid output never enters step_results
@pytest.mark.asyncio
async def test_invalid_output_never_enters_step_results(monkeypatch):
    """When a tool produces invalid output, PlanExecutor marks step FAILED and step_results is not updated."""
    # Temporarily monkeypatch system_health executor to return an invalid payload
    original_executor = tool_registry._executors["system_health"]
    monkeypatch.setitem(
        tool_registry._executors,
        "system_health",
        lambda _params: {"corrupted": "bad_data", "leaked_secret": "secret_123"},
    )

    step = ExecutionStep(
        step_id="step_fail",
        tool_name="system_health",
        parameters={"check_storage": True},
        title="Health Check Step",
        description="Health check step description",
        capability="system",
    )
    plan = ExecutionPlan(plan_id="plan_fail", goal="Test Output Failure", steps=[step])
    state = TaskState(task_id="task_fail", objective="Test Failure", status=TaskStatus.RUNNING, plan=plan)

    event_queue = asyncio.Queue()
    executor = PlanExecutor()
    result = await executor.execute_step(step, state, event_queue)

    assert result.success is False
    assert step.status == StepStatus.FAILED
    assert "step_fail" not in state.step_results
    assert "leaked_secret" not in (step.error or "")


# 7. Downstream dependent steps do not execute using invalid output
@pytest.mark.asyncio
async def test_downstream_dependent_steps_do_not_execute_on_invalid_output(monkeypatch, tmp_path):
    """When Step 1 fails output validation, Step 2 is skipped and never executed."""
    step1_invoked = []
    step2_invoked = []

    def corrupt_step1(_params):
        step1_invoked.append(True)
        return {"invalid": "payload"}  # Fails SystemHealthOutput

    def valid_step2(_params):
        step2_invoked.append(True)
        return {"status": "healthy", "python_version": "3.14", "registered_tools_count": 11, "privacy_mode_default": "local_only", "storage": {}, "directories_initialized": True}

    monkeypatch.setitem(tool_registry._executors, "system_health", corrupt_step1)

    step1 = ExecutionStep(
        step_id="step_1",
        tool_name="system_health",
        parameters={},
        title="Prerequisite Step",
        description="Prerequisite step description",
        capability="system",
    )
    step2 = ExecutionStep(
        step_id="step_2",
        tool_name="system_health",
        parameters={},
        dependencies=["step_1"],
        title="Dependent Step",
        description="Dependent step description",
        capability="system",
    )
    plan = ExecutionPlan(plan_id="plan_dag_fail", goal="Test DAG Failure Cascade", steps=[step1, step2])

    import app.agent.orchestrator as orchestrator_module
    from app.agent.checkpoint import SQLiteCheckpointer
    monkeypatch.setattr(orchestrator_module, "checkpointer", SQLiteCheckpointer(tmp_path / "tasks.db"))

    orchestrator = AgentOrchestrator()
    state = await orchestrator._run_task(
        objective="Test DAG",
        supplied_plan=plan,
        task_id="task_dag_fail",
        privacy_mode="local_only",
    )

    assert step1_invoked == [True]
    assert step2_invoked == []  # Step 2 never ran!
    assert state.plan.steps[0].status == StepStatus.FAILED
    assert state.plan.steps[1].status == StepStatus.SKIPPED
    assert state.plan.steps[1].error == "Skipped because a prerequisite failed."
    assert "step_1" not in state.step_results
    assert "step_2" not in state.step_results


# 8. Valid output remains usable by downstream steps
@pytest.mark.asyncio
async def test_valid_output_usable_by_downstream_steps(monkeypatch, tmp_path):
    """When Step 1 passes validation, it populates step_results and downstream steps execute."""
    step1 = ExecutionStep(
        step_id="step_1",
        tool_name="system_health",
        parameters={"check_storage": False},
        title="Health Check Step 1",
        description="Health check step 1 description",
        capability="system",
    )
    step2 = ExecutionStep(
        step_id="step_2",
        tool_name="system_health",
        parameters={"check_storage": False},
        dependencies=["step_1"],
        title="Health Check Step 2",
        description="Health check step 2 description",
        capability="system",
    )
    plan = ExecutionPlan(plan_id="plan_dag_success", goal="Test DAG Success", steps=[step1, step2])

    import app.agent.orchestrator as orchestrator_module
    from app.agent.checkpoint import SQLiteCheckpointer
    monkeypatch.setattr(orchestrator_module, "checkpointer", SQLiteCheckpointer(tmp_path / "tasks.db"))

    orchestrator = AgentOrchestrator()
    state = await orchestrator._run_task(
        objective="Test Success",
        supplied_plan=plan,
        task_id="task_dag_success",
        privacy_mode="local_only",
    )

    assert state.plan.steps[0].status == StepStatus.COMPLETED
    assert state.plan.steps[1].status == StepStatus.COMPLETED
    assert "step_1" in state.step_results
    assert "step_2" in state.step_results
    assert state.step_results["step_1"]["data"]["status"] == "healthy"


# 9. Output validation failures produce safe typed failures
@pytest.mark.asyncio
async def test_output_validation_failure_produces_safe_typed_failure(isolated_registry):
    with legacy_tool_scope("probe_tool", {"mode": "unexpected_field"}):
        result = await isolated_registry.execute_tool(
            "probe_tool", {"mode": "unexpected_field"}, user_approval_granted=True
        )
    assert result.success is False
    # Error message must be concise, typed, and predictable
    assert result.error == "Output validation failed for tool 'probe_tool'."
    # No sensitive keywords or stack trace strings in error
    assert "Traceback" not in result.error
    assert "leak" not in result.error


# 10. Raw invalid output is absent from traces and API responses
@pytest.mark.asyncio
async def test_raw_invalid_output_absent_from_events_and_state(monkeypatch):
    """Ensure that tainted data from an invalid output cannot leak into TaskEvents or step output."""
    secret_marker = "SUPER_SECRET_PAYLOAD_987654"
    monkeypatch.setitem(
        tool_registry._executors,
        "system_health",
        lambda _p: {"corrupted": True, "secret": secret_marker},
    )

    step = ExecutionStep(
        step_id="step_leak_check",
        tool_name="system_health",
        parameters={},
        title="Check Leaks",
        description="Check leaks step description",
        capability="system",
    )
    plan = ExecutionPlan(plan_id="plan_leak", goal="Leak Check", steps=[step])
    state = TaskState(task_id="task_leak", objective="Leak Check", status=TaskStatus.RUNNING, plan=plan)

    event_queue = asyncio.Queue()
    executor = PlanExecutor()
    await executor.execute_step(step, state, event_queue)

    # Collect all emitted events
    events: list[TaskEvent] = []
    while not event_queue.empty():
        events.append(await event_queue.get())

    for ev in events:
        import json
        dumped = json.dumps(ev.model_dump(mode="json"))
        assert secret_marker not in dumped, f"Secret marker leaked in event {ev.event_type}"

    assert secret_marker not in (step.error or "")
    assert step.output is None


# 11. Existing cancellation and authorization behavior remains intact
@pytest.mark.asyncio
async def test_cancellation_and_authorization_remain_intact_with_runtime_validation():
    # 1. Authorization check still blocks execution before tool run
    with pytest.raises(ExecutionHandoffError):
        await tool_registry._execute_tool("system_health", {})

    # 2. Cancellation check before tool start
    step = ExecutionStep(
        step_id="step_cancel",
        tool_name="system_health",
        parameters={},
        title="Cancel Step",
        description="Cancel step description",
        capability="system",
    )
    plan = ExecutionPlan(plan_id="plan_c", goal="Cancel", steps=[step])
    state = TaskState(task_id="task_c", objective="Cancel", status=TaskStatus.CANCEL_REQUESTED, plan=plan)

    executor = PlanExecutor()
    result = await executor.execute_step(step, state)
    assert result.success is False
    assert result.error == "Tool execution cancelled before start."
    assert step.status == StepStatus.SKIPPED


# 12. Deliberate production tool corruption fails closed
@pytest.mark.asyncio
async def test_deliberate_production_tool_corruption_fails_closed(monkeypatch):
    """Corrupt a real tool's output and verify execute_direct_read_only_tool fails closed."""
    monkeypatch.setitem(
        tool_registry._executors,
        "system_health",
        lambda _params: {"invalid_key": "some_value"},
    )
    result = await tool_registry.execute_direct_read_only_tool("system_health", {})
    assert result.success is False
    assert result.error == "Output validation failed for tool 'system_health'."
    assert result.data is None
