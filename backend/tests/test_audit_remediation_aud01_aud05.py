import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.execution_handoff import (
    ExecutionHandoff,
    ExecutionHandoffError,
    HandoffFailureCode,
    _HANDOFF_FACTORY_TOKEN,
    authorized_step_scope,
    authorized_tool_invocation,
    direct_read_only_tool_invocation,
    handoff_execution_scope,
    require_authorized_tool_arguments,
)
from app.agent.execution_trace import (
    build_handoff_trace,
    sanitize_execution_trace,
    trace_construction_failure,
)
from app.agent.intelligence_contracts import (
    EvaluationResult,
    EvaluationStatus,
    IntentCategory,
    IssueCode,
    ProviderIdentity,
    StepFailureCategory,
    TraceConstructionStatus,
    ValidationOutcome,
)
from app.agent.orchestrator import AgentOrchestrator
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus, TaskEvent, TaskState, TaskStatus
import app.api.routes.tasks as task_routes
from app.core.config import settings, validate_service_environment
import app.core.config as config_module
from app.core.security import validate_safe_path
from app.privacy.encryption import encryption_service
from app.tools.registry import tool_registry


from app.agent.intelligence_pipeline import IntelligencePipeline


async def _create_test_handoff() -> ExecutionHandoff:
    result = await IntelligencePipeline().run("Check system health", privacy_mode="local_only")
    return ExecutionHandoff.from_pipeline_result(
        result,
        privacy_mode="local_only",
        registry=tool_registry,
    )


# ==============================================================================
# AUD-05: Terminal Trace Consistency & Recovery Persistence
# ==============================================================================

@pytest.mark.asyncio
async def test_trace_builder_returning_none(monkeypatch, tmp_path):
    """When build_handoff_trace returns None, orchestrator must set trace_construction_failure."""
    runner = AgentOrchestrator()
    task_id = "task_trace_builder_none"
    handoff = await _create_test_handoff()
    state = TaskState(
        task_id=task_id,
        objective="private objective",
        privacy_mode="local_only",
        status=TaskStatus.RUNNING,
        plan=handoff.plan.model_copy(deep=True),
    )

    monkeypatch.setattr("app.agent.orchestrator.build_handoff_trace", lambda *args, **kwargs: None)

    ready = runner._finalize_handoff_trace(handoff, state, terminal_status=TaskStatus.COMPLETED)
    assert ready is False
    assert state.execution_trace is not None
    assert state.execution_trace.trace_construction_status == TraceConstructionStatus.FAILED
    assert state.execution_trace.evaluation_status == EvaluationStatus.INCONCLUSIVE
    assert state.execution_evaluation is not None
    assert state.execution_evaluation.evaluation_status == EvaluationStatus.INCONCLUSIVE
    # Must not leak private objective
    assert "private objective" not in state.execution_trace.model_dump_json()


def test_trace_sanitization_failure_terminal_event(tmp_path, monkeypatch):
    """Corrupted or unsanitizable trace snapshot in terminal event falls back safely."""
    store = SQLiteCheckpointer(tmp_path / "sanitize_failure.db")
    task_id = "task_corrupted_trace"
    state = TaskState(task_id=task_id, objective="priv", status=TaskStatus.FAILED)
    store.save_checkpoint(state)

    corrupted_data = {
        "status": "failed",
        "trace_snapshot": {"invalid": "not a valid trace structure at all"},
    }
    event = TaskEvent(
        event_type="task_failed",
        task_id=task_id,
        message="Failure event",
        data=corrupted_data,
    )
    seq = store.append_event(event)
    events = store.read_events(task_id)
    assert len(events) == 1
    stored_event = events[0][1]
    snapshot = stored_event.data.get("trace_snapshot")
    assert isinstance(snapshot, dict)
    assert snapshot.get("trace_construction_status") == "failed"
    assert snapshot.get("evaluation_status") == "inconclusive"


@pytest.mark.asyncio
async def test_terminal_event_creation_with_absent_or_failed_trace(monkeypatch):
    """_emit_event on terminal status with absent trace produces a typed failure snapshot."""
    runner = AgentOrchestrator()
    task_id = "task_absent_trace"
    state = TaskState(task_id=task_id, objective="priv", status=TaskStatus.RUNNING)
    state.set_execution_trace(None)

    emitted_events = []

    def fake_append(event, *args, **kwargs):
        emitted_events.append(event)
        return 1

    monkeypatch.setattr("app.agent.orchestrator.checkpointer.append_event", fake_append)

    await runner._emit_event(
        TaskEvent(event_type="task_failed", task_id=task_id, message="failed"),
        checkpoint_state=None,
    )

    assert len(emitted_events) == 1
    payload = emitted_events[0]
    assert "trace_snapshot" in payload.data
    snapshot = payload.data["trace_snapshot"]
    assert snapshot["trace_construction_status"] == "failed"
    assert snapshot["evaluation_status"] == "inconclusive"


@pytest.mark.asyncio
async def test_failure_and_cancellation_with_trace_construction_errors(monkeypatch, tmp_path):
    """Cancellation with trace construction error still produces a safe, honest terminal trace."""
    runner = AgentOrchestrator()
    task_id = "task_cancel_trace_err"
    handoff = await _create_test_handoff()
    state = TaskState(
        task_id=task_id,
        objective="private query",
        privacy_mode="local_only",
        status=TaskStatus.CANCEL_REQUESTED,
        plan=handoff.plan.model_copy(deep=True),
    )

    def failing_builder(*args, **kwargs):
        raise ValueError("Simulated trace builder crash")

    monkeypatch.setattr("app.agent.orchestrator.build_handoff_trace", failing_builder)

    runner._finalize_handoff_trace(handoff, state, terminal_status=TaskStatus.CANCELLED)
    assert state.execution_trace is not None
    assert state.execution_trace.trace_construction_status == TraceConstructionStatus.FAILED
    assert state.execution_trace.evaluation_status == EvaluationStatus.INCONCLUSIVE
    assert "private query" not in state.execution_trace.model_dump_json()


def test_recovery_checkpoint_and_event_atomicity(tmp_path):
    """Recovery commits checkpoint and terminal event atomically."""
    store = SQLiteCheckpointer(tmp_path / "recovery_atomic.db")
    task_id = "task_interrupted_work"
    state = TaskState(
        task_id=task_id,
        objective="private",
        status=TaskStatus.RUNNING,
        plan=ExecutionPlan(
            plan_id="plan_1",
            goal="work",
            steps=[ExecutionStep(
                step_id="step_1",
                title="step",
                description="step",
                capability="sys",
                tool_name="system_health",
                status=StepStatus.IN_PROGRESS,
            )],
        ),
    )
    store.save_checkpoint(state)

    recovered_count = store.recover_interrupted_tasks()
    assert recovered_count == 1

    recovered_state = store.load_checkpoint(task_id)
    assert recovered_state is not None
    assert recovered_state.status == TaskStatus.FAILED
    assert recovered_state.recovery_required is True
    assert recovered_state.plan.steps[0].status == StepStatus.INTERRUPTED

    events = store.read_events(task_id)
    assert len(events) == 1
    terminal_event = events[0][1]
    assert terminal_event.event_type == "task_failed"
    assert terminal_event.data.get("recovery_required") is True
    assert terminal_event.data.get("trace_snapshot") is not None


def test_recovery_handles_preexisting_inconsistent_terminal_records(tmp_path):
    """Subsequent recovery detects terminal checkpoints missing terminal events."""
    store = SQLiteCheckpointer(tmp_path / "recovery_inconsistent.db")
    task_id = "task_missing_terminal_event"
    state = TaskState(
        task_id=task_id,
        objective="private",
        status=TaskStatus.COMPLETED,
    )
    store.save_checkpoint(state)

    # Inconsistent state: checkpoint is COMPLETED, but 0 events exist in task_events
    events_before = store.read_events(task_id)
    assert len(events_before) == 0

    recovered_count = store.recover_interrupted_tasks()
    assert recovered_count == 1

    recovered_state = store.load_checkpoint(task_id)
    assert recovered_state.status == TaskStatus.COMPLETED
    assert recovered_state.recovery_required is True

    events_after = store.read_events(task_id)
    assert len(events_after) == 1
    assert events_after[0][1].event_type == "task_completed"


def test_recovery_interrupted_during_persistence(tmp_path, monkeypatch):
    """If persistence fails mid-transaction, nothing is committed and next recovery succeeds."""
    store = SQLiteCheckpointer(tmp_path / "recovery_rollback.db")
    task_id = "task_crash_recovery"
    state = TaskState(
        task_id=task_id,
        objective="private",
        status=TaskStatus.RUNNING,
    )
    store.save_checkpoint(state)

    original_append = store.append_event

    def broken_append(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated disk error during recovery event append")

    monkeypatch.setattr(store, "append_event", broken_append)

    with pytest.raises(sqlite3.OperationalError):
        store.recover_interrupted_tasks()

    # The checkpoint status must NOT have changed because transaction rolled back
    current_state = store.load_checkpoint(task_id)
    assert current_state.status == TaskStatus.RUNNING
    assert len(store.read_events(task_id)) == 0

    # Restore normal behavior and verify recovery recovers it cleanly
    monkeypatch.setattr(store, "append_event", original_append)
    count = store.recover_interrupted_tasks()
    assert count == 1
    assert store.load_checkpoint(task_id).status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_task_detail_versus_sse_terminal_trace_consistency(tmp_path, monkeypatch):
    """Task detail and reconstructed event replay expose identical evaluation information."""
    store = SQLiteCheckpointer(tmp_path / "trace_consistency.db")
    monkeypatch.setattr(task_routes, "checkpointer", store)
    task_id = "task_consistency_check"
    handoff = await _create_test_handoff()
    state = TaskState(
        task_id=task_id,
        objective="priv",
        status=TaskStatus.RUNNING,
        plan=handoff.plan.model_copy(deep=True),
    )
    store.save_checkpoint(state)

    # Simulate recovery
    store.recover_interrupted_tasks()

    # Replay trace from task_routes helper
    loaded_state = store.load_checkpoint(task_id)
    detail_trace = task_routes._execution_trace_for_state(loaded_state)

    # Trace from event payload
    events = store.read_events(task_id)
    event_trace_raw = events[-1][1].data.get("trace_snapshot")
    event_trace = sanitize_execution_trace(event_trace_raw)

    assert detail_trace is not None
    assert event_trace is not None
    assert detail_trace.evaluation_status == event_trace.evaluation_status
    assert detail_trace.trace_construction_status == event_trace.trace_construction_status


# ==============================================================================
# AUD-04: Registry Execution Boundary & Authorization
# ==============================================================================

@pytest.mark.asyncio
async def test_direct_execute_tool_without_authorization(monkeypatch):
    """Direct _execute_tool invocation without scope must raise ExecutionHandoffError."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    with pytest.raises(ExecutionHandoffError):
        await tool_registry._execute_tool("system_health", {})


@pytest.mark.asyncio
async def test_execute_tool_wrong_task_id(monkeypatch):
    """Step scope with incorrect task_id must be rejected."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    handoff = await _create_test_handoff()
    task_A = TaskState(
        task_id="task_A", objective="", privacy_mode="local_only",
        status=TaskStatus.RUNNING, plan=handoff.plan.model_copy(deep=True),
    )
    step = task_A.plan.steps[0]

    with handoff_execution_scope(handoff, task_id="task_A", privacy_mode="local_only"):
        with pytest.raises(ExecutionHandoffError):
            with authorized_step_scope(step, task_A.model_copy(update={"task_id": "task_WRONG"})):
                pass


@pytest.mark.asyncio
async def test_execute_tool_wrong_step_id(monkeypatch):
    """Step scope with unapproved step_id must be rejected."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    handoff = await _create_test_handoff()
    task_A = TaskState(
        task_id="task_A", objective="", privacy_mode="local_only",
        status=TaskStatus.RUNNING, plan=handoff.plan.model_copy(deep=True),
    )
    step_unapproved = task_A.plan.steps[0].model_copy(update={"step_id": "step_UNAPPROVED"})

    with handoff_execution_scope(handoff, task_id="task_A", privacy_mode="local_only"):
        with pytest.raises(ExecutionHandoffError):
            with authorized_step_scope(step_unapproved, task_A):
                pass


@pytest.mark.asyncio
async def test_execute_tool_wrong_tool_name(monkeypatch):
    """Authorized tool invocation with unapproved tool_name must be rejected."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    handoff = await _create_test_handoff()
    task_A = TaskState(
        task_id="task_A", objective="", privacy_mode="local_only",
        status=TaskStatus.RUNNING, plan=handoff.plan.model_copy(deep=True),
    )
    step = task_A.plan.steps[0]

    with handoff_execution_scope(handoff, task_id="task_A", privacy_mode="local_only"):
        with authorized_step_scope(step, task_A):
            with pytest.raises(ExecutionHandoffError):
                with authorized_tool_invocation("inspect_dataset", {}):
                    pass


@pytest.mark.asyncio
async def test_execute_tool_altered_arguments(monkeypatch):
    """Invoking tool with altered arguments must be rejected at execution boundary."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    handoff = await _create_test_handoff()
    task_A = TaskState(
        task_id="task_A", objective="", privacy_mode="local_only",
        status=TaskStatus.RUNNING, plan=handoff.plan.model_copy(deep=True),
    )
    step = task_A.plan.steps[0]

    with handoff_execution_scope(handoff, task_id="task_A", privacy_mode="local_only"):
        with authorized_step_scope(step, task_A):
            with authorized_tool_invocation(step.tool_name, {"allowed": True}):
                with pytest.raises(ExecutionHandoffError):
                    await tool_registry._execute_tool(step.tool_name, {"altered": True})


@pytest.mark.asyncio
async def test_reused_or_expired_authorization(monkeypatch):
    """An authorization token cannot be reused for a second execution."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    handoff = await _create_test_handoff()
    task_A = TaskState(
        task_id="task_A", objective="", privacy_mode="local_only",
        status=TaskStatus.RUNNING, plan=handoff.plan.model_copy(deep=True),
    )
    step = task_A.plan.steps[0]

    with handoff_execution_scope(handoff, task_id="task_A", privacy_mode="local_only"):
        with authorized_step_scope(step, task_A):
            with authorized_tool_invocation(step.tool_name, {}):
                # First execution: consumes authorization
                res1 = await tool_registry.execute_tool(step.tool_name, {}, user_approval_granted=True)
                assert res1.success is True

                # Second execution: authorization is expired/consumed!
                with pytest.raises(ExecutionHandoffError):
                    await tool_registry._execute_tool(step.tool_name, {})


@pytest.mark.asyncio
async def test_concurrent_tasks_cross_use_prevention(monkeypatch):
    """Concurrent tasks cannot cross-use or steal an authorized invocation."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    handoff = await _create_test_handoff()
    task_A = TaskState(
        task_id="task_A", objective="", privacy_mode="local_only",
        status=TaskStatus.RUNNING, plan=handoff.plan.model_copy(deep=True),
    )
    step = task_A.plan.steps[0]

    with handoff_execution_scope(handoff, task_id="task_A", privacy_mode="local_only"):
        with authorized_step_scope(step, task_A):
            with authorized_tool_invocation(step.tool_name, {}):
                async def rogue_task():
                    with pytest.raises(ExecutionHandoffError):
                        await tool_registry._execute_tool(step.tool_name, {})

                await asyncio.create_task(rogue_task())


@pytest.mark.asyncio
async def test_direct_read_only_route_capability_restrictions(monkeypatch):
    """Direct route cannot invoke agent-only or mutating tools."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")

    # Read-only tool succeeds
    result = await tool_registry.execute_direct_read_only_tool("system_health", {})
    assert result.success is True

    # Mutating tool raises PermissionError
    with pytest.raises(PermissionError):
        await tool_registry.execute_direct_read_only_tool(
            "generate_report",
            {"title": "t", "executive_summary": "s", "findings": []},
        )

    # Direct invocation bypassing registry with mutating tool raises ExecutionHandoffError
    with pytest.raises(ExecutionHandoffError):
        with direct_read_only_tool_invocation("generate_report", {}):
            pass


@pytest.mark.asyncio
async def test_exceptions_and_cancellation_during_execution(monkeypatch):
    """Exception or cancellation cleanly releases invocation context without leak."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    handoff = await _create_test_handoff()
    task_A = TaskState(
        task_id="task_A", objective="", privacy_mode="local_only",
        status=TaskStatus.RUNNING, plan=handoff.plan.model_copy(deep=True),
    )
    step = task_A.plan.steps[0]

    for err in [RuntimeError("tool failure"), asyncio.CancelledError()]:
        with handoff_execution_scope(handoff, task_id="task_A", privacy_mode="local_only"):
            with authorized_step_scope(step, task_A):
                with pytest.raises(type(err)):
                    with authorized_tool_invocation(step.tool_name, {}):
                        raise err

        # Scope was cleanly restored: calling _execute_tool raises
        with pytest.raises(ExecutionHandoffError):
            await tool_registry._execute_tool(step.tool_name, {})


# ==============================================================================
# AUD-01: Test Environment Safety
# ==============================================================================

def test_environment_safety_production_development_test(monkeypatch):
    """Production and development reject legacy execution; test only allows it when pytest is latched."""
    monkeypatch.setattr(config_module, "_PYTEST_PROCESS_AT_IMPORT", True)
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    assert config_module.test_harness_enabled() is False

    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    assert config_module.test_harness_enabled() is False

    monkeypatch.setattr(settings, "ENVIRONMENT", "test")
    assert config_module.test_harness_enabled() is True


def test_misconfigured_service_startup_fails_explicitly(monkeypatch):
    """In a real service process (not imported by pytest), test environment raises on validation."""
    monkeypatch.setattr(config_module, "_PYTEST_PROCESS_AT_IMPORT", False)
    monkeypatch.setattr(settings, "ENVIRONMENT", "test")
    assert config_module.test_harness_enabled() is False
    with pytest.raises(RuntimeError, match="not permitted in a service process"):
        validate_service_environment()


def test_attempted_runtime_request_based_environment_override(monkeypatch):
    """HTTP settings update cannot change ENVIRONMENT or enable test harness."""
    from app.api.routes.settings import UpdateSettingsRequest, update_settings

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    req = UpdateSettingsRequest.model_validate({"privacy_mode": "local_only"})
    update_settings(req)
    assert settings.ENVIRONMENT == "production"
    assert config_module.test_harness_enabled() is False


# ==============================================================================
# AUD-03: Symlink & Path Containment Verification
# ==============================================================================

def test_path_containment_platform_independent(tmp_path, monkeypatch):
    """Resolved paths escaping uploads directory are rejected."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.csv"
    outside_file.write_text("secret,data", encoding="utf-8")

    monkeypatch.setattr(settings, "UPLOADS_DIR", uploads_dir)

    # 1. Direct path outside uploads
    with pytest.raises(PermissionError, match="Access denied"):
        validate_safe_path(outside_file)

    # 2. Path traversal attempt (uploads/../outside/secret.csv)
    traversal_path = uploads_dir / ".." / "outside" / "secret.csv"
    with pytest.raises(PermissionError, match="Access denied"):
        validate_safe_path(traversal_path)

    # 3. Path traversal with subdirectories
    sub_traversal = uploads_dir / "sub" / ".." / ".." / "outside" / "secret.csv"
    with pytest.raises(PermissionError, match="Access denied"):
        validate_safe_path(sub_traversal)

    # 4. Mocked path where path.resolve() returns an outside path
    mock_outside = MagicMock()
    mock_outside.resolve.return_value = outside_file.resolve()
    mock_outside.is_file.return_value = True
    with pytest.raises(PermissionError, match="Access denied"):
        validate_safe_path(mock_outside)
