import asyncio
import json
import time
import threading
import uuid
import sqlite3
from contextlib import closing

import pytest
from pydantic import BaseModel

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.orchestrator import AgentOrchestrator
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus, TaskEvent, TaskState, TaskStatus
from app.core.config import settings
from app.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from app.llm.router import llm_router
from app.tools.registry import tool_registry
from app.tools.schemas import ExecutionLocation, PermissionLevel, ToolCategory, ToolResult


class ProbeInput(BaseModel):
    label: str
    delay: float = 0.05


class SyncSleepInput(BaseModel):
    sleep_seconds: float


class RetryProbeInput(BaseModel):
    pass


class Phase2GateOutput(BaseModel):
    completed: bool


class Phase2ProbeOutput(BaseModel):
    label: str


class Phase2SyncSleepOutput(BaseModel):
    slept: float


class Phase2RetryOutput(BaseModel):
    status: str | None = None


_probe_state: dict = {}
_sync_started = threading.Event()
_retry_attempts = {"unsafe": 0, "safe": 0}


async def _dependency_probe(params: ProbeInput):
    _probe_state["active"] += 1
    _probe_state["max_active"] = max(_probe_state["max_active"], _probe_state["active"])
    _probe_state["started"][params.label] = time.perf_counter()
    try:
        if params.label == "fail":
            return {}
        await asyncio.sleep(params.delay)
        _probe_state["finished"][params.label] = time.perf_counter()
        return {"label": params.label}
    finally:
        _probe_state["active"] -= 1


def _sync_sleep(params: SyncSleepInput):
    _sync_started.set()
    time.sleep(params.sleep_seconds)
    return {"slept": params.sleep_seconds}


def _failed_retry_probe(kind: str):
    def invoke(_params: RetryProbeInput):
        _retry_attempts[kind] += 1
        return ToolResult(tool_name=f"phase2_{kind}_retry", success=False, error="temporary failure")
    return invoke


async def _phase2_gate_tool(_params: RetryProbeInput):
    _probe_state["gate_started"].set()
    await _probe_state["gate_release"].wait()
    return {"completed": True}


tool_registry.register(
    name="phase2_gate_tool",
    category=ToolCategory.SYSTEM,
    description="Test-only async tool for cancellation while another step waits for capacity.",
    input_model=RetryProbeInput,
    output_model=Phase2GateOutput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY,
)(_phase2_gate_tool)

tool_registry.register(
    name="phase2_dependency_probe",
    category=ToolCategory.SYSTEM,
    description="Test-only async tool for DAG execution tests.",
    input_model=ProbeInput,
    output_model=Phase2ProbeOutput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY,
)(_dependency_probe)

tool_registry.register(
    name="phase2_sync_sleep",
    category=ToolCategory.SYSTEM,
    description="Test-only synchronous tool for timeout and event-loop tests.",
    input_model=SyncSleepInput,
    output_model=Phase2SyncSleepOutput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY,
    timeout_seconds=1,
)(_sync_sleep)

tool_registry.register(
    name="phase2_unsafe_retry",
    category=ToolCategory.SYSTEM,
    description="Test-only operation with ambiguous side effects.",
    input_model=RetryProbeInput,
    output_model=Phase2RetryOutput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY,
)(_failed_retry_probe("unsafe"))

tool_registry.register(
    name="phase2_safe_retry",
    category=ToolCategory.SYSTEM,
    description="Test-only read operation explicitly safe to retry.",
    input_model=RetryProbeInput,
    output_model=Phase2RetryOutput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY,
    retry_safe=True,
)(_failed_retry_probe("safe"))


class FixedPlanProvider(BaseLLMProvider):
    def __init__(self, steps):
        self.steps = steps

    def is_available(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[LLMMessage],
        tools=None,
        temperature=0.2,
        max_tokens=2048,
        response_format_json=False,
    ) -> LLMResponse:
        plan = {"plan_id": "plan_phase2", "steps": self.steps}
        return LLMResponse(content=json.dumps(plan), model="phase2-test", provider="test")


@pytest.fixture
def execution_data_dir():
    path = settings.DATA_DIR / f"phase2_execution_{uuid.uuid4().hex}"
    path.mkdir()
    yield path
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
    path.rmdir()


def _plan_step(step_id, label, dependencies=()):
    return {
        "step_id": step_id,
        "title": f"Probe {label}",
        "description": "Exercise actual execution ordering.",
        "capability": "system",
        "tool_name": "phase2_dependency_probe",
        "parameters": {"label": label, "delay": 0.05},
        "dependencies": list(dependencies),
    }


@pytest.mark.asyncio
async def test_independent_steps_run_concurrently_and_dependents_wait(monkeypatch, execution_data_dir):
    _probe_state.clear()
    _probe_state.update(active=0, max_active=0, started={}, finished={})
    provider = FixedPlanProvider([
        _plan_step("step_1", "first"),
        _plan_step("step_2", "second"),
        _plan_step("step_3", "third"),
        _plan_step("step_4", "fourth"),
        _plan_step("step_5", "fifth"),
        _plan_step("step_6", "dependent", ["step_1"]),
    ])
    monkeypatch.setattr(llm_router, "get_provider", lambda **kwargs: provider)
    import app.agent.orchestrator as orchestrator_module
    monkeypatch.setattr(orchestrator_module, "checkpointer", SQLiteCheckpointer(execution_data_dir / "tasks.db"))

    state = await AgentOrchestrator().run_task("safe probe", privacy_mode="local_only", task_id="stable-task-id")

    assert state.status == TaskStatus.COMPLETED
    assert state.task_id == "stable-task-id"
    assert 1 < _probe_state["max_active"] <= settings.MAX_CONCURRENT_TOOL_EXECUTIONS
    assert _probe_state["started"]["dependent"] >= _probe_state["finished"]["first"]


@pytest.mark.asyncio
async def test_failed_prerequisite_never_runs_dependent(monkeypatch, execution_data_dir):
    _probe_state.clear()
    _probe_state.update(active=0, max_active=0, started={}, finished={})
    provider = FixedPlanProvider([
        _plan_step("step_1", "fail"),
        _plan_step("step_2", "must-not-run", ["step_1"]),
    ])
    monkeypatch.setattr(llm_router, "get_provider", lambda **kwargs: provider)
    import app.agent.orchestrator as orchestrator_module
    monkeypatch.setattr(orchestrator_module, "checkpointer", SQLiteCheckpointer(execution_data_dir / "tasks.db"))

    state = await AgentOrchestrator().run_task("safe probe", privacy_mode="local_only", task_id="failed-dependency")

    assert state.status == TaskStatus.FAILED
    assert "must-not-run" not in _probe_state["started"]
    assert state.plan.steps[1].status.value == "skipped"


@pytest.mark.asyncio
async def test_synchronous_tool_runs_off_event_loop_and_timeout_waits_for_safe_boundary():
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.05)
            ticks += 1

    tick_task = asyncio.create_task(ticker())
    started = time.perf_counter()
    result = await tool_registry.execute_tool("phase2_sync_sleep", {"sleep_seconds": 1.1})
    elapsed = time.perf_counter() - started
    await tick_task

    assert ticks == 5
    assert elapsed >= 1.0
    assert not result.success
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_cancellation_waits_for_running_sync_tool_boundary(monkeypatch, execution_data_dir):
    _sync_started.clear()
    provider = FixedPlanProvider([{
        "step_id": "step_1",
        "title": "Slow synchronous operation",
        "description": "Exercise cancellation boundary.",
        "capability": "system",
        "tool_name": "phase2_sync_sleep",
        "parameters": {"sleep_seconds": 1.2},
        "dependencies": [],
    }])
    monkeypatch.setattr(llm_router, "get_provider", lambda **kwargs: provider)
    import app.agent.orchestrator as orchestrator_module
    monkeypatch.setattr(orchestrator_module, "checkpointer", SQLiteCheckpointer(execution_data_dir / "cancel.db"))
    runner = AgentOrchestrator()
    task = asyncio.create_task(runner.run_task("slow operation", privacy_mode="local_only", task_id="cancel-boundary"))

    for _ in range(100):
        if _sync_started.is_set():
            break
        await asyncio.sleep(0.02)
    assert _sync_started.is_set()
    assert runner.request_cancel("cancel-boundary") == "cancel_requested"
    assert not task.done()

    finished = await task
    assert finished.status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_prevents_a_step_waiting_for_tool_capacity_from_starting(monkeypatch):
    from app.agent.executor import PlanExecutor

    monkeypatch.setattr(settings, "MAX_CONCURRENT_TOOL_EXECUTIONS", 1)
    _probe_state.clear()
    _probe_state.update(gate_started=asyncio.Event(), gate_release=asyncio.Event(), active=0, max_active=0, started={}, finished={})
    state = TaskState(task_id="cancel-capacity-waiter", objective="safe", status=TaskStatus.RUNNING)
    active_step = ExecutionStep(
        step_id="step_active", title="Active", description="test", capability="system",
        tool_name="phase2_gate_tool",
    )
    waiting_step = ExecutionStep(
        step_id="step_waiting", title="Waiting", description="test", capability="system",
        tool_name="phase2_dependency_probe", parameters={"label": "must-not-start"},
    )
    executor = PlanExecutor()
    active = asyncio.create_task(executor.execute_step(active_step, state))
    await asyncio.wait_for(_probe_state["gate_started"].wait(), timeout=1)
    waiting = asyncio.create_task(executor.execute_step(waiting_step, state))
    await asyncio.sleep(0.05)

    state.transition_to(TaskStatus.CANCEL_REQUESTED)
    _probe_state["gate_release"].set()
    await asyncio.gather(active, waiting)

    assert waiting_step.status == StepStatus.SKIPPED
    assert "must-not-start" not in _probe_state["started"]


@pytest.mark.asyncio
async def test_worker_shutdown_waits_for_active_synchronous_tool(monkeypatch, execution_data_dir):
    from app.agent.service import TaskExecutionService
    import app.agent.service as service_module

    _sync_started.clear()
    provider = FixedPlanProvider([{
        "step_id": "step_1", "title": "Shutdown boundary", "description": "test",
        "capability": "system", "tool_name": "phase2_sync_sleep",
        "parameters": {"sleep_seconds": 1.1}, "dependencies": [],
    }])
    monkeypatch.setattr(llm_router, "get_provider", lambda **kwargs: provider)
    store = SQLiteCheckpointer(execution_data_dir / "shutdown.db")
    monkeypatch.setattr(service_module, "checkpointer", store)
    import app.agent.orchestrator as orchestrator_module
    monkeypatch.setattr(orchestrator_module, "checkpointer", store)
    runner = AgentOrchestrator()
    service = TaskExecutionService(runner)
    await service.enqueue(TaskState(task_id="shutdown-boundary", objective="safe", privacy_mode="local_only"))
    await service.start()

    for _ in range(100):
        if _sync_started.is_set():
            break
        await asyncio.sleep(0.02)
    assert _sync_started.is_set()
    started = time.perf_counter()
    await service.stop()
    assert time.perf_counter() - started >= 1.0
    assert store.load_checkpoint("shutdown-boundary").status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_only_explicitly_retry_safe_tools_are_retried():
    from app.agent.executor import PlanExecutor

    _retry_attempts.update(unsafe=0, safe=0)
    for name, kind in (("phase2_unsafe_retry", "unsafe"), ("phase2_safe_retry", "safe")):
        step = ExecutionStep(
            step_id=f"step_{kind}", title="Retry safety", description="test", capability="system",
            tool_name=name, max_retries=2,
        )
        state = TaskState(task_id=f"retry-{kind}", objective="safe")
        result = await PlanExecutor().execute_step(step, state)
        assert result is not None and not result.success
    assert _retry_attempts["unsafe"] == 1
    assert _retry_attempts["safe"] == 3

    _retry_attempts["safe"] = 0
    cancelled_state = TaskState(task_id="retry-cancelled", objective="safe")
    cancelled_state.transition_to(TaskStatus.PLANNING)
    cancelled_state.transition_to(TaskStatus.RUNNING)
    cancelled_state.transition_to(TaskStatus.CANCEL_REQUESTED)
    cancelled_step = ExecutionStep(
        step_id="step_safe_cancelled", title="Cancelled retry", description="test", capability="system",
        tool_name="phase2_safe_retry", max_retries=2,
    )
    await PlanExecutor().execute_step(cancelled_step, cancelled_state)
    assert _retry_attempts["safe"] == 0


@pytest.mark.asyncio
async def test_persisted_sse_projection_drops_sensitive_event_content(execution_data_dir, monkeypatch):
    import app.agent.orchestrator as orchestrator_module

    marker = "private-person@example.com raw-dataset-row"
    store = SQLiteCheckpointer(execution_data_dir / "events.db")
    state = TaskState(
        task_id="safe-event-projection",
        objective=marker,
        status=TaskStatus.RUNNING,
        plan=ExecutionPlan(plan_id="safe-plan", goal=marker, steps=[ExecutionStep(
            step_id="step_1", title=marker, description=marker, capability="dataset",
            tool_name="inspect_dataset", parameters={"file_path": marker}, status=StepStatus.IN_PROGRESS,
        )]),
    )
    store.save_checkpoint(state)
    monkeypatch.setattr(orchestrator_module, "checkpointer", store)
    runner = AgentOrchestrator()
    runner._active_tasks[state.task_id] = state
    await runner._emit_event(TaskEvent(
        event_type="step_started", task_id=state.task_id, step_id="step_1",
        message=marker, data={"arguments": marker, "rows": [marker]},
    ))

    records = store.read_events(state.task_id)
    assert len(records) == 1
    sequence, projected = records[0]
    assert sequence == 1
    assert projected.message == "A tool step started."
    assert marker not in str(projected.model_dump())
    assert projected.data["tool_name"] == "inspect_dataset"

    store.append_event(TaskEvent(
        event_type="step_completed", task_id=state.task_id, step_id="step_1",
        message=marker, data={"output": marker, "objective": marker, "duration_ms": 42},
    ))
    second = store.read_events(state.task_id, after_sequence=1)[0][1]
    assert second.message == "A tool step completed."
    assert second.data == {"duration_ms": 42}
    assert marker not in str(second.model_dump())

    store.append_event(TaskEvent(
        event_type=f"provider-error-{marker}", task_id=state.task_id,
        message=marker, data={"secret": marker},
    ))
    with closing(sqlite3.connect(store.db_path)) as conn:
        rows = conn.execute(
            "SELECT sequence, event_type, created_at, payload_json FROM task_events WHERE task_id = ? ORDER BY sequence",
            (state.task_id,),
        ).fetchall()
    assert [row[1] for row in rows] == ["step_started", "step_completed", "task_update"]
    assert all(marker not in row[1] and marker not in row[3] for row in rows)


@pytest.mark.asyncio
async def test_sse_replay_catches_events_produced_concurrently(execution_data_dir, monkeypatch):
    import app.agent.orchestrator as orchestrator_module
    import app.api.routes.tasks as task_routes

    store = SQLiteCheckpointer(execution_data_dir / "concurrent-sse.db")
    state = TaskState(task_id="concurrent-sse", objective="private", status=TaskStatus.RUNNING)
    store.save_checkpoint(state)
    monkeypatch.setattr(orchestrator_module, "checkpointer", store)
    monkeypatch.setattr(task_routes, "checkpointer", store)
    runner = AgentOrchestrator()
    runner._active_tasks[state.task_id] = state
    monkeypatch.setattr(task_routes, "orchestrator", runner)

    response = await task_routes.stream_task_events(state.task_id, last_event_id=None)
    iterator = response.body_iterator
    initial = await asyncio.wait_for(iterator.__anext__(), timeout=1)
    assert "task.snapshot" in initial

    pending_read = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0)
    await runner._emit_event(TaskEvent(
        event_type="step_started", task_id=state.task_id,
        message="sensitive raw content", data={"rows": ["private"]},
    ))
    produced = await asyncio.wait_for(pending_read, timeout=1)
    assert "id: 1" in produced
    assert "event: step_started" in produced
    assert "sensitive raw content" not in produced
    assert "private" not in produced

    state.transition_to(TaskStatus.COMPLETED)
    await runner._emit_event(TaskEvent(event_type="task_completed", task_id=state.task_id, message="done"))
    terminal = await asyncio.wait_for(iterator.__anext__(), timeout=1)
    assert "event: task_completed" in terminal
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(iterator.__anext__(), timeout=1)


@pytest.mark.asyncio
async def test_sse_replay_reports_expired_history_and_keeps_safe_cursor(execution_data_dir, monkeypatch):
    import app.api.routes.tasks as task_routes

    store = SQLiteCheckpointer(execution_data_dir / "history-gap.db")
    state = TaskState(
        task_id="history-gap-task", objective="PRIVATE_OBJECTIVE_390182",
        status=TaskStatus.COMPLETED,
    )
    store.save_checkpoint(state)
    for index in range(1, 6):
        store.append_event(TaskEvent(
            event_type="task_completed" if index == 5 else "task_started",
            task_id=state.task_id,
            message=f"PRIVATE_EVENT_{index}",
            data={"objective": "PRIVATE_OBJECTIVE_390182", "status": "completed"},
        ), retention=3)

    runner = AgentOrchestrator()
    runner._active_tasks[state.task_id] = state
    monkeypatch.setattr(task_routes, "checkpointer", store)
    monkeypatch.setattr(task_routes, "orchestrator", runner)

    latest, earliest, events = store.read_event_window(state.task_id, 0)
    assert (latest, earliest) == (5, 3)
    assert [sequence for sequence, _ in events] == [3, 4, 5]

    response = await task_routes.stream_task_events(state.task_id, last_event_id="0")
    iterator = response.body_iterator
    gap = await iterator.__anext__()
    assert "id: 5" in gap
    assert "event: task.history_gap" in gap
    assert '"status":"completed"' in gap
    assert '"history_gap":true' in gap
    assert "PRIVATE_OBJECTIVE_390182" not in gap
    assert "PRIVATE_EVENT" not in gap
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()


@pytest.mark.asyncio
async def test_sse_replay_handles_valid_empty_and_future_cursors(execution_data_dir, monkeypatch):
    import app.api.routes.tasks as task_routes
    from fastapi import HTTPException

    store = SQLiteCheckpointer(execution_data_dir / "event-cursors.db")
    state = TaskState(task_id="event-cursors", objective="private", status=TaskStatus.COMPLETED)
    store.save_checkpoint(state)
    for index in range(1, 6):
        store.append_event(TaskEvent(
            event_type="task_completed" if index == 5 else "task_started",
            task_id=state.task_id,
            message="safe event",
        ), retention=3)

    runner = AgentOrchestrator()
    runner._active_tasks[state.task_id] = state
    monkeypatch.setattr(task_routes, "checkpointer", store)
    monkeypatch.setattr(task_routes, "orchestrator", runner)

    valid = await task_routes.stream_task_events(state.task_id, last_event_id="3")
    valid_text = "".join([frame async for frame in valid.body_iterator])
    assert "event: task.history_gap" not in valid_text
    assert "id: 4" in valid_text and "id: 5" in valid_text

    # Simulate all retained rows expiring while the task sequence remains durable.
    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute("DELETE FROM task_events WHERE task_id = ?", (state.task_id,))
        conn.commit()
    latest, earliest, events = store.read_event_window(state.task_id, 0)
    assert latest == 5 and earliest is None and events == []
    empty = await task_routes.stream_task_events(state.task_id, last_event_id="0")
    empty_text = "".join([frame async for frame in empty.body_iterator])
    assert "event: task.history_gap" in empty_text
    assert "id: 5" in empty_text

    with pytest.raises(HTTPException) as future:
        await task_routes.stream_task_events(state.task_id, last_event_id="6")
    assert future.value.status_code == 400
