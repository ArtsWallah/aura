import pytest
from fastapi import HTTPException

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.execution_handoff import ExecutionHandoff
from app.agent.execution_trace import (
    MAX_TRACE_BYTES,
    build_handoff_trace,
    sanitize_execution_trace,
)
from app.agent.intelligence_contracts import (
    EvaluationResult,
    EvaluationStatus,
    ExecutionTrace,
    IssueCode,
    ProviderIdentity,
    StepFailureCategory,
)
from app.agent.intelligence_pipeline import IntelligencePipeline
from app.agent.orchestrator import AgentOrchestrator
from app.agent.state import StepStatus, TaskEvent, TaskState, TaskStatus
from app.tools.registry import tool_registry
import app.api.routes.tasks as task_routes


async def _handoff(objective="Check system health"):
    result = await IntelligencePipeline().run(objective, privacy_mode="local_only")
    return ExecutionHandoff.from_pipeline_result(result, privacy_mode="local_only")


def _trace_state(handoff, *, status=TaskStatus.RUNNING):
    marker = "RAW_TRACE_PRIVATE_MARKER_774"
    state = TaskState(
        task_id="trace-task",
        objective=marker,
        file_path=f"C:/private/{marker}.csv",
        privacy_mode="local_only",
        status=status,
        plan=handoff.plan.model_copy(deep=True),
    )
    step = state.plan.steps[0]
    step.status = StepStatus.COMPLETED
    step.retry_count = 1
    step.output = {"raw_output": marker, "arguments": marker}
    state.step_results[step.step_id] = {
        "data": step.output,
        "artifacts": [f"C:/private/{marker}.csv"],
    }
    state.set_execution_evaluation(EvaluationResult(
        execution_valid=True,
        objective_coverage=1.0,
        evaluation_status=EvaluationStatus.SUCCEEDED,
    ))
    state.set_execution_trace(build_handoff_trace(handoff, state))
    return state, marker


@pytest.mark.asyncio
async def test_trace_uses_registry_labels_and_excludes_raw_execution_data():
    handoff = await _handoff("Check system health and include marker alpha-private-88")
    state, marker = _trace_state(handoff)
    trace = state.execution_trace
    definition = tool_registry.get_tool_definition(trace.step_statuses[0].tool_name)

    assert trace.intent_category.value == "system_health"
    assert trace.provider_identity is ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
    assert trace.step_statuses[0].tool_label == definition.name.replace("_", " ").title()
    assert trace.step_statuses[0].tool_category == definition.category
    assert trace.step_statuses[0].status is StepStatus.COMPLETED
    assert trace.step_statuses[0].retry_count == 1
    assert trace.evaluation_status is EvaluationStatus.SUCCEEDED
    assert marker not in trace.model_dump_json()
    assert "C:/private" not in trace.model_dump_json()
    assert "raw_output" not in trace.model_dump_json()
    assert "arguments" not in trace.model_dump_json()


def test_trace_sanitizer_replaces_labels_and_omits_unknown_tools():
    payload = {
        "intent_category": "system_health",
        "intent_confidence": 0.91,
        "provider_identity": "gemini",
        "selected_tools": ["system_health", "missing_tool"],
        "step_statuses": [
            {
                "step_id": "step_known",
                "tool_name": "system_health",
                "tool_label": "MODEL-GENERATED PRIVATE LABEL",
                "tool_category": "report",
                "status": "completed",
                "retry_count": 0,
                "retry_limit": 2,
                "duration_ms": 2,
            },
            {
                "step_id": "step_unknown",
                "tool_name": "missing_tool",
                "status": "completed",
            },
        ],
        "validation_outcome": "accepted",
        "evaluation_status": "succeeded",
    }
    trace = sanitize_execution_trace(payload)
    assert trace is not None
    assert len(trace.step_statuses) == 1
    assert trace.selected_tools == ["system_health"]
    assert trace.step_statuses[0].tool_label == "System Health"
    assert trace.step_statuses[0].tool_category.value == "system"
    assert "MODEL-GENERATED" not in trace.model_dump_json()

    artifacts = sanitize_execution_trace({
        "intent_category": "visualization",
        "provider_identity": "local_deterministic_planner",
        "selected_tools": ["generate_chart"],
        "step_statuses": [{
            "step_id": "step_chart",
            "tool_name": "generate_chart",
            "status": "completed",
            "artifacts": [
                {"artifact_type": "chart_png", "available": True, "count": 1},
                {"artifact_type": "report_html", "available": True, "count": 1},
            ],
        }],
        "validation_outcome": "accepted",
    })
    assert artifacts is not None
    assert [item.artifact_type.value for item in artifacts.step_statuses[0].artifacts] == ["chart_png"]


@pytest.mark.asyncio
async def test_artifact_and_failure_metadata_are_typed_and_task_status_dto_is_allowlisted(monkeypatch, tmp_path):
    handoff = await _handoff()
    state, _ = _trace_state(handoff, status=TaskStatus.FAILED)
    step = state.plan.steps[0]
    step.status = StepStatus.FAILED
    step.error = "secret exception text"
    state.step_results.clear()
    state.set_execution_evaluation(EvaluationResult(
        execution_valid=False,
        objective_coverage=0.0,
        evaluation_status=EvaluationStatus.EXECUTION_FAILED,
        issues=[IssueCode.TOOL_FAILED],
    ))
    state.set_execution_trace(build_handoff_trace(handoff, state))
    assert state.execution_trace.step_statuses[0].failure_category is StepFailureCategory.TOOL_FAILURE
    assert state.execution_trace.evaluation_status is EvaluationStatus.EXECUTION_FAILED

    monkeypatch.setattr(task_routes, "checkpointer", SQLiteCheckpointer(tmp_path / "status-trace.db"))
    response = task_routes._status_response(state)
    payload = response.model_dump(mode="json")
    assert set(payload) == {
        "task_id", "status", "privacy_mode", "created_at", "updated_at", "execution_trace"
    }
    serialized = str(payload)
    assert "objective" not in serialized
    assert "plan_steps" not in serialized and "arguments" not in serialized
    assert "secret exception text" not in str(payload)


@pytest.mark.asyncio
async def test_trace_size_is_bounded_and_legacy_task_has_no_trace():
    handoff = await _handoff()
    trace = build_handoff_trace(handoff, _trace_state(handoff)[0])
    assert trace is not None
    assert len(trace.model_dump_json().encode("utf-8")) <= MAX_TRACE_BYTES

    legacy = TaskState(task_id="legacy-no-trace", objective="", status=TaskStatus.COMPLETED)
    response = task_routes._status_response(legacy)
    assert response.execution_trace is None


@pytest.mark.asyncio
async def test_sse_live_replay_and_status_use_the_same_safe_trace_schema(monkeypatch, tmp_path):
    handoff = await _handoff()
    state, marker = _trace_state(handoff, status=TaskStatus.COMPLETED)
    store = SQLiteCheckpointer(tmp_path / "trace-events.db")
    store.save_checkpoint(state)
    snapshot = state.execution_trace.model_dump(mode="json")
    trace_step = snapshot["step_statuses"][0]
    store.append_event(TaskEvent(
        event_type="plan_created", task_id=state.task_id, message="untrusted",
        data={"trace_snapshot": snapshot, "objective": marker, "tool_arguments": {"x": marker}},
    ))
    store.append_event(TaskEvent(
        event_type="step_completed", task_id=state.task_id, step_id=trace_step["step_id"],
        message="untrusted", data={"trace_step": trace_step, "raw_output": marker},
    ))
    store.append_event(TaskEvent(
        event_type="task_completed", task_id=state.task_id, message="untrusted",
        data={"trace_snapshot": snapshot, "provider_prompt": marker},
    ))
    monkeypatch.setattr(task_routes, "checkpointer", store)
    monkeypatch.setattr(task_routes, "orchestrator", AgentOrchestrator())
    task_routes.orchestrator._active_tasks[state.task_id] = state

    status_response = task_routes.get_task(state.task_id)
    assert status_response.execution_trace == state.execution_trace
    response = await task_routes.stream_task_events(state.task_id, last_event_id="0")
    frames = [frame async for frame in response.body_iterator]
    text = "".join(frames)
    assert "event: plan_created" in text
    assert "event: step_completed" in text
    assert "event: task_completed" in text
    assert marker not in text and "provider_prompt" not in text
    stored = store.read_events(state.task_id)
    for _, event in stored:
        if "trace_snapshot" in event.data:
            ExecutionTrace.model_validate(event.data["trace_snapshot"])
        if "trace_step" in event.data:
            assert event.data["trace_step"]["tool_label"] == "System Health"


@pytest.mark.asyncio
async def test_history_gap_includes_trace_snapshot_and_task_ids_remain_scoped(monkeypatch, tmp_path):
    handoff = await _handoff()
    state, _ = _trace_state(handoff, status=TaskStatus.COMPLETED)
    store = SQLiteCheckpointer(tmp_path / "trace-gap.db")
    store.save_checkpoint(state)
    snapshot = state.execution_trace.model_dump(mode="json")
    for index in range(4):
        store.append_event(TaskEvent(
            event_type="task_completed" if index == 3 else "task_started",
            task_id=state.task_id,
            message="safe",
            data={"trace_snapshot": snapshot},
        ), retention=2)
    runner = AgentOrchestrator()
    runner._active_tasks[state.task_id] = state
    monkeypatch.setattr(task_routes, "checkpointer", store)
    monkeypatch.setattr(task_routes, "orchestrator", runner)

    response = await task_routes.stream_task_events(state.task_id, last_event_id="0")
    gap = "".join([frame async for frame in response.body_iterator])
    assert "event: task.history_gap" in gap
    assert '"trace_snapshot"' in gap
    assert "event: task_completed" not in gap

    with pytest.raises(HTTPException) as missing_status:
        task_routes.get_task("arbitrary-other-task")
    assert missing_status.value.status_code == 404
    with pytest.raises(HTTPException) as missing_events:
        await task_routes.stream_task_events("arbitrary-other-task", last_event_id="0")
    assert missing_events.value.status_code == 404
