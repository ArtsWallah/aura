import pytest
import threading

from app.agent import orchestrator as orchestrator_module
from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.execution_handoff import (
    ExecutionHandoff,
    ExecutionHandoffError,
    HandoffFailureCode,
)
from app.agent.intelligence_contracts import (
    EvaluationStatus,
    IntentCategory,
    ProviderIdentity,
)
from app.agent.intelligence_pipeline import IntelligencePipeline, PipelineDisposition
from app.agent.orchestrator import AgentOrchestrator
from app.agent.planner import TaskPlanner
from app.agent.state import StepStatus, TaskStatus
from app.agent.state import ExecutionPlan, ExecutionStep, TaskEvent, TaskState
from app.agent.executor import PlanExecutor
from app.api.routes.tasks import _execution_trace_for_state, _status_response
from app.core.config import settings
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.schemas import ToolResult


async def _ready_result():
    return await IntelligencePipeline().run("Check system health", privacy_mode="local_only")


def _handoff(result):
    return ExecutionHandoff.from_pipeline_result(
        result,
        privacy_mode="local_only",
        registry=tool_registry,
    )


@pytest.mark.asyncio
async def test_ready_result_creates_allowlisted_handoff_and_owns_plan_copy():
    result = await IntelligencePipeline().run(
        "Check system health and return my private marker amber-917",
        privacy_mode="local_only",
    )
    source_plan = result.planning_response.plan
    handoff = _handoff(result)
    original_goal = handoff.plan.goal

    source_plan.goal = "mutated source objective"
    assert handoff.plan.goal == original_goal
    assert handoff.provider_identity is ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
    assert handoff.intent_category is IntentCategory.SYSTEM_HEALTH
    assert handoff.evaluation_status is EvaluationStatus.PLAN_VALID
    assert handoff.plan.goal == IntentCategory.SYSTEM_HEALTH.value
    assert set(handoff.model_dump()) == {
        "handoff_id", "plan", "privacy_mode", "provider_identity", "intent_category",
        "intent_confidence", "intent_is_multi_step", "required_tool_groups",
        "evaluation_status", "objective_coverage",
        "validation_registry_id", "plan_digest", "intent_digest",
    }
    assert "mutated source objective" not in str(handoff.model_dump())
    assert "amber-917" not in str(handoff.model_dump())


@pytest.mark.asyncio
@pytest.mark.parametrize("objective, expected", [
    ("Analyze this", PipelineDisposition.CLARIFICATION_REQUIRED),
    ("Teleport a database to Mars", PipelineDisposition.REJECTED),
])
async def test_non_ready_pipeline_results_cannot_create_handoff(objective, expected):
    result = await IntelligencePipeline().run(objective, privacy_mode="local_only")
    assert result.disposition is expected
    with pytest.raises(ExecutionHandoffError) as failure:
        _handoff(result)
    assert failure.value.code is HandoffFailureCode.INVALID_SOURCE


@pytest.mark.asyncio
async def test_missing_plan_invalid_evaluation_and_provider_mismatch_are_rejected():
    result = await _ready_result()
    missing_plan = result.model_copy(deep=True)
    missing_plan.planning_response.plan = None
    with pytest.raises(ExecutionHandoffError):
        _handoff(missing_plan)

    bad_evaluation = result.model_copy(deep=True)
    bad_evaluation.evaluation.evaluation_status = EvaluationStatus.INCONCLUSIVE
    with pytest.raises(ExecutionHandoffError):
        _handoff(bad_evaluation)

    bad_identity = result.model_copy(deep=True)
    bad_identity.provider_identity = ProviderIdentity.GEMINI
    with pytest.raises(ExecutionHandoffError):
        _handoff(bad_identity)

    missing_identity = result.model_copy(deep=True)
    missing_identity.provider_identity = None
    with pytest.raises(ExecutionHandoffError):
        _handoff(missing_identity)


@pytest.mark.asyncio
async def test_handoff_rejects_registry_and_privacy_mismatch():
    result = await _ready_result()
    mismatched = result.model_copy(update={"validation_registry_id": "other-registry"})
    with pytest.raises(ExecutionHandoffError) as failure:
        _handoff(mismatched)
    assert failure.value.code is HandoffFailureCode.INVALID_SOURCE

    with pytest.raises(ExecutionHandoffError) as failure:
        ExecutionHandoff.from_pipeline_result(
            result, privacy_mode="hybrid", registry=tool_registry
        )
    assert failure.value.code is HandoffFailureCode.INVALID_SOURCE

    with pytest.raises(ExecutionHandoffError) as failure:
        ExecutionHandoff.from_pipeline_result(
            result, privacy_mode="local_only", registry=ToolRegistry()
        )
    assert failure.value.code is HandoffFailureCode.REGISTRY_MISMATCH

    with pytest.raises(ExecutionHandoffError) as failure:
        ExecutionHandoff.model_validate({"plan": result.planning_response.plan.model_dump()})
    assert failure.value.code is HandoffFailureCode.INVALID_HANDOFF


@pytest.mark.asyncio
async def test_handoff_integrity_covers_safe_intent_coverage_metadata():
    handoff = _handoff(await _ready_result())
    handoff.required_tool_groups.append(["generate_report"])
    with pytest.raises(ExecutionHandoffError) as failure:
        handoff.validate_integrity(privacy_mode="local_only", registry=tool_registry)
    assert failure.value.code is HandoffFailureCode.INVALID_HANDOFF


@pytest.mark.asyncio
async def test_registry_unavailable_fails_with_safe_code(monkeypatch):
    result = await _ready_result()

    def unavailable(_name):
        raise RuntimeError("sensitive registry internals")

    monkeypatch.setattr(tool_registry, "get_tool_definition", unavailable)
    with pytest.raises(ExecutionHandoffError) as failure:
        _handoff(result)
    assert failure.value.code is HandoffFailureCode.REGISTRY_UNAVAILABLE
    assert "sensitive" not in str(failure.value)


@pytest.mark.asyncio
async def test_tampered_handoff_never_executes_tools(monkeypatch, tmp_path):
    result = await _ready_result()
    handoff = _handoff(result)
    handoff.plan.goal = "changed after handoff"
    executed = []

    async def record_execution(*args, **kwargs):
        executed.append(True)

    monkeypatch.setattr(orchestrator_module.executor, "execute_step", record_execution)
    monkeypatch.setattr(
        orchestrator_module,
        "checkpointer",
        SQLiteCheckpointer(tmp_path / "handoff.db"),
    )
    with pytest.raises(ExecutionHandoffError) as failure:
        await AgentOrchestrator().run_handoff(handoff)
    assert failure.value.code is HandoffFailureCode.PLAN_TAMPERED
    assert executed == []


@pytest.mark.asyncio
async def test_execution_revalidates_current_registry_and_privacy_before_tools(monkeypatch):
    result = await _ready_result()
    handoff = _handoff(result)
    tool_name = handoff.plan.steps[0].tool_name
    original_definition = tool_registry.get_tool_definition(tool_name)
    tool_registry._tools[tool_name] = original_definition.model_copy(
        update={"requires_approval": True}
    )
    executed = []

    async def record_execution(*args, **kwargs):
        executed.append(True)

    monkeypatch.setattr(orchestrator_module.executor, "execute_step", record_execution)
    try:
        with pytest.raises(ExecutionHandoffError) as failure:
            await AgentOrchestrator().run_handoff(handoff)
        assert failure.value.code is HandoffFailureCode.PLAN_REVALIDATION_FAILED
    finally:
        tool_registry._tools[tool_name] = original_definition
    assert executed == []

    with pytest.raises(ExecutionHandoffError) as failure:
        await AgentOrchestrator().run_handoff(handoff, privacy_mode="hybrid")
    assert failure.value.code is HandoffFailureCode.PRIVACY_MISMATCH
    assert executed == []


@pytest.mark.asyncio
async def test_handoff_path_executes_supplied_plan_without_legacy_planner(monkeypatch, tmp_path):
    result = await _ready_result()
    handoff = _handoff(result)
    expected_plan = handoff.plan.model_copy(deep=True)
    observed = []

    async def record_execution(step, state, event_queue=None):
        observed.append(step.model_copy(deep=True))
        step.status = StepStatus.COMPLETED
        step.output = {"synthetic": True}
        return ToolResult(tool_name=step.tool_name, success=True, data=step.output)

    async def legacy_planner_forbidden(*args, **kwargs):
        raise AssertionError("handoff path must not invoke the legacy planner")

    monkeypatch.setattr(orchestrator_module.executor, "execute_step", record_execution)
    monkeypatch.setattr(TaskPlanner, "create_plan", legacy_planner_forbidden)
    monkeypatch.setattr(
        orchestrator_module,
        "checkpointer",
        SQLiteCheckpointer(tmp_path / "handoff-execution.db"),
    )
    state = await AgentOrchestrator().run_handoff(handoff, task_id="handoff-test-task")

    assert state.status is TaskStatus.COMPLETED
    assert state.execution_evaluation.evaluation_status is EvaluationStatus.SUCCEEDED
    assert expected_plan.steps == observed
    assert state.plan.plan_id == expected_plan.plan_id
    assert state.objective == ""
    assert len(observed) == len(expected_plan.steps)


@pytest.mark.asyncio
async def test_production_executor_and_registry_reject_calls_without_handoff(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    step = ExecutionStep(
        step_id="step_health",
        title="System health",
        description="Check system health",
        capability="system",
        tool_name="system_health",
    )
    state = TaskState(
        task_id="unauthorized-executor-call",
        objective="",
        privacy_mode="local_only",
        status=TaskStatus.RUNNING,
        plan=ExecutionPlan(plan_id="plan_health", goal="system_health", steps=[step]),
    )
    invoked = []

    def record_tool(_args):
        invoked.append(True)
        return {"status": "unexpected"}

    monkeypatch.setitem(tool_registry._executors, "system_health", record_tool)
    with pytest.raises(ExecutionHandoffError):
        await PlanExecutor().execute_step(step, state)
    with pytest.raises(ExecutionHandoffError):
        await tool_registry.execute_tool(
            "system_health", {}, user_approval_granted=True
        )
    assert invoked == []


@pytest.mark.asyncio
async def test_direct_tool_route_uses_only_approved_read_only_tools_in_production(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    result = await tool_registry.execute_direct_read_only_tool("system_health", {})
    assert result.success is True
    with pytest.raises(PermissionError):
        await tool_registry.execute_direct_read_only_tool(
            "generate_report",
            {"title": "x", "executive_summary": "x", "findings": []},
        )


@pytest.mark.asyncio
async def test_valid_handoff_executes_through_production_registry_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(
        orchestrator_module,
        "checkpointer",
        SQLiteCheckpointer(tmp_path / "production-handoff.db"),
    )
    handoff = _handoff(await _ready_result())
    state = await AgentOrchestrator().run_handoff(handoff)
    assert state.status is TaskStatus.COMPLETED
    assert state.execution_evaluation is not None
    assert state.execution_trace is not None
    assert state.execution_trace.evaluation_status is state.execution_evaluation.evaluation_status


@pytest.mark.asyncio
async def test_terminal_status_trace_checkpoint_and_sse_are_consistent(monkeypatch, tmp_path):
    import app.api.routes.tasks as task_routes

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    store = SQLiteCheckpointer(tmp_path / "terminal-trace.db")
    monkeypatch.setattr(orchestrator_module, "checkpointer", store)
    monkeypatch.setattr(task_routes, "checkpointer", store)
    runner = AgentOrchestrator()
    handoff = _handoff(await _ready_result())
    task_id = "controlled-terminal-race"
    finalizing = threading.Event()
    release_finalizer = threading.Event()
    observed: dict[str, object] = {}
    original_finalize = runner._finalize_handoff_trace

    def pause_finalizer(handoff_arg, state_arg, *, terminal_status=None):
        finalizing.set()
        if not release_finalizer.wait(timeout=5):
            raise TimeoutError("Test finalizer synchronization timed out.")
        return original_finalize(
            handoff_arg, state_arg, terminal_status=terminal_status
        )

    monkeypatch.setattr(runner, "_finalize_handoff_trace", pause_finalizer)

    def observe_during_finalization():
        if not finalizing.wait(timeout=5):
            observed["error"] = "finalizer was not reached"
            release_finalizer.set()
            return
        current = runner.get_task_state(task_id)
        observed["status"] = current.status if current else None
        observed["trace_status"] = (
            current.execution_trace.evaluation_status
            if current and current.execution_trace
            else None
        )
        observed["terminal_events"] = [
            event.event_type for _, event in store.read_events(task_id)
            if event.event_type in {"task_completed", "task_failed", "task_cancelled"}
        ]
        release_finalizer.set()

    observer = threading.Thread(target=observe_during_finalization, daemon=True)
    observer.start()
    state = await runner.run_handoff(handoff, task_id=task_id)
    observer.join(timeout=5)

    assert "error" not in observed
    assert observed["status"] not in {
        TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED
    }
    assert observed["terminal_events"] == []
    assert state.status is TaskStatus.COMPLETED
    assert state.execution_trace is not None
    assert state.execution_trace.evaluation_status is EvaluationStatus.SUCCEEDED

    terminal_event = next(
        event for _, event in store.read_events(task_id)
        if event.event_type == "task_completed"
    )
    assert terminal_event.data["trace_snapshot"]["evaluation_status"] == "succeeded"
    checkpoint_state = store.load_checkpoint(task_id)
    assert checkpoint_state is not None
    assert checkpoint_state.status is TaskStatus.COMPLETED
    replayed_trace = _execution_trace_for_state(checkpoint_state)
    assert replayed_trace is not None
    assert replayed_trace.evaluation_status is EvaluationStatus.SUCCEEDED
    assert _status_response(state).execution_trace.evaluation_status is EvaluationStatus.SUCCEEDED


def test_interrupted_recovery_persists_inconclusive_terminal_trace(monkeypatch, tmp_path):
    import app.api.routes.tasks as task_routes

    store = SQLiteCheckpointer(tmp_path / "recovered-terminal-trace.db")
    monkeypatch.setattr(task_routes, "checkpointer", store)
    state = TaskState(
        task_id="recovered-terminal-trace",
        objective="",
        privacy_mode="local_only",
        status=TaskStatus.RUNNING,
        plan=ExecutionPlan(
            plan_id="plan_interrupted",
            goal="system_health",
            steps=[ExecutionStep(
                step_id="step_health",
                title="System health",
                description="System health",
                capability="system",
                tool_name="system_health",
                status=StepStatus.IN_PROGRESS,
            )],
        ),
    )
    store.save_checkpoint(state)
    handoff = type("TraceSource", (), {
        "intent_category": IntentCategory.SYSTEM_HEALTH,
        "intent_confidence": 0.9,
        "provider_identity": ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER,
    })()
    initial_trace = orchestrator_module.build_handoff_trace(handoff, state)
    store.append_event(TaskEvent(
        event_type="plan_created",
        task_id=state.task_id,
        message="Plan created.",
        data={"trace_snapshot": initial_trace.model_dump(mode="json")},
    ))

    assert store.recover_interrupted_tasks() == 1
    recovered = store.load_checkpoint(state.task_id)
    assert recovered is not None
    assert recovered.status is TaskStatus.FAILED
    terminal_event = next(
        event for _, event in store.read_events(state.task_id)
        if event.event_type == "task_failed"
    )
    assert terminal_event.data["trace_snapshot"]["evaluation_status"] == "inconclusive"
    assert terminal_event.data["trace_snapshot"]["step_statuses"][0]["status"] == "interrupted"
    replayed = _execution_trace_for_state(recovered)
    assert replayed is not None
    assert replayed.evaluation_status is EvaluationStatus.INCONCLUSIVE
    assert replayed.step_statuses[0].status is StepStatus.INTERRUPTED
