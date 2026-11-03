import pytest

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
