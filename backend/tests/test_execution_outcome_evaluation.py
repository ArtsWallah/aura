import asyncio

import pytest

from app.agent import orchestrator as orchestrator_module
from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.execution_handoff import ExecutionHandoff
from app.agent.intelligence_contracts import EvaluationStatus
from app.agent.intelligence_evaluator import IntelligenceEvaluator
from app.agent.intelligence_pipeline import IntelligencePipeline
from app.agent.orchestrator import AgentOrchestrator
from app.agent.state import StepStatus, TaskState, TaskStatus
from app.tools.registry import tool_registry
from app.tools.schemas import ToolResult


async def _multi_step_handoff(step_count=2):
    marker = "RAW-OBJECTIVE-MARKER-441"
    result = await IntelligencePipeline().run(
        f"Check system health and note {marker}", privacy_mode="local_only"
    )
    plan = result.planning_response.plan
    plan.steps[0].step_id = "step_health_a"
    for index in range(1, step_count):
        plan.steps.append(plan.steps[0].model_copy(update={
            "step_id": f"step_health_{chr(ord('a') + index)}",
            "dependencies": [plan.steps[-1].step_id],
            "status": StepStatus.PENDING,
            "output": None,
        }))
    handoff = ExecutionHandoff.from_pipeline_result(
        result, privacy_mode="local_only", registry=tool_registry
    )
    return handoff, marker


class RecordingEvaluator:
    def __init__(self, *, fail=False):
        self.request = None
        self.fail = fail
        self.delegate = IntelligenceEvaluator()

    def evaluate(self, request):
        self.request = request
        if self.fail:
            raise RuntimeError("private evaluator failure")
        return self.delegate.evaluate(request)


async def _run_scripted(outcomes, *, task_status=TaskStatus.COMPLETED, evaluator=None):
    handoff, marker = await _multi_step_handoff(step_count=len(outcomes))
    recorder = evaluator or RecordingEvaluator()
    runner = AgentOrchestrator(execution_evaluator=recorder)

    async def fake_execution(**kwargs):
        state = TaskState(
            task_id="outcome-test",
            objective="",
            privacy_mode="local_only",
            status=TaskStatus.RUNNING,
            plan=kwargs["supplied_plan"],
        )
        for step, scripted in zip(state.plan.steps, outcomes, strict=True):
            step.status = scripted
            step.output = {"sensitive_output": marker} if scripted == StepStatus.COMPLETED else None
            step.retry_count = 1 if scripted == StepStatus.COMPLETED else step.max_retries
            state.step_results[step.step_id] = {
                "data": step.output,
                "artifacts": [f"C:/private/{marker}.png"] if scripted == StepStatus.COMPLETED else [],
            }
        state.status = task_status
        return state

    runner._run_task = fake_execution
    state = await runner.run_handoff(handoff, privacy_mode="local_only")
    return state, recorder, marker


@pytest.mark.asyncio
async def test_handoff_full_execution_evaluates_safe_typed_evidence_only():
    state, recorder, marker = await _run_scripted([StepStatus.COMPLETED, StepStatus.COMPLETED])
    assert state.execution_evaluation.evaluation_status is EvaluationStatus.SUCCEEDED
    assert recorder.request.evaluation_target == "execution"
    assert set(recorder.request.model_dump()) == {
        "intent", "steps", "validation", "evaluation_target", "outcomes", "privacy_mode"
    }
    assert set(recorder.request.steps[0].model_dump()) == {"step_id", "tool_name", "dependencies"}
    assert set(recorder.request.outcomes[0].model_dump()) == {
        "step_id", "status", "result_kind", "result_count", "artifact_count", "retry_count",
        "retry_limit", "failure_category",
    }
    assert marker not in recorder.request.model_dump_json()
    assert "C:/private" not in recorder.request.model_dump_json()
    assert "sensitive_output" not in recorder.request.model_dump_json()
    assert "execution_evaluation" not in state.model_dump()


@pytest.mark.asyncio
async def test_partial_success_failure_and_dependency_blocked_outcomes():
    partial, _, _ = await _run_scripted(
        [StepStatus.COMPLETED, StepStatus.FAILED], task_status=TaskStatus.FAILED
    )
    assert partial.status is TaskStatus.FAILED
    assert partial.execution_evaluation.evaluation_status is EvaluationStatus.PARTIAL_SUCCESS

    blocked, recorder, _ = await _run_scripted(
        [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED], task_status=TaskStatus.FAILED
    )
    assert blocked.execution_evaluation.evaluation_status is EvaluationStatus.PARTIAL_SUCCESS
    assert recorder.request.outcomes[2].failure_category.value == "dependency_blocked"


@pytest.mark.asyncio
async def test_total_failure_cancelled_execution_and_evaluation_failure_are_safe():
    failed, _, _ = await _run_scripted(
        [StepStatus.FAILED, StepStatus.SKIPPED], task_status=TaskStatus.FAILED
    )
    assert failed.execution_evaluation.evaluation_status is EvaluationStatus.EXECUTION_FAILED

    cancelled, _, _ = await _run_scripted(
        [StepStatus.SKIPPED, StepStatus.SKIPPED], task_status=TaskStatus.CANCELLED
    )
    assert cancelled.status is TaskStatus.CANCELLED
    assert cancelled.execution_evaluation.evaluation_status is EvaluationStatus.INCONCLUSIVE

    interrupted, _, _ = await _run_scripted(
        [StepStatus.INTERRUPTED, StepStatus.PENDING], task_status=TaskStatus.FAILED
    )
    assert interrupted.execution_evaluation.evaluation_status is EvaluationStatus.INCONCLUSIVE

    broken, _, _ = await _run_scripted(
        [StepStatus.COMPLETED, StepStatus.COMPLETED], evaluator=RecordingEvaluator(fail=True)
    )
    assert broken.status is TaskStatus.COMPLETED
    assert broken.execution_evaluation.evaluation_status is EvaluationStatus.INCONCLUSIVE


@pytest.mark.asyncio
async def test_recovered_state_without_process_local_evidence_is_inconclusive():
    handoff, _ = await _multi_step_handoff()
    runner = AgentOrchestrator()
    recovered = TaskState(
        task_id="recovered-task",
        objective="",
        privacy_mode="local_only",
        status=TaskStatus.FAILED,
        recovery_required=True,
        plan=handoff.plan.model_copy(deep=True),
    )
    result = runner._evaluate_handoff_outcome(handoff, recovered)
    assert result.evaluation_status is EvaluationStatus.INCONCLUSIVE
    assert "execution_evaluation" not in recovered.model_dump()


@pytest.mark.asyncio
async def test_cancellation_requested_during_active_step_cannot_evaluate_as_success(monkeypatch, tmp_path):
    result = await IntelligencePipeline().run("Check system health", privacy_mode="local_only")
    handoff = ExecutionHandoff.from_pipeline_result(result, privacy_mode="local_only")
    runner = AgentOrchestrator()
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_execution(step, task_state, event_queue=None):
        step.status = StepStatus.IN_PROGRESS
        started.set()
        await release.wait()
        step.status = StepStatus.COMPLETED
        step.output = {"safe_synthetic_result": True}
        task_state.step_results[step.step_id] = {"data": step.output, "artifacts": []}
        return ToolResult(tool_name=step.tool_name, success=True, data=step.output)

    monkeypatch.setattr(orchestrator_module.executor, "execute_step", blocked_execution)
    monkeypatch.setattr(
        orchestrator_module,
        "checkpointer",
        SQLiteCheckpointer(tmp_path / "cancel-outcome.db"),
    )
    operation = asyncio.create_task(runner.run_handoff(handoff))
    await started.wait()
    task_id = next(iter(runner._active_tasks))
    assert runner.request_cancel(task_id) == "cancel_requested"
    release.set()
    state = await operation

    assert state.status is TaskStatus.CANCELLED
    assert state.execution_evaluation.evaluation_status is EvaluationStatus.INCONCLUSIVE


@pytest.mark.asyncio
async def test_legacy_run_task_does_not_attach_handoff_evaluation():
    runner = AgentOrchestrator(execution_evaluator=RecordingEvaluator())
    state = TaskState(task_id="legacy-task", objective="", status=TaskStatus.COMPLETED)

    async def legacy_execution(*args, **kwargs):
        return state

    runner._run_task = legacy_execution
    returned = await runner.run_task("private objective", privacy_mode="local_only")
    assert returned is state
    assert returned.execution_evaluation is None
