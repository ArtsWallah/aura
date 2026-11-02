import asyncio
import importlib
import json

import pytest

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.execution_handoff import ExecutionHandoff, HandoffFailureCode, ExecutionHandoffError
from app.agent.intelligence_contracts import ProviderIdentity, RequiredInputKind
from app.agent.intelligence_pipeline import IntelligencePipeline, PipelineDisposition, PipelineFailureCode
from app.agent.service import ProcessLocalTaskPayload, TaskExecutionService
from app.agent.state import TaskStatus


tasks_module = importlib.import_module("app.api.routes.tasks")
service_module = importlib.import_module("app.agent.service")


class FixedPipeline:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class FakeWorkerOrchestrator:
    def __init__(self, store):
        self.store = store
        self.handoffs = []
        self.legacy_calls = []
        self.called = asyncio.Event()

    async def run_handoff(self, handoff, *, privacy_mode, task_id, initial_state):
        self.handoffs.append((handoff, privacy_mode, task_id))
        initial_state.transition_to(TaskStatus.PLANNING)
        initial_state.transition_to(TaskStatus.RUNNING)
        initial_state.transition_to(TaskStatus.COMPLETED)
        self.store.save_checkpoint(initial_state)
        self.called.set()
        return initial_state

    async def run_task(self, **kwargs):
        self.legacy_calls.append(kwargs)
        raise AssertionError("new task submissions must never use run_task")

    def get_task_state(self, _task_id):
        return None


async def _ready():
    return await IntelligencePipeline().run(
        "Check system health", privacy_mode="local_only"
    )


def _request(**kwargs):
    values = {"objective": "Check system health", "privacy_mode": "local_only"}
    values.update(kwargs)
    return tasks_module.CreateTaskRequest(**values)


@pytest.mark.asyncio
async def test_ready_api_submission_queues_one_exact_handoff_and_worker_uses_handoff(
    tmp_path, monkeypatch
):
    result = await _ready()
    store = SQLiteCheckpointer(tmp_path / "ready.db")
    monkeypatch.setattr(service_module, "checkpointer", store)
    runner = FakeWorkerOrchestrator(store)
    service = TaskExecutionService(runner)
    service._running = True
    service._wake = asyncio.Event()
    pipeline = FixedPipeline(result=result)
    monkeypatch.setattr(tasks_module, "intelligence_pipeline", pipeline)
    monkeypatch.setattr(tasks_module, "task_execution_service", service)

    accepted = await tasks_module.create_task(_request())
    assert accepted.status == TaskStatus.QUEUED
    assert len(store.list_checkpoints()) == 1
    assert store.latest_event_sequence(accepted.task_id) == 1
    assert pipeline.calls[0]["privacy_mode"] == "local_only"
    assert runner.legacy_calls == []

    worker = asyncio.create_task(service._worker_loop("handoff-worker"))
    await asyncio.wait_for(runner.called.wait(), timeout=3)
    service._running = False
    service._wake.set()
    await asyncio.wait_for(worker, timeout=3)

    assert len(runner.handoffs) == 1
    handoff, mode, task_id = runner.handoffs[0]
    assert isinstance(handoff, ExecutionHandoff)
    assert mode == "local_only" and task_id == accepted.task_id
    assert handoff.plan == result.planning_response.plan
    assert handoff.provider_identity == result.provider_identity
    assert runner.legacy_calls == []
    assert store.load_checkpoint(accepted.task_id).status is TaskStatus.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", [
    PipelineDisposition.CLARIFICATION_REQUIRED,
    PipelineDisposition.REJECTED,
])
async def test_non_ready_submission_creates_no_task_or_queue_row(
    disposition, tmp_path, monkeypatch
):
    if disposition is PipelineDisposition.CLARIFICATION_REQUIRED:
        result = await IntelligencePipeline().run("Analyze this", privacy_mode="local_only")
    else:
        result = await IntelligencePipeline().run("Teleport a database to Mars", privacy_mode="local_only")
    assert result.disposition is disposition
    store = SQLiteCheckpointer(tmp_path / f"{disposition.value}.db")
    runner = FakeWorkerOrchestrator(store)
    service = TaskExecutionService(runner)
    pipeline = FixedPipeline(result=result)
    monkeypatch.setattr(service_module, "checkpointer", store)
    monkeypatch.setattr(tasks_module, "task_execution_service", service)
    monkeypatch.setattr(tasks_module, "intelligence_pipeline", pipeline)

    response = await tasks_module.create_task(_request())
    assert response.status_code == (200 if disposition is PipelineDisposition.CLARIFICATION_REQUIRED else 422)
    assert store.list_checkpoints() == []
    assert service._payloads == {}
    assert runner.handoffs == [] and runner.legacy_calls == []
    assert "objective" not in response.body.decode().lower()
    if disposition is PipelineDisposition.CLARIFICATION_REQUIRED:
        body = json.loads(response.body)
        assert set(body) == {"outcome", "question", "required_input"}
        assert "task_id" not in body


@pytest.mark.asyncio
async def test_pipeline_exception_and_handoff_failure_never_enqueue_or_fallback(monkeypatch, tmp_path):
    store = SQLiteCheckpointer(tmp_path / "planning-failure.db")
    runner = FakeWorkerOrchestrator(store)
    service = TaskExecutionService(runner)
    monkeypatch.setattr(service_module, "checkpointer", store)
    monkeypatch.setattr(tasks_module, "task_execution_service", service)
    monkeypatch.setattr(tasks_module, "intelligence_pipeline", FixedPipeline(error=RuntimeError("raw secret")))

    failed = await tasks_module.create_task(_request())
    assert failed.status_code == 503
    assert "raw secret" not in failed.body.decode()
    assert store.list_checkpoints() == []

    monkeypatch.setattr(tasks_module, "intelligence_pipeline", FixedPipeline(result=await _ready()))

    def fail_handoff(cls, *args, **kwargs):
        raise ExecutionHandoffError(HandoffFailureCode.REGISTRY_MISMATCH)

    monkeypatch.setattr(ExecutionHandoff, "from_pipeline_result", classmethod(fail_handoff))
    failed_handoff = await tasks_module.create_task(_request())
    assert failed_handoff.status_code == 422
    assert store.list_checkpoints() == []
    assert runner.legacy_calls == []


@pytest.mark.asyncio
async def test_pipeline_failure_code_maps_to_sanitized_rejection(monkeypatch, tmp_path):
    result = await IntelligencePipeline().run("Teleport a database to Mars", privacy_mode="local_only")
    assert result.failure_code is PipelineFailureCode.UNSUPPORTED_INTENT
    store = SQLiteCheckpointer(tmp_path / "reject.db")
    monkeypatch.setattr(service_module, "checkpointer", store)
    monkeypatch.setattr(tasks_module, "task_execution_service", TaskExecutionService())
    monkeypatch.setattr(tasks_module, "intelligence_pipeline", FixedPipeline(result=result))

    response = await tasks_module.create_task(_request(objective="private unsupported request"))
    assert response.status_code == 422
    assert response.body.decode() == (
        '{"outcome":"rejected","code":"unsupported_request",'
        '"message":"This request is not supported."}'
    )
    assert store.list_checkpoints() == []


def test_client_cannot_supply_plan_or_handoff_fields():
    for field, value in (
        ("plan", {"steps": []}),
        ("handoff", {"plan": {"steps": []}}),
        ("provider_identity", "gemini"),
        ("registry_identity", "fake-registry"),
        ("execution_arguments", {"path": "private"}),
        ("clarification_state", {"required_input": "dataset"}),
    ):
        with pytest.raises(ValueError):
            tasks_module.CreateTaskRequest.model_validate({
                "objective": "Check system health",
                field: value,
            })


@pytest.mark.asyncio
async def test_only_allowlisted_local_clarification_questions_are_returned():
    base = await IntelligencePipeline().run("Analyze this", privacy_mode="local_only")
    safe_planning = base.planning_response.model_copy(update={
        "clarification_question": "Which numeric column should be checked for outliers?",
    })
    safe_result = base.model_copy(update={"planning_response": safe_planning})
    response = tasks_module._clarification_response(safe_result)
    assert response.question == "Which numeric column should be checked for outliers?"
    assert response.required_input is RequiredInputKind.CLARIFICATION

    marker = "provider-echo-private-objective-773"
    unsafe_planning = base.planning_response.model_copy(update={
        "provider_identity": ProviderIdentity.GEMINI,
        "clarification_question": f"Please repeat {marker}",
    })
    unsafe_result = base.model_copy(update={"planning_response": unsafe_planning})
    fallback = tasks_module._clarification_response(unsafe_result)
    assert fallback.question == "Could you clarify the task and the result you want?"
    assert marker not in fallback.model_dump_json()

    malformed = base.model_copy(update={"intent": None})
    malformed_fallback = tasks_module._clarification_response(malformed)
    assert malformed_fallback.question == "Could you clarify the task and the result you want?"


@pytest.mark.asyncio
async def test_revised_submission_runs_a_fresh_pipeline_and_queues_only_ready_result(
    tmp_path, monkeypatch
):
    real_pipeline = IntelligencePipeline()

    class CountingPipeline:
        def __init__(self):
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs.copy())
            return await real_pipeline.run(**kwargs)

    pipeline = CountingPipeline()
    store = SQLiteCheckpointer(tmp_path / "fresh-reassessment.db")
    service = TaskExecutionService(FakeWorkerOrchestrator(store))
    monkeypatch.setattr(service_module, "checkpointer", store)
    monkeypatch.setattr(tasks_module, "task_execution_service", service)
    monkeypatch.setattr(tasks_module, "intelligence_pipeline", pipeline)

    clarification = await tasks_module.create_task(_request(objective="Analyze this", privacy_mode="hybrid"))
    assert clarification.status_code == 200
    assert store.list_checkpoints() == []
    assert service._payloads == {}

    accepted = await tasks_module.create_task(
        _request(objective="Check system health", privacy_mode="hybrid")
    )
    assert accepted.status.value == "queued"
    assert len(pipeline.calls) == 2
    assert [call["objective"] for call in pipeline.calls] == [
        "Analyze this", "Check system health"
    ]
    assert all(call["privacy_mode"] == "hybrid" for call in pipeline.calls)
    assert set(pipeline.calls[1]) == {"objective", "available_inputs", "privacy_mode"}
    assert len(store.list_checkpoints()) == 1
    assert len(service._payloads) == 1
    payload = service._payloads[accepted.task_id]
    assert payload.handoff_required is True
    assert payload.handoff.plan.steps[0].tool_name == "system_health"


@pytest.mark.asyncio
async def test_missing_handoff_payload_fails_without_legacy_fallback(tmp_path, monkeypatch):
    store = SQLiteCheckpointer(tmp_path / "missing-handoff.db")
    monkeypatch.setattr(service_module, "checkpointer", store)
    runner = FakeWorkerOrchestrator(store)
    service = TaskExecutionService(runner)
    task_id = "task_missing_handoff"
    from app.agent.state import TaskState

    queued = TaskState(
        task_id=task_id,
        objective="",
        privacy_mode="local_only",
        status=TaskStatus.QUEUED,
    )
    store.enqueue_task(queued)
    store.append_event(tasks_module.TaskEvent(
        event_type="task.created", task_id=task_id, message="Task accepted.", data={"status": "queued"}
    ))
    service._payloads[task_id] = ProcessLocalTaskPayload(
        task_state=queued, handoff=None, handoff_required=True
    )
    service._running = True
    service._wake = asyncio.Event()

    worker = asyncio.create_task(service._worker_loop("missing-handoff-worker"))
    for _ in range(100):
        state = store.load_checkpoint(task_id)
        if state and state.status is TaskStatus.FAILED:
            break
        await asyncio.sleep(0.01)
    service._running = False
    service._wake.set()
    await asyncio.wait_for(worker, timeout=3)

    failed = store.load_checkpoint(task_id)
    assert failed.status is TaskStatus.FAILED
    assert failed.recovery_required is False
    assert runner.handoffs == [] and runner.legacy_calls == []


@pytest.mark.asyncio
async def test_restart_with_missing_process_payload_never_reconstructs_or_executes(tmp_path, monkeypatch):
    store = SQLiteCheckpointer(tmp_path / "restart.db")
    monkeypatch.setattr(service_module, "checkpointer", store)
    runner = FakeWorkerOrchestrator(store)
    service = TaskExecutionService(runner)
    from app.agent.state import TaskState

    state = TaskState(
        task_id="task_after_restart",
        objective="",
        privacy_mode="local_only",
        status=TaskStatus.QUEUED,
    )
    store.enqueue_task(state)
    store.append_event(tasks_module.TaskEvent(
        event_type="task.created", task_id=state.task_id, message="Task accepted.", data={"status": "queued"}
    ))
    # A newly constructed service has no process-local handoff payload.
    service._running = True
    service._wake = asyncio.Event()
    worker = asyncio.create_task(service._worker_loop("restart-worker"))
    for _ in range(100):
        recovered = store.load_checkpoint(state.task_id)
        if recovered and recovered.status is TaskStatus.FAILED:
            break
        await asyncio.sleep(0.01)
    service._running = False
    service._wake.set()
    await asyncio.wait_for(worker, timeout=3)
    assert recovered.status is TaskStatus.FAILED
    assert recovered.recovery_required is True
    assert runner.handoffs == [] and runner.legacy_calls == []
