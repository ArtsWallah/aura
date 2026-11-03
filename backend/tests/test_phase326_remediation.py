from types import SimpleNamespace

import pytest

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.execution_handoff import ExecutionHandoffError
from app.agent.execution_trace import build_handoff_trace
from app.agent.orchestrator import AgentOrchestrator
from app.agent.service import TaskExecutionService
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus, TaskState
from app.core.config import settings
import app.core.config as config_module
import app.api.routes.tasks as task_routes


def _chart_trace_state(task_id: str, artifact_path: str) -> TaskState:
    state = TaskState(
        task_id=task_id,
        objective="",
        privacy_mode="local_only",
        plan=ExecutionPlan(
            plan_id=f"plan_{task_id}",
            goal="visualization",
            steps=[ExecutionStep(
                step_id="step_chart",
                title="Generate chart",
                description="Generate chart",
                capability="visualization",
                tool_name="generate_chart",
                status=StepStatus.COMPLETED,
            )],
        ),
    )
    state.plan.steps[0].output = {"ok": True}
    state.step_results["step_chart"] = {"artifacts": [artifact_path]}
    state.artifacts = [artifact_path]
    return state


def _trace_for(state: TaskState, manifest: list[dict[str, str]]):
    handoff = SimpleNamespace(
        intent_category="visualization",
        intent_confidence=0.9,
        provider_identity="local_deterministic_planner",
    )
    return build_handoff_trace(handoff, state, artifact_manifest=manifest)


@pytest.mark.asyncio
async def test_production_legacy_execution_entries_reject_before_any_tool_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    runner = AgentOrchestrator()
    calls = []

    async def record_tool(*args, **kwargs):
        calls.append(True)

    monkeypatch.setattr("app.agent.orchestrator.executor.execute_step", record_tool)
    with pytest.raises(ExecutionHandoffError):
        await runner.run_task("unsafe legacy request", privacy_mode="local_only")
    with pytest.raises(ExecutionHandoffError):
        await runner._run_task("unsafe direct request", privacy_mode="local_only")

    service = TaskExecutionService(runner)
    with pytest.raises(ExecutionHandoffError):
        await service.enqueue(TaskState(task_id="legacy-enqueue", objective="unsafe"))

    assert calls == []
    assert runner._active_tasks == {}
    assert service._payloads == {}


def test_task_list_contract_is_summary_only(monkeypatch):
    row = {
        "task_id": "task_summary",
        "status": "completed",
        "privacy_mode": "local_only",
        "created_at": 1.0,
        "updated_at": 2.0,
    }
    monkeypatch.setattr(task_routes.checkpointer, "list_checkpoints", lambda limit: [row])
    listed = task_routes.list_tasks()
    assert len(listed) == 1
    assert listed[0].model_dump() == row
    assert "execution_trace" not in listed[0].model_dump()


def test_test_compatibility_requires_latched_pytest_process_and_safe_environment(monkeypatch):
    monkeypatch.setattr(config_module, "_PYTEST_PROCESS_AT_IMPORT", True)
    monkeypatch.setattr(settings, "ENVIRONMENT", "test")
    assert config_module.test_harness_enabled() is True
    config_module.validate_service_environment()

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    assert config_module.test_harness_enabled() is False
    config_module.validate_service_environment()
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    config_module.validate_service_environment()

    # An ordinary settings request cannot set ENVIRONMENT.
    from app.api.routes.settings import UpdateSettingsRequest, update_settings
    request = UpdateSettingsRequest.model_validate({"environment": "test"})
    update_settings(request)
    assert settings.ENVIRONMENT == "development"

    monkeypatch.setattr(config_module, "_PYTEST_PROCESS_AT_IMPORT", False)
    monkeypatch.setattr(settings, "ENVIRONMENT", "test")
    assert config_module.test_harness_enabled() is False
    with pytest.raises(RuntimeError):
        config_module.validate_service_environment()


def test_trace_reports_only_existing_task_manifest_artifacts(monkeypatch, tmp_path):
    charts = tmp_path / "charts"
    reports = tmp_path / "reports"
    charts.mkdir()
    reports.mkdir()
    monkeypatch.setattr(settings, "CHARTS_DIR", charts)
    monkeypatch.setattr(settings, "REPORTS_DIR", reports)
    store = SQLiteCheckpointer(tmp_path / "trace-artifacts.db")

    valid_path = charts / "chart_aaaaaaaaaa.png.enc"
    valid_path.write_bytes(b"encrypted-chart")
    valid_state = _chart_trace_state("task_chart_a", str(valid_path))
    valid_manifest = store.artifact_manifest_for_task(valid_state)
    trace = _trace_for(valid_state, valid_manifest)
    assert trace is not None
    assert trace.step_statuses[0].artifacts[0].available is True
    assert trace.step_statuses[0].artifacts[0].count == 1
    assert str(valid_path) not in trace.model_dump_json()

    missing_path = charts / "chart_bbbbbbbbbb.png.enc"
    missing_state = _chart_trace_state("task_chart_missing", str(missing_path))
    missing_trace = _trace_for(missing_state, store.artifact_manifest_for_task(missing_state))
    assert missing_trace is not None
    assert missing_trace.step_statuses[0].artifacts[0].available is False

    outside_path = tmp_path / "outside" / "chart_cccccccccc.png.enc"
    outside_path.parent.mkdir()
    outside_path.write_bytes(b"not-task-scoped")
    outside_state = _chart_trace_state("task_chart_outside", str(outside_path))
    outside_trace = _trace_for(outside_state, store.artifact_manifest_for_task(outside_state))
    assert outside_trace is not None
    assert outside_trace.step_statuses[0].artifacts[0].available is False

    task_b_state = _chart_trace_state("task_chart_b", str(valid_path))
    task_b_manifest = store.artifact_manifest_for_task(task_b_state)
    task_a_state = _chart_trace_state("task_chart_a", str(valid_path))
    cross_task_trace = _trace_for(task_a_state, task_b_manifest)
    assert cross_task_trace is not None
    assert cross_task_trace.step_statuses[0].artifacts[0].available is False
