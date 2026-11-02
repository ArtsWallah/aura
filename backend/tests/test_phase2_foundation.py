import json
import sqlite3
import uuid
from contextlib import closing

import pytest
from cryptography.fernet import Fernet

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.plan_validation import PlanValidationError, validate_execution_plan
from app.agent.planner import TaskPlanner
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus, TaskState, TaskStatus
from app.core.config import Settings, settings
from app.llm.mock_provider import TestReasoningProvider as MockReasoningProvider
from app.tools.registry import tool_registry
from app.tools.schemas import PermissionLevel


def _plan(*steps):
    return ExecutionPlan(plan_id="plan_test", goal="safe test objective", steps=list(steps))


def _step(step_id="step_1", tool_name="system_health", parameters=None, dependencies=None):
    return ExecutionStep(
        step_id=step_id,
        title="Safe test step",
        description="Test plan validation.",
        capability="system",
        tool_name=tool_name,
        parameters=parameters or {},
        dependencies=dependencies or [],
    )


@pytest.fixture
def phase2_temp_dir():
    path = settings.DATA_DIR / f"phase2_foundation_{uuid.uuid4().hex}"
    path.mkdir()
    yield path
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
    path.rmdir()


def test_task_state_transitions_are_explicit_and_terminal_states_cannot_resume():
    state = TaskState(task_id="phase2-state", objective="safe")
    assert state.status == TaskStatus.QUEUED
    state.transition_to(TaskStatus.PLANNING)
    state.transition_to(TaskStatus.RUNNING)
    state.transition_to(TaskStatus.CANCEL_REQUESTED)
    state.transition_to(TaskStatus.CANCELLED)
    with pytest.raises(ValueError, match="Invalid task transition"):
        state.transition_to(TaskStatus.RUNNING)
    assert state.status == TaskStatus.CANCELLED


def test_execution_capacity_settings_reject_unbounded_or_invalid_values():
    with pytest.raises(ValueError):
        Settings(MAX_CONCURRENT_TASKS=0)
    with pytest.raises(ValueError):
        Settings(MAX_CONCURRENT_TOOL_EXECUTIONS=1000)


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [("pending", "queued"), ("executing", "running"), ("evaluating", "running")],
)
def test_legacy_checkpoint_statuses_load_into_new_state_model(legacy, expected):
    state = TaskState.model_validate({"task_id": "legacy", "objective": "safe", "status": legacy})
    assert state.status.value == expected


def test_fallback_plan_is_validated_before_it_can_be_returned():
    planner = TaskPlanner(MockReasoningProvider())
    plan = planner._create_fallback_plan("check system status", None)
    assert validate_execution_plan(plan, privacy_mode="local_only") is plan


def test_plan_validation_rejects_duplicate_step_ids():
    with pytest.raises(PlanValidationError) as exc:
        validate_execution_plan(_plan(_step(), _step()))
    assert exc.value.code == "duplicate_step_id"


def test_plan_validation_rejects_missing_dependencies_and_cycles():
    with pytest.raises(PlanValidationError) as missing:
        validate_execution_plan(_plan(_step(dependencies=["step_missing"])))
    assert missing.value.code == "missing_dependency"

    with pytest.raises(PlanValidationError) as cycle:
        validate_execution_plan(_plan(
            _step("step_a", dependencies=["step_b"]),
            _step("step_b", dependencies=["step_a"]),
        ))
    assert cycle.value.code == "dependency_cycle"


def test_plan_validation_rejects_unknown_tools_and_invalid_parameters():
    with pytest.raises(PlanValidationError) as unknown:
        validate_execution_plan(_plan(_step(tool_name="not_registered")))
    assert unknown.value.code == "unknown_tool"

    with pytest.raises(PlanValidationError) as invalid:
        validate_execution_plan(_plan(_step(
            tool_name="inspect_dataset",
            parameters={"file_path": "managed.csv.enc", "sample_rows": "not-an-integer"},
        )))
    assert invalid.value.code == "invalid_tool_parameters"
    assert "not-an-integer" not in str(invalid.value)


def test_plan_validation_enforces_step_limit_privacy_and_approval(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PLAN_STEPS", 1)
    with pytest.raises(PlanValidationError) as too_many:
        validate_execution_plan(_plan(_step("step_a"), _step("step_b")))
    assert too_many.value.code == "step_limit"

    with pytest.raises(PlanValidationError) as privacy:
        validate_execution_plan(_plan(_step(tool_name="web_research", parameters={"query": "safe"})), "local_only")
    assert privacy.value.code == "privacy_policy"
    with pytest.raises(PlanValidationError) as hybrid_privacy:
        validate_execution_plan(_plan(_step(tool_name="web_research", parameters={"query": "safe"})), "hybrid")
    assert hybrid_privacy.value.code == "privacy_policy"

    original = tool_registry._tools["generate_report"]
    tool_registry._tools["generate_report"] = original.model_copy(update={"requires_approval": True})
    try:
        args = {"title": "t", "executive_summary": "s", "findings": []}
        with pytest.raises(PlanValidationError) as approval:
            validate_execution_plan(_plan(_step(tool_name="generate_report", parameters=args)))
        assert approval.value.code == "approval_required"
    finally:
        tool_registry._tools["generate_report"] = original


def test_checkpoint_schema_migration_preserves_encrypted_rows_without_or_with_wrong_key(phase2_temp_dir, monkeypatch):
    key = Fernet.generate_key()
    wrong_key = Fernet.generate_key()
    db_path = phase2_temp_dir / "legacy_checkpoints.db"
    state_json = {
        "task_id": "legacy-encrypted",
        "objective": "sensitive objective retained",
        "privacy_mode": "hybrid",
        "status": "executing",
        "created_at": 1.0,
        "updated_at": 2.0,
    }
    encrypted = Fernet(key).encrypt(json.dumps(state_json).encode("utf-8")).decode("ascii")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("""
            CREATE TABLE task_checkpoints (
                task_id TEXT PRIMARY KEY, objective TEXT NOT NULL, privacy_mode TEXT NOT NULL,
                status TEXT NOT NULL, state_json TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO task_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("legacy-encrypted", "", "hybrid", "executing", encrypted, 1.0, 2.0),
        )
        conn.commit()

    monkeypatch.setattr(settings, "ENCRYPTION_KEY", None)
    no_key_store = SQLiteCheckpointer(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT state_json FROM task_checkpoints").fetchone()[0] == encrypted
        added_columns = {row[1] for row in conn.execute("PRAGMA table_info(task_checkpoints)")}
        assert {"claimed_by", "claimed_at", "event_seq", "recovery_required", "artifact_manifest"} <= added_columns
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='task_events'").fetchone()
    assert no_key_store.load_checkpoint("legacy-encrypted") is None

    monkeypatch.setattr(settings, "ENCRYPTION_KEY", wrong_key.decode("ascii"))
    SQLiteCheckpointer(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT state_json FROM task_checkpoints").fetchone()[0] == encrypted

    monkeypatch.setattr(settings, "ENCRYPTION_KEY", key.decode("ascii"))
    valid_store = SQLiteCheckpointer(db_path)
    loaded = valid_store.load_checkpoint("legacy-encrypted")
    assert loaded is not None
    assert loaded.status == TaskStatus.RUNNING
    assert loaded.objective == "sensitive objective retained"
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT state_json FROM task_checkpoints").fetchone()[0] == encrypted


def test_legacy_plaintext_checkpoint_migration_preserves_user_data_when_key_is_valid(phase2_temp_dir):
    db_path = phase2_temp_dir / "legacy_plaintext.db"
    state = TaskState(task_id="legacy-plain", objective="preserve this prior user objective")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE task_checkpoints (task_id TEXT PRIMARY KEY, objective TEXT NOT NULL, privacy_mode TEXT NOT NULL, status TEXT NOT NULL, state_json TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
        conn.execute(
            "INSERT INTO task_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
            (state.task_id, state.objective, state.privacy_mode, state.status.value, json.dumps(state.model_dump()), state.created_at, state.updated_at),
        )
        conn.commit()

    store = SQLiteCheckpointer(db_path)
    loaded = store.load_checkpoint(state.task_id)
    assert loaded is not None
    assert loaded.objective == state.objective
    with closing(sqlite3.connect(db_path)) as conn:
        encrypted = conn.execute("SELECT state_json FROM task_checkpoints WHERE task_id = ?", (state.task_id,)).fetchone()[0]
    assert state.objective not in encrypted


def test_persistent_queue_claim_is_atomic_and_reuses_the_task_id(phase2_temp_dir):
    store = SQLiteCheckpointer(phase2_temp_dir / "queue.db")
    state = TaskState(task_id="one-stable-task-id", objective="safe objective")
    store.enqueue_task(state)

    claimed = store.claim_next_queued_task("worker-a")
    duplicate = store.claim_next_queued_task("worker-b")

    assert claimed is not None
    assert claimed.task_id == state.task_id
    assert claimed.objective == ""
    assert claimed.file_path is None
    assert duplicate is None
    store.release_task_claim(state.task_id, "worker-a")


@pytest.mark.asyncio
async def test_worker_polling_cannot_claim_until_concurrent_enqueue_publishes_payload(
    phase2_temp_dir, monkeypatch
):
    import asyncio
    import threading
    import app.agent.service as service_module
    from app.agent.service import TaskExecutionService

    store = SQLiteCheckpointer(phase2_temp_dir / "enqueue-race.db")
    monkeypatch.setattr(service_module, "checkpointer", store)
    committed = threading.Event()
    allow_enqueue_return = threading.Event()
    original_enqueue = store.enqueue_task

    def delayed_enqueue(task):
        original_enqueue(task)
        committed.set()
        assert allow_enqueue_return.wait(timeout=3)

    monkeypatch.setattr(store, "enqueue_task", delayed_enqueue)

    class RecordingOrchestrator:
        def __init__(self):
            self.seen = []

        async def run_task(self, **kwargs):
            self.seen.append((kwargs["task_id"], kwargs["objective"]))
            state = kwargs["initial_state"]
            state.transition_to(TaskStatus.PLANNING)
            state.transition_to(TaskStatus.RUNNING)
            state.transition_to(TaskStatus.COMPLETED)
            store.save_checkpoint(state)

    runner = RecordingOrchestrator()
    service = TaskExecutionService(runner)
    service._running = True
    service._wake = asyncio.Event()

    submissions = [
        asyncio.create_task(service.enqueue(TaskState(task_id=f"race-{n}", objective=f"private-{n}")))
        for n in range(2)
    ]
    await asyncio.wait_for(asyncio.to_thread(committed.wait), timeout=2)
    worker = asyncio.create_task(service._worker_loop("race-worker"))
    await asyncio.sleep(0.1)
    assert runner.seen == []
    assert service._payloads == {}

    allow_enqueue_return.set()
    await asyncio.gather(*submissions)
    service._wake.set()
    for _ in range(100):
        if len(runner.seen) == 2:
            break
        await asyncio.sleep(0.02)
    service._running = False
    service._wake.set()
    await asyncio.wait_for(worker, timeout=2)

    assert sorted(runner.seen) == [("race-0", "private-0"), ("race-1", "private-1")]
    assert store.latest_event_sequence("race-0") == 1
    assert store.latest_event_sequence("race-1") == 1


@pytest.mark.asyncio
async def test_enqueue_event_failure_rolls_back_queue_row_and_payload(phase2_temp_dir, monkeypatch):
    import app.agent.service as service_module
    from app.agent.service import TaskExecutionService

    store = SQLiteCheckpointer(phase2_temp_dir / "enqueue-rollback.db")
    monkeypatch.setattr(service_module, "checkpointer", store)

    def fail_event(_event):
        raise RuntimeError("event store unavailable")

    monkeypatch.setattr(store, "append_event", fail_event)
    service = TaskExecutionService()
    state = TaskState(task_id="enqueue-rollback", objective="private")
    with pytest.raises(RuntimeError, match="event store unavailable"):
        await service.enqueue(state)

    assert service._payloads == {}
    assert store.load_checkpoint(state.task_id) is None
    assert store.claim_next_queued_task("rollback-check") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["enqueue", "event"])
async def test_cancelled_enqueue_drains_database_thread_and_rolls_back_before_worker_claim(
    phase2_temp_dir, monkeypatch, blocked_stage
):
    import asyncio
    import sqlite3
    import threading
    from contextlib import closing
    import app.agent.service as service_module
    from app.agent.service import TaskExecutionService

    store = SQLiteCheckpointer(phase2_temp_dir / f"cancel-enqueue-{blocked_stage}.db")
    monkeypatch.setattr(service_module, "checkpointer", store)
    committed = threading.Event()
    release_return = threading.Event()
    executions = []

    def block_after_commit(operation):
        def wrapped(*args):
            result = operation(*args)
            committed.set()
            assert release_return.wait(timeout=4)
            return result
        return wrapped

    if blocked_stage == "enqueue":
        store.enqueue_task = block_after_commit(store.enqueue_task)
    else:
        store.append_event = block_after_commit(store.append_event)

    class RecordingOrchestrator:
        async def run_task(self, **kwargs):
            executions.append(kwargs["task_id"])

    service = TaskExecutionService(RecordingOrchestrator())
    service._running = True
    service._wake = asyncio.Event()
    state = TaskState(task_id=f"cancelled-enqueue-{blocked_stage}", objective="private objective")
    enqueue_task = asyncio.create_task(service.enqueue(state))
    await asyncio.wait_for(asyncio.to_thread(committed.wait), timeout=2)

    worker = asyncio.create_task(service._worker_loop("cancel-race-worker"))
    enqueue_task.cancel()
    await asyncio.sleep(0.05)
    assert not enqueue_task.done()
    assert executions == []

    release_return.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(enqueue_task, timeout=2)

    service._running = False
    service._wake.set()
    await asyncio.wait_for(worker, timeout=2)

    assert executions == []
    assert state.task_id not in service._payloads
    assert store.load_checkpoint(state.task_id) is None
    assert store.claim_next_queued_task("post-cancel-check") is None
    with closing(sqlite3.connect(store.db_path)) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (state.task_id,)
        ).fetchone()[0] == 0


def test_restart_recovery_marks_inflight_work_ambiguous_without_replaying_it(phase2_temp_dir):
    store = SQLiteCheckpointer(phase2_temp_dir / "recovery.db")
    state = TaskState(
        task_id="interrupted-task",
        objective="sensitive objective",
        status=TaskStatus.RUNNING,
        plan=_plan(ExecutionStep(
            step_id="step_1", title="Possible side effect", description="safe", capability="report",
            tool_name="generate_report", parameters={"title": "private", "executive_summary": "private", "findings": []},
            status=StepStatus.IN_PROGRESS,
        )),
    )
    store.save_checkpoint(state)

    assert store.recover_interrupted_tasks() == 1
    recovered = store.load_checkpoint(state.task_id)
    assert recovered is not None
    assert recovered.status == TaskStatus.FAILED
    assert recovered.recovery_required is True
    assert recovered.plan.steps[0].status == StepStatus.INTERRUPTED
    assert "private" not in str(recovered.recovery_reason)


def test_restart_does_not_requeue_payload_less_queued_task(phase2_temp_dir):
    store = SQLiteCheckpointer(phase2_temp_dir / "queued-recovery.db")
    state = TaskState(task_id="queued-no-payload", objective="must not be persisted")
    store.enqueue_task(state)

    assert store.recover_interrupted_tasks() == 1
    recovered = store.load_checkpoint(state.task_id)
    assert recovered.status == TaskStatus.FAILED
    assert recovered.recovery_required is True
    assert recovered.objective == ""
    events = store.read_events(state.task_id)
    assert events[-1][1].event_type == "task_failed"
