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
        assert {"claimed_by", "claimed_at", "event_seq", "recovery_required"} <= added_columns
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
