import time
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, PrivateAttr, model_validator

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING = "waiting"
    RETRYING = "retrying"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    # Source compatibility while existing orchestration code is migrated.
    PENDING = "queued"
    EXECUTING = "running"
    EVALUATING = "running"


TASK_STATUS_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset({TaskStatus.PLANNING, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.PLANNING: frozenset({TaskStatus.RUNNING, TaskStatus.WAITING, TaskStatus.CANCEL_REQUESTED, TaskStatus.FAILED}),
    TaskStatus.RUNNING: frozenset({TaskStatus.WAITING, TaskStatus.RETRYING, TaskStatus.CANCEL_REQUESTED, TaskStatus.COMPLETED, TaskStatus.FAILED}),
    TaskStatus.WAITING: frozenset({TaskStatus.RUNNING, TaskStatus.RETRYING, TaskStatus.CANCEL_REQUESTED, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.RETRYING: frozenset({TaskStatus.RUNNING, TaskStatus.WAITING, TaskStatus.CANCEL_REQUESTED, TaskStatus.FAILED}),
    TaskStatus.CANCEL_REQUESTED: frozenset({TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"

class ExecutionStep(BaseModel):
    step_id: str
    title: str
    description: str
    capability: str
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    output: Any = None
    error: str | None = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=2, ge=0)
    status: StepStatus = StepStatus.PENDING
    duration_ms: float = 0.0

class ExecutionPlan(BaseModel):
    plan_id: str
    goal: str
    steps: list[ExecutionStep] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)

class TaskMetrics(BaseModel):
    total_execution_time_ms: float = 0.0
    tool_calls_count: int = 0
    llm_calls_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries_count: int = 0
    success_rate: float = 1.0

class TaskState(BaseModel):
    _execution_evaluation: Any = PrivateAttr(default=None)
    _execution_trace: Any = PrivateAttr(default=None)

    task_id: str
    objective: str
    file_path: str | None = None
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] = "hybrid"
    status: TaskStatus = TaskStatus.QUEUED
    plan: ExecutionPlan | None = None
    step_results: dict[str, Any] = Field(default_factory=dict)
    final_deliverable: dict[str, Any] | None = None
    artifacts: list[str] = Field(default_factory=list)
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    error_summary: str | None = None
    recovery_required: bool = False
    recovery_reason: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @property
    def execution_evaluation(self) -> Any:
        """Process-local safe evaluation; excluded from checkpoints and API DTOs."""
        return self._execution_evaluation

    def set_execution_evaluation(self, result: Any) -> None:
        self._execution_evaluation = result

    @property
    def execution_trace(self) -> Any:
        """Process-local validated trace; excluded from checkpoints by default."""
        return self._execution_trace

    def set_execution_trace(self, result: Any) -> None:
        self._execution_trace = result

    @model_validator(mode="before")
    @classmethod
    def map_legacy_status(cls, value: Any) -> Any:
        if isinstance(value, dict):
            legacy = {
                "pending": "queued",
                "executing": "running",
                "evaluating": "running",
            }
            status = value.get("status")
            if isinstance(status, str) and status in legacy:
                value = {**value, "status": legacy[status]}
        return value

    def transition_to(self, next_status: TaskStatus) -> None:
        """Apply a permitted lifecycle transition or reject it without mutation."""
        next_status = TaskStatus(next_status)
        allowed = TASK_STATUS_TRANSITIONS[self.status]
        if next_status not in allowed:
            raise ValueError(f"Invalid task transition: {self.status.value} -> {next_status.value}.")
        self.status = next_status
        self.updated_at = time.time()

class TaskEvent(BaseModel):
    event_type: str # "plan_created", "step_started", "step_completed", "step_failed", "evaluation", "task_completed"
    task_id: str
    timestamp: float = Field(default_factory=time.time)
    step_id: str | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)

