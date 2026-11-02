import time
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

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
    retry_count: int = 0
    max_retries: int = 2
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
    task_id: str
    objective: str
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] = "hybrid"
    status: TaskStatus = TaskStatus.PENDING
    plan: ExecutionPlan | None = None
    step_results: dict[str, Any] = Field(default_factory=dict)
    final_deliverable: dict[str, Any] | None = None
    artifacts: list[str] = Field(default_factory=list)
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    error_summary: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

class TaskEvent(BaseModel):
    event_type: str # "plan_created", "step_started", "step_completed", "step_failed", "evaluation", "task_completed"
    task_id: str
    timestamp: float = Field(default_factory=time.time)
    step_id: str | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)

