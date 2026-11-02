from typing import Any
from pydantic import BaseModel, Field

from app.agent.state import ExecutionStep, StepStatus, TaskState
from app.tools.schemas import ToolResult
from app.core.logging import logger

class EvaluationResult(BaseModel):
    is_valid: bool
    requires_retry: bool = False
    suggested_parameter_correction: dict[str, Any] | None = None
    reason: str
    confidence: float = 1.0

class StepEvaluator:
    """
    Dedicated evaluator that validates intermediate tool execution outputs,
    detects recoverable anomalies, and advises on parameter corrections or retries.
    """
    def evaluate_step(
        self,
        step: ExecutionStep,
        tool_result: ToolResult,
        task_state: TaskState
    ) -> EvaluationResult:
        # 1. Did the tool fail hard?
        if not tool_result.success:
            err = tool_result.error or "Unknown tool execution failure."
            logger.warning(f"Step '{step.step_id}' failed: {err}")
            
            # Check if retry limit exceeded
            if step.retry_count >= step.max_retries:
                return EvaluationResult(
                    is_valid=False,
                    requires_retry=False,
                    reason=f"Exceeded maximum retry limit ({step.max_retries}). Error: {err}"
                )

            # Heuristic self-corrections
            correction = None
            if "file not found" in err.lower():
                # Check if file path is available in task context or uploaded files
                suggested_file = task_state.step_results.get("uploaded_file")
                if suggested_file:
                    correction = dict(step.parameters)
                    correction["file_path"] = suggested_file
            elif "column" in err.lower() and "does not exist" in err.lower():
                # Check inspect_dataset output in previous steps to auto-correct column name
                for prev_step_id in step.dependencies:
                    prev_res = task_state.step_results.get(prev_step_id, {})
                    data = prev_res.get("data", {})
                    columns = data.get("columns", [])
                    if columns:
                        # Pick closest match or default numeric/categorical column
                        correction = dict(step.parameters)
                        if "group_by" in correction and columns:
                            cats = data.get("categorical_columns", [])
                            if cats:
                                correction["group_by"] = cats[0]
                        if "metric_column" in correction:
                            nums = data.get("numeric_columns", [])
                            if nums:
                                correction["metric_column"] = nums[0]

            return EvaluationResult(
                is_valid=False,
                requires_retry=True,
                suggested_parameter_correction=correction,
                reason=f"Tool error: {err}. Attempting recovery (retry #{step.retry_count + 1})."
            )

        # 2. Check for empty or malformed data payload
        data = tool_result.data
        if data is None or (isinstance(data, dict) and not data):
            return EvaluationResult(
                is_valid=False,
                requires_retry=False,
                reason="Tool returned an empty payload."
            )

        # 3. Step-specific schema and utility validations
        if step.tool_name == "inspect_dataset":
            if data.get("total_rows", 0) == 0:
                return EvaluationResult(
                    is_valid=False,
                    requires_retry=False,
                    reason="Dataset has 0 rows; cannot proceed with downstream calculations."
                )

        elif step.tool_name == "generate_chart":
            artifacts = tool_result.artifacts or data.get("artifacts", [])
            if not artifacts:
                return EvaluationResult(
                    is_valid=False,
                    requires_retry=True,
                    reason="Visualization tool succeeded but produced no chart image artifact."
                )

        return EvaluationResult(
            is_valid=True,
            requires_retry=False,
            reason="Step execution verified successfully against schema and utility standards.",
            confidence=1.0
        )

    def evaluate_task_completion(self, task_state: TaskState) -> tuple[bool, str]:
        """
        Final evaluation of the entire task DAG before concluding.
        """
        if not task_state.plan or not task_state.plan.steps:
            return False, "Task has no execution plan."

        failed_steps = [s.step_id for s in task_state.plan.steps if s.status == StepStatus.FAILED]
        if failed_steps:
            return False, f"Cannot complete task: steps {failed_steps} failed."

        incomplete = [s.step_id for s in task_state.plan.steps if s.status != StepStatus.COMPLETED]
        if incomplete:
            return False, f"Pending steps remain: {incomplete}."

        return True, "All planned steps completed and verified."

evaluator = StepEvaluator()
