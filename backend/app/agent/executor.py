import asyncio
import re
import time
from pathlib import Path
from typing import Any
from app.agent.state import ExecutionStep, StepStatus, TaskState, TaskEvent, TaskStatus
from app.agent.evaluator import evaluator
from app.tools.registry import tool_registry
from app.tools.schemas import ToolResult
from app.core.logging import logger
from app.core.config import settings

class PlanExecutor:
    def __init__(self):
        pass

    @staticmethod
    def _is_retryable_failure(error: str | None) -> bool:
        if not error:
            return False
        lowered = error.lower()
        transient_markers = (
            "timed out", "temporarily unavailable", "temporary failure",
            "connection reset", "connection refused", "rate limit", "http 429", "http 5",
        )
        return any(marker in lowered for marker in transient_markers)

    async def execute_step(
        self,
        step: ExecutionStep,
        task_state: TaskState,
        event_queue: asyncio.Queue | None = None
    ) -> ToolResult:
        if task_state.status == TaskStatus.CANCEL_REQUESTED:
            step.status = StepStatus.SKIPPED
            return ToolResult(tool_name=step.tool_name, success=False, error="Tool execution cancelled before start.")
        step.status = StepStatus.IN_PROGRESS
        start_time = time.perf_counter()
        
        if event_queue:
            await event_queue.put(TaskEvent(
                event_type="step_started",
                task_id=task_state.task_id,
                step_id=step.step_id,
                message=f"Executing: {step.title} [{step.tool_name}]",
                data={"parameters": step.parameters}
            ))

        # Dynamic parameter resolution from prior step results
        resolved_params = self._resolve_step_parameters(step, task_state)
        
        retry = True
        result: ToolResult | None = None
        
        while retry:
            logger.info("Running planned tool step.")
            result = await tool_registry.execute_tool(
                name=step.tool_name,
                arguments=resolved_params,
                user_approval_granted=True, # The submitted task authorizes its validated plan.
                cancellation_check=lambda: task_state.status == TaskStatus.CANCEL_REQUESTED,
            )

            if result.error == "Tool execution cancelled before start.":
                step.status = StepStatus.SKIPPED
                retry = False
                break
            
            task_state.metrics.tool_calls_count += 1
            
            # Dedicated step evaluation
            eval_res = evaluator.evaluate_step(step, result, task_state)
            
            if eval_res.is_valid:
                step.status = StepStatus.COMPLETED
                step.output = result.data
                step.duration_ms = (time.perf_counter() - start_time) * 1000.0
                
                # Register artifacts
                if result.artifacts:
                    task_state.artifacts.extend(result.artifacts)
                    
                task_state.step_results[step.step_id] = {
                    "tool": step.tool_name,
                    "data": result.data,
                    "artifacts": result.artifacts
                }
                
                if event_queue:
                    await event_queue.put(TaskEvent(
                        event_type="step_completed",
                        task_id=task_state.task_id,
                        step_id=step.step_id,
                        message=f"Completed: {step.title}",
                        data={"output": result.data, "duration_ms": step.duration_ms}
                    ))
                retry = False
            else:
                tool_definition = tool_registry.get_tool_definition(step.tool_name)
                if (
                    eval_res.requires_retry
                    and task_state.status != TaskStatus.CANCEL_REQUESTED
                    and tool_definition is not None
                    and tool_definition.retry_safe
                    and (eval_res.suggested_parameter_correction is not None or self._is_retryable_failure(result.error))
                    and step.retry_count < step.max_retries
                ):
                    step.retry_count += 1
                    task_state.metrics.retries_count += 1
                    if task_state.status == TaskStatus.RUNNING:
                        task_state.transition_to(TaskStatus.RETRYING)
                    logger.warning("Retrying failed tool step.")
                    
                    if eval_res.suggested_parameter_correction:
                        resolved_params = eval_res.suggested_parameter_correction
                        
                    if event_queue:
                        await event_queue.put(TaskEvent(
                            event_type="step_retry",
                            task_id=task_state.task_id,
                            step_id=step.step_id,
                            message=f"Recoverable issue in {step.title}. {eval_res.reason}",
                            data={"retry_count": step.retry_count}
                        ))
                    await asyncio.sleep(0.5)
                    if task_state.status == TaskStatus.RETRYING:
                        task_state.transition_to(TaskStatus.RUNNING)
                else:
                    step.status = StepStatus.FAILED
                    step.error = eval_res.reason
                    step.duration_ms = (time.perf_counter() - start_time) * 1000.0
                    task_state.error_summary = f"Step '{step.title}' failed: {eval_res.reason}"
                    
                    if event_queue:
                        await event_queue.put(TaskEvent(
                            event_type="step_failed",
                            task_id=task_state.task_id,
                            step_id=step.step_id,
                            message=f"Failed: {step.title}. {eval_res.reason}",
                            data={"error": eval_res.reason}
                        ))
                    retry = False

        return result

    def _resolve_step_parameters(self, step: ExecutionStep, task_state: TaskState) -> dict[str, Any]:
        """
        Dynamically propagate dependency outputs into downstream step inputs.
        E.g. If generate_report has empty chart_paths, inject artifacts produced by previous generate_chart steps.
        """
        params = dict(step.parameters)
        
        # Inject only charts already produced in this task. Their order matches
        # the task-scoped chart index used by the artifact routes.
        if step.tool_name == "generate_report":
            chart_root = settings.CHARTS_DIR.resolve()
            chart_artifacts = []
            for artifact in task_state.artifacts:
                candidate = Path(artifact).resolve()
                if (
                    candidate.parent == chart_root
                    and re.fullmatch(r"chart_[a-f0-9]{10}\.png\.enc", candidate.name)
                    and candidate.is_file()
                ):
                    chart_artifacts.append(str(candidate))
            params["chart_paths"] = chart_artifacts
            params["task_id"] = task_state.task_id

            # If findings are generic, enhance with quantitative figures from analytics steps
            for dep_id in step.dependencies:
                dep_res = task_state.step_results.get(dep_id, {})
                data = dep_res.get("data", {})
                if "underperforming_groups" in data and data["underperforming_groups"]:
                    under = data["underperforming_groups"]
                    params.setdefault("findings", []).append(
                        f"Underperforming segments identified below 5% contribution threshold: {', '.join(under)}."
                    )
                if "total_metric_value" in data:
                    params.setdefault("findings", []).append(
                        f"Total aggregate {data.get('metric_column', 'metric')} evaluated: {data['total_metric_value']:,.2f}."
                    )
                    
        return params

executor = PlanExecutor()

