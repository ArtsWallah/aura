import asyncio
import time
import uuid
from typing import AsyncGenerator
from app.agent.state import ExecutionPlan, TaskState, TaskStatus, StepStatus, TaskEvent, TaskMetrics
from app.agent.planner import TaskPlanner
from app.agent.executor import executor
from app.agent.evaluator import evaluator
from app.agent.checkpoint import checkpointer
from app.agent.execution_handoff import ExecutionHandoff, ExecutionHandoffError, HandoffFailureCode
from app.agent.intelligence_contracts import (
    EvaluationResult,
    EvaluationStatus,
    ExecutionTrace,
    IntentAssessment,
    IssueCode,
    StepFailureCategory,
    ValidationOutcome,
)
from app.agent.intelligence_evaluator import (
    EvaluationInput,
    IntelligenceEvaluator,
    PlanStepSummary,
    PlanValidationEvidence,
    StepOutcomeEvidence,
    evidence_result_kind,
    intelligence_evaluator,
)
from app.agent.execution_trace import build_handoff_trace, update_trace_step
from app.llm.router import llm_router
from app.core.config import settings
from app.core.logging import logger
from app.privacy.outbound import privacy_scope
from app.tools.registry import tool_registry

class AgentOrchestrator:
    def __init__(self, execution_evaluator: IntelligenceEvaluator | None = None):
        self._active_tasks: dict[str, TaskState] = {}
        self._event_queues: dict[str, list[asyncio.Queue[tuple[int, TaskEvent]]]] = {}
        self._cancel_signals: dict[str, asyncio.Event] = {}
        self.execution_evaluator = execution_evaluator or intelligence_evaluator

    def request_cancel(self, task_id: str) -> str | None:
        """Request cancellation; active sync tools finish before cancellation is final."""
        state = self.get_task_state(task_id)
        if state is None:
            return None
        if state.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            return state.status.value
        if state.status == TaskStatus.QUEUED:
            state.transition_to(TaskStatus.CANCELLED)
            checkpointer.save_checkpoint(state)
            return state.status.value
        if state.status != TaskStatus.CANCEL_REQUESTED:
            state.transition_to(TaskStatus.CANCEL_REQUESTED)
            checkpointer.save_checkpoint(state)
        signal = self._cancel_signals.get(task_id)
        if signal:
            signal.set()
        return state.status.value

    def get_task_state(self, task_id: str) -> TaskState | None:
        if task_id in self._active_tasks:
            return self._active_tasks[task_id]
        # Checkpoint loading restores only the encrypted, validated artifact
        # manifest alongside the minimal lifecycle state.
        return checkpointer.load_checkpoint(task_id)

    def register_event_listener(self, task_id: str) -> asyncio.Queue[tuple[int, TaskEvent]]:
        q = asyncio.Queue()
        self._event_queues.setdefault(task_id, []).append(q)
        return q

    def remove_event_listener(self, task_id: str, q: asyncio.Queue[tuple[int, TaskEvent]]):
        if task_id in self._event_queues and q in self._event_queues[task_id]:
            self._event_queues[task_id].remove(q)

    async def _emit_event(self, event: TaskEvent):
        allowed = {
            "task_started", "plan_created", "step_started", "step_completed",
            "step_failed", "step_retry", "task_completed", "task_failed", "task_cancelled",
        }
        event_type = event.event_type if event.event_type in allowed else "task_update"
        state = self._active_tasks.get(event.task_id)
        step = next((item for item in state.plan.steps if item.step_id == event.step_id), None) if state and state.plan and event.step_id else None
        ordinal = next((index + 1 for index, item in enumerate(state.plan.steps) if item.step_id == event.step_id), None) if state and state.plan and event.step_id else None
        messages = {
            "task_started": "Task execution started.",
            "plan_created": "A validated plan is ready.",
            "step_started": "A tool step started.",
            "step_completed": "A tool step completed.",
            "step_failed": "A tool step failed.",
            "step_retry": "A retry-safe step will be retried.",
            "task_completed": "Task completed.",
            "task_failed": "Task failed.",
            "task_cancelled": "Task cancelled.",
            "task_update": "Task status updated.",
        }
        safe_data: dict[str, object] = {}
        if event_type == "plan_created" and state and state.plan:
            safe_data = {"step_count": len(state.plan.steps)}
        elif step is not None:
            safe_data = {
                "step_index": ordinal,
                "tool_name": step.tool_name,
                "status": step.status.value,
            }
            if event_type == "step_completed":
                safe_data["duration_ms"] = max(0, int(step.duration_ms))
            if event_type == "step_retry":
                safe_data["retry_count"] = max(0, int(step.retry_count))
            if event_type == "step_failed":
                safe_data["error_category"] = "tool_failure"
        if state and isinstance(state.execution_trace, ExecutionTrace):
            if event_type in {"step_started", "step_retry", "step_completed", "step_failed"} and step is not None:
                updated_trace = update_trace_step(
                    state.execution_trace,
                    step,
                    state,
                    event_type,
                    retry_count=event.data.get("retry_count") if isinstance(event.data.get("retry_count"), int) else None,
                    artifact_manifest=checkpointer.artifact_manifest_for_task(state),
                )
                if updated_trace is not None:
                    state.set_execution_trace(updated_trace)
                    safe_data["trace_step"] = next(
                        item.model_dump(mode="json")
                        for item in updated_trace.step_statuses
                        if item.step_id == step.step_id
                    )
            if event_type == "plan_created" or event_type in {"task_completed", "task_failed", "task_cancelled"}:
                safe_data["trace_snapshot"] = state.execution_trace.model_dump(mode="json")
        safe_event = TaskEvent(
            event_type=event_type,
            task_id=event.task_id,
            timestamp=event.timestamp,
            step_id=event.step_id if step is not None else None,
            message=messages[event_type],
            data=safe_data,
        )
        sequence = await asyncio.to_thread(checkpointer.append_event, safe_event)
        queues = self._event_queues.get(event.task_id, [])
        for q in queues:
            await q.put((sequence, safe_event))

    async def run_task(
        self,
        objective: str,
        file_path: str | None = None,
        privacy_mode: str = "hybrid",
        task_id: str | None = None,
        initial_state: TaskState | None = None,
    ) -> TaskState:
        if settings.ENVIRONMENT != "test":
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
        with privacy_scope(privacy_mode):
            return await self._run_task(objective, file_path, privacy_mode, task_id, initial_state)

    async def run_handoff(
        self,
        handoff: ExecutionHandoff,
        *,
        task_id: str | None = None,
        privacy_mode: str | None = None,
        initial_state: TaskState | None = None,
    ) -> TaskState:
        """Execute a pipeline handoff after integrity and live-policy checks."""
        if not isinstance(handoff, ExecutionHandoff):
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
        mode = handoff.privacy_mode if privacy_mode is None else privacy_mode
        plan = handoff.validate_integrity(privacy_mode=mode, registry=tool_registry)
        if initial_state is not None and (
            initial_state.status != TaskStatus.QUEUED
            or initial_state.privacy_mode != mode
            or (task_id is not None and initial_state.task_id != task_id)
        ):
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
        if task_id is not None:
            try:
                queued_state = checkpointer.load_checkpoint(task_id)
                if task_id in self._active_tasks:
                    raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
                if initial_state is None and queued_state is not None:
                    raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
                if initial_state is not None and (
                    queued_state is None or queued_state.status != TaskStatus.QUEUED
                ):
                    raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
            except ExecutionHandoffError:
                raise
            except Exception:
                raise ExecutionHandoffError(HandoffFailureCode.REGISTRY_UNAVAILABLE) from None
        with privacy_scope(mode):
            state = await self._run_task(
                objective="",
                file_path=None,
                privacy_mode=mode,
                task_id=task_id,
                initial_state=initial_state,
                supplied_plan=plan,
                handoff_context=handoff,
            )
        if state.execution_evaluation is None:
            self._finalize_handoff_trace(handoff, state)
        return state

    def _finalize_handoff_trace(self, handoff: ExecutionHandoff, state: TaskState) -> None:
        state.set_execution_evaluation(self._evaluate_handoff_outcome(handoff, state))
        trace = build_handoff_trace(
            handoff,
            state,
            artifact_manifest=checkpointer.artifact_manifest_for_task(state),
        )
        if trace is not None:
            state.set_execution_trace(trace)

    def _evaluate_handoff_outcome(
        self,
        handoff: ExecutionHandoff,
        state: TaskState,
    ) -> EvaluationResult:
        """Build bounded evidence from final steps; raw results never enter the evaluator."""
        try:
            plan = state.plan
            if plan is None or not plan.steps or state.recovery_required:
                raise ValueError("Execution evidence is unavailable.")
            if state.status == TaskStatus.CANCELLED and all(
                step.status == StepStatus.COMPLETED for step in plan.steps
            ):
                raise ValueError("Cancellation makes objective completion inconclusive.")
            summaries = [
                PlanStepSummary(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    dependencies=list(step.dependencies),
                )
                for step in plan.steps
            ]
            outcomes = []
            for step in plan.steps:
                result_kind = evidence_result_kind(step.tool_name)
                result_count = None
                artifact_count = None
                if step.status == StepStatus.COMPLETED and result_kind is not None:
                    # One is an availability marker, not a row count or payload.
                    result_count = 1 if bool(step.output) else 0
                    result_record = state.step_results.get(step.step_id)
                    artifacts = result_record.get("artifacts") if isinstance(result_record, dict) else None
                    if step.tool_name in {"generate_chart", "generate_report"}:
                        artifact_count = min(len(artifacts), 10_000) if isinstance(artifacts, list) else 0
                if step.status == StepStatus.FAILED:
                    failure_category = StepFailureCategory.TOOL_FAILURE
                elif step.status == StepStatus.SKIPPED:
                    if state.status == TaskStatus.CANCELLED:
                        failure_category = StepFailureCategory.CANCELLED
                    elif any(
                        next((dependency.status for dependency in plan.steps if dependency.step_id == dep), None)
                        != StepStatus.COMPLETED
                        for dep in step.dependencies
                    ):
                        failure_category = StepFailureCategory.DEPENDENCY_BLOCKED
                    else:
                        failure_category = StepFailureCategory.UNKNOWN
                elif step.status == StepStatus.INTERRUPTED or step.status in {StepStatus.PENDING, StepStatus.IN_PROGRESS}:
                    failure_category = StepFailureCategory.INTERRUPTED
                else:
                    failure_category = StepFailureCategory.NONE
                outcomes.append(StepOutcomeEvidence(
                    step_id=step.step_id,
                    status=step.status,
                    result_kind=result_kind,
                    result_count=result_count,
                    artifact_count=artifact_count,
                    retry_count=min(step.retry_count, 10_000),
                    retry_limit=min(step.max_retries, 10_000),
                    failure_category=failure_category,
                ))

            intent = IntentAssessment(
                category=handoff.intent_category,
                confidence=handoff.intent_confidence,
                required_tool_groups=handoff.required_tool_groups,
                is_multi_step=handoff.intent_is_multi_step,
            )
            evidence = EvaluationInput(
                intent=intent,
                steps=summaries,
                validation=PlanValidationEvidence(outcome=ValidationOutcome.ACCEPTED),
                evaluation_target="execution",
                outcomes=outcomes,
                privacy_mode=handoff.privacy_mode,
            )
            return self.execution_evaluator.evaluate(evidence)
        except Exception:
            return EvaluationResult(
                execution_valid=False,
                objective_coverage=0.0,
                evaluation_status=EvaluationStatus.INCONCLUSIVE,
                issues=[IssueCode.MISSING_EVIDENCE],
            )

    async def _run_task(
        self,
        objective: str,
        file_path: str | None = None,
        privacy_mode: str = "hybrid",
        task_id: str | None = None,
        initial_state: TaskState | None = None,
        supplied_plan: ExecutionPlan | None = None,
        handoff_context: ExecutionHandoff | None = None,
    ) -> TaskState:
        if settings.ENVIRONMENT != "test":
            if not isinstance(handoff_context, ExecutionHandoff) or supplied_plan is None:
                raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
            validated_plan = handoff_context.validate_integrity(
                privacy_mode=privacy_mode,
                registry=tool_registry,
            )
            if supplied_plan != validated_plan:
                raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
        t_id = task_id or f"task_{uuid.uuid4().hex[:10]}"
        queued_state = checkpointer.load_checkpoint(t_id) if initial_state is not None else None
        if initial_state is not None and queued_state is None:
            raise RuntimeError("Queued task state could not be decrypted; refusing to overwrite it.")
        if queued_state and queued_state.status == TaskStatus.CANCELLED:
            return queued_state
        if queued_state and queued_state.status == TaskStatus.QUEUED:
            state = initial_state.model_copy(deep=True)
            state.transition_to(TaskStatus.PLANNING)
        else:
            state = TaskState(
                task_id=t_id,
                objective=objective,
                file_path=file_path,
                privacy_mode=privacy_mode,
                status=TaskStatus.PLANNING
            )
        self._active_tasks[t_id] = state
        cancel_signal = asyncio.Event()
        self._cancel_signals[t_id] = cancel_signal
        checkpointer.save_checkpoint(state)
        
        start_time = time.perf_counter()

        await self._emit_event(TaskEvent(
            event_type="task_started",
            task_id=t_id,
            message=f"Starting task execution under privacy mode: '{privacy_mode}'.",
            data={"objective": objective, "privacy_mode": privacy_mode}
        ))

        try:
            # 1. Routing & Planning
            planner = None
            if supplied_plan is None:
                provider = llm_router.get_provider(privacy_mode=privacy_mode)
                planner = TaskPlanner(provider)
                plan = await planner.create_plan(objective, file_path, privacy_mode=privacy_mode)
            else:
                plan = supplied_plan
            state.plan = plan
            if handoff_context is not None:
                initial_trace = build_handoff_trace(
                    handoff_context,
                    state,
                    artifact_manifest=checkpointer.artifact_manifest_for_task(state),
                )
                if initial_trace is not None:
                    state.set_execution_trace(initial_trace)
            if cancel_signal.is_set():
                if handoff_context is not None:
                    self._finalize_handoff_trace(handoff_context, state)
                state.transition_to(TaskStatus.CANCELLED)
                checkpointer.save_checkpoint(state)
                await self._emit_event(TaskEvent(event_type="task_cancelled", task_id=t_id, message="cancelled"))
                return state
            state.transition_to(TaskStatus.RUNNING)
            if planner is not None:
                state.metrics.llm_calls_count += planner.provider_calls
            checkpointer.save_checkpoint(state)

            await self._emit_event(TaskEvent(
                event_type="plan_created",
                task_id=t_id,
                message=f"Plan synthesized with {len(plan.steps)} steps.",
                data={"steps": [s.model_dump() for s in plan.steps]}
            ))

            # 2. Dependency-aware Execution Loop (DAG)
            queue_for_executor = asyncio.Queue()
            
            # Start background forwarder for executor events
            async def forward_events():
                while True:
                    ev = await queue_for_executor.get()
                    await self._emit_event(ev)
                    queue_for_executor.task_done()
                    
            forward_task = asyncio.create_task(forward_events())

            completed_step_ids: set[str] = set()
            total_steps = len(plan.steps)

            while len(completed_step_ids) < total_steps:
                if cancel_signal.is_set():
                    for pending_step in plan.steps:
                        if pending_step.status == StepStatus.PENDING:
                            pending_step.status = StepStatus.SKIPPED
                            pending_step.error = "Skipped after cancellation was requested."
                    if state.status == TaskStatus.CANCEL_REQUESTED:
                        state.transition_to(TaskStatus.CANCELLED)
                    checkpointer.save_checkpoint(state)
                    break
                # Find all pending steps whose dependencies are satisfied
                ready_steps = [
                    s for s in plan.steps
                    if s.status == StepStatus.PENDING and all(dep in completed_step_ids for dep in s.dependencies)
                ]

                if not ready_steps:
                    # Check if any steps failed or are stuck
                    failed = [s for s in plan.steps if s.status == StepStatus.FAILED]
                    if failed:
                        state.transition_to(TaskStatus.FAILED)
                        state.error_summary = f"Execution halted due to failed steps: {[s.step_id for s in failed]}"
                        break
                    else:
                        break # No more actionable steps

                # Execute ready steps in parallel when more than 1 are independent
                logger.info(f"Executing a batch of {len(ready_steps)} tool step(s).")
                batch_tasks = [
                    executor.execute_step(step, state, queue_for_executor)
                    for step in ready_steps
                ]
                
                await asyncio.gather(*batch_tasks)

                for step in ready_steps:
                    if step.status == StepStatus.COMPLETED:
                        completed_step_ids.add(step.step_id)
                    elif step.status == StepStatus.FAILED and not cancel_signal.is_set():
                        state.transition_to(TaskStatus.FAILED)
                        break

                checkpointer.save_checkpoint(state)
                if state.status == TaskStatus.FAILED:
                    blocked = {s.step_id for s in plan.steps if s.status == StepStatus.FAILED}
                    changed = True
                    while changed:
                        changed = False
                        for pending_step in plan.steps:
                            if pending_step.status == StepStatus.PENDING and any(dep in blocked for dep in pending_step.dependencies):
                                pending_step.status = StepStatus.SKIPPED
                                pending_step.error = "Skipped because a prerequisite failed."
                                blocked.add(pending_step.step_id)
                                changed = True
                    checkpointer.save_checkpoint(state)
                    break

            await queue_for_executor.join()
            forward_task.cancel()
            await asyncio.gather(forward_task, return_exceptions=True)

            if state.status == TaskStatus.CANCELLED:
                if handoff_context is not None:
                    self._finalize_handoff_trace(handoff_context, state)
                await self._emit_event(TaskEvent(event_type="task_cancelled", task_id=t_id, message="cancelled"))
                self._cancel_signals.pop(t_id, None)
                return state

            # 3. Final Task Evaluation
            is_successful, eval_reason = evaluator.evaluate_task_completion(state)
            state.updated_at = time.time()
            state.metrics.total_execution_time_ms = (time.perf_counter() - start_time) * 1000.0

            if is_successful:
                state.transition_to(TaskStatus.COMPLETED)
                
                # Extract final deliverable
                report_step = next((s for s in reversed(plan.steps) if s.tool_name == "generate_report" and s.status == StepStatus.COMPLETED), None)
                if report_step and report_step.output:
                    state.final_deliverable = report_step.output
                else:
                    # Aggregate last output
                    last_step = next((s for s in reversed(plan.steps) if s.status == StepStatus.COMPLETED), None)
                    state.final_deliverable = last_step.output if last_step else {"message": "Execution finished."}

                if handoff_context is not None:
                    self._finalize_handoff_trace(handoff_context, state)

                await self._emit_event(TaskEvent(
                    event_type="task_completed",
                    task_id=t_id,
                    message="Task completed successfully.",
                    data={"deliverable": state.final_deliverable, "metrics": state.metrics.model_dump()}
                ))
            else:
                if state.status != TaskStatus.FAILED:
                    state.transition_to(TaskStatus.FAILED)
                state.error_summary = eval_reason
                if handoff_context is not None:
                    self._finalize_handoff_trace(handoff_context, state)
                await self._emit_event(TaskEvent(
                    event_type="task_failed",
                    task_id=t_id,
                    message=f"Task could not be verified: {eval_reason}",
                    data={"error": eval_reason}
                ))

        except Exception as e:
            was_cancel_requested = state.status == TaskStatus.CANCEL_REQUESTED
            if was_cancel_requested:
                state.transition_to(TaskStatus.CANCELLED)
            elif state.status not in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
                state.transition_to(TaskStatus.FAILED)
            state.error_summary = "Internal task error."
            state.updated_at = time.time()
            state.metrics.total_execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("Task execution failed with an internal error.")
            if handoff_context is not None:
                self._finalize_handoff_trace(handoff_context, state)
            await self._emit_event(TaskEvent(
                event_type="task_cancelled" if was_cancel_requested else "task_failed",
                task_id=t_id,
                message="Task cancelled at a safe execution boundary." if was_cancel_requested else "Task execution failed.",
                data={"status": "cancelled" if was_cancel_requested else "failed"}
            ))

        if cancel_signal.is_set() and state.status == TaskStatus.CANCEL_REQUESTED:
            for pending_step in state.plan.steps if state.plan else []:
                if pending_step.status == StepStatus.PENDING:
                    pending_step.status = StepStatus.SKIPPED
                    pending_step.error = "Skipped after cancellation was requested."
            state.transition_to(TaskStatus.CANCELLED)
        checkpointer.save_checkpoint(state)
        self._cancel_signals.pop(t_id, None)
        return state

orchestrator = AgentOrchestrator()

