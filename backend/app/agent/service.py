"""Lifecycle-managed workers for the SQLite-backed persistent task queue."""
import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agent.checkpoint import checkpointer
from app.agent.execution_handoff import ExecutionHandoff, ExecutionHandoffError, HandoffFailureCode
from app.agent.execution_trace import trace_construction_failure
from app.agent.orchestrator import AgentOrchestrator, orchestrator
from app.agent.state import TaskEvent, TaskState, TaskStatus
from app.core.config import settings, test_harness_enabled
from app.core.logging import logger
from app.privacy.encryption import encryption_service
from app.tools.registry import tool_registry


@dataclass(frozen=True)
class ProcessLocalTaskPayload:
    """Executable request data retained only in this process."""

    task_state: TaskState
    handoff: ExecutionHandoff | None
    handoff_required: bool


class TaskExecutionService:
    def __init__(self, task_orchestrator: AgentOrchestrator = orchestrator):
        self.orchestrator = task_orchestrator
        self._running = False
        self._wake: asyncio.Event | None = None
        self._workers: list[asyncio.Task] = []
        self._payloads: dict[str, ProcessLocalTaskPayload] = {}
        # Keep a committed queue row invisible to workers until its process-local
        # executable payload and initial event have both been published.
        self._queue_lock = asyncio.Lock()

    @staticmethod
    async def _run_blocking_to_completion(
        operation: Callable[..., Any], *args: Any
    ) -> tuple[Any, Exception | None, bool]:
        """Drain a database thread even if its awaiting coroutine is cancelled."""
        work = asyncio.create_task(asyncio.to_thread(operation, *args))
        cancellation_requested = False
        while True:
            try:
                return await asyncio.shield(work), None, cancellation_requested
            except asyncio.CancelledError:
                cancellation_requested = True
                # Do not release queue coordination while the SQLite thread can
                # still commit a row or event.
                continue
            except Exception as exc:
                return None, exc, cancellation_requested

    async def _rollback_enqueue(self, task_id: str) -> tuple[bool, bool]:
        self._payloads.pop(task_id, None)
        rolled_back, error, cancellation_requested = await self._run_blocking_to_completion(
            checkpointer.discard_unclaimed_queued_task, task_id
        )
        if error is not None or not rolled_back:
            logger.error("Task enqueue could not be rolled back safely.")
            return False, cancellation_requested
        return True, cancellation_requested

    async def start(self) -> None:
        if self._running:
            return
        # The queue/checkpoints require authenticated encryption; never start
        # workers that could claim or mutate state with an unavailable key.
        encryption_service()
        # Interrupted operations have ambiguous side effects. Recovery marks them
        # for review and never queues them for automatic replay.
        await asyncio.to_thread(checkpointer.recover_interrupted_tasks, set(self._payloads))
        self._running = True
        self._wake = asyncio.Event()
        self._wake.set()
        self._workers = [
            asyncio.create_task(self._worker_loop(f"worker-{index}-{uuid.uuid4().hex[:8]}"), name=f"aura-worker-{index}")
            for index in range(settings.MAX_CONCURRENT_TASKS)
        ]
        logger.info("AURA task workers started.")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._wake is not None:
            self._wake.set()
        # Workers finish their current task before exiting; queued rows remain durable.
        if self._workers:
            done, pending = await asyncio.wait(
                self._workers,
                timeout=settings.WORKER_SHUTDOWN_GRACE_SECONDS,
            )
            if pending:
                logger.warning("Workers exceeded the shutdown grace period; cancelling pending workers.")
                for worker in pending:
                    worker.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            if done:
                await asyncio.gather(*done, return_exceptions=True)
        self._workers.clear()
        logger.info("AURA task workers stopped.")

    async def enqueue(self, task: TaskState) -> None:
        """Keep legacy queueing available only to the non-production test harness."""
        if not test_harness_enabled():
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
        await self._enqueue_payload(ProcessLocalTaskPayload(
            task_state=task.model_copy(deep=True),
            handoff=None,
            handoff_required=False,
        ))

    async def enqueue_handoff(self, task: TaskState, handoff: ExecutionHandoff) -> None:
        """Queue a validated plan without retaining the raw objective or path."""
        if task.status != TaskStatus.QUEUED:
            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
        handoff.validate_integrity(privacy_mode=task.privacy_mode, registry=tool_registry)
        safe_state = task.model_copy(deep=True)
        safe_state.objective = ""
        safe_state.file_path = None
        await self._enqueue_payload(ProcessLocalTaskPayload(
            task_state=safe_state,
            handoff=handoff.model_copy(deep=True),
            handoff_required=True,
        ))

    async def _enqueue_payload(self, payload: ProcessLocalTaskPayload) -> None:
        task = payload.task_state
        async with self._queue_lock:
            _, persist_error, persist_cancelled = await self._run_blocking_to_completion(
                checkpointer.enqueue_task, task
            )
            if persist_error is not None or persist_cancelled:
                rollback_ok, rollback_cancelled = await self._rollback_enqueue(task.task_id)
                if not rollback_ok:
                    raise RuntimeError("Task enqueue rollback failed; the task was not made available to workers.") from persist_error
                if persist_cancelled or rollback_cancelled:
                    raise asyncio.CancelledError
                if persist_error is not None:
                    raise persist_error

            self._payloads[task.task_id] = payload
            _, event_error, event_cancelled = await self._run_blocking_to_completion(
                checkpointer.append_event,
                TaskEvent(
                    event_type="task.created",
                    task_id=task.task_id,
                    message="Task accepted.",
                    data={"status": "queued"},
                ),
            )
            if event_error is not None or event_cancelled:
                rollback_ok, rollback_cancelled = await self._rollback_enqueue(task.task_id)
                if not rollback_ok:
                    raise RuntimeError("Task enqueue rollback failed; the task was not made available to workers.") from event_error
                if event_cancelled or rollback_cancelled:
                    raise asyncio.CancelledError
                if event_error is not None:
                    raise event_error
        if self._wake is not None:
            self._wake.set()

    async def cancel(self, task_id: str) -> str | None:
        current = self.orchestrator.get_task_state(task_id)
        was_queued = current is not None and current.status == TaskStatus.QUEUED
        status = self.orchestrator.request_cancel(task_id)
        if status == "cancelled":
            self._payloads.pop(task_id, None)
            if was_queued:
                await asyncio.to_thread(checkpointer.append_event, TaskEvent(
                    event_type="task_cancelled",
                    task_id=task_id,
                    message="Task cancelled before execution.",
                    data={"status": "cancelled"},
                ))
        return status

    async def _worker_loop(self, worker_id: str) -> None:
        while self._running:
            task = None
            payload = None
            try:
                async with self._queue_lock:
                    task, claim_error, claim_cancelled = await self._run_blocking_to_completion(
                        checkpointer.claim_next_queued_task, worker_id
                    )
                    if claim_error is not None:
                        raise claim_error
                    if claim_cancelled:
                        if task is not None:
                            _, release_error, _ = await self._run_blocking_to_completion(
                                checkpointer.release_task_claim, task.task_id, worker_id
                            )
                            if release_error is not None:
                                logger.error("Cancelled task worker could not release its queue claim.")
                        raise asyncio.CancelledError
                    payload = self._payloads.pop(task.task_id, None) if task is not None else None
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("Task worker could not claim queued work.")
                if task is not None:
                    await asyncio.to_thread(checkpointer.release_task_claim, task.task_id, worker_id)
                task = None

            if task is not None:
                if payload is None:
                    task.recovery_required = True
                    task.recovery_reason = "Queued task input was not available in this process."
                    if task.status != TaskStatus.FAILED and task.status not in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}:
                        task.transition_to(TaskStatus.FAILED)
                    task.error_summary = "Execution could not be safely resumed; submit the task again."
                    trace = checkpointer._recovered_trace_snapshot(task) or trace_construction_failure()
                    event_type = {
                        TaskStatus.COMPLETED: "task_completed",
                        TaskStatus.CANCELLED: "task_cancelled",
                    }.get(task.status, "task_failed")
                    event = TaskEvent(
                        event_type=event_type, task_id=task.task_id,
                        message="Task could not be resumed after restart.",
                        data={"status": task.status.value, "recovery_required": True, "trace_snapshot": trace.model_dump(mode="json")},
                    )
                    await asyncio.to_thread(checkpointer.save_checkpoint_and_append_event, task, event)
                    await asyncio.to_thread(checkpointer.release_task_claim, task.task_id, worker_id)
                    continue
                if payload.task_state.task_id != task.task_id:
                    task.recovery_required = True
                    task.recovery_reason = "Queued task input did not match its process-local task identifier."
                    if task.status != TaskStatus.FAILED and task.status not in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}:
                        task.transition_to(TaskStatus.FAILED)
                    task.error_summary = "Execution could not be safely resumed; submit the task again."
                    trace = checkpointer._recovered_trace_snapshot(task) or trace_construction_failure()
                    event_type = {
                        TaskStatus.COMPLETED: "task_completed",
                        TaskStatus.CANCELLED: "task_cancelled",
                    }.get(task.status, "task_failed")
                    event = TaskEvent(
                        event_type=event_type, task_id=task.task_id,
                        message="Task could not be resumed after restart.",
                        data={"status": task.status.value, "recovery_required": True, "trace_snapshot": trace.model_dump(mode="json")},
                    )
                    await asyncio.to_thread(checkpointer.save_checkpoint_and_append_event, task, event)
                    await asyncio.to_thread(checkpointer.release_task_claim, task.task_id, worker_id)
                    continue
                try:
                    if payload.handoff_required:
                        if not isinstance(payload.handoff, ExecutionHandoff):
                            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
                        await self.orchestrator.run_handoff(
                            payload.handoff,
                            privacy_mode=payload.task_state.privacy_mode,
                            task_id=task.task_id,
                            initial_state=task,
                        )
                    else:
                        if not test_harness_enabled():
                            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
                        if payload.handoff is not None:
                            raise ExecutionHandoffError(HandoffFailureCode.INVALID_HANDOFF)
                        await self.orchestrator.run_task(
                            objective=payload.task_state.objective,
                            file_path=payload.task_state.file_path,
                            privacy_mode=payload.task_state.privacy_mode,
                            task_id=payload.task_state.task_id,
                            initial_state=payload.task_state,
                        )
                except Exception:
                    logger.error("Task worker stopped after an execution error.")
                    failed_state = self.orchestrator.get_task_state(task.task_id) or task
                    try:
                        if failed_state.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                            if failed_state.status == TaskStatus.CANCEL_REQUESTED:
                                failed_state.transition_to(TaskStatus.CANCELLED)
                                event_type = "task_cancelled"
                                status_str = "cancelled"
                                msg = "Task cancelled at safe boundary."
                            else:
                                failed_state.transition_to(TaskStatus.FAILED)
                                failed_state.error_summary = failed_state.error_summary or "Task worker execution failed."
                                event_type = "task_failed"
                                status_str = "failed"
                                msg = "Task worker execution failed."
                            trace = checkpointer._recovered_trace_snapshot(failed_state) or trace_construction_failure()
                            event = TaskEvent(
                                event_type=event_type,
                                task_id=failed_state.task_id,
                                message=msg,
                                data={"status": status_str, "trace_snapshot": trace.model_dump(mode="json")},
                            )
                            await asyncio.to_thread(checkpointer.save_checkpoint_and_append_event, failed_state, event)
                    except Exception:
                        logger.error("Task worker could not persist the failure state.")
                finally:
                    await asyncio.to_thread(checkpointer.release_task_claim, task.task_id, worker_id)
                continue

            if self._wake is None:
                return
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=settings.WORKER_POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass


task_execution_service = TaskExecutionService()
