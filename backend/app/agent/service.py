"""Lifecycle-managed workers for the SQLite-backed persistent task queue."""
import asyncio
import uuid

from app.agent.checkpoint import checkpointer
from app.agent.orchestrator import AgentOrchestrator, orchestrator
from app.agent.state import TaskEvent, TaskState
from app.core.config import settings
from app.core.logging import logger
from app.privacy.encryption import encryption_service


class TaskExecutionService:
    def __init__(self, task_orchestrator: AgentOrchestrator = orchestrator):
        self.orchestrator = task_orchestrator
        self._running = False
        self._wake: asyncio.Event | None = None
        self._workers: list[asyncio.Task] = []
        self._payloads: dict[str, TaskState] = {}
        # Keep a committed queue row invisible to workers until its process-local
        # executable payload and initial event have both been published.
        self._queue_lock = asyncio.Lock()

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
            _, pending = await asyncio.wait(
                self._workers,
                timeout=settings.WORKER_SHUTDOWN_GRACE_SECONDS,
            )
            if pending:
                logger.warning("Workers exceeded the shutdown grace period; waiting for safe tool boundaries.")
                await asyncio.gather(*pending, return_exceptions=True)
            completed = [worker for worker in self._workers if worker not in pending]
            if completed:
                await asyncio.gather(*completed, return_exceptions=True)
        self._workers.clear()
        logger.info("AURA task workers stopped.")

    async def enqueue(self, task: TaskState) -> None:
        async with self._queue_lock:
            await asyncio.to_thread(checkpointer.enqueue_task, task)
            self._payloads[task.task_id] = task.model_copy(deep=True)
            try:
                await asyncio.to_thread(checkpointer.append_event, TaskEvent(
                    event_type="task.created",
                    task_id=task.task_id,
                    message="Task accepted.",
                    data={"status": "queued"},
                ))
            except Exception:
                self._payloads.pop(task.task_id, None)
                await asyncio.to_thread(checkpointer.discard_unclaimed_queued_task, task.task_id)
                raise
        if self._wake is not None:
            self._wake.set()

    async def cancel(self, task_id: str) -> str | None:
        current = self.orchestrator.get_task_state(task_id)
        was_queued = current is not None and current.status.value == "queued"
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
            try:
                async with self._queue_lock:
                    task = await asyncio.to_thread(checkpointer.claim_next_queued_task, worker_id)
                    payload = self._payloads.pop(task.task_id, None) if task is not None else None
            except Exception:
                logger.error("Task worker could not claim queued work.")
                task = None

            if task is not None:
                if payload is None:
                    task.recovery_required = True
                    task.recovery_reason = "Queued task input was not available in this process."
                    task.transition_to(type(task.status).FAILED)
                    task.error_summary = "Execution could not be safely resumed; submit the task again."
                    await asyncio.to_thread(checkpointer.save_checkpoint, task)
                    await asyncio.to_thread(checkpointer.append_event, TaskEvent(
                        event_type="task_failed", task_id=task.task_id,
                        message="Task could not be resumed after restart.",
                        data={"status": "failed", "recovery_required": True},
                    ))
                    await asyncio.to_thread(checkpointer.release_task_claim, task.task_id, worker_id)
                    continue
                try:
                    await self.orchestrator.run_task(
                        objective=payload.objective,
                        file_path=payload.file_path,
                        privacy_mode=payload.privacy_mode,
                        task_id=payload.task_id,
                        initial_state=payload,
                    )
                except Exception:
                    logger.error("Task worker stopped after an execution error.")
                    failed_state = self.orchestrator.get_task_state(task.task_id) or task
                    try:
                        if failed_state.status.value not in {"completed", "failed", "cancelled"}:
                            failed_state.transition_to(type(failed_state.status).FAILED)
                        failed_state.error_summary = "Task worker execution failed."
                        await asyncio.to_thread(checkpointer.save_checkpoint, failed_state)
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
