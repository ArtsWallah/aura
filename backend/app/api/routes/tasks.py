import json
import asyncio
import uuid
from typing import Literal
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.orchestrator import orchestrator
from app.agent.state import TaskState, TaskStatus, TaskEvent
from app.agent.checkpoint import checkpointer
from app.agent.service import task_execution_service
from app.core.config import settings
from app.core.security import validate_safe_path

router = APIRouter(prefix="/tasks", tags=["Tasks"])

class CreateTaskRequest(BaseModel):
    objective: str = Field(description="High-level natural language instruction or query.")
    file_path: str | None = Field(default=None, description="Path to uploaded dataset/document.")
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] | None = Field(
        default=None,
        description="Privacy execution mode."
    )

@router.post("", response_model=TaskState, status_code=status.HTTP_202_ACCEPTED)
async def create_task(request: CreateTaskRequest):
    task_id = f"task_{uuid.uuid4().hex}"
    if request.file_path:
        try:
            validate_safe_path(request.file_path)
        except PermissionError as exc:
            raise HTTPException(status_code=400, detail="Only managed uploads and the bundled sample dataset may be used.") from exc
    privacy_mode = request.privacy_mode or settings.DEFAULT_PRIVACY_MODE
    state = TaskState(
        task_id=task_id,
        objective=request.objective,
        file_path=request.file_path,
        privacy_mode=privacy_mode,
        status=TaskStatus.QUEUED,
    )
    await task_execution_service.enqueue(state)
    return state

@router.get("", response_model=list[dict])
def list_tasks(limit: int = 25):
    return checkpointer.list_checkpoints(limit=limit)

@router.get("/{task_id}", response_model=TaskState)
def get_task(task_id: str):
    state = orchestrator.get_task_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return state

@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    state = orchestrator.get_task_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    resulting_status = await task_execution_service.cancel(task_id)
    message = "Cancellation requested; active synchronous work will finish before cancellation is confirmed."
    return {"message": message, "status": resulting_status}

@router.get("/{task_id}/events")
async def stream_task_events(task_id: str, last_event_id: str | None = Header(default=None, alias="Last-Event-ID")):
    """
    Server-Sent Events (SSE) streaming live updates for a task.
    """
    state = orchestrator.get_task_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found.")
    try:
        after_sequence = int(last_event_id or 0)
        if after_sequence < 0:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Last-Event-ID must be a non-negative integer.") from exc
    queue = orchestrator.register_event_listener(task_id)

    async def event_generator():
        cursor = after_sequence
        terminal_types = {"task_completed", "task_failed", "task_cancelled"}

        def encode(sequence: int, event: TaskEvent) -> str:
            payload = json.dumps(event.model_dump(), separators=(",", ":"))
            return f"id: {sequence}\nevent: {event.event_type}\ndata: {payload}\n\n"

        try:
            batch = await asyncio.to_thread(checkpointer.read_events, task_id, cursor)
            for sequence, event in batch:
                cursor = sequence
                yield encode(sequence, event)
                if event.event_type in terminal_types:
                    return

            # A created task always has a durable task.created event. This safe
            # snapshot covers pre-migration rows that have no event history.
            if cursor == 0:
                safe_snapshot = TaskEvent(
                    event_type="task.snapshot",
                    task_id=task_id,
                    message="Current task status.",
                    data={"status": state.status.value},
                )
                yield f"event: task.snapshot\ndata: {json.dumps(safe_snapshot.model_dump())}\n\n"

            while True:
                try:
                    await asyncio.wait_for(queue.get(), timeout=20.0)
                    while True:
                        batch = await asyncio.to_thread(checkpointer.read_events, task_id, cursor)
                        if not batch:
                            break
                        for sequence, event in batch:
                            if sequence <= cursor:
                                continue
                            cursor = sequence
                            yield encode(sequence, event)
                            if event.event_type in terminal_types:
                                return
                        if len(batch) < 1000:
                            break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            orchestrator.remove_event_listener(task_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

