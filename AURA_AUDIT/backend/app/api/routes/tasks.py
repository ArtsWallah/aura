import json
import asyncio
from typing import Literal
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.orchestrator import orchestrator
from app.agent.state import TaskState, TaskStatus, TaskEvent
from app.agent.checkpoint import checkpointer
from app.core.config import settings

router = APIRouter(prefix="/tasks", tags=["Tasks"])

class CreateTaskRequest(BaseModel):
    objective: str = Field(description="High-level natural language instruction or query.")
    file_path: str | None = Field(default=None, description="Path to uploaded dataset/document.")
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] = Field(
        default="hybrid",
        description="Privacy execution mode."
    )

@router.post("", response_model=TaskState)
async def create_task(request: CreateTaskRequest, background_tasks: BackgroundTasks):
    # Launch orchestrator
    # We run the task as a background task so client receives immediate task_id and can listen to SSE
    task_id = f"task_{checkpointer.db_path.name[:4]}_{asyncio.get_event_loop().time():.0f}"
    
    # Run synchronously or in background
    # Running directly in async handler or background task
    state = await orchestrator.run_task(
        objective=request.objective,
        file_path=request.file_path,
        privacy_mode=request.privacy_mode
    )
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
def cancel_task(task_id: str):
    state = orchestrator.get_task_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    state.status = TaskStatus.CANCELLED
    checkpointer.save_checkpoint(state)
    return {"message": f"Task '{task_id}' cancelled.", "status": state.status}

@router.get("/{task_id}/events")
async def stream_task_events(task_id: str):
    """
    Server-Sent Events (SSE) streaming live updates for a task.
    """
    queue = orchestrator.register_event_listener(task_id)

    async def event_generator():
        try:
            # Yield initial state
            state = orchestrator.get_task_state(task_id)
            if state:
                yield f"event: initial_state\ndata: {json.dumps(state.model_dump())}\n\n"

            while True:
                try:
                    event: TaskEvent = await asyncio.wait_for(queue.get(), timeout=20.0)
                    payload = json.dumps(event.model_dump())
                    yield f"event: {event.event_type}\ndata: {payload}\n\n"
                    
                    if event.event_type in ["task_completed", "task_failed"]:
                        break
                except asyncio.TimeoutError:
                    # Keep-alive ping
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

