import json
import asyncio
import uuid
import re
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.agent.orchestrator import orchestrator
from app.agent.state import TaskState, TaskStatus, TaskEvent
from app.agent.checkpoint import checkpointer
from app.agent.service import task_execution_service
from app.core.config import settings
from app.core.security import validate_safe_path
from app.privacy.encryption import encryption_service

router = APIRouter(prefix="/tasks", tags=["Tasks"])

class CreateTaskRequest(BaseModel):
    objective: str = Field(description="High-level natural language instruction or query.")
    file_path: str | None = Field(default=None, description="Path to uploaded dataset/document.")
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] | None = Field(
        default=None,
        description="Privacy execution mode."
    )


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"]
    created_at: float
    updated_at: float


class TaskArtifactManifest(BaseModel):
    task_id: str
    title: str
    html_url: str | None = None
    markdown_url: str | None = None
    chart_urls: list[str] = Field(default_factory=list)


class TaskArtifactContentResponse(BaseModel):
    task_id: str
    artifact_type: Literal["markdown"]
    content: str


def _status_response(state: TaskState) -> TaskStatusResponse:
    return TaskStatusResponse(
        task_id=state.task_id,
        status=state.status,
        privacy_mode=state.privacy_mode,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )

@router.post("", response_model=TaskStatusResponse, status_code=status.HTTP_202_ACCEPTED)
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
    return _status_response(state)

@router.get("", response_model=list[TaskStatusResponse])
def list_tasks(limit: int = 25):
    return [TaskStatusResponse(**row) for row in checkpointer.list_checkpoints(limit=limit)]

@router.get("/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str):
    state = orchestrator.get_task_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found.")
    return _status_response(state)


def _completed_artifact_state(task_id: str) -> TaskState:
    state = orchestrator.get_task_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found.")
    if state.status != TaskStatus.COMPLETED or not state.final_deliverable:
        raise HTTPException(status_code=404, detail="Task artifacts are not available.")
    return state


def _scoped_report_path(state: TaskState, extension: Literal["html", "md"]) -> Path:
    report_id = state.final_deliverable.get("report_id", "") if state.final_deliverable else ""
    if not isinstance(report_id, str) or not re.fullmatch(r"report_[a-f0-9]{10}", report_id):
        raise HTTPException(status_code=404, detail="Task artifact not found.")
    root = settings.REPORTS_DIR.resolve()
    artifact_name = f"{report_id}.{extension}.enc"
    path = (root / artifact_name).resolve()
    listed_path = any(Path(artifact).resolve() == path for artifact in state.artifacts)
    if not listed_path or path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="Task artifact not found.")
    return path


def _scoped_chart_path(state: TaskState, chart_index: int) -> Path:
    root = settings.CHARTS_DIR.resolve()
    chart_names = []
    for artifact in state.artifacts:
        candidate = Path(artifact).resolve()
        if (
            candidate.parent == root
            and re.fullmatch(r"chart_[a-f0-9]{10}\.png\.enc", candidate.name)
            and candidate.is_file()
        ):
            chart_names.append(candidate.name)
    if chart_index < 0 or chart_index >= len(chart_names):
        raise HTTPException(status_code=404, detail="Task artifact not found.")
    path = (root / chart_names[chart_index]).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="Task artifact not found.")
    return path


@router.get("/{task_id}/artifacts", response_model=TaskArtifactManifest)
def get_task_artifacts(task_id: str):
    """Return a task-scoped manifest, never filesystem paths or report contents."""
    state = _completed_artifact_state(task_id)

    html_path = _scoped_report_path(state, "html")
    markdown_path = _scoped_report_path(state, "md")
    chart_urls = []
    chart_count = 0
    for artifact in state.artifacts:
        try:
            _scoped_chart_path(state, chart_count)
        except HTTPException:
            break
        chart_count += 1
    for index in range(chart_count):
        _scoped_chart_path(state, index)
        chart_urls.append(f"/api/tasks/{task_id}/artifacts/chart/{index}")

    return TaskArtifactManifest(
        task_id=task_id,
        title=str(state.final_deliverable.get("title") or "Analytical Deliverable")[:200],
        html_url=f"/api/tasks/{task_id}/artifacts/html" if html_path else None,
        markdown_url=f"/api/tasks/{task_id}/artifacts/markdown" if markdown_path else None,
        chart_urls=chart_urls,
    )


@router.get("/{task_id}/artifacts/markdown", response_model=TaskArtifactContentResponse)
def get_task_markdown_artifact(task_id: str):
    state = _completed_artifact_state(task_id)
    try:
        content = encryption_service().decrypt_bytes(_scoped_report_path(state, "md").read_bytes()).decode("utf-8")
    except (RuntimeError, ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Task artifact decryption unavailable.") from exc
    return TaskArtifactContentResponse(task_id=task_id, artifact_type="markdown", content=content)


@router.get("/{task_id}/artifacts/html", response_class=Response)
def get_task_html_artifact(task_id: str):
    state = _completed_artifact_state(task_id)
    try:
        content = encryption_service().decrypt_bytes(_scoped_report_path(state, "html").read_bytes())
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Task artifact decryption unavailable.") from exc
    return Response(content=content, media_type="text/html", headers={
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; sandbox",
    })


@router.get("/{task_id}/artifacts/chart/{chart_index}", response_class=Response)
def get_task_chart_artifact(task_id: str, chart_index: int):
    state = _completed_artifact_state(task_id)
    try:
        content = encryption_service().decrypt_bytes(_scoped_chart_path(state, chart_index).read_bytes())
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Task artifact decryption unavailable.") from exc
    return Response(content=content, media_type="image/png", headers={
        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    })

@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    state = orchestrator.get_task_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found.")
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
    latest_sequence = checkpointer.latest_event_sequence(task_id)
    if after_sequence > latest_sequence:
        raise HTTPException(status_code=400, detail="Last-Event-ID cannot exceed the latest task event.")
    queue = orchestrator.register_event_listener(task_id)

    async def event_generator():
        cursor = after_sequence
        terminal_types = {"task_completed", "task_failed", "task_cancelled"}

        def encode(sequence: int, event: TaskEvent) -> str:
            payload = json.dumps(event.model_dump(), separators=(",", ":"))
            return f"id: {sequence}\nevent: {event.event_type}\ndata: {payload}\n\n"

        async def replay_window() -> tuple[list[str], bool, int]:
            """Replay retained events, replacing an expired history with a safe snapshot."""
            nonlocal cursor
            latest, earliest, batch = await asyncio.to_thread(
                checkpointer.read_event_window, task_id, cursor
            )
            if (earliest is None and cursor < latest) or (
                earliest is not None and cursor < earliest - 1
            ):
                current = orchestrator.get_task_state(task_id)
                current_status = current.status.value if current else "unknown"
                gap = TaskEvent(
                    event_type="task.history_gap",
                    task_id=task_id,
                    message="Earlier task events expired; the current task status is shown.",
                    data={"status": current_status, "history_gap": True},
                )
                cursor = latest
                return [f"id: {latest}\nevent: task.history_gap\ndata: "
                        f"{json.dumps(gap.model_dump(), separators=(',', ':'))}\n\n"], False, latest

            frames = []
            terminal = False
            for sequence, event in batch:
                if sequence <= cursor:
                    continue
                cursor = sequence
                frames.append(encode(sequence, event))
                if event.event_type in terminal_types:
                    terminal = True
                    break
            return frames, terminal, latest

        try:
            frames, terminal, _ = await replay_window()
            for frame in frames:
                yield frame
            if terminal:
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

            # A reconnect at the terminal sequence has nothing left to replay.
            # Close it immediately instead of waiting for a queue notification
            # that can never arrive. A fresh read also covers completion during
            # the initial replay query.
            current = orchestrator.get_task_state(task_id)
            if current and current.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                return

            while True:
                try:
                    await asyncio.wait_for(queue.get(), timeout=20.0)
                    while True:
                        frames, terminal, latest = await replay_window()
                        for frame in frames:
                            yield frame
                        if terminal:
                            return
                        if cursor >= latest:
                            break
                        if len(frames) < 1000:
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

