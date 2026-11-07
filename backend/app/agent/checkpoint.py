import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any
from cryptography.fernet import InvalidToken
from app.core.config import settings
from app.agent.state import TaskState, TaskStatus, StepStatus, TaskEvent
from app.agent.execution_trace import apply_trace_step, sanitize_execution_trace, sanitize_trace_step, trace_construction_failure
from app.agent.intelligence_contracts import EvaluationStatus, StepFailureCategory
from app.core.logging import logger
from app.privacy.encryption import encryption_service

class SQLiteCheckpointer:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (settings.DATA_DIR / "aura_checkpoints.db")
        self._init_db()
        if settings.ENCRYPTION_KEY:
            self._migrate_legacy_rows()

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_checkpoints (
                    task_id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    privacy_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    claimed_by TEXT,
                    claimed_at REAL,
                    event_seq INTEGER NOT NULL DEFAULT 0,
                    recovery_required INTEGER NOT NULL DEFAULT 0,
                    artifact_manifest TEXT
                )
            """)
            existing = {row[1] for row in conn.execute("PRAGMA table_info(task_checkpoints)").fetchall()}
            additive_columns = {
                "claimed_by": "TEXT",
                "claimed_at": "REAL",
                "event_seq": "INTEGER NOT NULL DEFAULT 0",
                "recovery_required": "INTEGER NOT NULL DEFAULT 0",
                "artifact_manifest": "TEXT",
            }
            for column, declaration in additive_columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE task_checkpoints ADD COLUMN {column} {declaration}")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_events (
                    task_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (task_id, sequence)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_checkpoints_status_updated ON task_checkpoints(status, updated_at)")
            conn.execute("""
                UPDATE task_checkpoints SET status = CASE status
                    WHEN 'pending' THEN 'queued'
                    WHEN 'executing' THEN 'running'
                    WHEN 'evaluating' THEN 'running'
                    ELSE status
                END
                WHERE status IN ('pending', 'executing', 'evaluating')
            """)
            conn.commit()

    def save_checkpoint(self, task_state: TaskState):
        try:
            values = self._checkpoint_values(task_state)
            with closing(sqlite3.connect(self.db_path)) as conn:
                self._write_checkpoint(conn, values)
                conn.commit()
        except Exception:
            logger.error("Failed to persist encrypted task checkpoint.")
            raise RuntimeError("Encrypted task persistence failed; configure and verify ENCRYPTION_KEY.")

    def _checkpoint_values(self, task_state: TaskState) -> tuple[Any, ...]:
        manifest_json = json.dumps(self._build_artifact_manifest(task_state), separators=(",", ":"))
        manifest_ciphertext = encryption_service().encrypt_bytes(manifest_json.encode("utf-8")).decode("ascii")
        persisted = task_state.model_copy(deep=True)
        # Persist only lifecycle metadata; raw inputs, arguments, outputs and rows
        # remain process-local and are never written to SQLite.
        persisted.objective = ""
        persisted.file_path = None
        persisted.step_results = {}
        persisted.final_deliverable = None
        persisted.artifacts = []
        persisted.error_summary = None
        if persisted.plan:
            persisted.plan.goal = ""
            for step in persisted.plan.steps:
                step.title = ""
                step.description = ""
                step.parameters = {}
                step.output = None
                step.error = None
        state_json = encryption_service().encrypt_bytes(
            json.dumps(persisted.model_dump()).encode("utf-8")
        ).decode("ascii")
        return (
            task_state.task_id,
            task_state.privacy_mode,
            task_state.status.value,
            state_json,
            task_state.created_at,
            task_state.updated_at,
            int(task_state.recovery_required),
            manifest_ciphertext,
        )

    @staticmethod
    def _write_checkpoint(conn: sqlite3.Connection, values: tuple[Any, ...]) -> None:
        task_id, privacy_mode, status, state_json, created_at, updated_at, recovery_required, manifest_ciphertext = values
        conn.execute("""
                INSERT INTO task_checkpoints (
                        task_id, objective, privacy_mode, status, state_json, created_at, updated_at,
                        recovery_required, artifact_manifest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    objective = excluded.objective,
                    privacy_mode = excluded.privacy_mode,
                    status = excluded.status,
                    state_json = excluded.state_json,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    recovery_required = excluded.recovery_required,
                    artifact_manifest = excluded.artifact_manifest
                """, (
            task_id,
            "",  # Sensitive request text is intentionally not persisted.
            privacy_mode,
            status,
            state_json,
            created_at,
            updated_at,
            recovery_required,
            manifest_ciphertext,
        ))

    @staticmethod
    def _artifact_type_and_id(filename: str) -> tuple[str, str] | None:
        if re.fullmatch(r"report_[a-f0-9]{10}\.html\.enc", filename):
            return "report_html", filename.removesuffix(".html.enc") + "_html"
        if re.fullmatch(r"report_[a-f0-9]{10}\.md\.enc", filename):
            return "report_markdown", filename.removesuffix(".md.enc") + "_markdown"
        if re.fullmatch(r"chart_[a-f0-9]{10}\.png\.enc", filename):
            return "chart_png", filename.removesuffix(".png.enc")
        match = re.fullmatch(r"transformed_([a-f0-9]{10})_([a-f0-9]{10})\.csv\.enc", filename)
        if match:
            return "transformed_dataset", f"transformed_{match.group(2)}"
        return None

    @staticmethod
    def _build_artifact_manifest(task_state: TaskState) -> list[dict[str, str]]:
        """Build typed references to encrypted artifacts directly inside approved roots."""
        roots = {
            "reports": settings.REPORTS_DIR.resolve(),
            "charts": settings.CHARTS_DIR.resolve(),
            "transforms": settings.TRANSFORMS_DIR.resolve(),
        }
        candidates = list(task_state.artifacts)
        deliverable = task_state.final_deliverable or {}
        for key in ("html_path", "markdown_path"):
            value = deliverable.get(key)
            if isinstance(value, str):
                candidates.append(value)

        manifest: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for artifact in candidates:
            try:
                original = Path(artifact)
                resolved = original.resolve(strict=True)
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            if resolved.name != original.name:
                continue
            kind = SQLiteCheckpointer._artifact_type_and_id(resolved.name)
            if kind is None:
                continue
            artifact_type, artifact_id = kind
            if artifact_type == "chart_png":
                directory = "charts"
            elif artifact_type in {"report_html", "report_markdown"}:
                directory = "reports"
            elif artifact_type == "transformed_dataset":
                directory = "transforms"
                expected_task_hash = hashlib.sha256(task_state.task_id.encode("utf-8")).hexdigest()[:10]
                match = re.fullmatch(r"transformed_([a-f0-9]{10})_([a-f0-9]{10})\.csv\.enc", resolved.name)
                if not match or match.group(1) != expected_task_hash:
                    continue
            else:
                continue
            if resolved.parent != roots[directory] or not resolved.is_file():
                continue
            key = (artifact_type, resolved.name)
            if key in seen:
                continue
            seen.add(key)
            manifest.append({
                "artifact_id": artifact_id,
                "task_id": task_state.task_id,
                "artifact_type": artifact_type,
                "relative_path": resolved.name,
            })
        return manifest

    def artifact_manifest_for_task(self, task_state: TaskState) -> list[dict[str, str]]:
        """Return validated, existing artifact references bound to this task ID."""
        return self._build_artifact_manifest(task_state)

    def _restore_artifact_manifest(self, state: TaskState, encrypted: str | None) -> None:
        """Restore paths only after authenticating and validating every manifest entry."""
        if not encrypted:
            return
        raw = encryption_service().decrypt_bytes(encrypted.encode("ascii"))
        entries = json.loads(raw.decode("utf-8"))
        if not isinstance(entries, list) or len(entries) > 1000:
            raise ValueError("Invalid artifact manifest.")

        roots = {
            "reports": settings.REPORTS_DIR.resolve(),
            "charts": settings.CHARTS_DIR.resolve(),
            "transforms": settings.TRANSFORMS_DIR.resolve(),
        }
        restored: list[str] = []
        report_id: str | None = None
        seen: set[tuple[str, str]] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "artifact_id", "task_id", "artifact_type", "relative_path"
            }:
                raise ValueError("Invalid artifact manifest entry.")
            artifact_type = entry["artifact_type"]
            relative_path = entry["relative_path"]
            if entry["task_id"] != state.task_id or not isinstance(relative_path, str):
                raise ValueError("Artifact manifest task mismatch.")
            if artifact_type == "chart_png":
                directory = "charts"
            elif artifact_type in {"report_html", "report_markdown"}:
                directory = "reports"
            elif artifact_type == "transformed_dataset":
                directory = "transforms"
                expected_task_hash = hashlib.sha256(state.task_id.encode("utf-8")).hexdigest()[:10]
                match = re.fullmatch(r"transformed_([a-f0-9]{10})_([a-f0-9]{10})\.csv\.enc", relative_path)
                if not match or match.group(1) != expected_task_hash:
                    raise ValueError("Artifact manifest task hash mismatch.")
            else:
                raise ValueError("Unsupported artifact type.")
            kind = self._artifact_type_and_id(relative_path)
            if kind is None or kind[0] != artifact_type or kind[1] != entry["artifact_id"]:
                raise ValueError("Artifact manifest identity mismatch.")
            if (
                Path(relative_path).is_absolute()
                or "/" in relative_path
                or "\\" in relative_path
                or ":" in relative_path
            ):
                raise ValueError("Artifact manifest path is not relative.")
            key = (artifact_type, relative_path)
            if key in seen:
                raise ValueError("Duplicate artifact manifest entry.")
            seen.add(key)
            candidate = (roots[directory] / relative_path).resolve(strict=True)
            if candidate.parent != roots[directory] or candidate.name != relative_path or not candidate.is_file():
                raise ValueError("Artifact path escaped its approved directory.")
            restored.append(str(candidate))
            if artifact_type == "report_html":
                report_id = relative_path.removesuffix(".html.enc")

        state.artifacts = restored
        if state.status == TaskStatus.COMPLETED and report_id is not None:
            # Only metadata required by the task-scoped report routes is restored.
            state.final_deliverable = {
                "report_id": report_id,
                "title": "Analytical Deliverable",
            }

    def enqueue_task(self, task_state: TaskState) -> None:
        if task_state.status != TaskStatus.QUEUED:
            raise ValueError("Only queued tasks can be added to the persistent queue.")
        self.save_checkpoint(task_state)

    def discard_unclaimed_queued_task(self, task_id: str) -> bool:
        """Atomically roll back a not-yet-claimed enqueue and its initial events."""
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, claimed_by FROM task_checkpoints WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                conn.commit()
                return True
            if row[0] != TaskStatus.QUEUED.value or row[1] is not None:
                conn.rollback()
                return False
            conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
            conn.execute(
                "DELETE FROM task_checkpoints WHERE task_id = ? AND status = ? AND claimed_by IS NULL",
                (task_id, TaskStatus.QUEUED.value),
            )
            conn.commit()
            return True

    def claim_next_queued_task(self, worker_id: str) -> TaskState | None:
        """Atomically claim one queued row so workers cannot execute it twice."""
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT task_id FROM task_checkpoints WHERE status = ? AND claimed_by IS NULL ORDER BY created_at LIMIT 1",
                (TaskStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            task_id = row[0]
            updated = conn.execute(
                "UPDATE task_checkpoints SET claimed_by = ?, claimed_at = ? WHERE task_id = ? AND status = ? AND claimed_by IS NULL",
                (worker_id, time.time(), task_id, TaskStatus.QUEUED.value),
            ).rowcount
            conn.commit()
        if updated != 1:
            return None
        return self.load_checkpoint(task_id)

    def release_task_claim(self, task_id: str, worker_id: str) -> None:
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute(
                "UPDATE task_checkpoints SET claimed_by = NULL, claimed_at = NULL WHERE task_id = ? AND claimed_by = ?",
                (task_id, worker_id),
            )
            conn.commit()

    def append_event(
        self,
        event: TaskEvent,
        retention: int = 1000,
        *,
        _connection: sqlite3.Connection | None = None,
    ) -> int:
        """Atomically assign an ID and persist a strict privacy-safe event projection."""
        known_types = {
            "task.created", "task_started", "plan_created", "step_started", "step_completed",
            "step_failed", "step_retry", "task_completed", "task_failed", "task_cancelled",
        }
        event_type = event.event_type if event.event_type in known_types else "task_update"
        safe_messages = {
            "task.created": "Task accepted.",
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
        allowed_statuses = {status.value for status in TaskStatus}
        safe_data: dict[str, Any] = {}
        status = event.data.get("status")
        if isinstance(status, str) and status in allowed_statuses:
            safe_data["status"] = status
        for key in ("step_count", "step_index", "duration_ms", "retry_count"):
            value = event.data.get(key)
            if isinstance(value, int) and 0 <= value <= 1_000_000:
                safe_data[key] = value
        tool_name = event.data.get("tool_name")
        if isinstance(tool_name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", tool_name):
            safe_data["tool_name"] = tool_name
        error_category = event.data.get("error_category")
        if isinstance(error_category, str) and error_category in {"tool_failure", "provider_failure", "internal_error"}:
            safe_data["error_category"] = error_category
        if event.data.get("recovery_required") is True:
            safe_data["recovery_required"] = True
        trace_snapshot = event.data.get("trace_snapshot")
        if isinstance(trace_snapshot, dict):
            safe_trace = sanitize_execution_trace(trace_snapshot)
            if safe_trace is not None:
                if event_type in {"task_completed", "task_failed", "task_cancelled"} and safe_trace.evaluation_status is None:
                    safe_trace = trace_construction_failure(timing=safe_trace.timing)
                safe_data["trace_snapshot"] = safe_trace.model_dump(mode="json")
            else:
                if event_type in {"task_completed", "task_failed", "task_cancelled"}:
                    safe_data["trace_snapshot"] = trace_construction_failure().model_dump(mode="json")
        elif event_type in {"task_completed", "task_failed", "task_cancelled"}:
            safe_data["trace_snapshot"] = trace_construction_failure().model_dump(mode="json")
        trace_step = event.data.get("trace_step")
        if isinstance(trace_step, dict):
            safe_step = sanitize_trace_step(trace_step)
            if safe_step is not None and safe_step.step_id == event.step_id:
                safe_data["trace_step"] = safe_step.model_dump(mode="json")
        safe_event = TaskEvent(
            event_type=event_type,
            task_id=event.task_id,
            timestamp=event.timestamp,
            step_id=event.step_id if isinstance(event.step_id, str) and re.fullmatch(r"step_[A-Za-z0-9][A-Za-z0-9_-]{0,63}", event.step_id) else None,
            message=safe_messages[event_type],
            data=safe_data,
        )
        if _connection is not None:
            return self._append_projected_event(_connection, safe_event, retention)
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            sequence = self._append_projected_event(conn, safe_event, retention)
            conn.commit()
            return sequence

    @staticmethod
    def _append_projected_event(
        conn: sqlite3.Connection,
        safe_event: TaskEvent,
        retention: int,
    ) -> int:
        payload = json.dumps(safe_event.model_dump())
        row = conn.execute(
            "SELECT event_seq FROM task_checkpoints WHERE task_id = ?", (safe_event.task_id,)
        ).fetchone()
        if row is None:
            raise KeyError("Task event has no checkpoint.")
        sequence = int(row[0]) + 1
        conn.execute(
            "UPDATE task_checkpoints SET event_seq = ? WHERE task_id = ?",
            (sequence, safe_event.task_id),
        )
        conn.execute(
            "INSERT INTO task_events(task_id, sequence, event_type, created_at, payload_json) VALUES (?, ?, ?, ?, ?)",
            (safe_event.task_id, sequence, safe_event.event_type, safe_event.timestamp, payload),
        )
        conn.execute(
            "DELETE FROM task_events WHERE task_id = ? AND sequence <= ?",
            (safe_event.task_id, max(0, sequence - retention)),
        )
        return sequence

    def save_checkpoint_and_append_event(
        self,
        task_state: TaskState,
        event: TaskEvent,
        retention: int = 1000,
    ) -> tuple[int, TaskEvent]:
        """Commit a terminal checkpoint and its sanitized event in one transaction."""
        expected_event = {
            TaskStatus.COMPLETED: "task_completed",
            TaskStatus.FAILED: "task_failed",
            TaskStatus.CANCELLED: "task_cancelled",
        }.get(task_state.status)
        if (
            event.task_id != task_state.task_id
            or event.event_type != expected_event
        ):
            raise ValueError("Terminal checkpoint and event do not match.")
        values = self._checkpoint_values(task_state)
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._write_checkpoint(conn, values)
            sequence = self.append_event(event, retention, _connection=conn)
            row = conn.execute(
                "SELECT payload_json FROM task_events WHERE task_id = ? AND sequence = ?",
                (event.task_id, sequence),
            ).fetchone()
            safe_event = TaskEvent(**json.loads(row[0]))
            conn.commit()
        return sequence, safe_event

    def read_events(self, task_id: str, after_sequence: int = 0, limit: int = 1000) -> list[tuple[int, TaskEvent]]:
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            rows = conn.execute(
                "SELECT sequence, payload_json FROM task_events WHERE task_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (task_id, max(0, after_sequence), min(max(1, limit), 1000)),
            ).fetchall()
        return [(int(sequence), TaskEvent(**json.loads(payload))) for sequence, payload in rows]

    def read_event_window(
        self, task_id: str, after_sequence: int = 0, limit: int = 1000
    ) -> tuple[int, int | None, list[tuple[int, TaskEvent]]]:
        """Read the event cursor and retained window from one SQLite snapshot."""
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute("BEGIN")
            row = conn.execute(
                "SELECT event_seq, (SELECT MIN(sequence) FROM task_events WHERE task_id = ?) "
                "FROM task_checkpoints WHERE task_id = ?",
                (task_id, task_id),
            ).fetchone()
            latest = int(row[0]) if row else 0
            earliest = int(row[1]) if row and row[1] is not None else None
            rows = conn.execute(
                "SELECT sequence, payload_json FROM task_events "
                "WHERE task_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (task_id, max(0, after_sequence), min(max(1, limit), 1000)),
            ).fetchall()
            conn.commit()
        events = [(int(sequence), TaskEvent(**json.loads(payload))) for sequence, payload in rows]
        return latest, earliest, events

    def latest_event_sequence(self, task_id: str) -> int:
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            row = conn.execute(
                "SELECT event_seq FROM task_checkpoints WHERE task_id = ?", (task_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def recover_interrupted_tasks(self, available_queued_ids: set[str] | None = None) -> int:
        """Fail ambiguous interrupted work without replaying any tool operation."""
        available_queued_ids = available_queued_ids or set()
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            rows = conn.execute(
                "SELECT task_id, status FROM task_checkpoints WHERE status IN (?, ?, ?, ?, ?, ?)",
                tuple(status.value for status in (
                    TaskStatus.QUEUED,
                    TaskStatus.PLANNING, TaskStatus.RUNNING, TaskStatus.WAITING,
                    TaskStatus.RETRYING, TaskStatus.CANCEL_REQUESTED,
                )),
            ).fetchall()
            inconsistent_terminal_rows = conn.execute("""
                SELECT c.task_id, c.status
                FROM task_checkpoints c
                WHERE c.status IN ('completed', 'failed', 'cancelled')
                AND NOT EXISTS (
                    SELECT 1 FROM task_events e
                    WHERE e.task_id = c.task_id
                    AND e.event_type = CASE c.status
                        WHEN 'completed' THEN 'task_completed'
                        WHEN 'failed' THEN 'task_failed'
                        WHEN 'cancelled' THEN 'task_cancelled'
                    END
                )
            """).fetchall()
            conn.execute("UPDATE task_checkpoints SET claimed_by = NULL, claimed_at = NULL")
            conn.commit()

        recovered = 0
        handled_task_ids: set[str] = set()
        for task_id, persisted_status in rows:
            if persisted_status == TaskStatus.QUEUED.value and task_id in available_queued_ids:
                continue
            state = self.load_checkpoint(task_id)
            if state is None:
                # A missing/wrong key makes the payload unreadable; leave it untouched.
                continue
            if state.plan:
                for step in state.plan.steps:
                    if step.status == StepStatus.IN_PROGRESS:
                        step.status = StepStatus.INTERRUPTED
                        step.error = "Execution was interrupted by process shutdown; automatic replay is disabled."
            state.recovery_required = True
            state.recovery_reason = (
                "Queued task input was process-local and is unavailable after restart."
                if persisted_status == TaskStatus.QUEUED.value
                else "The prior process stopped during execution; outcome may be ambiguous."
            )
            if state.status != TaskStatus.FAILED:
                state.transition_to(TaskStatus.FAILED)
            state.error_summary = "Execution could not be safely resumed; review before manually resubmitting."
            trace = self._recovered_trace_snapshot(state)
            if trace is None:
                trace = trace_construction_failure()
            event = TaskEvent(
                event_type="task_failed",
                task_id=task_id,
                message="Task outcome requires review after restart.",
                data={
                    "status": "failed",
                    "recovery_required": True,
                    "trace_snapshot": trace.model_dump(mode="json"),
                },
            )
            self.save_checkpoint_and_append_event(state, event)
            handled_task_ids.add(task_id)
            recovered += 1

        for task_id, persisted_status in inconsistent_terminal_rows:
            if task_id in handled_task_ids:
                continue
            state = self.load_checkpoint(task_id)
            if state is None:
                continue
            state.recovery_required = True
            state.recovery_reason = "Terminal task event was missing; outcome marked for review."
            terminal_event_type = {
                TaskStatus.COMPLETED: "task_completed",
                TaskStatus.FAILED: "task_failed",
                TaskStatus.CANCELLED: "task_cancelled",
            }.get(state.status, "task_failed")
            state.error_summary = "Execution could not be safely verified; review before resubmitting."
            trace = self._recovered_trace_snapshot(state)
            if trace is None:
                trace = trace_construction_failure()
            event = TaskEvent(
                event_type=terminal_event_type,
                task_id=task_id,
                message="Task outcome requires review after restart.",
                data={
                    "status": state.status.value,
                    "recovery_required": True,
                    "trace_snapshot": trace.model_dump(mode="json"),
                },
            )
            self.save_checkpoint_and_append_event(state, event)
            recovered += 1

        return recovered

    def _recovered_trace_snapshot(self, state: TaskState):
        """Mark the last safe trace snapshot inconclusive for SSE and detail replay."""
        trace = None
        for _, event in self.read_events(state.task_id, 0, 1000):
            snapshot = event.data.get("trace_snapshot")
            if isinstance(snapshot, dict):
                trace = sanitize_execution_trace(snapshot)
            elif trace is not None and isinstance(event.data.get("trace_step"), dict):
                trace = apply_trace_step(trace, event.data)
        if trace is None:
            return None
        if state.plan is not None:
            persisted_steps = {step.step_id: step for step in state.plan.steps}
            steps = []
            for item in trace.step_statuses:
                persisted = persisted_steps.get(item.step_id)
                if persisted is None:
                    continue
                steps.append(item.model_copy(update={
                    "status": persisted.status,
                    "retry_count": persisted.retry_count,
                    "failure_category": (
                        StepFailureCategory.INTERRUPTED
                        if persisted.status == StepStatus.INTERRUPTED
                        else item.failure_category
                    ),
                }))
            trace = trace.model_copy(update={"step_statuses": steps})
        return sanitize_execution_trace(trace.model_copy(update={
            "evaluation_status": EvaluationStatus.INCONCLUSIVE,
        }))

    def load_checkpoint(self, task_id: str) -> TaskState | None:
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT state_json, artifact_manifest FROM task_checkpoints WHERE task_id = ?",
                    (task_id,),
                )
                row = cursor.fetchone()
                if row:
                    raw = row[0]
                    encrypted_manifest = row[1]
                    try:
                        data = json.loads(encryption_service().decrypt_bytes(raw.encode("ascii")).decode("utf-8"))
                    except (InvalidToken, ValueError, UnicodeEncodeError):
                        # Preserve legacy task contents while encrypting them in place.
                        data = json.loads(raw)
                        legacy_state = TaskState(**data)
                        encrypted = encryption_service().encrypt_bytes(
                            json.dumps(legacy_state.model_dump()).encode("utf-8")
                        ).decode("ascii")
                        conn.execute(
                            "UPDATE task_checkpoints SET objective = '', privacy_mode = ?, status = ?, state_json = ?, updated_at = ? WHERE task_id = ?",
                            (legacy_state.privacy_mode, legacy_state.status.value, encrypted, legacy_state.updated_at, task_id),
                        )
                        conn.commit()
                    state = TaskState(**data)
                    try:
                        self._restore_artifact_manifest(state, encrypted_manifest)
                    except Exception:
                        # A damaged manifest must not make status lookup fail or
                        # expose an unvalidated path. Artifacts remain unavailable.
                        state.artifacts = []
                        state.final_deliverable = None
                        logger.error("Task artifact manifest could not be restored safely.")
                    return state
        except Exception:
            logger.error("Failed to load task checkpoint.")
        return None

    def list_checkpoints(self, limit: int = 50) -> list[dict[str, Any]]:
        results = []
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT task_id, privacy_mode, status, created_at, updated_at
                    FROM task_checkpoints
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))
                for row in cursor.fetchall():
                    results.append({
                        "task_id": row[0],
                        "privacy_mode": row[1],
                        "status": row[2],
                        "created_at": row[3],
                        "updated_at": row[4]
                    })
        except Exception:
            logger.error("Failed to list task checkpoints.")
        return results

    def _migrate_legacy_rows(self):
        """Encrypt pre-Phase-1 plaintext checkpoint rows in place, preserving state."""
        try:
            cipher = encryption_service()
        except (RuntimeError, ValueError):
            logger.error("Checkpoint migration skipped because ENCRYPTION_KEY is unavailable or invalid; existing rows were preserved.")
            return
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute("SELECT state_json FROM task_checkpoints").fetchall()
            for (raw,) in rows:
                try:
                    cipher.decrypt_bytes(raw.encode("ascii"))
                except (InvalidToken, ValueError, UnicodeEncodeError):
                    try:
                        state = TaskState(**json.loads(raw))
                    except Exception:
                        logger.error("A legacy checkpoint could not be migrated; original row was preserved.")
                        continue
                    encrypted = cipher.encrypt_bytes(json.dumps(state.model_dump()).encode("utf-8")).decode("ascii")
                    conn.execute(
                        "UPDATE task_checkpoints SET state_json = ?, objective = '', status = ? WHERE task_id = ?",
                        (encrypted, state.status.value, state.task_id)
                    )
            conn.commit()

checkpointer = SQLiteCheckpointer()

