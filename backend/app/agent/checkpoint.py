import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from app.core.config import settings
from app.agent.state import TaskState, TaskStatus
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
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    def save_checkpoint(self, task_state: TaskState):
        try:
            state_dict = task_state.model_dump()
            state_json = encryption_service().encrypt_bytes(json.dumps(state_dict).encode("utf-8")).decode("ascii")
            
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO task_checkpoints (
                        task_id, objective, privacy_mode, status, state_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_state.task_id,
                    "",  # Objective exists only inside encrypted state_json.
                    task_state.privacy_mode,
                    task_state.status.value,
                    state_json,
                    task_state.created_at,
                    task_state.updated_at
                ))
                conn.commit()
        except Exception:
            logger.error("Failed to persist encrypted task checkpoint.")
            raise RuntimeError("Encrypted task persistence failed; configure and verify ENCRYPTION_KEY.")

    def load_checkpoint(self, task_id: str) -> TaskState | None:
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT state_json FROM task_checkpoints WHERE task_id = ?", (task_id,))
                row = cursor.fetchone()
                if row:
                    raw = row[0]
                    try:
                        data = json.loads(encryption_service().decrypt_bytes(raw.encode("ascii")).decode("utf-8"))
                    except (ValueError, UnicodeEncodeError):
                        # Legacy plaintext is immediately rewritten in encrypted form.
                        data = json.loads(raw)
                        self.save_checkpoint(TaskState(**data))
                    return TaskState(**data)
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
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute("SELECT state_json FROM task_checkpoints").fetchall()
            for (raw,) in rows:
                try:
                    encryption_service().decrypt_bytes(raw.encode("ascii"))
                except (ValueError, UnicodeEncodeError):
                    try:
                        state = TaskState(**json.loads(raw))
                    except Exception:
                        logger.error("A legacy checkpoint could not be migrated; original row was preserved.")
                        continue
                    encrypted = encryption_service().encrypt_bytes(json.dumps(state.model_dump()).encode("utf-8")).decode("ascii")
                    conn.execute("UPDATE task_checkpoints SET state_json = ?, objective = '' WHERE task_id = ?", (encrypted, state.task_id))
            conn.commit()

checkpointer = SQLiteCheckpointer()

