import json
import sqlite3
from pathlib import Path
from typing import Any
from app.core.config import settings
from app.agent.state import TaskState, TaskStatus
from app.core.logging import logger

class SQLiteCheckpointer:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (settings.DATA_DIR / "aura_checkpoints.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
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
            state_json = json.dumps(state_dict)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO task_checkpoints (
                        task_id, objective, privacy_mode, status, state_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_state.task_id,
                    task_state.objective,
                    task_state.privacy_mode,
                    task_state.status.value,
                    state_json,
                    task_state.created_at,
                    task_state.updated_at
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist checkpoint for task {task_state.task_id}: {e}")

    def load_checkpoint(self, task_id: str) -> TaskState | None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT state_json FROM task_checkpoints WHERE task_id = ?", (task_id,))
                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    return TaskState(**data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint for task {task_id}: {e}")
        return None

    def list_checkpoints(self, limit: int = 50) -> list[dict[str, Any]]:
        results = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT task_id, objective, privacy_mode, status, created_at, updated_at
                    FROM task_checkpoints
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))
                for row in cursor.fetchall():
                    results.append({
                        "task_id": row[0],
                        "objective": row[1],
                        "privacy_mode": row[2],
                        "status": row[3],
                        "created_at": row[4],
                        "updated_at": row[5]
                    })
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
        return results

checkpointer = SQLiteCheckpointer()
