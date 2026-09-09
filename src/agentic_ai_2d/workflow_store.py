"""Durable local records required to resume a Phase 2 review session."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from pydantic import BaseModel


class ImmutableWorkflowRecordError(ValueError):
    """Raised when code attempts to replace a persisted workflow record."""


class WorkflowStore:
    """Store immutable versioned JSON records beside the local artifact store."""

    def __init__(self, path: Path | str) -> None:
        self._db = sqlite3.connect(Path(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS workflow_records (project_id TEXT NOT NULL, project_version INTEGER NOT NULL, "
            "record_type TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (project_id, project_version, record_type))"
        )
        self._db.commit()

    def save(self, project_id: str, project_version: int, record_type: str, payload: BaseModel | dict) -> None:
        """Insert a record once, allowing only an identical retry after a crash."""
        content = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
        try:
            self._db.execute(
                "INSERT INTO workflow_records VALUES (?, ?, ?, ?)",
                (project_id, project_version, record_type, encoded),
            )
            self._db.commit()
        except sqlite3.IntegrityError as error:
            existing = self.get(project_id, project_version, record_type)
            if existing == content:
                return
            raise ImmutableWorkflowRecordError(
                f"workflow record already exists for {project_id} version {project_version}: {record_type}"
            ) from error

    def get(self, project_id: str, project_version: int, record_type: str) -> dict | None:
        row = self._db.execute(
            "SELECT payload_json FROM workflow_records WHERE project_id = ? AND project_version = ? AND record_type = ?",
            (project_id, project_version, record_type),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def close(self) -> None:
        self._db.close()
