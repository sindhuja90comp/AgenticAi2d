"""Durable SQLite event storage for resumable Phase 2 workflows."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from threading import RLock

from .domain import DomainEvent, EventType, TransitionRejectedEvent, WorkflowTransitionEvent
from .repository import DuplicateEventIdError
from ..models.project_spec import ProjectStatus


class SqliteEventRepository:
    """Append-only event repository that survives a local process restart."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS workflow_events (event_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, "
            "project_version INTEGER NOT NULL, idempotency_key TEXT UNIQUE NOT NULL, event_type TEXT NOT NULL, "
            "event_class TEXT NOT NULL, payload_json TEXT NOT NULL)"
        )
        self._db.commit()

    def append(self, event: DomainEvent) -> DomainEvent:
        with self._lock:
            existing = self.find_by_idempotency_key(event.idempotency_key)
            if existing is not None:
                return existing
            payload = _encode(event)
            try:
                self._db.execute(
                    "INSERT INTO workflow_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (event.event_id, event.project_id, event.project_version, event.idempotency_key,
                     event.event_type.value, type(event).__name__, payload),
                )
                self._db.commit()
            except sqlite3.IntegrityError as error:
                raise DuplicateEventIdError(f"event_id {event.event_id!r} is already in use") from error
            return event

    def list_events(self, project_id: str, project_version: int) -> tuple[DomainEvent, ...]:
        rows = self._db.execute(
            "SELECT event_class, payload_json FROM workflow_events WHERE project_id = ? AND project_version = ? ORDER BY rowid",
            (project_id, project_version),
        )
        return tuple(_decode(event_class, payload) for event_class, payload in rows)

    def find_by_idempotency_key(self, idempotency_key: str) -> DomainEvent | None:
        row = self._db.execute(
            "SELECT event_class, payload_json FROM workflow_events WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return None if row is None else _decode(row[0], row[1])

    def close(self) -> None:
        self._db.close()


def _encode(event: DomainEvent) -> str:
    payload = asdict(event)
    payload["event_type"] = event.event_type.value
    payload["occurred_at"] = event.occurred_at.isoformat()
    for key in ("from_status", "to_status", "attempted_event_type", "current_status"):
        if key in payload and payload[key] is not None:
            payload[key] = payload[key].value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _decode(event_class: str, payload_json: str) -> DomainEvent:
    payload = json.loads(payload_json)
    payload["event_type"] = EventType(payload["event_type"])
    payload["occurred_at"] = datetime.fromisoformat(payload["occurred_at"])
    if "from_status" in payload and payload["from_status"] is not None:
        payload["from_status"] = ProjectStatus(payload["from_status"])
    if "to_status" in payload:
        payload["to_status"] = ProjectStatus(payload["to_status"])
    if "attempted_event_type" in payload:
        payload["attempted_event_type"] = EventType(payload["attempted_event_type"])
    if "current_status" in payload:
        payload["current_status"] = ProjectStatus(payload["current_status"])
    classes = {"WorkflowTransitionEvent": WorkflowTransitionEvent, "TransitionRejectedEvent": TransitionRejectedEvent}
    return classes[event_class](**payload)
