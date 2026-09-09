"""Append-only event repository abstractions."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Protocol

from .domain import DomainEvent


class DuplicateEventIdError(ValueError):
    """Raised when an event ID is reused for a different immutable event."""


class EventRepository(Protocol):
    """Persistence boundary for an append-only, idempotent event stream."""

    def append(self, event: DomainEvent) -> DomainEvent:
        """Append an event or return the existing event for its idempotency key."""

    def list_events(self, project_id: str, project_version: int) -> Sequence[DomainEvent]:
        """Return events for one immutable project version in append order."""

    def find_by_idempotency_key(self, idempotency_key: str) -> DomainEvent | None:
        """Look up the first event published for a stable idempotency key."""


class InMemoryEventRepository:
    """Thread-safe repository used by unit tests and local development."""

    def __init__(self) -> None:
        self._events: list[DomainEvent] = []
        self._events_by_id: dict[str, DomainEvent] = {}
        self._events_by_idempotency_key: dict[str, DomainEvent] = {}
        self._lock = RLock()

    def append(self, event: DomainEvent) -> DomainEvent:
        with self._lock:
            existing_by_key = self._events_by_idempotency_key.get(event.idempotency_key)
            if existing_by_key is not None:
                return existing_by_key

            existing_by_id = self._events_by_id.get(event.event_id)
            if existing_by_id is not None:
                if existing_by_id == event:
                    return existing_by_id
                raise DuplicateEventIdError(f"event_id {event.event_id!r} is already in use")

            self._events.append(event)
            self._events_by_id[event.event_id] = event
            self._events_by_idempotency_key[event.idempotency_key] = event
            return event

    def list_events(self, project_id: str, project_version: int) -> tuple[DomainEvent, ...]:
        with self._lock:
            return tuple(
                event
                for event in self._events
                if event.project_id == project_id and event.project_version == project_version
            )

    def find_by_idempotency_key(self, idempotency_key: str) -> DomainEvent | None:
        with self._lock:
            return self._events_by_idempotency_key.get(idempotency_key)
