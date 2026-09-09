"""Typed, append-only event contracts for workflow state reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from ..models.project_spec import ProjectStatus


class EventType(StrEnum):
    STORYBOARD_CREATED = "storyboard_created"
    PLANNING_STARTED = "planning_started"
    PROJECT_APPROVED = "project_approved"
    CONTRACTS_VALIDATED = "contracts_validated"
    ASSETS_CACHED_AND_VALIDATED = "assets_cached_and_validated"
    PREVIEW_CREATED = "preview_created"
    PREVIEW_APPROVED = "preview_approved"
    PREVIEW_REJECTED = "preview_rejected"
    RENDER_QA_PASSED = "render_qa_passed"
    REPAIR_BUDGET_EXHAUSTED = "repair_budget_exhausted"
    REVISION_CREATED = "revision_created"
    ARTIFACT_PUBLISHED = "artifact_published"
    TRANSITION_REJECTED = "transition_rejected"


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Common metadata required for every event stored in the event stream."""

    event_id: str
    project_id: str
    project_version: int
    event_type: EventType
    occurred_at: datetime
    actor: str
    correlation_id: str
    idempotency_key: str
    payload_digest: str

    def __post_init__(self) -> None:
        required_values = {
            "event_id": self.event_id,
            "project_id": self.project_id,
            "actor": self.actor,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "payload_digest": self.payload_digest,
        }
        if self.project_version < 1:
            raise ValueError("project_version must be at least 1")
        if any(not value for value in required_values.values()):
            raise ValueError("event metadata fields must not be empty")
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class WorkflowTransitionEvent(DomainEvent):
    """Records an accepted workflow-state transition."""

    from_status: ProjectStatus | None
    to_status: ProjectStatus


@dataclass(frozen=True, slots=True)
class ArtifactPublishedEvent(DomainEvent):
    """Records a durable artifact reference without embedding binary content."""

    artifact_id: str
    artifact_digest: str


@dataclass(frozen=True, slots=True)
class TransitionRejectedEvent(DomainEvent):
    """Records a rejected transition attempt for auditability."""

    attempted_event_type: EventType
    current_status: ProjectStatus
    reason: str


def event_metadata() -> tuple[str, datetime]:
    """Create a unique event ID and a timezone-aware UTC timestamp."""
    return str(uuid4()), datetime.now(timezone.utc)
