"""Deterministic workflow transitions reconstructed from append-only events."""

from __future__ import annotations

from .events.domain import (
    EventType,
    TransitionRejectedEvent,
    WorkflowTransitionEvent,
    event_metadata,
)
from .events.repository import EventRepository
from .models.project_spec import ProjectSpec, ProjectStatus


class InvalidTransitionError(ValueError):
    """Raised after an invalid transition is recorded in the event stream."""


class InvalidRevisionError(ValueError):
    """Raised when a proposed revision breaks immutable version rules."""


class WorkflowStateMachine:
    """Apply only legal transitions and materialize current status from events."""

    _TRANSITIONS: dict[tuple[ProjectStatus, EventType], ProjectStatus] = {
        (ProjectStatus.DRAFT, EventType.STORYBOARD_CREATED): ProjectStatus.AWAITING_APPROVAL,
        (ProjectStatus.DRAFT, EventType.PLANNING_STARTED): ProjectStatus.PLANNING,
        (ProjectStatus.AWAITING_APPROVAL, EventType.PROJECT_APPROVED): ProjectStatus.PLANNING,
        (ProjectStatus.PLANNING, EventType.CONTRACTS_VALIDATED): ProjectStatus.ASSET_GENERATION,
        (ProjectStatus.ASSET_GENERATION, EventType.ASSETS_CACHED_AND_VALIDATED): ProjectStatus.RENDERING,
        (ProjectStatus.ASSET_GENERATION, EventType.PREVIEW_CREATED): ProjectStatus.AWAITING_PREVIEW_REVIEW,
        (ProjectStatus.AWAITING_PREVIEW_REVIEW, EventType.PREVIEW_APPROVED): ProjectStatus.PREVIEW_APPROVED,
        (ProjectStatus.AWAITING_PREVIEW_REVIEW, EventType.PREVIEW_REJECTED): ProjectStatus.REVISION_REQUESTED,
        (ProjectStatus.PREVIEW_APPROVED, EventType.PROJECT_APPROVED): ProjectStatus.RENDERING,
        (ProjectStatus.RENDERING, EventType.RENDER_QA_PASSED): ProjectStatus.COMPLETED,
        (ProjectStatus.RENDERING, EventType.REPAIR_BUDGET_EXHAUSTED): ProjectStatus.QA_FAILED,
    }

    def __init__(self, repository: EventRepository) -> None:
        self._repository = repository

    def current_status(self, project: ProjectSpec) -> ProjectStatus:
        """Materialize the status from transition events, falling back to initial draft."""
        transitions = [
            event
            for event in self._repository.list_events(project.project_id, project.version)
            if isinstance(event, WorkflowTransitionEvent)
        ]
        if not transitions:
            return project.status
        return transitions[-1].to_status

    def transition(
        self,
        project: ProjectSpec,
        event_type: EventType,
        *,
        actor: str,
        correlation_id: str,
        idempotency_key: str,
        payload_digest: str,
    ) -> WorkflowTransitionEvent:
        """Append an accepted transition or audit and reject an illegal movement."""
        existing = self._repository.find_by_idempotency_key(idempotency_key)
        if existing is not None:
            if (
                isinstance(existing, WorkflowTransitionEvent)
                and existing.project_id == project.project_id
                and existing.project_version == project.version
                and existing.event_type is event_type
            ):
                return existing
            raise InvalidTransitionError("idempotency key is already associated with another event")

        current = self.current_status(project)
        target = self._TRANSITIONS.get((current, event_type))
        if target is None:
            self._record_rejection(
                project,
                event_type,
                current,
                actor,
                correlation_id,
                idempotency_key,
                payload_digest,
            )
            raise InvalidTransitionError(
                f"{event_type.value} is not legal from {current.value} for project version {project.version}"
            )

        event_id, occurred_at = event_metadata()
        event = WorkflowTransitionEvent(
            event_id=event_id,
            project_id=project.project_id,
            project_version=project.version,
            event_type=event_type,
            occurred_at=occurred_at,
            actor=actor,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            payload_digest=payload_digest,
            from_status=current,
            to_status=target,
        )
        stored = self._repository.append(event)
        if not isinstance(stored, WorkflowTransitionEvent):
            raise InvalidTransitionError("idempotency key belongs to a non-transition event")
        return stored

    def create_revision(
        self,
        previous: ProjectSpec,
        revision: ProjectSpec,
        *,
        actor: str,
        correlation_id: str,
        idempotency_key: str,
        payload_digest: str,
    ) -> WorkflowTransitionEvent:
        """Create a new draft version without mutating a prior project specification."""
        existing = self._repository.find_by_idempotency_key(idempotency_key)
        if existing is not None:
            if (
                isinstance(existing, WorkflowTransitionEvent)
                and existing.project_id == revision.project_id
                and existing.project_version == revision.version
                and existing.event_type is EventType.REVISION_CREATED
            ):
                return existing
            raise InvalidRevisionError("idempotency key is already associated with another event")

        previous_status = self.current_status(previous)
        if previous_status not in {ProjectStatus.AWAITING_APPROVAL, ProjectStatus.REVISION_REQUESTED, ProjectStatus.QA_FAILED}:
            raise InvalidRevisionError("revisions are only allowed while awaiting approval, after preview rejection, or after QA failure")
        if revision.project_id != previous.project_id:
            raise InvalidRevisionError("revision project_id must match its predecessor")
        if revision.version != previous.version + 1:
            raise InvalidRevisionError("revision version must be exactly one greater than its predecessor")
        if revision.parent_version != previous.version:
            raise InvalidRevisionError("revision parent_version must reference its predecessor")
        if revision.status is not ProjectStatus.DRAFT:
            raise InvalidRevisionError("a new revision must begin in draft status")
        if self._repository.list_events(revision.project_id, revision.version):
            raise InvalidRevisionError("a revision version may only be created once")

        event_id, occurred_at = event_metadata()
        event = WorkflowTransitionEvent(
            event_id=event_id,
            project_id=revision.project_id,
            project_version=revision.version,
            event_type=EventType.REVISION_CREATED,
            occurred_at=occurred_at,
            actor=actor,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            payload_digest=payload_digest,
            from_status=previous_status,
            to_status=ProjectStatus.DRAFT,
        )
        stored = self._repository.append(event)
        if not isinstance(stored, WorkflowTransitionEvent):
            raise InvalidRevisionError("idempotency key belongs to a non-transition event")
        return stored

    def _record_rejection(
        self,
        project: ProjectSpec,
        attempted_event_type: EventType,
        current_status: ProjectStatus,
        actor: str,
        correlation_id: str,
        idempotency_key: str,
        payload_digest: str,
    ) -> None:
        event_id, occurred_at = event_metadata()
        rejection = TransitionRejectedEvent(
            event_id=event_id,
            project_id=project.project_id,
            project_version=project.version,
            event_type=EventType.TRANSITION_REJECTED,
            occurred_at=occurred_at,
            actor=actor,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            payload_digest=payload_digest,
            attempted_event_type=attempted_event_type,
            current_status=current_status,
            reason=f"{attempted_event_type.value} is not legal from {current_status.value}",
        )
        self._repository.append(rejection)
