from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agentic_ai_2d.events import EventType, InMemoryEventRepository, TransitionRejectedEvent
from agentic_ai_2d.models.project_spec import ProjectSpec, ProjectStatus
from agentic_ai_2d.state_machine import (
    InvalidRevisionError,
    InvalidTransitionError,
    WorkflowStateMachine,
)


def project(version: int = 1, parent_version: int | None = None) -> ProjectSpec:
    return ProjectSpec.model_validate({
        "project_id": "proj_12345678", "version": version, "parent_version": parent_version,
        "status": "draft", "created_at": datetime.now(timezone.utc),
        "input": {"audio_asset_id": "asset_instruction_12345678", "user_prompt": "A rabbit shares.", "language": "en"},
        "output": {"aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": 30, "format": "mp4"},
        "creative_constraints": {"target_duration_seconds": 15, "style_preset": "storybook_2d_flat", "character_ids": ["char_rabbit_v1"], "narration_source": "synthetic_tts", "captions_enabled": True, "user_approval_required": True}
    })


def transition(machine: WorkflowStateMachine, spec: ProjectSpec, event_type: EventType, key: str) -> None:
    machine.transition(spec, event_type, actor="worker", correlation_id="corr_123", idempotency_key=key, payload_digest="sha256:abc")


def test_legal_transitions_materialize_current_status() -> None:
    repository = InMemoryEventRepository()
    machine = WorkflowStateMachine(repository)
    spec = project()

    transition(machine, spec, EventType.STORYBOARD_CREATED, "storyboard-1")
    transition(machine, spec, EventType.PROJECT_APPROVED, "approval-1")
    transition(machine, spec, EventType.CONTRACTS_VALIDATED, "contracts-1")
    transition(machine, spec, EventType.ASSETS_CACHED_AND_VALIDATED, "assets-1")
    transition(machine, spec, EventType.RENDER_QA_PASSED, "qa-1")

    assert machine.current_status(spec) is ProjectStatus.COMPLETED


def test_invalid_transition_is_rejected_and_recorded() -> None:
    repository = InMemoryEventRepository()
    spec = project()
    machine = WorkflowStateMachine(repository)

    with pytest.raises(InvalidTransitionError):
        transition(machine, spec, EventType.PROJECT_APPROVED, "invalid-approval")

    events = repository.list_events(spec.project_id, spec.version)
    assert len(events) == 1
    assert isinstance(events[0], TransitionRejectedEvent)


def test_event_repository_returns_existing_event_for_idempotency_key() -> None:
    repository = InMemoryEventRepository()
    machine = WorkflowStateMachine(repository)
    spec = project()

    first = machine.transition(spec, EventType.STORYBOARD_CREATED, actor="writer", correlation_id="corr", idempotency_key="same-key", payload_digest="sha256:one")
    duplicate = machine.transition(spec, EventType.STORYBOARD_CREATED, actor="writer", correlation_id="corr", idempotency_key="same-key", payload_digest="sha256:one")

    assert duplicate == first
    assert len(repository.list_events(spec.project_id, spec.version)) == 1


def test_revision_requires_a_new_immutable_version() -> None:
    repository = InMemoryEventRepository()
    machine = WorkflowStateMachine(repository)
    original = project()
    transition(machine, original, EventType.STORYBOARD_CREATED, "storyboard-1")

    revision = project(version=2, parent_version=1)
    machine.create_revision(original, revision, actor="user", correlation_id="corr", idempotency_key="revision-2", payload_digest="sha256:revision")

    assert machine.current_status(revision) is ProjectStatus.DRAFT
    with pytest.raises(InvalidRevisionError):
        machine.create_revision(original, project(version=1, parent_version=None), actor="user", correlation_id="corr", idempotency_key="bad-revision", payload_digest="sha256:bad")


def test_project_spec_cannot_be_mutated_in_place() -> None:
    spec = project()

    with pytest.raises(ValidationError):
        spec.version = 2


def test_qa_failed_project_creates_a_new_draft_revision() -> None:
    repository = InMemoryEventRepository()
    machine = WorkflowStateMachine(repository)
    original = project()
    transition(machine, original, EventType.STORYBOARD_CREATED, "storyboard-1")
    transition(machine, original, EventType.PROJECT_APPROVED, "approval-1")
    transition(machine, original, EventType.CONTRACTS_VALIDATED, "contracts-1")
    transition(machine, original, EventType.ASSETS_CACHED_AND_VALIDATED, "assets-1")
    transition(machine, original, EventType.REPAIR_BUDGET_EXHAUSTED, "qa-failed-1")

    revision = project(version=2, parent_version=1)
    machine.create_revision(original, revision, actor="user", correlation_id="corr", idempotency_key="revision-2", payload_digest="sha256:revision")

    assert machine.current_status(original) is ProjectStatus.QA_FAILED
    assert machine.current_status(revision) is ProjectStatus.DRAFT


def test_preview_gate_blocks_rendering_until_explicit_approval() -> None:
    repository = InMemoryEventRepository()
    machine = WorkflowStateMachine(repository)
    spec = project()

    transition(machine, spec, EventType.PLANNING_STARTED, "planning-1")
    transition(machine, spec, EventType.CONTRACTS_VALIDATED, "contracts-1")
    transition(machine, spec, EventType.PREVIEW_CREATED, "preview-1")

    assert machine.current_status(spec) is ProjectStatus.AWAITING_PREVIEW_REVIEW
    with pytest.raises(InvalidTransitionError):
        transition(machine, spec, EventType.RENDER_QA_PASSED, "render-before-approval")
    transition(machine, spec, EventType.PREVIEW_APPROVED, "preview-approved-1")
    transition(machine, spec, EventType.PROJECT_APPROVED, "render-approved-1")
    assert machine.current_status(spec) is ProjectStatus.RENDERING
