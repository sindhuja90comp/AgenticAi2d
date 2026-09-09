from datetime import datetime, timezone

from agentic_ai_2d.events import EventType
from agentic_ai_2d.events.sqlite_repository import SqliteEventRepository
from agentic_ai_2d.models.project_spec import ProjectSpec, ProjectStatus
from agentic_ai_2d.state_machine import WorkflowStateMachine
from agentic_ai_2d.workflow_store import WorkflowStore


def _project(version: int = 1, parent_version: int | None = None) -> ProjectSpec:
    return ProjectSpec.model_validate({
        "project_id": "proj_12345678", "version": version, "parent_version": parent_version,
        "status": "draft", "created_at": datetime.now(timezone.utc),
        "input": {"audio_asset_id": "asset_instruction_12345678", "user_prompt": "Bird hops.", "language": "en"},
        "output": {"aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": 30, "format": "mp4"},
        "creative_constraints": {"target_duration_seconds": 15, "style_preset": "storybook_2d_flat", "character_ids": ["char_bird_v1"], "narration_source": "synthetic_tts", "captions_enabled": True, "user_approval_required": True},
    })


def _transition(machine: WorkflowStateMachine, project: ProjectSpec, event: EventType, key: str) -> None:
    machine.transition(project, event, actor="test", correlation_id="corr", idempotency_key=key, payload_digest="a" * 64)


def test_rejected_v1_and_v2_preview_survive_restart_and_can_be_approved(tmp_path) -> None:
    path = tmp_path / "phase2-workflow.db"
    records = WorkflowStore(path)
    v1 = _project()
    records.save(v1.project_id, 1, "project", v1)
    for record_type in ("transcript", "storyboard_draft", "preview", "approval_decision", "revision_request"):
        records.save(v1.project_id, 1, record_type, {"record_type": record_type})

    machine = WorkflowStateMachine(SqliteEventRepository(path))
    _transition(machine, v1, EventType.PLANNING_STARTED, "v1-planning")
    _transition(machine, v1, EventType.CONTRACTS_VALIDATED, "v1-contracts")
    _transition(machine, v1, EventType.PREVIEW_CREATED, "v1-preview")
    _transition(machine, v1, EventType.PREVIEW_REJECTED, "v1-rejected")

    v2 = _project(version=2, parent_version=1)
    records.save(v2.project_id, 2, "project", v2)
    machine.create_revision(v1, v2, actor="test", correlation_id="corr", idempotency_key="v2-created", payload_digest="b" * 64)
    _transition(machine, v2, EventType.PLANNING_STARTED, "v2-planning")
    _transition(machine, v2, EventType.CONTRACTS_VALIDATED, "v2-contracts")
    _transition(machine, v2, EventType.PREVIEW_CREATED, "v2-preview")
    records.save(v2.project_id, 2, "transcript", {"record_type": "transcript"})
    records.save(v2.project_id, 2, "storyboard_draft", {"record_type": "storyboard_draft"})
    records.save(v2.project_id, 2, "preview", {"record_type": "preview"})
    records.close()

    reopened_records = WorkflowStore(path)
    reopened_machine = WorkflowStateMachine(SqliteEventRepository(path))
    assert reopened_records.get(v2.project_id, 2, "preview") == {"record_type": "preview"}
    assert reopened_machine.current_status(v2) is ProjectStatus.AWAITING_PREVIEW_REVIEW

    _transition(reopened_machine, v2, EventType.PREVIEW_APPROVED, "v2-approved")
    _transition(reopened_machine, v2, EventType.PROJECT_APPROVED, "v2-rendering")
    assert reopened_machine.current_status(v2) is ProjectStatus.RENDERING
