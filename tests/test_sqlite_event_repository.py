from datetime import datetime, timezone

from agentic_ai_2d.events import EventType
from agentic_ai_2d.events.domain import WorkflowTransitionEvent
from agentic_ai_2d.events.sqlite_repository import SqliteEventRepository
from agentic_ai_2d.models.project_spec import ProjectStatus


def test_sqlite_events_survive_a_restart(tmp_path) -> None:
    path = tmp_path / "phase2-workflow.db"
    event = WorkflowTransitionEvent(
        event_id="event_123", project_id="proj_12345678", project_version=1,
        event_type=EventType.PREVIEW_CREATED, occurred_at=datetime.now(timezone.utc),
        actor="preview", correlation_id="corr", idempotency_key="preview-1", payload_digest="a" * 64,
        from_status=ProjectStatus.ASSET_GENERATION, to_status=ProjectStatus.AWAITING_PREVIEW_REVIEW,
    )
    repository = SqliteEventRepository(path)
    repository.append(event)
    repository.close()

    reopened = SqliteEventRepository(path)
    restored = reopened.find_by_idempotency_key("preview-1")

    assert restored == event
    assert reopened.list_events("proj_12345678", 1) == (event,)
