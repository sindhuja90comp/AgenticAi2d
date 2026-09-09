import pytest

from agentic_ai_2d.workflow_store import ImmutableWorkflowRecordError, WorkflowStore


def test_workflow_store_persists_versioned_records(tmp_path) -> None:
    path = tmp_path / "workflow.db"
    store = WorkflowStore(path)
    store.save("proj_12345678", 1, "preview", {"preview_id": "preview_12345678"})

    assert WorkflowStore(path).get("proj_12345678", 1, "preview") == {"preview_id": "preview_12345678"}


def test_workflow_records_are_immutable_but_identical_retries_are_safe(tmp_path) -> None:
    store = WorkflowStore(tmp_path / "workflow.db")
    store.save("proj_12345678", 1, "project", {"version": 1})
    store.save("proj_12345678", 1, "project", {"version": 1})

    with pytest.raises(ImmutableWorkflowRecordError):
        store.save("proj_12345678", 1, "project", {"version": 2})
