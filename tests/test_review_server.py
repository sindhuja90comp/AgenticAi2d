from http.client import HTTPConnection
from threading import Thread
from urllib.parse import urlencode

from agentic_ai_2d.models.phase2 import PreviewPackage
from agentic_ai_2d.review_server import build_review_server
from agentic_ai_2d.storage import ArtifactManager
from agentic_ai_2d.workflow_store import WorkflowStore


def _package(tmp_path) -> tuple[PreviewPackage, ArtifactManager]:
    storage = ArtifactManager(tmp_path)
    preview_video = storage.store_bytes(b"video", filename="preview.mp4", media_type="video/mp4")
    contact_sheet = storage.store_bytes(b"sheet", filename="sheet.png", media_type="image/png")
    frames = [storage.store_bytes(f"frame-{number}".encode(), filename=f"frame-{number}.png", media_type="image/png") for number in range(3)]
    package = PreviewPackage(
        preview_id="preview_12345678", project_id="proj_12345678", project_version=1, timeline_digest="a" * 64,
        preview_asset_id=preview_video.asset_id, contact_sheet_asset_id=contact_sheet.asset_id,
        review_frame_asset_ids=[frame.asset_id for frame in frames], review_frame_numbers=[0, 10, 20],
    )
    WorkflowStore(tmp_path / "phase2-workflow.db").save(package.project_id, 1, "preview", package)
    return package, storage


def _server(tmp_path, package: PreviewPackage):
    server = build_review_server(tmp_path / "phase2-workflow.db", package.project_id, 1, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, HTTPConnection(*server.server_address)


def _token(page: str) -> str:
    return page.split("name='csrf_token' value='")[1].split("'", 1)[0]


def test_review_page_serves_preview_artifacts_and_binds_rejection(tmp_path) -> None:
    package, _storage = _package(tmp_path)
    server, thread, connection = _server(tmp_path, package)
    connection.request("GET", "/")
    response = connection.getresponse()
    page = response.read().decode()
    assert response.status == 200
    assert "<video controls" in page and "Contact sheet" in page and "Review frames" in page
    assert "<pre>" not in page and "GROQ_API_KEY" not in page
    assert response.getheader("Content-Security-Policy") is not None

    connection.request("GET", f"/artifacts/{package.preview_asset_id}")
    assert connection.getresponse().read() == b"video"
    connection.request("GET", "/artifacts/asset_not_in_preview")
    assert connection.getresponse().status == 404

    payload = urlencode({"action": "reject", "feedback": "Move the wings higher.", "csrf_token": _token(page), "preview_id": package.preview_id, "preview_digest": package.timeline_digest})
    connection.request("POST", "/decision", payload, {"Content-Type": "application/x-www-form-urlencoded"})
    assert connection.getresponse().status == 303
    thread.join(timeout=2)
    decision = WorkflowStore(tmp_path / "phase2-workflow.db").get(package.project_id, 1, "approval_decision")
    assert decision is not None
    assert decision["preview_id"] == package.preview_id and decision["preview_digest"] == package.timeline_digest
    assert decision["action"] == "rejected"
    server.server_close()


def test_review_server_rejects_stale_preview_digest(tmp_path) -> None:
    package, _storage = _package(tmp_path)
    server, thread, connection = _server(tmp_path, package)
    connection.request("GET", "/")
    page = connection.getresponse().read().decode()
    payload = urlencode({"action": "approve", "csrf_token": _token(page), "preview_id": package.preview_id, "preview_digest": "b" * 64})
    connection.request("POST", "/decision", payload, {"Content-Type": "application/x-www-form-urlencoded"})
    assert connection.getresponse().status == 409
    server.shutdown(); thread.join(timeout=2); server.server_close()
