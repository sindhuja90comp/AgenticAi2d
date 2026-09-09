"""Loopback-only visual review server for persisted Phase 2 preview packages."""

from __future__ import annotations

from hashlib import sha256
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
from threading import Thread
from urllib.parse import parse_qs, unquote, urlparse

from .models.phase2 import ApprovalAction, ApprovalDecision, PreviewPackage
from .storage import ArtifactManager
from .workflow_store import ImmutableWorkflowRecordError, WorkflowStore


def build_review_server(
    store_path: Path | str, project_id: str, version: int, *, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    """Build a local server that exposes only artifacts named by one preview package."""
    storage = ArtifactManager(Path(store_path).parent)
    csrf_token = token_urlsafe(32)

    def get_record(record_type: str) -> dict | None:
        # ThreadingHTTPServer handles each request on a separate thread.
        session = WorkflowStore(store_path)
        try:
            return session.get(project_id, version, record_type)
        finally:
            session.close()

    def save_record(record_type: str, payload: ApprovalDecision) -> None:
        session = WorkflowStore(store_path)
        try:
            session.save(project_id, version, record_type, payload)
        finally:
            session.close()

    def preview_package() -> PreviewPackage:
        record = get_record("preview")
        if record is None:
            raise LookupError("no saved preview exists for this project version")
        return PreviewPackage.model_validate(record)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                try:
                    preview = preview_package()
                except (LookupError, ValueError) as error:
                    self.send_error(HTTPStatus.NOT_FOUND, str(error))
                    return
                self._send_html(_review_page(preview, csrf_token, get_record("approval_decision") is not None))
                return
            if path.startswith("/artifacts/"):
                asset_id = unquote(path.removeprefix("/artifacts/"))
                try:
                    preview = preview_package()
                    if asset_id not in {preview.preview_asset_id, preview.contact_sheet_asset_id, *preview.review_frame_asset_ids}:
                        self.send_error(HTTPStatus.NOT_FOUND, "artifact is not part of this preview")
                        return
                    metadata = storage.get_metadata(asset_id)
                    content = storage.read_bytes(metadata)
                except (LookupError, KeyError, FileNotFoundError, ValueError) as error:
                    self.send_error(HTTPStatus.NOT_FOUND, str(error))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", metadata.media_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/decision":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 8_192:
                self.send_error(HTTPStatus.BAD_REQUEST, "invalid decision payload size")
                return
            form = parse_qs(self.rfile.read(size).decode("utf-8"))
            try:
                preview = preview_package()
                action = form.get("action", [""])[0]
                if action not in {"approve", "reject"}:
                    raise ValueError("action must be approve or reject")
                if form.get("csrf_token", [""])[0] != csrf_token:
                    raise ValueError("invalid review token")
                if form.get("preview_id", [""])[0] != preview.preview_id:
                    raise ValueError("preview ID does not match the saved preview")
                if form.get("preview_digest", [""])[0] != preview.timeline_digest:
                    raise ValueError("preview digest does not match the saved preview")
                decision = _decision_for(preview, action, form.get("feedback", [""])[0].strip() or None)
                save_record("approval_decision", decision)
            except (ImmutableWorkflowRecordError, LookupError, ValueError) as error:
                self.send_error(HTTPStatus.CONFLICT, str(error))
                return
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/")
            self.end_headers()
            self.server.decision = decision  # type: ignore[attr-defined]
            Thread(target=self.server.shutdown, daemon=True).start()

        def _send_html(self, page: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; media-src 'self'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.decision = None  # type: ignore[attr-defined]
    return server


def serve_review(
    store_path: Path | str, project_id: str, version: int, *, host: str = "127.0.0.1", port: int = 8765
) -> ApprovalDecision | None:
    """Serve one review page until a digest-bound decision is submitted."""
    server = build_review_server(store_path, project_id, version, host=host, port=port)
    try:
        server.serve_forever()
        return server.decision  # type: ignore[attr-defined,no-any-return]
    finally:
        server.server_close()


def _decision_for(preview: PreviewPackage, action: str, feedback: str | None) -> ApprovalDecision:
    approval_action = ApprovalAction.APPROVED if action == "approve" else ApprovalAction.REJECTED
    fingerprint = sha256(f"{preview.project_id}:{preview.project_version}:{preview.preview_id}:{preview.timeline_digest}:{approval_action}:{feedback or ''}".encode()).hexdigest()[:16]
    return ApprovalDecision(
        approval_id=f"approval_{fingerprint}", project_id=preview.project_id, project_version=preview.project_version,
        preview_id=preview.preview_id, preview_digest=preview.timeline_digest, action=approval_action, feedback=feedback,
    )


def _review_page(preview: PreviewPackage, csrf_token: str, decision_recorded: bool) -> str:
    def artifact_url(asset_id: str) -> str:
        return f"/artifacts/{escape(asset_id, quote=True)}"

    frames = "".join(
        f"<figure><img src='{artifact_url(asset_id)}' alt='Review frame {frame}'><figcaption>Frame {frame}</figcaption></figure>"
        for asset_id, frame in zip(preview.review_frame_asset_ids, preview.review_frame_numbers, strict=True)
    )
    controls = "<p class='notice'>A decision has already been recorded for this immutable preview.</p>" if decision_recorded else f"""
<form method='post' action='/decision'><input type='hidden' name='csrf_token' value='{escape(csrf_token, quote=True)}'><input type='hidden' name='preview_id' value='{escape(preview.preview_id, quote=True)}'><input type='hidden' name='preview_digest' value='{escape(preview.timeline_digest, quote=True)}'>
<label for='feedback'>Revision feedback (required when rejecting)</label><textarea id='feedback' name='feedback' maxlength='4000'></textarea>
<div><button class='approve' name='action' value='approve'>Approve preview</button><button class='reject' name='action' value='reject'>Reject and create revision</button></div></form>"""
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Phase 2 Preview Review</title><style>body{{font-family:Georgia,serif;max-width:1100px;margin:2rem auto;background:#f7f4eb;color:#23332a}} video,img{{max-width:100%;border:1px solid #9dad93;border-radius:8px}} .frames{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}} figure{{margin:0}} textarea{{display:block;width:100%;min-height:7rem;margin:.5rem 0 1rem}} button{{padding:.7rem 1rem;margin-right:.5rem;border:0;border-radius:5px;font-weight:bold}} .approve{{background:#477a45;color:white}} .reject{{background:#9b4236;color:white}} .notice{{padding:1rem;background:#e8e2cb}}</style></head><body><h1>Preview Review</h1><p>Project {escape(preview.project_id)}, version {preview.project_version}</p><h2>Preview video</h2><video controls preload='metadata' src='{artifact_url(preview.preview_asset_id)}'></video><h2>Contact sheet</h2><img src='{artifact_url(preview.contact_sheet_asset_id)}' alt='Preview contact sheet'><h2>Review frames</h2><section class='frames'>{frames}</section><h2>Decision</h2>{controls}</body></html>"""
