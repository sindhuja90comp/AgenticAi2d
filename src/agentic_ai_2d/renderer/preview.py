"""Local low-resolution preview rendering and review-frame packaging."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageOps

from ..models.phase2 import PreviewPackage
from ..models.timeline import Timeline
from ..storage import ArtifactManager
from .adapter import LocalFfmpegRenderer


class PreviewRenderer:
    """Render local preview media and store immutable review artifacts."""

    def __init__(self, renderer: LocalFfmpegRenderer, storage: ArtifactManager, *, ffmpeg_binary: str = "ffmpeg") -> None:
        self._renderer = renderer
        self._storage = storage
        self._ffmpeg_binary = ffmpeg_binary

    def render(self, timeline: Timeline) -> PreviewPackage:
        preview_timeline = timeline.model_copy(update={"width": 1280, "height": 720})
        result = self._renderer.render(preview_timeline)
        preview = self._storage.store_bytes(result.output_path.read_bytes(), filename="preview.mp4", media_type="video/mp4")
        frame_numbers = [0, max(0, preview_timeline.duration_frames // 2), preview_timeline.duration_frames - 1]
        with tempfile.TemporaryDirectory(prefix="agentic-ai-2d-preview-") as directory:
            frame_paths = [Path(directory) / f"review-{index}.png" for index in range(3)]
            for frame_number, path in zip(frame_numbers, frame_paths, strict=True):
                self._extract_frame(result.output_path, frame_number, preview_timeline.fps, path)
            frame_assets = [self._storage.store_bytes(path.read_bytes(), filename=path.name, media_type="image/png") for path in frame_paths]
            sheet = self._contact_sheet(frame_paths)
        contact_sheet = self._storage.store_bytes(sheet, filename="preview-contact-sheet.png", media_type="image/png")
        digest = sha256(str(preview_timeline.model_dump(mode="json")).encode("utf-8")).hexdigest()
        return PreviewPackage(
            preview_id=f"preview_{digest[:16]}", project_id=timeline.project_id, project_version=timeline.project_version,
            timeline_digest=digest, preview_asset_id=preview.asset_id, contact_sheet_asset_id=contact_sheet.asset_id,
            review_frame_asset_ids=[frame.asset_id for frame in frame_assets], review_frame_numbers=frame_numbers,
        )

    def _extract_frame(self, video: Path, frame_number: int, fps: int, output: Path) -> None:
        completed = subprocess.run(
            [self._ffmpeg_binary, "-y", "-ss", f"{frame_number / fps:.6f}", "-i", str(video), "-frames:v", "1", str(output)],
            check=False, capture_output=True, text=True, timeout=120,
        )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(f"FFmpeg could not extract preview frame: {completed.stderr}")

    @staticmethod
    def _contact_sheet(frame_paths: list[Path]) -> bytes:
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        try:
            width, height = max(image.width for image in images), max(image.height for image in images)
            sheet = Image.new("RGB", (width * len(images), height), "white")
            for index, image in enumerate(images):
                sheet.paste(ImageOps.contain(image, (width, height)), (index * width, 0))
            with BytesIO() as buffer:
                sheet.save(buffer, format="PNG")
                return buffer.getvalue()
        finally:
            for image in images:
                image.close()
