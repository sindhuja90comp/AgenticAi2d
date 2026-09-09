"""Deterministic checks that run before any expensive render activity."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Set

from ..models.narration_track import NarrationTrack
from ..models.timeline import Timeline, VisualTrackKind


@dataclass(frozen=True, slots=True)
class QaResult:
    passed: bool
    failures: tuple[str, ...]


class PreRenderQaEngine:
    """Validate bounds, asset availability, and narration alignment deterministically."""

    def evaluate(
        self,
        timeline: Timeline,
        narration: NarrationTrack,
        available_asset_ids: Set[str],
    ) -> QaResult:
        failures: list[str] = []
        if timeline.project_id != narration.project_id or timeline.project_version != narration.project_version:
            failures.append("Timeline and NarrationTrack project identity must match")
        if timeline.fps != narration.fps:
            failures.append("Timeline and NarrationTrack fps must match")
        if timeline.duration_frames != narration.duration_frames:
            failures.append("Timeline and NarrationTrack duration must match")

        for track in timeline.visual_tracks:
            occupied: list[tuple[int, int]] = []
            for clip in track.clips:
                self._validate_clip(clip.clip_id, clip.asset_id, clip.start_frame, clip.duration_frames, timeline.duration_frames, available_asset_ids, failures)
                clip_end = clip.start_frame + clip.duration_frames - 1
                if any(clip.start_frame <= end and clip_end >= start for start, end in occupied):
                    failures.append(f"Visual track {track.track_id} contains overlapping clips")
                occupied.append((clip.start_frame, clip_end))
                if track.kind is VisualTrackKind.CHARACTER_MOUTH and clip.viseme_timeline_id is None:
                    failures.append(f"Mouth clip {clip.clip_id} is missing a viseme timeline")

        if len(timeline.audio_tracks) != 1:
            failures.append("Phase 1 requires exactly one audio track")
        else:
            audio = timeline.audio_tracks[0]
            self._validate_clip(audio.track_id, audio.asset_id, audio.start_frame, audio.duration_frames, timeline.duration_frames, available_asset_ids, failures)
            if audio.asset_id != narration.audio_asset_id:
                failures.append("Timeline audio asset does not match NarrationTrack")
            if audio.duration_frames != narration.duration_frames:
                failures.append("Timeline audio duration does not match NarrationTrack")

        return QaResult(passed=not failures, failures=tuple(failures))

    @staticmethod
    def _validate_clip(
        clip_id: str,
        asset_id: str,
        start_frame: int,
        duration_frames: int,
        timeline_duration: int,
        available_asset_ids: Set[str],
        failures: list[str],
    ) -> None:
        if asset_id not in available_asset_ids:
            failures.append(f"Clip {clip_id} references a missing asset")
        if start_frame + duration_frames > timeline_duration:
            failures.append(f"Clip {clip_id} exceeds timeline boundary")
