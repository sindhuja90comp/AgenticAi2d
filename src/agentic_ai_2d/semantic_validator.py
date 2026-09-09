"""Cross-document validation for immutable Phase 1 production contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass

from .models.narration_track import NarrationTrack
from .models.project_spec import ProjectSpec
from .models.storyboard import Storyboard
from .models.timeline import Timeline, VisualTrackKind
from .models.viseme_timeline import VisemeTimeline


@dataclass(frozen=True)
class MouthAssetBinding:
    """Optional catalog metadata for verifying mouth-asset compatibility."""

    character_id: str
    mouth_set_id: str


class SemanticValidationError(ValueError):
    """Raised when one or more cross-contract invariants are violated."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Semantic validation failed: " + "; ".join(errors))


class SemanticValidator:
    """Validate the production set without mutating any supplied contract."""

    def validate(
        self,
        project: ProjectSpec,
        storyboard: Storyboard,
        narration: NarrationTrack,
        visemes: Iterable[VisemeTimeline],
        timeline: Timeline,
        available_asset_ids: Set[str],
        mouth_asset_bindings: Mapping[str, MouthAssetBinding] | None = None,
    ) -> None:
        viseme_list = list(visemes)
        errors: list[str] = []

        self._validate_identity(project, storyboard, narration, viseme_list, timeline, errors)
        self._validate_project_duration(project, storyboard, timeline, errors)
        self._validate_scenes(storyboard, errors)
        self._validate_narration(storyboard, narration, errors)
        self._validate_visemes(narration, viseme_list, errors)
        self._validate_asset_references(
            project, storyboard, narration, viseme_list, available_asset_ids, errors
        )
        self._validate_timeline(
            narration,
            viseme_list,
            timeline,
            available_asset_ids,
            mouth_asset_bindings,
            errors,
        )

        if errors:
            raise SemanticValidationError(errors)

    @staticmethod
    def _validate_identity(
        project: ProjectSpec,
        storyboard: Storyboard,
        narration: NarrationTrack,
        visemes: list[VisemeTimeline],
        timeline: Timeline,
        errors: list[str],
    ) -> None:
        documents = [storyboard, narration, timeline, *visemes]
        for document in documents:
            if document.project_id != project.project_id:
                errors.append(f"{type(document).__name__} project_id does not match ProjectSpec")
            if document.project_version != project.version:
                errors.append(f"{type(document).__name__} project_version does not match ProjectSpec")
            if document.fps != project.output.fps:
                errors.append(f"{type(document).__name__} fps does not match ProjectSpec")

    @staticmethod
    def _validate_project_duration(
        project: ProjectSpec,
        storyboard: Storyboard,
        timeline: Timeline,
        errors: list[str],
    ) -> None:
        expected_duration = project.creative_constraints.target_duration_seconds * project.output.fps
        for name, duration in (("Storyboard", storyboard.duration_frames), ("Timeline", timeline.duration_frames)):
            if duration != expected_duration:
                errors.append(f"{name} duration_frames must equal target_duration_seconds * fps")

    @staticmethod
    def _validate_scenes(storyboard: Storyboard, errors: list[str]) -> None:
        previous_end = -1
        seen_ids: set[str] = set()
        seen_sequences: set[int] = set()
        for expected_sequence, scene in enumerate(storyboard.scenes, start=1):
            end_frame = scene.start_frame + scene.duration_frames - 1
            if scene.scene_id in seen_ids:
                errors.append("Storyboard scene IDs must be unique")
            if scene.sequence in seen_sequences:
                errors.append("Storyboard scene sequences must be unique")
            if scene.sequence != expected_sequence:
                errors.append("Storyboard scenes must have contiguous sequence values starting at 1")
            if scene.start_frame != previous_end + 1:
                errors.append("Storyboard scenes must be contiguous and non-overlapping")
            for cue in scene.camera_cues:
                if cue.frame_offset >= scene.duration_frames:
                    errors.append(f"Camera cue exceeds scene {scene.scene_id} duration")
            for cue in scene.actor_cues:
                if cue.start_frame_offset + cue.duration_frames > scene.duration_frames:
                    errors.append(f"Actor cue exceeds scene {scene.scene_id} duration")
            seen_ids.add(scene.scene_id)
            seen_sequences.add(scene.sequence)
            previous_end = end_frame
        if previous_end + 1 != storyboard.duration_frames:
            errors.append("Storyboard scenes must exactly cover storyboard duration")

    @staticmethod
    def _validate_narration(
        storyboard: Storyboard, narration: NarrationTrack, errors: list[str]
    ) -> None:
        expected = {scene.narration.utterance_id: scene.narration.text for scene in storyboard.scenes}
        actual = {utterance.utterance_id: utterance.text for utterance in narration.utterances}
        if len(actual) != len(narration.utterances):
            errors.append("Narration utterance IDs must be unique")
        if expected != actual:
            errors.append("Storyboard narration utterance IDs and text must exactly match NarrationTrack")

        previous_end = -1
        for utterance in narration.utterances:
            if utterance.end_frame < utterance.start_frame:
                errors.append(f"Utterance {utterance.utterance_id} ends before it starts")
            if utterance.start_frame <= previous_end:
                errors.append("Narration utterances must be ordered and non-overlapping")
            if utterance.end_frame >= narration.duration_frames:
                errors.append("Narration utterance exceeds track duration")
            word_previous_end = utterance.start_frame - 1
            for word in utterance.words:
                if word.end_frame < word.start_frame:
                    errors.append(f"Word {word.text!r} ends before it starts")
                if word.start_frame <= word_previous_end:
                    errors.append("Narration words must be ordered and non-overlapping")
                if word.start_frame < utterance.start_frame or word.end_frame > utterance.end_frame:
                    errors.append("Narration word exceeds its utterance range")
                word_previous_end = word.end_frame
            previous_end = utterance.end_frame

    @staticmethod
    def _validate_visemes(
        narration: NarrationTrack,
        visemes: list[VisemeTimeline],
        errors: list[str],
    ) -> None:
        for viseme in visemes:
            if viseme.narration_track_id != narration.narration_track_id:
                errors.append("VisemeTimeline narration_track_id does not match NarrationTrack")
            if viseme.input_audio_asset_id != narration.audio_asset_id:
                errors.append("VisemeTimeline input_audio_asset_id does not match NarrationTrack")
            if viseme.duration_frames != narration.duration_frames:
                errors.append("VisemeTimeline duration_frames does not match NarrationTrack")
            previous_end = -1
            for cue in viseme.cues:
                if cue.end_frame < cue.start_frame:
                    errors.append("Viseme cue ends before it starts")
                if cue.start_frame <= previous_end:
                    errors.append("Viseme cues must be ordered and non-overlapping")
                if cue.end_frame >= viseme.duration_frames:
                    errors.append("Viseme cue exceeds timeline duration")
                previous_end = cue.end_frame

    @staticmethod
    def _validate_asset_references(
        project: ProjectSpec,
        storyboard: Storyboard,
        narration: NarrationTrack,
        visemes: list[VisemeTimeline],
        available_asset_ids: Set[str],
        errors: list[str],
    ) -> None:
        references = [
            ("ProjectSpec input audio", project.input.audio_asset_id),
            ("NarrationTrack audio", narration.audio_asset_id),
        ]
        references.extend(
            (f"Storyboard scene {scene.scene_id} background", scene.background_asset_id)
            for scene in storyboard.scenes
        )
        references.extend(
            (f"Storyboard scene {scene.scene_id} pose", cue.pose_asset_id)
            for scene in storyboard.scenes
            for cue in scene.actor_cues
        )
        references.extend(
            ("VisemeTimeline input audio", viseme.input_audio_asset_id) for viseme in visemes
        )
        for source, asset_id in references:
            if asset_id not in available_asset_ids:
                errors.append(f"{source} asset {asset_id} is not available")

    @staticmethod
    def _validate_timeline(
        narration: NarrationTrack,
        visemes: list[VisemeTimeline],
        timeline: Timeline,
        available_asset_ids: Set[str],
        mouth_asset_bindings: Mapping[str, MouthAssetBinding] | None,
        errors: list[str],
    ) -> None:
        visemes_by_id = {viseme.viseme_timeline_id: viseme for viseme in visemes}
        if len(visemes_by_id) != len(visemes):
            errors.append("VisemeTimeline IDs must be unique")
        for track in timeline.visual_tracks:
            occupied: list[tuple[int, int]] = []
            for clip in track.clips:
                end_frame = clip.start_frame + clip.duration_frames - 1
                if end_frame >= timeline.duration_frames:
                    errors.append(f"Timeline clip {clip.clip_id} exceeds timeline duration")
                if clip.asset_id not in available_asset_ids:
                    errors.append(f"Timeline asset {clip.asset_id} is not available")
                for occupied_start, occupied_end in occupied:
                    if clip.start_frame <= occupied_end and end_frame >= occupied_start:
                        errors.append(f"Timeline track {track.track_id} contains overlapping clips")
                        break
                occupied.append((clip.start_frame, end_frame))
                if track.kind is VisualTrackKind.CHARACTER_MOUTH:
                    if clip.viseme_timeline_id is None:
                        errors.append("Character mouth clips require viseme_timeline_id")
                        continue
                    viseme = visemes_by_id.get(clip.viseme_timeline_id)
                    if viseme is None:
                        errors.append("Character mouth clip references an unknown VisemeTimeline")
                    elif mouth_asset_bindings is not None:
                        binding = mouth_asset_bindings.get(clip.asset_id)
                        if binding is None:
                            errors.append("Character mouth asset is missing catalog binding metadata")
                        elif binding.character_id != viseme.character_id or binding.mouth_set_id != viseme.mouth_set_id:
                            errors.append("Character mouth asset does not match referenced VisemeTimeline")
                elif clip.viseme_timeline_id is not None:
                    errors.append("Only character mouth clips may reference a VisemeTimeline")

        for track in timeline.audio_tracks:
            if track.asset_id not in available_asset_ids:
                errors.append(f"Timeline audio asset {track.asset_id} is not available")
            if track.asset_id != narration.audio_asset_id:
                errors.append("Timeline audio asset does not match NarrationTrack")
            if track.duration_frames != narration.duration_frames:
                errors.append("Timeline audio duration does not match NarrationTrack")
