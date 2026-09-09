"""Compile Phase 1 scene contracts into one deterministic Timeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Literal

from ..models.narration_track import NarrationTrack
from ..models.project_spec import ProjectSpec
from ..models.storyboard import ActorPosition, Storyboard
from ..models.timeline import AudioTrack, Timeline, VisualClip, VisualTrack, VisualTrackKind
from ..models.viseme_timeline import VisemeTimeline
from ..storage import ArtifactMetadata


class SceneCompilationError(ValueError):
    """Raised when cached assets cannot satisfy a valid scene contract."""


class SceneCompiler:
    """Build a Timeline using only validated contracts and cached asset references."""

    _POSITION = {
        ActorPosition.LEFT: (20.0, 70.0, 0.85),
        ActorPosition.CENTER: (50.0, 70.0, 1.0),
        ActorPosition.RIGHT: (80.0, 70.0, 0.85),
    }
    _BACKGROUND_Z_INDEX = 0
    _CHARACTER_Z_INDEX = 10

    def compile(
        self,
        project: ProjectSpec,
        storyboard: Storyboard,
        narration: NarrationTrack,
        visemes: Sequence[VisemeTimeline],
        cached_assets: Mapping[str, ArtifactMetadata],
        mouth_asset_ids: Mapping[str, str],
        *,
        audio_kind: Literal["narration", "sfx"] = "narration",
    ) -> Timeline:
        self._validate_inputs(project, storyboard, narration, visemes, cached_assets, mouth_asset_ids)
        visual_tracks = [self._background_track(storyboard)]
        visual_tracks.extend(self._character_tracks(storyboard))
        visual_tracks.extend(self._mouth_tracks(visemes, mouth_asset_ids))
        fingerprint = self._fingerprint(project, storyboard, narration, visemes)
        return Timeline(
            timeline_id=f"timeline_{fingerprint}",
            project_id=project.project_id,
            project_version=project.version,
            fps=project.output.fps,
            width=project.output.width,
            height=project.output.height,
            duration_frames=storyboard.duration_frames,
            visual_tracks=visual_tracks,
            audio_tracks=[
                AudioTrack(
                    track_id=f"atrack_{fingerprint}",
                    kind=audio_kind,
                    asset_id=narration.audio_asset_id,
                    start_frame=0,
                    duration_frames=narration.duration_frames,
                )
            ],
        )

    def _validate_inputs(
        self,
        project: ProjectSpec,
        storyboard: Storyboard,
        narration: NarrationTrack,
        visemes: Sequence[VisemeTimeline],
        cached_assets: Mapping[str, ArtifactMetadata],
        mouth_asset_ids: Mapping[str, str],
    ) -> None:
        if storyboard.project_id != project.project_id or storyboard.project_version != project.version:
            raise SceneCompilationError("Storyboard does not belong to this ProjectSpec version")
        if narration.project_id != project.project_id or narration.project_version != project.version:
            raise SceneCompilationError("NarrationTrack does not belong to this ProjectSpec version")
        if storyboard.fps != project.output.fps or narration.fps != project.output.fps:
            raise SceneCompilationError("Storyboard and NarrationTrack must match project fps")
        if narration.duration_frames != storyboard.duration_frames:
            raise SceneCompilationError("NarrationTrack duration must match Storyboard duration")
        asset_ids = set(cached_assets)
        required_assets = {narration.audio_asset_id}
        for scene in storyboard.scenes:
            required_assets.add(scene.background_asset_id)
            required_assets.update(cue.pose_asset_id for cue in scene.actor_cues)
        for viseme in visemes:
            if viseme.project_id != project.project_id or viseme.project_version != project.version:
                raise SceneCompilationError("VisemeTimeline does not belong to this ProjectSpec version")
            if viseme.narration_track_id != narration.narration_track_id:
                raise SceneCompilationError("VisemeTimeline does not match NarrationTrack")
            if viseme.duration_frames != narration.duration_frames:
                raise SceneCompilationError("VisemeTimeline duration must match NarrationTrack")
            mouth_asset_id = mouth_asset_ids.get(viseme.character_id)
            if mouth_asset_id is None:
                raise SceneCompilationError(f"No mouth asset is configured for {viseme.character_id}")
            required_assets.add(mouth_asset_id)
        missing = required_assets - asset_ids
        if missing:
            raise SceneCompilationError(f"Cached assets are missing: {', '.join(sorted(missing))}")

    def _background_track(self, storyboard: Storyboard) -> VisualTrack:
        clips = [
            VisualClip(
                clip_id=f"clip_bg_{scene.scene_id.removeprefix('scene_')}",
                asset_id=scene.background_asset_id,
                start_frame=scene.start_frame,
                duration_frames=scene.duration_frames,
            )
            for scene in storyboard.scenes
        ]
        return VisualTrack(
            track_id=f"vtrack_bg_{storyboard.storyboard_id.removeprefix('story_')}",
            kind=VisualTrackKind.BACKGROUND,
            z_index=self._BACKGROUND_Z_INDEX,
            clips=clips,
        )

    def _character_tracks(self, storyboard: Storyboard) -> list[VisualTrack]:
        tracks: list[VisualTrack] = []
        for scene in storyboard.scenes:
            for cue_index, cue in enumerate(scene.actor_cues, start=1):
                if cue.transform is None:
                    x_percent, y_percent, scale = self._POSITION[cue.position]
                    rotation_degrees, anchor_x_percent, anchor_y_percent, z_index = 0, 50, 50, self._CHARACTER_Z_INDEX
                else:
                    x_percent, y_percent, scale = cue.transform.x_percent, cue.transform.y_percent, cue.transform.scale
                    rotation_degrees = cue.transform.rotation_degrees
                    anchor_x_percent = cue.transform.anchor_x_percent
                    anchor_y_percent = cue.transform.anchor_y_percent
                    z_index = cue.transform.z_index
                tracks.append(
                    VisualTrack(
                        track_id=(
                            f"vtrack_pose_{scene.scene_id.removeprefix('scene_')}_"
                            f"{cue.character_id.removeprefix('char_')}_{cue_index:02d}"
                        ),
                        kind=VisualTrackKind.CHARACTER_POSE,
                        z_index=z_index,
                        clips=[VisualClip(
                            clip_id=(
                                f"clip_pose_{scene.scene_id.removeprefix('scene_')}_"
                                f"{cue.character_id.removeprefix('char_')}_{cue_index:02d}"
                            ),
                            asset_id=cue.pose_asset_id,
                            start_frame=scene.start_frame + cue.start_frame_offset,
                            duration_frames=cue.duration_frames,
                            x_percent=x_percent, y_percent=y_percent, scale=scale,
                            rotation_degrees=rotation_degrees, anchor_x_percent=anchor_x_percent,
                            anchor_y_percent=anchor_y_percent, motion=cue.motion,
                        )],
                    )
                )
        return tracks

    def _mouth_tracks(
        self, visemes: Sequence[VisemeTimeline], mouth_asset_ids: Mapping[str, str]
    ) -> list[VisualTrack]:
        return [
            VisualTrack(
                track_id=f"vtrack_mouth_{viseme.character_id.removeprefix('char_')}",
                kind=VisualTrackKind.CHARACTER_MOUTH,
                z_index=self._CHARACTER_Z_INDEX + index * 2 + 1,
                clips=[
                    VisualClip(
                        clip_id=f"clip_mouth_{viseme.viseme_timeline_id.removeprefix('viseme_')}",
                        asset_id=mouth_asset_ids[viseme.character_id],
                        start_frame=0,
                        duration_frames=viseme.duration_frames,
                        x_percent=50,
                        y_percent=70,
                        scale=1,
                        viseme_timeline_id=viseme.viseme_timeline_id,
                    )
                ],
            )
            for index, viseme in enumerate(sorted(visemes, key=lambda item: item.character_id))
        ]

    @staticmethod
    def _fingerprint(
        project: ProjectSpec,
        storyboard: Storyboard,
        narration: NarrationTrack,
        visemes: Sequence[VisemeTimeline],
    ) -> str:
        source = ":".join(
            [
                project.project_id,
                str(project.version),
                storyboard.storyboard_id,
                narration.narration_track_id,
                *(item.viseme_timeline_id for item in visemes),
            ]
        )
        return sha256(source.encode("utf-8")).hexdigest()[:16]
