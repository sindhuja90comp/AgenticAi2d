"""Writer adapter that turns an approved instruction into a valid Storyboard."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol

from ..ingestion import TimestampedTranscript
from ..models.project_spec import ProjectSpec
from ..assets import Scenario
from ..models.storyboard import (
    ActorCue, ActorPosition, CameraCue, CameraPreset, MotionKind, MotionSpec, Narration, Scene, Storyboard,
)


class WriterAgent(Protocol):
    def generate(
        self,
        project: ProjectSpec,
        transcript: TimestampedTranscript,
        *,
        background_asset_id: str,
        pose_asset_id: str,
    ) -> Storyboard:
        """Generate a schema-valid storyboard from spoken intent."""


class DeterministicWriterAgent:
    """A bounded local writer for tests and the initial template-driven MVP."""

    def generate(
        self,
        project: ProjectSpec,
        transcript: TimestampedTranscript,
        *,
        background_asset_id: str,
        pose_asset_id: str,
    ) -> Storyboard:
        duration_frames = project.creative_constraints.target_duration_seconds * project.output.fps
        fingerprint = self._fingerprint(project, transcript)
        narration_text = transcript.text.strip()
        utterance_id = f"utt_{fingerprint}"
        scene = Scene(
            scene_id=f"scene_{fingerprint}",
            sequence=1,
            start_frame=0,
            duration_frames=duration_frames,
            background_asset_id=background_asset_id,
            narration=Narration(utterance_id=utterance_id, text=narration_text),
            camera_cues=[CameraCue(frame_offset=0, preset=CameraPreset.WIDE)],
            actor_cues=[
                ActorCue(
                    character_id=project.creative_constraints.character_ids[0],
                    pose_asset_id=pose_asset_id,
                    start_frame_offset=0,
                    duration_frames=duration_frames,
                    position=ActorPosition.CENTER,
                )
            ],
        )
        return Storyboard(
            storyboard_id=f"story_{fingerprint}",
            project_id=project.project_id,
            project_version=project.version,
            fps=project.output.fps,
            duration_frames=duration_frames,
            scenes=[scene],
        )

    @staticmethod
    def _fingerprint(project: ProjectSpec, transcript: TimestampedTranscript) -> str:
        source = f"{project.project_id}:{project.version}:{transcript.text}".encode("utf-8")
        return sha256(source).hexdigest()[:16]


class Phase1StoryboardPlanner:
    """Natural-language planner for the two approved deterministic micro-scenes.

    A future local LLM may propose this same contract, but this bounded planner is
    deliberately deterministic and rejects prompts outside the approved scope.
    """

    def generate(
        self, project: ProjectSpec, prompt: str, *, background_asset_id: str, pose_asset_id: str,
        accent_asset_id: str | None = None,
    ) -> Storyboard:
        normalized = prompt.lower()
        if "butterfly" in normalized and "flower" in normalized:
            scenario, character, motion, position, narration = (
                Scenario.BUTTERFLY_FLOWER, "char_butterfly_v1",
                MotionSpec(kind=MotionKind.BUTTERFLY_FLAP, amplitude_percent=14, period_frames=12),
                ActorPosition.CENTER, "A butterfly gently flaps above a swaying flower.",
            )
        elif "bird" in normalized and "fence" in normalized:
            scenario, character, motion, position, narration = (
                Scenario.BIRD_FENCE, "char_bird_v1",
                MotionSpec(kind=MotionKind.BIRD_HOP, amplitude_percent=13, period_frames=45, hop_count=5, start_x_percent=20, end_x_percent=80),
                ActorPosition.LEFT, "A cheerful bird hops forward along a wooden fence.",
            )
        else:
            raise ValueError("Phase 1 supports prompts containing 'butterfly' and 'flower' or 'bird' and 'fence'")
        if project.creative_constraints.character_ids[0] != character:
            raise ValueError(f"ProjectSpec character must be {character} for {scenario.value}")
        duration_frames = project.creative_constraints.target_duration_seconds * project.output.fps
        fingerprint = sha256(f"{project.project_id}:{project.version}:{scenario.value}".encode()).hexdigest()[:16]
        return Storyboard(
            storyboard_id=f"story_{fingerprint}", project_id=project.project_id, project_version=project.version,
            fps=project.output.fps, duration_frames=duration_frames,
            scenes=[Scene(
                scene_id=f"scene_{fingerprint}", sequence=1, start_frame=0, duration_frames=duration_frames,
                background_asset_id=background_asset_id,
                narration=Narration(utterance_id=f"utt_{fingerprint}", text=narration),
                camera_cues=[CameraCue(frame_offset=0, preset=CameraPreset.WIDE)],
                actor_cues=[
                    ActorCue(character_id=character, pose_asset_id=pose_asset_id, start_frame_offset=0,
                             duration_frames=duration_frames, position=position, motion=motion),
                    *([] if accent_asset_id is None else [
                        ActorCue(character_id="char_flower_v1", pose_asset_id=accent_asset_id,
                                 start_frame_offset=0, duration_frames=duration_frames,
                                 position=ActorPosition.CENTER,
                                 motion=MotionSpec(kind=MotionKind.FLOWER_SWAY, amplitude_percent=3, period_frames=90))
                    ]),
                ],
            )],
        )
