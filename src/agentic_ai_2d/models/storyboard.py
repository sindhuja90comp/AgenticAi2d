"""Storyboard contract emitted by the Writer Agent."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from .common import AssetId, CharacterId, ContractModel, ProjectId, UtteranceId
from .phase2 import TransformAnchor


class CameraPreset(StrEnum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    STATIC = "static"


class ActorPosition(StrEnum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class MotionKind(StrEnum):
    BUTTERFLY_FLAP = "butterfly_flap"
    FLOWER_SWAY = "flower_sway"
    BIRD_HOP = "bird_hop"
    STATIC = "static"


class MotionSpec(ContractModel):
    """Bounded Phase 1 motion parameters consumed by the deterministic renderer."""

    kind: MotionKind
    amplitude_percent: float = Field(default=0, ge=0, le=30)
    period_frames: int = Field(default=30, ge=1, le=600)
    hop_count: int = Field(default=0, ge=0, le=12)
    start_x_percent: float | None = Field(default=None, ge=0, le=100)
    end_x_percent: float | None = Field(default=None, ge=0, le=100)
    landing_surface_y_percent: float | None = Field(default=None, ge=0, le=100)
    foot_y_percent: float | None = Field(default=None, ge=0, le=100)


class Narration(ContractModel):
    utterance_id: UtteranceId
    text: str = Field(min_length=1, max_length=1_000)


class CameraCue(ContractModel):
    frame_offset: int = Field(ge=0)
    preset: CameraPreset


class ActorCue(ContractModel):
    character_id: CharacterId
    pose_asset_id: AssetId
    start_frame_offset: int = Field(ge=0)
    duration_frames: int = Field(ge=1)
    position: ActorPosition
    transform: TransformAnchor | None = None
    motion: MotionSpec = Field(default_factory=lambda: MotionSpec(kind=MotionKind.STATIC))


class Scene(ContractModel):
    scene_id: str = Field(pattern=r"^scene_[a-zA-Z0-9_-]{8,128}$")
    sequence: int = Field(ge=1)
    start_frame: int = Field(ge=0)
    duration_frames: int = Field(ge=1)
    background_asset_id: AssetId
    narration: Narration
    camera_cues: list[CameraCue] = Field(min_length=1)
    actor_cues: list[ActorCue] = Field(min_length=1, max_length=3)


class Storyboard(ContractModel):
    schema_id = "https://agenticai2d.local/schemas/Storyboard.schema.json"

    storyboard_id: str = Field(pattern=r"^story_[a-zA-Z0-9_-]{8,128}$")
    project_id: ProjectId
    project_version: int = Field(ge=1)
    fps: Literal[24, 30]
    duration_frames: int = Field(ge=1, le=1_350)
    scenes: list[Scene] = Field(min_length=1, max_length=3)

    @property
    def scene_ids(self) -> set[str]:
        return {scene.scene_id for scene in self.scenes}
