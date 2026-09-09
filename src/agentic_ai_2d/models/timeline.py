"""Master render contract consumed by the deterministic renderer."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from .common import AssetId, ContractModel, ProjectId
from .storyboard import MotionSpec


class VisualTrackKind(StrEnum):
    BACKGROUND = "background"
    CHARACTER_POSE = "character_pose"
    CHARACTER_MOUTH = "character_mouth"
    CAPTION = "caption"


class VisualClip(ContractModel):
    clip_id: str = Field(pattern=r"^clip_[a-zA-Z0-9_-]{8,128}$")
    asset_id: AssetId
    start_frame: int = Field(ge=0)
    duration_frames: int = Field(ge=1)
    x_percent: float | None = Field(default=None, ge=0, le=100)
    y_percent: float | None = Field(default=None, ge=0, le=100)
    scale: float | None = Field(default=None, ge=0.1, le=3)
    rotation_degrees: float = Field(default=0, ge=-180, le=180)
    anchor_x_percent: float = Field(default=50, ge=0, le=100)
    anchor_y_percent: float = Field(default=50, ge=0, le=100)
    viseme_timeline_id: str | None = Field(
        default=None, pattern=r"^viseme_[a-zA-Z0-9_-]{8,128}$"
    )
    motion: MotionSpec | None = None


class VisualTrack(ContractModel):
    track_id: str = Field(pattern=r"^vtrack_[a-zA-Z0-9_-]{8,128}$")
    kind: VisualTrackKind
    z_index: int = Field(ge=0, le=100)
    clips: list[VisualClip] = Field(min_length=1)


class AudioTrack(ContractModel):
    track_id: str = Field(pattern=r"^atrack_[a-zA-Z0-9_-]{8,128}$")
    kind: Literal["narration", "sfx"]
    asset_id: AssetId
    start_frame: Literal[0]
    duration_frames: int = Field(ge=1)


class Timeline(ContractModel):
    schema_id = "https://agenticai2d.local/schemas/Timeline.schema.json"

    timeline_id: str = Field(pattern=r"^timeline_[a-zA-Z0-9_-]{8,128}$")
    project_id: ProjectId
    project_version: int = Field(ge=1)
    fps: Literal[24, 30]
    width: Literal[1280, 1920]
    height: Literal[720, 1080, 1920]
    duration_frames: int = Field(ge=1, le=1_350)
    visual_tracks: list[VisualTrack] = Field(min_length=1)
    audio_tracks: list[AudioTrack] = Field(min_length=1, max_length=1)
