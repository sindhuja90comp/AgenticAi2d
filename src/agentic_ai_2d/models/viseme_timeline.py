"""Rhubarb mouth-shape timeline normalized to output frames."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from .common import AssetId, CharacterId, ContractModel, MouthSetId, ProjectId


class MouthShape(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    X = "X"


class VisemeCue(ContractModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    mouth_shape: MouthShape


class VisemeTimeline(ContractModel):
    schema_id = "https://agenticai2d.local/schemas/VisemeTimeline.schema.json"

    viseme_timeline_id: str = Field(pattern=r"^viseme_[a-zA-Z0-9_-]{8,128}$")
    project_id: ProjectId
    project_version: int = Field(ge=1)
    narration_track_id: str = Field(pattern=r"^narr_[a-zA-Z0-9_-]{8,128}$")
    input_audio_asset_id: AssetId
    generator: Literal["rhubarb"]
    fps: Literal[24, 30]
    duration_frames: int = Field(ge=1)
    character_id: CharacterId
    mouth_set_id: MouthSetId
    cues: list[VisemeCue] = Field(min_length=1)
