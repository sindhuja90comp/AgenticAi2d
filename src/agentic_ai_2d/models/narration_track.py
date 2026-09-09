"""Narration audio and word-level frame alignment contract."""

from __future__ import annotations

from .common import AssetId, ContractModel, ProjectId, UtteranceId
from .project_spec import NarrationSource
from pydantic import Field
from typing import Literal


class WordAlignment(ContractModel):
    text: str = Field(min_length=1)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)


class Utterance(ContractModel):
    utterance_id: UtteranceId
    text: str = Field(min_length=1)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    words: list[WordAlignment] = Field(min_length=1)


class NarrationTrack(ContractModel):
    schema_id = "https://agenticai2d.local/schemas/NarrationTrack.schema.json"

    narration_track_id: str = Field(pattern=r"^narr_[a-zA-Z0-9_-]{8,128}$")
    project_id: ProjectId
    project_version: int = Field(ge=1)
    audio_asset_id: AssetId
    source: NarrationSource
    language: Literal["en"]
    sample_rate_hz: Literal[44100, 48000]
    fps: Literal[24, 30]
    duration_frames: int = Field(ge=1)
    utterances: list[Utterance] = Field(min_length=1)
