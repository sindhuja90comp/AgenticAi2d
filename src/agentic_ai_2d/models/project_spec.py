"""ProjectSpec contract for a user-requested animation project."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from .common import AssetId, CharacterId, ContractModel, ProjectId


class ProjectStatus(StrEnum):
    DRAFT = "draft"
    AWAITING_APPROVAL = "awaiting_approval"
    PLANNING = "planning"
    ASSET_GENERATION = "asset_generation"
    AWAITING_PREVIEW_REVIEW = "awaiting_preview_review"
    REVISION_REQUESTED = "revision_requested"
    PREVIEW_APPROVED = "preview_approved"
    RENDERING = "rendering"
    COMPLETED = "completed"
    QA_FAILED = "qa_failed"


class NarrationSource(StrEnum):
    USER_RECORDING = "user_recording"
    SYNTHETIC_TTS = "synthetic_tts"


class InputSpec(ContractModel):
    audio_asset_id: AssetId
    user_prompt: str = Field(min_length=1, max_length=10_000)
    language: Literal["en"]


class OutputSpec(ContractModel):
    aspect_ratio: Literal["16:9", "9:16", "1:1"]
    width: Literal[1280, 1920]
    height: Literal[720, 1080, 1920]
    fps: Literal[24, 30]
    format: Literal["mp4"]


class CreativeConstraints(ContractModel):
    target_duration_seconds: int = Field(ge=15, le=45)
    style_preset: Literal["storybook_2d_flat"]
    character_ids: list[CharacterId] = Field(min_length=1, max_length=2)
    narration_source: NarrationSource
    captions_enabled: bool
    user_approval_required: Literal[True]

    @field_validator("character_ids")
    @classmethod
    def character_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("character_ids must be unique")
        return value


class ProjectSpec(ContractModel):
    """Immutable user-facing project specification."""

    schema_id = "https://agenticai2d.local/schemas/ProjectSpec.schema.json"

    project_id: ProjectId
    version: int = Field(ge=1)
    parent_version: int | None = Field(default=None, ge=1)
    status: ProjectStatus
    created_at: datetime
    input: InputSpec
    output: OutputSpec
    creative_constraints: CreativeConstraints
