"""Schema-constrained planning and review contracts for Phase 2."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .common import AssetId, CharacterId, ContractModel, ProjectId


Digest = str
CatalogAssetKey = str


class AssetKind(StrEnum):
    BACKGROUND = "background"
    CHARACTER_PART = "character_part"
    PROP = "prop"


class NarrationMode(StrEnum):
    SYNTHETIC_TTS = "synthetic_tts"
    USER_RECORDING = "user_recording"


class DraftMotionKind(StrEnum):
    STATIC = "static"
    BUTTERFLY_WING_FLAP = "butterfly_wing_flap"
    FLOWER_SWAY = "flower_sway"
    BIRD_HOP = "bird_hop"


class ApprovalAction(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class TransformAnchor(ContractModel):
    """Canvas placement and transform pivot, expressed as percentages."""

    x_percent: float = Field(ge=0, le=100)
    y_percent: float = Field(ge=0, le=100)
    scale: float = Field(default=1, ge=0.1, le=3)
    rotation_degrees: float = Field(default=0, ge=-180, le=180)
    anchor_x_percent: float = Field(default=50, ge=0, le=100)
    anchor_y_percent: float = Field(default=50, ge=0, le=100)
    z_index: int = Field(ge=0, le=100)


class DraftMotion(ContractModel):
    kind: DraftMotionKind
    amplitude_percent: float = Field(default=0, ge=0, le=30)
    period_frames: int = Field(default=30, ge=1, le=600)


class AssetRequest(ContractModel):
    """A local LLM request for a single item from the approved asset catalog."""

    schema_id = "https://agenticai2d.local/schemas/AssetRequest.schema.json"

    asset_request_id: str = Field(pattern=r"^areq_[a-zA-Z0-9_-]{8,128}$")
    catalog_key: CatalogAssetKey = Field(pattern=r"^catalog_[a-zA-Z0-9_-]{3,128}$")
    kind: AssetKind
    character_id: CharacterId | None = None
    required_parts: list[Literal["body", "left_wing", "right_wing", "head", "stem"]] = Field(
        default_factory=list, max_length=5
    )

    @model_validator(mode="after")
    def validate_parts(self) -> "AssetRequest":
        if self.kind is AssetKind.CHARACTER_PART and not self.required_parts:
            raise ValueError("character_part asset requests require at least one part")
        if self.kind is not AssetKind.CHARACTER_PART and self.required_parts:
            raise ValueError("only character_part asset requests may include required_parts")
        return self


class DraftActor(ContractModel):
    character_id: CharacterId
    asset_request_id: str = Field(pattern=r"^areq_[a-zA-Z0-9_-]{8,128}$")
    transform: TransformAnchor
    motion: DraftMotion = Field(default_factory=lambda: DraftMotion(kind=DraftMotionKind.STATIC))


class DraftScene(ContractModel):
    scene_id: str = Field(pattern=r"^scene_[a-zA-Z0-9_-]{8,128}$")
    sequence: int = Field(ge=1, le=3)
    duration_frames: int = Field(ge=1, le=1_350)
    background_request_id: str = Field(pattern=r"^areq_[a-zA-Z0-9_-]{8,128}$")
    actors: list[DraftActor] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def actor_layers_are_unique(self) -> "DraftScene":
        layers = [actor.transform.z_index for actor in self.actors]
        if len(layers) != len(set(layers)):
            raise ValueError("draft actor z_index values must be unique within a scene")
        return self


class BirdFenceGeometry(ContractModel):
    """Editable geometry in the asset canvas's 1920 by 1080 pixel coordinates.

    Rod bottom equals pole top; bird landing height is derived from rod top.
    Keeping these dependent coordinates derived prevents gaps and intersections.
    """

    pole_x_positions: list[int] = Field(default_factory=lambda: list(range(80, 1880, 250)), min_length=1, max_length=24)
    pole_top_y: int = Field(default=790, ge=1, le=1079)
    pole_bottom_y: int = Field(default=1030, ge=1, le=1080)
    pole_width: int = Field(default=55, ge=1, le=200)
    rod_left_x: int = Field(default=0, ge=0, le=1919)
    rod_right_x: int = Field(default=1920, ge=1, le=1920)
    rod_thickness: int = Field(default=100, ge=1, le=200)
    bird_scale: float = Field(default=0.68, ge=0.1, le=1.5)
    hop_start_x_percent: float = Field(default=38, ge=0, le=100)
    hop_end_x_percent: float = Field(default=62, ge=0, le=100)
    hop_height: float = Field(default=54, ge=0, le=324)
    hop_period_frames: int = Field(default=24, ge=1, le=600)
    hop_count: int = Field(default=4, ge=1, le=12)

    @property
    def rod_top_y(self) -> int:
        return self.pole_top_y - self.rod_thickness

    @model_validator(mode="after")
    def aligned_geometry(self) -> "BirdFenceGeometry":
        if not 0 <= self.rod_top_y < self.pole_top_y < self.pole_bottom_y:
            raise ValueError("rod must fit above the poles and poles must extend downward")
        if self.rod_left_x >= self.rod_right_x:
            raise ValueError("rod left must be before rod right")
        if len(set(self.pole_x_positions)) != len(self.pole_x_positions):
            raise ValueError("pole positions must be distinct")
        if any(x < self.rod_left_x or x + self.pole_width > self.rod_right_x for x in self.pole_x_positions):
            raise ValueError("every pole must fit underneath the rod")
        ordered = sorted(self.pole_x_positions)
        if any(b < a + self.pole_width for a, b in zip(ordered, ordered[1:])):
            raise ValueError("poles must not overlap")
        # Conservative full sprite bounds also keep feet on the rod throughout travel.
        size = round(1920 * 0.35 * self.bird_scale)
        for x in (self.hop_start_x_percent, self.hop_end_x_percent):
            if x * 19.2 - size / 2 < self.rod_left_x or x * 19.2 + size / 2 > self.rod_right_x:
                raise ValueError("bird hop endpoints must fit on the rod")
        if self.rod_top_y - size - self.hop_height < 0:
            raise ValueError("bird and hop must fit above the rod within the canvas")
        return self


class StoryboardDraft(ContractModel):
    """Validated LLM plan before assets are resolved or a preview is rendered."""

    schema_id = "https://agenticai2d.local/schemas/StoryboardDraft.schema.json"

    storyboard_draft_id: str = Field(pattern=r"^storydraft_[a-zA-Z0-9_-]{8,128}$")
    project_id: ProjectId
    project_version: int = Field(ge=1)
    transcript_asset_id: AssetId
    narration_mode: NarrationMode
    narration_text: str | None = Field(default=None, min_length=1, max_length=1_000)
    asset_requests: list[AssetRequest] = Field(min_length=1, max_length=32)
    scenes: list[DraftScene] = Field(min_length=1, max_length=3)
    bird_fence_geometry: BirdFenceGeometry | None = None

    @model_validator(mode="after")
    def validate_references_and_narration(self) -> "StoryboardDraft":
        request_ids = [request.asset_request_id for request in self.asset_requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("asset request IDs must be unique")
        if self.narration_mode is NarrationMode.SYNTHETIC_TTS and self.narration_text is None:
            raise ValueError("synthetic narration requires narration_text")
        if self.narration_mode is NarrationMode.USER_RECORDING and self.narration_text is not None:
            raise ValueError("user recording narration must not provide replacement narration_text")
        request_id_set = set(request_ids)
        scene_ids = [scene.scene_id for scene in self.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene IDs must be unique")
        for scene in self.scenes:
            if self.bird_fence_geometry is not None:
                geometry = self.bird_fence_geometry
                if geometry.hop_count * geometry.hop_period_frames >= scene.duration_frames:
                    raise ValueError("scene must include every complete hop and a final landing frame")
            references = [scene.background_request_id, *(actor.asset_request_id for actor in scene.actors)]
            missing = set(references) - request_id_set
            if missing:
                raise ValueError(f"scene references unknown asset requests: {', '.join(sorted(missing))}")
        return self


class PreviewPackage(ContractModel):
    schema_id = "https://agenticai2d.local/schemas/PreviewPackage.schema.json"

    preview_id: str = Field(pattern=r"^preview_[a-zA-Z0-9_-]{8,128}$")
    project_id: ProjectId
    project_version: int = Field(ge=1)
    timeline_digest: Digest = Field(pattern=r"^[a-f0-9]{64}$")
    preview_asset_id: AssetId
    contact_sheet_asset_id: AssetId
    review_frame_asset_ids: list[AssetId] = Field(min_length=3, max_length=3)
    review_frame_numbers: list[int] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def review_frames_are_ordered(self) -> "PreviewPackage":
        if self.review_frame_numbers != sorted(self.review_frame_numbers):
            raise ValueError("review frame numbers must be ordered")
        if len(self.review_frame_numbers) != len(set(self.review_frame_numbers)):
            raise ValueError("review frame numbers must be unique")
        return self


class ApprovalDecision(ContractModel):
    schema_id = "https://agenticai2d.local/schemas/ApprovalDecision.schema.json"

    approval_id: str = Field(pattern=r"^approval_[a-zA-Z0-9_-]{8,128}$")
    project_id: ProjectId
    project_version: int = Field(ge=1)
    preview_id: str = Field(pattern=r"^preview_[a-zA-Z0-9_-]{8,128}$")
    preview_digest: Digest = Field(pattern=r"^[a-f0-9]{64}$")
    action: ApprovalAction
    feedback: str | None = Field(default=None, min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def rejected_decisions_require_feedback(self) -> "ApprovalDecision":
        if self.action is ApprovalAction.REJECTED and self.feedback is None:
            raise ValueError("rejected previews require feedback")
        return self


class RevisionRequest(ContractModel):
    schema_id = "https://agenticai2d.local/schemas/RevisionRequest.schema.json"

    revision_request_id: str = Field(pattern=r"^revision_[a-zA-Z0-9_-]{8,128}$")
    project_id: ProjectId
    source_project_version: int = Field(ge=1)
    preview_id: str = Field(pattern=r"^preview_[a-zA-Z0-9_-]{8,128}$")
    preview_digest: Digest = Field(pattern=r"^[a-f0-9]{64}$")
    feedback: str = Field(min_length=1, max_length=4_000)
