import pytest
from pydantic import ValidationError

from agentic_ai_2d.models.phase2 import (
    ApprovalDecision,
    ApprovalAction,
    AssetRequest,
    PreviewPackage,
    RevisionRequest,
    StoryboardDraft,
)


def asset_request_payload(request_id: str, catalog_key: str, *, kind: str, parts: list[str] | None = None) -> dict:
    return {
        "asset_request_id": request_id,
        "catalog_key": catalog_key,
        "kind": kind,
        "character_id": "char_butterfly_v1" if kind == "character_part" else None,
        "required_parts": parts or [],
    }


def valid_draft_payload() -> dict:
    background = asset_request_payload("areq_background01", "catalog_meadow_v1", kind="background")
    butterfly = asset_request_payload(
        "areq_butterfly01", "catalog_butterfly_v1", kind="character_part", parts=["body", "left_wing", "right_wing"]
    )
    flower = asset_request_payload(
        "areq_flower0001", "catalog_flower_v1", kind="character_part", parts=["head", "stem"]
    )
    return {
        "storyboard_draft_id": "storydraft_butterfly01",
        "project_id": "proj_butterfly01",
        "project_version": 1,
        "transcript_asset_id": "asset_transcript_12345678",
        "narration_mode": "synthetic_tts",
        "narration_text": "A butterfly rests above a flower that sways in the breeze.",
        "asset_requests": [background, butterfly, flower],
        "scenes": [{
            "scene_id": "scene_butterfly01",
            "sequence": 1,
            "duration_frames": 360,
            "background_request_id": "areq_background01",
            "actors": [
                {
                    "character_id": "char_butterfly_v1",
                    "asset_request_id": "areq_butterfly01",
                    "transform": {"x_percent": 52, "y_percent": 42, "z_index": 20},
                    "motion": {"kind": "butterfly_wing_flap", "amplitude_percent": 14, "period_frames": 12},
                },
                {
                    "character_id": "char_flower_v1",
                    "asset_request_id": "areq_flower0001",
                    "transform": {"x_percent": 50, "y_percent": 72, "anchor_x_percent": 50, "anchor_y_percent": 100, "z_index": 10},
                    "motion": {"kind": "flower_sway", "amplitude_percent": 3, "period_frames": 90},
                },
            ],
        }],
    }


def test_storyboard_draft_accepts_bounded_asset_and_anchor_plan() -> None:
    draft = StoryboardDraft.model_validate(valid_draft_payload())

    flower = draft.scenes[0].actors[1]
    assert flower.transform.anchor_y_percent == 100
    assert flower.transform.z_index < draft.scenes[0].actors[0].transform.z_index


def test_phase2_contracts_reject_invalid_references_and_review_decisions() -> None:
    payload = valid_draft_payload()
    payload["scenes"][0]["actors"][0]["asset_request_id"] = "areq_missing0001"
    with pytest.raises(ValidationError, match="unknown asset requests"):
        StoryboardDraft.model_validate(payload)

    with pytest.raises(ValidationError, match="character_part"):
        AssetRequest.model_validate(asset_request_payload("areq_badpart01", "catalog_bad_v1", kind="character_part"))

    with pytest.raises(ValidationError, match="rejected previews require feedback"):
        ApprovalDecision.model_validate({
            "approval_id": "approval_12345678", "project_id": "proj_butterfly01", "project_version": 1,
            "preview_id": "preview_12345678", "preview_digest": "a" * 64, "action": ApprovalAction.REJECTED,
        })


def test_preview_and_revision_contracts_bind_to_a_versioned_preview() -> None:
    preview = PreviewPackage.model_validate({
        "preview_id": "preview_12345678", "project_id": "proj_butterfly01", "project_version": 1,
        "timeline_digest": "b" * 64, "preview_asset_id": "asset_preview_12345678",
        "contact_sheet_asset_id": "asset_sheet_12345678",
        "review_frame_asset_ids": ["asset_frame0001", "asset_frame0002", "asset_frame0003"],
        "review_frame_numbers": [0, 120, 359],
    })
    revision = RevisionRequest.model_validate({
        "revision_request_id": "revision_12345678", "project_id": preview.project_id,
        "source_project_version": preview.project_version, "preview_id": preview.preview_id,
        "preview_digest": "c" * 64, "feedback": "Move the butterfly above the flower.",
    })

    assert revision.preview_id == preview.preview_id


@pytest.mark.parametrize("model", [StoryboardDraft, AssetRequest, PreviewPackage, ApprovalDecision, RevisionRequest])
def test_phase2_contracts_export_draft_2020_12_schemas(model) -> None:
    schema = model.draft202012_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].startswith("https://agenticai2d.local/schemas/")
    assert schema["additionalProperties"] is False
