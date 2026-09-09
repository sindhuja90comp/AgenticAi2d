from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agentic_ai_2d.models.project_spec import (
    CreativeConstraints,
    InputSpec,
    OutputSpec,
    ProjectSpec,
    ProjectStatus,
)
from agentic_ai_2d.models.storyboard import Storyboard


def project_payload() -> dict:
    return {
        "project_id": "proj_12345678",
        "version": 1,
        "status": "draft",
        "created_at": datetime.now(timezone.utc),
        "input": {
            "audio_asset_id": "asset_instruction_12345678",
            "user_prompt": "Tell a story about a rabbit learning to share.",
            "language": "en",
        },
        "output": {
            "aspect_ratio": "16:9",
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "format": "mp4",
        },
        "creative_constraints": {
            "target_duration_seconds": 15,
            "style_preset": "storybook_2d_flat",
            "character_ids": ["char_rabbit_v1"],
            "narration_source": "synthetic_tts",
            "captions_enabled": True,
            "user_approval_required": True,
        },
    }


def test_project_spec_accepts_valid_payload() -> None:
    project = ProjectSpec.model_validate(project_payload())

    assert project.status is ProjectStatus.DRAFT
    assert project.output.fps == 30


def test_project_spec_rejects_extra_fields_and_duplicate_characters() -> None:
    payload = project_payload()
    payload["unexpected"] = "not allowed"
    with pytest.raises(ValidationError):
        ProjectSpec.model_validate(payload)

    payload = project_payload()
    payload["creative_constraints"]["character_ids"] = ["char_rabbit_v1", "char_rabbit_v1"]
    with pytest.raises(ValidationError, match="unique"):
        ProjectSpec.model_validate(payload)


def test_project_spec_exports_draft_2020_12_schema() -> None:
    schema = ProjectSpec.draft202012_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith("ProjectSpec.schema.json")
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) >= {"project_id", "version", "status"}


def test_storyboard_rejects_invalid_fps() -> None:
    payload = {
        "storyboard_id": "story_12345678",
        "project_id": "proj_12345678",
        "project_version": 1,
        "fps": 25,
        "duration_frames": 450,
        "scenes": [],
    }
    with pytest.raises(ValidationError):
        Storyboard.model_validate(payload)
