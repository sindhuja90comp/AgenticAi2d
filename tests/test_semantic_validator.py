from datetime import datetime, timezone

import pytest

from agentic_ai_2d.models.narration_track import NarrationTrack
from agentic_ai_2d.models.project_spec import ProjectSpec
from agentic_ai_2d.models.storyboard import Storyboard
from agentic_ai_2d.models.timeline import Timeline
from agentic_ai_2d.models.viseme_timeline import VisemeTimeline
from agentic_ai_2d.semantic_validator import SemanticValidationError, SemanticValidator


def contracts() -> tuple[ProjectSpec, Storyboard, NarrationTrack, VisemeTimeline, Timeline, set[str]]:
    project = ProjectSpec.model_validate({
        "project_id": "proj_12345678", "version": 1, "status": "draft",
        "created_at": datetime.now(timezone.utc),
        "input": {"audio_asset_id": "asset_instruction_12345678", "user_prompt": "A rabbit shares.", "language": "en"},
        "output": {"aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": 30, "format": "mp4"},
        "creative_constraints": {"target_duration_seconds": 15, "style_preset": "storybook_2d_flat", "character_ids": ["char_rabbit_v1"], "narration_source": "synthetic_tts", "captions_enabled": True, "user_approval_required": True}
    })
    storyboard = Storyboard.model_validate({
        "storyboard_id": "story_12345678", "project_id": project.project_id, "project_version": 1, "fps": 30, "duration_frames": 450,
        "scenes": [{"scene_id": "scene_12345678", "sequence": 1, "start_frame": 0, "duration_frames": 450, "background_asset_id": "asset_background_12345678", "narration": {"utterance_id": "utt_12345678", "text": "A rabbit shares a carrot."}, "camera_cues": [{"frame_offset": 0, "preset": "wide"}], "actor_cues": [{"character_id": "char_rabbit_v1", "pose_asset_id": "asset_rabbit_pose_12345678", "start_frame_offset": 0, "duration_frames": 450, "position": "center"}]}]
    })
    narration = NarrationTrack.model_validate({
        "narration_track_id": "narr_12345678", "project_id": project.project_id, "project_version": 1, "audio_asset_id": "asset_narration_12345678", "source": "synthetic_tts", "language": "en", "sample_rate_hz": 48000, "fps": 30, "duration_frames": 450,
        "utterances": [{"utterance_id": "utt_12345678", "text": "A rabbit shares a carrot.", "start_frame": 0, "end_frame": 449, "words": [{"text": "A", "start_frame": 0, "end_frame": 99}, {"text": "rabbit", "start_frame": 100, "end_frame": 199}, {"text": "shares", "start_frame": 200, "end_frame": 299}, {"text": "a", "start_frame": 300, "end_frame": 349}, {"text": "carrot.", "start_frame": 350, "end_frame": 449}]}]
    })
    viseme = VisemeTimeline.model_validate({
        "viseme_timeline_id": "viseme_12345678", "project_id": project.project_id, "project_version": 1, "narration_track_id": narration.narration_track_id, "input_audio_asset_id": narration.audio_asset_id, "generator": "rhubarb", "fps": 30, "duration_frames": 450, "character_id": "char_rabbit_v1", "mouth_set_id": "mouthset_rabbit_v1", "cues": [{"start_frame": 0, "end_frame": 224, "mouth_shape": "A"}, {"start_frame": 225, "end_frame": 449, "mouth_shape": "X"}]
    })
    timeline = Timeline.model_validate({
        "timeline_id": "timeline_12345678", "project_id": project.project_id, "project_version": 1, "fps": 30, "width": 1920, "height": 1080, "duration_frames": 450,
        "visual_tracks": [{"track_id": "vtrack_12345678", "kind": "background", "z_index": 0, "clips": [{"clip_id": "clip_background_12345678", "asset_id": "asset_background_12345678", "start_frame": 0, "duration_frames": 450}]}, {"track_id": "vtrack_mouth_12345678", "kind": "character_mouth", "z_index": 2, "clips": [{"clip_id": "clip_mouth_12345678", "asset_id": "asset_rabbit_mouth_12345678", "start_frame": 0, "duration_frames": 450, "viseme_timeline_id": viseme.viseme_timeline_id}]}],
        "audio_tracks": [{"track_id": "atrack_12345678", "kind": "narration", "asset_id": narration.audio_asset_id, "start_frame": 0, "duration_frames": 450}]
    })
    assets = {
        project.input.audio_asset_id,
        "asset_background_12345678",
        "asset_rabbit_pose_12345678",
        "asset_rabbit_mouth_12345678",
        narration.audio_asset_id,
    }
    return project, storyboard, narration, viseme, timeline, assets


def test_valid_contract_set_passes_semantic_validation() -> None:
    project, storyboard, narration, viseme, timeline, assets = contracts()
    SemanticValidator().validate(project, storyboard, narration, [viseme], timeline, assets)


def test_validator_rejects_non_contiguous_scenes() -> None:
    project, storyboard, narration, viseme, timeline, assets = contracts()
    broken = storyboard.model_copy(update={"scenes": [storyboard.scenes[0].model_copy(update={"start_frame": 1})]})
    with pytest.raises(SemanticValidationError, match="contiguous"):
        SemanticValidator().validate(project, broken, narration, [viseme], timeline, assets)


def test_validator_rejects_missing_asset_and_mismatched_utterance() -> None:
    project, storyboard, narration, viseme, timeline, _ = contracts()
    broken_narration = narration.model_copy(update={"utterances": [narration.utterances[0].model_copy(update={"text": "Different text."})]})
    with pytest.raises(SemanticValidationError) as error:
        SemanticValidator().validate(project, storyboard, broken_narration, [viseme], timeline, set())
    assert "exactly match" in str(error.value)
    assert "not available" in str(error.value)
