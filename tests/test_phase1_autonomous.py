from datetime import datetime, timezone

from agentic_ai_2d.agents import Phase1StoryboardPlanner
from agentic_ai_2d.assets import Phase1AssetGenerator, Scenario
from agentic_ai_2d.models.project_spec import ProjectSpec
from agentic_ai_2d.models.storyboard import MotionKind


def project(character_ids: list[str]) -> ProjectSpec:
    return ProjectSpec.model_validate({
        "project_id": "proj_autonomous01", "version": 1, "status": "draft", "created_at": datetime.now(timezone.utc),
        "input": {"audio_asset_id": "asset_instruction_12345678", "user_prompt": "scene", "language": "en"},
        "output": {"aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": 30, "format": "mp4"},
        "creative_constraints": {"target_duration_seconds": 15, "style_preset": "storybook_2d_flat", "character_ids": character_ids, "narration_source": "synthetic_tts", "captions_enabled": True, "user_approval_required": True},
    })


def test_asset_generator_creates_transparent_phase1_layers(tmp_path) -> None:
    pack = Phase1AssetGenerator(tmp_path).generate(Scenario.BUTTERFLY_FLOWER)
    assert pack.background.is_file() and pack.subject.is_file() and pack.mouth_sheet.is_file()
    assert pack.accent is not None and pack.accent.is_file()


def test_bird_asset_pack_contains_separate_body_and_wing_layers(tmp_path) -> None:
    pack = Phase1AssetGenerator(tmp_path).generate(Scenario.BIRD_FENCE)
    assert pack.parts is not None
    assert all(pack.parts[part].is_file() for part in ("body", "left_wing", "right_wing"))


def test_planner_encodes_butterfly_and_flower_motion() -> None:
    storyboard = Phase1StoryboardPlanner().generate(
        project(["char_butterfly_v1", "char_flower_v1"]), "A butterfly visits a flower.",
        background_asset_id="asset_background_12345678", pose_asset_id="asset_butterfly_12345678",
        accent_asset_id="asset_flower_12345678",
    )
    assert [cue.motion.kind for cue in storyboard.scenes[0].actor_cues] == [MotionKind.BUTTERFLY_FLAP, MotionKind.FLOWER_SWAY]


def test_planner_encodes_bird_hop_path() -> None:
    storyboard = Phase1StoryboardPlanner().generate(
        project(["char_bird_v1"]), "A bird hops along a wooden fence.",
        background_asset_id="asset_background_12345678", pose_asset_id="asset_bird_12345678",
    )
    motion = storyboard.scenes[0].actor_cues[0].motion
    assert motion.kind is MotionKind.BIRD_HOP
    assert (motion.start_x_percent, motion.end_x_percent, motion.hop_count) == (20, 80, 5)
