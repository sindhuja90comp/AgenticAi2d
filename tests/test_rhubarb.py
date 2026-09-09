from datetime import datetime, timezone

from agentic_ai_2d.agents import (
    DeterministicTtsAgent,
    DeterministicWriterAgent,
    LocalRhubarbAdapter,
    SimulatedRhubarbAdapter,
)
from agentic_ai_2d.ingestion import StaticWhisperAdapter
from agentic_ai_2d.models.project_spec import ProjectSpec
from agentic_ai_2d.storage import ArtifactManager


def test_rhubarb_adapter_creates_ordered_viseme_alignment(tmp_path) -> None:
    storage = ArtifactManager(tmp_path)
    input_audio = storage.store_bytes(b"recorded voice", filename="instruction.wav", media_type="audio/wav")
    project = ProjectSpec.model_validate({
        "project_id": "proj_12345678", "version": 1, "status": "draft", "created_at": datetime.now(timezone.utc),
        "input": {"audio_asset_id": input_audio.asset_id, "user_prompt": "Tell a story.", "language": "en"},
        "output": {"aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": 30, "format": "mp4"},
        "creative_constraints": {"target_duration_seconds": 15, "style_preset": "storybook_2d_flat", "character_ids": ["char_rabbit_v1"], "narration_source": "synthetic_tts", "captions_enabled": True, "user_approval_required": True}
    })
    transcript = StaticWhisperAdapter("A rabbit shares", 15).transcribe(input_audio)
    background = storage.store_bytes(b"background", filename="background.png", media_type="image/png")
    pose = storage.store_bytes(b"pose", filename="pose.png", media_type="image/png")
    storyboard = DeterministicWriterAgent().generate(project, transcript, background_asset_id=background.asset_id, pose_asset_id=pose.asset_id)
    narration = DeterministicTtsAgent().generate(project, storyboard, storage)

    visemes = SimulatedRhubarbAdapter().generate(project, narration, character_id="char_rabbit_v1", mouth_set_id="mouthset_rabbit_v1")

    assert visemes.input_audio_asset_id == narration.audio_asset_id
    assert visemes.cues[0].start_frame == 0
    assert visemes.cues[-1].end_frame == narration.duration_frames - 1
    assert all(left.end_frame < right.start_frame for left, right in zip(visemes.cues, visemes.cues[1:]))
    assert {cue.mouth_shape.value for cue in visemes.cues} <= {"A", "B", "C", "D", "E", "F", "G", "H", "X"}


def test_local_rhubarb_cues_are_converted_to_non_overlapping_frame_intervals() -> None:
    cues = LocalRhubarbAdapter._to_frame_cues(
        [
            {"start": 0.0, "end": 0.1, "value": "A"},
            {"start": 0.1, "end": 0.2, "value": "B"},
            {"start": 0.2, "end": 0.3, "value": "B"},
        ],
        fps=30,
        duration_frames=30,
    )

    assert [(cue.start_frame, cue.end_frame, cue.mouth_shape.value) for cue in cues] == [
        (0, 2, "A"),
        (3, 8, "B"),
    ]
