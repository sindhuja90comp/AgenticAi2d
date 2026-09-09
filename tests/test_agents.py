from datetime import datetime, timezone

from agentic_ai_2d.agents import DeterministicTtsAgent, DeterministicWriterAgent
from agentic_ai_2d.ingestion import LocalWhisperAdapter, StaticWhisperAdapter
from agentic_ai_2d.models.project_spec import ProjectSpec
from agentic_ai_2d.storage import ArtifactManager


def project() -> ProjectSpec:
    return ProjectSpec.model_validate({
        "project_id": "proj_12345678", "version": 1, "status": "draft", "created_at": datetime.now(timezone.utc),
        "input": {"audio_asset_id": "asset_instruction_12345678", "user_prompt": "Tell a rabbit story.", "language": "en"},
        "output": {"aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": 30, "format": "mp4"},
        "creative_constraints": {"target_duration_seconds": 15, "style_preset": "storybook_2d_flat", "character_ids": ["char_rabbit_v1"], "narration_source": "synthetic_tts", "captions_enabled": True, "user_approval_required": True}
    })


def test_transcription_writer_and_tts_emit_valid_contracts(tmp_path) -> None:
    storage = ArtifactManager(tmp_path)
    input_audio = storage.store_bytes(b"recorded voice", filename="instruction.wav", media_type="audio/wav")
    spec = project().model_copy(update={"input": project().input.model_copy(update={"audio_asset_id": input_audio.asset_id})})
    transcript = StaticWhisperAdapter("A rabbit shares a carrot", 15).transcribe(input_audio)
    background = storage.store_bytes(b"background", filename="background.png", media_type="image/png")
    pose = storage.store_bytes(b"pose", filename="pose.png", media_type="image/png")

    storyboard = DeterministicWriterAgent().generate(spec, transcript, background_asset_id=background.asset_id, pose_asset_id=pose.asset_id)
    narration = DeterministicTtsAgent().generate(spec, storyboard, storage)

    assert storyboard.duration_frames == 450
    assert storyboard.scenes[0].narration.text == transcript.text
    assert narration.duration_frames == storyboard.duration_frames
    assert narration.utterances[0].words[-1].end_frame == 449
    assert narration.audio_asset_id in storage.asset_ids


def test_local_whisper_adapter_normalizes_provider_word_timestamps(tmp_path) -> None:
    storage = ArtifactManager(tmp_path)
    audio = storage.store_bytes(b"audio", filename="instruction.wav", media_type="audio/wav")

    class FakeWhisperModel:
        def transcribe(self, *_args, **_kwargs):
            return {
                "text": "A rabbit shares.",
                "segments": [
                    {
                        "words": [
                            {"word": " A", "start": 0.0, "end": 0.2},
                            {"word": " rabbit", "start": 0.2, "end": 0.6},
                            {"word": " shares.", "start": 0.6, "end": 1.0},
                        ]
                    }
                ],
            }

    LocalWhisperAdapter._models[("test", "cpu")] = FakeWhisperModel()
    transcript = LocalWhisperAdapter(storage, model_name="test").transcribe(audio)

    assert transcript.text == "A rabbit shares."
    assert [word.text for word in transcript.words] == ["A", "rabbit", "shares."]
