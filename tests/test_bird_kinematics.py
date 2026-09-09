from datetime import datetime, timezone

from agentic_ai_2d.compiler import SceneCompiler
from agentic_ai_2d.models.narration_track import NarrationTrack, Utterance, WordAlignment
from agentic_ai_2d.models.phase2 import TransformAnchor
from agentic_ai_2d.models.project_spec import ProjectSpec
from agentic_ai_2d.models.storyboard import ActorCue, ActorPosition, CameraCue, CameraPreset, MotionKind, MotionSpec, Narration, Scene, Storyboard
from agentic_ai_2d.storage import ArtifactManager


def test_compiler_keeps_multipart_wing_pivots_layers_and_shared_hop_motion(tmp_path) -> None:
    storage = ArtifactManager(tmp_path / "artifacts")
    background = storage.store_bytes(b"background", filename="fence.png", media_type="image/png")
    body = storage.store_bytes(b"body", filename="body.png", media_type="image/png")
    left_wing = storage.store_bytes(b"left", filename="left.png", media_type="image/png")
    right_wing = storage.store_bytes(b"right", filename="right.png", media_type="image/png")
    audio = storage.store_bytes(b"audio", filename="narration.wav", media_type="audio/wav")
    project = ProjectSpec.model_validate({
        "project_id": "proj_birdkin01", "version": 1, "status": "draft", "created_at": datetime.now(timezone.utc),
        "input": {"audio_asset_id": audio.asset_id, "user_prompt": "Bird flaps.", "language": "en"},
        "output": {"aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": 30, "format": "mp4"},
        "creative_constraints": {"target_duration_seconds": 15, "style_preset": "storybook_2d_flat", "character_ids": ["char_bird_v1"], "narration_source": "synthetic_tts", "captions_enabled": True, "user_approval_required": True},
    })
    body_transform = TransformAnchor(x_percent=48, y_percent=62, scale=0.8, z_index=20)
    left_transform = TransformAnchor(x_percent=48, y_percent=62, scale=0.8, anchor_x_percent=36.3, anchor_y_percent=40.3, z_index=18)
    right_transform = TransformAnchor(x_percent=48, y_percent=62, scale=0.8, anchor_x_percent=63.7, anchor_y_percent=40.3, z_index=19)
    cues = [
        ActorCue(character_id="char_bird_v1", pose_asset_id=body.asset_id, start_frame_offset=0, duration_frames=450, position=ActorPosition.CENTER, transform=body_transform),
        ActorCue(character_id="char_bird_v1", pose_asset_id=left_wing.asset_id, start_frame_offset=0, duration_frames=450, position=ActorPosition.CENTER, transform=left_transform, motion=MotionSpec(kind=MotionKind.BIRD_HOP, amplitude_percent=5, period_frames=24, hop_count=4, start_x_percent=32, end_x_percent=56)),
        ActorCue(character_id="char_bird_v1", pose_asset_id=right_wing.asset_id, start_frame_offset=0, duration_frames=450, position=ActorPosition.CENTER, transform=right_transform, motion=MotionSpec(kind=MotionKind.BIRD_HOP, amplitude_percent=5, period_frames=24, hop_count=4, start_x_percent=44, end_x_percent=68)),
    ]
    storyboard = Storyboard(
        storyboard_id="story_birdkin01", project_id=project.project_id, project_version=1, fps=30, duration_frames=450,
        scenes=[Scene(scene_id="scene_birdkin01", sequence=1, start_frame=0, duration_frames=450, background_asset_id=background.asset_id, narration=Narration(utterance_id="utt_birdkin01", text="A bird flaps."), camera_cues=[CameraCue(frame_offset=0, preset=CameraPreset.WIDE)], actor_cues=cues)],
    )
    narration = NarrationTrack(narration_track_id="narr_birdkin01", project_id=project.project_id, project_version=1, audio_asset_id=audio.asset_id, source="synthetic_tts", language="en", sample_rate_hz=48000, fps=30, duration_frames=450, utterances=[Utterance(utterance_id="utt_birdkin01", text="A bird flaps.", start_frame=0, end_frame=449, words=[WordAlignment(text="bird", start_frame=0, end_frame=449)])])

    timeline = SceneCompiler().compile(project, storyboard, narration, [], {asset.asset_id: asset for asset in (background, body, left_wing, right_wing, audio)}, {})
    tracks = sorted((track for track in timeline.visual_tracks if track.kind.value == "character_pose"), key=lambda track: track.z_index)

    assert len({track.track_id for track in tracks}) == 3
    assert [track.z_index for track in tracks] == [18, 19, 20]
    assert [(track.clips[0].anchor_x_percent, track.clips[0].anchor_y_percent) for track in tracks[:2]] == [(36.3, 40.3), (63.7, 40.3)]
    assert all(track.clips[0].motion.kind is MotionKind.BIRD_HOP for track in tracks[:2])
    assert all((track.clips[0].x_percent, track.clips[0].y_percent, track.clips[0].scale) == (48, 62, 0.8) for track in tracks)
