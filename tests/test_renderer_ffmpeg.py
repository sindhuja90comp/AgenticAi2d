import json
import wave
from io import BytesIO

from PIL import Image, ImageDraw

from agentic_ai_2d.models.storyboard import MotionKind, MotionSpec
from agentic_ai_2d.models.timeline import AudioTrack, Timeline, VisualClip, VisualTrack, VisualTrackKind
from agentic_ai_2d.models.viseme_timeline import MouthShape, VisemeCue, VisemeTimeline
from agentic_ai_2d.renderer import LocalFfmpegRenderer
from agentic_ai_2d.storage import ArtifactManager


def test_local_ffmpeg_renderer_creates_a_real_mp4(tmp_path) -> None:
    storage = ArtifactManager(tmp_path / "artifacts")
    background = storage.store_bytes(_background_png(), filename="background.png", media_type="image/png")
    pose = storage.store_bytes(_pose_png(), filename="pose.png", media_type="image/png")
    mouth = storage.store_bytes(_mouth_sheet_png(), filename="mouth.png", media_type="image/png")
    audio = storage.store_bytes(_silence_wav(duration_seconds=1.0), filename="narration.wav", media_type="audio/wav")
    viseme = VisemeTimeline(
        viseme_timeline_id="viseme_12345678",
        project_id="proj_12345678",
        project_version=1,
        narration_track_id="narr_12345678",
        input_audio_asset_id=audio.asset_id,
        generator="rhubarb",
        fps=24,
        duration_frames=24,
        character_id="char_bird_v1",
        mouth_set_id="mouthset_bird_v1",
        cues=[
            VisemeCue(start_frame=0, end_frame=7, mouth_shape=MouthShape.A),
            VisemeCue(start_frame=8, end_frame=15, mouth_shape=MouthShape.C),
            VisemeCue(start_frame=16, end_frame=23, mouth_shape=MouthShape.X),
        ],
    )
    timeline = Timeline(
        timeline_id="timeline_12345678",
        project_id="proj_12345678",
        project_version=1,
        fps=24,
        width=1280,
        height=720,
        duration_frames=24,
        visual_tracks=[
            VisualTrack(
                track_id="vtrack_bg_12345678",
                kind=VisualTrackKind.BACKGROUND,
                z_index=0,
                clips=[
                    VisualClip(
                        clip_id="clip_bg_12345678",
                        asset_id=background.asset_id,
                        start_frame=0,
                        duration_frames=24,
                    )
                ],
            ),
            VisualTrack(
                track_id="vtrack_pose_12345678",
                kind=VisualTrackKind.CHARACTER_POSE,
                z_index=10,
                clips=[
                    VisualClip(
                        clip_id="clip_pose_12345678",
                        asset_id=pose.asset_id,
                        start_frame=0,
                        duration_frames=24,
                        x_percent=50,
                        y_percent=65,
                        scale=1,
                        motion=MotionSpec(
                            kind=MotionKind.BIRD_HOP,
                            amplitude_percent=8,
                            hop_count=2,
                            start_x_percent=35,
                            end_x_percent=65,
                            period_frames=24,
                        ),
                    )
                ],
            ),
            VisualTrack(
                track_id="vtrack_mouth_12345678",
                kind=VisualTrackKind.CHARACTER_MOUTH,
                z_index=11,
                clips=[
                    VisualClip(
                        clip_id="clip_mouth_12345678",
                        asset_id=mouth.asset_id,
                        start_frame=0,
                        duration_frames=24,
                        x_percent=50,
                        y_percent=65,
                        scale=1,
                        viseme_timeline_id=viseme.viseme_timeline_id,
                    )
                ],
            ),
        ],
        audio_tracks=[
            AudioTrack(
                track_id="atrack_12345678",
                kind="narration",
                asset_id=audio.asset_id,
                start_frame=0,
                duration_frames=24,
            )
        ],
    )

    result = LocalFfmpegRenderer(
        storage,
        output_root=tmp_path / "renders",
        visemes=[viseme],
    ).render(timeline)

    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["renderer"] == "local_ffmpeg"
    assert any(stream["codec_type"] == "video" for stream in manifest["media_info"]["streams"])
    assert any(stream["codec_type"] == "audio" for stream in manifest["media_info"]["streams"])


def _background_png() -> bytes:
    image = Image.new("RGBA", (1280, 720), "#d8edf2")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 520, 1280, 720), fill="#9fd36e")
    draw.ellipse((480, 180, 800, 500), fill="#ffe27a", outline="#775d20", width=8)
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _pose_png() -> bytes:
    image = Image.new("RGBA", (360, 360), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 80, 320, 300), fill="#6d8fdb", outline="#27407a", width=10)
    draw.polygon(((260, 150), (350, 180), (260, 220)), fill="#f1b04d", outline="#86540d", width=8)
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _mouth_sheet_png() -> bytes:
    image = Image.new("RGBA", (180, 630), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    colors = ["#3c1e27", "#622a35", "#8c3944", "#b44a53", "#cf6c6b", "#b44a53", "#8c3944", "#622a35", "#2c151d"]
    for index, color in enumerate(colors):
        y = index * 70
        draw.ellipse((40, y + 18, 140, y + 52), fill=color)
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _silence_wav(*, duration_seconds: float, sample_rate: int = 48_000) -> bytes:
    frames = int(round(duration_seconds * sample_rate))
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00\x00" * frames)
        return buffer.getvalue()
