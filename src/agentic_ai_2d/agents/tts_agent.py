"""Narration adapter that creates audio assets and NarrationTrack contracts."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from math import ceil
from pathlib import Path
import subprocess
import tempfile
from typing import Protocol
import wave

from ..models.narration_track import NarrationTrack, Utterance, WordAlignment
from ..models.project_spec import NarrationSource, ProjectSpec
from ..models.storyboard import Scene, Storyboard
from ..storage import ArtifactManager, ArtifactMetadata


class TtsAgent(Protocol):
    def generate(
        self,
        project: ProjectSpec,
        storyboard: Storyboard,
        storage: ArtifactManager,
    ) -> NarrationTrack:
        """Emit the narration audio artifact and its frame-level alignment contract."""


class DeterministicTtsAgent:
    """Generates valid silent WAV placeholders with deterministic frame alignments."""

    SAMPLE_RATE_HZ = 48_000

    def generate(
        self,
        project: ProjectSpec,
        storyboard: Storyboard,
        storage: ArtifactManager,
    ) -> NarrationTrack:
        duration_frames = storyboard.duration_frames
        if project.creative_constraints.narration_source is NarrationSource.USER_RECORDING:
            audio_asset_id = project.input.audio_asset_id
        else:
            audio = self._silence_wav(duration_frames / project.output.fps)
            metadata = storage.store_bytes(
                audio, filename="narration.wav", media_type="audio/wav"
            )
            audio_asset_id = metadata.asset_id

        utterances = [self._align_scene(scene) for scene in storyboard.scenes]
        identifier_source = f"{project.project_id}:{project.version}:{audio_asset_id}".encode("utf-8")
        return NarrationTrack(
            narration_track_id=f"narr_{sha256(identifier_source).hexdigest()[:16]}",
            project_id=project.project_id,
            project_version=project.version,
            audio_asset_id=audio_asset_id,
            source=project.creative_constraints.narration_source,
            language="en",
            sample_rate_hz=self.SAMPLE_RATE_HZ,
            fps=project.output.fps,
            duration_frames=duration_frames,
            utterances=utterances,
        )

    @staticmethod
    def _align_scene(scene: Scene) -> Utterance:
        text = scene.narration.text
        words = text.split()
        frame_count = scene.duration_frames
        word_frames = max(1, ceil(frame_count / len(words)))
        alignments: list[WordAlignment] = []
        for index, word in enumerate(words):
            start = scene.start_frame + index * word_frames
            end = min(scene.start_frame + frame_count - 1, start + word_frames - 1)
            alignments.append(WordAlignment(text=word, start_frame=start, end_frame=end))
        # Extend the final word to the scene boundary to keep the narration continuous.
        alignments[-1] = alignments[-1].model_copy(
            update={"end_frame": scene.start_frame + frame_count - 1}
        )
        return Utterance(
            utterance_id=scene.narration.utterance_id,
            text=text,
            start_frame=scene.start_frame,
            end_frame=scene.start_frame + frame_count - 1,
            words=alignments,
        )

    def _silence_wav(self, duration_seconds: float) -> bytes:
        frames = int(round(duration_seconds * self.SAMPLE_RATE_HZ))
        with BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.SAMPLE_RATE_HZ)
                wav.writeframes(b"\x00\x00" * frames)
            return buffer.getvalue()


class KokoroTtsAgent(DeterministicTtsAgent):
    """Generate local Kokoro narration WAV files while preserving NarrationTrack."""

    SOURCE_SAMPLE_RATE_HZ = 24_000
    SAMPLE_RATE_HZ = 48_000

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        language_code: str = "a",
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        self._voice = voice
        self._language_code = language_code
        self._ffmpeg_binary = ffmpeg_binary
        self._pipeline: object | None = None

    def generate(
        self,
        project: ProjectSpec,
        storyboard: Storyboard,
        storage: ArtifactManager,
    ) -> NarrationTrack:
        expected_frames = storyboard.duration_frames
        if project.creative_constraints.narration_source is NarrationSource.USER_RECORDING:
            audio_asset_id = self._normalize_user_recording(project, storage)
        else:
            audio = self._synthesize_wav(
                " ".join(scene.narration.text for scene in storyboard.scenes),
                expected_frames / project.output.fps,
            )
            audio_asset_id = storage.store_bytes(
                audio, filename="narration.wav", media_type="audio/wav"
            ).asset_id

        utterances = [self._align_scene(scene) for scene in storyboard.scenes]
        identifier_source = f"{project.project_id}:{project.version}:{audio_asset_id}".encode("utf-8")
        return NarrationTrack(
            narration_track_id=f"narr_{sha256(identifier_source).hexdigest()[:16]}",
            project_id=project.project_id,
            project_version=project.version,
            audio_asset_id=audio_asset_id,
            source=project.creative_constraints.narration_source,
            language="en",
            sample_rate_hz=self.SAMPLE_RATE_HZ,
            fps=project.output.fps,
            duration_frames=expected_frames,
            utterances=utterances,
        )

    def _synthesize_wav(self, text: str, target_duration_seconds: float) -> bytes:
        try:
            import numpy as np
            import soundfile as sf
            from kokoro import KPipeline
        except ImportError as error:
            raise RuntimeError(
                "Kokoro is unavailable. Install the project's live dependencies first."
            ) from error
        if self._pipeline is None:
            self._pipeline = KPipeline(lang_code=self._language_code)
        samples = [audio for _, _, audio in self._pipeline(text, voice=self._voice)]
        if not samples:
            raise RuntimeError("Kokoro returned no audio samples")
        merged = np.concatenate(samples)
        # Kokoro emits 24 kHz audio; Phase 1 contracts require 44.1 or 48 kHz.
        merged = np.repeat(merged, self.SAMPLE_RATE_HZ // self.SOURCE_SAMPLE_RATE_HZ)
        target_samples = int(round(target_duration_seconds * self.SAMPLE_RATE_HZ))
        if len(merged) > target_samples:
            raise RuntimeError("Kokoro narration exceeds the approved storyboard duration")
        if len(merged) < target_samples:
            merged = np.pad(merged, (0, target_samples - len(merged)))
        with BytesIO() as buffer:
            sf.write(buffer, merged, self.SAMPLE_RATE_HZ, format="WAV", subtype="PCM_16")
            return buffer.getvalue()

    def _normalize_user_recording(self, project: ProjectSpec, storage: ArtifactManager) -> str:
        source = storage.get_metadata(project.input.audio_asset_id)
        with tempfile.TemporaryDirectory(prefix="agentic-ai-2d-tts-") as temp_dir:
            output_path = Path(temp_dir) / "narration.wav"
            completed = subprocess.run(
                [
                    self._ffmpeg_binary,
                    "-y",
                    "-i",
                    str(storage.absolute_path(source)),
                    "-ac",
                    "1",
                    "-ar",
                    str(self.SAMPLE_RATE_HZ),
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode != 0 or not output_path.is_file():
                raise RuntimeError(f"FFmpeg could not normalize user narration: {completed.stderr}")
            return storage.store_bytes(
                output_path.read_bytes(), filename="narration.wav", media_type="audio/wav"
            ).asset_id
