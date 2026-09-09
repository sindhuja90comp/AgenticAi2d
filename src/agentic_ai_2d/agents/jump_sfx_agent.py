"""Create local, frame-aligned sound effects for the bird hop rig."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from math import pi, sin
import struct
import wave

from ..models.narration_track import NarrationTrack, Utterance, WordAlignment
from ..models.project_spec import NarrationSource, ProjectSpec
from ..models.storyboard import MotionKind, Scene, Storyboard
from ..storage import ArtifactManager


class JumpSfxAgent:
    """Generate short local sounds at the shared bird rig's hop starts."""

    SAMPLE_RATE_HZ = 48_000
    HOP_START_FRAMES = (0, 24, 48, 72)
    EFFECT_DURATION_SECONDS = 0.18

    def generate(
        self,
        project: ProjectSpec,
        storyboard: Storyboard,
        storage: ArtifactManager,
    ) -> NarrationTrack:
        starts = set()
        for scene in storyboard.scenes:
            for cue in scene.actor_cues:
                motion = cue.motion
                if motion.kind is MotionKind.BIRD_HOP and motion.hop_count:
                    frames = min(cue.duration_frames - 1, motion.period_frames * motion.hop_count)
                    starts.update(scene.start_frame + cue.start_frame_offset + i * frames / motion.hop_count
                                  for i in range(motion.hop_count))
        audio = self._jump_effects_wav(storyboard.duration_frames / project.output.fps, project.output.fps,
                                       hop_start_frames=sorted(starts))
        metadata = storage.store_bytes(audio, filename="bird_jump_effects.wav", media_type="audio/wav")
        identifier = sha256(f"{project.project_id}:{project.version}:{metadata.asset_id}".encode("utf-8")).hexdigest()[:16]
        return NarrationTrack(
            narration_track_id=f"narr_{identifier}",
            project_id=project.project_id,
            project_version=project.version,
            audio_asset_id=metadata.asset_id,
            # The existing track contract uses this source enum. This track contains no speech.
            source=NarrationSource.SYNTHETIC_TTS,
            language="en",
            sample_rate_hz=self.SAMPLE_RATE_HZ,
            fps=project.output.fps,
            duration_frames=storyboard.duration_frames,
            # Retain text alignment metadata required by the immutable storyboard contract.
            # It is never passed to a speech synthesizer in this agent.
            utterances=[self._align_scene(scene) for scene in storyboard.scenes],
        )

    @staticmethod
    def _align_scene(scene: Scene) -> Utterance:
        words = scene.narration.text.split()
        if len(words) > scene.duration_frames:
            raise ValueError("scene needs at least one frame per narration word")
        alignments = [
            WordAlignment(
                text=word,
                start_frame=scene.start_frame + index * scene.duration_frames // len(words),
                end_frame=scene.start_frame + (index + 1) * scene.duration_frames // len(words) - 1,
            )
            for index, word in enumerate(words)
        ]
        alignments[-1] = alignments[-1].model_copy(
            update={"end_frame": scene.start_frame + scene.duration_frames - 1}
        )
        return Utterance(
            utterance_id=scene.narration.utterance_id,
            text=scene.narration.text,
            start_frame=scene.start_frame,
            end_frame=scene.start_frame + scene.duration_frames - 1,
            words=alignments,
        )

    def _jump_effects_wav(self, duration_seconds: float, fps: int, *, hop_start_frames=None) -> bytes:
        frame_count = int(round(duration_seconds * self.SAMPLE_RATE_HZ))
        samples = [0.0] * frame_count
        effect_samples = int(self.EFFECT_DURATION_SECONDS * self.SAMPLE_RATE_HZ)
        for hop_frame in self.HOP_START_FRAMES if hop_start_frames is None else hop_start_frames:
            start = int(round(hop_frame / fps * self.SAMPLE_RATE_HZ))
            for index in range(effect_samples):
                target = start + index
                if target >= frame_count:
                    break
                progress = index / effect_samples
                envelope = (1.0 - progress) ** 2
                frequency = 620 - 280 * progress
                # A small click plus a falling tone reads as a light take-off hop.
                click = 0.18 * (1.0 - progress) if index < 180 else 0.0
                samples[target] += 0.28 * envelope * sin(2 * pi * frequency * index / self.SAMPLE_RATE_HZ) + click
        pcm = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)) for sample in samples)
        with BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.SAMPLE_RATE_HZ)
                wav.writeframes(pcm)
            return buffer.getvalue()
