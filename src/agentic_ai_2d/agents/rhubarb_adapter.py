"""Rhubarb adapter boundary and deterministic local viseme normalization."""

from __future__ import annotations

from hashlib import sha256
import json
from math import ceil, floor
from pathlib import Path
import subprocess
import tempfile
from typing import Protocol

from ..models.narration_track import NarrationTrack
from ..models.project_spec import ProjectSpec
from ..models.viseme_timeline import MouthShape, VisemeCue, VisemeTimeline
from ..storage import ArtifactManager


class RhubarbAdapter(Protocol):
    def generate(
        self,
        project: ProjectSpec,
        narration: NarrationTrack,
        *,
        character_id: str,
        mouth_set_id: str,
    ) -> VisemeTimeline:
        """Produce a frame-normalized viseme timeline from narration audio alignment."""


class SimulatedRhubarbAdapter:
    """Deterministic local normalizer used until the Rhubarb binary is configured."""

    _SHAPE_BY_INITIAL = {
        "a": MouthShape.A,
        "b": MouthShape.B,
        "m": MouthShape.B,
        "p": MouthShape.B,
        "c": MouthShape.C,
        "d": MouthShape.C,
        "n": MouthShape.C,
        "s": MouthShape.C,
        "t": MouthShape.C,
        "z": MouthShape.C,
        "e": MouthShape.E,
        "f": MouthShape.F,
        "v": MouthShape.F,
        "g": MouthShape.G,
        "j": MouthShape.G,
        "k": MouthShape.G,
        "l": MouthShape.H,
        "r": MouthShape.H,
        "o": MouthShape.X,
        "u": MouthShape.X,
    }

    def generate(
        self,
        project: ProjectSpec,
        narration: NarrationTrack,
        *,
        character_id: str,
        mouth_set_id: str,
    ) -> VisemeTimeline:
        cues: list[VisemeCue] = []
        for utterance in narration.utterances:
            for word in utterance.words:
                shape = self._shape_for_word(word.text)
                cue = VisemeCue(
                    start_frame=word.start_frame,
                    end_frame=word.end_frame,
                    mouth_shape=shape,
                )
                if cues and cues[-1].mouth_shape is shape and cues[-1].end_frame + 1 == cue.start_frame:
                    cues[-1] = cues[-1].model_copy(update={"end_frame": cue.end_frame})
                else:
                    cues.append(cue)
        identifier_source = f"{narration.narration_track_id}:{character_id}:{mouth_set_id}".encode("utf-8")
        return VisemeTimeline(
            viseme_timeline_id=f"viseme_{sha256(identifier_source).hexdigest()[:16]}",
            project_id=project.project_id,
            project_version=project.version,
            narration_track_id=narration.narration_track_id,
            input_audio_asset_id=narration.audio_asset_id,
            generator="rhubarb",
            fps=narration.fps,
            duration_frames=narration.duration_frames,
            character_id=character_id,
            mouth_set_id=mouth_set_id,
            cues=cues,
        )

    def _shape_for_word(self, word: str) -> MouthShape:
        for character in word.lower():
            if character.isalpha():
                return self._SHAPE_BY_INITIAL.get(character, MouthShape.X)
        return MouthShape.X


class LocalRhubarbAdapter:
    """Execute the local Rhubarb binary and normalize JSON cues to frame intervals."""

    def __init__(
        self,
        storage: ArtifactManager,
        *,
        binary: Path | str,
        timeout_seconds: int = 120,
    ) -> None:
        self._storage = storage
        self._binary = Path(binary)
        self._timeout_seconds = timeout_seconds

    def generate(
        self,
        project: ProjectSpec,
        narration: NarrationTrack,
        *,
        character_id: str,
        mouth_set_id: str,
    ) -> VisemeTimeline:
        if not self._binary.is_file():
            raise FileNotFoundError(f"Rhubarb binary was not found at {self._binary}")
        source = self._storage.get_metadata(narration.audio_asset_id)
        with tempfile.TemporaryDirectory(prefix="agentic-ai-2d-rhubarb-") as temp_dir:
            work_dir = Path(temp_dir)
            audio_path = work_dir / "narration.wav"
            dialog_path = work_dir / "dialog.txt"
            output_path = work_dir / "mouth-cues.json"
            audio_path.write_bytes(self._storage.read_bytes(source))
            dialog_path.write_text(
                "\n".join(utterance.text for utterance in narration.utterances),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    str(self._binary),
                    "--dialogFile",
                    str(dialog_path),
                    "--recognizer",
                    "pocketSphinx",
                    "--extendedShapes",
                    "GHX",
                    "--exportFormat",
                    "json",
                    "--machineReadable",
                    "--output",
                    str(output_path),
                    str(audio_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
            if completed.returncode != 0 or not output_path.is_file():
                raise RuntimeError(f"Rhubarb failed: {completed.stderr}")
            try:
                raw_cues = json.loads(output_path.read_text(encoding="utf-8"))["mouthCues"]
            except (KeyError, json.JSONDecodeError) as error:
                raise RuntimeError("Rhubarb produced malformed JSON output") from error

        cues = self._to_frame_cues(raw_cues, narration.fps, narration.duration_frames)
        if not cues:
            raise RuntimeError("Rhubarb produced no usable mouth cues")
        identifier_source = f"{narration.narration_track_id}:{character_id}:{mouth_set_id}".encode("utf-8")
        return VisemeTimeline(
            viseme_timeline_id=f"viseme_{sha256(identifier_source).hexdigest()[:16]}",
            project_id=project.project_id,
            project_version=project.version,
            narration_track_id=narration.narration_track_id,
            input_audio_asset_id=narration.audio_asset_id,
            generator="rhubarb",
            fps=narration.fps,
            duration_frames=narration.duration_frames,
            character_id=character_id,
            mouth_set_id=mouth_set_id,
            cues=cues,
        )

    @staticmethod
    def _to_frame_cues(raw_cues: object, fps: int, duration_frames: int) -> list[VisemeCue]:
        if not isinstance(raw_cues, list):
            raise RuntimeError("Rhubarb mouthCues must be a list")
        cues: list[VisemeCue] = []
        for raw in raw_cues:
            if not isinstance(raw, dict):
                raise RuntimeError("Rhubarb mouth cue must be an object")
            try:
                shape = MouthShape(str(raw["value"]))
                start_seconds = float(raw["start"])
                end_seconds = float(raw["end"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("Rhubarb mouth cue is invalid") from error
            start_frame = max(0, floor(start_seconds * fps))
            end_frame = min(duration_frames - 1, ceil(end_seconds * fps) - 1)
            if end_frame < start_frame:
                continue
            if cues and start_frame <= cues[-1].end_frame:
                start_frame = cues[-1].end_frame + 1
            if end_frame < start_frame:
                continue
            if cues and cues[-1].mouth_shape is shape and cues[-1].end_frame + 1 >= start_frame:
                cues[-1] = cues[-1].model_copy(update={"end_frame": max(cues[-1].end_frame, end_frame)})
            else:
                cues.append(
                    VisemeCue(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        mouth_shape=shape,
                    )
                )
        return cues
