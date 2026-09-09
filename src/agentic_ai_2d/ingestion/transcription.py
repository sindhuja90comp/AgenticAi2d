"""Timestamped transcript types and a replaceable Whisper adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from ..storage import ArtifactManager, ArtifactMetadata


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    text: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("transcript words must not be empty")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("transcript word timestamps are invalid")


@dataclass(frozen=True, slots=True)
class TimestampedTranscript:
    text: str
    language: str
    words: tuple[TranscriptWord, ...]

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.words:
            raise ValueError("a transcript requires text and at least one word")
        previous_end = -1.0
        for word in self.words:
            if word.start_seconds < previous_end:
                raise ValueError("transcript words must be ordered and non-overlapping")
            previous_end = word.end_seconds


class WhisperAdapter(Protocol):
    """Adapter implemented by a production Whisper or speech-to-text provider."""

    def transcribe(self, audio: ArtifactMetadata, *, language: str = "en") -> TimestampedTranscript:
        """Convert a stored audio artifact into word-level timestamps."""


class StaticWhisperAdapter:
    """Deterministic local adapter used until a real Whisper provider is configured."""

    def __init__(self, transcript_text: str, duration_seconds: float) -> None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        words = transcript_text.split()
        if not words:
            raise ValueError("transcript_text must contain at least one word")
        self._text = transcript_text
        self._duration_seconds = duration_seconds
        self._words = words

    def transcribe(self, audio: ArtifactMetadata, *, language: str = "en") -> TimestampedTranscript:
        if not audio.asset_id.startswith("asset_"):
            raise ValueError("audio must be a stored asset")
        word_duration = self._duration_seconds / len(self._words)
        words = tuple(
            TranscriptWord(
                text=word,
                start_seconds=index * word_duration,
                end_seconds=(index + 1) * word_duration,
            )
            for index, word in enumerate(self._words)
        )
        return TimestampedTranscript(self._text, language, words)


class TextPromptAdapter:
    """Canonical text-input path that produces a transcript artifact, not fake audio."""

    def transcribe(self, prompt: str, *, language: str = "en") -> TimestampedTranscript:
        words = prompt.split()
        if not words:
            raise ValueError("text prompt must contain at least one word")
        return TimestampedTranscript(prompt.strip(), language, tuple(
            TranscriptWord(word, index * 0.25, (index + 1) * 0.25) for index, word in enumerate(words)
        ))

    def store(self, storage: ArtifactManager, transcript: TimestampedTranscript) -> ArtifactMetadata:
        return storage.store_contract({"source": "text_prompt", "text": transcript.text, "language": transcript.language})


class LocalWhisperAdapter:
    """Run OpenAI Whisper locally and normalize its word timestamps."""

    _models: ClassVar[dict[tuple[str, str], Any]] = {}

    def __init__(
        self,
        storage: ArtifactManager,
        *,
        model_name: str = "small.en",
        device: str = "cpu",
        download_root: Path | str | None = None,
    ) -> None:
        self._storage = storage
        self._model_name = model_name
        self._device = device
        self._download_root = str(download_root) if download_root is not None else None

    def transcribe(self, audio: ArtifactMetadata, *, language: str = "en") -> TimestampedTranscript:
        """Transcribe a stored audio artifact using a cached local Whisper model."""
        if language != "en":
            raise ValueError("Phase 1 LocalWhisperAdapter currently supports English only")
        model = self._load_model()
        result = model.transcribe(
            str(self._storage.absolute_path(audio)),
            language=language,
            word_timestamps=True,
            fp16=False,
            verbose=False,
        )
        words: list[TranscriptWord] = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                text = str(word.get("word", "")).strip()
                start = word.get("start")
                end = word.get("end")
                if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    raise RuntimeError("Whisper returned an incomplete word timestamp")
                words.append(TranscriptWord(text=text, start_seconds=float(start), end_seconds=float(end)))
        if not words:
            raise RuntimeError("Whisper returned no word timestamps")
        text = str(result.get("text", "")).strip() or " ".join(word.text for word in words)
        return TimestampedTranscript(text=text, language=language, words=tuple(words))

    def _load_model(self) -> Any:
        cache_key = (self._model_name, self._device)
        model = self._models.get(cache_key)
        if model is not None:
            return model
        try:
            import whisper
        except ImportError as error:
            raise RuntimeError(
                "Local Whisper is unavailable. Install the project's live dependencies first."
            ) from error
        model = whisper.load_model(
            self._model_name,
            device=self._device,
            download_root=self._download_root,
        )
        self._models[cache_key] = model
        return model
