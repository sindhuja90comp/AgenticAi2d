"""Speech-to-text contracts and provider-neutral transcription adapters."""

from .transcription import (
    LocalWhisperAdapter,
    StaticWhisperAdapter,
    TextPromptAdapter,
    TimestampedTranscript,
    TranscriptWord,
    WhisperAdapter,
)

__all__ = [
    "LocalWhisperAdapter",
    "StaticWhisperAdapter",
    "TextPromptAdapter",
    "TimestampedTranscript",
    "TranscriptWord",
    "WhisperAdapter",
]
