# Live Adapters Plan

## Purpose

This document defines the approved path for replacing Phase 1's deterministic test adapters with local engines. The production pipeline will remain contract-first: every live adapter reads immutable artifacts, writes immutable artifacts, and returns the same Pydantic contracts already used by the compiler, QA engine, and workflow state machine.

No implementation changes are made by this document.

## Licensing Decision

Phase 1 now standardizes on an FFmpeg renderer so the local stack stays compatible with a strict open-source-only requirement. FFmpeg handles the constrained timeline composition directly, which avoids the extra Node/Remotion runtime and keeps the render worker aligned with the existing immutable artifact model.

The recommended decisions are:

| Engine | Local execution | License position | Phase 1 decision |
| --- | --- | --- | --- |
| Whisper | Yes | Open-source code and local model inference | Use local Whisper. |
| Kokoro | Yes | Open-weight model with Apache-licensed weights stated by its project | Preferred local TTS engine. |
| Coqui TTS | Yes | Open-source toolkit; model licenses vary | Supported alternative, only after model-license review. |
| Rhubarb | Yes | Local CLI tool | Use local Rhubarb binary. |
| FFmpeg renderer | Yes | Open-source CLI/runtime | Default Phase 1 renderer. |
The FFmpeg renderer composites static pose PNGs, background PNGs, mouth-sprite overlays, and narration audio directly from the validated `Timeline` plus immutable asset references.

## Local macOS Prerequisites

Install and validate these tools before modifying application adapters.

| Requirement | Purpose | Validation |
| --- | --- | --- |
| Xcode Command Line Tools | C/C++ compilation and native Python packages | `xcode-select -p` |
| Homebrew | Installs system dependencies | `brew --version` |
| Python 3.11+ virtual environment | Runs the pipeline and local ML adapters | `python3 --version` |
| FFmpeg | Audio normalization, Whisper decoding, final media support | `ffmpeg -version` |
| espeak-ng | Kokoro English G2P dependency | `espeak-ng --version` |
| Rhubarb binary | Generates mouth cue JSON | `rhubarb --version` |

Recommended system installation commands:

```zsh
xcode-select --install
brew install ffmpeg espeak-ng cmake boost
```

Use a project-local Python virtual environment; do not install machine-learning packages into the system Python:

```zsh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install Whisper and its Python dependencies in that environment. Whisper requires FFmpeg and documents Homebrew installation for macOS. [Whisper setup](https://github.com/openai/whisper/blob/main/README.md?plain=1)

```zsh
python -m pip install openai-whisper torch
```

Install one TTS engine, not both initially:

```zsh
# Preferred initial option
python -m pip install 'kokoro>=0.9.4' soundfile 'misaki[en]'

# Alternative, only after confirming a supported Python/model combination
python -m pip install coqui-tts
```

Kokoro's project documents `pip install kokoro`, `soundfile`, and Apple Silicon MPS fallback. [Kokoro installation and macOS notes](https://github.com/hexgrad/kokoro)

Install Rhubarb from its macOS release and expose its absolute binary path through configuration. Rhubarb accepts WAVE or Ogg input, emits JSON, and supports a dialog file to improve recognition. [Rhubarb CLI reference](https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/README.adoc)

```zsh
# Example only: download and unpack the macOS release first.
export RHUBARB_BINARY="$PWD/tools/rhubarb/rhubarb"
"$RHUBARB_BINARY" --version
```

For Apple Silicon, test both CPU and MPS execution. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` only when an unsupported operation needs CPU fallback; record the selected device in the artifact manifest.

## Configuration and Secrets

Add a future `.env.example` with paths and pinned model choices. Do not commit `.env` or downloaded model files.

```dotenv
ARTIFACT_ROOT=./var/artifacts
WHISPER_MODEL=small.en
WHISPER_DEVICE=mps
TTS_ENGINE=kokoro
KOKORO_VOICE=af_heart
RHUBARB_BINARY=/absolute/path/to/rhubarb
FFMPEG_BINARY=/opt/homebrew/bin/ffmpeg
FFPROBE_BINARY=/opt/homebrew/bin/ffprobe
RENDER_ENGINE=ffmpeg
```

The first run will download model weights unless the selected engines are pre-seeded in their local model caches. For fully offline deployment, download and checksum every model beforehand, then configure each engine's cache path to the pre-seeded location.

## Adapter Replacement Plan

### 1. Local Whisper transcription

Current boundary: `src/agentic_ai_2d/ingestion/transcription.py`

Add `LocalWhisperAdapter`, implementing the existing `WhisperAdapter` protocol.

1. Resolve `ArtifactMetadata.relative_path` through `ArtifactManager.root`; never accept an arbitrary user-provided file path.
2. Load the configured Whisper model once per worker process and cache it by `(model_name, device)`.
3. Invoke `model.transcribe(audio_path, language="en", word_timestamps=True, fp16=False)` on macOS unless the installed backend explicitly supports the selected accelerator path.
4. Convert Whisper word timestamps into `TranscriptWord` values. Reject missing, negative, non-monotonic, or overlapping timestamps.
5. Return `TimestampedTranscript` and store a canonical transcript JSON artifact containing source audio asset ID, model revision, language, and word timestamps.
6. Emit an `ArtifactPublishedEvent` for the transcript artifact using the existing event repository.

Use local Whisper model inference, not an API call. Whisper supports local installation through `openai-whisper` and requires FFmpeg for media decoding. [Whisper README](https://github.com/openai/whisper/blob/main/README.md?plain=1)

### 2. Local Kokoro or Coqui TTS

Current boundary: `src/agentic_ai_2d/agents/tts_agent.py`

Add a `LocalTtsAgent` protocol implementation selected by configuration:

- `KokoroTtsAgent` is the default. It loads `KPipeline`, synthesizes each approved narration utterance locally, concatenates the generated PCM samples, and writes a 24 kHz or configured WAV through `soundfile`.
- `CoquiTtsAgent` is an alternative subprocess or library adapter. It must use a pinned model name and record its model license, version, and voice settings in output metadata. Coqui's toolkit supports local CLI inference, but published compatibility guidance varies by release, so its exact Python version must be verified in the target environment. [Coqui TTS repository](https://github.com/coqui-ai/TTS)

Required code changes:

1. Replace `DeterministicTtsAgent._silence_wav()` with an engine call that generates PCM/WAV data.
2. Preserve the existing `NarrationTrack` contract. Generate word alignment from the approved narration text and measured output audio duration; do not infer alignment from an assumed fixed duration.
3. Normalize all narration to mono WAV with FFmpeg before Rhubarb: 48 kHz for project audio storage or a Rhubarb-compatible WAV sample rate as validated in integration tests.
4. For `narration_source="user_recording"`, bypass synthesis, normalize the approved user recording with FFmpeg, and align the approved transcript to it before creating `NarrationTrack`.
5. Store the WAV plus canonical alignment JSON through `ArtifactManager`; include engine name, engine/model revision, voice, sample rate, and source contract digest in artifact metadata.
6. Reject a measured audio duration that cannot fit the approved storyboard duration. The repair path is to revise the narration or storyboard, not to silently time-stretch content.

### 3. Local Rhubarb lip sync

Current boundary: `src/agentic_ai_2d/agents/rhubarb_adapter.py`

Add `LocalRhubarbAdapter`, implementing the existing `RhubarbAdapter` protocol. It will execute the configured binary with `subprocess.run()` using an argument list, `shell=False`, a timeout, and a temporary per-job working directory.

For each narration asset:

1. Materialize the normalized WAV and a UTF-8 narration text file in the job directory.
2. Run Rhubarb with JSON output, dialog text, English recognition, extended shapes, and machine-readable logs:

```text
<rhubarb-binary> --dialogFile <dialog.txt> --recognizer pocketSphinx \
  --extendedShapes GHX --exportFormat json --machineReadable \
  --output <rhubarb.json> <narration.wav>
```

3. Fail the activity on non-zero exit status, timeout, malformed JSON, unsupported mouth shape, or unexpected duration mismatch. Persist stderr as a diagnostic artifact without exposing it as a user-facing video asset.
4. Parse `mouthCues`, then convert seconds to the project frame rate using a fixed policy:
   - `start_frame = floor(start_seconds * fps)`
   - `end_frame = ceil(end_seconds * fps) - 1`
   - Clamp results to `[0, duration_frames - 1]`, discard empty intervals, then merge adjacent equal-shape cues.
5. Produce the existing `VisemeTimeline` with only `A, B, C, D, E, F, G, H, X` values.
6. Run `SemanticValidator` after conversion; any gap or overlap is a content-quality failure, subject to the existing one-repair budget.

Rhubarb's CLI supports JSON export, `--dialogFile`, `--machineReadable`, the English `pocketSphinx` recognizer, and the extended `G`, `H`, and `X` shapes. [Rhubarb command options](https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/README.adoc)

### 4. Local FFmpeg renderer

Current boundary: `src/agentic_ai_2d/renderer/adapter.py`

The real implementation replaces the live renderer backend only; the `RendererAdapter` interface and `RenderResult` remain unchanged.

The Python `LocalFfmpegRenderer` will:

1. Write immutable job inputs to a fresh render directory: `timeline.json`, `asset-map.json`, and `render-config.json`.
2. Build `asset-map.json` from `ArtifactMetadata`, mapping each asset ID to an absolute local path. Timeline contracts remain asset-ID based and never gain machine-specific paths.
3. Invoke `ffmpeg` through `subprocess.run()` with `shell=False`, a timeout, and captured stderr for diagnostics.
4. Build a deterministic filtergraph that composites backgrounds, pose PNGs, motion transforms, mouth-sprite crops, and narration audio at the Timeline's width, height, fps, and duration.
5. Verify that `output.mp4` exists, is non-empty, and passes `ffprobe` stream validation before publishing it through `ArtifactManager`.
6. Store `render_manifest.json` with Timeline digest, FFmpeg version, FFprobe version, asset digests, output digest, elapsed time, and probed stream metadata.

## Planned Python File Changes

| File | Planned change |
| --- | --- |
| `storage/artifacts.py` | Persist engine provenance and add durable metadata lookup/reload for process restarts. |
| `ingestion/transcription.py` | Add `LocalWhisperAdapter`, model configuration, timestamp conversion, and transcript artifact publishing. |
| `agents/tts_agent.py` | Add `KokoroTtsAgent` and optional `CoquiTtsAgent`; preserve the `TtsAgent` protocol. |
| `agents/rhubarb_adapter.py` | Add `LocalRhubarbAdapter` that executes Rhubarb safely and normalizes JSON mouth cues. |
| `renderer/adapter.py` | Add `LocalFfmpegRenderer` with FFmpeg filtergraph composition and FFprobe validation. |
| `compiler/scene_compiler.py` | No contract redesign; add any renderer asset-map helper needed to resolve stored asset metadata. |
| `run_pipeline.py` | Add a command-line entry point that wires live adapters together from configuration. |
| `tests/` | Add integration tests marked `live` and retain deterministic tests as the default fast suite. |

## `run_pipeline.py` Execution Flow

## Autonomous Phase 1 Micro-Scenes

The executable Phase 1 path accepts a text prompt and generates all required PNG layers locally in
`assets/generated/`; it does not ask the user for artwork. The intentionally bounded planner supports:

- `butterfly` plus `flower`: transparent butterfly, flower, background, and mouth-sheet layers; the renderer flaps the butterfly and sways the flower.
- `bird` plus `fence`: transparent bird, fence background, and mouth-sheet layers; the renderer moves the bird through a deterministic series of hop arcs.

This is a deterministic scenario planner, not a general local LLM. A future local LLM must emit the same
validated `Storyboard` motion contract before the Scene Compiler accepts it.

After this plan is approved and implemented, the command will be:

```zsh
source .venv-live/bin/activate
python3 run_pipeline.py --project-id proj_butterfly01 --prompt "A butterfly rests on a flower"
```

Expected execution sequence:

1. Store the recorded voice input with `ArtifactManager`, producing an immutable asset ID and SHA-256 digest.
2. Run local Whisper to create the timestamped transcript artifact.
3. Run the Writer Agent to create `Storyboard`; stop in `awaiting_approval` until the user approves it.
4. After approval, run Kokoro or Coqui locally to create narration WAV and `NarrationTrack`.
5. Normalize narration WAV and run local Rhubarb to create `VisemeTimeline`.
6. Compile the approved contracts and cached assets to `Timeline`.
7. Run deterministic pre-render QA. A failure follows the existing retry/repair rules and does not render a video.
8. Invoke the configured local renderer. The FFmpeg worker renders a real H.264/AAC MP4 directly from the compiled Timeline and immutable asset set.
9. Validate the MP4 with `ffprobe`, store it immutably, write `render_manifest.json`, publish the render event, and transition the project to `completed`.

The final playable file will be exposed at a stable user-facing path such as `var/exports/proj_rabbit001/v1/output.mp4`; its content-addressed copy remains in artifact storage for reproducibility.

## Acceptance Criteria Before Implementation

- All engine binaries and local model weights are available without an external API request during a test run.
- A 15-second English fixture produces a valid Whisper transcript, WAV narration, Rhubarb JSON, Timeline, and an MP4 recognized by `ffprobe`.
- Re-running the same immutable inputs produces artifacts with recorded engine/model versions and deterministic contract validation results.
- Infrastructure failures preserve the current retry policy; schema/content/safety failures preserve the repair boundaries defined in `SYSTEM_ARCHITECTURE.md`.
- The selected FFmpeg renderer is documented and produces a valid MP4 plus captured-frame verification assets during integration checks.
