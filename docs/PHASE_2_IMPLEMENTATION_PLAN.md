# Phase 2 Implementation Plan: Local AI 2D Animation Agent

## Purpose

Phase 2 turns the Phase 1 renderer prototype into a local-first AI animation agent. The agent receives a live spoken instruction, transcribes it locally with Whisper, sends only plain-language planning text to Groq's cloud planner, uses constrained local software tools to build a visual draft, waits for human review, and renders only the approved version with local FFmpeg.

This document is the implementation agreement for Phase 2. The LLM may propose creative content, but it must never directly construct shell commands, FFmpeg filters, file paths, or unvalidated timeline data. Validated contracts are the boundary between intelligence and execution.

## Phase 2 Outcome

For a supported request such as "Put a butterfly on top of a flower and let the flower lean in the wind," the system will:

1. Capture or accept the user's voice instruction.
2. Transcribe it locally with Whisper.
3. Use Groq's `openai/gpt-oss-20b` text planner to propose a structured scene plan.
4. Generate or select approved 2D assets, then create a low-resolution visual preview.
5. Pause for the user to approve, reject, or request a revision after seeing the preview.
6. Render an approved MP4 with the same validated scene data and FFmpeg.

Phase 2 remains intentionally bounded: English only, 15-45 seconds, at most three scenes, and an approved asset/style catalog. It is not an unrestricted text-to-video system.

## Architecture Rules

| Rule | Decision |
| --- | --- |
| Intelligence boundary | Groq receives only plain-language instruction and approved-catalog text. It returns JSON that is validated locally; it cannot call FFmpeg or access arbitrary paths. |
| Tool boundary | Asset, layout, preview, audio, and render tools accept only validated contracts and artifact IDs. |
| Review boundary | The final render can begin only from an explicitly approved preview version. |
| Reproducibility | Every version records input audio, transcript, LLM model/prompt version, assets, layout, preview, approval, and render manifest. |
| Safe revision | Feedback creates a new project version. It never mutates an approved or rendered version. |
| Rendering | FFmpeg remains the only final video renderer. |
| Cloud structural planner | Use Groq's supported `openai/gpt-oss-20b` endpoint. This replaced the deprecated Llama endpoint; pin the provider, model ID, prompt version, request settings, and benchmark results. |
| Local runtime policy | Whisper, asset tools, preview, and FFmpeg remain local on macOS Monterey (12.x) Intel. Do not source-compile runtime dependencies on this legacy OS. |
| Asset strategy | Use only the curated, approved asset catalog in Phase 2. Each animatable asset is pre-segmented into parts. |
| Review UI | A local browser page displays preview media and records approval or feedback. |
| Default narration | The user's first recording is an instruction for the agent; Kokoro generates final narration unless the user explicitly chooses to preserve their recording. |

## Six-Step Pipeline Architecture

```text
Voice instruction
  -> Whisper transcript
  -> Text-only Groq scene plan
  -> Asset and layout tools
  -> Preview MP4 and review frames
  -> Human approval or revision
  -> FFmpeg final MP4
```

### 1. Voice Input: Whisper Ears

**Purpose:** turn the user's spoken instruction into a timestamped, immutable transcript.

**Inputs**

- A microphone recording or uploaded WAV/MP3/M4A file.
- Optional language selection; Phase 2 initially supports English only.

**Implementation**

- Extend `run_pipeline.py` with `--input-audio`, `--record`, and `--narration-source` options.
- Store the original audio through `ArtifactManager` before transcription.
- Invoke `LocalWhisperAdapter` in `transcription.py` and store a transcript artifact containing text, word timestamps, model name, model revision, and device.
- Reject empty, unsupported, or malformed recordings before planning.

**Output contract:** `TimestampedTranscript` plus a stored transcript artifact.

**Important distinction:** the voice instruction tells the agent what to create. If the user wants that same recording in the final animation, `narration_source` is `user_recording`. Otherwise, the LLM creates approved narration text and Kokoro creates a different narration voice after approval.

### 2. Scene Generation: Groq Text Planner

**Purpose:** convert the transcript into a creative but bounded scene plan instead of selecting a rigid keyword template.

**Implementation**

- Add a provider-neutral planner protocol with Groq as the first implementation. Pin `openai/gpt-oss-20b`, prompt version, temperature, and request settings in metadata.
- The planner request may contain only the instruction transcript as plain text and approved catalog descriptions as plain text. Never send audio, images, video, artifact IDs, file paths, FFmpeg data, or API keys.
- Implement the first adapter in `agents/planner.py`. It requests JSON output and validates the returned object locally before any asset or renderer tool runs.
- Replace the Phase 1 keyword-only path in `writer_agent.py` with an LLM planning path that emits a schema-constrained `StoryboardDraft`.
- Give the LLM an explicit catalog of allowed characters, actions, backgrounds, camera shots, motion types, and style rules.
- Validate the response with Pydantic and `SemanticValidator`. If it is invalid, allow one structured repair request; otherwise stop and show a useful error.

**Required planning additions**

- User intent summary and scene list.
- Narration text or an explicit instruction to preserve the user's recording.
- Character and prop requests by semantic name, never by local file path.
- Per-actor position, scale, layer order, facing direction, and motion.
- Motion anchor/pivot information for natural movement.
- Camera framing and expected duration.

**Output contract:** a validated `StoryboardDraft`; the existing `Storyboard` becomes the approved, immutable version.

### 3. Asset and Layout Compilation: Tool Hands

**Purpose:** turn a scene plan into real, positionally correct assets and a safe render timeline.

**Implementation**

- Add an `AssetRequest` contract and an `AssetTool` boundary. In Phase 2, the asset tool selects only from the approved curated asset catalog and stores every selected image through `ArtifactManager`.
- Do not let the LLM invent asset IDs. The asset tool resolves semantic requests to immutable asset IDs.
- Extend actor/layout contracts with numerical `x_percent`, `y_percent`, `scale`, `z_index`, `anchor_x_percent`, and `anchor_y_percent` values. Keep values bounded and validate overlap, out-of-frame placement, and unsupported layer combinations.
- Update `scene_compiler.py` to preserve the LLM-approved layout rather than mapping every actor to `left`, `center`, or `right` defaults.
- Update `adapter.py` so transforms honor the specified anchor point.

**Technical fixes required from Phase 1**

| Problem | Phase 2 correction |
| --- | --- |
| Butterfly and flower occupy the same center position | The planner sets separate coordinates and layer order; the compiler rejects unintended large overlap. |
| Flower covers butterfly | Butterfly gets a higher `z_index` when it is on top of the flower. |
| Flower rotates around its center | Set the flower pivot to the bottom center of the stem. Rotate the complete flower layer around that pivot so the head moves in an arc. |
| Butterfly looks like one object shrinking | Split butterfly art into body and left/right wing layers, then rotate or scale wings around body-side anchors. |
| Static, front-facing composition | Support side/perspective asset variants in the asset catalog and require the LLM to select one intentionally. |

**Output contract:** a validated asset map and `TimelineDraft` with concrete positions, layer order, anchors, and motions.

### 4. Visual Storyboard Generation: Preview Eyes

**Purpose:** let a person judge the actual visual composition before approving the expensive final render.

**Implementation**

- Add a `PreviewRenderer` mode to the FFmpeg renderer. It uses the same timeline and assets as final rendering, but renders at 854x480, lower bitrate, and a short review duration or lower preview frame rate.
- Export at least three review frames: first frame, a representative middle frame, and final frame. Export a contact sheet and a playable preview MP4.
- Store `preview.mp4`, `preview-contact-sheet.png`, individual frames, and a `preview_manifest.json` as immutable artifacts.
- Add automated image checks for missing layers, blank frames, out-of-bounds actors, and excessive visual overlap. These checks complement, but do not replace, human judgment.

**Output contract:** `PreviewPackage` containing preview artifact IDs, a preview timeline digest, and review-frame timestamps.

### 5. Human-in-the-Loop Review: Approval Gate

**Purpose:** make visual approval real rather than automatically marking a storyboard approved.

**Implementation**

- Add workflow states for `awaiting_preview_review`, `revision_requested`, and `preview_approved`.
- Stop the pipeline after preview generation. Do not invoke the final renderer at this point.
- Present the preview MP4 and review frames in a lightweight local browser page that accepts `approve`, `reject`, or structured feedback.
- Convert feedback into a `RevisionRequest` contract. The LLM receives the prior approved inputs, preview notes, and allowed catalog, then produces a new draft version.
- Require explicit approval for the exact preview digest and project version. Approval for version 1 cannot render version 2.

**Output contract:** `ApprovalDecision` or `RevisionRequest`, both bound to a project version and preview digest.

### 6. Final Rendering: FFmpeg Hands

**Purpose:** produce the deliverable video only after preview approval.

**Implementation**

- Recompile the approved timeline at final output size and FPS; do not use unapproved feedback or mutable assets.
- By default, generate approved narration text with Kokoro, then run Rhubarb against that audio. If the user explicitly selects preserved narration, normalize the approved recording and run Rhubarb against its Whisper transcript.
- Render H.264 video and AAC audio with `LocalFfmpegRenderer`.
- Validate MP4 streams, duration, dimensions, audio presence, and non-empty review frames with `ffprobe` and FFmpeg frame extraction.
- Store the final MP4, captured frame, renderer manifest, model/tool versions, and approval digest.

**Output contract:** `RenderResult` with MP4 path, final-frame artifact, and manifest.

## End-to-End Execution Flow

1. Create `ProjectSpec` version 1 and store the user audio.
2. Whisper transcribes the instruction and stores `TimestampedTranscript`.
3. Groq emits a text-only JSON blueprint from transcript plus approved catalog text; local code validates and converts it to `StoryboardDraft`.
4. Validate and resolve asset requests; create an asset map.
5. Compile a layout-aware `TimelineDraft`.
6. Render and store `PreviewPackage`.
7. Pause in `awaiting_preview_review`.
8. If revision is requested, create project version 2 and repeat steps 3-7 using the feedback contract.
9. If approved, create or normalize final narration, generate lip sync, compile the final timeline, and run pre-render QA.
10. FFmpeg renders the final MP4; FFprobe and captured-frame checks validate it.
11. Publish `RenderResult`, mark the approved version complete, and preserve all artifacts.

## Implementation Roadmap

### Step 1: Blueprint and Contract Foundation

**Mandatory start gate:** complete Task 1 and Task 2 below before writing any downstream pipeline code, agent execution logic, tool integration, preview UI, or rendering changes.

**Local runtime requirement:** Whisper, assets, preview, and FFmpeg run on macOS Monterey (12.x) Intel hardware. Do not use Homebrew source builds or source-compiled LLVM, Rust, Go, or other toolchains to obtain local runtime dependencies on this legacy OS.

#### Task 1: Model Benchmarking and Pinning

- Run cloud speed, JSON-output, and local schema-validation tests against Groq's supported `openai/gpt-oss-20b` endpoint.
- Measure request latency, valid-schema rate, repair rate, rate-limit behavior, and cost for the target account.
- Pin the Groq provider, exact model ID, request settings, prompt version, and JSON-output mode.
- Save the benchmark results and selected configuration as a project artifact or checked-in configuration record.

#### Task 2: Contract Scaffolding

- Write and verify the Phase 2 Pydantic schemas before any LLM or tool executes: `StoryboardDraft`, `AssetRequest`, timeline transform and anchor fields, `PreviewPackage`, `ApprovalDecision`, and `RevisionRequest`.
- Export schema snapshots and add valid, invalid, and cross-contract validation tests.
- Confirm that the schemas express the required bounds for positions, layers, motion anchors, asset references, preview digests, approvals, and revisions.

#### Step 1 Completion Work

- Use the approved curated multipart asset catalog and local browser review UI defined in the confirmed decisions below.
- Update JSON schema snapshots, semantic validation, state transitions, event types, and artifact provenance.
- Define no-deviation tests: LLM output must be validated before tools run; final rendering requires the matching approval digest.

### Step 2: Whisper and Live Audio Integration

- Add audio upload/recording CLI or local UI support.
- Wire `LocalWhisperAdapter` into `run_pipeline.py`.
- Add the two narration paths: user recording is preserved; synthetic narration uses Kokoro only after preview approval.
- Test a real voice recording from ingestion through transcript storage.

### Step 3: Dynamic Groq Scene Planning

- Implement the text-only Groq planner adapter and a schema-constrained prompt.
- Replace the template selector in `writer_agent.py` for Phase 2 requests.
- Add repair-once behavior, model provenance, prompt snapshots, and tests for invalid/unsafe model responses.
- Keep the Phase 1 templates as deterministic fallback fixtures, not as the Phase 2 planner.

### Step 4: Asset, Layout, and Natural Motion

- Implement the asset catalog/tool boundary and asset requests.
- Extend storyboard and timeline transforms for exact coordinates, z-index, scale, and anchor pivots.
- Fix flower base-pivot sway and butterfly body/wing separation.
- Add visual-layout tests for the butterfly-on-flower scenario, including overlap and layer-order assertions.

### Step 5: Preview and Human Approval

- Add low-resolution FFmpeg preview rendering, three review frames, and a contact sheet.
- Implement durable approval/revision events and the `awaiting_preview_review` pause.
- Add a minimal local review interface that shows the preview and records feedback.
- Test that no final MP4 is rendered before explicit approval.

### Step 6: Approved Final Render and Quality Gates

- Wire approved narration, Rhubarb, final timeline compilation, FFmpeg rendering, FFprobe validation, and captured-frame validation.
- Record final provenance: input digest, Whisper model, LLM model/prompt, asset digests, preview digest, approval digest, Rhubarb version, FFmpeg version, and output digest.
- Run a real end-to-end test for both user-recorded narration and Kokoro narration.

### Step 7: Reliability and Release Readiness

- Replace the in-memory event repository with durable local persistence before treating the review flow as reliable.
- Add restart recovery, job timeouts, error artifacts, and clear user-facing failure messages.
- Restore or initialize Git version control, pin dependencies/models, and add a repeatable setup command.
- Measure preview time, final render time, transcription time, and failure rate; set practical limits before release.

## Acceptance Criteria

- A real English voice instruction creates a stored local Whisper transcript and a validated Groq JSON scene plan.
- The agent supports at least the approved butterfly-on-flower example without hard-coded keyword placement.
- A preview MP4, contact sheet, and three review frames exist before final rendering.
- The system stops for explicit approval and creates a new immutable version for each revision.
- The approved butterfly appears above the flower; its wings animate independently; the flower bends from its stem base with the head moving in an arc.
- User voice and Kokoro narration both produce audio that is present in the final MP4.
- Final MP4 passes FFprobe, includes H.264 video and AAC audio, and has a captured-frame artifact.
- Tests cover contracts, invalid LLM output, approval gating, layout/layering, pivot behavior, and at least one real end-to-end render.
- The final manifest connects the MP4 to the exact approved preview and every model/tool version used.

## Confirmed Phase 2 Decisions

| Area | Confirmed decision |
| --- | --- |
| Structural planner | Groq `openai/gpt-oss-20b` receives only plain-language planning text and returns a JSON blueprint for local validation. It replaced the deprecated Llama endpoint. |
| Asset strategy | Use only a curated, approved asset catalog. Assets must contain pre-segmented animatable parts, for example flower head/stem and butterfly body/left wing/right wing. No unconstrained image generation in Phase 2. |
| Review UI | Build a lightweight browser-based local UI that shows the preview MP4, contact sheet, review frames, and clear approve/reject controls. |
| Narration | Treat the initial user recording as an instruction for local Whisper and the Groq text planner. Generate final narration with Kokoro by default. Preserve the user's recording only when they explicitly select that option. |

## Explicit Non-Goals for Phase 2

- Unrestricted generation of any visual style, character, or animation genre.
- Letting an LLM execute shell commands or write render filtergraphs.
- Rendering an unreviewed scene as the final deliverable.
- Silent changes to assets, narration, or layout after approval.
