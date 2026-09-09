"""Run the Phase 2 local-first pipeline through its mandatory preview review gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import os
from pathlib import Path

from agentic_ai_2d.agents import GroqTextPlanner, JumpSfxAgent, storyboard_from_draft
from agentic_ai_2d.assets import Phase1AssetGenerator, Scenario, bird_fence_catalog
from agentic_ai_2d.compiler import SceneCompiler
from agentic_ai_2d.events import EventType, SqliteEventRepository
from agentic_ai_2d.ingestion import LocalWhisperAdapter
from agentic_ai_2d.models.phase2 import ApprovalAction, ApprovalDecision, RevisionRequest
from agentic_ai_2d.models.project_spec import ProjectSpec, ProjectStatus
from agentic_ai_2d.qa import PreRenderQaEngine
from agentic_ai_2d.renderer import LocalFfmpegRenderer, PreviewRenderer
from agentic_ai_2d.semantic_validator import SemanticValidator
from agentic_ai_2d.state_machine import WorkflowStateMachine
from agentic_ai_2d.storage import ArtifactManager
from agentic_ai_2d.review_server import serve_review
from agentic_ai_2d.workflow_store import WorkflowStore


SCENARIOS = {
    "bird_fence": {
        "scenario": Scenario.BIRD_FENCE,
        "character_id": "char_bird_v1",
        "mouth_set_id": "mouthset_bird_v1",
        "catalog_description": "catalog_fence_v1: background; catalog_bird_body_v1: character_part, character char_bird_v1, parts body; catalog_bird_left_wing_v1: character_part, character char_bird_v1, parts left_wing; catalog_bird_right_wing_v1: character_part, character char_bird_v1, parts right_wing. Use the body and both wing entries as a single multipart bird rig. For this instruction, all three parts must make four synchronized bird_hop jumps forward; do not use butterfly_wing_flap and do not request a whole-bird asset.",
        "background_key": "catalog_fence_v1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--project-version", type=int, default=1, help="Saved version to resume for review.")
    parser.add_argument("--input-audio", type=Path, help="Local voice instruction WAV, MP3, or M4A.")
    parser.add_argument("--resume-review", action="store_true", help="Read a saved browser review decision for this project.")
    parser.add_argument("--serve-saved-review", action="store_true", help="Open the local review page for a saved project version.")
    parser.add_argument("--duration", default=15, type=int, choices=range(15, 46))
    parser.add_argument("--fps", default=30, type=int, choices=(24, 30))
    parser.add_argument("--whisper-model", default="small.en")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="bird_fence")
    parser.add_argument("--review-action", choices=("approve", "reject"))
    parser.add_argument("--serve-review", action="store_true", help="Open the local browser review server after preview creation.")
    parser.add_argument("--review-feedback", default="Please revise the preview composition.")
    parser.add_argument("--assets-root", type=Path, default=Path("assets/generated"))
    parser.add_argument("--artifact-root", type=Path, default=Path("var/artifacts"))
    parser.add_argument("--output-root", type=Path, default=Path("var/exports"))
    parser.add_argument("--ffmpeg-binary", default=os.getenv("FFMPEG_BINARY", "ffmpeg"))
    parser.add_argument("--ffprobe-binary", default=os.getenv("FFPROBE_BINARY", "ffprobe"))
    parser.add_argument("--rhubarb-binary", type=Path, default=Path(os.getenv("RHUBARB_BINARY", "tools/rhubarb/rhubarb")))
    parser.add_argument("--kokoro-voice", default=os.getenv("KOKORO_VOICE", "af_heart"))
    return parser.parse_args()


def transition(machine: WorkflowStateMachine, project: ProjectSpec, event_type: EventType, digest: str) -> None:
    machine.transition(project, event_type, actor="run_pipeline", correlation_id=project.project_id,
                       idempotency_key=f"{project.project_id}:v{project.version}:{event_type.value}", payload_digest=digest)


def make_project(args: argparse.Namespace, audio_asset_id: str, transcript: str, character_id: str) -> ProjectSpec:
    return ProjectSpec.model_validate({
        "project_id": args.project_id, "version": 1, "status": ProjectStatus.DRAFT, "created_at": datetime.now(timezone.utc),
        "input": {"audio_asset_id": audio_asset_id, "user_prompt": transcript, "language": "en"},
        "output": {"aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": args.fps, "format": "mp4"},
        "creative_constraints": {"target_duration_seconds": args.duration, "style_preset": "storybook_2d_flat",
            "character_ids": [character_id], "narration_source": "synthetic_tts",
            "captions_enabled": True, "user_approval_required": True},
    })


def main(args: argparse.Namespace | None = None) -> int:
    args = parse_args() if args is None else args
    if args.serve_saved_review:
        records = WorkflowStore(args.artifact_root / "phase2-workflow.db")
        if records.get(args.project_id, args.project_version, "preview") is None:
            raise RuntimeError("no saved preview exists for this project version")
        print("Open http://127.0.0.1:8765 to approve or reject this preview.")
        decision = serve_review(args.artifact_root / "phase2-workflow.db", args.project_id, args.project_version)
        if decision is not None:
            args.serve_saved_review = False
            args.serve_review = True
            args.resume_review = True
            return main(args)
        return 0
    if args.resume_review:
        records = WorkflowStore(args.artifact_root / "phase2-workflow.db")
        version = args.project_version
        decision_record = records.get(args.project_id, version, "approval_decision")
        preview = records.get(args.project_id, version, "preview")
        if decision_record is None or preview is None:
            raise RuntimeError("no saved browser review decision exists for this project")
        decision = ApprovalDecision.model_validate(decision_record)
        if (
            decision.project_id != args.project_id or decision.project_version != version
            or decision.preview_id != preview["preview_id"] or decision.preview_digest != preview["timeline_digest"]
        ):
            raise RuntimeError("saved approval decision is not bound to the current preview")
        if decision.action is ApprovalAction.REJECTED:
            project_record = records.get(args.project_id, version, "project")
            transcript_record = records.get(args.project_id, 1, "transcript")
            if project_record is None or transcript_record is None:
                raise RuntimeError("saved project records are incomplete")
            previous = ProjectSpec.model_validate(project_record)
            revision = previous.model_copy(update={"version": version + 1, "parent_version": version, "status": ProjectStatus.DRAFT})
            machine = WorkflowStateMachine(SqliteEventRepository(args.artifact_root / "phase2-workflow.db"))
            transition(machine, previous, EventType.PREVIEW_REJECTED, preview["timeline_digest"])
            machine.create_revision(
                previous, revision, actor="run_pipeline", correlation_id=revision.project_id,
                idempotency_key=f"{revision.project_id}:v{revision.version}:revision_created",
                payload_digest=preview["timeline_digest"],
            )
            config = SCENARIOS[args.scenario]
            feedback = decision.feedback or ""
            previous_draft = records.get(args.project_id, version, "storyboard_draft") or {}
            geometry_context = json.dumps(previous_draft.get("bird_fence_geometry"))
            draft = GroqTextPlanner().plan_draft(
                f"{transcript_record['text']}\n\nCurrent geometry settings: {geometry_context}\n\nRevision feedback: {feedback}", project_id=revision.project_id,
                project_version=revision.version, transcript_asset_id=transcript_record["asset_id"],
                catalog_description=f"{config['catalog_description']} Produce exactly one scene with duration_frames {revision.creative_constraints.target_duration_seconds * revision.output.fps}.",
            )
            records.save(revision.project_id, revision.version, "project", revision)
            records.save(revision.project_id, revision.version, "storyboard_draft", draft)
            records.save(revision.project_id, revision.version, "revision_request", {"parent_version": version, "feedback": feedback})
            transition(machine, revision, EventType.PLANNING_STARTED, preview["timeline_digest"])
            storage = ArtifactManager(args.artifact_root)
            pack = Phase1AssetGenerator(args.assets_root).generate(config["scenario"], geometry=draft.bird_fence_geometry)
            stored_catalog = {
                key: storage.store_bytes(entry.path.read_bytes(), filename=entry.path.name, media_type="image/png")
                for key, entry in bird_fence_catalog(pack).items()
            }
            storyboard = storyboard_from_draft(draft, revision, catalog_asset_ids={key: item.asset_id for key, item in stored_catalog.items()}, user_recording_text=transcript_record["text"])
            transition(machine, revision, EventType.CONTRACTS_VALIDATED, preview["timeline_digest"])
            narration = JumpSfxAgent().generate(revision, storyboard, storage)
            assets = {item.asset_id: item for item in (storage.get_metadata(revision.input.audio_asset_id), *stored_catalog.values(), storage.get_metadata(narration.audio_asset_id))}
            timeline = SceneCompiler().compile(revision, storyboard, narration, [], assets, {}, audio_kind="sfx")
            preview = PreviewRenderer(LocalFfmpegRenderer(storage, output_root=args.output_root / "previews", ffmpeg_binary=args.ffmpeg_binary, ffprobe_binary=args.ffprobe_binary), storage, ffmpeg_binary=args.ffmpeg_binary).render(timeline)
            records.save(revision.project_id, revision.version, "preview", preview)
            transition(machine, revision, EventType.PREVIEW_CREATED, preview.timeline_digest)
            print(f"Version {revision.version} preview ready: {storage.absolute_path(storage.get_metadata(preview.preview_asset_id))}")
            if args.serve_review:
                args.project_version = revision.version
                args.resume_review = False
                args.serve_saved_review = True
                return main(args)
            return 0
        project_record = records.get(args.project_id, version, "project")
        draft_record = records.get(args.project_id, version, "storyboard_draft")
        transcript_record = records.get(args.project_id, 1, "transcript")
        if project_record is None or draft_record is None or transcript_record is None:
            raise RuntimeError("saved project records are incomplete")
        from agentic_ai_2d.models.phase2 import StoryboardDraft

        storage = ArtifactManager(args.artifact_root)
        project = ProjectSpec.model_validate(project_record)
        draft = StoryboardDraft.model_validate(draft_record)
        config = SCENARIOS[args.scenario]
        pack = Phase1AssetGenerator(args.assets_root).generate(config["scenario"], geometry=draft.bird_fence_geometry)
        stored_catalog = {
            key: storage.store_bytes(entry.path.read_bytes(), filename=entry.path.name, media_type="image/png")
            for key, entry in bird_fence_catalog(pack).items()
        }
        storyboard = storyboard_from_draft(draft, project, catalog_asset_ids={key: item.asset_id for key, item in stored_catalog.items()}, user_recording_text=transcript_record["text"])
        machine = WorkflowStateMachine(SqliteEventRepository(args.artifact_root / "phase2-workflow.db"))
        transition(machine, project, EventType.PREVIEW_APPROVED, preview["timeline_digest"])
        transition(machine, project, EventType.PROJECT_APPROVED, preview["timeline_digest"])
        narration = JumpSfxAgent().generate(project, storyboard, storage)
        assets = {item.asset_id: item for item in (storage.get_metadata(project.input.audio_asset_id), *stored_catalog.values(), storage.get_metadata(narration.audio_asset_id))}
        timeline = SceneCompiler().compile(project, storyboard, narration, [], assets, {}, audio_kind="sfx")
        SemanticValidator().validate(project, storyboard, narration, [], timeline, set(assets))
        result = LocalFfmpegRenderer(storage, output_root=args.output_root, ffmpeg_binary=args.ffmpeg_binary, ffprobe_binary=args.ffprobe_binary).render(timeline)
        transition(machine, project, EventType.RENDER_QA_PASSED, result.output_digest)
        print(result.output_path)
        return 0
    if args.input_audio is None:
        raise ValueError("--input-audio is required unless --resume-review is used")
    if not args.input_audio.is_file():
        raise FileNotFoundError(f"input audio is missing: {args.input_audio}")
    storage = ArtifactManager(args.artifact_root)
    workflow_path = args.artifact_root / "phase2-workflow.db"
    records = WorkflowStore(workflow_path)
    audio = storage.store_bytes(args.input_audio.read_bytes(), filename=args.input_audio.name, media_type="audio/*")
    transcript = LocalWhisperAdapter(storage, model_name=args.whisper_model).transcribe(audio)
    transcript_asset = storage.store_contract({"source": "local_whisper", "text": transcript.text, "language": transcript.language,
        "words": [{"text": word.text, "start_seconds": word.start_seconds, "end_seconds": word.end_seconds} for word in transcript.words]})
    scenario_config = SCENARIOS[args.scenario]
    project = make_project(args, audio.asset_id, transcript.text, scenario_config["character_id"])
    records.save(project.project_id, project.version, "project", project)
    records.save(project.project_id, project.version, "transcript", {"asset_id": transcript_asset.asset_id, "text": transcript.text})
    machine = WorkflowStateMachine(SqliteEventRepository(workflow_path))
    transition(machine, project, EventType.PLANNING_STARTED, transcript_asset.digest)

    planner = GroqTextPlanner()
    draft = planner.plan_draft(transcript.text, project_id=project.project_id, project_version=project.version,
        transcript_asset_id=transcript_asset.asset_id,
        catalog_description=f"{scenario_config['catalog_description']} Produce exactly one scene with duration_frames {args.duration * args.fps}.")
    pack = Phase1AssetGenerator(args.assets_root).generate(scenario_config["scenario"], geometry=draft.bird_fence_geometry)
    catalog = bird_fence_catalog(pack)
    stored_catalog = {key: storage.store_bytes(entry.path.read_bytes(), filename=entry.path.name, media_type="image/png") for key, entry in catalog.items()}
    catalog_assets = {key: item.asset_id for key, item in stored_catalog.items()}
    draft_asset = storage.store_contract(draft)
    records.save(project.project_id, project.version, "storyboard_draft", draft)
    records.save(project.project_id, project.version, "planner_provenance", planner.last_provenance)
    transition(machine, project, EventType.CONTRACTS_VALIDATED, draft_asset.digest)
    storyboard = storyboard_from_draft(draft, project, catalog_asset_ids=catalog_assets, user_recording_text=transcript.text)

    # Bird previews and approved renders use only local hop effects, never explanatory speech.
    preview_narration = JumpSfxAgent().generate(project, storyboard, storage)
    assets = {item.asset_id: item for item in (audio, *stored_catalog.values(), storage.get_metadata(preview_narration.audio_asset_id))}
    preview_timeline = SceneCompiler().compile(project, storyboard, preview_narration, [], assets, {}, audio_kind="sfx")
    SemanticValidator().validate(project, storyboard, preview_narration, [], preview_timeline, set(assets))
    preview = PreviewRenderer(LocalFfmpegRenderer(storage, output_root=args.output_root / "previews", ffmpeg_binary=args.ffmpeg_binary,
        ffprobe_binary=args.ffprobe_binary), storage, ffmpeg_binary=args.ffmpeg_binary).render(preview_timeline)
    preview_asset = storage.store_contract(preview)
    records.save(project.project_id, project.version, "preview", preview)
    transition(machine, project, EventType.PREVIEW_CREATED, preview_asset.digest)
    print(f"Preview ready: {storage.absolute_path(storage.get_metadata(preview.preview_asset_id))}")
    print(f"Contact sheet: {storage.absolute_path(storage.get_metadata(preview.contact_sheet_asset_id))}")
    if args.review_action is None:
        if args.serve_review:
            print("Open http://127.0.0.1:8765 to approve or reject this preview.")
            decision = serve_review(workflow_path, project.project_id, project.version)
            if decision is not None:
                args.resume_review = True
                return main(args)
        print("Pipeline paused in awaiting_preview_review. Inspect the preview, then rerun with --review-action approve or --review-action reject.")
        return 0

    if args.review_action == "reject":
        decision = ApprovalDecision(approval_id=f"approval_{preview.preview_id.removeprefix('preview_')}", project_id=project.project_id,
            project_version=project.version, preview_id=preview.preview_id, preview_digest=preview.timeline_digest,
            action=ApprovalAction.REJECTED, feedback=args.review_feedback)
        storage.store_contract(decision)
        records.save(project.project_id, project.version, "approval_decision", decision)
        revision = RevisionRequest(revision_request_id=f"revision_{preview.preview_id.removeprefix('preview_')}", project_id=project.project_id,
            source_project_version=project.version, preview_id=preview.preview_id, preview_digest=preview.timeline_digest, feedback=args.review_feedback)
        transition(machine, project, EventType.PREVIEW_REJECTED, storage.store_contract(revision).digest)
        records.save(project.project_id, project.version, "revision_request", revision)
        print("Preview rejected. A revision request was stored; no final MP4 was rendered.")
        return 0

    decision = ApprovalDecision(approval_id=f"approval_{preview.preview_id.removeprefix('preview_')}", project_id=project.project_id,
        project_version=project.version, preview_id=preview.preview_id, preview_digest=preview.timeline_digest, action=ApprovalAction.APPROVED)
    transition(machine, project, EventType.PREVIEW_APPROVED, storage.store_contract(decision).digest)
    records.save(project.project_id, project.version, "approval_decision", decision)
    transition(machine, project, EventType.PROJECT_APPROVED, preview.timeline_digest)
    narration = JumpSfxAgent().generate(project, storyboard, storage)
    assets[narration.audio_asset_id] = storage.get_metadata(narration.audio_asset_id)
    timeline = SceneCompiler().compile(project, storyboard, narration, [], assets, {}, audio_kind="sfx")
    SemanticValidator().validate(project, storyboard, narration, [], timeline, set(assets))
    qa = PreRenderQaEngine().evaluate(timeline, narration, set(assets))
    if not qa.passed:
        transition(machine, project, EventType.REPAIR_BUDGET_EXHAUSTED, "qa_failed")
        raise RuntimeError("pre-render QA failed: " + "; ".join(qa.failures))
    result = LocalFfmpegRenderer(storage, output_root=args.output_root, ffmpeg_binary=args.ffmpeg_binary,
        ffprobe_binary=args.ffprobe_binary).render(timeline)
    transition(machine, project, EventType.RENDER_QA_PASSED, result.output_digest)
    print(result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
