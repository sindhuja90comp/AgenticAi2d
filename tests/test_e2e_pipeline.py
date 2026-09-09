from datetime import datetime, timezone

from agentic_ai_2d.agents import DeterministicTtsAgent, DeterministicWriterAgent, SimulatedRhubarbAdapter
from agentic_ai_2d.compiler import SceneCompiler
from agentic_ai_2d.events import EventType, InMemoryEventRepository
from agentic_ai_2d.ingestion import StaticWhisperAdapter
from agentic_ai_2d.models.project_spec import ProjectSpec, ProjectStatus
from agentic_ai_2d.qa import PreRenderQaEngine
from agentic_ai_2d.renderer import SimulatedRemotionRenderer
from agentic_ai_2d.semantic_validator import SemanticValidator
from agentic_ai_2d.state_machine import WorkflowStateMachine
from agentic_ai_2d.storage import ArtifactManager


def test_e2e_pipeline_creates_a_valid_completed_render(tmp_path) -> None:
    storage = ArtifactManager(tmp_path / "artifacts")
    input_audio = storage.store_bytes(b"recorded voice", filename="instruction.wav", media_type="audio/wav")
    project = ProjectSpec.model_validate({
        "project_id": "proj_12345678", "version": 1, "status": "draft", "created_at": datetime.now(timezone.utc),
        "input": {"audio_asset_id": input_audio.asset_id, "user_prompt": "Make a rabbit story.", "language": "en"},
        "output": {"aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": 30, "format": "mp4"},
        "creative_constraints": {"target_duration_seconds": 15, "style_preset": "storybook_2d_flat", "character_ids": ["char_rabbit_v1"], "narration_source": "synthetic_tts", "captions_enabled": True, "user_approval_required": True}
    })
    background = storage.store_bytes(b"background", filename="background.png", media_type="image/png")
    pose = storage.store_bytes(b"pose", filename="rabbit-pose.png", media_type="image/png")
    mouth = storage.store_bytes(b"mouth", filename="rabbit-mouth.png", media_type="image/png")
    assets = {metadata.asset_id: metadata for metadata in [input_audio, background, pose, mouth]}

    transcript = StaticWhisperAdapter("A rabbit shares a carrot", 15).transcribe(input_audio)
    storyboard = DeterministicWriterAgent().generate(project, transcript, background_asset_id=background.asset_id, pose_asset_id=pose.asset_id)
    narration = DeterministicTtsAgent().generate(project, storyboard, storage)
    assets[narration.audio_asset_id] = storage.get_metadata(narration.audio_asset_id)
    viseme = SimulatedRhubarbAdapter().generate(project, narration, character_id="char_rabbit_v1", mouth_set_id="mouthset_rabbit_v1")
    timeline = SceneCompiler().compile(project, storyboard, narration, [viseme], assets, {"char_rabbit_v1": mouth.asset_id})

    SemanticValidator().validate(project, storyboard, narration, [viseme], timeline, set(assets))
    qa = PreRenderQaEngine().evaluate(timeline, narration, set(assets))
    assert qa.passed, qa.failures

    repository = InMemoryEventRepository()
    machine = WorkflowStateMachine(repository)
    for event_type in [EventType.STORYBOARD_CREATED, EventType.PROJECT_APPROVED, EventType.CONTRACTS_VALIDATED, EventType.ASSETS_CACHED_AND_VALIDATED]:
        machine.transition(project, event_type, actor="pipeline", correlation_id="e2e", idempotency_key=event_type.value, payload_digest="sha256:e2e")

    result = SimulatedRemotionRenderer(tmp_path / "renders").render(timeline)
    machine.transition(project, EventType.RENDER_QA_PASSED, actor="renderer", correlation_id="e2e", idempotency_key="render-qa", payload_digest=result.output_digest)

    assert result.output_path.name == "output.mp4"
    assert result.output_path.exists()
    assert result.manifest_path.name == "render_manifest.json"
    assert result.manifest_path.exists()
    assert machine.current_status(project) is ProjectStatus.COMPLETED
