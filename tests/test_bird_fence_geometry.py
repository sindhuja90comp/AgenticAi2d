"""Regression coverage for editable geometry, contact, and persisted revisions."""
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from agentic_ai_2d.agents.planner import GroqPlannerError, GroqTextPlanner, storyboard_draft_from_blueprint
from agentic_ai_2d.assets.generator import Phase1AssetGenerator, Scenario
from agentic_ai_2d.models.phase2 import BirdFenceGeometry, PreviewPackage
from agentic_ai_2d.workflow_store import WorkflowStore


def blueprint(geometry):
    parts = ['body', 'left_wing', 'right_wing']
    return {
        'narration_mode': 'synthetic_tts', 'narration_text': 'A bird hops along the fence.',
        'bird_fence_geometry': geometry.model_dump(),
        'asset_requests': [dict(catalog_key='catalog_fence_v1', kind='background')]
        + [dict(catalog_key=f'catalog_bird_{part}_v1', kind='character_part',
                character_id='char_bird_v1', required_parts=[part]) for part in parts],
        'scenes': [dict(sequence=1, duration_frames=450, background_catalog_key='catalog_fence_v1',
                        actors=[dict(catalog_key=f'catalog_bird_{part}_v1', character_id='char_bird_v1',
                                     x_percent=50, y_percent=80, scale=1, z_index=20+i,
                                     motion=dict(kind='bird_hop', amplitude_percent=10, period_frames=112))
                                for i, part in enumerate(parts)])],
    }


@pytest.mark.parametrize('change', [
    {'pole_top_y': 50}, {'pole_bottom_y': 700}, {'pole_x_positions': [-1]},
    {'pole_x_positions': [80, 90]}, {'hop_start_x_percent': 0},
    {'bird_scale': 1.5}, {'hop_count': 13}, {'rod_thickness': float('nan')},
])
def test_invalid_geometry_is_rejected_before_generation(change):
    with pytest.raises(ValidationError):
        BirdFenceGeometry(**change)


def test_null_geometry_cannot_bypass_alignment_in_new_blueprints():
    data = blueprint(BirdFenceGeometry())
    data['bird_fence_geometry'] = None
    with pytest.raises(GroqPlannerError):
        storyboard_draft_from_blueprint(data, project_id='proj_geometry01', project_version=1,
                                       transcript_asset_id='asset_transcript_12345678')


@pytest.mark.parametrize('include_geometry', [True, False])
def test_live_planner_requires_explicit_shared_geometry(include_geometry):
    data = blueprint(BirdFenceGeometry())
    if not include_geometry:
        del data['bird_fence_geometry']
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    planner = GroqTextPlanner(client=client)
    kwargs = dict(project_id='proj_geometry01', project_version=1,
                  transcript_asset_id='asset_transcript_12345678',
                  catalog_description='catalog_bird_body_v1 catalog_bird_left_wing_v1 catalog_bird_right_wing_v1')
    if include_geometry:
        assert planner.plan_draft('A bird hops.', **kwargs).bird_fence_geometry == BirdFenceGeometry()
    else:
        with pytest.raises(GroqPlannerError, match='explicit bird_fence_geometry'):
            planner.plan_draft('A bird hops.', **kwargs)


@pytest.mark.parametrize('geometry', [BirdFenceGeometry(), BirdFenceGeometry(
    pole_x_positions=[200, 600, 1000, 1400], pole_top_y=850, pole_bottom_y=1050,
    rod_left_x=100, rod_right_x=1800, rod_thickness=60, bird_scale=0.6,
    hop_start_x_percent=30, hop_end_x_percent=70, hop_height=90,
)])
def test_fence_edges_touch_and_scaled_feet_land_on_rod(tmp_path, geometry):
    pack = Phase1AssetGenerator(tmp_path).generate(Scenario.BIRD_FENCE, geometry=geometry)
    with Image.open(pack.background) as background:
        x = geometry.pole_x_positions[0]
        # Above the rod is landscape; rod bottom and pole top occupy adjacent rows.
        assert background.getpixel((x, geometry.rod_top_y - 1))[:3] in [(204, 236, 255), (145, 198, 108)]
        assert background.getpixel((x, geometry.rod_top_y))[:3] == (93, 54, 32)
        assert background.getpixel((x, geometry.pole_top_y - 1))[:3] == (93, 54, 32)
        assert background.getpixel((x, geometry.pole_top_y))[:3] == (93, 54, 32)
        assert background.getpixel((x - 1, geometry.pole_top_y))[:3] != (93, 54, 32)
    with Image.open(pack.parts['body']) as body:
        foot_edge = body.getchannel('A').getbbox()[3]
        for width, height in [(1920, 1080), (1280, 720)]:
            size = round(width * 0.35 * geometry.bird_scale)
            center_y = Phase1AssetGenerator.bird_landing_y_percent(geometry, width, height) * height / 100
            rendered_foot = center_y - size / 2 + foot_edge * size / body.height
            assert rendered_foot == pytest.approx(geometry.rod_top_y * height / 1080)


def test_initial_revision_and_approval_use_persisted_geometry(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('geometry_pipeline', Path(__file__).parents[1] / 'run_pipeline.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    original = BirdFenceGeometry()
    revised = BirdFenceGeometry(pole_top_y=860, rod_thickness=70, hop_height=100,
                               hop_start_x_percent=32, hop_end_x_percent=68, hop_period_frames=36)
    plans = iter([original, revised])
    instructions, previews, renders = [], [], []
    sound_starts = []
    synthesize = pipeline.JumpSfxAgent._jump_effects_wav

    def capture_sound(self, duration_seconds, fps, *, hop_start_frames=None):
        sound_starts.append(hop_start_frames)
        return synthesize(self, duration_seconds, fps, hop_start_frames=hop_start_frames)

    monkeypatch.setattr(pipeline.JumpSfxAgent, '_jump_effects_wav', capture_sound)

    class Planner:
        last_provenance = {}

        def plan_draft(self, instruction, **kwargs):
            instructions.append(instruction)
            kwargs.pop('catalog_description')
            return storyboard_draft_from_blueprint(blueprint(next(plans)), **kwargs)

    class Preview:
        def __init__(self, renderer, storage, **kwargs):
            self.storage = storage

        def render(self, timeline):
            previews.append(timeline)
            payload = timeline.model_dump_json().encode()
            digest = sha256(payload).hexdigest()
            asset = self.storage.store_bytes(payload, filename='preview.json', media_type='application/json')
            return PreviewPackage(preview_id=f'preview_{digest[:16]}', project_id=timeline.project_id,
                                  project_version=timeline.project_version, timeline_digest=digest,
                                  preview_asset_id=asset.asset_id, contact_sheet_asset_id=asset.asset_id,
                                  review_frame_asset_ids=[asset.asset_id]*3, review_frame_numbers=[0,225,449])

    class Renderer:
        def __init__(self, *args, **kwargs):
            pass

        def render(self, timeline):
            renders.append(timeline)
            return SimpleNamespace(output_path=tmp_path / 'test.mp4', output_digest='a'*64)

    monkeypatch.setattr(pipeline, 'GroqTextPlanner', Planner)
    monkeypatch.setattr(pipeline, 'PreviewRenderer', Preview)
    monkeypatch.setattr(pipeline, 'LocalFfmpegRenderer', Renderer)
    monkeypatch.setattr(pipeline, 'LocalWhisperAdapter', lambda *a, **k: SimpleNamespace(
        transcribe=lambda audio: SimpleNamespace(text='A bird hops.', language='en', words=[])))
    audio = tmp_path / 'instruction.wav'
    audio.write_bytes(b'fake input for mocked transcriber')
    monkeypatch.setattr('sys.argv', ['run_pipeline.py', '--project-id', 'proj_geometry01',
                                   '--input-audio', str(audio), '--artifact-root', str(tmp_path/'artifacts'),
                                   '--assets-root', str(tmp_path/'assets'), '--output-root', str(tmp_path/'exports')])
    args = pipeline.parse_args()
    assert pipeline.main(args) == 0
    store = WorkflowStore(args.artifact_root / 'phase2-workflow.db')

    def decide(version, action):
        preview = store.get(args.project_id, version, 'preview')
        store.save(args.project_id, version, 'approval_decision', dict(
            approval_id=f'approval_geometry0{version}', project_id=args.project_id, project_version=version,
            preview_id=preview['preview_id'], preview_digest=preview['timeline_digest'],
            action=action, feedback='Raise the fence and increase bounce height.' if action == 'rejected' else None))

    decide(1, 'rejected')
    args.resume_review = True
    assert pipeline.main(args) == 0
    assert json.loads(instructions[1].split('Current geometry settings: ')[1].split('\n\n')[0]) == original.model_dump()
    assert store.get(args.project_id, 2, 'storyboard_draft')['bird_fence_geometry'] == revised.model_dump()
    assert previews[0].visual_tracks[0].clips[0].asset_id != previews[1].visual_tracks[0].clips[0].asset_id
    for timeline, geometry in zip(previews, [original, revised]):
        clips = [track.clips[0] for track in timeline.visual_tracks if track.kind.value == 'character_pose']
        assert len(clips) == 3
        assert len({(c.x_percent, c.y_percent, c.scale) for c in clips}) == 1
        assert all(c.motion.start_x_percent == geometry.hop_start_x_percent for c in clips)
        assert all(c.motion.amplitude_percent == pytest.approx(geometry.hop_height / 1080 * 100) for c in clips)
        assert all(c.motion.period_frames == geometry.hop_period_frames for c in clips)
        assert all(c.y_percent == pytest.approx(Phase1AssetGenerator.bird_landing_y_percent(geometry)) for c in clips)
    decide(2, 'approved')
    args.project_version = 2
    assert pipeline.main(args) == 0
    assert renders[0] == previews[1]
    assert sound_starts == [[0,24,48,72], [0,36,72,108], [0,36,72,108]]
    store.close()


def test_real_render_contacts_rod_at_start_and_between_hops(tmp_path):
    import subprocess
    import wave
    from io import BytesIO
    from agentic_ai_2d.models.storyboard import MotionSpec
    from agentic_ai_2d.models.timeline import Timeline, VisualTrack, VisualClip, AudioTrack
    from agentic_ai_2d.renderer import LocalFfmpegRenderer
    from agentic_ai_2d.storage import ArtifactManager

    geometry = BirdFenceGeometry(hop_count=2)
    pack = Phase1AssetGenerator(tmp_path/'assets').generate(Scenario.BIRD_FENCE, geometry=geometry)
    storage = ArtifactManager(tmp_path/'artifacts')
    background = storage.store_bytes(pack.background.read_bytes(), filename='background.png', media_type='image/png')
    bird = storage.store_bytes(pack.parts['body'].read_bytes(), filename='body.png', media_type='image/png')
    with BytesIO() as data:
        with wave.open(data, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(48000)
            audio.writeframes(b'\0' * (48000 * 2 * 2))
        sound = storage.store_bytes(data.getvalue(), filename='silence.wav', media_type='audio/wav')
    timeline = Timeline(timeline_id='timeline_contact01', project_id='proj_contact01', project_version=1,
                        fps=30, width=1280, height=720, duration_frames=60,
                        visual_tracks=[
                            VisualTrack(track_id='vtrack_background01', kind='background', z_index=0,
                                        clips=[VisualClip(clip_id='clip_background01', asset_id=background.asset_id,
                                                          start_frame=0, duration_frames=60)]),
                            VisualTrack(track_id='vtrack_character01', kind='character_pose', z_index=20,
                                        clips=[VisualClip(clip_id='clip_character01', asset_id=bird.asset_id,
                                                          start_frame=0, duration_frames=60, scale=geometry.bird_scale,
                                                          x_percent=38, y_percent=49,
                                                          motion=MotionSpec(kind='bird_hop', amplitude_percent=5,
                                                                            period_frames=24, hop_count=2,
                                                                            start_x_percent=38, end_x_percent=62,
                                                                            landing_surface_y_percent=geometry.rod_top_y/1080*100,
                                                                            foot_y_percent=Phase1AssetGenerator.bird_foot_y_percent()))]),
                        ], audio_tracks=[AudioTrack(track_id='atrack_contact01', kind='sfx', asset_id=sound.asset_id,
                                                   start_frame=0, duration_frames=60)])
    result = LocalFfmpegRenderer(storage, output_root=tmp_path/'exports').render(timeline)
    for frame, expected_foot in [(0,460), (12,424), (24,460), (48,460)]:
        path = tmp_path/f'frame-{frame}.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(result.output_path), '-vf',
                        f'select=eq(n\\,{frame})', '-frames:v', '1', str(path)], check=True, capture_output=True)
        with Image.open(path) as image:
            # Brown foot pixels above the fence, excluding the rod itself. Allow a
            # pixel of resampling/codec antialiasing at the geometric contact edge.
            foot_rows = [y for y in range(380,460) if any(
                r > b * 1.3 and g < 120
                for r,g,b in [image.getpixel((x,y))[:3] for x in range(250,1000)])]
            assert abs((max(foot_rows) + 1) - expected_foot) <= 2
