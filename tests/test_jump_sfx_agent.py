import wave
from io import BytesIO
from types import SimpleNamespace

from agentic_ai_2d.agents.jump_sfx_agent import JumpSfxAgent


def test_long_text_alignment_stays_ordered_and_inside_scene():
    scene = SimpleNamespace(start_frame=100, duration_frames=450,
                            narration=SimpleNamespace(text=' '.join(['word'] * 83), utterance_id='utt_longtext01'))
    utterance = JumpSfxAgent._align_scene(scene)
    assert utterance.words[0].start_frame == 100
    assert utterance.words[-1].end_frame == 549
    assert all(100 <= word.start_frame <= word.end_frame <= 549 for word in utterance.words)
    assert all(a.end_frame < b.start_frame for a,b in zip(utterance.words, utterance.words[1:]))


def test_jump_sfx_has_four_effects_at_the_four_hop_start_frames() -> None:
    agent = JumpSfxAgent()
    assert agent.HOP_START_FRAMES == (0, 24, 48, 72)
    audio = agent._jump_effects_wav(duration_seconds=3.2, fps=30)
    with wave.open(BytesIO(audio), "rb") as wav:
        assert wav.getframerate() == 48_000
        assert wav.getnframes() == 153_600
        samples = wav.readframes(wav.getnframes())
    # Each hop start contains the deliberate click/tone; the gaps remain silent.
    for frame in agent.HOP_START_FRAMES:
        offset = int(frame / 30 * agent.SAMPLE_RATE_HZ) * 2
        assert samples[offset:offset + 300] != b"\x00" * 300
    quiet_offset = int(12 / 30 * agent.SAMPLE_RATE_HZ) * 2
    assert samples[quiet_offset:quiet_offset + 300] == b"\x00" * 300
