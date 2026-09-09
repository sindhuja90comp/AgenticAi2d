from pathlib import Path
import runpy

from agentic_ai_2d.models.phase2 import StoryboardDraft


def test_ollama_benchmark_uses_phase2_schema() -> None:
    module = runpy.run_path(Path("tools/benchmark_ollama.py"), run_name="benchmark_ollama_test")

    assert "butterfly" in module["PROMPT"].lower()
    assert StoryboardDraft.schema_id.endswith("StoryboardDraft.schema.json")
    assert callable(module["benchmark_model"])
