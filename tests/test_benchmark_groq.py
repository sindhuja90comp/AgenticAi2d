import runpy
from pathlib import Path


def test_groq_benchmark_declares_supported_model() -> None:
    module = runpy.run_path(Path("tools/benchmark_groq.py"), run_name="benchmark_groq_test")
    assert module["GROQ_PLANNER_MODEL"] == "openai/gpt-oss-20b"
