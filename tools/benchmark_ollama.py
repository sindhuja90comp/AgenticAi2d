#!/usr/bin/env python3
"""Benchmark local Ollama models against the Phase 2 StoryboardDraft schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from pydantic import ValidationError

from agentic_ai_2d.models.phase2 import StoryboardDraft


PROMPT = """Return exactly one valid StoryboardDraft JSON object for this request: A butterfly rests above a flower that sways in the breeze.

Use project_id "proj_butterfly01", project_version 1, transcript_asset_id "asset_transcript_12345678", and narration_mode "synthetic_tts".
Provide narration_text. Create exactly three asset requests:
1. A background request with kind "background" and required_parts as an empty list.
2. A butterfly request with kind "character_part", character_id "char_butterfly_v1", and required_parts ["body", "left_wing", "right_wing"].
3. A flower request with kind "character_part", character_id "char_flower_v1", and required_parts ["head", "stem"].

Create exactly one scene. Every x_percent and y_percent must be between 0 and 100. Place the butterfly above the flower with a higher z_index. Set the flower anchor_y_percent to 100 and use flower_sway motion. Use butterfly_wing_flap motion for the butterfly.
Return JSON only and obey the provided JSON schema exactly."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="Ollama model tag to benchmark; repeat for each candidate.")
    parser.add_argument("--runs", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--ollama-binary", default="ollama")
    parser.add_argument("--output", type=Path, default=Path("var/benchmarks/ollama-phase2.json"))
    return parser.parse_args()


def run_model(binary: str, model: str, prompt: str = PROMPT) -> tuple[float, dict[str, object] | None, str | None]:
    started_at = time.perf_counter()
    try:
        completed = subprocess.run(
            [binary, "run", model, "--format", json.dumps(StoryboardDraft.model_json_schema()), prompt],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        return elapsed_seconds, None, "model response exceeded the 180-second benchmark timeout"
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    if completed.returncode != 0:
        return elapsed_seconds, None, completed.stderr.strip() or "ollama returned a non-zero exit code"
    try:
        return elapsed_seconds, json.loads(completed.stdout), None
    except json.JSONDecodeError as error:
        return elapsed_seconds, None, f"invalid JSON: {error}"


def benchmark_model(binary: str, model: str, runs: int) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for _ in range(runs):
        elapsed_seconds, payload, error = run_model(binary, model)
        valid = False
        repair_elapsed_seconds: float | None = None
        repaired_valid = False
        if payload is not None:
            try:
                StoryboardDraft.model_validate(payload)
                valid = True
            except ValidationError as validation_error:
                error = str(validation_error)
                repair_prompt = (
                    "Correct this JSON so it validates against the supplied schema. Return JSON only. "
                    f"Validation error: {error}\nPrior JSON: {json.dumps(payload, separators=(',', ':'))}"
                )
                repair_elapsed_seconds, repaired_payload, repair_error = run_model(binary, model, repair_prompt)
                if repaired_payload is not None:
                    try:
                        StoryboardDraft.model_validate(repaired_payload)
                        repaired_valid = True
                    except ValidationError as repaired_validation_error:
                        repair_error = str(repaired_validation_error)
                if not repaired_valid:
                    error = repair_error or error
        results.append(
            {
                "elapsed_seconds": elapsed_seconds,
                "schema_valid": valid,
                "repair_elapsed_seconds": repair_elapsed_seconds,
                "repaired_schema_valid": repaired_valid,
                "error": error,
            }
        )
    valid_runs = sum(1 for result in results if result["schema_valid"])
    valid_after_repair_runs = sum(
        1 for result in results if result["schema_valid"] or result["repaired_schema_valid"]
    )
    return {
        "model": model,
        "runs": results,
        "valid_schema_rate": valid_runs / runs,
        "valid_after_repair_rate": valid_after_repair_runs / runs,
        "mean_latency_seconds": round(sum(result["elapsed_seconds"] for result in results) / runs, 3),
    }


def main() -> int:
    args = parse_args()
    try:
        version = subprocess.run(
            [args.ollama_binary, "--version"], check=False, capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError as error:
        raise RuntimeError("Ollama is unavailable; install it before benchmarking") from error
    if version.returncode != 0:
        raise RuntimeError("Ollama is unavailable; install it before benchmarking")
    payload = {
        "benchmark": "phase2_storyboard_draft",
        "ollama_version": version.stdout.strip(),
        "schema_id": StoryboardDraft.schema_id,
        "models": [benchmark_model(args.ollama_binary, model, args.runs) for model in args.model],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
