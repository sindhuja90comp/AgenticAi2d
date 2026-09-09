"""Measure Groq planner latency and local contract validity for Phase 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from agentic_ai_2d.agents import GROQ_PLANNER_MODEL, GroqTextPlanner


CATALOG = "catalog_fence_v1: background; catalog_bird_body_v1: character_part, character char_bird_v1, parts body; catalog_bird_left_wing_v1: character_part, character char_bird_v1, parts left_wing. Produce exactly one scene with duration_frames 450."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("var/benchmarks/groq-phase2.json"))
    args = parser.parse_args()
    results = []
    for _ in range(args.runs):
        planner = GroqTextPlanner()
        started = time.perf_counter()
        try:
            planner.plan_draft("A bird hops along a fence.", project_id="proj_benchmark01", project_version=1,
                               transcript_asset_id="asset_transcript_12345678", catalog_description=CATALOG)
            results.append({"valid": True, **planner.last_provenance, "wall_seconds": round(time.perf_counter() - started, 3)})
        except Exception as error:
            results.append({"valid": False, "error": str(error), "wall_seconds": round(time.perf_counter() - started, 3)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": GROQ_PLANNER_MODEL, "runs": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
