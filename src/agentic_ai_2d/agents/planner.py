"""Text-only Groq planner for bounded Phase 2 animation blueprints."""

from __future__ import annotations

import json
import os
import time
from hashlib import sha256
from typing import Any

from pydantic import ValidationError

from ..models.phase2 import BirdFenceGeometry, NarrationMode, StoryboardDraft
from ..assets.generator import Phase1AssetGenerator
from ..models.project_spec import ProjectSpec
from ..models.storyboard import (
    ActorCue, ActorPosition, CameraCue, CameraPreset, MotionKind, MotionSpec, Narration, Scene, Storyboard,
)


GROQ_PLANNER_MODEL = "openai/gpt-oss-20b"
_MAX_COMPLETION_TOKENS = 3_072
_BIRD_CATALOG_KEYS = [
    "catalog_fence_v1",
    "catalog_bird_body_v1",
    "catalog_bird_left_wing_v1",
    "catalog_bird_right_wing_v1",
]

_BLUEPRINT_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["narration_mode", "narration_text", "asset_requests", "scenes"],
    "properties": {
        "narration_mode": {"type": "string", "enum": ["synthetic_tts"]},
        "narration_text": {"type": "string", "minLength": 1, "maxLength": 1000},
        "asset_requests": {"type": "array", "minItems": 1, "maxItems": 32, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["catalog_key", "kind", "character_id", "required_parts"],
            "properties": {
                "catalog_key": {"type": "string", "enum": _BIRD_CATALOG_KEYS},
                "kind": {"type": "string", "enum": ["background", "character_part", "prop"]},
                "character_id": {"type": ["string", "null"], "enum": ["char_bird_v1", None]},
                "required_parts": {"type": "array", "items": {"type": "string", "enum": ["body", "left_wing", "right_wing", "head", "stem"]}},
            },
        }},
        "scenes": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["sequence", "duration_frames", "background_catalog_key", "actors"],
            "properties": {
                "sequence": {"type": "integer", "minimum": 1, "maximum": 3},
                "duration_frames": {"type": "integer", "minimum": 1, "maximum": 1350},
                "background_catalog_key": {"type": "string", "enum": ["catalog_fence_v1"]},
                "actors": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["catalog_key", "character_id", "x_percent", "y_percent", "scale", "z_index", "anchor_x_percent", "anchor_y_percent", "motion"],
                    "properties": {
                        "catalog_key": {"type": "string", "enum": _BIRD_CATALOG_KEYS[1:]}, "character_id": {"type": "string", "enum": ["char_bird_v1"]},
                        "x_percent": {"type": "number", "minimum": 0, "maximum": 100}, "y_percent": {"type": "number", "minimum": 0, "maximum": 100},
                        "scale": {"type": "number", "minimum": 0.1, "maximum": 3}, "z_index": {"type": "integer", "minimum": 0, "maximum": 100},
                        "anchor_x_percent": {"type": "number", "minimum": 0, "maximum": 100}, "anchor_y_percent": {"type": "number", "minimum": 0, "maximum": 100},
                        "motion": {"type": "object", "additionalProperties": False,
                            "required": ["kind", "amplitude_percent", "period_frames"], "properties": {
                                "kind": {"type": "string", "enum": ["static", "butterfly_wing_flap", "flower_sway", "bird_hop"]},
                                "amplitude_percent": {"type": "number", "minimum": 0, "maximum": 30},
                                "period_frames": {"type": "integer", "minimum": 1, "maximum": 600},
                            }},
                    },
                }},
            },
        }},
    },
}

_SYSTEM_PROMPT = """You are the structural planner for a bounded 2D animation pipeline.
Return one JSON object only: no Markdown, explanation, shell commands, FFmpeg filters,
file paths, URLs, asset IDs, or executable code. Use only the approved catalog described
by the user. Produce a blueprint with bird_fence_geometry, narration_mode, narration_text,
asset_requests, and scenes. Every asset request must contain catalog_key, kind, character_id, and required_parts.
Every scene must contain sequence, duration_frames, background_catalog_key, and actors.
Every actor must contain catalog_key, character_id, x_percent, y_percent, scale, z_index,
anchor_x_percent, anchor_y_percent, and motion. Motion must contain kind, amplitude_percent,
and period_frames. Use these exact enum values only: narration_mode must be "synthetic_tts";
motion.kind must be "static", "butterfly_wing_flap", "flower_sway", or "bird_hop".
narration_text must be a non-empty English sentence. Do not use the words "silent", "sway",
or "flap" as enum values. Keep all values within the requested bounds. When the approved
catalog contains bird body, left-wing, and right-wing entries, request all three and place all
three as separate actors. For a bird that hops, all three parts use bird_hop together; do not
use butterfly_wing_flap. Use a left-wing pivot left of center and a right-wing pivot right of
center, near each wing root."""

# The remote schema and local validator share the same coordinate definitions.
_geometry_schema = BirdFenceGeometry.model_json_schema()
_geometry_schema["required"] = list(_geometry_schema["properties"])
for _property in _geometry_schema["properties"].values():
    _property.pop("default", None)
_BLUEPRINT_SCHEMA["properties"]["bird_fence_geometry"] = _geometry_schema
_BLUEPRINT_SCHEMA["required"].append("bird_fence_geometry")
_SYSTEM_PROMPT += """
Always include bird_fence_geometry, the authoritative shared bird rig and fence settings.
Use character_id "char_bird_v1" for every bird request and actor, and null for the background.
Coordinates use a 1920x1080 canvas, with y increasing downward. pole_x_positions are
pole left edges; pole_top_y and pole_bottom_y define their vertical span. The horizontal
rod spans rod_left_x to rod_right_x; its bottom is pole_top_y and its top is
pole_top_y minus rod_thickness. Every pole must fit under the rod without overlapping.
The bird's visible feet land exactly on rod top automatically; do not add a separate
landing offset. Adjust pole_top_y or rod_thickness to move that surface. bird_scale,
hop_start_x_percent, hop_end_x_percent, hop_height (pixels), hop_period_frames and
hop_count control the entire multipart bird together. These settings supersede individual
bird actor placements and motion; keep all parts registered on their shared sprite canvas.
Keep the full bird within the rod endpoints and above the canvas bottom, including its
hop peak. Default settings: pole_x_positions [80,330,580,830,1080,1330,1580,1830],
pole_top_y 790, pole_bottom_y 1030, pole_width 55, rod_left_x 0, rod_right_x 1920,
rod_thickness 100, bird_scale 0.68, hop_start_x_percent 38, hop_end_x_percent 62,
hop_height 54, hop_period_frames 24, hop_count 4. On revision preserve current settings
unless feedback requests changing them. Apply geometry feedback through these fields.
"""
_SYSTEM_PROMPT += "\nRequired output JSON Schema (also applies in JSON-object mode):\n" + json.dumps(_BLUEPRINT_SCHEMA, separators=(",", ":"))
_SYSTEM_PROMPT += '\nWrite bird_fence_geometry FIRST, as a top-level object. Example geometry (adapt to the instruction):\n' + json.dumps({"bird_fence_geometry": BirdFenceGeometry().model_dump()})


class GroqPlannerError(RuntimeError):
    """Raised when Groq cannot provide a usable JSON animation blueprint."""


class GroqTextPlanner:
    """Plan a scene remotely while keeping media and execution data on the Mac."""

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        self.last_provenance: dict[str, Any] = {}
        self._request_digest: str | None = None
        self._catalog_digest: str | None = None
        self._response_digest: str | None = None
        self._repair_response_digest: str | None = None
        self._response_mode = "json_schema"
        if client is not None:
            self._client = client
            return

        try:
            from dotenv import load_dotenv
        except ImportError as error:
            raise GroqPlannerError("Install the planner dependencies with: pip install -e '.[planner]'") from error
        load_dotenv()
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise GroqPlannerError("GROQ_API_KEY is required for the Groq text planner")

        try:
            from groq import Groq
        except ImportError as error:
            raise GroqPlannerError("Install the planner dependencies with: pip install -e '.[planner]'") from error
        self._client = Groq(api_key=api_key)

    def plan(self, instruction: str, *, catalog_description: str = "") -> dict[str, Any]:
        """Return a JSON-only blueprint from text-only planning context.

        The caller must pass a plain-language instruction and an optional plain-language
        approved-catalog description. Audio, images, local paths, artifact IDs, and render
        settings are intentionally outside this interface.
        """
        instruction = instruction.strip()
        catalog_description = catalog_description.strip()
        self._response_mode = "json_schema"
        if not instruction:
            raise ValueError("instruction must not be empty")

        user_message = f"Instruction: {instruction}"
        if catalog_description:
            user_message += f"\n\nApproved catalog: {catalog_description}"
        self._request_digest = _text_digest(f"{_SYSTEM_PROMPT}\n\n{user_message}")
        self._catalog_digest = _text_digest(catalog_description)

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
        try:
            response = self._client.chat.completions.create(
                model=GROQ_PLANNER_MODEL, messages=messages,
                response_format={"type": "json_schema", "json_schema": {"name": "animation_blueprint", "strict": True, "schema": _BLUEPRINT_SCHEMA}},
                temperature=0, max_completion_tokens=_MAX_COMPLETION_TOKENS,
            )
            content = response.choices[0].message.content
        except Exception as error:
            if "json_validate_failed" not in str(error):
                raise GroqPlannerError(f"Groq planning request failed: {error}") from error
            # Groq can reject strict-schema decoding before returning a response. Keep the
            # same text-only boundary and validate the fallback locally against Pydantic.
            try:
                response = self._client.chat.completions.create(
                    model=GROQ_PLANNER_MODEL, messages=messages, response_format={"type": "json_object"},
                    temperature=0, max_completion_tokens=_MAX_COMPLETION_TOKENS,
                )
                content = response.choices[0].message.content
                self._response_mode = "json_object_fallback"
            except Exception as fallback_error:
                raise GroqPlannerError(f"Groq planning fallback failed: {fallback_error}") from fallback_error
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise GroqPlannerError("Groq returned an incomplete planning response") from error

        if not content:
            raise GroqPlannerError("Groq returned an empty planning response")
        self._response_digest = _text_digest(content)
        try:
            blueprint = json.loads(content)
        except json.JSONDecodeError as error:
            raise GroqPlannerError("Groq returned invalid JSON") from error
        if not isinstance(blueprint, dict):
            raise GroqPlannerError("Groq blueprint must be a JSON object")
        return blueprint

    def plan_draft(
        self,
        instruction: str,
        *,
        project_id: str,
        project_version: int,
        transcript_asset_id: str,
        catalog_description: str,
    ) -> StoryboardDraft:
        """Turn an untrusted text-only Groq response into a local contract."""
        started = time.perf_counter()
        self._response_digest = None
        self._repair_response_digest = None
        try:
            blueprint = self.plan(instruction, catalog_description=catalog_description)
        except GroqPlannerError as error:
            self._record_provenance(started, repair_attempted=False, valid=False, validation_error=str(error))
            raise
        try:
            draft = storyboard_draft_from_blueprint(
                _complete_bird_catalog_blueprint(blueprint, catalog_description), project_id=project_id,
                project_version=project_version, transcript_asset_id=transcript_asset_id
            )
            _validate_bird_multipart_plan(draft, catalog_description)
            self._record_provenance(started, repair_attempted=False, valid=True)
            return draft
        except GroqPlannerError as first_error:
            try:
                repaired = self._repair(blueprint, str(first_error), instruction=instruction, catalog_description=catalog_description)
                draft = storyboard_draft_from_blueprint(
                    _complete_bird_catalog_blueprint(repaired, catalog_description), project_id=project_id,
                    project_version=project_version, transcript_asset_id=transcript_asset_id
                )
                _validate_bird_multipart_plan(draft, catalog_description)
            except GroqPlannerError as error:
                self._record_provenance(started, repair_attempted=True, valid=False, validation_error=str(error))
                raise
            self._record_provenance(started, repair_attempted=True, valid=True)
            return draft

    def _repair(self, blueprint: dict[str, Any], error: str, *, instruction: str = "", catalog_description: str = "") -> dict[str, Any]:
        """Allow exactly one text-only repair request for an invalid blueprint."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Original instruction (including revision settings and feedback): {instruction}\nApproved catalog: {catalog_description}\nRepair this JSON blueprint to satisfy the schema and original instruction. Validation error: {error}\nBlueprint: {json.dumps(blueprint)}"},
        ]
        try:
            response = self._client.chat.completions.create(
                model=GROQ_PLANNER_MODEL, messages=messages,
                response_format={"type": "json_schema", "json_schema": {"name": "animation_blueprint", "strict": True, "schema": _BLUEPRINT_SCHEMA}},
                temperature=0, max_completion_tokens=_MAX_COMPLETION_TOKENS,
            )
        except Exception as repair_error:
            if "json_validate_failed" not in str(repair_error):
                raise GroqPlannerError(f"Groq repair request failed: {repair_error}") from repair_error
            try:
                response = self._client.chat.completions.create(
                    model=GROQ_PLANNER_MODEL, messages=messages, response_format={"type": "json_object"},
                    temperature=0, max_completion_tokens=_MAX_COMPLETION_TOKENS,
                )
                self._response_mode = "json_object_fallback"
            except Exception as fallback_error:
                raise GroqPlannerError(f"Groq repair fallback failed: {fallback_error}") from fallback_error
        try:
            content = response.choices[0].message.content
            self._repair_response_digest = _text_digest(content)
            repaired = json.loads(content)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as repair_error:
            raise GroqPlannerError("Groq repair returned invalid JSON") from repair_error
        if not isinstance(repaired, dict):
            raise GroqPlannerError("Groq repair must return a JSON object")
        return repaired

    def _record_provenance(
        self, started: float, *, repair_attempted: bool, valid: bool, validation_error: str | None = None
    ) -> None:
        """Store digests and request metadata without retaining user text or API secrets."""
        self.last_provenance = {
            "provider": "groq",
            "model": GROQ_PLANNER_MODEL,
            "prompt_digest": self._request_digest,
            "catalog_digest": self._catalog_digest,
            "response_digest": self._response_digest,
            "repair_response_digest": self._repair_response_digest,
            "request_settings": {"temperature": 0, "response_format": self._response_mode, "strict": self._response_mode == "json_schema", "max_completion_tokens": _MAX_COMPLETION_TOKENS},
            "repair_attempted": repair_attempted,
            "validation_result": "valid" if valid else "invalid",
            "validation_error": validation_error,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }


def _text_digest(value: str) -> str:
    """Return a stable digest so provenance never stores planning text itself."""
    return sha256(value.encode("utf-8")).hexdigest()


def _validate_bird_multipart_plan(draft: StoryboardDraft, catalog_description: str) -> None:
    """Require all local bird layers when this bounded catalog is offered to the planner."""
    required_keys = {"catalog_bird_body_v1", "catalog_bird_left_wing_v1", "catalog_bird_right_wing_v1"}
    if not all(key in catalog_description for key in required_keys):
        return
    requests = {request.catalog_key: request for request in draft.asset_requests}
    missing = required_keys - set(requests)
    if missing:
        raise GroqPlannerError(f"bird plan is missing required multipart assets: {', '.join(sorted(missing))}")
    for scene in draft.scenes:
        actors = {actor.asset_request_id: actor for actor in scene.actors}
        missing_actor_keys = {key for key in required_keys if requests[key].asset_request_id not in actors}
        if missing_actor_keys:
            raise GroqPlannerError(f"bird scene is missing multipart actors: {', '.join(sorted(missing_actor_keys))}")
        if draft.bird_fence_geometry is not None:
            # Shared geometry supplies motion and registration for every rig part.
            continue
        left = actors[requests["catalog_bird_left_wing_v1"].asset_request_id]
        right = actors[requests["catalog_bird_right_wing_v1"].asset_request_id]
        if left.motion.kind.value != "bird_hop" or right.motion.kind.value != "bird_hop":
            raise GroqPlannerError("all bird parts must use the shared bird_hop motion")
        if left.transform.anchor_x_percent >= 50 or right.transform.anchor_x_percent <= 50:
            raise GroqPlannerError("bird wing pivots must be on their body-side roots")


def _complete_bird_catalog_blueprint(blueprint: dict[str, Any], catalog_description: str) -> dict[str, Any]:
    """Complete omitted fixed catalog layers locally; no new assets or creative content are invented."""
    required_keys = {"catalog_bird_body_v1", "catalog_bird_left_wing_v1", "catalog_bird_right_wing_v1"}
    if not all(key in catalog_description for key in required_keys):
        return blueprint
    if "bird_fence_geometry" not in blueprint:
        raise GroqPlannerError("bird plans require explicit bird_fence_geometry settings, including revisions")
    if not isinstance(blueprint.get("asset_requests"), list) or not isinstance(blueprint.get("scenes"), list):
        return blueprint
    completed = json.loads(json.dumps(blueprint))
    requests = completed["asset_requests"]
    aliases = {"catalog_background_fence_v1": "catalog_fence_v1"}
    for request in requests:
        if isinstance(request, dict):
            request["catalog_key"] = aliases.get(request.get("catalog_key"), request.get("catalog_key"))
    request_keys = {request.get("catalog_key") for request in requests if isinstance(request, dict)}
    templates = {
        "catalog_fence_v1": {"catalog_key": "catalog_fence_v1", "kind": "background", "character_id": None, "required_parts": []},
        "catalog_bird_body_v1": {"catalog_key": "catalog_bird_body_v1", "kind": "character_part", "character_id": "char_bird_v1", "required_parts": ["body"]},
        "catalog_bird_left_wing_v1": {"catalog_key": "catalog_bird_left_wing_v1", "kind": "character_part", "character_id": "char_bird_v1", "required_parts": ["left_wing"]},
        "catalog_bird_right_wing_v1": {"catalog_key": "catalog_bird_right_wing_v1", "kind": "character_part", "character_id": "char_bird_v1", "required_parts": ["right_wing"]},
    }
    requests.extend(template for key, template in templates.items() if key not in request_keys)
    actor_templates = {
        "catalog_bird_body_v1": {"catalog_key": "catalog_bird_body_v1", "character_id": "char_bird_v1", "x_percent": 50, "y_percent": 65, "scale": 0.68, "z_index": 20, "anchor_x_percent": 50, "anchor_y_percent": 50, "motion": {"kind": "bird_hop", "amplitude_percent": 5, "period_frames": 30}},
        "catalog_bird_left_wing_v1": {"catalog_key": "catalog_bird_left_wing_v1", "character_id": "char_bird_v1", "x_percent": 44, "y_percent": 65, "scale": 0.68, "z_index": 18, "anchor_x_percent": 36.3, "anchor_y_percent": 40.3, "motion": {"kind": "bird_hop", "amplitude_percent": 5, "period_frames": 30}},
        "catalog_bird_right_wing_v1": {"catalog_key": "catalog_bird_right_wing_v1", "character_id": "char_bird_v1", "x_percent": 56, "y_percent": 65, "scale": 0.68, "z_index": 19, "anchor_x_percent": 63.7, "anchor_y_percent": 40.3, "motion": {"kind": "bird_hop", "amplitude_percent": 5, "period_frames": 30}},
    }
    for scene in completed["scenes"]:
        if not isinstance(scene, dict) or not isinstance(scene.get("actors"), list):
            continue
        scene["background_catalog_key"] = aliases.get(scene.get("background_catalog_key"), scene.get("background_catalog_key", "catalog_fence_v1"))
        actor_keys = {actor.get("catalog_key") for actor in scene["actors"] if isinstance(actor, dict)}
        scene["actors"].extend(template for key, template in actor_templates.items() if key not in actor_keys)
    return completed


def storyboard_draft_from_blueprint(
    blueprint: dict[str, Any], *, project_id: str, project_version: int, transcript_asset_id: str
) -> StoryboardDraft:
    """Assign local IDs and validate a text-only remote blueprint exactly once."""
    try:
        raw_requests = blueprint["asset_requests"]
        raw_scenes = blueprint["scenes"]
        if not isinstance(raw_requests, list) or not isinstance(raw_scenes, list):
            raise TypeError("asset_requests and scenes must be arrays")
        fingerprint = sha256(
            json.dumps(blueprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        requests_by_catalog: dict[str, str] = {}
        asset_requests: list[dict[str, Any]] = []
        for index, request in enumerate(raw_requests, start=1):
            catalog_key = request["catalog_key"]
            if catalog_key in requests_by_catalog:
                raise ValueError(f"duplicate catalog request {catalog_key!r}")
            request_id = f"areq_{fingerprint}_{index:02d}"
            requests_by_catalog[catalog_key] = request_id
            asset_requests.append({
                "asset_request_id": request_id,
                "catalog_key": catalog_key,
                "kind": request["kind"],
                "character_id": request.get("character_id"),
                "required_parts": request.get("required_parts", []),
            })

        scenes: list[dict[str, Any]] = []
        for index, scene in enumerate(raw_scenes, start=1):
            actors = []
            for actor in scene["actors"]:
                motion = actor.get("motion", {"kind": "static"})
                actors.append({
                    "character_id": actor["character_id"],
                    "asset_request_id": requests_by_catalog[actor["catalog_key"]],
                    "transform": {
                        "x_percent": actor["x_percent"], "y_percent": actor["y_percent"],
                        "scale": actor.get("scale", 1), "rotation_degrees": actor.get("rotation_degrees", 0),
                        "anchor_x_percent": actor.get("anchor_x_percent", 50),
                        "anchor_y_percent": actor.get("anchor_y_percent", 50), "z_index": actor["z_index"],
                    },
                    "motion": {
                        "kind": motion["kind"], "amplitude_percent": motion.get("amplitude_percent", 0),
                        "period_frames": motion.get("period_frames", 30),
                    },
                })
            scenes.append({
                "scene_id": f"scene_{fingerprint}_{index:02d}", "sequence": scene["sequence"],
                "duration_frames": scene["duration_frames"],
                "background_request_id": requests_by_catalog[scene["background_catalog_key"]], "actors": actors,
            })
        return StoryboardDraft.model_validate({
            "storyboard_draft_id": f"storydraft_{fingerprint}", "project_id": project_id,
            "project_version": project_version, "transcript_asset_id": transcript_asset_id,
            "narration_mode": blueprint["narration_mode"], "narration_text": blueprint.get("narration_text"),
            "asset_requests": asset_requests, "scenes": scenes,
            "bird_fence_geometry": BirdFenceGeometry.model_validate(
                blueprint.get("bird_fence_geometry", BirdFenceGeometry().model_dump())
            ),
        })
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise GroqPlannerError(f"Groq blueprint does not satisfy StoryboardDraft: {error}") from error


def storyboard_from_draft(
    draft: StoryboardDraft,
    project: ProjectSpec,
    *,
    catalog_asset_ids: dict[str, str],
    user_recording_text: str,
) -> Storyboard:
    """Resolve approved catalog assets locally and preserve Phase 2 transforms."""
    if draft.project_id != project.project_id or draft.project_version != project.version:
        raise ValueError("StoryboardDraft does not belong to this ProjectSpec version")
    if draft.narration_mode is not NarrationMode.SYNTHETIC_TTS:
        raise ValueError("Phase 2 instruction recordings use synthetic narration by default")
    request_assets = {
        request.asset_request_id: catalog_asset_ids[request.catalog_key] for request in draft.asset_requests
    }
    request_catalog_keys = {request.asset_request_id: request.catalog_key for request in draft.asset_requests}
    narration_text = draft.narration_text or user_recording_text
    if not narration_text.strip():
        raise ValueError("a storyboard requires narration text")
    if sum(scene.duration_frames for scene in draft.scenes) != project.creative_constraints.target_duration_seconds * project.output.fps:
        raise ValueError("StoryboardDraft scenes must equal the requested project duration")

    motion_kinds = {
        "butterfly_wing_flap": MotionKind.BUTTERFLY_FLAP,
        "flower_sway": MotionKind.FLOWER_SWAY,
        "bird_hop": MotionKind.BIRD_HOP,
        "static": MotionKind.STATIC,
    }
    start_frame = 0
    scenes: list[Scene] = []
    for scene in draft.scenes:
        actor_cues = []
        for actor in scene.actors:
            catalog_key = request_catalog_keys[actor.asset_request_id]
            transform = actor.transform
            motion_kind = actor.motion.kind.value
            layout = None
            if catalog_key in _BIRD_LAYOUT and draft.bird_fence_geometry is not None:
                geometry = draft.bird_fence_geometry
                transform = transform.model_copy(update={
                    "x_percent": geometry.hop_start_x_percent,
                    "y_percent": Phase1AssetGenerator.bird_landing_y_percent(geometry, project.output.width, project.output.height),
                    "scale": geometry.bird_scale, "rotation_degrees": 0,
                    "anchor_x_percent": 50, "anchor_y_percent": 50,
                    "z_index": _BIRD_LAYOUT[catalog_key]["transform"]["z_index"],
                })
                layout = {"amplitude_percent": geometry.hop_height / 1080 * 100,
                          "period_frames": geometry.hop_period_frames, "hop_count": geometry.hop_count,
                          "start_x_percent": geometry.hop_start_x_percent, "end_x_percent": geometry.hop_end_x_percent}
                motion_kind = "bird_hop"
            elif catalog_key in _BIRD_LAYOUT:
                # Fixed approved parts share one shoulder/body coordinate system. The body
                # stays above the two wings so transparent wing pixels cannot cover it.
                layout = _BIRD_LAYOUT[catalog_key]
                transform = transform.model_copy(update=layout["transform"])
                motion_kind = layout["motion_kind"]
            actor_cues.append(ActorCue(
                character_id=actor.character_id, pose_asset_id=request_assets[actor.asset_request_id],
                start_frame_offset=0, duration_frames=scene.duration_frames, position=ActorPosition.CENTER,
                transform=transform,
                motion=MotionSpec(
                    kind=motion_kinds[motion_kind], amplitude_percent=layout["amplitude_percent"] if catalog_key in _BIRD_LAYOUT else actor.motion.amplitude_percent,
                    period_frames=layout["period_frames"] if catalog_key in _BIRD_LAYOUT else actor.motion.period_frames,
                    hop_count=layout["hop_count"] if catalog_key in _BIRD_LAYOUT else 0,
                    start_x_percent=layout["start_x_percent"] if catalog_key in _BIRD_LAYOUT else None,
                    end_x_percent=layout["end_x_percent"] if catalog_key in _BIRD_LAYOUT else None,
                    landing_surface_y_percent=draft.bird_fence_geometry.rod_top_y / 1080 * 100
                        if catalog_key in _BIRD_LAYOUT and draft.bird_fence_geometry else None,
                    foot_y_percent=Phase1AssetGenerator.bird_foot_y_percent()
                        if catalog_key in _BIRD_LAYOUT and draft.bird_fence_geometry else None,
                ),
            ))
        scenes.append(Scene(
            scene_id=scene.scene_id, sequence=scene.sequence, start_frame=start_frame,
            duration_frames=scene.duration_frames, background_asset_id=request_assets[scene.background_request_id],
            narration=Narration(utterance_id=f"utt_{scene.scene_id.removeprefix('scene_')}", text=narration_text),
            camera_cues=[CameraCue(frame_offset=0, preset=CameraPreset.WIDE)], actor_cues=actor_cues,
        ))
        start_frame += scene.duration_frames
    return Storyboard(
        storyboard_id=f"story_{draft.storyboard_draft_id.removeprefix('storydraft_')}", project_id=project.project_id,
        project_version=project.version, fps=project.output.fps, duration_frames=start_frame, scenes=scenes,
    )


_BIRD_LAYOUT = {
    "catalog_bird_body_v1": {
        "transform": {"x_percent": 50, "y_percent": 65, "scale": 0.68, "anchor_x_percent": 50, "anchor_y_percent": 50, "z_index": 20},
        "motion_kind": "bird_hop", "amplitude_percent": 5, "period_frames": 24, "hop_count": 4, "start_x_percent": 38, "end_x_percent": 62,
    },
    "catalog_bird_left_wing_v1": {
        "transform": {"x_percent": 44, "y_percent": 65, "scale": 0.68, "anchor_x_percent": 36.3, "anchor_y_percent": 40.3, "z_index": 18},
        "motion_kind": "bird_hop", "amplitude_percent": 5, "period_frames": 24, "hop_count": 4, "start_x_percent": 32, "end_x_percent": 56,
    },
    "catalog_bird_right_wing_v1": {
        "transform": {"x_percent": 56, "y_percent": 65, "scale": 0.68, "anchor_x_percent": 63.7, "anchor_y_percent": 40.3, "z_index": 19},
        "motion_kind": "bird_hop", "amplitude_percent": 5, "period_frames": 24, "hop_count": 4, "start_x_percent": 44, "end_x_percent": 68,
    },
}
