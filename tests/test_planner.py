import pytest

from agentic_ai_2d.agents.planner import GROQ_PLANNER_MODEL, GroqPlannerError, GroqTextPlanner, storyboard_draft_from_blueprint


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.request: dict | None = None

    def create(self, **kwargs):
        self.request = kwargs
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self, content: str) -> None:
        self.completions = FakeCompletions(content)
        self.chat = type("Chat", (), {"completions": self.completions})()


class SequencedCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = type("Message", (), {"content": self.contents.pop(0)})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class SequencedClient:
    def __init__(self, contents: list[str]) -> None:
        self.completions = SequencedCompletions(contents)
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_groq_text_planner_sends_text_only_context_and_parses_json() -> None:
    client = FakeClient('{"narration_mode":"synthetic_tts","scenes":[]}')
    planner = GroqTextPlanner(client=client)

    blueprint = planner.plan(
        "Place a butterfly above a flower.",
        catalog_description="butterfly body and wings; flower head and stem",
    )

    assert blueprint["narration_mode"] == "synthetic_tts"
    assert client.completions.request["model"] == GROQ_PLANNER_MODEL
    assert client.completions.request["response_format"]["type"] == "json_schema"
    assert client.completions.request["response_format"]["json_schema"]["strict"] is True
    assert all(isinstance(message["content"], str) for message in client.completions.request["messages"])


def test_groq_text_planner_rejects_empty_instructions_and_invalid_json() -> None:
    planner = GroqTextPlanner(client=FakeClient("not json"))

    with pytest.raises(ValueError, match="must not be empty"):
        planner.plan(" ")
    with pytest.raises(GroqPlannerError, match="invalid JSON"):
        planner.plan("Make a butterfly move.")


def test_text_only_blueprint_is_assigned_local_ids_and_validated_as_a_draft() -> None:
    blueprint = {
        "narration_mode": "synthetic_tts", "narration_text": "A butterfly rests above a flower.",
        "asset_requests": [
            {"catalog_key": "catalog_meadow_v1", "kind": "background", "character_id": None, "required_parts": []},
            {"catalog_key": "catalog_butterfly_v1", "kind": "character_part", "character_id": "char_butterfly_v1", "required_parts": ["body", "left_wing", "right_wing"]},
        ],
        "scenes": [{"sequence": 1, "duration_frames": 450, "background_catalog_key": "catalog_meadow_v1", "actors": [{
            "catalog_key": "catalog_butterfly_v1", "character_id": "char_butterfly_v1", "x_percent": 55,
            "y_percent": 42, "scale": 1, "z_index": 20, "anchor_x_percent": 50, "anchor_y_percent": 50,
            "motion": {"kind": "butterfly_wing_flap", "amplitude_percent": 10, "period_frames": 12},
        }]}],
    }

    draft = storyboard_draft_from_blueprint(
        blueprint, project_id="proj_butterfly01", project_version=1, transcript_asset_id="asset_transcript_12345678"
    )

    assert draft.asset_requests[0].asset_request_id.startswith("areq_")
    assert draft.scenes[0].actors[0].transform.z_index == 20


def test_planner_repairs_an_invalid_blueprint_once_and_records_text_safe_provenance() -> None:
    invalid = '{"narration_mode":"synthetic_tts","narration_text":"Bird.","asset_requests":[],"scenes":[]}'
    valid = '''{"narration_mode":"synthetic_tts","narration_text":"A bird hops.","asset_requests":[{"catalog_key":"catalog_fence_v1","kind":"background","character_id":null,"required_parts":[]},{"catalog_key":"catalog_bird_body_v1","kind":"character_part","character_id":"char_bird_v1","required_parts":["body"]}],"scenes":[{"sequence":1,"duration_frames":450,"background_catalog_key":"catalog_fence_v1","actors":[{"catalog_key":"catalog_bird_body_v1","character_id":"char_bird_v1","x_percent":50,"y_percent":50,"scale":1,"z_index":1,"anchor_x_percent":50,"anchor_y_percent":50,"motion":{"kind":"bird_hop","amplitude_percent":5,"period_frames":30}}]}]}'''
    client = SequencedClient([invalid, valid])
    planner = GroqTextPlanner(client=client)

    draft = planner.plan_draft(
        "A bird hops.", project_id="proj_birdtest01", project_version=1,
        transcript_asset_id="asset_transcript_12345678", catalog_description="A local bird catalog.",
    )

    assert draft.narration_text == "A bird hops."
    assert len(client.completions.requests) == 2
    assert planner.last_provenance["repair_attempted"] is True
    assert planner.last_provenance["validation_result"] == "valid"
    assert len(planner.last_provenance["prompt_digest"]) == 64
    assert len(planner.last_provenance["response_digest"]) == 64
    assert len(planner.last_provenance["repair_response_digest"]) == 64


def test_schema_is_explicit_in_prompt_and_repair_preserves_revision_context():
    import json
    from agentic_ai_2d.agents.planner import _BLUEPRINT_SCHEMA
    client = SequencedClient(['{"scenes":[]}', '{"scenes":[]}'])
    planner = GroqTextPlanner(client=client)
    instruction = 'Current geometry: pole_top_y 790. Revision: move pole_top_y to 850.'
    planner.plan(instruction, catalog_description='approved bird catalog')
    planner._repair({}, 'missing bird_fence_geometry', instruction=instruction,
                    catalog_description='approved bird catalog')
    for request in client.completions.requests:
        assert json.dumps(_BLUEPRINT_SCHEMA, separators=(',', ':')) in request['messages'][0]['content']
        assert '"bird_fence_geometry": {' in request['messages'][0]['content']
    repair_text = client.completions.requests[1]['messages'][1]['content']
    assert instruction in repair_text
    assert 'approved bird catalog' in repair_text
