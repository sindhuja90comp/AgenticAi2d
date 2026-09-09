from agentic_ai_2d.assets import Phase1AssetGenerator, Scenario, bird_fence_catalog


def test_bird_catalog_resolves_only_local_approved_parts(tmp_path) -> None:
    catalog = bird_fence_catalog(Phase1AssetGenerator(tmp_path).generate(Scenario.BIRD_FENCE))
    assert catalog["catalog_bird_body_v1"].part == "body"
    assert catalog["catalog_bird_left_wing_v1"].path.is_file()
    assert catalog["catalog_bird_right_wing_v1"].part == "right_wing"
    assert "catalog_bird_v1" not in catalog
