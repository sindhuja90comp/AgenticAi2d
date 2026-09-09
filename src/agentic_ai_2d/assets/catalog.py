"""Curated local asset catalog used by Phase 2 planners and resolvers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .generator import GeneratedAssetPack


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    key: str
    path: Path
    character_id: str | None = None
    part: str | None = None


def bird_fence_catalog(pack: GeneratedAssetPack) -> dict[str, CatalogEntry]:
    """Return only approved bird/fence assets, including separated bird layers."""
    if not pack.parts:
        raise ValueError("bird asset pack is missing multipart layers")
    return {
        "catalog_fence_v1": CatalogEntry("catalog_fence_v1", pack.background),
        "catalog_bird_body_v1": CatalogEntry("catalog_bird_body_v1", pack.parts["body"], "char_bird_v1", "body"),
        "catalog_bird_left_wing_v1": CatalogEntry("catalog_bird_left_wing_v1", pack.parts["left_wing"], "char_bird_v1", "left_wing"),
        "catalog_bird_right_wing_v1": CatalogEntry("catalog_bird_right_wing_v1", pack.parts["right_wing"], "char_bird_v1", "right_wing"),
    }
