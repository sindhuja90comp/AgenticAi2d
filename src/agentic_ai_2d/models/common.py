"""Shared strict model configuration and identifier constraints."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

ProjectId = Annotated[str, Field(pattern=r"^proj_[a-zA-Z0-9_-]{8,64}$")]
AssetId = Annotated[str, Field(pattern=r"^asset_[a-zA-Z0-9_-]{8,128}$")]
CharacterId = Annotated[
    str, Field(pattern=r"^char_[a-zA-Z0-9_-]{3,128}_v[1-9][0-9]*$")
]
MouthSetId = Annotated[
    str, Field(pattern=r"^mouthset_[a-zA-Z0-9_-]{3,128}_v[1-9][0-9]*$")
]
UtteranceId = Annotated[str, Field(pattern=r"^utt_[a-zA-Z0-9_-]{8,128}$")]


class ContractModel(BaseModel):
    """Base class for API contracts with JSON Schema export helpers."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: ClassVar[str]

    @classmethod
    def draft202012_schema(cls) -> dict[str, object]:
        """Return the Pydantic schema identified as Draft 2020-12."""
        schema = cls.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = cls.schema_id
        return schema
