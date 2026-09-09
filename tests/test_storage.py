from hashlib import sha256

from agentic_ai_2d.storage import ArtifactManager, ArtifactType


def test_artifact_storage_is_content_addressed_and_immutable(tmp_path) -> None:
    manager = ArtifactManager(tmp_path)

    first = manager.store_bytes(b"voice bytes", filename="instruction.wav", media_type="audio/wav")
    duplicate = manager.store_bytes(b"voice bytes", filename="other-name.wav", media_type="audio/wav")

    assert first == duplicate
    assert first.asset_id == f"asset_{sha256(b'voice bytes').hexdigest()}"
    assert manager.read_bytes(first) == b"voice bytes"
    assert first.asset_id in manager.asset_ids


def test_json_contracts_are_canonicalized_before_hashing(tmp_path) -> None:
    manager = ArtifactManager(tmp_path)

    first = manager.store_contract({"b": 2, "a": 1})
    second = manager.store_contract({"a": 1, "b": 2})

    assert first == second
    assert first.artifact_type is ArtifactType.JSON_CONTRACT
    assert manager.read_bytes(first) == b'{"a":1,"b":2}'
