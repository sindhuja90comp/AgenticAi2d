"""Filesystem-backed, content-addressed artifact storage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Mapping

from ..models.common import ContractModel


class ArtifactType(StrEnum):
    RAW_FILE = "raw_file"
    JSON_CONTRACT = "json_contract"


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Immutable reference to a content-addressed artifact."""

    asset_id: str
    digest: str
    artifact_type: ArtifactType
    relative_path: str
    media_type: str
    size_bytes: int


class ArtifactManager:
    """Persist bytes once and return stable IDs derived from SHA-256 content hashes."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._by_digest: dict[str, ArtifactMetadata] = {}
        self._lock = RLock()
        self._root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._root / "state.db", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS artifacts (digest TEXT PRIMARY KEY, asset_id TEXT UNIQUE, artifact_type TEXT, relative_path TEXT, media_type TEXT, size_bytes INTEGER)")
        for row in self._db.execute("SELECT digest, asset_id, artifact_type, relative_path, media_type, size_bytes FROM artifacts"):
            self._by_digest[row[0]] = ArtifactMetadata(row[1], row[0], ArtifactType(row[2]), row[3], row[4], row[5])

    @property
    def root(self) -> Path:
        return self._root

    @property
    def asset_ids(self) -> frozenset[str]:
        return frozenset(metadata.asset_id for metadata in self._by_digest.values())

    def get_metadata(self, asset_id: str) -> ArtifactMetadata:
        """Return metadata already registered for an immutable asset ID."""
        for metadata in self._by_digest.values():
            if metadata.asset_id == asset_id:
                return metadata
        raise KeyError(f"unknown asset_id {asset_id!r}")

    def absolute_path(self, metadata: ArtifactMetadata) -> Path:
        """Resolve an immutable metadata reference to its local artifact path."""
        path = self._root / metadata.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"artifact file is missing for {metadata.asset_id}")
        return path

    def store_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        media_type: str = "application/octet-stream",
    ) -> ArtifactMetadata:
        """Store raw bytes at an immutable digest path and return its metadata."""
        if not content:
            raise ValueError("artifacts must contain at least one byte")
        digest = sha256(content).hexdigest()
        suffix = Path(filename).suffix.lower() or ".bin"
        return self._store(digest, content, ArtifactType.RAW_FILE, suffix, media_type)

    def store_contract(self, contract: ContractModel | Mapping[str, Any]) -> ArtifactMetadata:
        """Canonicalize and store a JSON contract so equal content has one digest."""
        if isinstance(contract, ContractModel):
            payload: Mapping[str, Any] = contract.model_dump(mode="json")
            contract_name = type(contract).__name__.lower()
        else:
            payload = contract
            contract_name = "contract"
        content = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        digest = sha256(content).hexdigest()
        return self._store(
            digest,
            content,
            ArtifactType.JSON_CONTRACT,
            ".json",
            f"application/vnd.agentic-ai-2d.{contract_name}+json",
        )

    def read_bytes(self, metadata: ArtifactMetadata) -> bytes:
        """Read an artifact only after confirming its on-disk digest is intact."""
        path = self.absolute_path(metadata)
        content = path.read_bytes()
        if sha256(content).hexdigest() != metadata.digest:
            raise ValueError(f"artifact digest mismatch for {metadata.asset_id}")
        return content

    def _store(
        self,
        digest: str,
        content: bytes,
        artifact_type: ArtifactType,
        suffix: str,
        media_type: str,
    ) -> ArtifactMetadata:
        with self._lock:
            existing = self._by_digest.get(digest)
            if existing is not None:
                return existing

            relative_path = Path("objects") / "sha256" / digest[:2] / digest / f"payload{suffix}"
            path = self._root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"refusing to overwrite non-matching artifact path {path}")
            if not path.exists():
                path.write_bytes(content)

            metadata = ArtifactMetadata(
                asset_id=f"asset_{digest}",
                digest=digest,
                artifact_type=artifact_type,
                relative_path=str(relative_path),
                media_type=media_type,
                size_bytes=len(content),
            )
            self._by_digest[digest] = metadata
            self._db.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?, ?, ?, ?)", (digest, metadata.asset_id, metadata.artifact_type.value, metadata.relative_path, metadata.media_type, metadata.size_bytes))
            self._db.commit()
            return metadata
