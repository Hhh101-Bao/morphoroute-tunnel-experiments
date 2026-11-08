from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, sha256_json


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _metadata_path(self, checkpoint_name: str) -> Path:
        return self.checkpoint_dir / f"{checkpoint_name}.meta.json"

    def record(
        self,
        checkpoint_name: str,
        *,
        config: ExperimentConfig,
        dataset_manifest_sha256: str,
        update: int,
        best_d2_score: float | None,
    ) -> Path:
        payload = {
            "schema_version": 1,
            "checkpoint_name": checkpoint_name,
            "config_sha256": sha256_json(config.canonical_dict()),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "profile": config.profile,
            "seed": config.seed,
            "update": int(update),
            "best_d2_score": best_d2_score,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return atomic_write_json(self._metadata_path(checkpoint_name), payload)

    def validate_resume(
        self,
        checkpoint_name: str,
        *,
        config: ExperimentConfig,
        dataset_manifest_sha256: str,
    ) -> dict[str, Any]:
        metadata_path = self._metadata_path(checkpoint_name)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_config_hash = sha256_json(config.canonical_dict())
        if payload.get("config_sha256") != expected_config_hash:
            raise ValueError("checkpoint config does not match the current run")
        if payload.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError("checkpoint dataset manifest does not match")
        if payload.get("profile") != config.profile or payload.get("seed") != config.seed:
            raise ValueError("checkpoint profile or seed does not match")
        return payload
