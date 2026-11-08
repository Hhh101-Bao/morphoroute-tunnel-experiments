from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .io import atomic_write_json, sha256_file, sha256_json


class RunManifest:
    @staticmethod
    def start(
        config: ExperimentConfig,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        config.validate()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset_manifest = config.data_root / "metadata" / "split_manifest.csv"
        payload: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "profile": config.profile,
            "seed": config.seed,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "config_sha256": sha256_json(config.canonical_dict()),
            "dataset_manifest_sha256": (
                sha256_file(dataset_manifest) if dataset_manifest.is_file() else None
            ),
            "code_revision": None,
        }
        atomic_write_json(output_path / "run_manifest.json", payload)
        return payload
