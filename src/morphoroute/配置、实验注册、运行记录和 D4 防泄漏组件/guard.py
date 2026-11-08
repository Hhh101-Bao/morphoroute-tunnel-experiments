from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import atomic_write_json, sha256_file, sha256_json


D4_CONFIRMATION_PHRASE = "I UNDERSTAND D4 IS ONE-SHOT"


@dataclass(frozen=True)
class D4AuthorizationRequest:
    experiment_id: str
    freeze_manifest: Path
    output_root: Path
    confirmation_phrase: str


class D4Guard:
    def authorize(self, request: D4AuthorizationRequest) -> dict[str, Any]:
        if request.experiment_id != "EXP-11":
            raise PermissionError("D4 authorization is restricted to EXP-11")
        if request.confirmation_phrase != D4_CONFIRMATION_PHRASE:
            raise PermissionError("D4 confirmation phrase does not match")
        if not request.freeze_manifest.is_file():
            raise PermissionError("D4 freeze manifest is missing")

        try:
            payload = json.loads(request.freeze_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PermissionError("D4 freeze manifest is unreadable") from error

        if payload.get("status") != "frozen" or payload.get("experiment_id") != "EXP-10":
            raise PermissionError("D4 freeze manifest has invalid status")
        if payload.get("schema_version") != 1:
            raise PermissionError("D4 freeze manifest schema is unsupported")

        try:
            frozen_at = datetime.fromisoformat(payload["frozen_at_utc"])
        except (KeyError, TypeError, ValueError) as error:
            raise PermissionError("D4 freeze timestamp is invalid") from error
        if frozen_at.tzinfo is None:
            raise PermissionError("D4 freeze timestamp must include a timezone")
        if frozen_at > datetime.now(timezone.utc):
            raise PermissionError("D4 freeze timestamp is in the future")

        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise PermissionError("D4 freeze manifest has no artifacts")

        verified: list[str] = []
        for name, record in artifacts.items():
            if not isinstance(record, dict):
                raise PermissionError(f"D4 artifact {name} is malformed")
            path_text = record.get("path")
            expected_hash = record.get("sha256")
            if not path_text or not expected_hash:
                raise PermissionError(f"D4 artifact {name} lacks path or hash")
            artifact_path = Path(path_text)
            if not artifact_path.is_absolute():
                artifact_path = request.freeze_manifest.parent / artifact_path
            if not artifact_path.is_file():
                raise PermissionError(f"D4 artifact {name} is missing")
            if sha256_file(artifact_path) != expected_hash:
                raise PermissionError(f"D4 artifact hash mismatch: {name}")
            verified.append(str(name))

        marker = request.output_root / "one_shot_status.json"
        recovery_count = 0
        if marker.exists():
            try:
                previous = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise PermissionError("D4 one-shot marker is unreadable") from error
            feedback_exists = (
                any(
                    (request.output_root / "D4_predictions").rglob("*.json")
                )
                or (request.output_root / "D4_combined_metrics.csv").exists()
                or (request.output_root / "E_gate_decision.json").exists()
            )
            recoverable = (
                previous.get("status")
                == "infrastructure_failed_no_feedback"
                and previous.get("freeze_manifest_sha256")
                == sha256_file(request.freeze_manifest)
                and not feedback_exists
            )
            if not recoverable:
                raise PermissionError(
                    "D4 formal run has already started or completed"
                )
            recovery_count = int(previous.get("recovery_count", 0)) + 1

        authorization = {
            "authorized_experiment": "EXP-11",
            "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_manifest_sha256": sha256_file(request.freeze_manifest),
            "verified_artifacts": sorted(verified),
            "status": "started",
            "recovery_count": recovery_count,
        }
        atomic_write_json(marker, authorization)
        return authorization
