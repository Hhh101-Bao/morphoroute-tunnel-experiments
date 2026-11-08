import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.guard import (
    D4_CONFIRMATION_PHRASE,
    D4AuthorizationRequest,
    D4Guard,
)
from morphoroute.core.io import sha256_file


class D4GuardTests(unittest.TestCase):
    def _valid_request(self, root: Path) -> D4AuthorizationRequest:
        checkpoint = root / "best.pth"
        checkpoint.write_bytes(b"frozen-checkpoint")
        freeze_manifest = root / "d4_freeze_manifest.json"
        freeze_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "frozen",
                    "experiment_id": "EXP-10",
                    "frozen_at_utc": "2026-07-29T00:00:00+00:00",
                    "artifacts": {
                        "checkpoint": {
                            "path": str(checkpoint),
                            "sha256": sha256_file(checkpoint),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return D4AuthorizationRequest(
            experiment_id="EXP-11",
            freeze_manifest=freeze_manifest,
            output_root=root / "output",
            confirmation_phrase=D4_CONFIRMATION_PHRASE,
        )

    def test_rejects_missing_freeze_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = D4AuthorizationRequest(
                experiment_id="EXP-11",
                freeze_manifest=root / "missing.json",
                output_root=root / "output",
                confirmation_phrase=D4_CONFIRMATION_PHRASE,
            )
            with self.assertRaisesRegex(PermissionError, "freeze"):
                D4Guard().authorize(request)

    def test_rejects_wrong_confirmation_phrase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self._valid_request(root)
            request = D4AuthorizationRequest(
                experiment_id=request.experiment_id,
                freeze_manifest=request.freeze_manifest,
                output_root=request.output_root,
                confirmation_phrase="yes",
            )
            with self.assertRaisesRegex(PermissionError, "confirmation"):
                D4Guard().authorize(request)

    def test_rejects_artifact_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self._valid_request(root)
            (root / "best.pth").write_bytes(b"changed")
            with self.assertRaisesRegex(PermissionError, "hash"):
                D4Guard().authorize(request)

    def test_rejects_second_started_or_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self._valid_request(root)
            request.output_root.mkdir(parents=True)
            (request.output_root / "one_shot_status.json").write_text(
                json.dumps({"status": "started"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(PermissionError, "already"):
                D4Guard().authorize(request)

    def test_allows_audited_recovery_before_any_prediction_feedback(self):
        from morphoroute.core.io import atomic_write_json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._valid_request(root)
            marker = request.output_root / "one_shot_status.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                marker,
                {
                    "status": "infrastructure_failed_no_feedback",
                    "freeze_manifest_sha256": sha256_file(
                        request.freeze_manifest
                    ),
                    "recovery_count": 0,
                },
            )

            authorization = D4Guard().authorize(request)

            self.assertEqual(authorization["status"], "started")
            self.assertEqual(authorization["recovery_count"], 1)

    def test_accepts_complete_untampered_freeze_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self._valid_request(Path(tmp))
            result = D4Guard().authorize(request)
            self.assertEqual(result["authorized_experiment"], "EXP-11")
            self.assertEqual(result["verified_artifacts"], ["checkpoint"])


if __name__ == "__main__":
    unittest.main()
