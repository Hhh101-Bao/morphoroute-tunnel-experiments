import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, sha256_file
from morphoroute.core.run_manifest import RunManifest


class CoreIOTests(unittest.TestCase):
    def test_atomic_json_replaces_previous_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "state.json"
            atomic_write_json(target, {"status": "first"})
            atomic_write_json(target, {"status": "complete", "中文": "可读"})

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"status": "complete", "中文": "可读"},
            )
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_sha256_matches_known_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "literal.txt"
            target.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(target),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_run_manifest_records_profile_and_config_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            output_root = root / "out"
            data_root.mkdir()
            config = ExperimentConfig.from_mapping(
                {
                    "experiment_id": "EXP-01",
                    "profile": "local_6gb",
                    "data_root": str(data_root),
                    "output_root": str(output_root),
                    "seed": 7,
                    "allowed_splits": ["D1", "D2", "D3"],
                }
            )
            manifest = RunManifest.start(config, output_root / "EXP-01")

            self.assertEqual(manifest["status"], "running")
            self.assertEqual(manifest["profile"], "local_6gb")
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(len(manifest["config_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
