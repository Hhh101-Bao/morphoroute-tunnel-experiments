import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.config import ExperimentConfig
from morphoroute.runtime.checkpoints import CheckpointManager


class CheckpointResumeTests(unittest.TestCase):
    def _config(self, root: Path, **overrides):
        mapping = {
            "experiment_id": "EXP-05",
            "profile": "local_6gb",
            "data_root": str(root / "data"),
            "output_root": str(root / "out"),
            "seed": 42,
            "allowed_splits": ["D1", "D2", "D3"],
            "parameters": {"input_size": 640},
        }
        mapping.update(overrides)
        return ExperimentConfig.from_mapping(mapping)

    def test_matching_checkpoint_returns_exact_resume_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            manager = CheckpointManager(root / "checkpoints")
            config = self._config(root)
            manager.record(
                "latest.pth",
                config=config,
                dataset_manifest_sha256="a" * 64,
                update=1234,
                best_d2_score=0.51,
            )
            state = manager.validate_resume(
                "latest.pth",
                config=config,
                dataset_manifest_sha256="a" * 64,
            )
            self.assertEqual(state["update"], 1234)
            self.assertEqual(state["best_d2_score"], 0.51)

    def test_rejects_profile_or_data_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            manager = CheckpointManager(root / "checkpoints")
            config = self._config(root)
            manager.record(
                "latest.pth",
                config=config,
                dataset_manifest_sha256="a" * 64,
                update=1,
                best_d2_score=0.1,
            )
            with self.assertRaisesRegex(ValueError, "dataset"):
                manager.validate_resume(
                    "latest.pth",
                    config=config,
                    dataset_manifest_sha256="b" * 64,
                )
            other_profile = self._config(root, profile="sop_full")
            with self.assertRaisesRegex(ValueError, "config"):
                manager.validate_resume(
                    "latest.pth",
                    config=other_profile,
                    dataset_manifest_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
