import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.config import ExperimentConfig


class ExperimentConfigTests(unittest.TestCase):
    def _mapping(self, **overrides):
        value = {
            "experiment_id": "EXP-01",
            "profile": "local_6gb",
            "data_root": "C:/dataset",
            "output_root": "C:/project/outputs",
            "seed": 42,
            "allowed_splits": ["D1", "D2", "D3"],
            "parameters": {"input_size": 640},
        }
        value.update(overrides)
        return value

    def test_rejects_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "profile"):
            ExperimentConfig.from_mapping(self._mapping(profile="paperish"))

    def test_rejects_unknown_split(self):
        with self.assertRaisesRegex(ValueError, "allowed_splits"):
            ExperimentConfig.from_mapping(
                self._mapping(allowed_splits=["D1", "validation"])
            )

    def test_rejects_output_inside_dataset(self):
        config = ExperimentConfig.from_mapping(
            self._mapping(
                data_root="C:/dataset",
                output_root="C:/dataset/results",
            )
        )
        with self.assertRaisesRegex(ValueError, "output_root"):
            config.validate()

    def test_canonical_dict_has_stable_path_and_split_types(self):
        config = ExperimentConfig.from_mapping(self._mapping())
        canonical = config.canonical_dict()
        self.assertEqual(canonical["allowed_splits"], ["D1", "D2", "D3"])
        self.assertEqual(canonical["data_root"], "C:/dataset")
        self.assertEqual(canonical["parameters"], {"input_size": 640})


if __name__ == "__main__":
    unittest.main()
