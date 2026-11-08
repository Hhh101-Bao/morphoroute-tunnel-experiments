import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.registry import EXPERIMENTS, validate_prerequisites


class ExperimentRegistryTests(unittest.TestCase):
    def test_registry_contains_all_nineteen_sop_cards(self):
        expected = {
            "EXP-00",
            "EXP-01",
            "EXP-02",
            "EXP-03",
            "EXP-04",
            "EXP-05",
            "EXP-06",
            "EXP-07",
            "EXP-08A",
            "EXP-08B",
            "EXP-08C",
            "EXP-08D",
            "EXP-09",
            "EXP-10",
            "EXP-11",
            "EXP-12",
            "EXP-13",
            "EXP-14",
            "EXP-15",
        }
        self.assertEqual(set(EXPERIMENTS), expected)

    def test_exp11_is_only_regular_d4_consumer(self):
        consumers = [
            spec.experiment_id
            for spec in EXPERIMENTS.values()
            if "D4" in spec.allowed_splits
        ]
        self.assertEqual(consumers, ["EXP-11"])

    def test_prerequisite_must_have_completed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            exp01 = output_root / "experiments" / "EXP-01"
            exp01.mkdir(parents=True)
            (exp01 / "run_manifest.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "EXP-01"):
                validate_prerequisites(EXPERIMENTS["EXP-03"], output_root)


if __name__ == "__main__":
    unittest.main()
