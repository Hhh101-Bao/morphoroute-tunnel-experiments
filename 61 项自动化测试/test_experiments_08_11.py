import importlib
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.registry import EXPERIMENTS


MODULE_NAMES = {
    "EXP-08A": "exp08a",
    "EXP-08B": "exp08b",
    "EXP-08C": "exp08c",
    "EXP-08D": "exp08d",
    "EXP-09": "exp09",
    "EXP-10": "exp10",
    "EXP-11": "exp11",
}


class MechanismAndExternalExperimentTests(unittest.TestCase):
    def _config(self, root: Path, experiment_id: str):
        return ExperimentConfig.from_mapping(
            {
                "experiment_id": experiment_id,
                "profile": "local_6gb",
                "data_root": str(root / "data"),
                "output_root": str(root / "out"),
                "seed": 42,
                "allowed_splits": list(EXPERIMENTS[experiment_id].allowed_splits),
                "parameters": {
                    "model": "morphoroute_b9",
                    "input_size": 640,
                    "max_updates": 50000,
                    "physical_batch": 1,
                    "effective_batch": 8,
                    "amp": True,
                },
            }
        )

    def _plan(self, root: Path, experiment_id: str):
        module = importlib.import_module(
            f"morphoroute.experiments.{MODULE_NAMES[experiment_id]}"
        )
        return module.execute(self._config(root, experiment_id), dry_run=True)

    def test_each_entry_declares_registered_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for experiment_id in MODULE_NAMES:
                plan = self._plan(root, experiment_id)
                self.assertEqual(
                    set(plan["planned_outputs"]),
                    set(EXPERIMENTS[experiment_id].expected_outputs),
                    experiment_id,
                )

    def test_mechanism_contracts_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m1 = self._plan(root, "EXP-08A")
            self.assertTrue(m1["same_checkpoint"])
            self.assertTrue(m1["same_parameter_subset"])
            m3 = self._plan(root, "EXP-08C")
            self.assertEqual(
                set(m3["required_covariates"]),
                {"class_name", "instance_area"},
            )
            m4 = self._plan(root, "EXP-08D")
            self.assertEqual(
                set(m4["interventions"]),
                {
                    "normal",
                    "disable_line",
                    "disable_area",
                    "swap_outputs",
                    "force_line",
                    "force_area",
                },
            )

    def test_ablation_and_freeze_contracts_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ablation = self._plan(root, "EXP-09")
            self.assertEqual(ablation["variants"], [f"B{i}" for i in range(12)])
            self.assertIn(["B9", "B10"], ablation["planned_contrasts"])
            freeze = self._plan(root, "EXP-10")
            self.assertEqual(
                set(freeze["mandatory_freeze_fields"]),
                {
                    "checkpoint",
                    "code",
                    "config",
                    "environment",
                    "thresholds",
                    "calibrator",
                    "postprocess",
                    "evaluator",
                },
            )
            external = self._plan(root, "EXP-11")
            self.assertTrue(external["one_shot"])
            self.assertEqual(external["authorized_split"], "D4")


if __name__ == "__main__":
    unittest.main()
