import importlib
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.registry import EXPERIMENTS
from morphoroute.experiments.exp13 import require_physical_scale
from morphoroute.experiments.exp15 import claim_branch


MODULE_NAMES = {
    "EXP-12": "exp12",
    "EXP-13": "exp13",
    "EXP-14": "exp14",
    "EXP-15": "exp15",
}


class LateExperimentTests(unittest.TestCase):
    def _config(self, root: Path, experiment_id: str):
        return ExperimentConfig.from_mapping(
            {
                "experiment_id": experiment_id,
                "profile": "local_6gb",
                "data_root": str(root / "data"),
                "output_root": str(root / "out"),
                "seed": 42,
                "allowed_splits": list(EXPERIMENTS[experiment_id].allowed_splits),
                "parameters": {"model": "morphoroute_b9"},
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

    def test_calibration_and_efficiency_contracts_are_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reliability = self._plan(root, "EXP-12")
            self.assertEqual(reliability["fit_split"], "D2")
            self.assertEqual(reliability["validation_split"], "D3")
            self.assertEqual(reliability["external_action"], "apply_frozen_only")
            efficiency = self._plan(root, "EXP-14")
            self.assertEqual(efficiency["batch_size"], 1)
            self.assertTrue(efficiency["same_hardware_required"])
            self.assertTrue(efficiency["synchronized_timing"])

    def test_absolute_units_require_scale_metadata(self):
        self.assertEqual(require_physical_scale(0.25), 0.25)
        with self.assertRaisesRegex(ValueError, "scale"):
            require_physical_scale(None)

    def test_claim_branch_covers_all_m_and_e_states(self):
        self.assertEqual(claim_branch(True, True)["code"], "M+/E+")
        self.assertEqual(claim_branch(True, False)["code"], "M+/E-")
        self.assertEqual(claim_branch(False, True)["code"], "M-/E+")
        self.assertEqual(claim_branch(False, False)["code"], "M-/E-")
        self.assertNotIn("已证明负迁移", claim_branch(False, True)["claim"])


if __name__ == "__main__":
    unittest.main()
