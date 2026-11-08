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
    "EXP-00": "exp00",
    "EXP-01": "exp01",
    "EXP-02": "exp02",
    "EXP-03": "exp03",
    "EXP-04": "exp04",
    "EXP-05": "exp05",
    "EXP-06": "exp06",
    "EXP-07": "exp07",
}


class EarlyExperimentTests(unittest.TestCase):
    def _config(self, root: Path, experiment_id: str) -> ExperimentConfig:
        models = {
            "EXP-02": "yolov8n-seg.pt",
            "EXP-03": "yolov8n-seg.pt",
            "EXP-04": "mask_rcnn_r50",
            "EXP-05": "mask2former_b0",
            "EXP-06": "resource_matched_shared",
            "EXP-07": "routing_controls",
        }
        return ExperimentConfig.from_mapping(
            {
                "experiment_id": experiment_id,
                "profile": "local_6gb",
                "data_root": str(root / "data"),
                "output_root": str(root / "out"),
                "seed": 42,
                "allowed_splits": list(EXPERIMENTS[experiment_id].allowed_splits),
                "parameters": {
                    "model": models.get(experiment_id, "analysis"),
                    "input_size": 640,
                    "max_updates": 50000,
                    "physical_batch": 1,
                    "effective_batch": 8,
                    "amp": True,
                },
            }
        )

    def test_each_entry_declares_every_registered_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for experiment_id, module_name in MODULE_NAMES.items():
                module = importlib.import_module(
                    f"morphoroute.experiments.{module_name}"
                )
                plan = module.execute(self._config(root, experiment_id), dry_run=True)
                self.assertEqual(
                    set(plan["planned_outputs"]),
                    set(EXPERIMENTS[experiment_id].expected_outputs),
                    experiment_id,
                )
                self.assertTrue(plan["steps"], experiment_id)

    def test_yolo_reproduction_is_excluded_from_innovation_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = importlib.import_module("morphoroute.experiments.exp02")
            plan = module.execute(self._config(root, "EXP-02"), dry_run=True)
            self.assertEqual(plan["comparison_scope"], "rq1_yolo_only")
            self.assertFalse(plan["eligible_for_innovation_selection"])


if __name__ == "__main__":
    unittest.main()
