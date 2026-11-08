import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.config import ExperimentConfig
from morphoroute.runtime.mmdet_adapter import MMDetectionAdapter
from morphoroute.runtime.ultralytics_adapter import UltralyticsAdapter


class AdapterDryRunTests(unittest.TestCase):
    def _config(self, root: Path, experiment_id: str, model: str):
        return ExperimentConfig.from_mapping(
            {
                "experiment_id": experiment_id,
                "profile": "local_6gb",
                "data_root": str(root / "data"),
                "output_root": str(root / "out"),
                "seed": 17,
                "allowed_splits": ["D1", "D2", "D3"],
                "parameters": {
                    "model": model,
                    "input_size": 640,
                    "max_updates": 50000,
                    "physical_batch": 1,
                    "effective_batch": 8,
                    "amp": True,
                },
            }
        )

    def test_ultralytics_plan_uses_d1_train_d2_validation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = UltralyticsAdapter().build_plan(
                self._config(root, "EXP-03", "yolov8n-seg"),
                root / "derived" / "yolo",
            )
            self.assertEqual(plan.train_split, "D1")
            self.assertEqual(plan.validation_split, "D2")
            self.assertNotIn("D3", plan.training_inputs)
            self.assertNotIn("D4", plan.training_inputs)
            self.assertEqual(plan.gradient_accumulation, 8)
            self.assertTrue(plan.amp)

    def test_mmdet_plan_preserves_update_and_batch_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = MMDetectionAdapter().build_plan(
                self._config(root, "EXP-05", "mask2former_b0"),
                root / "derived" / "coco",
            )
            self.assertEqual(plan.max_updates, 50000)
            self.assertEqual(plan.physical_batch, 1)
            self.assertEqual(plan.effective_batch, 8)
            self.assertEqual(plan.gradient_accumulation, 8)
            self.assertEqual(plan.input_size, 640)
            self.assertIn("mask2former_b0", plan.model)


if __name__ == "__main__":
    unittest.main()
