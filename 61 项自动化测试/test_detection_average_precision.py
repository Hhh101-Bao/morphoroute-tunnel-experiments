from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class DetectionAveragePrecisionTests(unittest.TestCase):
    def _bundle(self):
        from morphoroute.evaluation.schema import (
            PredictionBundle,
            PredictionImage,
            PredictionInstance,
        )

        target_mask = np.zeros((10, 10), dtype=bool)
        target_mask[2:8, 2:8] = True
        false_mask = np.zeros((10, 10), dtype=bool)
        false_mask[:2, :2] = True
        target = PredictionInstance("crack", 1.0, (2, 2, 6, 6), target_mask)
        high_score_false = PredictionInstance(
            "crack", 0.95, (0, 0, 2, 2), false_mask
        )
        true_prediction = PredictionInstance(
            "crack", 0.90, (2, 2, 6, 6), target_mask
        )
        return PredictionBundle(
            experiment_id="EXP-T",
            split="D3",
            class_names=("crack",),
            images=(
                PredictionImage(
                    image_id="i",
                    group_id="g",
                    width=10,
                    height=10,
                    predictions=(high_score_false, true_prediction),
                    targets=(target,),
                ),
            ),
        )

    def test_high_scoring_false_positive_reduces_ranked_ap(self):
        from morphoroute.evaluation.pipeline import evaluate_bundle

        result = evaluate_bundle(self._bundle())

        self.assertGreater(result.mask_ap50, 0.0)
        self.assertLess(result.mask_ap50, 1.0)
        self.assertEqual(result.mask_ap50, result.box_ap50)
        self.assertIsNotNone(result.mask_ap)
        self.assertIsNotNone(result.mask_ap75)


if __name__ == "__main__":
    unittest.main()
