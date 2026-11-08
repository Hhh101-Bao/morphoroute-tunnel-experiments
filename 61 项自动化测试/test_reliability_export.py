from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ReliabilityExportTests(unittest.TestCase):
    def test_bundle_is_exported_as_five_class_logits_with_target_labels(self):
        from morphoroute.evaluation.reliability_export import (
            reliability_rows_from_bundle,
        )
        from morphoroute.evaluation.schema import (
            PredictionBundle,
            PredictionImage,
            PredictionInstance,
        )

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        target = PredictionInstance("crack", 1.0, (2, 2, 4, 4), mask)
        prediction = PredictionInstance("crack", 0.9, (2, 2, 4, 4), mask)
        bundle = PredictionBundle(
            experiment_id="EXP-09",
            split="D2",
            class_names=(
                "crack",
                "water_leakage",
                "spalling",
                "efflorescence",
                "exposed_rebar",
            ),
            images=(
                PredictionImage(
                    image_id="i1",
                    group_id="g1",
                    width=8,
                    height=8,
                    predictions=(prediction,),
                    targets=(target,),
                ),
            ),
        )

        rows = reliability_rows_from_bundle(bundle, model="B9", seed=42)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["split"], "D2")
        self.assertEqual(rows[0]["label"], 0)
        self.assertFalse(rows[0]["high_risk"])
        self.assertEqual(len(json.loads(rows[0]["logits_json"])), 5)


if __name__ == "__main__":
    unittest.main()
