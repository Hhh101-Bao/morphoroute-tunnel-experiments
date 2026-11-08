from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class MMDetRouteDiagnosticTests(unittest.TestCase):
    def test_query_diagnostics_are_reduced_per_image(self):
        from morphoroute.runtime.mmdet_predictions import (
            route_diagnostics_from_model,
        )

        probabilities = torch.tensor(
            [[[0.9, 0.1], [0.4, 0.6], [0.5, 0.5]]]
        )
        active = torch.tensor([[1, 1, 2]])
        entropy = -(
            probabilities
            * torch.log(probabilities.clamp_min(1e-8))
        ).sum(dim=-1)
        model = SimpleNamespace(
            panoptic_head=SimpleNamespace(
                latest_route_diagnostics={
                    "route_probabilities": probabilities,
                    "active_experts": active,
                    "route_entropy": entropy,
                },
                latest_training_stage="inference",
            )
        )

        row = route_diagnostics_from_model(model, image_id="image-1")

        self.assertEqual(row["image_id"], "image-1")
        self.assertAlmostEqual(row["mean_active_experts"], 4 / 3)
        self.assertAlmostEqual(row["dual_activation_rate"], 1 / 3)
        self.assertAlmostEqual(row["line_top1_fraction"], 2 / 3)
        self.assertAlmostEqual(row["area_top1_fraction"], 1 / 3)
        self.assertEqual(row["training_stage"], "inference")


if __name__ == "__main__":
    unittest.main()
