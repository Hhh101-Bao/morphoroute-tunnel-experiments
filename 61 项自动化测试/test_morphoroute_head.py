from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class MorphoRouteHeadTests(unittest.TestCase):
    def test_soft_cldice_is_zero_for_identical_thin_structures(self):
        from morphoroute.mmdet.models.morphoroute_head import (
            soft_cldice_loss,
        )

        mask = torch.zeros(1, 1, 9, 9)
        mask[:, :, 4, 1:8] = 1.0

        self.assertLess(float(soft_cldice_loss(mask, mask)), 1e-5)

    def test_only_matched_queries_receive_route_targets(self):
        from morphoroute.mmdet.models.morphology_targets import (
            MorphologyTargetEncoder,
        )
        from morphoroute.models.morphology import MorphologyPrior, extract_morphology

        line = np.zeros((12, 12), dtype=bool)
        line[5, 1:11] = True
        area = np.zeros((12, 12), dtype=bool)
        area[2:10, 3:9] = True
        prior = MorphologyPrior.fit(
            np.stack([extract_morphology(line), extract_morphology(area)])
        )
        encoder = MorphologyTargetEncoder(prior)

        targets, weights = encoder(
            [line, area],
            matched_query_indices=[2, 7],
            query_count=10,
        )

        self.assertEqual(weights.nonzero().flatten().tolist(), [2, 7])
        self.assertTrue(
            torch.allclose(
                targets.sum(dim=-1)[weights.bool()],
                torch.ones(2),
            )
        )

    def test_b9_returns_segmentation_structure_and_route_losses(self):
        from morphoroute.mmdet.models.morphoroute_head import (
            MorphoRouteMask2FormerHead,
        )

        head = MorphoRouteMask2FormerHead(
            in_channels=8,
            query_dim=8,
            variant_id="B9",
            tau=0.75,
        )
        features = torch.randn(1, 8, 8, 8, requires_grad=True)
        queries = torch.randn(1, 3, 8, requires_grad=True)
        target_masks = torch.zeros(1, 3, 8, 8)
        target_masks[:, :, 2:6, 2:6] = 1
        route_targets = torch.tensor(
            [[[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]]]
        )
        route_weights = torch.ones(1, 3)

        losses = head.loss(
            features,
            queries,
            target_masks,
            route_targets,
            route_weights,
        )

        self.assertTrue(
            {
                "loss_cls",
                "loss_mask",
                "loss_dice",
                "loss_route",
                "loss_cldice",
                "loss_boundary",
            }
            <= set(losses)
        )
        sum(losses.values()).backward()

    def test_b9_inference_uses_query_features_without_ground_truth(self):
        from morphoroute.mmdet.models.morphoroute_head import (
            MorphoRouteMask2FormerHead,
        )

        head = MorphoRouteMask2FormerHead(
            in_channels=8,
            query_dim=8,
            variant_id="B9",
            tau=0.75,
        ).eval()
        output = head.predict(
            torch.randn(1, 8, 8, 8),
            torch.randn(1, 3, 8),
        )

        self.assertEqual(output["route_probabilities"].shape[-1], 2)
        self.assertFalse(output["ground_truth_inputs_used"])

    def test_b10_disables_low_confidence_dual_activation(self):
        from morphoroute.mmdet.models.morphoroute_head import (
            MorphoRouteMask2FormerHead,
        )

        head = MorphoRouteMask2FormerHead(
            in_channels=8,
            query_dim=8,
            variant_id="B10",
            tau=0.75,
        ).eval()
        output = head.predict(
            torch.randn(1, 8, 8, 8),
            torch.randn(1, 3, 8),
        )

        self.assertTrue(torch.all(output["active_experts"] == 1))

    def test_b9_training_transitions_from_shared_to_soft_to_sparse(self):
        from morphoroute.mmdet.models.morphoroute_head import (
            MorphoRouteMask2FormerHead,
        )

        head = MorphoRouteMask2FormerHead(
            in_channels=8,
            query_dim=8,
            variant_id="B9",
            tau=0.75,
            stage_warmup_end=1,
            stage_sparse_start=2,
        ).train()
        features = torch.randn(1, 8, 8, 8)
        queries = torch.randn(1, 3, 8)

        head.training_update = 0
        stage_a = head(features, queries)
        head.training_update = 1
        stage_b = head(features, queries)
        head.training_update = 2
        stage_c = head(features, queries)

        self.assertEqual(stage_a["training_stage"], "A")
        self.assertTrue(torch.all(stage_a["active_experts"] == 0))
        self.assertEqual(stage_b["training_stage"], "B")
        self.assertTrue(torch.all(stage_b["active_experts"] == 2))
        self.assertEqual(stage_c["training_stage"], "C")
        self.assertTrue(
            torch.all(
                (stage_c["active_experts"] == 1)
                | (stage_c["active_experts"] == 2)
            )
        )


if __name__ == "__main__":
    unittest.main()
