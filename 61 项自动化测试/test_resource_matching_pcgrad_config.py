from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ResourceMatchingTests(unittest.TestCase):
    def test_registered_matches_are_within_five_percent(self):
        from morphoroute.training.resource_matching import (
            build_resource_matches,
        )

        matches = build_resource_matches(
            channels=64,
            feature_height=32,
            feature_width=32,
            tolerance=0.05,
        )

        self.assertLessEqual(matches.b2.parameter_relative_error, 0.05)
        self.assertLessEqual(matches.b3.flops_relative_error, 0.05)
        self.assertGreater(matches.b2.adapter_rank, 0)
        self.assertGreater(matches.b3.adapter_depth, 0)
        self.assertGreater(len(matches.parameter_candidates), 1)
        self.assertGreater(len(matches.flops_candidates), 1)

    def test_b9_parameter_target_matches_instantiated_increment(self):
        from morphoroute.mmdet.models.morphoroute_head import (
            MorphoRouteMask2FormerHead,
        )
        from morphoroute.training.resource_matching import (
            build_resource_matches,
        )

        b0 = MorphoRouteMask2FormerHead(
            in_channels=64,
            query_dim=64,
            variant_id="B0",
        )
        b9 = MorphoRouteMask2FormerHead(
            in_channels=64,
            query_dim=64,
            variant_id="B9",
        )
        increment = sum(
            parameter.numel() for parameter in b9.parameters()
        ) - sum(parameter.numel() for parameter in b0.parameters())
        matches = build_resource_matches(
            channels=64,
            feature_height=16,
            feature_width=16,
        )

        self.assertEqual(matches.target_parameters, increment)

    def test_variant_factory_applies_selected_capacity_controls(self):
        from morphoroute.mmdet.models.variant_factory import (
            build_variant_model_cfg,
        )
        from morphoroute.training.resource_matching import (
            build_resource_matches,
        )

        matches = build_resource_matches(
            channels=64,
            feature_height=16,
            feature_width=16,
            tolerance=0.05,
        )
        b2 = build_variant_model_cfg(
            "B2",
            in_channels=64,
            query_dim=64,
            resource_matches=matches,
        )
        b3 = build_variant_model_cfg(
            "B3",
            in_channels=64,
            query_dim=64,
            resource_matches=matches,
        )

        self.assertEqual(b2["capacity_adapter_rank"], matches.b2.adapter_rank)
        self.assertEqual(b3["compute_adapter_depth"], matches.b3.adapter_depth)


class PCGradConfigurationTests(unittest.TestCase):
    def test_stage_a_zero_structure_terms_fall_back_to_primary_gradient(self):
        from morphoroute.mmdet.models.pcgrad_runtime import (
            split_pcgrad_losses,
        )

        parameter = torch.tensor(2.0, requires_grad=True)
        losses = {
            "loss_mask": parameter.square(),
            "loss_route": parameter * 0.0,
            "loss_cldice": parameter * 0.0,
            "loss_boundary": parameter * 0.0,
        }

        self.assertIsNone(split_pcgrad_losses(losses))

    def test_b11_uses_registered_true_fp32_pcgrad_wrapper_and_detector(self):
        from morphoroute.mmdet.config_builder import (
            morphoroute_model_overrides,
        )

        overrides = morphoroute_model_overrides(
            variant_id="B11",
            morphology_prior_path="prior.json",
            tau=0.75,
            capacity_match=None,
        )

        self.assertEqual(overrides["model"]["type"], "MorphoRouteMask2Former")
        self.assertEqual(
            overrides["model"]["panoptic_head"]["loss_cls"]["class_weight"],
            [1.0] * 5 + [0.1],
        )
        self.assertEqual(
            overrides["optim_wrapper"]["type"],
            "PCGradFP32OptimWrapper",
        )

    def test_non_pcgrad_variant_keeps_standard_optimizer(self):
        from morphoroute.mmdet.config_builder import (
            morphoroute_model_overrides,
        )

        overrides = morphoroute_model_overrides(
            variant_id="B9",
            morphology_prior_path="prior.json",
            tau=0.75,
            capacity_match=None,
        )

        self.assertNotIn("optim_wrapper", overrides)


if __name__ == "__main__":
    unittest.main()
