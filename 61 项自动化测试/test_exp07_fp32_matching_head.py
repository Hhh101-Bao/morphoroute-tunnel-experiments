from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

import morphoroute.mmdet.models  # noqa: F401 - populate registry
from mmdet.registry import MODELS
from morphoroute.models.experts import SparseExpertDecoder


class _Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, value):
        return value * self.factor


class _Router(nn.Module):
    def forward(self, value):
        shape = (*value.shape[:-1], 2)
        return value.new_zeros(shape)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MatchingHeadPrecisionTest(unittest.TestCase):
    def test_matching_head_keeps_large_finite_mask_logits_in_fp32(self):
        head_class = MODELS.get("MorphoRouteMask2FormerHead")
        self.assertIsNotNone(head_class)

        fake = SimpleNamespace(
            transformer_decoder=SimpleNamespace(post_norm=nn.Identity()),
            capacity_adapter=nn.Identity(),
            cls_embed=_Scale(1.0),
            mask_embed=_Scale(32.0),
            compute_adapter=nn.Identity(),
            morphology_router=_Router(),
            variant={
                "router": {"kind": "learned_generic"},
                "experts": {
                    "enabled": True,
                    "line": True,
                    "area": True,
                    "homogeneous": False,
                },
            },
            num_classes=5,
            _route_logits_by_layer=[],
            _shared_mask_feature_cache=None,
            _expert_feature_cache=None,
            training=True,
            training_update=2000,
            stage_warmup_end=1000,
            stage_sparse_start=40000,
            latest_training_stage="A",
            sparse_decoder=SparseExpertDecoder(0.75).cuda(),
            line_expert=nn.Identity(),
            area_expert=nn.Identity(),
            homogeneous_area_expert=nn.Identity(),
            intervention_mode="normal",
            latest_route_diagnostics=None,
            num_heads=1,
        )

        decoder_out = torch.ones(
            (1, 2, 256), device="cuda", dtype=torch.float16
        )
        mask_feature = torch.full(
            (1, 256, 4, 4),
            32.0,
            device="cuda",
            dtype=torch.float16,
        )

        with torch.cuda.amp.autocast(enabled=True):
            cls_pred, mask_pred, attention = head_class._forward_head(
                fake,
                decoder_out,
                mask_feature,
                (4, 4),
            )

        self.assertEqual(cls_pred.dtype, torch.float32)
        self.assertEqual(mask_pred.dtype, torch.float32)
        self.assertTrue(torch.isfinite(cls_pred).all().item())
        self.assertTrue(torch.isfinite(mask_pred).all().item())
        self.assertEqual(float(mask_pred.max().item()), 262144.0)
        self.assertEqual(attention.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
