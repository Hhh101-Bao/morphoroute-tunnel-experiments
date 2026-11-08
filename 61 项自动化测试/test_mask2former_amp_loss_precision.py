from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class Mask2FormerAmpLossPrecisionTests(unittest.TestCase):
    def test_mmdet_loss_matching_receives_fp32_tensors(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the AMP regression test")

        from mmdet.models.dense_heads import (
            Mask2FormerHead as MMDetMask2FormerHead,
        )
        from morphoroute.mmdet.models.morphoroute_head import (
            MorphoRouteMask2FormerHead,
        )

        observed = {}

        def probe(
            instance,
            cls_scores,
            mask_preds,
            batch_gt_instances,
            batch_img_metas,
        ):
            denominator = (
                cls_scores.new_tensor([6.0])
                * 12544
            )

            observed["cls_dtype"] = cls_scores.dtype
            observed["mask_dtype"] = mask_preds.dtype
            observed["autocast_enabled"] = torch.is_autocast_enabled()
            observed["denominator"] = denominator

            zero = cls_scores.float().sum() * 0.0
            return zero, zero, zero

        head = MorphoRouteMask2FormerHead.__new__(
            MorphoRouteMask2FormerHead
        )

        cls_scores = torch.ones(
            1,
            10,
            6,
            dtype=torch.float16,
            device="cuda",
        )
        mask_preds = torch.ones(
            1,
            10,
            8,
            8,
            dtype=torch.float16,
            device="cuda",
        )

        with patch.object(
            MMDetMask2FormerHead,
            "_loss_by_feat_single",
            probe,
        ):
            with torch.cuda.amp.autocast():
                MorphoRouteMask2FormerHead._loss_by_feat_single(
                    head,
                    cls_scores,
                    mask_preds,
                    [],
                    [],
                )

        self.assertEqual(observed["cls_dtype"], torch.float32)
        self.assertEqual(observed["mask_dtype"], torch.float32)
        self.assertFalse(observed["autocast_enabled"])
        self.assertTrue(
            torch.isfinite(observed["denominator"]).all().item()
        )
        self.assertEqual(
            observed["denominator"].item(),
            75264.0,
        )


if __name__ == "__main__":
    unittest.main()
