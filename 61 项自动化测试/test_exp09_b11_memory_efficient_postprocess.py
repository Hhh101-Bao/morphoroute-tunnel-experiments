from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class MemoryEfficientPostprocessSourceTests(unittest.TestCase):
    def test_streaming_implementation_avoids_batched_fp32_candidates(self):
        source_path = (
            SOURCE_ROOT
            / "morphoroute"
            / "MMDetection 配置与扩展组件"
            / "models"
            / "memory_efficient_fusion.py"
        )
        self.assertTrue(source_path.is_file())
        text = source_path.read_text(encoding="utf-8")
        self.assertIn(
            "class MorphoRouteMemoryEfficientMaskFormerFusionHead", text
        )
        self.assertIn("for query_index in unique_query_indices", text)
        self.assertNotIn("mask_pred = mask_pred[query_indices]", text)
        self.assertNotIn("(mask_pred > 0).float()", text)

    def test_launcher_runs_full_d2_probe_before_formal_resume(self):
        launcher = (
            PROJECT_ROOT
            / "工程辅助运行脚本"
            / "launch_exp09_b11_true_fp32_recovery.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("verify_exp09_b11_memory_efficient_d2.py", launcher)
        self.assertIn("EXP09_B11_MEMORY_EFFICIENT_D2=PASSED", launcher)
        self.assertIn("iter_80000.pth", launcher)


@unittest.skipUnless(
    importlib.util.find_spec("torch")
    and importlib.util.find_spec("mmengine")
    and importlib.util.find_spec("mmdet"),
    "requires the AutoDL torch/MMDetection runtime",
)
class MemoryEfficientPostprocessRuntimeTests(unittest.TestCase):
    def test_matches_official_instance_postprocess_on_small_tensors(self):
        import torch
        from mmdet.models.seg_heads.panoptic_fusion_heads import (
            MaskFormerFusionHead,
        )

        from morphoroute.mmdet.models.memory_efficient_fusion import (
            MorphoRouteMemoryEfficientMaskFormerFusionHead,
        )

        test_cfg = {
            "instance_on": True,
            "max_per_image": 12,
            "panoptic_on": False,
            "semantic_on": False,
        }
        official = MaskFormerFusionHead(
            num_things_classes=3,
            num_stuff_classes=0,
            test_cfg=test_cfg,
        )
        streaming = MorphoRouteMemoryEfficientMaskFormerFusionHead(
            num_things_classes=3,
            num_stuff_classes=0,
            test_cfg=test_cfg,
        )
        generator = torch.Generator().manual_seed(20260925)
        mask_cls = torch.randn(7, 4, generator=generator)
        mask_pred = torch.randn(7, 23, 31, generator=generator)

        expected = official.instance_postprocess(mask_cls, mask_pred)
        actual = streaming.instance_postprocess(mask_cls, mask_pred)

        self.assertTrue(torch.equal(actual.labels, expected.labels))
        self.assertTrue(torch.equal(actual.masks, expected.masks))
        self.assertTrue(torch.equal(actual.bboxes, expected.bboxes))
        self.assertTrue(
            torch.allclose(actual.scores, expected.scores, atol=1e-6, rtol=0)
        )


if __name__ == "__main__":
    unittest.main()
