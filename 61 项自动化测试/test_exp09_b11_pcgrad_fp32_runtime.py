from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

try:
    import numpy as np
    import torch
    from mmengine.optim import AmpOptimWrapper
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    from mmdet.structures.mask import BitmapMasks

    from morphoroute.mmdet.models.safe_batch_pad import (
        MorphoRouteSafeBatchFixedSizePad,
    )
    from morphoroute.runtime.pcgrad_fp32 import (
        install_pcgrad_fp32_optim_wrapper,
    )
    from morphoroute.mmdet.models.morphology_targets import (
        MorphologyTargetEncoder,
    )
    from morphoroute.models.morphology import (
        MorphologyPrior,
        extract_morphology,
    )

    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False


@unittest.skipUnless(
    RUNTIME_AVAILABLE,
    "torch/mmengine runtime is available only in the training environment",
)
class Exp09B11PcgradFp32RuntimeTests(unittest.TestCase):
    def test_empty_augmented_mask_is_excluded_only_from_route_supervision(self):
        line = np.zeros((9, 9), dtype=bool)
        line[4, 1:8] = True
        area = np.zeros((9, 9), dtype=bool)
        area[2:7, 2:7] = True
        prior = MorphologyPrior.fit(
            np.stack([extract_morphology(line), extract_morphology(area)])
        )
        encoder = MorphologyTargetEncoder(prior)

        targets, weights = encoder(
            [np.zeros((9, 9), dtype=bool), line],
            matched_query_indices=[2, 7],
            query_count=10,
        )

        self.assertEqual(weights.nonzero().flatten().tolist(), [7])
        self.assertEqual(float(weights[2]), 0.0)
        self.assertEqual(float(targets[2].sum()), 0.0)
        self.assertAlmostEqual(float(targets[7].sum()), 1.0, places=6)

    def test_safe_batch_pad_preserves_oversized_width_and_mask_metadata(self):
        transform = MorphoRouteSafeBatchFixedSizePad(
            size=(640, 640),
            pad_mask=True,
        )
        inputs = torch.zeros((1, 3, 512, 704), dtype=torch.float32)
        sample = DetDataSample()
        instances = InstanceData()
        instances.masks = BitmapMasks(
            np.zeros((2, 512, 704), dtype=np.uint8),
            512,
            704,
        )
        sample.gt_instances = instances

        padded, samples = transform(inputs, [sample])

        self.assertEqual(tuple(padded.shape[-2:]), (640, 704))
        self.assertEqual(samples[0].batch_input_shape, (640, 704))
        self.assertEqual(samples[0].pad_shape, (640, 704))
        self.assertEqual(samples[0].gt_instances.masks.height, 640)
        self.assertEqual(samples[0].gt_instances.masks.width, 704)
        self.assertEqual(
            samples[0].gt_instances.masks.masks.shape,
            (2, 640, 704),
        )

    def test_true_fp32_pcgrad_accumulates_updates_and_loads_amp_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper_class = install_pcgrad_fp32_optim_wrapper(directory)
            self.assertFalse(issubclass(wrapper_class, AmpOptimWrapper))

            parameter = torch.nn.Parameter(torch.tensor([1.0]))
            optimizer = torch.optim.AdamW([parameter], lr=0.01)
            wrapper = wrapper_class(
                optimizer=optimizer,
                accumulative_counts=2,
                clip_grad={"max_norm": 0.1, "norm_type": 2},
            )

            initial = parameter.detach().clone()
            wrapper.update_params_from_loss_dict(
                {
                    "loss_cls": parameter.square().sum(),
                    "loss_route": 0.5 * parameter.square().sum(),
                }
            )
            self.assertTrue(torch.equal(parameter.detach(), initial))

            wrapper.update_params_from_loss_dict(
                {
                    "loss_cls": parameter.square().sum(),
                    "loss_route": 0.5 * parameter.square().sum(),
                }
            )
            self.assertTrue(torch.isfinite(parameter).all())
            self.assertFalse(torch.equal(parameter.detach(), initial))

            state = wrapper.state_dict()
            state["loss_scaler"] = {
                "scale": 1024.0,
                "growth_interval": 2000,
            }
            replacement = torch.nn.Parameter(torch.tensor([1.0]))
            replacement_wrapper = wrapper_class(
                optimizer=torch.optim.AdamW([replacement], lr=0.01),
                accumulative_counts=2,
                clip_grad={"max_norm": 0.1, "norm_type": 2},
            )
            replacement_wrapper.load_state_dict(state)
            self.assertNotIn(
                "loss_scaler", replacement_wrapper.state_dict()
            )

    def test_nonfinite_loss_is_rejected_before_update(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper_class = install_pcgrad_fp32_optim_wrapper(directory)
            parameter = torch.nn.Parameter(torch.tensor([1.0]))
            wrapper = wrapper_class(
                optimizer=torch.optim.AdamW([parameter], lr=0.01),
                accumulative_counts=1,
            )
            initial = parameter.detach().clone()
            with self.assertRaisesRegex(Exception, "non-finite loss"):
                wrapper.update_params_from_loss_dict(
                    {
                        "loss_cls": (
                            parameter.sum() * torch.tensor(float("nan"))
                        ),
                        "loss_route": parameter.square().sum(),
                    }
                )
            self.assertTrue(torch.equal(parameter.detach(), initial))


if __name__ == "__main__":
    unittest.main()
