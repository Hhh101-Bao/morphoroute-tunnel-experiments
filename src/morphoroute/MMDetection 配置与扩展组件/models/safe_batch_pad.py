from __future__ import annotations

from typing import Optional

import torch.nn.functional as functional
from torch import Tensor, nn

from mmdet.registry import MODELS


@MODELS.register_module(force=True)
class MorphoRouteSafeBatchFixedSizePad(nn.Module):
    """Pad to at least ``size`` while preserving oversized dimensions.

    MMDetection's ``BatchFixedSizePad`` records the configured fixed size in
    sample metadata even when one input dimension is already larger and is
    therefore not cropped by ``torch.nn.functional.pad``.  That mismatch can
    make bitmap masks impossible to reshape.  This implementation records and
    uses the actual padded tensor shape for images, masks, and segmentation.
    """

    def __init__(
        self,
        size: tuple[int, int],
        img_pad_value: float = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
    ) -> None:
        super().__init__()
        if len(size) != 2 or min(int(value) for value in size) <= 0:
            raise ValueError(f"size must contain two positive values: {size}")
        self.size = tuple(int(value) for value in size)
        self.img_pad_value = img_pad_value
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value

    def forward(
        self,
        inputs: Tensor,
        data_samples: Optional[list] = None,
    ) -> tuple[Tensor, Optional[list]]:
        source_height, source_width = inputs.shape[-2:]
        target_height = max(source_height, self.size[0])
        target_width = max(source_width, self.size[1])
        pad_height = target_height - source_height
        pad_width = target_width - source_width

        if pad_height or pad_width:
            inputs = functional.pad(
                inputs,
                (0, pad_width, 0, pad_height),
                value=self.img_pad_value,
            )

        if data_samples is None:
            return inputs, data_samples

        actual_shape = (target_height, target_width)
        for data_sample in data_samples:
            data_sample.set_metainfo(
                {
                    "batch_input_shape": actual_shape,
                    "pad_shape": actual_shape,
                }
            )
            if self.pad_mask and "masks" in data_sample.gt_instances:
                data_sample.gt_instances.masks = (
                    data_sample.gt_instances.masks.pad(
                        actual_shape,
                        pad_val=self.mask_pad_value,
                    )
                )
            if self.pad_seg and "gt_sem_seg" in data_sample:
                data_sample.gt_sem_seg.sem_seg = functional.pad(
                    data_sample.gt_sem_seg.sem_seg,
                    (0, pad_width, 0, pad_height),
                    value=self.seg_pad_value,
                )

        return inputs, data_samples
