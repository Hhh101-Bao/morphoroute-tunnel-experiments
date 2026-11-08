from __future__ import annotations

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.seg_heads.panoptic_fusion_heads import (
    MaskFormerFusionHead,
)
from mmdet.registry import MODELS
from mmdet.structures.mask import mask2bbox


@MODELS.register_module()
class MorphoRouteMemoryEfficientMaskFormerFusionHead(
    MaskFormerFusionHead
):
    """Output-equivalent MaskFormer instance postprocess with bounded VRAM.

    MMDetection's reference implementation expands all selected full-
    resolution logits and creates multiple FP32 tensors of shape
    ``(max_per_image, H, W)``.  Large original images can therefore exceed
    GPU memory even with a batch size of one.  This implementation keeps the
    same class ranking, interpolation, threshold, mask-quality score, boxes,
    and output ordering, while evaluating one unique query at a time.
    """

    def instance_postprocess(
        self,
        mask_cls: Tensor,
        mask_pred: Tensor,
    ) -> InstanceData:
        max_per_image = self.test_cfg.get("max_per_image", 100)
        num_queries = mask_cls.shape[0]

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = (
            torch.arange(self.num_classes, device=mask_cls.device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, top_indices = scores.flatten(0, 1).topk(
            max_per_image,
            sorted=False,
        )
        labels_per_image = labels[top_indices]
        query_indices = top_indices // self.num_classes

        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        query_indices = query_indices[is_thing]

        selected_count = int(query_indices.numel())
        height, width = mask_pred.shape[-2:]
        mask_pred_binary = torch.empty(
            (selected_count, height, width),
            dtype=torch.bool,
            device=mask_pred.device,
        )
        mask_scores_per_image = torch.empty_like(scores_per_image)

        unique_query_indices = torch.unique(query_indices, sorted=False)
        for query_index in unique_query_indices:
            candidate_positions = torch.nonzero(
                query_indices == query_index,
                as_tuple=False,
            ).flatten()
            logits = mask_pred[int(query_index.item())]
            binary = logits > 0
            probabilities = logits.sigmoid()
            probabilities.masked_fill_(~binary, 0)
            quality = probabilities.sum() / (
                binary.sum().to(probabilities.dtype) + 1e-6
            )

            mask_scores_per_image[candidate_positions] = quality
            for candidate_position in candidate_positions:
                mask_pred_binary[int(candidate_position.item())].copy_(
                    binary
                )

        det_scores = scores_per_image * mask_scores_per_image
        bboxes = mask2bbox(mask_pred_binary)

        results = InstanceData()
        results.bboxes = bboxes
        results.labels = labels_per_image
        results.scores = det_scores
        results.masks = mask_pred_binary
        return results
