from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
import json
from pathlib import Path
import numpy as np

from morphoroute.models.experts import AreaExpert, LineExpert, SparseExpertDecoder
from morphoroute.models.router import MorphologyRouter
from morphoroute.models.variants import build_variant_config
from morphoroute.models.morphology import (
    MorphologyPrior,
    extract_observable_morphology,
)
from morphoroute.models.capacity_controls import (
    FlopsMatchedSharedAdapter,
    ParameterMatchedQueryAdapter,
    ParameterMatchedSharedAdapter,
)
from morphoroute.models.interventions import apply_intervention


try:
    from mmdet.registry import MODELS
except ImportError:
    MODELS = None


def _register(cls):
    return cls


def _soft_erode(mask):
    vertical = -functional.max_pool2d(
        -mask,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0),
    )
    horizontal = -functional.max_pool2d(
        -mask,
        kernel_size=(1, 3),
        stride=1,
        padding=(0, 1),
    )
    return torch.minimum(vertical, horizontal)


def _soft_open(mask):
    eroded = _soft_erode(mask)
    return functional.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)


def soft_skeletonize(mask, *, iterations: int = 10):
    opened = _soft_open(mask)
    skeleton = functional.relu(mask - opened)
    eroded = mask
    for _ in range(iterations):
        eroded = _soft_erode(eroded)
        opened = _soft_open(eroded)
        delta = functional.relu(eroded - opened)
        skeleton = skeleton + functional.relu(
            delta - skeleton * delta
        )
    return skeleton


def soft_cldice_loss(
    probability,
    target,
    *,
    iterations: int = 10,
    epsilon: float = 1e-6,
):
    prediction_skeleton = soft_skeletonize(
        probability,
        iterations=iterations,
    )
    target_skeleton = soft_skeletonize(
        target,
        iterations=iterations,
    )
    topology_precision = (
        (prediction_skeleton * target).sum(dim=(-2, -1))
        + epsilon
    ) / (
        prediction_skeleton.sum(dim=(-2, -1)) + epsilon
    )
    topology_sensitivity = (
        (target_skeleton * probability).sum(dim=(-2, -1))
        + epsilon
    ) / (
        target_skeleton.sum(dim=(-2, -1)) + epsilon
    )
    return (
        1
        - (2 * topology_precision * topology_sensitivity)
        / (topology_precision + topology_sensitivity + epsilon)
    ).mean()


@_register
class MorphoRouteMask2FormerHead(nn.Module):
    """Query-routing module used by the MorphoRoute Mask2Former extension.

    The module consumes RGB-derived pixel/query features only. Morphology masks
    enter exclusively through ``loss`` as matched-query supervision.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        query_dim: int,
        variant_id: str = "B9",
        tau: float = 0.75,
        num_classes: int = 5,
        capacity_adapter_rank: int = 0,
        compute_adapter_depth: int = 0,
        intervention_mode: str = "normal",
        stage_warmup_end: int = 1000,
        stage_sparse_start: int = 40000,
        **_: object,
    ) -> None:
        super().__init__()
        if stage_warmup_end < 0 or stage_sparse_start <= stage_warmup_end:
            raise ValueError(
                "stage boundaries must satisfy "
                "0 <= stage_warmup_end < stage_sparse_start"
            )
        self.variant_id = variant_id
        self.variant = build_variant_config(variant_id)
        self.num_classes = num_classes
        self.stage_warmup_end = int(stage_warmup_end)
        self.stage_sparse_start = int(stage_sparse_start)
        self.training_update = 0
        self.pixel_projection = nn.Conv2d(in_channels, query_dim, kernel_size=1)
        self.query_projection = nn.Linear(query_dim, query_dim)
        self.classifier = nn.Linear(query_dim, num_classes + 1)
        expert_cfg = self.variant["experts"]
        self.router = (
            MorphologyRouter(query_dim)
            if expert_cfg["enabled"]
            else None
        )
        self.line_expert = (
            LineExpert(query_dim)
            if expert_cfg["line"] or expert_cfg["homogeneous"]
            else nn.Identity()
        )
        self.area_expert = (
            AreaExpert(query_dim)
            if expert_cfg["area"]
            else nn.Identity()
        )
        self.homogeneous_area_expert = (
            LineExpert(query_dim)
            if expert_cfg["homogeneous"]
            else nn.Identity()
        )
        self.sparse_decoder = SparseExpertDecoder(
            tau if self.variant["router"]["dual_activation"] else 0.5
        )
        self.intervention_mode = intervention_mode
        self.capacity_adapter = (
            ParameterMatchedSharedAdapter(query_dim, capacity_adapter_rank)
            if capacity_adapter_rank > 0
            else nn.Identity()
        )
        self.compute_adapter = (
            FlopsMatchedSharedAdapter(query_dim, compute_adapter_depth)
            if compute_adapter_depth > 0
            else nn.Identity()
        )

    def _expert_features(self, pixels):
        expert_cfg = self.variant["experts"]
        line = (
            self.line_expert(pixels)
            if expert_cfg["enabled"] and expert_cfg["line"]
            else pixels
        )
        area = (
            self.area_expert(pixels)
            if expert_cfg["enabled"] and expert_cfg["area"]
            else pixels
        )
        if expert_cfg["enabled"] and expert_cfg["homogeneous"]:
            line = self.line_expert(pixels)
            area = self.homogeneous_area_expert(pixels)
        return line, area

    def forward(self, features, queries):
        pixels = self.compute_adapter(
            self.capacity_adapter(self.pixel_projection(features))
        )
        projected_queries = self.query_projection(queries)
        class_logits = self.classifier(projected_queries)
        route_logits = (
            self.router(projected_queries)
            if self.router is not None
            else projected_queries.new_zeros(
                (*projected_queries.shape[:-1], 2)
            )
        )
        if not self.training:
            training_stage = "inference"
        elif self.training_update < self.stage_warmup_end:
            training_stage = "A"
        elif self.training_update < self.stage_sparse_start:
            training_stage = "B"
        else:
            training_stage = "C"
        experts_active = bool(
            self.variant["experts"]["enabled"] and training_stage != "A"
        )
        if not experts_active:
            mask_logits = torch.einsum(
                "bqd,bdhw->bqhw",
                projected_queries,
                pixels,
            )
            probabilities = torch.softmax(route_logits, dim=-1)
            diagnostics = {
                "route_probabilities": probabilities,
                "active_experts": torch.zeros(
                    probabilities.shape[:-1],
                    dtype=torch.long,
                    device=probabilities.device,
                ),
                "route_entropy": -(
                    probabilities
                    * torch.log(probabilities.clamp_min(1e-8))
                ).sum(dim=-1),
            }
        else:
            self.sparse_decoder.sparse_training = training_stage == "C"
            line, area = self._expert_features(pixels)
            line, area, route_logits = apply_intervention(
                line,
                area,
                route_logits,
                self.intervention_mode,
            )
            mask_logits, diagnostics = self.sparse_decoder(
                projected_queries,
                line,
                area,
                route_logits,
            )
        return {
            "class_logits": class_logits,
            "mask_logits": mask_logits,
            "route_logits": route_logits,
            "training_stage": training_stage,
            **diagnostics,
        }

    @staticmethod
    def _dice_loss(probability, target):
        intersection = (probability * target).sum(dim=(-2, -1))
        denominator = probability.sum(dim=(-2, -1)) + target.sum(
            dim=(-2, -1)
        )
        return (1.0 - (2 * intersection + 1) / (denominator + 1)).mean()

    @staticmethod
    def _boundary(mask):
        eroded = -functional.max_pool2d(-mask, 3, 1, 1)
        return (mask - eroded).abs()

    def loss(
        self,
        features,
        queries,
        target_masks,
        route_targets,
        route_weights,
    ):
        output = self(features, queries)
        mask_logits = output["mask_logits"]
        targets = target_masks.float()
        probability = torch.sigmoid(mask_logits)
        loss_mask = functional.binary_cross_entropy_with_logits(
            mask_logits,
            targets,
        )
        loss_dice = self._dice_loss(probability, targets)
        weights = route_weights.float()
        route_per_query = functional.kl_div(
            functional.log_softmax(output["route_logits"], dim=-1),
            route_targets,
            reduction="none",
        ).sum(dim=-1)
        if output["training_stage"] == "A":
            loss_route = route_per_query.sum() * 0.0
        else:
            loss_route = (
                (route_per_query * weights).sum()
                / weights.sum().clamp_min(1.0)
            )
        if (
            self.variant["loss"]["structure"]
            and output["training_stage"] != "A"
        ):
            loss_cldice = soft_cldice_loss(
                probability,
                targets,
            )
            loss_boundary = functional.l1_loss(
                self._boundary(probability),
                self._boundary(targets),
            )
        else:
            zero = mask_logits.sum() * 0.0
            loss_cldice = zero
            loss_boundary = zero
        loss_cls = output["class_logits"].sum() * 0.0
        return {
            "loss_cls": loss_cls,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_route": loss_route,
            "loss_cldice": loss_cldice,
            "loss_boundary": loss_boundary,
        }

    @torch.no_grad()
    def predict(self, features, queries):
        output = self(features, queries)
        return {
            "class_logits": output["class_logits"],
            "mask_logits": output["mask_logits"],
            "route_probabilities": output["route_probabilities"],
            "active_experts": output["active_experts"],
            "route_entropy": output["route_entropy"],
            "ground_truth_inputs_used": False,
        }


if MODELS is not None:
    from mmdet.models.dense_heads import Mask2FormerHead as _MMDetMask2FormerHead

    @MODELS.register_module(force=True)
    class MorphoRouteMask2FormerHead(_MMDetMask2FormerHead):
        """MMDetection-compatible Mask2Former head with query routing."""

        def __init__(
            self,
            *args,
            variant_id: str = "B9",
            tau: float = 0.75,
            morphology_prior_path: str | None = None,
            route_loss_weight: float = 1.0,
            cldice_loss_weight: float = 1.0,
            boundary_loss_weight: float = 1.0,
            capacity_adapter_rank: int = 0,
            compute_adapter_depth: int = 0,
            intervention_mode: str = "normal",
            stage_warmup_end: int = 1000,
            stage_sparse_start: int = 40000,
            **kwargs,
        ):
            if stage_warmup_end < 0 or stage_sparse_start <= stage_warmup_end:
                raise ValueError(
                    "stage boundaries must satisfy "
                    "0 <= stage_warmup_end < stage_sparse_start"
                )
            feat_channels = int(kwargs.get("feat_channels", 256))
            out_channels = int(kwargs.get("out_channels", 256))
            super().__init__(*args, **kwargs)
            self.variant_id = variant_id
            self.variant = build_variant_config(variant_id)
            self.stage_warmup_end = int(stage_warmup_end)
            self.stage_sparse_start = int(stage_sparse_start)
            self.training_update = 0
            self.latest_training_stage = "A"
            self.route_loss_weight = route_loss_weight
            self.cldice_loss_weight = cldice_loss_weight
            self.boundary_loss_weight = boundary_loss_weight
            self.morphology_prior = None
            if morphology_prior_path:
                payload = json.loads(
                    Path(morphology_prior_path).read_text(encoding="utf-8")
                )
                self.morphology_prior = MorphologyPrior.from_dict(payload)
            expert_cfg = self.variant["experts"]
            self.morphology_router = (
                MorphologyRouter(feat_channels)
                if expert_cfg["enabled"]
                else None
            )
            self.line_expert = (
                LineExpert(out_channels)
                if expert_cfg["line"] or expert_cfg["homogeneous"]
                else nn.Identity()
            )
            self.area_expert = (
                AreaExpert(out_channels)
                if expert_cfg["area"]
                else nn.Identity()
            )
            self.homogeneous_area_expert = (
                LineExpert(out_channels)
                if expert_cfg["homogeneous"]
                else nn.Identity()
            )
            self.sparse_decoder = SparseExpertDecoder(
                tau if self.variant["router"]["dual_activation"] else 0.5
            )
            self.intervention_mode = intervention_mode
            self.capacity_adapter = (
                ParameterMatchedQueryAdapter(
                    out_channels,
                    capacity_adapter_rank,
                )
                if capacity_adapter_rank > 0
                else nn.Identity()
            )
            self.compute_adapter = (
                FlopsMatchedSharedAdapter(
                    out_channels,
                    compute_adapter_depth,
                )
                if compute_adapter_depth > 0
                else nn.Identity()
            )
            self._route_logits_by_layer = []
            self._shared_mask_feature_cache = None
            self._expert_feature_cache = None
            self.ignored_empty_morphology_targets = 0

        def _forward_head(
            self,
            decoder_out,
            mask_feature,
            attn_mask_target_size,
        ):
            decoder_out = self.transformer_decoder.post_norm(decoder_out)
            decoder_out = self.capacity_adapter(decoder_out)
            cls_pred = self.cls_embed(decoder_out)
            mask_embed = self.mask_embed(decoder_out)
            if self._shared_mask_feature_cache is None:
                self._shared_mask_feature_cache = self.compute_adapter(
                    mask_feature
                )
            mask_feature = self._shared_mask_feature_cache
            route_logits = (
                self.morphology_router(decoder_out)
                if self.morphology_router is not None
                else decoder_out.new_zeros((*decoder_out.shape[:-1], 2))
            )
            if self.variant["router"]["kind"] == "class_hard_diagnostic":
                predicted_class = cls_pred[..., : self.num_classes].argmax(
                    dim=-1
                )
                class_route = torch.full_like(route_logits, -20.0)
                class_route[..., 0] = torch.where(
                    predicted_class == 0,
                    torch.full_like(predicted_class, 20.0, dtype=route_logits.dtype),
                    torch.full_like(predicted_class, -20.0, dtype=route_logits.dtype),
                )
                class_route[..., 1] = -class_route[..., 0]
                route_logits = class_route
            self._route_logits_by_layer.append(route_logits)
            expert_cfg = self.variant["experts"]
            if not self.training:
                training_stage = "inference"
            elif self.training_update < self.stage_warmup_end:
                training_stage = "A"
            elif self.training_update < self.stage_sparse_start:
                training_stage = "B"
            else:
                training_stage = "C"
            self.latest_training_stage = training_stage
            experts_active = bool(
                expert_cfg["enabled"] and training_stage != "A"
            )
            if not experts_active:
                mask_pred = torch.einsum(
                    "bqc,bchw->bqhw",
                    mask_embed,
                    mask_feature,
                )
            else:
                self.sparse_decoder.sparse_training = training_stage == "C"
                if self._expert_feature_cache is None:
                    line_features = (
                        self.line_expert(mask_feature)
                        if expert_cfg["line"] or expert_cfg["homogeneous"]
                        else mask_feature
                    )
                    area_features = (
                        self.area_expert(mask_feature)
                        if expert_cfg["area"]
                        else (
                            self.homogeneous_area_expert(mask_feature)
                            if expert_cfg["homogeneous"]
                            else mask_feature
                        )
                    )
                    self._expert_feature_cache = (
                        line_features,
                        area_features,
                    )
                line_features, area_features = self._expert_feature_cache
                line_features, area_features, route_logits = apply_intervention(
                    line_features,
                    area_features,
                    route_logits,
                    self.intervention_mode,
                )
                mask_pred, diagnostics = self.sparse_decoder(
                    mask_embed,
                    line_features,
                    area_features,
                    route_logits,
                )
                self.latest_route_diagnostics = diagnostics
            attn_mask = functional.interpolate(
                mask_pred,
                attn_mask_target_size,
                mode="bilinear",
                align_corners=False,
            )
            attn_mask = (
                attn_mask.flatten(2)
                .unsqueeze(1)
                .repeat((1, self.num_heads, 1, 1))
                .flatten(0, 1)
            )
            return cls_pred, mask_pred, (attn_mask.sigmoid() < 0.5).detach()

        def forward(self, x, batch_data_samples):
            self._route_logits_by_layer = []
            self._shared_mask_feature_cache = None
            self._expert_feature_cache = None
            try:
                return super().forward(x, batch_data_samples)
            finally:
                self._shared_mask_feature_cache = None
                self._expert_feature_cache = None

        @staticmethod
        def _boundary(mask):
            eroded = -functional.max_pool2d(-mask, 3, 1, 1)
            return (mask - eroded).abs()

        @staticmethod
        def _soft_cldice_loss(probability, target):
            return soft_cldice_loss(probability, target)

        def _morphology_losses(
            self,
            cls_scores,
            mask_preds,
            batch_gt_instances,
            batch_img_metas,
        ):
            zero = mask_preds.sum() * 0.0
            needs_route = bool(
                self.morphology_prior is not None
                and self.variant["router"]["morphology_supervision"]
                and self.latest_training_stage != "A"
            )
            needs_structure = bool(
                self.variant["loss"]["structure"]
                and self.latest_training_stage != "A"
            )
            if not needs_route and not needs_structure:
                return zero, zero, zero
            route_logits = self._route_logits_by_layer[-1]
            route_losses = []
            topology_losses = []
            boundary_losses = []
            for image_index, (scores, masks, gt, metadata) in enumerate(
                zip(
                    cls_scores,
                    mask_preds,
                    batch_gt_instances,
                    batch_img_metas,
                )
            ):
                target_result = self._get_targets_single(
                    scores,
                    masks,
                    gt,
                    metadata,
                )
                pos_inds = target_result[4]
                sampling_result = target_result[6]
                if len(pos_inds) == 0:
                    continue
                assigned = sampling_result.pos_assigned_gt_inds
                gt_masks = gt.masks[assigned]
                if needs_route:
                    mask_arrays = [
                        mask.detach().cpu().numpy().astype(bool)
                        for mask in gt_masks
                    ]
                    features, observable_positions = (
                        extract_observable_morphology(mask_arrays)
                    )
                    ignored = len(mask_arrays) - len(observable_positions)
                    if ignored:
                        self.ignored_empty_morphology_targets += ignored
                        print(
                            "MORPHOROUTE_EMPTY_MORPHOLOGY_POLICY "
                            "action=exclude_route_supervision "
                            f"ignored_in_batch={ignored} "
                            "primary_and_structure_losses=preserved "
                            "fabricated_targets=0 "
                            "cumulative_ignored="
                            f"{self.ignored_empty_morphology_targets}",
                            flush=True,
                        )
                    if len(observable_positions):
                        positions = torch.as_tensor(
                            observable_positions,
                            device=pos_inds.device,
                            dtype=torch.long,
                        )
                        route_pos_inds = pos_inds[positions]
                        soft_targets = torch.as_tensor(
                            self.morphology_prior.soft_targets(features),
                            device=route_logits.device,
                            dtype=route_logits.dtype,
                        )
                        route_losses.append(
                            functional.kl_div(
                                functional.log_softmax(
                                    route_logits[
                                        image_index,
                                        route_pos_inds,
                                    ],
                                    dim=-1,
                                ),
                                soft_targets,
                                reduction="batchmean",
                            )
                        )
                if needs_structure:
                    selected_masks = functional.interpolate(
                        masks[pos_inds].unsqueeze(1),
                        size=gt_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1).sigmoid()
                    targets = gt_masks.float()
                    topology_losses.append(
                        self._soft_cldice_loss(selected_masks, targets)
                    )
                    boundary_losses.append(
                        functional.l1_loss(
                            self._boundary(selected_masks),
                            self._boundary(targets),
                        )
                    )
            return (
                (
                    torch.stack(route_losses).mean()
                    if route_losses
                    else zero
                ),
                (
                    torch.stack(topology_losses).mean()
                    if topology_losses
                    else zero
                ),
                (
                    torch.stack(boundary_losses).mean()
                    if boundary_losses
                    else zero
                ),
            )

        def loss_by_feat(
            self,
            all_cls_scores,
            all_mask_preds,
            batch_gt_instances,
            batch_img_metas,
        ):
            losses = super().loss_by_feat(
                all_cls_scores,
                all_mask_preds,
                batch_gt_instances,
                batch_img_metas,
            )
            route, topology, boundary = self._morphology_losses(
                all_cls_scores[-1],
                all_mask_preds[-1],
                batch_gt_instances,
                batch_img_metas,
            )
            losses["loss_route"] = route * self.route_loss_weight
            losses["loss_cldice"] = topology * self.cldice_loss_weight
            losses["loss_boundary"] = boundary * self.boundary_loss_weight
            return losses
