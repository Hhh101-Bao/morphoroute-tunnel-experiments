from __future__ import annotations

from .optional_torch import (
    TORCH_AVAILABLE,
    functional,
    nn,
    require_torch,
    torch,
)


if TORCH_AVAILABLE:

    class MorphoRouteLoss(nn.Module):
        def __init__(
            self,
            *,
            mask_weight: float = 1.0,
            topology_weight: float = 1.0,
            boundary_weight: float = 1.0,
            route_weight: float = 1.0,
            prior_weight: float = 0.05,
        ) -> None:
            super().__init__()
            self.mask_weight = mask_weight
            self.topology_weight = topology_weight
            self.boundary_weight = boundary_weight
            self.route_weight = route_weight
            self.prior_weight = prior_weight

        @staticmethod
        def _dice(probability, target):
            axes = tuple(range(2, probability.ndim))
            intersection = (probability * target).sum(dim=axes)
            denominator = probability.sum(dim=axes) + target.sum(dim=axes)
            return 1.0 - ((2 * intersection + 1.0) / (denominator + 1.0)).mean()

        @staticmethod
        def _boundary_map(mask):
            eroded = -functional.max_pool2d(
                -mask,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            return (mask - eroded).abs()

        def forward(
            self,
            mask_logits,
            target_masks,
            route_logits,
            route_targets,
            line_weights,
            area_weights,
        ):
            targets = target_masks.float()
            probability = torch.sigmoid(mask_logits)
            mask_bce = functional.binary_cross_entropy_with_logits(
                mask_logits, targets
            )
            mask_dice = self._dice(probability, targets)
            topology = self._dice(
                functional.max_pool2d(probability, 3, 1, 1),
                functional.max_pool2d(targets, 3, 1, 1),
            )
            boundary = functional.l1_loss(
                self._boundary_map(probability),
                self._boundary_map(targets),
            )
            route_log_probability = functional.log_softmax(route_logits, dim=-1)
            route = functional.kl_div(
                route_log_probability,
                route_targets,
                reduction="batchmean",
            )
            mean_route = torch.softmax(route_logits, dim=-1).mean(dim=(0, 1))
            mean_target = route_targets.mean(dim=(0, 1))
            prior = functional.kl_div(
                torch.log(mean_route.clamp_min(1e-8)),
                mean_target,
                reduction="sum",
            )
            weighted_topology = topology * line_weights.mean()
            weighted_boundary = boundary * area_weights.mean()
            total = (
                self.mask_weight * (mask_bce + mask_dice)
                + self.topology_weight * weighted_topology
                + self.boundary_weight * weighted_boundary
                + self.route_weight * route
                + self.prior_weight * prior
            )
            return {
                "loss_total": total,
                "loss_mask_bce": mask_bce,
                "loss_mask_dice": mask_dice,
                "loss_topology": weighted_topology,
                "loss_boundary": weighted_boundary,
                "loss_route": route,
                "loss_prior": prior,
            }

else:

    class MorphoRouteLoss:
        def __init__(self, *args, **kwargs) -> None:
            require_torch()
