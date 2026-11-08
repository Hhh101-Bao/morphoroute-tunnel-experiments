from __future__ import annotations

from .optional_torch import (
    TORCH_AVAILABLE,
    functional,
    nn,
    require_torch,
    torch,
)


if TORCH_AVAILABLE:

    class DirectionalSnakeConv(nn.Module):
        def __init__(self, channels: int, max_offset: float = 2.0) -> None:
            super().__init__()
            self.max_offset = max_offset
            self.offset = nn.Conv2d(channels, 2, kernel_size=3, padding=1)
            self.horizontal = nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=channels,
            )
            self.vertical = nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                groups=channels,
            )
            self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)

        def forward(self, features):
            batch, _, height, width = features.shape
            offsets = torch.tanh(self.offset(features)) * self.max_offset
            y_coordinates = torch.linspace(
                -1.0, 1.0, height, device=features.device, dtype=features.dtype
            )
            x_coordinates = torch.linspace(
                -1.0, 1.0, width, device=features.device, dtype=features.dtype
            )
            grid_y, grid_x = torch.meshgrid(
                y_coordinates, x_coordinates, indexing="ij"
            )
            base_grid = torch.stack([grid_x, grid_y], dim=-1)
            base_grid = base_grid.unsqueeze(0).expand(batch, -1, -1, -1)
            normalized_x = offsets[:, 0] * (2.0 / max(width - 1, 1))
            normalized_y = offsets[:, 1] * (2.0 / max(height - 1, 1))
            offset_grid = torch.stack([normalized_x, normalized_y], dim=-1)
            sampled = functional.grid_sample(
                features,
                base_grid + offset_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            return self.fuse(
                torch.cat([self.horizontal(sampled), self.vertical(sampled)], dim=1)
            )


    class LineExpert(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                DirectionalSnakeConv(channels),
                nn.GroupNorm(1, channels),
                nn.GELU(),
                DirectionalSnakeConv(channels),
            )

        def forward(self, features):
            return features + self.network(features)


    class AreaExpert(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.context = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
            self.boundary_gate = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)

        def forward(self, features):
            fused = self.fuse(
                torch.cat([self.local(features), self.context(features)], dim=1)
            )
            return features + fused * self.boundary_gate(features)


    class SparseExpertDecoder(nn.Module):
        def __init__(self, threshold: float) -> None:
            super().__init__()
            if not 0.5 <= threshold <= 1.0:
                raise ValueError("threshold must lie in [0.5, 1.0]")
            self.register_buffer(
                "_threshold_tensor",
                torch.tensor(float(threshold), dtype=torch.float32),
                persistent=True,
            )
            self.sparse_training = False

        @property
        def threshold(self) -> float:
            return float(self._threshold_tensor.item())

        @threshold.setter
        def threshold(self, value: float) -> None:
            selected = float(value)
            if not 0.5 <= selected <= 1.0:
                raise ValueError("threshold must lie in [0.5, 1.0]")
            self._threshold_tensor.fill_(selected)

        def forward(
            self,
            queries,
            line_features,
            area_features,
            route_logits,
        ):
            probabilities = torch.softmax(route_logits, dim=-1)
            line_masks = torch.einsum("bqd,bdhw->bqhw", queries, line_features)
            area_masks = torch.einsum("bqd,bdhw->bqhw", queries, area_features)
            both = torch.stack([line_masks, area_masks], dim=-1)
            if self.training and not self.sparse_training:
                masks = (both * probabilities[..., None, None, :]).sum(dim=-1)
                active = torch.full(
                    probabilities.shape[:-1],
                    2,
                    device=probabilities.device,
                    dtype=torch.long,
                )
            else:
                confidence, top = probabilities.max(dim=-1)
                top_masks = torch.gather(
                    both,
                    dim=-1,
                    index=top[..., None, None, None].expand(
                        -1, -1, both.shape[2], both.shape[3], 1
                    ),
                ).squeeze(-1)
                dual_masks = (both * probabilities[..., None, None, :]).sum(dim=-1)
                dual = confidence < self.threshold
                masks = torch.where(dual[..., None, None], dual_masks, top_masks)
                active = torch.where(
                    dual,
                    torch.full_like(top, 2),
                    torch.ones_like(top),
                )
            return masks, {
                "route_probabilities": probabilities,
                "active_experts": active,
                "route_entropy": -(
                    probabilities * torch.log(probabilities.clamp_min(1e-8))
                ).sum(dim=-1),
            }

else:

    class DirectionalSnakeConv:
        def __init__(self, *args, **kwargs) -> None:
            require_torch()

    class LineExpert(DirectionalSnakeConv):
        pass

    class AreaExpert(DirectionalSnakeConv):
        pass

    class SparseExpertDecoder(DirectionalSnakeConv):
        pass
