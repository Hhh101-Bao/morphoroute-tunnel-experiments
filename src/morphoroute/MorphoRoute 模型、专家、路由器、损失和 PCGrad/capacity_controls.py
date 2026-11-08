from __future__ import annotations

from .optional_torch import TORCH_AVAILABLE, nn, require_torch


if TORCH_AVAILABLE:

    class ParameterMatchedSharedAdapter(nn.Module):
        def __init__(self, channels: int, rank: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.GroupNorm(1, channels),
                nn.Conv2d(channels, rank, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(rank, channels, kernel_size=1),
            )

        def forward(self, features):
            return features + self.network(features)


    class ParameterMatchedQueryAdapter(nn.Module):
        def __init__(self, channels: int, rank: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, rank),
                nn.GELU(),
                nn.Linear(rank, channels),
            )

        def forward(self, queries):
            return queries + self.network(queries)


    class SharedComputeBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.GroupNorm(1, channels),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    groups=channels,
                ),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1),
            )

        def forward(self, features):
            return features + self.network(features)


    class FlopsMatchedSharedAdapter(nn.Module):
        def __init__(self, channels: int, depth: int) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                *(SharedComputeBlock(channels) for _ in range(depth))
            )

        def forward(self, features):
            return self.blocks(features)

else:

    class ParameterMatchedSharedAdapter:
        def __init__(self, *args, **kwargs) -> None:
            require_torch()

    class FlopsMatchedSharedAdapter(ParameterMatchedSharedAdapter):
        pass

    class ParameterMatchedQueryAdapter(ParameterMatchedSharedAdapter):
        pass
