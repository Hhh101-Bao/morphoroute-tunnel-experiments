from __future__ import annotations

from .optional_torch import TORCH_AVAILABLE, nn, require_torch


if TORCH_AVAILABLE:

    class MorphologyRouter(nn.Module):
        def __init__(self, query_dim: int, hidden_dim: int | None = None) -> None:
            super().__init__()
            hidden = hidden_dim or query_dim
            self.network = nn.Sequential(
                nn.LayerNorm(query_dim),
                nn.Linear(query_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2),
            )

        def forward(self, queries):
            return self.network(queries)

else:

    class MorphologyRouter:
        def __init__(self, *args, **kwargs) -> None:
            require_torch()
