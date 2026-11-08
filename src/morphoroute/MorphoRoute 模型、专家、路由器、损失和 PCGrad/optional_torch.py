from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    functional = None
    TORCH_AVAILABLE = False


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "This model component requires PyTorch in WSL ENV-MAIN. "
            "Run scripts/setup_wsl.sh before training."
        )
