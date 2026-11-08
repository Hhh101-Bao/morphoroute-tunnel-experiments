import os
from pathlib import Path

from .morphoroute_head import MorphoRouteMask2FormerHead
from .morphology_targets import MorphologyTargetEncoder
from .variant_factory import build_variant_model_cfg
from .pcgrad_runtime import split_pcgrad_losses
from .routing_threshold_hook import choose_routing_threshold
from .safe_batch_pad import MorphoRouteSafeBatchFixedSizePad
from .memory_efficient_fusion import (
    MorphoRouteMemoryEfficientMaskFormerFusionHead,
)
from . import budget_gate_hook as _budget_gate_hook
from morphoroute.runtime.pcgrad_fp32 import (
    install_pcgrad_fp32_optim_wrapper,
)


_pcgrad_artifact_dir = os.environ.get(
    "EXP09_B11_NUMERICAL_ARTIFACT_DIR",
    str(Path.cwd() / "outputs" / "numerical_artifacts" / "exp09_b11"),
)
install_pcgrad_fp32_optim_wrapper(_pcgrad_artifact_dir)

__all__ = [
    "MorphoRouteMask2FormerHead",
    "MorphologyTargetEncoder",
    "build_variant_model_cfg",
    "split_pcgrad_losses",
    "choose_routing_threshold",
    "MorphoRouteSafeBatchFixedSizePad",
    "MorphoRouteMemoryEfficientMaskFormerFusionHead",
]
