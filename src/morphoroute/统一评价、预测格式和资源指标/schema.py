from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .mask_codec import EncodedMask, MaskLike


@dataclass(frozen=True)
class PredictionInstance:
    class_name: str
    score: float
    bbox: tuple[float, float, float, float]
    mask: MaskLike
    logits: tuple[float, ...] = ()
    query_id: int | None = None
    route_probabilities: tuple[float, ...] = ()
    active_experts: tuple[str, ...] = ()
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.mask, EncodedMask):
            binary = self.mask
        else:
            binary = np.asarray(self.mask, dtype=bool)
            if binary.ndim != 2:
                raise ValueError("instance mask must be two-dimensional")
        object.__setattr__(self, "mask", binary)
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("instance score must be in [0, 1]")


@dataclass(frozen=True)
class PredictionImage:
    image_id: str
    group_id: str
    width: int
    height: int
    predictions: tuple[PredictionInstance, ...]
    targets: tuple[PredictionInstance, ...]
    source: str = ""
    domain: str = ""
    image_path: str | None = None


@dataclass(frozen=True)
class PredictionBundle:
    experiment_id: str
    split: str
    class_names: tuple[str, ...]
    images: tuple[PredictionImage, ...]
    framework: str = ""
    checkpoint_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
