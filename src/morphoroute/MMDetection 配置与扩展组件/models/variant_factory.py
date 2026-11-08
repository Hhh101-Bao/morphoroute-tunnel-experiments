from __future__ import annotations

from typing import Any

from morphoroute.models.variants import build_variant_config
from morphoroute.training.resource_matching import ResourceMatches


def build_variant_model_cfg(
    variant_id: str,
    *,
    in_channels: int,
    query_dim: int,
    tau: float = 0.75,
    num_classes: int = 5,
    resource_matches: ResourceMatches | None = None,
) -> dict[str, Any]:
    variant = build_variant_config(variant_id)
    result = {
        "type": "MorphoRouteMask2FormerHead",
        "in_channels": in_channels,
        "query_dim": query_dim,
        "variant_id": variant_id,
        "tau": tau,
        "num_classes": num_classes,
        "variant_contract": variant,
    }
    if resource_matches is not None and variant_id == "B2":
        result["capacity_adapter_rank"] = resource_matches.b2.adapter_rank
    if resource_matches is not None and variant_id == "B3":
        result["compute_adapter_depth"] = resource_matches.b3.adapter_depth
    return result
