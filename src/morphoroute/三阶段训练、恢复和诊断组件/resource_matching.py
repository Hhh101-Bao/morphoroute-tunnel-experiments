from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ResourceCandidate:
    adapter_rank: int = 0
    adapter_depth: int = 0
    parameters: int = 0
    flops: int = 0
    parameter_relative_error: float = 0.0
    flops_relative_error: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceMatches:
    target_parameters: int
    target_flops: int
    b2: ResourceCandidate
    b3: ResourceCandidate
    parameter_candidates: tuple[ResourceCandidate, ...]
    flops_candidates: tuple[ResourceCandidate, ...]
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_parameters": self.target_parameters,
            "target_flops": self.target_flops,
            "b2": self.b2.to_dict(),
            "b3": self.b3.to_dict(),
            "parameter_candidates": [
                candidate.to_dict() for candidate in self.parameter_candidates
            ],
            "flops_candidates": [
                candidate.to_dict() for candidate in self.flops_candidates
            ],
            "tolerance": self.tolerance,
            "resource_definition": (
                "Incremental decoder/head resources relative to B0; "
                "multiply-add is counted as two FLOPs."
            ),
        }


def _b9_resources(
    channels: int,
    feature_height: int,
    feature_width: int,
    num_queries: int,
) -> tuple[int, int]:
    # Exact trainable-parameter formulas for LineExpert, AreaExpert and router.
    parameters = 34 * channels * channels + 65 * channels + 6
    spatial = feature_height * feature_width
    expert_macs = spatial * (33 * channels * channels + 48 * channels)
    router_macs = num_queries * (channels * channels + 2 * channels)
    return parameters, 2 * (expert_macs + router_macs)


def _parameter_adapter_parameters(channels: int, rank: int) -> int:
    # LayerNorm + C->rank->C pointwise shared bottleneck.
    return rank * (2 * channels + 1) + 3 * channels


def _compute_block_resources(
    channels: int,
    feature_height: int,
    feature_width: int,
) -> tuple[int, int]:
    # LayerNorm-like GroupNorm + depthwise 3x3 + pointwise 1x1.
    parameters = channels * channels + 13 * channels
    macs = feature_height * feature_width * (
        channels * channels + 9 * channels
    )
    return parameters, 2 * macs


def build_resource_matches(
    *,
    channels: int = 256,
    feature_height: int = 256,
    feature_width: int = 256,
    num_queries: int = 100,
    tolerance: float = 0.05,
) -> ResourceMatches:
    """Select B2/B3 controls using only static, D2-independent resources."""

    if min(channels, feature_height, feature_width, num_queries) <= 0:
        raise ValueError("resource dimensions must be positive")
    if not 0 < tolerance <= 0.25:
        raise ValueError("matching tolerance must lie in (0, 0.25]")
    target_parameters, target_flops = _b9_resources(
        channels,
        feature_height,
        feature_width,
        num_queries,
    )

    parameter_candidates = []
    for rank in range(1, 40 * channels + 1):
        parameters = _parameter_adapter_parameters(channels, rank)
        relative_error = abs(parameters - target_parameters) / target_parameters
        parameter_candidates.append(
            ResourceCandidate(
                adapter_rank=rank,
                parameters=parameters,
                flops=2
                * feature_height
                * feature_width
                * rank
                * 2
                * channels,
                parameter_relative_error=relative_error,
            )
        )
    b2 = min(
        parameter_candidates,
        key=lambda candidate: (
            candidate.parameter_relative_error,
            candidate.adapter_rank,
        ),
    )

    block_parameters, block_flops = _compute_block_resources(
        channels,
        feature_height,
        feature_width,
    )
    flops_candidates = []
    for depth in range(1, 129):
        flops = depth * block_flops
        relative_error = abs(flops - target_flops) / target_flops
        flops_candidates.append(
            ResourceCandidate(
                adapter_depth=depth,
                parameters=depth * block_parameters,
                flops=flops,
                flops_relative_error=relative_error,
            )
        )
    b3 = min(
        flops_candidates,
        key=lambda candidate: (
            candidate.flops_relative_error,
            candidate.adapter_depth,
        ),
    )
    if b2.parameter_relative_error > tolerance:
        raise RuntimeError(
            "No registered B2 parameter match satisfies the tolerance: "
            f"{b2.parameter_relative_error:.4f} > {tolerance:.4f}"
        )
    if b3.flops_relative_error > tolerance:
        raise RuntimeError(
            "No registered B3 FLOPs match satisfies the tolerance: "
            f"{b3.flops_relative_error:.4f} > {tolerance:.4f}"
        )
    return ResourceMatches(
        target_parameters=target_parameters,
        target_flops=target_flops,
        b2=b2,
        b3=b3,
        parameter_candidates=tuple(parameter_candidates),
        flops_candidates=tuple(flops_candidates),
        tolerance=tolerance,
    )
