from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


VARIANT_IDS = tuple(f"B{index}" for index in range(12))


_BASE = {
    "model": {
        "shared_decoder": True,
        "decoder_width_scale": 1.0,
        "decoder_depth_scale": 1.0,
    },
    "loss": {"structure": False},
    "experts": {
        "enabled": False,
        "count": 0,
        "line": False,
        "area": False,
        "homogeneous": False,
    },
    "router": {
        "kind": "none",
        "morphology_supervision": False,
        "dual_activation": False,
    },
    "optimizer": {"pcgrad": False},
    "training": {"separate_models": False, "hard_morphology_split": False},
    "matching": {"parameter_target": None, "flops_target": None},
}


def _set(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


_MODIFICATIONS = {
    "B0": {},
    "B1": {"loss.structure": True},
    "B2": {
        "model.decoder_width_scale": 1.35,
        "matching.parameter_target": "B9",
    },
    "B3": {
        "model.decoder_depth_scale": 1.30,
        "matching.flops_target": "B9",
    },
    "B4": {
        "experts.enabled": True,
        "experts.count": 2,
        "experts.homogeneous": True,
        "router.kind": "learned_generic",
    },
    "B5": {
        "experts.enabled": True,
        "experts.count": 2,
        "experts.homogeneous": True,
        "router.kind": "class_hard_diagnostic",
    },
    "B6": {
        "experts.enabled": True,
        "experts.count": 2,
        "experts.homogeneous": True,
        "router.kind": "morphology_soft",
        "router.morphology_supervision": True,
    },
    "B7": {
        "experts.enabled": True,
        "experts.count": 2,
        "experts.line": True,
        "router.kind": "morphology_soft",
        "router.morphology_supervision": True,
    },
    "B8": {
        "experts.enabled": True,
        "experts.count": 2,
        "experts.area": True,
        "router.kind": "morphology_soft",
        "router.morphology_supervision": True,
    },
    "B9": {
        "loss.structure": True,
        "experts.enabled": True,
        "experts.count": 2,
        "experts.line": True,
        "experts.area": True,
        "router.kind": "morphology_soft",
        "router.morphology_supervision": True,
        "router.dual_activation": True,
    },
    "B10": {
        "loss.structure": True,
        "experts.enabled": True,
        "experts.count": 2,
        "experts.line": True,
        "experts.area": True,
        "router.kind": "morphology_soft",
        "router.morphology_supervision": True,
        "router.dual_activation": False,
    },
    "B11": {
        "loss.structure": True,
        "experts.enabled": True,
        "experts.count": 2,
        "experts.line": True,
        "experts.area": True,
        "router.kind": "morphology_soft",
        "router.morphology_supervision": True,
        "router.dual_activation": True,
        "optimizer.pcgrad": True,
    },
}


def build_variant_config(variant_id: str) -> dict[str, Any]:
    if variant_id not in _MODIFICATIONS:
        raise ValueError(f"unknown ablation variant: {variant_id}")
    config = deepcopy(_BASE)
    for key, value in _MODIFICATIONS[variant_id].items():
        _set(config, key, value)
    config["variant_id"] = variant_id
    return config


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(_flatten(item, dotted))
        else:
            flattened[dotted] = item
    return flattened


def diff_keys(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    ignored = {"variant_id"}
    return {
        key
        for key in set(left_flat).union(right_flat)
        if key not in ignored and left_flat.get(key) != right_flat.get(key)
    }
