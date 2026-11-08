from __future__ import annotations

from .optional_torch import TORCH_AVAILABLE, require_torch, torch


INTERVENTIONS = (
    "normal",
    "disable_line",
    "disable_area",
    "swap_outputs",
    "force_line",
    "force_area",
)


def apply_intervention(line_output, area_output, route_logits, mode: str):
    if not TORCH_AVAILABLE:
        require_torch()
    if mode not in INTERVENTIONS:
        raise ValueError(f"unknown expert intervention: {mode}")
    if mode == "normal":
        return line_output, area_output, route_logits
    if mode == "disable_line":
        return torch.zeros_like(line_output), area_output, route_logits
    if mode == "disable_area":
        return line_output, torch.zeros_like(area_output), route_logits
    if mode == "swap_outputs":
        return area_output, line_output, route_logits
    forced = torch.full_like(route_logits, -20.0)
    forced[..., 0 if mode == "force_line" else 1] = 20.0
    return line_output, area_output, forced
