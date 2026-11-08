from __future__ import annotations

from .optional_torch import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:

    def flattened_gradients(loss, parameters, *, retain_graph: bool):
        gradients = torch.autograd.grad(
            loss,
            list(parameters),
            retain_graph=retain_graph,
            allow_unused=True,
        )
        available = [
            gradient.detach().reshape(-1)
            for gradient in gradients
            if gradient is not None
        ]
        if not available:
            raise ValueError("selected parameter subset has no gradients")
        return torch.cat(available)


    def gradient_conflict_record(line_loss, area_loss, parameters):
        parameter_list = list(parameters)
        line = flattened_gradients(line_loss, parameter_list, retain_graph=True)
        area = flattened_gradients(area_loss, parameter_list, retain_graph=True)
        line_norm = torch.linalg.vector_norm(line)
        area_norm = torch.linalg.vector_norm(area)
        cosine = torch.dot(line, area) / (line_norm * area_norm + 1e-12)
        return {
            "gradient_cosine": float(cosine.item()),
            "negative_cosine": bool(cosine.item() < 0),
            "line_norm": float(line_norm.item()),
            "area_norm": float(area_norm.item()),
            "norm_ratio": float((line_norm / (area_norm + 1e-12)).item()),
        }

else:

    def flattened_gradients(*args, **kwargs):
        require_torch()

    def gradient_conflict_record(*args, **kwargs):
        require_torch()
