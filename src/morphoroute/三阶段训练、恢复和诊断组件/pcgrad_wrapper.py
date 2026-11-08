from __future__ import annotations

import torch

from morphoroute.models.pcgrad import project_conflicting_pair


class AmpPCGradWrapper:
    """Projects conflicting task gradients after AMP unscale and before clipping."""

    def __init__(self, optimizer, *, scaler=None, max_norm: float | None = None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.max_norm = max_norm

    def step(self, first_loss, second_loss) -> None:
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        scale = float(self.scaler.get_scale()) if self.scaler is not None else 1.0
        first = torch.autograd.grad(
            first_loss * scale,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        second = torch.autograd.grad(
            second_loss * scale,
            parameters,
            allow_unused=True,
        )
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, left, right in zip(parameters, first, second):
            if left is None and right is None:
                continue
            if left is None:
                gradient = right
            elif right is None:
                gradient = left
            else:
                projected_left, projected_right = project_conflicting_pair(
                    left / scale,
                    right / scale,
                )
                gradient = (projected_left + projected_right) / 2
                parameter.grad = gradient
                continue
            parameter.grad = gradient / scale
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
