from __future__ import annotations

from .optional_torch import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:

    def project_conflicting_pair(first, second, epsilon: float = 1e-12):
        dot = torch.dot(first.reshape(-1), second.reshape(-1))
        if dot >= 0:
            return first, second
        first_projected = first - dot * second / (second.pow(2).sum() + epsilon)
        second_projected = second - dot * first / (first.pow(2).sum() + epsilon)
        return first_projected, second_projected


    class PCGrad:
        def __init__(self, optimizer) -> None:
            self.optimizer = optimizer

        def zero_grad(self, set_to_none: bool = True) -> None:
            self.optimizer.zero_grad(set_to_none=set_to_none)

        def step_from_losses(self, first_loss, second_loss) -> None:
            parameters = [
                parameter
                for group in self.optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            ]
            first_gradients = torch.autograd.grad(
                first_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            second_gradients = torch.autograd.grad(
                second_loss,
                parameters,
                allow_unused=True,
            )
            for parameter, first, second in zip(
                parameters, first_gradients, second_gradients
            ):
                if first is None and second is None:
                    parameter.grad = None
                    continue
                if first is None:
                    parameter.grad = second
                    continue
                if second is None:
                    parameter.grad = first
                    continue
                projected_first, projected_second = project_conflicting_pair(
                    first, second
                )
                parameter.grad = (projected_first + projected_second) / 2
            self.optimizer.step()

else:

    def project_conflicting_pair(*args, **kwargs):
        require_torch()

    class PCGrad:
        def __init__(self, *args, **kwargs) -> None:
            require_torch()
