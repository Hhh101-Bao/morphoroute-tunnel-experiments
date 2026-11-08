from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


POLICY_REVISION = "morphoroute_exp09_b11_memory_bounded_postprocess_v4"
VALID_VARIANT = "B11"
_COMPATIBLE_WRAPPERS = frozenset(
    {"PCGradAmpOptimWrapper", "PCGradFP32OptimWrapper"}
)
_STRUCTURE_LOSS_TOKENS = ("route", "cldice", "boundary")


def _get(container, key, default=None):
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def _set(container, key, value) -> None:
    try:
        container[key] = value
    except TypeError:
        setattr(container, key, value)


def _pop(container, key) -> None:
    if hasattr(container, "pop"):
        container.pop(key, None)
    elif hasattr(container, key):
        delattr(container, key)


def apply_b11_true_fp32_config(configuration, *, variant_id: str):
    """Replace only B11's AMP PCGrad execution wrapper with true FP32.

    Training budget, optimizer hyperparameters, scheduler, accumulation,
    losses, data order, and model definition are deliberately untouched.
    """

    if variant_id != VALID_VARIANT:
        raise ValueError(
            f"EXP-09 PCGrad true FP32 is limited to B11, got {variant_id!r}"
        )
    optim_wrapper = _get(configuration, "optim_wrapper")
    if optim_wrapper is None:
        raise KeyError("resolved configuration has no optim_wrapper")
    wrapper_type = str(_get(optim_wrapper, "type", ""))
    if wrapper_type not in _COMPATIBLE_WRAPPERS:
        raise RuntimeError(
            "unexpected optimizer wrapper for B11 true FP32: "
            f"{wrapper_type!r}"
        )
    if _get(optim_wrapper, "optimizer") is None:
        raise KeyError("resolved optim_wrapper has no optimizer")

    _set(optim_wrapper, "type", "PCGradFP32OptimWrapper")
    _pop(optim_wrapper, "loss_scale")
    _pop(optim_wrapper, "dtype")

    clip_grad = _get(optim_wrapper, "clip_grad")
    if clip_grad is None:
        clip_grad = {"max_norm": 0.1, "norm_type": 2}
        _set(optim_wrapper, "clip_grad", clip_grad)
    _set(clip_grad, "error_if_nonfinite", True)

    model = _get(configuration, "model")
    data_preprocessor = _get(model, "data_preprocessor")
    batch_augments = _get(data_preprocessor, "batch_augments", ())
    padding_augment_count = 0
    for augment in batch_augments:
        augment_type = str(_get(augment, "type", ""))
        if augment_type == "BatchFixedSizePad":
            _set(augment, "type", "MorphoRouteSafeBatchFixedSizePad")
            padding_augment_count += 1
        elif augment_type == "MorphoRouteSafeBatchFixedSizePad":
            padding_augment_count += 1
    if padding_augment_count != 1:
        raise RuntimeError(
            "B11 true-FP32 recovery requires exactly one fixed-size "
            f"batch padding augment, found {padding_augment_count}"
        )

    panoptic_fusion_head = _get(model, "panoptic_fusion_head")
    if panoptic_fusion_head is None:
        raise KeyError("resolved B11 model has no panoptic_fusion_head")
    _set(
        panoptic_fusion_head,
        "type",
        "MorphoRouteMemoryEfficientMaskFormerFusionHead",
    )

    audit = {
        "policy_revision": POLICY_REVISION,
        "variant_id": variant_id,
        "amp_enabled": False,
        "grad_scaler_enabled": False,
        "bf16_enabled": False,
        "cuda_matmul_tf32_enabled": False,
        "cudnn_tf32_enabled": False,
        "float32_matmul_precision": "highest",
        "optimizer_wrapper": "PCGradFP32OptimWrapper",
        "pcgrad_enabled": True,
        "checkpoint_optimizer_state_preserved": True,
        "discarded_checkpoint_state": "loss_scaler_only",
        "nonfinite_behavior": "save_evidence_and_fail_before_update",
        "hungarian_behavior": "save_evidence_and_raise_without_repair",
        "batch_padding_policy": (
            "preserve_oversized_dimension_and_record_actual_shape"
        ),
        "empty_morphology_policy": (
            "exclude_route_supervision_without_fabricating_target"
        ),
        "instance_postprocess_policy": (
            "stream_queries_without_batched_fp32_mask_copies"
        ),
    }
    _set(configuration, "numerical_stability_audit", audit)
    return configuration


def sanitize_pcgrad_fp32_checkpoint_state(
    state_dict: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """Copy optimizer-wrapper state and remove only obsolete AMP metadata."""

    sanitized = dict(state_dict)
    removed = "loss_scaler" in sanitized
    sanitized.pop("loss_scaler", None)
    return sanitized, removed


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_failure(
    destination: Path,
    *,
    event: str,
    error: BaseException,
) -> Path:
    report = destination / f"{event}_{_stamp()}.json"
    _atomic_write_json(
        report,
        {
            "event": event,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_revision": POLICY_REVISION,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    return report


def _validate_gradient_values(gradients, *, label: str) -> None:
    import torch

    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise RuntimeError(
                f"non-finite {label} gradient at parameter index {index}"
            )


def _optimizer_parameters(optimizer) -> list[object]:
    parameters = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            identifier = id(parameter)
            if parameter.requires_grad and identifier not in seen:
                seen.add(identifier)
                parameters.append(parameter)
    return parameters


def _split_pcgrad_losses(losses: Mapping[str, object]):
    import torch

    primary = []
    structure = []
    for name, value in losses.items():
        if "loss" not in name:
            continue
        values = value if isinstance(value, (list, tuple)) else (value,)
        target = (
            structure
            if any(token in name.lower() for token in _STRUCTURE_LOSS_TOKENS)
            else primary
        )
        target.extend(item.mean() for item in values if torch.is_tensor(item))
    if not primary or not structure:
        return None
    structure_loss = sum(structure)
    if float(structure_loss.detach().abs().item()) == 0.0:
        return None
    return sum(primary), structure_loss


def install_pcgrad_fp32_optim_wrapper(artifact_dir: str | Path):
    """Register true-FP32 PCGrad with pre-update finite-value guards."""

    import torch
    from mmengine.optim import OptimWrapper
    from mmengine.registry import OPTIM_WRAPPERS

    from morphoroute.models.pcgrad import project_conflicting_pair
    from morphoroute.runtime.strict_fp32 import (
        validate_finite_gradients,
        validate_finite_loss,
    )

    existing = OPTIM_WRAPPERS.get("PCGradFP32OptimWrapper")
    if existing is not None:
        return existing

    destination = Path(artifact_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    @OPTIM_WRAPPERS.register_module(force=True)
    class PCGradFP32OptimWrapper(OptimWrapper):
        """Effective-batch PCGrad without autocast or GradScaler."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pcgrad_first = None
            self._pcgrad_second = None

        def load_state_dict(self, state_dict):
            sanitized, removed = sanitize_pcgrad_fp32_checkpoint_state(
                state_dict
            )
            if removed:
                print(
                    "EXP09_B11_TRUE_FP32_DISCARDED_AMP_LOSS_SCALER",
                    flush=True,
                )
            return super().load_state_dict(sanitized)

        def backward(self, loss, **kwargs):
            try:
                validate_finite_loss(loss)
            except Exception as error:
                report = _record_failure(
                    destination,
                    event="nonfinite_loss",
                    error=error,
                )
                print(f"EXP09_B11_NUMERICAL_ARTIFACT={report}", flush=True)
                raise
            return super().backward(loss, **kwargs)

        def step(self, **kwargs):
            try:
                validate_finite_gradients(
                    _optimizer_parameters(self.optimizer)
                )
            except Exception as error:
                report = _record_failure(
                    destination,
                    event="nonfinite_projected_gradient",
                    error=error,
                )
                print(f"EXP09_B11_NUMERICAL_ARTIFACT={report}", flush=True)
                raise
            return super().step(**kwargs)

        def _accumulate(self, first, second) -> None:
            if self._pcgrad_first is None:
                self._pcgrad_first = [
                    None if value is None else value.detach().clone()
                    for value in first
                ]
                self._pcgrad_second = [
                    None if value is None else value.detach().clone()
                    for value in second
                ]
                return
            for buffer, gradients in (
                (self._pcgrad_first, first),
                (self._pcgrad_second, second),
            ):
                for index, value in enumerate(gradients):
                    if value is None:
                        continue
                    if buffer[index] is None:
                        buffer[index] = value.detach().clone()
                    else:
                        buffer[index].add_(value.detach())

        def update_params_from_loss_dict(
            self,
            losses,
            step_kwargs=None,
            zero_kwargs=None,
        ) -> None:
            pair = _split_pcgrad_losses(losses)
            if pair is None:
                parsed = sum(
                    item.mean()
                    for name, value in losses.items()
                    if "loss" in name
                    for item in (
                        value if isinstance(value, (list, tuple)) else (value,)
                    )
                    if torch.is_tensor(item)
                )
                self.update_params(parsed, step_kwargs, zero_kwargs)
                return

            step_kwargs = step_kwargs or {}
            zero_kwargs = zero_kwargs or {}
            first_loss, second_loss = (
                self.scale_loss(loss) for loss in pair
            )
            parameters = _optimizer_parameters(self.optimizer)
            try:
                validate_finite_loss(first_loss)
                validate_finite_loss(second_loss)
                first = torch.autograd.grad(
                    first_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                second = torch.autograd.grad(
                    second_loss,
                    parameters,
                    allow_unused=True,
                )
                _validate_gradient_values(first, label="primary-task")
                _validate_gradient_values(second, label="structure-task")
                self._accumulate(first, second)
            except Exception as error:
                report = _record_failure(
                    destination,
                    event="nonfinite_pcgrad_input",
                    error=error,
                )
                print(f"EXP09_B11_NUMERICAL_ARTIFACT={report}", flush=True)
                raise

            self._inner_count += 1
            if not self.should_update():
                return

            try:
                for parameter, left, right in zip(
                    parameters,
                    self._pcgrad_first,
                    self._pcgrad_second,
                ):
                    if left is None and right is None:
                        continue
                    if left is None:
                        gradient = right
                    elif right is None:
                        gradient = left
                    else:
                        projected_left, projected_right = (
                            project_conflicting_pair(left, right)
                        )
                        gradient = (projected_left + projected_right) / 2
                    if not bool(torch.isfinite(gradient).all().item()):
                        raise RuntimeError(
                            "non-finite PCGrad projection rejected before update"
                        )
                    parameter.grad = gradient
                self.step(**step_kwargs)
            except Exception as error:
                report = _record_failure(
                    destination,
                    event="nonfinite_pcgrad_projection",
                    error=error,
                )
                print(f"EXP09_B11_NUMERICAL_ARTIFACT={report}", flush=True)
                raise
            finally:
                self._pcgrad_first = None
                self._pcgrad_second = None
            self.zero_grad(**zero_kwargs)

    return PCGradFP32OptimWrapper


def install_hungarian_finite_guard(artifact_dir: str | Path) -> None:
    """Reject invalid matching costs and preserve evidence without repair."""

    import numpy as np
    import torch
    import mmdet.models.task_modules.assigners.hungarian_assigner as module

    current = module.linear_sum_assignment
    if getattr(current, "_morphoroute_exp09_b11_fp32_guard", False):
        return

    destination = Path(artifact_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def guarded(cost_matrix, *args, **kwargs):
        matrix = (
            cost_matrix.detach().cpu().numpy()
            if torch.is_tensor(cost_matrix)
            else np.asarray(cost_matrix)
        )
        finite = np.isfinite(matrix)
        if not bool(finite.all()):
            event = _stamp()
            matrix_path = destination / f"invalid_hungarian_cost_{event}.npy"
            report_path = destination / f"invalid_hungarian_cost_{event}.json"
            np.save(matrix_path, matrix)
            finite_values = matrix[finite]
            _atomic_write_json(
                report_path,
                {
                    "event": "invalid_hungarian_cost_matrix",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "policy_revision": POLICY_REVISION,
                    "matrix_file": str(matrix_path),
                    "shape": list(matrix.shape),
                    "dtype": str(matrix.dtype),
                    "nan": int(np.isnan(matrix).sum()),
                    "posinf": int(np.isposinf(matrix).sum()),
                    "neginf": int(np.isneginf(matrix).sum()),
                    "finite_min": (
                        float(finite_values.min())
                        if finite_values.size
                        else None
                    ),
                    "finite_max": (
                        float(finite_values.max())
                        if finite_values.size
                        else None
                    ),
                },
            )
            raise RuntimeError(
                f"invalid Hungarian cost saved to {report_path}"
            )
        return current(cost_matrix, *args, **kwargs)

    guarded._morphoroute_exp09_b11_fp32_guard = True
    module.linear_sum_assignment = guarded
