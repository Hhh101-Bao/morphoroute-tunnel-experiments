from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


POLICY_REVISION = "morphoroute_strict_fp32_v1"
VALID_VARIANTS = frozenset({"B4", "B5", "B6"})
VALID_MODES = frozenset({"probe", "formal"})


class NonFiniteTrainingError(RuntimeError):
    """Raised before a non-finite value can modify model state."""


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


def apply_strict_fp32_config(
    configuration,
    *,
    variant_id: str,
    mode: str,
):
    """Apply only the approved EXP-07 numerical-execution changes."""

    if variant_id not in VALID_VARIANTS:
        raise ValueError(
            f"strict FP32 is limited to {sorted(VALID_VARIANTS)}, "
            f"got {variant_id!r}"
        )
    if mode not in VALID_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}"
        )

    optim_wrapper = _get(configuration, "optim_wrapper")
    if optim_wrapper is None:
        raise KeyError("resolved configuration has no optim_wrapper")
    optimizer = _get(optim_wrapper, "optimizer")
    if optimizer is None:
        raise KeyError("resolved optim_wrapper has no optimizer")

    _set(optim_wrapper, "type", "StrictFiniteOptimWrapper")
    _pop(optim_wrapper, "loss_scale")
    _pop(optim_wrapper, "dtype")

    clip_grad = _get(optim_wrapper, "clip_grad")
    if clip_grad is None:
        clip_grad = {"max_norm": 0.1, "norm_type": 2}
        _set(optim_wrapper, "clip_grad", clip_grad)
    _set(clip_grad, "error_if_nonfinite", True)

    audit = {
        "policy_revision": POLICY_REVISION,
        "variant_id": variant_id,
        "mode": mode,
        "amp_enabled": False,
        "grad_scaler_enabled": False,
        "bf16_enabled": False,
        "cuda_matmul_tf32_enabled": False,
        "cudnn_tf32_enabled": False,
        "float32_matmul_precision": "highest",
        "optimizer_wrapper": "StrictFiniteOptimWrapper",
        "nonfinite_behavior": "fail_before_parameter_update",
        "hungarian_behavior": "save_evidence_and_raise_without_repair",
    }
    _set(configuration, "numerical_stability_audit", audit)
    return configuration


def configure_torch_strict_fp32(torch_module) -> dict[str, object]:
    """Disable reduced-precision FP32 paths before model construction."""

    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.set_float32_matmul_precision("highest")
    return {
        "cuda_matmul_allow_tf32": bool(
            torch_module.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "matmul_precision": torch_module.get_float32_matmul_precision(),
    }


def validate_finite_loss(loss) -> None:
    import torch

    if not torch.is_tensor(loss):
        raise TypeError(f"loss must be a tensor, got {type(loss).__name__}")
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise NonFiniteTrainingError("non-finite loss rejected before backward")


def validate_finite_gradients(parameters: Iterable[object]) -> None:
    import torch

    for index, parameter in enumerate(parameters):
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise NonFiniteTrainingError(
                "non-finite gradient rejected before clipping/update "
                f"at parameter index {index}"
            )


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


def guarded_linear_sum_assignment(
    original_assignment,
    artifact_dir: str | Path,
    cost_matrix,
    *args,
    **kwargs,
):
    """Reject an invalid Hungarian matrix without changing its values."""

    import torch

    if torch.is_tensor(cost_matrix):
        matrix = cost_matrix.detach().cpu().numpy()
    else:
        matrix = np.asarray(cost_matrix)
    finite = np.isfinite(matrix)
    if not bool(finite.all()):
        destination = Path(artifact_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        event = _stamp()
        matrix_path = destination / f"invalid_hungarian_cost_{event}.npy"
        report_path = destination / f"invalid_hungarian_cost_{event}.json"
        np.save(matrix_path, matrix)
        finite_values = matrix[finite]
        report = {
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
                float(finite_values.min()) if finite_values.size else None
            ),
            "finite_max": (
                float(finite_values.max()) if finite_values.size else None
            ),
        }
        _atomic_write_json(report_path, report)
        raise NonFiniteTrainingError(
            f"invalid Hungarian cost saved to {report_path}"
        )
    return original_assignment(cost_matrix, *args, **kwargs)


def _optimizer_parameters(optimizer) -> list[object]:
    parameters = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            identifier = id(parameter)
            if identifier not in seen:
                seen.add(identifier)
                parameters.append(parameter)
    return parameters


def sanitize_optimizer_checkpoint_state(
    state_dict: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """Copy optimizer state while removing AMP-only scaler metadata."""

    sanitized = dict(state_dict)
    removed = "loss_scaler" in sanitized
    sanitized.pop("loss_scaler", None)
    return sanitized, removed


def install_strict_finite_optim_wrapper(
    artifact_dir: str | Path,
):
    """Register the MMEngine optimizer wrapper used by formal EXP-07."""

    from mmengine.optim import OptimWrapper
    from mmengine.registry import OPTIM_WRAPPERS

    existing = OPTIM_WRAPPERS.get("StrictFiniteOptimWrapper")
    if existing is not None:
        return existing

    destination = Path(artifact_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    @OPTIM_WRAPPERS.register_module(force=True)
    class StrictFiniteOptimWrapper(OptimWrapper):
        def load_state_dict(self, state_dict):
            sanitized, removed = sanitize_optimizer_checkpoint_state(
                state_dict
            )
            if removed:
                print(
                    "EXP07_STRICT_FP32_DISCARDED_AMP_LOSS_SCALER",
                    flush=True,
                )
            return super().load_state_dict(sanitized)

        def backward(self, loss, **kwargs):
            try:
                validate_finite_loss(loss)
            except NonFiniteTrainingError as error:
                _atomic_write_json(
                    destination / f"nonfinite_loss_{_stamp()}.json",
                    {
                        "event": "nonfinite_loss",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "policy_revision": POLICY_REVISION,
                        "error": str(error),
                    },
                )
                raise
            return super().backward(loss, **kwargs)

        def step(self, **kwargs):
            try:
                validate_finite_gradients(
                    _optimizer_parameters(self.optimizer)
                )
            except NonFiniteTrainingError as error:
                _atomic_write_json(
                    destination / f"nonfinite_gradient_{_stamp()}.json",
                    {
                        "event": "nonfinite_gradient",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "policy_revision": POLICY_REVISION,
                        "error": str(error),
                    },
                )
                raise
            return super().step(**kwargs)

    return StrictFiniteOptimWrapper


def install_hungarian_finite_guard(artifact_dir: str | Path) -> None:
    """Install an idempotent guard at MMDetection's SciPy boundary."""

    import mmdet.models.task_modules.assigners.hungarian_assigner as module

    current = module.linear_sum_assignment
    if getattr(current, "_morphoroute_strict_fp32_guard", False):
        return

    def guarded(cost_matrix, *args, **kwargs):
        return guarded_linear_sum_assignment(
            current,
            artifact_dir,
            cost_matrix,
            *args,
            **kwargs,
        )

    guarded._morphoroute_strict_fp32_guard = True
    module.linear_sum_assignment = guarded
