from __future__ import annotations

import json
import os
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import morphoroute.mmdet.models  # noqa: F401 - populate MMDetection registry
from mmdet.registry import MODELS
import mmdet.models.task_modules.assigners.hungarian_assigner as hungarian_assigner
from morphoroute.runtime.mmdet_adapter import OfficialMMDetBackend


ARTIFACT_DIR = Path(os.environ["EXP07_NUMERICAL_ARTIFACT_DIR"]).resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
STOP_ITER = int(os.environ.get("EXP07_NUMERICAL_PROBE_STOP_ITER", "60000"))
if STOP_ITER <= 40000:
    raise ValueError("EXP07_NUMERICAL_PROBE_STOP_ITER must exceed 40000")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _torch_stats(value: torch.Tensor) -> dict[str, object]:
    tensor = value.detach()
    numeric_floating = tensor.is_floating_point() or tensor.is_complex()
    finite = (
        torch.isfinite(tensor)
        if numeric_floating
        else torch.ones_like(tensor, dtype=torch.bool)
    )
    finite_values = tensor[finite]
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(tensor).sum().item()) if numeric_floating else 0,
        "posinf": (
            int(torch.isposinf(tensor).sum().item()) if numeric_floating else 0
        ),
        "neginf": (
            int(torch.isneginf(tensor).sum().item()) if numeric_floating else 0
        ),
        "finite_min": (
            float(finite_values.min().item()) if finite_values.numel() else None
        ),
        "finite_max": (
            float(finite_values.max().item()) if finite_values.numel() else None
        ),
        "finite_abs_max": (
            float(finite_values.abs().max().item())
            if finite_values.numel()
            else None
        ),
    }


def _numpy_stats(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(array).sum()),
        "posinf": int(np.isposinf(array).sum()),
        "neginf": int(np.isneginf(array).sum()),
        "finite_min": float(finite_values.min()) if finite_values.size else None,
        "finite_max": float(finite_values.max()) if finite_values.size else None,
        "finite_abs_max": (
            float(np.abs(finite_values).max()) if finite_values.size else None
        ),
    }


_original_assignment = hungarian_assigner.linear_sum_assignment


def _checked_assignment(cost_matrix, *args, **kwargs):
    if torch.is_tensor(cost_matrix):
        matrix = cost_matrix.detach().cpu().numpy()
    else:
        matrix = np.asarray(cost_matrix)
    if not np.isfinite(matrix).all():
        event = _stamp()
        matrix_file = ARTIFACT_DIR / f"invalid_cost_{event}.npy"
        report_file = ARTIFACT_DIR / f"invalid_cost_{event}.json"
        np.save(matrix_file, matrix)
        report = {
            "event": "invalid_hungarian_cost_matrix",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "matrix_file": str(matrix_file),
            "matrix": _numpy_stats(matrix),
        }
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("EXP07_HUNGARIAN_INVALID_COST=" + json.dumps(report, ensure_ascii=False))
        traceback.print_stack()
    return _original_assignment(cost_matrix, *args, **kwargs)


hungarian_assigner.linear_sum_assignment = _checked_assignment

head_class = MODELS.get("MorphoRouteMask2FormerHead")
if head_class is None:
    raise RuntimeError("MorphoRouteMask2FormerHead is not registered")

_original_loss = head_class._loss_by_feat_single


def _diagnostic_loss(
    self,
    cls_scores,
    mask_preds,
    batch_gt_instances,
    batch_img_metas,
):
    try:
        return _original_loss(
            self,
            cls_scores,
            mask_preds,
            batch_gt_instances,
            batch_img_metas,
        )
    except Exception as error:
        if "matrix contains invalid numeric entries" not in str(error):
            raise
        event = _stamp()
        report_file = ARTIFACT_DIR / f"loss_inputs_{event}.json"
        gt_report = []
        for index, instance in enumerate(batch_gt_instances):
            labels = getattr(instance, "labels", None)
            masks = getattr(instance, "masks", None)
            raw_masks = getattr(masks, "masks", masks)
            item = {"image_index": index}
            if torch.is_tensor(labels):
                item["labels"] = _torch_stats(labels)
            if torch.is_tensor(raw_masks):
                item["masks"] = _torch_stats(raw_masks)
            elif raw_masks is not None:
                item["masks"] = _numpy_stats(np.asarray(raw_masks))
            gt_report.append(item)
        report = {
            "event": "hungarian_failure_loss_inputs",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "training_update": int(getattr(self, "training_update", -1)),
            "training_stage": str(getattr(self, "latest_training_stage", "unknown")),
            "cls_scores": _torch_stats(cls_scores),
            "mask_preds": _torch_stats(mask_preds),
            "ground_truth": gt_report,
            "batch_img_metas": batch_img_metas,
        }
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print("EXP07_HUNGARIAN_LOSS_INPUTS=" + json.dumps(report, ensure_ascii=False, default=str))
        traceback.print_exc()
        raise


head_class._loss_by_feat_single = _diagnostic_loss

_original_configuration = OfficialMMDetBackend._configuration
_original_train = OfficialMMDetBackend.train


def _probe_configuration(resolved):
    configuration = _original_configuration(resolved)
    configured_max = int(configuration.train_cfg.max_iters)
    if STOP_ITER > configured_max:
        raise ValueError(
            f"probe stop iter {STOP_ITER} exceeds configured max {configured_max}"
        )
    configuration.train_cfg.max_iters = STOP_ITER
    configuration.dump(resolved.config_path)
    print(
        f"EXP07_NUMERICAL_PROBE_LIMIT max_iters={STOP_ITER} "
        f"original_max_iters={configured_max}",
        flush=True,
    )
    return configuration


def _probe_train(self, resolved, *, resume: bool):
    checkpoint = _original_train(self, resolved, resume=resume)
    print(
        f"EXP07_NUMERICAL_PROBE_REACHED_LIMIT "
        f"max_iters={STOP_ITER} checkpoint={checkpoint}",
        flush=True,
    )
    raise SystemExit(0)


OfficialMMDetBackend._configuration = staticmethod(_probe_configuration)
OfficialMMDetBackend.train = _probe_train

if len(sys.argv) < 3:
    raise SystemExit(
        "usage: exp07_hungarian_probe.py EFFICIENT_RUNNER ENTRY [arguments...]"
    )

efficient_runner = Path(sys.argv[1]).resolve()
entry = Path(sys.argv[2]).resolve()
sys.argv = [str(efficient_runner), str(entry), *sys.argv[3:]]
print(f"EXP07_NUMERICAL_PROBE_ACTIVE artifacts={ARTIFACT_DIR}", flush=True)
runpy.run_path(str(efficient_runner), run_name="__main__")
