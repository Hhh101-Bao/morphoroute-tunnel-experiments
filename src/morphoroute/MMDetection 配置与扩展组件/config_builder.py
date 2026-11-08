from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from morphoroute.core.config import ExperimentConfig
from morphoroute.data.materialize import MaterializedDataset
from morphoroute.runtime.base import plan_common
from morphoroute.runtime.acceleration import budget_gate_policy
from morphoroute.runtime.pcgrad_fp32 import apply_b11_true_fp32_config
from morphoroute.models.variants import build_variant_config
from morphoroute.training.resource_matching import (
    ResourceMatches,
    build_resource_matches,
)


CLASS_NAMES = (
    "crack",
    "water_leakage",
    "spalling",
    "efflorescence",
    "exposed_rebar",
)


def _mask2former_data_preprocessor(input_size: int) -> dict[str, Any]:
    return {
        "batch_augments": [
            {
                "type": "BatchFixedSizePad",
                "size": (input_size, input_size),
                "img_pad_value": 0,
                "pad_mask": True,
                "mask_pad_value": 0,
                "pad_seg": False,
            }
        ]
    }


@dataclass(frozen=True)
class MMDetResolvedConfig:
    model_kind: str
    base_config: str
    config: dict[str, Any]
    work_dir: Path
    config_path: Path


def _pipeline(input_size: int, *, training: bool) -> list[dict[str, Any]]:
    transforms: list[dict[str, Any]] = [
        {"type": "LoadImageFromFile", "backend_args": None},
        {"type": "LoadAnnotations", "with_bbox": True, "with_mask": True},
    ]
    if training:
        transforms.extend(
            [
                {
                    "type": "RandomResize",
                    "scale": (input_size, input_size),
                    "ratio_range": (0.8, 1.2),
                    "keep_ratio": True,
                },
                {"type": "RandomFlip", "prob": 0.5},
            ]
        )
    else:
        transforms.append(
            {
                "type": "Resize",
                "scale": (input_size, input_size),
                "keep_ratio": True,
            }
        )
    transforms.append(
        {
            "type": "PackDetInputs",
            "meta_keys": (
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
            ),
        }
    )
    return transforms


def _dataloader(
    annotation_file: Path,
    *,
    input_size: int,
    batch_size: int,
    training: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> dict[str, Any]:
    if num_workers < 0:
        raise ValueError("dataloader_workers must be non-negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    dataloader = {
        "_delete_": True,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": bool(pin_memory),
        "sampler": {
            "type": "DefaultSampler",
            "shuffle": training,
        },
        "batch_sampler": (
            {"type": "AspectRatioBatchSampler"} if training else None
        ),
        "dataset": {
            "type": "CocoDataset",
            "ann_file": str(annotation_file),
            "data_prefix": {"img": ""},
            "metainfo": {"classes": CLASS_NAMES},
            "filter_cfg": (
                {"filter_empty_gt": True, "min_size": 16}
                if training
                else {"filter_empty_gt": False}
            ),
            "pipeline": _pipeline(input_size, training=training),
            "test_mode": not training,
            "backend_args": None,
        },
    }
    if num_workers > 0:
        dataloader["prefetch_factor"] = prefetch_factor
    return dataloader


def _common_config(
    config: ExperimentConfig,
    dataset: MaterializedDataset,
    *,
    model: dict[str, Any],
    base_config: str,
    model_kind: str,
) -> MMDetResolvedConfig:
    common = plan_common(config)
    required_splits = {"D1", "D2", "D3"}
    if not required_splits.issubset(dataset.annotation_files):
        raise ValueError("MMDetection dataset requires D1, D2, and D3 COCO files")
    work_dir = (
        config.output_root
        / "experiments"
        / config.experiment_id
        / config.profile
        / f"seed_{config.seed}"
        / "runs"
    )
    run_name = config.parameters.get("run_name")
    if run_name:
        work_dir = work_dir / Path(str(run_name))
    max_updates = common["max_updates"]
    gradient_accumulation = common["gradient_accumulation"]
    dataloader_workers = int(config.parameters.get("dataloader_workers", 2))
    performance_mode = bool(config.parameters.get("performance_mode", False))
    pin_memory = bool(
        config.parameters.get("pin_memory", performance_mode)
    )
    prefetch_factor = int(config.parameters.get("prefetch_factor", 2))
    gate = budget_gate_policy(config.parameters)
    if gate is not None and gate.start_update > max_updates:
        raise ValueError("budget_gate.start_update cannot exceed max_updates")
    max_iterations = max_updates * gradient_accumulation
    default_validation_update_interval = 1 if max_updates < 100 else 5000
    validation_update_interval = int(
        config.parameters.get(
            "validation_update_interval",
            default_validation_update_interval,
        )
    )
    if validation_update_interval <= 0:
        raise ValueError("validation_update_interval must be positive")
    validation_update_interval = min(validation_update_interval, max_updates)
    validation_interval = (
        validation_update_interval * gradient_accumulation
    )
    model.setdefault("backbone", {})["with_cp"] = bool(
        config.parameters.get("gradient_checkpointing", True)
    )
    resolved: dict[str, Any] = {
        "model": model,
        "train_dataloader": _dataloader(
            dataset.annotation_files["D1"],
            input_size=common["input_size"],
            batch_size=common["physical_batch"],
            training=True,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        ),
        "val_dataloader": _dataloader(
            dataset.annotation_files["D2"],
            input_size=common["input_size"],
            batch_size=1,
            training=False,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        ),
        "test_dataloader": _dataloader(
            dataset.annotation_files["D3"],
            input_size=common["input_size"],
            batch_size=1,
            training=False,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        ),
        "train_cfg": {
            "_delete_": True,
            "type": "IterBasedTrainLoop",
            "max_iters": max_iterations,
            "val_interval": validation_interval,
        },
        "val_cfg": {"_delete_": True, "type": "ValLoop"},
        "test_cfg": {"_delete_": True, "type": "TestLoop"},
        "optim_wrapper": {
            "_delete_": True,
            "type": "AmpOptimWrapper" if common["amp"] else "OptimWrapper",
            "optimizer": {
                "type": "AdamW",
                "lr": float(config.parameters.get("learning_rate", 1e-4)),
                "weight_decay": 0.05,
            },
            "accumulative_counts": gradient_accumulation,
            "clip_grad": {"max_norm": 0.1, "norm_type": 2},
            "loss_scale": "dynamic",
        },
        "param_scheduler": [
            {
                "type": "LinearLR",
                "start_factor": 0.001,
                "by_epoch": False,
                "begin": 0,
                "end": min(1000, max_updates) * gradient_accumulation,
            },
            {
                "type": "MultiStepLR",
                "by_epoch": False,
                "begin": 0,
                "end": max_iterations,
                "milestones": [
                    max(
                        1,
                        int(max_updates * 0.8) * gradient_accumulation,
                    ),
                    max(
                        1,
                        int(max_updates * 0.9) * gradient_accumulation,
                    ),
                ],
                "gamma": 0.1,
            },
        ],
        "val_evaluator": {
            "_delete_": True,
            "type": "CocoMetric",
            "ann_file": str(dataset.annotation_files["D2"]),
            "metric": ["bbox", "segm"],
            "format_only": False,
            "backend_args": None,
        },
        "test_evaluator": {
            "_delete_": True,
            "type": "CocoMetric",
            "ann_file": str(dataset.annotation_files["D3"]),
            "metric": ["bbox", "segm"],
            "format_only": False,
            "backend_args": None,
        },
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook",
                "by_epoch": False,
                "interval": validation_interval,
                "save_best": "coco/segm_mAP",
                "rule": "greater",
                "max_keep_ckpts": 3,
            },
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "DetVisualizationHook"},
        },
        "visualizer": {
            "type": "DetLocalVisualizer",
            "vis_backends": [{"type": "LocalVisBackend"}],
            "name": "visualizer",
        },
        "randomness": {
            "seed": config.seed,
            "deterministic": not performance_mode,
        },
        "env_cfg": {
            "cudnn_benchmark": performance_mode,
            "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0},
            "dist_cfg": {"backend": "nccl"},
        },
        "log_processor": {"type": "LogProcessor", "window_size": 50, "by_epoch": False},
        "launcher": "none",
        "resume": bool(config.parameters.get("resume", True)),
        "work_dir": str(work_dir),
    }
    if gate is not None:
        resolved["custom_imports"] = {
            "imports": ["morphoroute.mmdet.models"],
            "allow_failed_imports": False,
        }
        resolved.setdefault("custom_hooks", []).append(
            {
                "type": "D2BudgetGateHook",
                "start_iteration": gate.start_update * gradient_accumulation,
                "update_divisor": gradient_accumulation,
                "metric": gate.metric,
                "patience": gate.patience,
                "min_delta": gate.min_delta,
                "output_filename": "budget_gate_D2.json",
            }
        )
    resolved["acceleration_audit"] = {
        "budget_gate_enabled": gate is not None,
        "release_budget_gate": bool(
            config.parameters.get("release_budget_gate", False)
        ),
        "selection_split": "D2" if gate is not None else None,
        "performance_mode": performance_mode,
        "dataloader_workers": dataloader_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": (
            prefetch_factor if dataloader_workers > 0 else None
        ),
    }
    return MMDetResolvedConfig(
        model_kind=model_kind,
        base_config=base_config,
        config=resolved,
        work_dir=work_dir,
        config_path=work_dir / "resolved_mmdet_config.py",
    )


def build_mask_rcnn_config(
    config: ExperimentConfig,
    dataset: MaterializedDataset,
) -> MMDetResolvedConfig:
    return _common_config(
        config,
        dataset,
        model={
            "roi_head": {
                "bbox_head": {"num_classes": len(CLASS_NAMES)},
                "mask_head": {"num_classes": len(CLASS_NAMES)},
            }
        },
        base_config="mmdet::mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py",
        model_kind="mask_rcnn",
    )


def build_mask2former_config(
    config: ExperimentConfig,
    dataset: MaterializedDataset,
) -> MMDetResolvedConfig:
    common = plan_common(config)
    return _common_config(
        config,
        dataset,
        model={
            "data_preprocessor": _mask2former_data_preprocessor(
                common["input_size"]
            ),
            "panoptic_head": {
                "num_things_classes": len(CLASS_NAMES),
                "num_stuff_classes": 0,
                "num_queries": int(
                    config.parameters.get("query_count", 100)
                ),
                "loss_cls": {
                    "class_weight": [1.0] * len(CLASS_NAMES) + [0.1]
                },
            },
            "panoptic_fusion_head": {
                "num_things_classes": len(CLASS_NAMES),
                "num_stuff_classes": 0,
            },
        },
        base_config=(
            "mmdet::mask2former/"
            "mask2former_r50_8xb2-lsj-50e_coco.py"
        ),
        model_kind="mask2former",
    )


def build_morphoroute_config(
    config: ExperimentConfig,
    dataset: MaterializedDataset,
    *,
    variant_id: str,
    morphology_prior_path: str | Path | None,
    tau: float = 0.75,
) -> MMDetResolvedConfig:
    common = plan_common(config)
    matches = build_resource_matches(
        channels=256,
        feature_height=max(1, common["input_size"] // 4),
        feature_width=max(1, common["input_size"] // 4),
        num_queries=int(config.parameters.get("query_count", 100)),
        tolerance=float(config.parameters.get("matching_tolerance", 0.05)),
    )
    match = matches if variant_id in {"B2", "B3"} else None
    max_updates = int(common["max_updates"])
    stage_warmup_end = int(
        config.parameters.get(
            "stage_warmup_end",
            min(1000, max(1, max_updates // 10)),
        )
    )
    stage_sparse_start = int(
        config.parameters.get(
            "stage_sparse_start",
            min(
                40000,
                max(stage_warmup_end + 1, int(max_updates * 0.8)),
            ),
        )
    )
    overrides = morphoroute_model_overrides(
        variant_id=variant_id,
        morphology_prior_path=morphology_prior_path,
        tau=tau,
        capacity_match=match,
        intervention_mode=str(
            config.parameters.get("intervention_mode", "normal")
        ),
        stage_warmup_end=stage_warmup_end,
        stage_sparse_start=stage_sparse_start,
        query_count=int(config.parameters.get("query_count", 100)),
    )
    overrides["model"]["data_preprocessor"] = (
        _mask2former_data_preprocessor(common["input_size"])
    )
    resolved = _common_config(
        config,
        dataset,
        model=overrides["model"],
        base_config=(
            "mmdet::mask2former/"
            "mask2former_r50_8xb2-lsj-50e_coco.py"
        ),
        model_kind=f"morphoroute_{variant_id}",
    )
    resolved.config["custom_imports"] = {
        "imports": ["morphoroute.mmdet.models"],
        "allow_failed_imports": False,
    }
    resolved.config["resource_match_audit"] = matches.to_dict()
    resolved.config["model"]["gradient_accumulation"] = int(
        common["gradient_accumulation"]
    )
    variant = build_variant_config(variant_id)
    dual_activation = bool(variant["router"]["dual_activation"])
    experts_enabled = bool(variant["experts"]["enabled"])
    structure_enabled = bool(variant["loss"]["structure"])
    if experts_enabled or structure_enabled:
        transition_update = (
            stage_sparse_start if experts_enabled else stage_warmup_end
        )
        resolved.config.setdefault("custom_hooks", []).append(
            {
                "type": "RoutingThresholdSelectionHook",
                "trigger_iteration": (
                    transition_update
                    * int(common["gradient_accumulation"])
                ),
                "candidates": list(
                    config.parameters.get(
                        "routing_threshold_candidates",
                        (0.50, 0.60, 0.70, 0.75, 0.80, 0.90),
                    )
                ),
                "metric": "coco/segm_mAP",
                "select_threshold": dual_activation,
                "output_filename": (
                    "routing_threshold_selection_D2.json"
                    if dual_activation
                    else (
                        "stage_C_transition.json"
                        if experts_enabled
                        else "stage_B_transition.json"
                    )
                ),
            }
        )
    if variant_id == "B9":
        max_updates = int(common["max_updates"])
        resolved.config["model"]["diagnostic_path"] = str(
            resolved.work_dir / "gradient_diagnostics.jsonl"
        )
        resolved.config["model"]["diagnostic_updates"] = sorted(
            {
                stage_warmup_end,
                max(
                    stage_warmup_end,
                    (stage_warmup_end + stage_sparse_start) // 2,
                ),
                max(0, max_updates - 1),
            }
        )
    if "optim_wrapper" in overrides:
        resolved.config["optim_wrapper"]["type"] = overrides[
            "optim_wrapper"
        ]["type"]
        if variant_id == "B11":
            apply_b11_true_fp32_config(
                resolved.config,
                variant_id=variant_id,
            )
    return resolved


def morphoroute_model_overrides(
    *,
    variant_id: str,
    morphology_prior_path: str | Path | None,
    tau: float,
    capacity_match: ResourceMatches | None,
    intervention_mode: str = "normal",
    stage_warmup_end: int = 1000,
    stage_sparse_start: int = 40000,
    query_count: int = 100,
) -> dict[str, Any]:
    variant = build_variant_config(variant_id)
    head: dict[str, Any] = {
        "type": "MorphoRouteMask2FormerHead",
        "num_things_classes": len(CLASS_NAMES),
        "num_stuff_classes": 0,
        "num_queries": int(query_count),
        "variant_id": variant_id,
        "tau": tau,
        "morphology_prior_path": (
            str(morphology_prior_path)
            if morphology_prior_path is not None
            else None
        ),
        "intervention_mode": intervention_mode,
        "stage_warmup_end": stage_warmup_end,
        "stage_sparse_start": stage_sparse_start,
        "loss_cls": {
            "class_weight": [1.0] * len(CLASS_NAMES) + [0.1]
        },
    }
    if capacity_match is not None and variant_id == "B2":
        head["capacity_adapter_rank"] = capacity_match.b2.adapter_rank
    if capacity_match is not None and variant_id == "B3":
        head["compute_adapter_depth"] = capacity_match.b3.adapter_depth
    result: dict[str, Any] = {
        "model": {
            "type": "MorphoRouteMask2Former",
            "panoptic_head": head,
            "panoptic_fusion_head": {
                "num_things_classes": len(CLASS_NAMES),
                "num_stuff_classes": 0,
            },
        }
    }
    if variant["optimizer"]["pcgrad"]:
        result["optim_wrapper"] = {"type": "PCGradFP32OptimWrapper"}
    return result
