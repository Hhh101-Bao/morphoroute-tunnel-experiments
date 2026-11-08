from __future__ import annotations

import shutil

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json
from morphoroute.data.coco_dataset import ensure_coco_dataset
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.evaluation.pipeline import write_prediction_bundle
from morphoroute.runtime.mmdet_adapter import train_validate_predict_mmdet
from morphoroute.runtime.resources import torch_resource_snapshot

from .common import begin, dry_plan, finish


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "生成同骨干 Mask R-CNN 五类 COCO 配置",
            "D1 训练、D2 有限阈值选择、D3 冻结评价",
            "记录参数、FLOPs、显存和 batch=1 时延",
        ],
        backend="mmdetection",
        comparison_scope="strong_two_stage_baseline",
    )
    if dry_run:
        return plan
    directory = begin(config)
    records = load_manifest(
        config.data_root / "metadata" / "split_manifest.csv",
        config.data_root,
    )
    class_mapping = load_class_mapping(
        config.data_root / "metadata" / "class_mapping.json"
    )
    materialized = ensure_coco_dataset(
        records,
        class_mapping,
        config.output_root / "derived",
        tuple(split for split in config.allowed_splits if split != "D4"),
    )
    result = train_validate_predict_mmdet(
        config,
        materialized,
        model_kind="mask_rcnn",
    )
    atomic_write_json(
        directory / "training_plan.json",
        result.resolved_config.config,
    )
    write_prediction_bundle(
        result.predictions,
        directory / "mask_rcnn_D3_predictions.json",
    )
    shutil.copy2(
        result.evaluation_outputs.metrics_csv,
        directory / "metrics.csv",
    )
    resources = torch_resource_snapshot()
    resources.update(
        {
            "framework": "mmdetection",
            "model_kind": "mask_rcnn",
            "best_checkpoint": str(result.best_checkpoint),
            "dataset_fingerprint": materialized.fingerprint.sha256,
        }
    )
    atomic_write_json(directory / "resources.json", resources)
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
