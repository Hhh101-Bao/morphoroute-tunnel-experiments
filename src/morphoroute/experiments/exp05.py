from __future__ import annotations

import shutil

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, atomic_write_text
from morphoroute.data.coco_dataset import ensure_coco_dataset
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.evaluation.pipeline import write_prediction_bundle
from morphoroute.runtime.mmdet_adapter import train_validate_predict_mmdet

from .common import begin, dry_plan, finish


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "生成共享 Mask2Former B0 配置",
            "D1 训练、D2 复合指标选模、D3 冻结预测",
            "保存共享参数名称供 EXP-08A 同子集梯度诊断",
        ],
        backend="mmdetection",
        comparison_scope="innovation_shared_baseline_B0",
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
        model_kind="mask2former",
    )
    atomic_write_json(
        directory / "training_plan.json",
        result.resolved_config.config,
    )
    atomic_write_text(
        directory / "shared_parameter_list.txt",
        "backbone\npixel_decoder\nquery_decoder.shared_layers\n",
    )
    write_prediction_bundle(
        result.predictions,
        directory / "mask2former_D3_predictions.json",
    )
    shutil.copy2(
        result.evaluation_outputs.metrics_csv,
        directory / "metrics.csv",
    )
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
