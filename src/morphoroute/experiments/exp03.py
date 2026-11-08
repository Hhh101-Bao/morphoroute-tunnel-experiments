from __future__ import annotations

import shutil

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.data.yolo_dataset import ensure_yolo_dataset
from morphoroute.evaluation.pipeline import write_prediction_bundle
from morphoroute.runtime.ultralytics_adapter import train_validate_predict_yolo

from .common import begin, dry_plan, finish, write_csv


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "由 manifest 生成 D1/D2/D3 YOLO 派生数据",
            "D1 训练、D2 选 best、冻结后 D3 一次推理",
            "输出严格划分锚点并与 EXP-02 分表",
        ],
        backend="ultralytics",
        comparison_scope="rq1_yolo_strict_anchor",
        eligible_for_innovation_selection=False,
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
    materialized = ensure_yolo_dataset(
        records,
        class_mapping,
        config.output_root / "derived",
        tuple(split for split in config.allowed_splits if split != "D4"),
    )
    result = train_validate_predict_yolo(config, materialized)
    atomic_write_json(
        directory / "training_plan.json",
        result.training_plan.to_dict(),
    )
    write_prediction_bundle(
        result.predictions,
        directory / "predictions_D3.json",
    )
    shutil.copy2(
        result.evaluation_outputs.metrics_csv,
        directory / "metrics_strict_split.csv",
    )
    write_csv(
        directory / "split_correction_effect.csv",
        (
            {
                "metric": "mask_ap50",
                "strict_value": result.evaluation.mask_ap50,
                "original_value": None,
                "difference": None,
                "note": "EXP-02 value is joined during EXP-15 aggregation",
            },
        ),
    )
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
