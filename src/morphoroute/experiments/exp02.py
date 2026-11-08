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
            "恢复原 YOLO 配置或标记为公开默认配置重建",
            "仅用原验证规则选择检查点",
            "结果单独写入 RQ1 复现表，不进入创新候选选择",
        ],
        backend="ultralytics",
        comparison_scope="rq1_yolo_only",
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
    evidence_level = config.parameters.get(
        "evidence_level", "public_default_reconstruction"
    )
    atomic_write_json(
        directory / "reproduction_provenance.json",
        {
            "evidence_level": evidence_level,
            "innovation_selection_eligible": False,
            "training_plan": result.training_plan.to_dict(),
            "dataset_fingerprint": materialized.fingerprint.sha256,
            "best_checkpoint": str(result.best_checkpoint),
        },
    )
    write_prediction_bundle(
        result.predictions,
        directory / "predictions_D3.json",
    )
    shutil.copy2(
        result.evaluation_outputs.metrics_csv,
        directory / "metrics_original_split.csv",
    )
    write_csv(
        directory / "error_audit_original.csv",
        (),
        ("image_id", "class_name", "error_type", "note"),
    )
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
