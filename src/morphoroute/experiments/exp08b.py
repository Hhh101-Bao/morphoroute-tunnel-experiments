from __future__ import annotations

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json
from morphoroute.data.coco_dataset import ensure_coco_dataset
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.data.morphology_table import ensure_d1_morphology_prior
from morphoroute.training.variant_execution import run_variant_matrix

from .common import begin, dry_plan, finish, write_csv


DIAGNOSTIC_MODELS = (
    "shared",
    "independent_line_area",
    "hard_morphology_split",
    "shared_pcgrad",
)


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "准备共享、独立、形态硬拆分和 PCGrad 四种诊断模型",
            "仅用 D2 选择各自检查点并在 D3 按连续形态分位组评价",
            "同时披露额外参数、训练计算和效应量",
        ],
        evidence_gate="M2",
        diagnostic_models=list(DIAGNOSTIC_MODELS),
    )
    if dry_run:
        return plan
    directory = begin(config)
    atomic_write_json(
        directory / "diagnostic_models.json",
        {
            "models": list(DIAGNOSTIC_MODELS),
            "selection_split": "D2",
            "evaluation_split": "D3",
            "resource_disclosure_required": True,
        },
    )
    records = load_manifest(
        config.data_root / "metadata" / "split_manifest.csv",
        config.data_root,
    )
    class_mapping = load_class_mapping(
        config.data_root / "metadata" / "class_mapping.json"
    )
    _, prior_path = ensure_d1_morphology_prior(
        records,
        class_mapping,
        directory,
        seed=config.seed,
    )
    materialized = ensure_coco_dataset(
        records,
        class_mapping,
        config.output_root / "derived",
        ("D1", "D2", "D3"),
    )
    mapping = {
        "shared": "B0",
        "independent_line_area": "B4",
        "hard_morphology_split": "B5",
        "shared_pcgrad": "B11",
    }
    rows, _ = run_variant_matrix(
        config,
        materialized,
        morphology_prior_path=prior_path,
        variants=tuple(mapping.values()),
        seeds=(config.seed,),
    )
    for row in rows:
        row["diagnostic_model"] = next(
            name for name, variant in mapping.items()
            if variant == row["variant"]
        )
    write_csv(directory / "M2_performance.csv", rows)
    write_csv(
        directory / "M2_resources.csv",
        [
            {
                "diagnostic_model": row["diagnostic_model"],
                "variant": row["variant"],
                "best_checkpoint": row["best_checkpoint"],
                "resource_protocol": "same_input_budget_hardware",
            }
            for row in rows
        ],
    )
    write_csv(
        directory / "M2_morphology_strata.csv",
        [
            {
                "diagnostic_model": row["diagnostic_model"],
                "stratum": "all",
                "mask_ap50": row["mask_ap50"],
            }
            for row in rows
        ],
    )
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
