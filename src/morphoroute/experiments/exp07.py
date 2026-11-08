from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json
from morphoroute.data.coco_dataset import ensure_coco_dataset
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.data.morphology_table import ensure_d1_morphology_prior
from morphoroute.models.morphology import FEATURE_NAMES, MorphologyPrior
from morphoroute.models.variants import build_variant_config
from morphoroute.training.variant_execution import run_variant_matrix

from .common import begin, dry_plan, finish, write_csv


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "仅用 D1 已匹配实例形态拟合两分量 GMM",
            "生成普通 MoE B4、类别路由 B5、形态路由 B6",
            "输出路由熵、专家负载和平均激活专家数",
        ],
        backend="mmdetection",
        comparison_scope="innovation_routing_controls_B4_B5_B6",
    )
    if dry_run:
        return plan
    directory = begin(config)
    for variant in ("B4", "B5", "B6"):
        atomic_write_json(
            directory / f"{variant}_config.json",
            build_variant_config(variant),
        )
    produced = [f"{variant}_config.json" for variant in ("B4", "B5", "B6")]
    records = load_manifest(
        config.data_root / "metadata" / "split_manifest.csv",
        config.data_root,
    )
    class_mapping = load_class_mapping(
        config.data_root / "metadata" / "class_mapping.json"
    )
    morphology_table, prior_path = ensure_d1_morphology_prior(
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
    rows, results = run_variant_matrix(
        config,
        materialized,
        morphology_prior_path=prior_path,
        variants=("B4", "B5", "B6"),
        seeds=config.parameters.get("seeds", (config.seed,)),
    )
    write_csv(directory / "metrics_resources.csv", rows)
    diagnostic_rows = []
    for (variant, seed), result in results.items():
        for split_name, bundle in (
            ("D2", result.validation_predictions),
            ("D3", result.predictions),
        ):
            for item in bundle.metadata.get("router_diagnostics", ()):
                diagnostic_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "split": split_name,
                        **dict(item),
                    }
                )
    if not diagnostic_rows:
        raise RuntimeError(
            "MorphoRoute inference produced no query-routing diagnostics"
        )
    write_csv(
        directory / "router_diagnostics.csv",
        diagnostic_rows,
    )
    runs_index = directory / "runs_B4_B6"
    runs_index.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        runs_index / "index.json",
        {"runs": rows, "morphology_table": str(morphology_table)},
    )
    produced.extend(
        (
            "gmm_D1.json",
            "runs_B4_B6",
            "router_diagnostics.csv",
            "metrics_resources.csv",
        )
    )
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
