from __future__ import annotations

import json

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, atomic_write_text
from morphoroute.data.coco_dataset import ensure_coco_dataset
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.data.morphology_table import ensure_d1_morphology_prior
from morphoroute.models.variants import VARIANT_IDS, build_variant_config, diff_keys
from morphoroute.training.variant_execution import run_variant_matrix

from .common import begin, dry_plan, finish, write_csv


PLANNED_CONTRASTS = (
    ("B0", "B1"),
    ("B0", "B2"),
    ("B0", "B3"),
    ("B4", "B6"),
    ("B5", "B6"),
    ("B6", "B7"),
    ("B6", "B8"),
    ("B6", "B9"),
    ("B9", "B10"),
    ("B9", "B11"),
)


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "按固定顺序生成 B0–B11 并保存配置差异",
            "逐变体、逐 seed 串行训练并保存 D3 逐图预测",
            "按预注册配对对比生成性能—资源消融表",
        ],
        variants=list(VARIANT_IDS),
        planned_contrasts=[list(pair) for pair in PLANNED_CONTRASTS],
    )
    if dry_run:
        return plan
    directory = begin(config)
    config_dir = directory / "configs"
    diff_dir = directory / "config_diffs"
    baseline = build_variant_config("B0")
    for variant in VARIANT_IDS:
        variant_config = build_variant_config(variant)
        atomic_write_json(config_dir / f"{variant}.json", variant_config)
        atomic_write_text(
            diff_dir / f"{variant}_vs_B0.txt",
            "\n".join(sorted(diff_keys(baseline, variant_config))) + "\n",
        )
    write_csv(
        directory / "ablation_pairwise.csv",
        [{"left": left, "right": right} for left, right in PLANNED_CONTRASTS],
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
    rows, _ = run_variant_matrix(
        config,
        materialized,
        morphology_prior_path=prior_path,
        variants=VARIANT_IDS,
    )
    write_csv(directory / "ablation_D3.csv", rows)
    pairwise_rows = []
    for left, right in PLANNED_CONTRASTS:
        left_rows = {
            int(row["seed"]): row
            for row in rows
            if row["variant"] == left
        }
        right_rows = {
            int(row["seed"]): row
            for row in rows
            if row["variant"] == right
        }
        paired_seeds = sorted(set(left_rows).intersection(right_rows))
        if not paired_seeds:
            raise RuntimeError(f"no paired ablation seeds: {left} vs {right}")
        left_ap_values = [
            float(left_rows[seed]["mask_ap"])
            for seed in paired_seeds
        ]
        right_ap_values = [
            float(right_rows[seed]["mask_ap"])
            for seed in paired_seeds
        ]
        left_values = [
            float(left_rows[seed]["mask_ap50"])
            for seed in paired_seeds
        ]
        right_values = [
            float(right_rows[seed]["mask_ap50"])
            for seed in paired_seeds
        ]
        pairwise_rows.append(
            {
                "left": left,
                "right": right,
                "paired_seeds": ",".join(str(seed) for seed in paired_seeds),
                "seed_count": len(paired_seeds),
                "evidence_tier": (
                    "confirmation" if len(paired_seeds) > 1 else "screening"
                ),
                "mean_difference_mask_ap": (
                    sum(right_ap_values) / len(right_ap_values)
                    - sum(left_ap_values) / len(left_ap_values)
                ),
                "mean_difference_mask_ap50": (
                    sum(right_values) / len(right_values)
                    - sum(left_values) / len(left_values)
                ),
            }
        )
    write_csv(directory / "ablation_pairwise.csv", pairwise_rows)
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
