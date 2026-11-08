from __future__ import annotations

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json
from morphoroute.data.coco_dataset import ensure_coco_dataset
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.data.morphology_table import ensure_d1_morphology_prior
from morphoroute.models.variants import build_variant_config
from morphoroute.training.variant_execution import run_variant_matrix
from morphoroute.training.resource_matching import build_resource_matches
from morphoroute.runtime.base import plan_common

from .common import begin, dry_plan, finish, write_csv


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "生成参数量匹配 B2 与 FLOPs 匹配 B3",
            "分别审计相对 B9 的匹配误差",
            "以相同训练预算输出性能—资源对照",
        ],
        backend="mmdetection",
        comparison_scope="innovation_resource_controls_B2_B3",
    )
    if dry_run:
        return plan
    directory = begin(config)
    tolerance = float(config.parameters.get("matching_tolerance", 0.05))
    common = plan_common(config)
    matches = build_resource_matches(
        channels=256,
        feature_height=max(1, common["input_size"] // 4),
        feature_width=max(1, common["input_size"] // 4),
        tolerance=tolerance,
    )
    b2_config = build_variant_config("B2")
    b2_config["selected_resource_match"] = matches.b2.to_dict()
    b3_config = build_variant_config("B3")
    b3_config["selected_resource_match"] = matches.b3.to_dict()
    atomic_write_json(directory / "B2_config.json", b2_config)
    atomic_write_json(directory / "B3_config.json", b3_config)
    atomic_write_json(directory / "matching_search.json", matches.to_dict())
    write_csv(
        directory / "matching_audit.csv",
        [
            {
                "variant": "B2",
                "target": "B9 parameters",
                "target_value": matches.target_parameters,
                "matched_value": matches.b2.parameters,
                "relative_error": matches.b2.parameter_relative_error,
                "tolerance": tolerance,
                "selected_control": f"adapter_rank={matches.b2.adapter_rank}",
            },
            {
                "variant": "B3",
                "target": "B9 FLOPs",
                "target_value": matches.target_flops,
                "matched_value": matches.b3.flops,
                "relative_error": matches.b3.flops_relative_error,
                "tolerance": tolerance,
                "selected_control": f"adapter_depth={matches.b3.adapter_depth}",
            },
        ],
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
        variants=("B2", "B3"),
        seeds=(config.seed,),
    )
    write_csv(directory / "metrics_resources.csv", rows)
    for variant in ("B2", "B3"):
        run_directory = directory / f"runs_{variant}"
        run_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            run_directory / "index.json",
            {
                "runs": [
                    row for row in rows if row["variant"] == variant
                ]
            },
        )
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
