from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json
from morphoroute.data.coco_dataset import ensure_coco_dataset
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.data.morphology_table import ensure_d1_morphology_prior
from morphoroute.evaluation.pipeline import (
    evaluate_bundle,
    write_prediction_bundle,
)
from morphoroute.mmdet.config_builder import build_morphoroute_config
from morphoroute.models.interventions import INTERVENTIONS
from morphoroute.runtime.mmdet_adapter import OfficialMMDetBackend

from .common import begin, dry_plan, finish, write_csv


def _find_b9_run(config: ExperimentConfig) -> dict[str, object]:
    candidates = list(
        (config.output_root / "experiments" / "EXP-09").rglob(
            "ablation_D3.csv"
        )
    )
    if not candidates:
        raise FileNotFoundError("EXP-08D cannot find EXP-09 ablation table")
    table = pd.read_csv(max(candidates, key=lambda path: path.stat().st_mtime_ns))
    rows = table[table["variant"].astype(str) == "B9"]
    if rows.empty:
        raise ValueError("EXP-09 ablation table has no B9 row")
    same_seed = rows[rows["seed"].astype(int) == config.seed]
    selected = same_seed.iloc[0] if not same_seed.empty else rows.iloc[0]
    return dict(selected)


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "在冻结 B9 与预注册 D3 query 上运行全部专家干预",
            "对裂缝拓扑与面状边界执行成对方向特异比较",
            "同时报告交叉损伤、专家负载和输出相似度",
        ],
        evidence_gate="M4",
        interventions=list(INTERVENTIONS),
    )
    if dry_run:
        return plan
    directory = begin(config)
    atomic_write_json(
        directory / "M4_interventions.json",
        {
            "interventions": list(INTERVENTIONS),
            "paired": True,
            "selection_split": "D2",
            "execution_split": "D3",
            "case_selection": "preregistered morphology quantiles",
        },
    )
    selected = _find_b9_run(config)
    checkpoint = Path(str(selected["best_checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen B9 checkpoint is missing: {checkpoint}")
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
    backend = OfficialMMDetBackend()
    metric_rows = []
    load_rows = []
    prediction_directory = directory / "expert_response_figures"
    for mode in INTERVENTIONS:
        parameters = dict(config.parameters)
        parameters.update(
            {
                "intervention_mode": mode,
                "run_name": f"interventions/{mode}",
                "resume": False,
            }
        )
        mode_config = replace(config, parameters=parameters)
        resolved = build_morphoroute_config(
            mode_config,
            materialized,
            variant_id="B9",
            morphology_prior_path=prior_path,
            tau=float(config.parameters.get("routing_tau", 0.75)),
        )
        # EXP08D_MEMORY_EFFICIENT_INFERENCE_V1
        from morphoroute.mmdet.models.memory_efficient_fusion import (
            MorphoRouteMemoryEfficientMaskFormerFusionHead,
        )
        configuration = backend._configuration(resolved)
        fusion_before = str(configuration.model.panoptic_fusion_head.type)
        if fusion_before not in {
            "MaskFormerFusionHead",
            "mmdet.MaskFormerFusionHead",
            "MorphoRouteMemoryEfficientMaskFormerFusionHead",
        }:
            raise RuntimeError(f"unexpected fusion head: {fusion_before!r}")
        configuration.model.panoptic_fusion_head.type = (
            MorphoRouteMemoryEfficientMaskFormerFusionHead.__name__
        )
        configuration.dump(resolved.config_path)
        print(
            f"EXP08D_MEMORY_EFFICIENT_INFERENCE mode={mode} "
            f"fusion_head={configuration.model.panoptic_fusion_head.type}",
            flush=True,
        )
        bundle = backend.predict(
            checkpoint,
            resolved,
            materialized,
            split="D3",
            shard_root=prediction_directory / "prediction_shards_memory_efficient_v1" / mode / "D3",
        )
        write_prediction_bundle(
            bundle,
            prediction_directory / f"{mode}_predictions_D3.json",
        )
        evaluation = evaluate_bundle(bundle)
        metric_rows.append(
            {
                "intervention": mode,
                "mask_ap50": evaluation.mask_ap50,
                "crack_cldice": evaluation.crack_cldice,
                "area_boundary_f1": evaluation.area_boundary_f1,
            }
        )
        load_rows.append(
            {
                "intervention": mode,
                "expected_active_experts": (
                    1 if mode in {"force_line", "force_area"} else 2
                ),
                "same_checkpoint": str(checkpoint),
                "shared_computation_changed": False,
            }
        )
    normal = next(row for row in metric_rows if row["intervention"] == "normal")
    effects = []
    for row in metric_rows:
        for metric in ("mask_ap50", "crack_cldice", "area_boundary_f1"):
            effects.append(
                {
                    "intervention": row["intervention"],
                    "metric": metric,
                    "paired_effect_vs_normal": float(row[metric])
                    - float(normal[metric]),
                }
            )
    write_csv(directory / "M4_paired_effects.csv", effects)
    write_csv(directory / "expert_load.csv", load_rows)
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
