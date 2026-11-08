from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from morphoroute.data.materialize import MaterializedDataset
from morphoroute.evaluation.pipeline import (
    EvaluationResult,
    evaluate_bundle,
    write_prediction_bundle,
)
from morphoroute.evaluation.reliability_export import (
    reliability_rows_from_bundle,
)
from morphoroute.runtime.mmdet_predictions import predict_mmdet_bundle


def run_frozen_external_models(
    checkpoint_index_path: str | Path,
    materialized: MaterializedDataset,
    *,
    output_dir: str | Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    index = json.loads(Path(checkpoint_index_path).read_text(encoding="utf-8"))
    models = index.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("frozen checkpoint index requires candidate and baseline")
    directory = Path(output_dir)
    prediction_dir = directory / "D4_predictions"
    metric_rows = []
    reliability_rows = []
    domain_rows = []
    for record in models:
        name = str(record["model"])
        checkpoint = Path(str(record["checkpoint"]))
        config_path = Path(str(record["config"]))
        if not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"frozen model artifact is missing for {name}"
            )
        bundle = predict_mmdet_bundle(
            config_path,
            checkpoint,
            materialized,
            split="D4",
        )
        write_prediction_bundle(
            bundle,
            prediction_dir / f"{name}_predictions_D4.json",
        )
        result = evaluate_bundle(bundle)
        available = [
            float(value)
            for value in result.per_class_ap.values()
            if value is not None
        ]
        metric_rows.append(
            {
                "model": name,
                "macro_mask_ap": result.macro_mask_ap,
                "worst_class_ap": min(available) if available else 0.0,
                "crack_cldice": (
                    result.crack_cldice
                    if result.crack_cldice is not None
                    else 0.0
                ),
                "crack_break_rate": (
                    result.crack_break_rate
                    if result.crack_break_rate is not None
                    else 0.0
                ),
                "crack_component_error": (
                    result.crack_component_error
                    if result.crack_component_error is not None
                    else 0.0
                ),
                "area_boundary_f1": (
                    result.area_boundary_f1
                    if result.area_boundary_f1 is not None
                    else 0.0
                ),
                "area_hd95": (
                    result.area_hd95
                    if result.area_hd95 is not None
                    else 0.0
                ),
                "area_relative_error": (
                    result.area_relative_error
                    if result.area_relative_error is not None
                    else 0.0
                ),
                "per_class_ap50_json": json.dumps(
                    result.per_class_ap50,
                    ensure_ascii=False,
                ),
            }
        )
        reliability_rows.extend(
            reliability_rows_from_bundle(
                bundle,
                model=name,
                seed=int(record.get("seed", 0)),
            )
        )
        domains = sorted(
            {
                image.domain or image.source or "unknown"
                for image in bundle.images
            }
        )
        for domain in domains:
            subset = replace(
                bundle,
                images=tuple(
                    image
                    for image in bundle.images
                    if (image.domain or image.source or "unknown") == domain
                ),
            )
            domain_result = evaluate_bundle(subset)
            domain_rows.append(
                {
                    "model": name,
                    "domain": domain,
                    "image_count": domain_result.image_count,
                    "macro_mask_ap": domain_result.macro_mask_ap,
                    "crack_cldice": domain_result.crack_cldice,
                    "crack_break_rate": domain_result.crack_break_rate,
                    "crack_component_error": (
                        domain_result.crack_component_error
                    ),
                    "area_boundary_f1": domain_result.area_boundary_f1,
                    "area_hd95": domain_result.area_hd95,
                    "area_relative_error": (
                        domain_result.area_relative_error
                    ),
                }
            )
    return metric_rows, reliability_rows, domain_rows


def evaluate_domains(
    bundle_path: str | Path,
) -> list[dict[str, object]]:
    """Reserved schema helper for externally loaded prediction bundles."""

    payload = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    return [
        {
            "model": payload.get("experiment_id", ""),
            "domain": "all",
            "image_count": len(payload.get("images", [])),
        }
    ]
