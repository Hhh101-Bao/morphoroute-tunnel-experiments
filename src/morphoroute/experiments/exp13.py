from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from morphoroute.analysis.quantification_pipeline import run_quantification
from morphoroute.core.config import ExperimentConfig
from morphoroute.metrics.quantification import measure_mask
from morphoroute.visualization.plots import plot_bland_altman

from .common import begin, dry_plan, finish


def require_physical_scale(scale_per_pixel):
    if scale_per_pixel is None or float(scale_per_pixel) <= 0:
        raise ValueError("positive scale metadata is required for physical units")
    return float(scale_per_pixel)


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "读取冻结量化清单、尺度来源和质量等级",
            "计算骨架弧长、骨架局部正交宽度和掩膜面积",
            "有尺度才输出物理单位，并报告误差、协议一致性和不可测率",
        ],
        absolute_units_require_scale=True,
    )
    if dry_run:
        return plan
    directory = begin(config)
    manifest_value = config.parameters.get("quantification_manifest_csv")
    if not manifest_value:
        prediction_candidates = list(
            (
                config.output_root / "experiments" / "EXP-09"
            ).rglob("B9/seed_*/predictions_D3.json")
        )
        if not prediction_candidates:
            prediction_candidates = list(
                (
                    config.output_root / "experiments" / "EXP-05"
                ).rglob("mask2former_D3_predictions.json")
            )
        if not prediction_candidates:
            raise FileNotFoundError(
                "EXP-13 cannot locate frozen D3 prediction bundles from "
                "EXP-09 B9 or EXP-05"
            )
        prediction_path = max(
            prediction_candidates,
            key=lambda path: path.stat().st_mtime_ns,
        )
        metadata_candidate = (
            config.data_root / "metadata" / "quantification_metadata.csv"
        )
        result = run_quantification(
            prediction_path,
            metadata_path=(
                metadata_candidate if metadata_candidate.is_file() else None
            ),
            output_dir=directory,
        )
        pd.DataFrame(
            [
                {
                    "instance_id": row["instance_id"],
                    "image_id": row["image_id"],
                    "class_name": row["class_name"],
                    "prediction_bundle": str(prediction_path),
                    "scale_per_pixel": row["scale_per_pixel"],
                }
                for row in result.measurements
            ]
        ).to_csv(directory / "quantification_manifest.csv", index=False)
        pd.DataFrame(
            columns=(
                "instance_id",
                "reference_length",
                "reference_width",
                "reference_area",
            )
        ).to_csv(directory / "measurements_reference.csv", index=False)
        pd.DataFrame(
            [
                {
                    "measurement": "physical_input_completeness",
                    "n": len(result.measurements),
                    "physical_complete": result.physical_complete,
                    "template_path": str(result.template_path),
                }
            ]
        ).to_csv(directory / "quantification_metrics.csv", index=False)
        return finish(
            config,
            directory,
            status="completed",
            produced_outputs=plan["planned_outputs"],
            notes=(
                f"physical_complete={result.physical_complete}",
                "No physical unit is emitted without positive scale metadata.",
            ),
        )
    manifest = pd.read_csv(Path(str(manifest_value)))
    required = {"instance_id", "class_name", "pred_mask_path", "scale_per_pixel"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"quantification manifest lacks: {sorted(missing)}")
    manifest.to_csv(directory / "quantification_manifest.csv", index=False)
    predictions = []
    for row in manifest.to_dict("records"):
        mask = np.asarray(Image.open(Path(str(row["pred_mask_path"])))) != 0
        raw_scale = row.get("scale_per_pixel")
        scale = None if pd.isna(raw_scale) else require_physical_scale(raw_scale)
        measurements = measure_mask(mask, scale_per_pixel=scale)
        predictions.append(
            {
                "instance_id": row["instance_id"],
                "class_name": row["class_name"],
                **measurements,
            }
        )
    prediction_table = pd.DataFrame(predictions)
    prediction_table.to_csv(directory / "measurements_pred.csv", index=False)
    reference_columns = [
        column for column in manifest.columns if column.startswith("reference_")
    ]
    reference_table = manifest[["instance_id", *reference_columns]]
    reference_table.to_csv(directory / "measurements_reference.csv", index=False)
    metric_rows = []
    for predicted_name, reference_name in (
        ("length_physical", "reference_length"),
        ("width_median_physical", "reference_width"),
        ("area_physical", "reference_area"),
    ):
        if reference_name not in manifest:
            continue
        merged = prediction_table[["instance_id", predicted_name]].merge(
            manifest[["instance_id", reference_name]], on="instance_id"
        ).dropna()
        if merged.empty:
            continue
        absolute_error = np.abs(
            merged[predicted_name].to_numpy(float)
            - merged[reference_name].to_numpy(float)
        )
        reference = merged[reference_name].to_numpy(float)
        metric_rows.append(
            {
                "measurement": predicted_name,
                "n": len(merged),
                "mae": float(absolute_error.mean()),
                "relative_error_mean": float(
                    np.mean(absolute_error / np.maximum(np.abs(reference), 1e-12))
                ),
            }
        )
        try:
            plot_bland_altman(
                reference,
                merged[predicted_name],
                directory / f"{predicted_name}_bland_altman",
            )
        except RuntimeError:
            pass
    pd.DataFrame(metric_rows).to_csv(directory / "quantification_metrics.csv", index=False)
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
        notes=(
            f"unmeasurable_rate={float((~prediction_table['measurable']).mean()):.6f}",
        ),
    )
