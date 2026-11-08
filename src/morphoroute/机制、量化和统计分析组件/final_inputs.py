from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from morphoroute.core.io import atomic_write_json
from morphoroute.evaluation.io import read_prediction_bundle
from morphoroute.evaluation.pipeline import evaluate_bundle


def _latest(root: Path, pattern: str) -> Path:
    candidates = list(root.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"missing upstream final input: {pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def build_final_records(
    output_root: str | Path,
    destination: str | Path,
) -> Path:
    root = Path(output_root)
    paths = list(
        (root / "experiments" / "EXP-09").rglob("predictions_D3.json")
    )
    if not paths:
        raise FileNotFoundError("EXP-15 cannot find EXP-09 D3 prediction bundles")
    rows = []
    for path in sorted(paths):
        bundle = read_prediction_bundle(path)
        if not bundle.images or not bundle.images[0].targets:
            raise ValueError(
                f"prediction bundle lacks reproducible targets: {path}"
            )
        model = str(
            bundle.metadata.get("variant_id")
            or path.parent.parent.name
        )
        seed = int(bundle.metadata.get("seed", path.parent.name.split("_")[-1]))
        smoke = bool(bundle.metadata.get("smoke_test", False))
        for image in bundle.images:
            result = evaluate_bundle(replace(bundle, images=(image,)))
            metrics = {
                "mask_ap": result.mask_ap,
                "mask_ap50": result.mask_ap50,
                "crack_cldice": result.crack_cldice,
                "crack_break_rate": result.crack_break_rate,
                "crack_component_error": result.crack_component_error,
                "area_boundary_f1": result.area_boundary_f1,
                "area_hd95": result.area_hd95,
                "area_relative_error": result.area_relative_error,
            }
            for metric, value in metrics.items():
                if value is None:
                    continue
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "group_id": image.group_id,
                        "image_id": image.image_id,
                        "metric": metric,
                        "value": float(value),
                        "run_status": "completed",
                        "smoke_test": smoke,
                        "prediction_bundle": str(path),
                    }
                )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination_path, index=False)
    return destination_path


def build_mechanism_evidence(
    output_root: str | Path,
    destination: str | Path,
) -> Path:
    root = Path(output_root) / "experiments"
    evidence = {
        "M1": False,
        "M2": False,
        "M3": False,
        "M4": False,
        "rules": {},
    }
    m1 = pd.read_csv(_latest(root / "EXP-08A", "M1_summary.csv"))
    negative_rate = float(m1["negative_cosine_rate"].mean())
    evidence["M1"] = negative_rate >= 0.10
    evidence["rules"]["M1"] = {
        "negative_cosine_rate": negative_rate,
        "threshold": 0.10,
    }

    m2 = pd.read_csv(_latest(root / "EXP-08B", "M2_performance.csv"))
    scores = m2.groupby("variant")["mask_ap50"].mean()
    m2_effect = float(scores.get("B11", float("-inf")) - scores.get("B0", 0.0))
    evidence["M2"] = m2_effect > 0
    evidence["rules"]["M2"] = {"B11_minus_B0_mask_ap50": m2_effect}

    m3 = pd.read_csv(_latest(root / "EXP-08C", "M3_model_results.csv"))
    morphology = m3[m3["term"].isin(
        (
            "log_aspect_ratio",
            "rectangularity",
            "elongation",
            "compactness",
            "skeleton_area_ratio",
        )
    )]
    max_coefficient = (
        float(morphology["coefficient"].abs().max())
        if not morphology.empty
        else 0.0
    )
    evidence["M3"] = max_coefficient > 0
    evidence["rules"]["M3"] = {
        "max_absolute_adjusted_coefficient": max_coefficient,
        "interpretation": "association only",
    }

    m4 = pd.read_csv(_latest(root / "EXP-08D", "M4_paired_effects.csv"))
    crack = m4[
        (m4["intervention"] == "disable_line")
        & (m4["metric"] == "crack_cldice")
    ]
    area = m4[
        (m4["intervention"] == "disable_area")
        & (m4["metric"] == "area_boundary_f1")
    ]
    line_harm = float(crack["paired_effect_vs_normal"].mean()) if len(crack) else 0.0
    area_harm = float(area["paired_effect_vs_normal"].mean()) if len(area) else 0.0
    evidence["M4"] = line_harm < 0 and area_harm < 0
    evidence["rules"]["M4"] = {
        "disable_line_crack_effect": line_harm,
        "disable_area_boundary_effect": area_harm,
    }
    return atomic_write_json(destination, evidence)


def resolve_effectiveness_decision(output_root: str | Path) -> Path:
    return _latest(
        Path(output_root) / "experiments" / "EXP-11",
        "E_gate_decision.json",
    )
