from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json
from morphoroute.metrics.calibration import apply_temperature, fit_temperature
from morphoroute.metrics.reliability import (
    brier_score,
    expected_calibration_error,
    risk_coverage_curve,
)
from morphoroute.visualization.plots import plot_risk_coverage

from .common import begin, dry_plan, finish


def _parse_logits(values) -> np.ndarray:
    return np.asarray([json.loads(str(value)) for value in values], dtype=float)


def _stage_manifest(
    config: ExperimentConfig,
    stage: str,
    payload: dict,
) -> None:
    summary_dir = config.output_root / "experiments" / "EXP-12"
    summary_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        summary_dir / f"{stage}_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "EXP-12",
            "stage": stage,
            "status": "completed",
            **payload,
        },
    )


def _resolve_prediction_table(config: ExperimentConfig, stage: str) -> Path:
    parameter_name = (
        "predictions_csv"
        if stage == "prefreeze"
        else "external_predictions_csv"
    )
    explicit = config.parameters.get(parameter_name)
    if explicit:
        return Path(str(explicit))
    if stage == "prefreeze":
        candidates = list(
            (
                config.output_root
                / "experiments"
                / "EXP-09"
            ).rglob("reliability_logits_D2_D3.csv")
        )
    else:
        candidates = list(
            (
                config.output_root
                / "experiments"
                / "EXP-11"
            ).rglob("reliability_logits_D4.csv")
        )
    if not candidates:
        raise FileNotFoundError(
            f"EXP-12 {stage} cannot find {parameter_name}; "
            "the upstream prediction bundle must include per-instance logits"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _execute_prefreeze(config: ExperimentConfig, directory: Path, plan: dict):
    table = pd.read_csv(_resolve_prediction_table(config, "prefreeze"))
    required = {"split", "logits_json", "label", "high_risk"}
    if missing := required.difference(table.columns):
        raise ValueError(f"reliability table lacks columns: {sorted(missing)}")
    d2 = table[table["split"] == "D2"]
    d3 = table[table["split"] == "D3"].copy()
    if d2.empty or d3.empty:
        raise ValueError("temperature fitting requires D2 and validation requires D3")
    temperature = fit_temperature(
        _parse_logits(d2["logits_json"]),
        d2["label"].to_numpy(int),
    )
    calibrator_path = directory / "calibrator_D2.json"
    atomic_write_json(
        calibrator_path,
        {"method": "temperature_scaling", "temperature": temperature, "fit_split": "D2"},
    )
    probabilities = apply_temperature(_parse_logits(d3["logits_json"]), temperature)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == d3["label"].to_numpy(int)
    error = (~correct).astype(float)
    curve = risk_coverage_curve(confidence, error)
    d3["calibrated_confidence"] = confidence
    d3["correct"] = correct
    d3.to_csv(directory / "calibration_D3.csv", index=False)
    review_rows = []
    high_risk = d3["high_risk"].astype(bool).to_numpy()
    uncertainty_order = np.argsort(confidence)
    for budget in (0.10, 0.20, 0.30):
        count = max(1, int(np.ceil(len(d3) * budget)))
        selected = uncertainty_order[:count]
        captured = high_risk[selected].sum()
        total = high_risk.sum()
        review_rows.append(
            {
                "budget": budget,
                "review_count": count,
                "high_risk_captured": int(captured),
                "high_risk_total": int(total),
                "capture_rate": float(captured / total) if total else None,
            }
        )
    pd.DataFrame(review_rows).to_csv(directory / "selective_review.csv", index=False)
    atomic_write_json(
        directory / "calibration_summary.json",
        {
            "temperature": temperature,
            "ece": expected_calibration_error(confidence, correct.astype(float)),
            "brier": brier_score(confidence, correct.astype(float)),
        },
    )
    plot_risk_coverage(
        curve["coverage"],
        curve["risk"],
        directory / "risk_coverage",
    )
    _stage_manifest(
        config,
        "prefreeze",
        {
            "temperature": temperature,
            "calibrator_path": str(calibrator_path),
            "fit_split": "D2",
            "validation_split": "D3",
        },
    )
    return finish(
        config,
        directory,
        status="prefreeze_completed",
        produced_outputs=plan["planned_outputs"],
    )


def _execute_external(config: ExperimentConfig, directory: Path, plan: dict):
    prefreeze_manifest = (
        config.output_root
        / "experiments"
        / "EXP-12"
        / "prefreeze_manifest.json"
    )
    payload = json.loads(prefreeze_manifest.read_text(encoding="utf-8"))
    calibrator = json.loads(
        Path(payload["calibrator_path"]).read_text(encoding="utf-8")
    )
    temperature = float(calibrator["temperature"])
    table = pd.read_csv(_resolve_prediction_table(config, "external"))
    if set(table["split"].astype(str)) != {"D4"}:
        raise ValueError("external reliability stage accepts D4 rows only")
    required = {"logits_json", "label", "high_risk"}
    if missing := required.difference(table.columns):
        raise ValueError(f"external reliability table lacks: {sorted(missing)}")
    probabilities = apply_temperature(
        _parse_logits(table["logits_json"]),
        temperature,
    )
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == table["label"].to_numpy(int)
    external = table.copy()
    external["calibrated_confidence"] = confidence
    external["correct"] = correct
    external.to_csv(directory / "calibration_D4.csv", index=False)
    atomic_write_json(
        directory / "external_calibration_summary.json",
        {
            "temperature": temperature,
            "fit_calls": 0,
            "ece": expected_calibration_error(confidence, correct.astype(float)),
            "brier": brier_score(confidence, correct.astype(float)),
        },
    )
    _stage_manifest(
        config,
        "external",
        {
            "temperature": temperature,
            "fit_calls": 0,
            "source": str(_resolve_prediction_table(config, "external")),
        },
    )
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    stage = str(config.parameters.get("execution_stage", "external"))
    plan = dry_plan(
        config,
        [
            "仅从 D2 冻结 logits 拟合 temperature scaling",
            "在 D3 计算 ECE、Brier、risk–coverage 和固定预算捕获率",
            "D4 只应用冻结校准器，不重新拟合或更换评分",
        ],
        fit_split="D2",
        validation_split="D3",
        external_action="apply_frozen_only",
        execution_stage=stage,
    )
    if dry_run:
        return plan
    directory = begin(config)
    if stage == "prefreeze":
        return _execute_prefreeze(config, directory, plan)
    if stage == "external":
        return _execute_external(config, directory, plan)
    raise ValueError(f"unsupported EXP-12 stage: {stage}")
