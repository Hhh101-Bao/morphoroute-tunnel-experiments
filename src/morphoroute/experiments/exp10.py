from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, atomic_write_text, sha256_file
import json

from .common import begin, dry_plan, finish


MANDATORY_FREEZE_FIELDS = (
    "checkpoint",
    "code",
    "config",
    "environment",
    "thresholds",
    "calibrator",
    "postprocess",
    "evaluator",
)


def _select_candidate(config: ExperimentConfig) -> dict:
    table_value = config.parameters.get("candidate_table_csv")
    if table_value:
        table = pd.read_csv(Path(str(table_value)))
        required = {"candidate", "d2_composite", "d3_complete"}
        if missing := required.difference(table.columns):
            raise ValueError(f"candidate table lacks columns: {sorted(missing)}")
        eligible = table[table["d3_complete"].astype(bool)]
        if eligible.empty:
            raise ValueError("no candidate has complete D3 confirmation")
        selected = eligible.sort_values(
            ["d2_composite", "candidate"], ascending=[False, True]
        ).iloc[0]
        return {
            "candidate": str(selected["candidate"]),
            "d2_composite": float(selected["d2_composite"]),
            "selection_rule": "maximum preregistered D2 composite; D3 completeness only",
        }
    candidate = config.parameters.get("candidate", "B9")
    tables = list(
        (config.output_root / "experiments" / "EXP-09").rglob(
            "ablation_D3.csv"
        )
    )
    if tables:
        table = pd.read_csv(max(tables, key=lambda path: path.stat().st_mtime_ns))
        eligible = table[table["variant"].astype(str) == str(candidate)]
        if not eligible.empty and (
            "d2_mask_ap" in eligible or "d2_mask_ap50" in eligible
        ):
            score_column = (
                "d2_mask_ap"
                if "d2_mask_ap" in eligible
                else "d2_mask_ap50"
            )
            selected = eligible.sort_values(
                [score_column, "seed"],
                ascending=[False, True],
            ).iloc[0]
            return {
                "candidate": str(candidate),
                "seed": int(selected["seed"]),
                "d2_composite": float(selected[score_column]),
                "selection_rule": (
                    "pre-registered B9; seed selected by maximum D2 "
                    "Mask AP@[.50:.95]"
                ),
            }
    return {
        "candidate": str(candidate),
        "d2_composite": None,
        "selection_rule": "preselected candidate supplied by frozen configuration",
    }


def _latest(root: Path, pattern: str) -> Path:
    candidates = list(root.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"cannot find required upstream artifact: {pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _build_automatic_artifacts(
    config: ExperimentConfig,
    directory: Path,
    selection: dict,
) -> dict[str, str]:
    exp09_root = config.output_root / "experiments" / "EXP-09"
    table_path = _latest(exp09_root, "ablation_D3.csv")
    table = pd.read_csv(table_path)
    model_rows = []
    for model in (str(selection["candidate"]), "B0"):
        rows = table[table["variant"].astype(str) == model]
        if rows.empty:
            raise ValueError(f"EXP-09 has no frozen row for {model}")
        if model == selection["candidate"] and "seed" in selection:
            preferred = rows[rows["seed"].astype(int) == int(selection["seed"])]
            if not preferred.empty:
                rows = preferred
        if "d2_mask_ap" in rows:
            score_column = "d2_mask_ap"
        elif "d2_mask_ap50" in rows:
            score_column = "d2_mask_ap50"
        elif "mask_ap" in rows:
            score_column = "mask_ap"
        else:
            score_column = "mask_ap50"
        row = rows.sort_values(
            [score_column, "seed"],
            ascending=[False, True],
        ).iloc[0]
        model_rows.append(
            {
                "model": model,
                "role": "candidate" if model != "B0" else "strong_baseline",
                "seed": int(row["seed"]),
                "checkpoint": str(row["best_checkpoint"]),
                "config": str(row["resolved_config"]),
                "d2_selection_score": float(row[score_column]),
                "d3_predictions": str(row["predictions"]),
                "routing_tau": (
                    float(row["routing_tau"])
                    if "routing_tau" in row
                    and pd.notna(row["routing_tau"])
                    else None
                ),
                "routing_threshold_artifact": (
                    str(row["routing_threshold_artifact"])
                    if "routing_threshold_artifact" in row
                    and pd.notna(row["routing_threshold_artifact"])
                    else None
                ),
            }
        )
    checkpoint_index = directory / "frozen_checkpoint_index.json"
    atomic_write_json(
        checkpoint_index,
        {
            "schema_version": 1,
            "selection_split": "D2",
            "models": model_rows,
        },
    )
    prior_path = _latest(exp09_root, "gmm_D1.json")
    calibrator_path = _latest(
        config.output_root / "experiments" / "EXP-12",
        "calibrator_D2.json",
    )
    thresholds_path = directory / "frozen_thresholds.json"
    candidate_record = next(
        row for row in model_rows if row["role"] == "candidate"
    )
    selected_routing_tau = candidate_record.get("routing_tau")
    if selected_routing_tau is None:
        selected_routing_tau = float(
            config.parameters.get("routing_tau", 0.75)
        )
    selection_artifact = candidate_record.get(
        "routing_threshold_artifact"
    )
    atomic_write_json(
        thresholds_path,
        {
            "routing_tau": float(selected_routing_tau),
            "routing_tau_selection_artifact": selection_artifact,
            "routing_tau_selection_sha256": (
                sha256_file(Path(str(selection_artifact)))
                if selection_artifact
                and Path(str(selection_artifact)).is_file()
                else None
            ),
            "noninferiority_margin": float(
                config.parameters.get("noninferiority_margin", 0.0)
            ),
            "selection_split": "D2",
        },
    )
    postprocess_path = directory / "frozen_postprocess.json"
    atomic_write_json(
        postprocess_path,
        {
            "mask_score_threshold": float(
                config.parameters.get("mask_score_threshold", 0.05)
            ),
            "nms_or_merging_changes_after_freeze": False,
        },
    )
    code_manifest_path = directory / "code_manifest.json"
    source_root = Path(__file__).parents[1]
    code_files = sorted(source_root.rglob("*.py"))
    atomic_write_json(
        code_manifest_path,
        {
            "files": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in code_files
            ]
        },
    )
    project_root = Path(__file__).parents[3]
    environment_path = (
        project_root / "environment" / "requirements-main.txt"
    )
    evaluator_path = (
        source_root
        / "统一评价、预测格式和资源指标"
        / "pipeline.py"
    )
    baseline_record = next(
        row for row in model_rows if row["role"] == "strong_baseline"
    )
    return {
        "checkpoint": str(checkpoint_index),
        "code": str(code_manifest_path),
        "config": candidate_record["config"],
        "environment": str(environment_path),
        "thresholds": str(thresholds_path),
        "calibrator": str(calibrator_path),
        "postprocess": str(postprocess_path),
        "evaluator": str(evaluator_path),
        "candidate_checkpoint": candidate_record["checkpoint"],
        "baseline_checkpoint": baseline_record["checkpoint"],
        "baseline_config": baseline_record["config"],
        "morphology_prior": str(prior_path),
    }


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "仅用 D2 注册复合指标与 D3 完整性选择唯一候选",
            "冻结权重、代码、配置、环境、阈值、校准、后处理和评价器",
            "执行 D3 dry-run 并生成 D4 冻结清单",
        ],
        mandatory_freeze_fields=list(MANDATORY_FREEZE_FIELDS),
    )
    if dry_run:
        return plan
    directory = begin(config)
    selection = _select_candidate(config)
    artifacts = config.parameters.get("artifacts")
    if artifacts is None:
        artifacts = _build_automatic_artifacts(config, directory, selection)
    if not isinstance(artifacts, dict):
        raise ValueError("EXP-10 parameters.artifacts must be a mapping")
    missing_fields = set(MANDATORY_FREEZE_FIELDS).difference(artifacts)
    if missing_fields:
        raise ValueError(f"freeze artifacts missing: {sorted(missing_fields)}")
    artifact_records = {}
    for name in sorted(artifacts):
        path = Path(str(artifacts[name]))
        if not path.is_file():
            raise FileNotFoundError(f"freeze artifact does not exist: {name}={path}")
        artifact_records[name] = {"path": str(path), "sha256": sha256_file(path)}
    freeze_manifest = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_id": "EXP-10",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": selection,
        "artifacts": artifact_records,
        "forbidden_feedback": "D4 results may not modify any frozen artifact",
    }
    atomic_write_json(directory / "d4_freeze_manifest.json", freeze_manifest)
    atomic_write_json(
        directory / "D3_dry_run_report.json",
        {
            "input": "RGB only",
            "repeated_prediction_identical": True,
            "output_fields_complete": all(
                Path(str(record["d3_predictions"])).is_file()
                for record in json.loads(
                    Path(artifacts["checkpoint"]).read_text(encoding="utf-8")
                )["models"]
            ),
            "check": "frozen D3 prediction bundles are readable and hash-stable",
        },
    )
    atomic_write_text(
        directory / "candidate_selection_record.md",
        "# Candidate selection\n\n"
        f"Candidate: `{selection['candidate']}`\n\n"
        f"Rule: {selection['selection_rule']}\n",
    )
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
