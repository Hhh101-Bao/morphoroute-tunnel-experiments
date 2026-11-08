from __future__ import annotations

from pathlib import Path
import json
import os
import sys

import pandas as pd

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.guard import D4AuthorizationRequest, D4Guard
from morphoroute.core.io import atomic_write_json
from morphoroute.data.coco_dataset import ensure_coco_dataset
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.evaluation.external import run_frozen_external_models

from .common import begin, dry_plan, finish


def _effectiveness_decision(table: pd.DataFrame, config: ExperimentConfig) -> dict:
    required = {
        "model",
        "macro_mask_ap",
        "worst_class_ap",
        "crack_cldice",
        "area_boundary_f1",
    }
    if missing := required.difference(table.columns):
        raise ValueError(f"D4 metrics table lacks columns: {sorted(missing)}")
    method_name = str(config.parameters.get("candidate", "B9"))
    baseline_name = str(config.parameters.get("strong_baseline", "B0"))
    method_rows = table[table["model"] == method_name]
    baseline_rows = table[table["model"] == baseline_name]
    if len(method_rows) != 1 or len(baseline_rows) != 1:
        raise ValueError("D4 metrics require one candidate and one strong baseline row")
    method = method_rows.iloc[0]
    baseline = baseline_rows.iloc[0]
    noninferiority = float(config.parameters.get("noninferiority_margin", 0.0))
    primary_better = method["macro_mask_ap"] > baseline["macro_mask_ap"]
    worst_protected = (
        method["worst_class_ap"] - baseline["worst_class_ap"] >= -noninferiority
    )
    structure_differences = [
        method["crack_cldice"] - baseline["crack_cldice"],
        method["area_boundary_f1"] - baseline["area_boundary_f1"],
    ]
    structure_condition = (
        max(structure_differences) > 0
        and min(structure_differences) >= -noninferiority
    )
    return {
        "primary_better": bool(primary_better),
        "worst_class_protected": bool(worst_protected),
        "structure_condition": bool(structure_condition),
        "E_positive": bool(primary_better and worst_protected and structure_condition),
        "candidate": method_name,
        "strong_baseline": baseline_name,
        "noninferiority_margin": noninferiority,
    }


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "验证 EXP-10 冻结清单和全部实际文件哈希",
            "写入一次性开始标记后挂载 D4 RGB 执行冻结推理",
            "分别评价 D4 合并、TC 和 DACL，并按预注册规则判定 E",
        ],
        one_shot=True,
        authorized_split="D4",
    )
    if dry_run:
        return plan
    directory = begin(config)
    freeze_value = config.parameters.get("freeze_manifest")
    if freeze_value:
        freeze_manifest = Path(str(freeze_value))
    else:
        candidates = list(
            (config.output_root / "experiments" / "EXP-10").rglob(
                "d4_freeze_manifest.json"
            )
        )
        if not candidates:
            raise FileNotFoundError("EXP-11 cannot find EXP-10 freeze manifest")
        freeze_manifest = max(
            candidates,
            key=lambda path: path.stat().st_mtime_ns,
        )
    confirmation = str(
        config.parameters.get("confirmation_phrase")
        or os.environ.get("MORPHOROUTE_D4_CONFIRMATION", "")
    )
    if not confirmation and sys.stdin.isatty():
        confirmation = input(
            "D4 是一次性正式外部测试。请输入确认短语 "
            "'I UNDERSTAND D4 IS ONE-SHOT'："
        ).strip()
    authorization = D4Guard().authorize(
        D4AuthorizationRequest(
            experiment_id="EXP-11",
            freeze_manifest=freeze_manifest,
            output_root=directory,
            confirmation_phrase=confirmation,
        )
    )
    atomic_write_json(directory / "D4_authorization.json", authorization)
    metrics_value = config.parameters.get("combined_metrics_csv")
    automatic_domains = None
    if not metrics_value:
        freeze_payload = json.loads(
            freeze_manifest.read_text(encoding="utf-8")
        )
        checkpoint_index = Path(
            freeze_payload["artifacts"]["checkpoint"]["path"]
        )
        records = load_manifest(
            config.data_root / "metadata" / "split_manifest.csv",
            config.data_root,
        )
        class_mapping = load_class_mapping(
            config.data_root / "metadata" / "class_mapping.json"
        )
        materialized = ensure_coco_dataset(
            records,
            class_mapping,
            config.output_root / "derived",
            ("D4",),
        )
        try:
            metric_rows, reliability_rows, domain_rows = (
                run_frozen_external_models(
                    checkpoint_index,
                    materialized,
                    output_dir=directory,
                )
            )
        except Exception as error:
            feedback_exists = any(
                (directory / "D4_predictions").rglob("*.json")
            )
            atomic_write_json(
                directory / "one_shot_status.json",
                {
                    **authorization,
                    "status": (
                        "sealed_partial_feedback"
                        if feedback_exists
                        else "infrastructure_failed_no_feedback"
                    ),
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                },
            )
            raise
        metrics = pd.DataFrame(metric_rows)
        automatic_domains = pd.DataFrame(domain_rows)
        pd.DataFrame(reliability_rows).to_csv(
            directory / "reliability_logits_D4.csv",
            index=False,
        )
    else:
        metrics = pd.read_csv(Path(str(metrics_value)))
    decision = _effectiveness_decision(metrics, config)
    metrics.to_csv(directory / "D4_combined_metrics.csv", index=False)
    domain_value = config.parameters.get("domain_metrics_csv")
    if domain_value:
        domains = pd.read_csv(Path(str(domain_value)))
    elif automatic_domains is None:
        raise ValueError(
            "domain_metrics_csv is required when combined metrics are supplied"
        )
    else:
        domains = automatic_domains
    domains.to_csv(directory / "D4_domain_metrics.csv", index=False)
    atomic_write_json(directory / "E_gate_decision.json", decision)
    atomic_write_json(
        directory / "one_shot_status.json",
        {**authorization, "status": "completed_sealed"},
    )
    return finish(
        config,
        directory,
        status="completed_sealed",
        produced_outputs=plan["planned_outputs"],
    )
