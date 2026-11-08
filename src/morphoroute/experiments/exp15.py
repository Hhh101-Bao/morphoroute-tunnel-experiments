from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, atomic_write_text, sha256_file
from morphoroute.statistics.bootstrap import paired_group_bootstrap
from morphoroute.statistics.effects import standardized_paired_effect
from morphoroute.statistics.multiple_testing import holm_adjust
from morphoroute.visualization.plots import plot_effect_forest
from morphoroute.visualization.tables import save_result_bundle
from morphoroute.analysis.final_inputs import (
    build_final_records,
    build_mechanism_evidence,
    resolve_effectiveness_decision,
)
from morphoroute.analysis.final_statistics import aggregate_final_results

from .common import begin, dry_plan, finish


def claim_branch(mechanism_positive: bool, effectiveness_positive: bool) -> dict:
    if mechanism_positive and effectiveness_positive:
        return {
            "code": "M+/E+",
            "claim": "连续形态条件性干扰具有多源证据，且冻结外部测试支持方法有效性。",
        }
    if mechanism_positive and not effectiveness_positive:
        return {
            "code": "M+/E-",
            "claim": "机制证据通过，但冻结外部效能不足；结论限于机制与失败边界。",
        }
    if not mechanism_positive and effectiveness_positive:
        return {
            "code": "M-/E+",
            "claim": "保留冻结外部性能结论；形态路由仅解释为有效结构归纳偏置。",
        }
    return {
        "code": "M-/E-",
        "claim": "机制与外部效能均未通过；报告负结果、数据限制和后续验证需求。",
    }


def _mechanism_positive(payload: dict) -> bool:
    evidence = {
        key: bool(payload.get(key, False)) for key in ("M1", "M2", "M3", "M4")
    }
    return sum(evidence.values()) >= 2 and (evidence["M1"] or evidence["M2"])


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "保留全部 seed 和区段结果并汇总均值±标准差",
            "按区段/序列配对 Bootstrap，主比较执行 Holm 校正",
            "独立读取 M/E 决策并锁定四分支论文表述与复现清单",
        ],
        bootstrap_unit="tunnel_section_or_sequence",
        failed_seeds_retained=True,
        claim_branches=["M+/E+", "M+/E-", "M-/E+", "M-/E-"],
    )
    if dry_run:
        return plan
    directory = begin(config)
    records_value = config.parameters.get("final_records_csv")
    mechanism_value = config.parameters.get("mechanism_evidence_json")
    effectiveness_value = config.parameters.get("effectiveness_decision_json")
    if not records_value:
        records_value = build_final_records(
            config.output_root,
            directory / "automatic_final_records.csv",
        )
    if not mechanism_value:
        mechanism_value = build_mechanism_evidence(
            config.output_root,
            directory / "automatic_mechanism_evidence.json",
        )
    if not effectiveness_value:
        effectiveness_value = resolve_effectiveness_decision(
            config.output_root
        )
    if not records_value or not mechanism_value or not effectiveness_value:
        return finish(
            config,
            directory,
            status="planned",
            produced_outputs=(),
            notes=(
                "Provide final_records_csv, mechanism_evidence_json, "
                "and effectiveness_decision_json.",
            ),
        )
    records = pd.read_csv(Path(str(records_value)))
    required = {"model", "seed", "group_id", "metric", "value", "run_status"}
    if missing := required.difference(records.columns):
        raise ValueError(f"final records lack columns: {sorted(missing)}")
    if records["run_status"].isna().any():
        raise ValueError("failed and completed seeds must carry explicit run_status")
    if bool(config.parameters.get("smoke_test", False)):
        atomic_write_json(
            directory / "statistical_results.json",
            {
                "status": "synthetic_smoke_only",
                "formal_claims_allowed": False,
                "record_count": len(records),
            },
        )
        atomic_write_text(
            directory / "claim_branch.md",
            "# 合成 smoke 验证\n\n该结果仅验证代码链，不进入论文统计或结论。\n",
        )
        (directory / "final_tables").mkdir(exist_ok=True)
        (directory / "final_figures").mkdir(exist_ok=True)
        atomic_write_json(
            directory / "reproducibility_manifest.json",
            {"smoke_test": True, "formal_claims_allowed": False},
        )
        return finish(
            config,
            directory,
            status="completed_smoke_only",
            produced_outputs=plan["planned_outputs"],
        )
    aggregate_final_results(records.to_dict("records"))
    final_tables = directory / "final_tables"
    final_figures = directory / "final_figures"
    final_tables.mkdir()
    final_figures.mkdir()
    summary = (
        records.groupby(["model", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            seeds=("seed", "nunique"),
            groups=("group_id", "nunique"),
        )
    )
    save_result_bundle(summary.to_dict("records"), final_tables / "seed_summary")

    comparisons = config.parameters.get("primary_comparisons", [["B9", "B0"]])
    comparison_rows = []
    p_values = []
    for left, right in comparisons:
        for metric in sorted(records["metric"].unique()):
            left_rows = records[
                (records["model"] == left) & (records["metric"] == metric)
            ][["group_id", "value"]].rename(columns={"value": "left_value"})
            right_rows = records[
                (records["model"] == right) & (records["metric"] == metric)
            ][["group_id", "value"]].rename(columns={"value": "right_value"})
            paired = left_rows.merge(right_rows, on="group_id")
            if paired.empty:
                continue
            bootstrap = paired_group_bootstrap(
                paired,
                group_col="group_id",
                value_a="left_value",
                value_b="right_value",
                n_resamples=int(config.parameters.get("bootstrap_resamples", 10000)),
                seed=config.seed,
            )
            distribution = np.asarray(bootstrap["distribution"])
            p_value = float(
                min(
                    1.0,
                    2
                    * min(
                        np.mean(distribution <= 0),
                        np.mean(distribution >= 0),
                    ),
                )
            )
            p_values.append(p_value)
            comparison_rows.append(
                {
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "difference": bootstrap["point_estimate"],
                    "ci_low": bootstrap["ci_low"],
                    "ci_high": bootstrap["ci_high"],
                    "effect": standardized_paired_effect(
                        paired["left_value"].to_numpy(float),
                        paired["right_value"].to_numpy(float),
                    ),
                    "p_value": p_value,
                }
            )
    if comparison_rows:
        adjusted = holm_adjust(p_values)
        for row, value in zip(comparison_rows, adjusted):
            row["holm_adjusted_p"] = float(value)
    save_result_bundle(comparison_rows, final_tables / "paired_comparisons")
    if comparison_rows:
        try:
            plot_effect_forest(
                [f"{row['left']}–{row['right']} {row['metric']}" for row in comparison_rows],
                [row["difference"] for row in comparison_rows],
                [row["ci_low"] for row in comparison_rows],
                [row["ci_high"] for row in comparison_rows],
                final_figures / "paired_effects",
            )
        except RuntimeError:
            pass

    mechanism = json.loads(Path(str(mechanism_value)).read_text(encoding="utf-8"))
    effectiveness = json.loads(
        Path(str(effectiveness_value)).read_text(encoding="utf-8")
    )
    m_positive = _mechanism_positive(mechanism)
    e_positive = bool(effectiveness.get("E_positive", False))
    branch = claim_branch(m_positive, e_positive)
    statistical_results = {
        "mechanism_positive": m_positive,
        "effectiveness_positive": e_positive,
        "claim_branch": branch,
        "comparisons": comparison_rows,
        "failed_seeds_retained": True,
    }
    statistical_path = directory / "statistical_results.json"
    atomic_write_json(statistical_path, statistical_results)
    atomic_write_text(
        directory / "claim_branch.md",
        f"# {branch['code']}\n\n{branch['claim']}\n",
    )
    artifact_paths = [
        statistical_path,
        directory / "claim_branch.md",
        final_tables / "seed_summary.json",
        final_tables / "paired_comparisons.json",
    ]
    atomic_write_json(
        directory / "reproducibility_manifest.json",
        {
            "artifacts": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in artifact_paths
            ],
            "bootstrap_seed": config.seed,
            "bootstrap_unit": "group_id",
        },
    )
    return finish(
        config,
        directory,
        status="completed_locked",
        produced_outputs=plan["planned_outputs"],
    )
