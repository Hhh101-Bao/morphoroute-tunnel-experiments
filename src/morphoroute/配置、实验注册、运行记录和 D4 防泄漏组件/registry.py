from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    title: str
    stage: int
    prerequisites: tuple[str, ...]
    allowed_splits: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    metadata_splits: tuple[str, ...] = ("D1", "D2", "D3")


def _spec(
    experiment_id: str,
    title: str,
    stage: int,
    prerequisites: tuple[str, ...],
    allowed_splits: tuple[str, ...],
    expected_outputs: tuple[str, ...],
    metadata_splits: tuple[str, ...] = ("D1", "D2", "D3"),
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        title=title,
        stage=stage,
        prerequisites=prerequisites,
        allowed_splits=allowed_splits,
        expected_outputs=expected_outputs,
        metadata_splits=metadata_splits,
    )


EXPERIMENTS = {
    "EXP-00": _spec(
        "EXP-00",
        "预注册与实验冻结",
        0,
        (),
        (),
        ("preregistration.json", "freeze_registry.csv", "analysis_plan.md"),
        ("D1", "D2", "D3", "D4"),
    ),
    "EXP-01": _spec(
        "EXP-01",
        "数据清单、去重与五类完整性审计",
        1,
        ("EXP-00",),
        ("D1", "D2", "D3"),
        (
            "data_audit.json",
            "class_distribution_verified.csv",
            "split_overlap_check.csv",
            "dataset_hashes.csv",
        ),
        ("D1", "D2", "D3", "D4"),
    ),
    "EXP-02": _spec(
        "EXP-02",
        "YOLOv8n-seg 原实验条件复现",
        2,
        ("EXP-00",),
        ("D1", "D2", "D3"),
        (
            "runs",
            "metrics_original_split.csv",
            "reproduction_provenance.json",
            "error_audit_original.csv",
        ),
    ),
    "EXP-03": _spec(
        "EXP-03",
        "YOLOv8n-seg 严格划分纠正复现",
        2,
        ("EXP-01",),
        ("D1", "D2", "D3"),
        (
            "runs",
            "metrics_strict_split.csv",
            "predictions_D3.json",
            "split_correction_effect.csv",
        ),
    ),
    "EXP-04": _spec(
        "EXP-04",
        "Mask R-CNN 两阶段强基线",
        3,
        ("EXP-01",),
        ("D1", "D2", "D3"),
        ("runs", "mask_rcnn_D3_predictions.json", "metrics.csv", "resources.json"),
    ),
    "EXP-05": _spec(
        "EXP-05",
        "同骨干 Mask2Former 共享主基线",
        3,
        ("EXP-01",),
        ("D1", "D2", "D3"),
        (
            "runs",
            "mask2former_D3_predictions.json",
            "shared_parameter_list.txt",
            "metrics.csv",
        ),
    ),
    "EXP-06": _spec(
        "EXP-06",
        "容量匹配与 FLOPs 匹配共享模型",
        3,
        ("EXP-05",),
        ("D1", "D2", "D3"),
        ("runs_B2", "runs_B3", "matching_audit.csv", "metrics_resources.csv"),
    ),
    "EXP-07": _spec(
        "EXP-07",
        "普通 MoE、类别硬路由与形态路由对照",
        3,
        ("EXP-05",),
        ("D1", "D2", "D3"),
        ("runs_B4_B6", "gmm_D1.json", "router_diagnostics.csv", "metrics_resources.csv"),
    ),
    "EXP-08A": _spec(
        "EXP-08A",
        "梯度冲突诊断 M1",
        4,
        ("EXP-09",),
        ("D1", "D2", "D3"),
        (
            "gradient_conflict_records.csv",
            "M1_summary.csv",
            "M1_sensitivity.csv",
            "M1_plot.pdf",
        ),
    ),
    "EXP-08B": _spec(
        "EXP-08B",
        "共享、独立、硬拆分与 PCGrad 性能诊断 M2",
        4,
        ("EXP-05",),
        ("D1", "D2", "D3"),
        ("runs", "M2_performance.csv", "M2_morphology_strata.csv", "M2_resources.csv"),
    ),
    "EXP-08C": _spec(
        "EXP-08C",
        "梯度冲突与连续形态关联 M3",
        4,
        ("EXP-08A",),
        ("D1", "D3"),
        ("M3_analysis_table.csv", "M3_model_results.csv", "M3_partial_dependence.pdf"),
    ),
    "EXP-08D": _spec(
        "EXP-08D",
        "专家交换、关闭与输入交换 M4",
        4,
        ("EXP-09",),
        ("D2", "D3"),
        (
            "M4_interventions.json",
            "M4_paired_effects.csv",
            "expert_load.csv",
            "expert_response_figures",
        ),
    ),
    "EXP-09": _spec(
        "EXP-09",
        "B0–B11 完整消融",
        5,
        ("EXP-07",),
        ("D1", "D2", "D3"),
        ("runs", "ablation_D3.csv", "config_diffs", "ablation_pairwise.csv"),
    ),
    "EXP-10": _spec(
        "EXP-10",
        "唯一候选选择与 D4 前冻结",
        5,
        (
            "EXP-03",
            "EXP-04",
            "EXP-05",
            "EXP-06",
            "EXP-07",
            "EXP-09",
            "EXP-12:prefreeze",
        ),
        ("D1", "D2", "D3"),
        (
            "d4_freeze_manifest.json",
            "candidate_selection_record.md",
            "D3_dry_run_report.json",
        ),
    ),
    "EXP-11": _spec(
        "EXP-11",
        "D4 一次性外部测试与效能门 E",
        6,
        ("EXP-10",),
        ("D4",),
        (
            "D4_predictions",
            "D4_combined_metrics.csv",
            "D4_domain_metrics.csv",
            "E_gate_decision.json",
        ),
        ("D4",),
    ),
    "EXP-12": _spec(
        "EXP-12",
        "校准、不确定性与选择性人工复核",
        7,
        ("EXP-12:prefreeze", "EXP-11"),
        ("D2", "D3"),
        (
            "calibrator_D2.json",
            "calibration_D3.csv",
            "selective_review.csv",
            "risk_coverage.pdf",
        ),
    ),
    "EXP-13": _spec(
        "EXP-13",
        "病害长度、正交宽度与面积量化",
        8,
        ("EXP-10",),
        ("D1", "D2", "D3"),
        (
            "quantification_manifest.csv",
            "measurements_pred.csv",
            "measurements_reference.csv",
            "quantification_metrics.csv",
        ),
    ),
    "EXP-14": _spec(
        "EXP-14",
        "完整序列推理与效率测试",
        8,
        ("EXP-10",),
        ("D3",),
        (
            "sequence_runtime.csv",
            "stage_latency.csv",
            "activation_profile.csv",
            "sequence_failure_audit.csv",
        ),
    ),
    "EXP-15": _spec(
        "EXP-15",
        "五种子统计、配对 Bootstrap 与论文结论锁定",
        9,
        ("EXP-11", "EXP-12", "EXP-13", "EXP-14"),
        (),
        (
            "final_tables",
            "final_figures",
            "statistical_results.json",
            "claim_branch.md",
            "reproducibility_manifest.json",
        ),
        ("D1", "D2", "D3", "D4"),
    ),
}


def validate_prerequisites(spec: ExperimentSpec, output_root: str | Path) -> None:
    from .dependencies import is_complete

    incomplete: list[str] = []
    for experiment_id in spec.prerequisites:
        if not is_complete(output_root, experiment_id):
            incomplete.append(experiment_id)
    if incomplete:
        raise RuntimeError(
            f"incomplete prerequisites for {spec.experiment_id}: "
            + ", ".join(incomplete)
        )
