from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from morphoroute.core.config import ExperimentConfig
from morphoroute.visualization.plots import plot_gradient_conflict
from morphoroute.visualization.tables import save_result_bundle

from .common import begin, dry_plan, finish


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "从同一 checkpoint、batch 和共享参数子集计算线/面梯度",
            "在 AMP unscale 后、梯度裁剪前记录余弦、负余弦和范数比",
            "按 group、训练阶段、层子集、分辨率和 seed 汇总",
        ],
        evidence_gate="M1",
        same_checkpoint=True,
        same_parameter_subset=True,
    )
    if dry_run:
        return plan
    directory = begin(config)
    source_value = config.parameters.get("gradient_records_csv")
    if not source_value:
        candidates = list(
            (config.output_root / "experiments" / "EXP-09").rglob(
                "gradient_diagnostics.jsonl"
            )
        )
        if not candidates:
            raise FileNotFoundError(
                "EXP-08A cannot find EXP-09 B9 gradient diagnostics"
            )
        records = []
        for path in candidates:
            records.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        table = pd.DataFrame(records)
    else:
        table = pd.read_csv(Path(str(source_value)))
    required = {
        "group_id",
        "gradient_cosine",
        "norm_ratio",
        "seed",
        "checkpoint_stage",
        "parameter_subset",
        "input_size",
    }
    if missing := required.difference(table.columns):
        raise ValueError(f"M1 table lacks columns: {sorted(missing)}")
    table["negative_cosine"] = table["gradient_cosine"] < 0
    group_summary = (
        table.groupby("group_id", as_index=False)
        .agg(
            gradient_cosine=("gradient_cosine", "mean"),
            negative_cosine_rate=("negative_cosine", "mean"),
            norm_ratio=("norm_ratio", "mean"),
        )
    )
    sensitivity = (
        table.groupby(
            ["seed", "checkpoint_stage", "parameter_subset", "input_size"],
            as_index=False,
        )
        .agg(
            gradient_cosine=("gradient_cosine", "mean"),
            negative_cosine_rate=("negative_cosine", "mean"),
            norm_ratio=("norm_ratio", "mean"),
        )
    )
    table.to_csv(directory / "gradient_conflict_records.csv", index=False)
    group_summary.to_csv(directory / "M1_summary.csv", index=False)
    sensitivity.to_csv(directory / "M1_sensitivity.csv", index=False)
    try:
        plot_gradient_conflict(
            sensitivity["checkpoint_stage"].astype(str).tolist(),
            sensitivity["gradient_cosine"].tolist(),
            directory / "M1_plot",
        )
    except RuntimeError:
        pass
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
