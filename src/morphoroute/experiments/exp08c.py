from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from morphoroute.core.config import ExperimentConfig
from morphoroute.models.morphology import FEATURE_NAMES
from morphoroute.visualization.plots import plot_morphology_association
from morphoroute.analysis.morphology_association import (
    fit_adjusted_association,
)

from .common import begin, dry_plan, finish


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "连接 group 级冲突记录与五维连续形态",
            "以类别和实例尺度为强制协变量拟合预注册模型",
            "输出调整后系数、置信区间和分箱趋势",
        ],
        evidence_gate="M3",
        required_covariates=["class_name", "instance_area"],
    )
    if dry_run:
        return plan
    directory = begin(config)
    source_value = config.parameters.get("analysis_table_csv")
    if not source_value:
        candidates = list(
            (config.output_root / "experiments" / "EXP-08A").rglob(
                "gradient_conflict_records.csv"
            )
        )
        if not candidates:
            raise FileNotFoundError(
                "EXP-08C cannot find EXP-08A gradient records"
            )
        source_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    else:
        source_path = Path(str(source_value))
    table = pd.read_csv(source_path)
    required = {
        "group_id",
        "gradient_cosine",
        "class_name",
        "instance_area",
        *FEATURE_NAMES,
    }
    if missing := required.difference(table.columns):
        raise ValueError(f"M3 table lacks columns: {sorted(missing)}")
    coefficient_table = fit_adjusted_association(
        table,
        n_resamples=int(
            config.parameters.get("association_bootstrap_resamples", 1000)
        ),
        seed=config.seed,
    )
    table.to_csv(directory / "M3_analysis_table.csv", index=False)
    coefficient_table.to_csv(directory / "M3_model_results.csv", index=False)
    try:
        plot_morphology_association(
            table["log_aspect_ratio"],
            table["gradient_cosine"],
            directory / "M3_partial_dependence",
        )
    except RuntimeError:
        pass
    return finish(
        config,
        directory,
        status="completed_association_not_causation",
        produced_outputs=plan["planned_outputs"],
    )
