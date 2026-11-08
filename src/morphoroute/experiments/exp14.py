from __future__ import annotations

from pathlib import Path

import pandas as pd

from morphoroute.core.config import ExperimentConfig
from morphoroute.visualization.plots import plot_latency_stack
from morphoroute.data.manifest import load_manifest
from morphoroute.runtime.sequence_benchmark import benchmark_frozen_models

from .common import begin, dry_plan, finish


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "在同一硬件、驱动、精度、batch=1 协议下固定预热并同步计时",
            "分别记录读取、预处理、前向、拼接后处理和写出时延",
            "报告显存、FLOPs、激活专家、双激活和长序列失败",
        ],
        batch_size=1,
        same_hardware_required=True,
        synchronized_timing=True,
    )
    if dry_run:
        return plan
    directory = begin(config)
    runtime_value = config.parameters.get("runtime_records_csv")
    if not runtime_value:
        freeze_candidates = list(
            (config.output_root / "experiments" / "EXP-10").rglob(
                "d4_freeze_manifest.json"
            )
        )
        if not freeze_candidates:
            raise FileNotFoundError("EXP-14 cannot find EXP-10 freeze manifest")
        import json

        freeze = json.loads(
            max(
                freeze_candidates,
                key=lambda path: path.stat().st_mtime_ns,
            ).read_text(encoding="utf-8")
        )
        checkpoint_index = freeze["artifacts"]["checkpoint"]["path"]
        records = load_manifest(
            config.data_root / "metadata" / "split_manifest.csv",
            config.data_root,
        )
        benchmark_rows = benchmark_frozen_models(
            checkpoint_index,
            records,
            warmup=int(config.parameters.get("timing_warmup", 5)),
        )
        table = pd.DataFrame(benchmark_rows)
    else:
        table = pd.read_csv(Path(str(runtime_value)))
    required = {
        "model",
        "sequence_id",
        "hardware_id",
        "protocol_id",
        "batch_size",
        "input_ms",
        "preprocess_ms",
        "forward_ms",
        "postprocess_ms",
        "output_ms",
        "peak_memory_mb",
        "active_experts",
        "dual_activation_ratio",
    }
    if missing := required.difference(table.columns):
        raise ValueError(f"runtime table lacks: {sorted(missing)}")
    if table["hardware_id"].nunique() != 1 or table["protocol_id"].nunique() != 1:
        raise ValueError("all efficiency comparisons must share hardware and protocol")
    if set(table["batch_size"]) != {1}:
        raise ValueError("efficiency comparison requires batch_size=1")
    table["end_to_end_ms"] = table[
        ["input_ms", "preprocess_ms", "forward_ms", "postprocess_ms", "output_ms"]
    ].sum(axis=1)
    if "missed" in table:
        table["missed_run_length"] = (
            table.sort_values(["model", "sequence_id", "record_id"])
            .groupby(["model", "sequence_id"])["missed"]
            .transform(
                lambda values: values.astype(int)
                .groupby((~values.astype(bool)).cumsum())
                .cumsum()
            )
        )
        table["flicker_rate"] = (
            table.sort_values(["model", "sequence_id", "record_id"])
            .groupby(["model", "sequence_id"])["missed"]
            .transform(
                lambda values: values.astype(int).diff().abs().fillna(0)
            )
        )
        table["failure_reason"] = table["missed"].map(
            {True: "no_instance_prediction", False: ""}
        )
    table.to_csv(directory / "sequence_runtime.csv", index=False)
    table[
        [
            "model",
            "sequence_id",
            "input_ms",
            "preprocess_ms",
            "forward_ms",
            "postprocess_ms",
            "output_ms",
            "end_to_end_ms",
        ]
    ].to_csv(directory / "stage_latency.csv", index=False)
    table[
        [
            "model",
            "sequence_id",
            "peak_memory_mb",
            "active_experts",
            "dual_activation_ratio",
        ]
    ].to_csv(directory / "activation_profile.csv", index=False)
    failure_columns = [
        column
        for column in ("missed_run_length", "flicker_rate", "failure_reason")
        if column in table
    ]
    table[["model", "sequence_id", *failure_columns]].to_csv(
        directory / "sequence_failure_audit.csv", index=False
    )
    try:
        grouped = table.groupby("model", as_index=False).mean(numeric_only=True)
        plot_latency_stack(
            grouped["model"].tolist(),
            {
                stage: grouped[f"{stage}_ms"].tolist()
                for stage in ("input", "preprocess", "forward", "postprocess", "output")
            },
            directory / "stage_latency",
        )
    except RuntimeError:
        pass
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
