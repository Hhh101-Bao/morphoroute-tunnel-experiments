from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, atomic_write_text
from morphoroute.core.registry import EXPERIMENTS
from morphoroute.core.run_manifest import RunManifest


def experiment_dir(config: ExperimentConfig) -> Path:
    return (
        config.output_root
        / "experiments"
        / config.experiment_id
        / config.profile
        / f"seed_{config.seed}"
    )


def dry_plan(
    config: ExperimentConfig,
    steps: Iterable[str],
    **extra: Any,
) -> dict[str, Any]:
    spec = EXPERIMENTS[config.experiment_id]
    payload = {
        "experiment_id": config.experiment_id,
        "title": spec.title,
        "stage": spec.stage,
        "profile": config.profile,
        "seed": config.seed,
        "allowed_splits": list(spec.allowed_splits),
        "metadata_splits": list(spec.metadata_splits),
        "prerequisites": list(spec.prerequisites),
        "planned_outputs": list(spec.expected_outputs),
        "steps": list(steps),
        "dry_run": True,
    }
    payload.update(extra)
    return payload


def begin(config: ExperimentConfig) -> Path:
    config.validate()
    directory = experiment_dir(config)
    RunManifest.start(config, directory)
    return directory


def finish(
    config: ExperimentConfig,
    directory: Path,
    *,
    status: str,
    produced_outputs: Iterable[str],
    notes: Iterable[str] = (),
) -> dict[str, Any]:
    manifest_path = directory / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": status,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "produced_outputs": sorted(set(produced_outputs)),
            "notes": list(notes),
        }
    )
    atomic_write_json(manifest_path, payload)
    summary_dir = config.output_root / "experiments" / config.experiment_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_dir / "run_manifest.json", payload)
    return payload


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> Path:
    rows_list = list(rows)
    if fieldnames is None:
        fieldnames = list(rows_list[0]) if rows_list else []
    fields = list(fieldnames)
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows_list)
    return atomic_write_text(path, buffer.getvalue())
