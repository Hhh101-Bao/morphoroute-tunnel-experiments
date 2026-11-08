from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import ExperimentConfig
from .registry import EXPERIMENTS, validate_prerequisites
from .run_manifest import RunManifest


Handler = Callable[[ExperimentConfig, Path], dict[str, Any]]


def run_registered_experiment(
    config: ExperimentConfig,
    handler: Handler,
    *,
    check_prerequisites: bool = True,
) -> dict[str, Any]:
    config.validate()
    if config.experiment_id not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {config.experiment_id}")
    spec = EXPERIMENTS[config.experiment_id]
    forbidden = set(config.allowed_splits).difference(spec.allowed_splits)
    if forbidden:
        raise PermissionError(
            f"{config.experiment_id} cannot access splits: {sorted(forbidden)}"
        )
    if check_prerequisites:
        validate_prerequisites(spec, config.output_root)

    run_dir = (
        config.output_root
        / "experiments"
        / config.experiment_id
        / config.profile
        / f"seed_{config.seed}"
    )
    RunManifest.start(config, run_dir)
    return handler(config, run_dir)
