from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.dependencies import ExecutionGraph, is_complete
from morphoroute.core.registry import EXPERIMENTS


MODULE_NAMES = {
    "EXP-00": "exp00",
    "EXP-01": "exp01",
    "EXP-02": "exp02",
    "EXP-03": "exp03",
    "EXP-04": "exp04",
    "EXP-05": "exp05",
    "EXP-06": "exp06",
    "EXP-07": "exp07",
    "EXP-08A": "exp08a",
    "EXP-08B": "exp08b",
    "EXP-08C": "exp08c",
    "EXP-08D": "exp08d",
    "EXP-09": "exp09",
    "EXP-10": "exp10",
    "EXP-11": "exp11",
    "EXP-12": "exp12",
    "EXP-13": "exp13",
    "EXP-14": "exp14",
    "EXP-15": "exp15",
}


def load_handler(experiment_id: str):
    if experiment_id not in MODULE_NAMES or experiment_id not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {experiment_id}")
    module = importlib.import_module(
        f"morphoroute.experiments.{MODULE_NAMES[experiment_id]}"
    )
    return module.execute


def execute_config(
    config_path: str | Path,
    *,
    dry_run: bool,
    resume: bool = False,
    analysis_only: bool = False,
):
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return execute_mapping(
        payload,
        dry_run=dry_run,
        resume=resume,
        analysis_only=analysis_only,
    )


def execute_mapping(
    payload: Mapping[str, Any],
    *,
    dry_run: bool,
    resume: bool = False,
    analysis_only: bool = False,
):
    payload = dict(payload)
    experiment_id = str(payload.get("experiment_id", ""))
    handler = load_handler(experiment_id)
    parameters = dict(payload.get("parameters", {}))
    if resume:
        parameters["resume"] = True
    if analysis_only:
        parameters["analysis_only"] = True
    payload["parameters"] = parameters
    config = ExperimentConfig.from_mapping(payload)
    config.validate()
    if not dry_run:
        stage = parameters.get("execution_stage")
        execution_node = (
            f"EXP-12:{stage}"
            if experiment_id == "EXP-12" and stage
            else experiment_id
        )
        incomplete = [
            requirement.producer
            for requirement in ExecutionGraph.from_registry().requirements(
                execution_node
            )
            if not is_complete(config.output_root, requirement.producer)
        ]
        if incomplete:
            raise RuntimeError(
                f"incomplete prerequisites for {execution_node}: "
                + ", ".join(incomplete)
            )
    return handler(config, dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one registered railway-tunnel experiment."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = execute_config(
            arguments.config,
            dry_run=arguments.dry_run,
            resume=arguments.resume,
            analysis_only=arguments.analysis_only,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
