from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.aliasing import install_module_alias
from morphoroute.core.layout import ProjectLayout
from morphoroute.core.standalone import EntrySettings, parse_entry_arguments
from morphoroute.core.dependencies import (
    ExecutionGraph,
    is_complete,
)
from morphoroute.runtime.environments import EnvironmentResolver
from morphoroute.runtime.wsl_relay import run_in_wsl, should_relay_to_wsl

LAYOUT = ProjectLayout.discover(PROJECT_ROOT)
sys.path.insert(0, str(LAYOUT.scripts))

_CONFIG_GENERATOR = install_module_alias(
    "_morphoroute_generate_configs",
    LAYOUT.script_file("三套实验配置生成器.py", "generate_configs.py"),
)
_EXPERIMENT_RUNNER = install_module_alias(
    "_morphoroute_run_experiment",
    LAYOUT.script_file("统一实验运行器.py", "run_experiment.py"),
)
FILE_NAMES = _CONFIG_GENERATOR.FILE_NAMES
execute_mapping = _EXPERIMENT_RUNNER.execute_mapping
_SYNTHETIC_DATASET = install_module_alias(
    "_morphoroute_synthetic_dataset",
    LAYOUT.script_file(
        "五类合成测试数据生成器.py",
        "create_synthetic_dataset.py",
    ),
)


ENTRY_FILE_NAMES = {
    "EXP-00": "exp00_preregister.py",
    "EXP-01": "exp01_data_audit.py",
    "EXP-02": "exp02_yolo_original.py",
    "EXP-03": "exp03_yolo_strict.py",
    "EXP-04": "exp04_mask_rcnn.py",
    "EXP-05": "exp05_mask2former.py",
    "EXP-06": "exp06_resource_matched.py",
    "EXP-07": "exp07_routing_controls.py",
    "EXP-08A": "exp08a_gradient_conflict.py",
    "EXP-08B": "exp08b_performance_diagnosis.py",
    "EXP-08C": "exp08c_morphology_association.py",
    "EXP-08D": "exp08d_expert_intervention.py",
    "EXP-09": "exp09_ablation.py",
    "EXP-10": "exp10_freeze.py",
    "EXP-11": "exp11_external_test.py",
    "EXP-12": "exp12_reliability.py",
    "EXP-13": "exp13_quantification.py",
    "EXP-14": "exp14_sequence_efficiency.py",
    "EXP-15": "exp15_statistics.py",
}


def _load_payload(
    experiment_id: str,
    settings: EntrySettings,
    config_path: Path | None,
) -> dict:
    selected_config = config_path or (
        LAYOUT.profile_config_dir(settings.profile) / FILE_NAMES[experiment_id]
    )
    payload = json.loads(selected_config.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != experiment_id:
        raise ValueError(
            f"entry {experiment_id} cannot run config {payload.get('experiment_id')}"
        )
    parameters = dict(payload.get("parameters", {}))
    parameters.update(
        {
            "smoke_test": settings.smoke_test,
            "auto_run_prerequisites": settings.auto_run_prerequisites,
            "resume": settings.resume,
        }
    )
    if settings.smoke_test:
        parameters.update(
            {
                "max_updates": 3,
                "input_size": 128,
                "query_count": 20,
                "stage_warmup_end": 1,
                "stage_sparse_start": 2,
                "routing_threshold_candidates": [0.50, 0.75],
                "seeds": [42],
                "physical_batch": 1,
                "effective_batch": 1,
                "gradient_accumulation": 1,
                "timing_warmup": 1,
                "timing_repeats": 1,
                "confirmation_phrase": "I UNDERSTAND D4 IS ONE-SHOT",
                "result_status": "synthetic_smoke_only",
            }
        )
    payload.update(
        {
            "profile": settings.profile,
            "data_root": str(settings.dataset_root),
            "output_root": str(settings.output_root),
            "parameters": parameters,
        }
    )
    return payload


def _prepare_smoke_settings(settings: EntrySettings) -> EntrySettings:
    if not settings.smoke_test:
        return settings
    smoke_root = (
        settings.output_root
        if settings.output_root.name == "smoke_runs"
        else settings.output_root / "smoke_runs"
    )
    synthetic_root = smoke_root / "synthetic_five_class_dataset"
    _SYNTHETIC_DATASET.create_synthetic_dataset(synthetic_root)
    return replace(
        settings,
        dataset_root=synthetic_root,
        output_root=smoke_root,
    )


def _run_windows_prerequisites(
    target: str,
    arguments,
) -> None:
    graph = ExecutionGraph.from_registry()
    for node_id in graph.plan(target)[:-1]:
        if is_complete(arguments.settings.output_root, node_id):
            continue
        experiment_id, _, stage = node_id.partition(":")
        entry = LAYOUT.experiments / ENTRY_FILE_NAMES[experiment_id]
        child_tokens = arguments.relay_tokens()
        child_tokens = [
            token
            for token in child_tokens
            if token not in {
                "--auto-run-prerequisites",
                "--no-auto-run-prerequisites",
            }
        ]
        child_tokens.append("--no-auto-run-prerequisites")
        if stage:
            if "--stage" in child_tokens:
                stage_index = child_tokens.index("--stage")
                del child_tokens[stage_index : stage_index + 2]
            child_tokens.extend(("--stage", stage))
        completed = subprocess.run(
            [sys.executable, str(entry), *child_tokens],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"automatic prerequisite {node_id} failed with "
                f"exit code {completed.returncode}"
            )
        if not is_complete(arguments.settings.output_root, node_id):
            raise RuntimeError(
                f"automatic prerequisite {node_id} did not write a "
                "completed manifest"
            )


def main_for(
    experiment_id: str,
    *,
    profile: str = "local_6gb",
    dataset_root: Path,
    output_root: Path | None = None,
    smoke_test: bool = False,
    auto_run_prerequisites: bool = True,
    resume: bool = True,
    entry_path: Path | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    defaults = EntrySettings(
        profile=profile,
        dataset_root=Path(dataset_root),
        output_root=Path(output_root) if output_root else PROJECT_ROOT / "outputs",
        smoke_test=smoke_test,
        auto_run_prerequisites=auto_run_prerequisites,
        resume=resume,
    )
    arguments = parse_entry_arguments(argv, defaults)
    arguments = replace(
        arguments,
        settings=_prepare_smoke_settings(arguments.settings),
    )
    if should_relay_to_wsl(dry_run=arguments.dry_run):
        target_node = (
            f"EXP-12:{arguments.execution_stage}"
            if experiment_id == "EXP-12" and arguments.execution_stage
            else experiment_id
        )
        if arguments.settings.auto_run_prerequisites:
            _run_windows_prerequisites(target_node, arguments)
        return run_in_wsl(
            PROJECT_ROOT,
            Path(entry_path) if entry_path else Path(sys.argv[0]).resolve(),
            arguments.relay_tokens(),
            environment=EnvironmentResolver.for_experiment(experiment_id),
        )
    try:
        payload = _load_payload(
            experiment_id,
            arguments.settings,
            arguments.config_path,
        )
        if arguments.execution_stage is not None:
            payload["parameters"]["execution_stage"] = (
                arguments.execution_stage
            )
        result = execute_mapping(
            payload,
            dry_run=arguments.dry_run,
            resume=arguments.settings.resume,
            analysis_only=arguments.analysis_only,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0
