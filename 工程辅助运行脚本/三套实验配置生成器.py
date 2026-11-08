from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.io import atomic_write_json
from morphoroute.core.layout import PROFILE_DIRECTORY_NAMES, ProjectLayout
from morphoroute.core.registry import EXPERIMENTS


FILE_NAMES = {
    "EXP-00": "exp00.json",
    "EXP-01": "exp01.json",
    "EXP-02": "exp02.json",
    "EXP-03": "exp03.json",
    "EXP-04": "exp04.json",
    "EXP-05": "exp05.json",
    "EXP-06": "exp06.json",
    "EXP-07": "exp07.json",
    "EXP-08A": "exp08a.json",
    "EXP-08B": "exp08b.json",
    "EXP-08C": "exp08c.json",
    "EXP-08D": "exp08d.json",
    "EXP-09": "exp09.json",
    "EXP-10": "exp10.json",
    "EXP-11": "exp11.json",
    "EXP-12": "exp12.json",
    "EXP-13": "exp13.json",
    "EXP-14": "exp14.json",
    "EXP-15": "exp15.json",
}


MODELS = {
    "EXP-00": "analysis",
    "EXP-01": "analysis",
    "EXP-02": "yolov8n-seg.pt",
    "EXP-03": "yolov8n-seg.pt",
    "EXP-04": "mask_rcnn_r50_fpn",
    "EXP-05": "mask2former_r50_b0",
    "EXP-06": "resource_matched_shared",
    "EXP-07": "routing_controls_B4_B5_B6",
    "EXP-08A": "gradient_probe_B0",
    "EXP-08B": "performance_diagnostics",
    "EXP-08C": "grouped_association",
    "EXP-08D": "expert_interventions_B9",
    "EXP-09": "ablation_B0_B11",
    "EXP-10": "freeze_candidate",
    "EXP-11": "frozen_external_inference",
    "EXP-12": "reliability_analysis",
    "EXP-13": "physical_quantification",
    "EXP-14": "sequence_efficiency",
    "EXP-15": "final_statistics",
}


REVERSIBLE_ACCELERATION_EXPERIMENTS = frozenset(
    {"EXP-05", "EXP-06", "EXP-07", "EXP-08B", "EXP-09"}
)

ACCELERATED_PROFILES = frozenset({"local_6gb", "autodl_5090_32gb"})

QUERY_64_EXPERIMENTS = frozenset(
    {
        "EXP-05",
        "EXP-06",
        "EXP-07",
        "EXP-08A",
        "EXP-08B",
        "EXP-08C",
        "EXP-08D",
        "EXP-09",
    }
)

GATE_START_BY_EXPERIMENT = {
    "EXP-05": 20000,
    "EXP-06": 20000,
    "EXP-07": 30000,
    "EXP-08B": 30000,
    "EXP-09": 30000,
}


PROFILE_PARAMETERS = {
    "sop_full": {
        "input_size": 1024,
        "physical_batch": 1,
        "effective_batch": 8,
        "max_updates": 50000,
        "amp": True,
        "gradient_checkpointing": True,
        "query_count": 100,
        "result_status": "paper_standard_unexecuted",
    },
    "local_6gb": {
        "input_size": 640,
        "physical_batch": 1,
        "effective_batch": 8,
        "max_updates": 50000,
        "amp": True,
        "gradient_checkpointing": True,
        "query_count": 100,
        "result_status": "local_validation",
    },
    "autodl_5090_32gb": {
        "input_size": 640,
        "physical_batch": 4,
        "effective_batch": 8,
        "max_updates": 50000,
        "amp": True,
        "gradient_checkpointing": False,
        "query_count": 100,
        "result_status": "autodl_5090_controlled_convergence",
    },
    "local_6gb_fallback": {
        "input_size": 512,
        "physical_batch": 1,
        "effective_batch": 8,
        "max_updates": 5000,
        "amp": True,
        "gradient_checkpointing": True,
        "query_count": 50,
        "result_status": "preexperiment",
    },
}


DATA_ROOT_BY_PROFILE = {
    "local_6gb": "/mnt/e/桌面/铁路隧道多病害实例分割_原始影像严格划分_v1",
    "autodl_5090_32gb": (
        "/root/autodl-tmp/铁路隧道多病害实例分割_原始影像严格划分_v1"
    ),
    "local_6gb_fallback": (
        "/mnt/e/桌面/铁路隧道多病害实例分割_原始影像严格划分_v1"
    ),
    "sop_full": (
        "/mnt/c/Users/Baoju/Desktop/"
        "铁路隧道多病害实例分割_原始影像严格划分_v1"
    ),
}


def generate_profile_configs(
    output_root: Path,
    *,
    profile_directory_names: Mapping[str, str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for profile, profile_parameters in PROFILE_PARAMETERS.items():
        data_root = DATA_ROOT_BY_PROFILE[profile]
        for experiment_id, spec in EXPERIMENTS.items():
            parameters = deepcopy(profile_parameters)
            parameters.update(
                {
                    "model": MODELS[experiment_id],
                    "execute_training": True,
                    "verify_hashes": False,
                }
            )
            if profile == "local_6gb" and experiment_id == "EXP-04":
                parameters["validation_update_interval"] = 5000
            if (
                profile in ACCELERATED_PROFILES
                and experiment_id in QUERY_64_EXPERIMENTS
            ):
                parameters["query_count"] = 64
            if (
                profile in ACCELERATED_PROFILES
                and experiment_id in REVERSIBLE_ACCELERATION_EXPERIMENTS
            ):
                parameters.update(
                    {
                        "budget_gate": {
                            "enabled": True,
                            "start_update": GATE_START_BY_EXPERIMENT[
                                experiment_id
                            ],
                            "metric": "coco/segm_mAP",
                            "patience": 2,
                            "min_delta": 0.001,
                        },
                        "release_budget_gate": False,
                        "reuse_completed_training": True,
                        "stage_warmup_end": 1000,
                        "stage_sparse_start": 20000,
                        "validation_update_interval": 5000,
                        "performance_mode": True,
                        "dataloader_workers": (
                            8 if profile == "autodl_5090_32gb" else 4
                        ),
                        "pin_memory": True,
                        "prefetch_factor": 2,
                    }
                )
            if profile in ACCELERATED_PROFILES and experiment_id == "EXP-09":
                parameters.update(
                    {
                        "screening_seeds": [42],
                        "confirmation_seeds": [42, 43, 44, 45, 46],
                        "confirmation_variants": ["B0", "B9", "B10", "B11"],
                    }
                )
            payload = {
                "experiment_id": experiment_id,
                "profile": profile,
                "data_root": data_root,
                "output_root": "./outputs",
                "seed": 42,
                "allowed_splits": list(spec.allowed_splits),
                "parameters": parameters,
            }
            directory_name = (
                profile_directory_names.get(profile, profile)
                if profile_directory_names is not None
                else profile
            )
            path = output_root / directory_name / FILE_NAMES[experiment_id]
            atomic_write_json(path, payload)
            paths.append(path)
    return paths


def main() -> int:
    root = ProjectLayout.discover(PROJECT_ROOT).configs
    paths = generate_profile_configs(
        root,
        profile_directory_names={
            profile: names[0]
            for profile, names in PROFILE_DIRECTORY_NAMES.items()
        },
    )
    print(json.dumps({"generated": len(paths), "root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
