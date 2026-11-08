from __future__ import annotations

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, atomic_write_text, sha256_file

from .common import begin, dry_plan, finish, write_csv


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "登记 RQ1–RQ8、主终点和结构终点",
            "登记 M/E 判据、强比较、随机种子和区段级 Bootstrap",
            "写入预注册文件并记录 SHA-256",
        ],
        comparison_scope="all_registered_questions",
    )
    if dry_run:
        return plan
    directory = begin(config)
    preregistration = {
        "schema_version": 1,
        "research_questions": [f"RQ{index}" for index in range(1, 9)],
        "primary_endpoint": "D4 combined macro Mask AP",
        "structure_endpoints": [
            "crack clDice",
            "crack break rate",
            "crack connected-component error",
            "area Boundary-F1",
            "area HD95 (pixels)",
            "area relative area error",
        ],
        "mechanism_gate": {
            "minimum_independent_evidence_types": 2,
            "requires_gradient_or_performance": True,
        },
        "effectiveness_gate": {
            "primary_endpoint_required": True,
            "worst_class_protected": True,
            "one_structure_better_other_noninferior": True,
        },
        "bootstrap_unit": "tunnel_section_or_sequence",
        "development_seeds": [17, 42, 89],
        "final_seeds": [17, 42, 89, 123, 2026],
        "yolo_scope": "RQ1 only; excluded from innovation selection",
    }
    preregistration_path = directory / "preregistration.json"
    atomic_write_json(preregistration_path, preregistration)
    write_csv(
        directory / "freeze_registry.csv",
        [
            {
                "artifact": "preregistration.json",
                "sha256": sha256_file(preregistration_path),
                "status": "frozen",
            }
        ],
    )
    atomic_write_text(
        directory / "analysis_plan.md",
        "# Analysis plan\n\n"
        "All primary comparisons use paired tunnel-section or sequence resampling. "
        "Holm adjustment is restricted to preregistered primary comparisons.\n",
    )
    return finish(
        config,
        directory,
        status="completed",
        produced_outputs=plan["planned_outputs"],
    )
