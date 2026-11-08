from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, sha256_file
from morphoroute.core.layout import ProjectLayout
from morphoroute.data.coco_dataset import ensure_coco_dataset
from morphoroute.data.conversion import load_class_mapping
from morphoroute.data.manifest import load_manifest
from morphoroute.data.materialize import MaterializedDataset
from morphoroute.data.morphology_table import ensure_d1_morphology_prior
from morphoroute.mmdet.config_builder import (
    MMDetResolvedConfig,
    build_morphoroute_config,
)
from morphoroute.runtime.mmdet_adapter import (
    postprocess_trained_mmdet_checkpoint,
)


@dataclass(frozen=True)
class B2RecoveryContext:
    experiment_id: str
    variant_id: str
    profile: str
    seed: int
    checkpoint: str
    checkpoint_sha256: str
    config_path: str
    config_sha256: str
    dataset_fingerprint: str
    cache_signature: str
    training_action: str = "forbidden"


def validate_b2_recovery_context(
    config: ExperimentConfig,
    materialized: MaterializedDataset,
    resolved: MMDetResolvedConfig,
    checkpoint: str | Path,
) -> B2RecoveryContext:
    checkpoint = Path(checkpoint)
    if config.experiment_id != "EXP-06":
        raise ValueError("recovery is restricted to EXP-06")
    if config.profile != "local_6gb" or config.seed != 42:
        raise ValueError("recovery requires local_6gb seed 42")
    if bool(config.parameters.get("resume", False)):
        raise ValueError("B2 recovery requires resume=false")
    if resolved.work_dir.name != "seed_42" or resolved.work_dir.parent.name != "B2":
        raise ValueError(f"resolved work directory is not B2/seed_42: {resolved.work_dir}")
    if not resolved.config_path.is_file():
        raise FileNotFoundError(
            f"resolved MMDetection config does not exist: {resolved.config_path}"
        )
    if not checkpoint.is_file() or checkpoint.parent != resolved.work_dir:
        raise FileNotFoundError(
            f"B2 best checkpoint is missing or outside work directory: {checkpoint}"
        )
    if not checkpoint.name.startswith("best_coco_segm_mAP"):
        raise ValueError(f"recovery requires the frozen best checkpoint: {checkpoint}")
    arguments_path = resolved.work_dir / "mmdet_run_arguments.json"
    if not arguments_path.is_file():
        raise FileNotFoundError(
            f"original MMDetection run arguments do not exist: {arguments_path}"
        )
    arguments = json.loads(arguments_path.read_text(encoding="utf-8"))
    expected_fingerprint = str(arguments.get("dataset_fingerprint", ""))
    actual_fingerprint = materialized.fingerprint.sha256
    if not expected_fingerprint or expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "dataset fingerprint mismatch: "
            f"expected {expected_fingerprint!r}, got {actual_fingerprint!r}"
        )
    cache_signature = str(arguments.get("training_cache_signature", ""))
    if not cache_signature:
        raise ValueError("original training cache signature is missing")
    return B2RecoveryContext(
        experiment_id=config.experiment_id,
        variant_id="B2",
        profile=config.profile,
        seed=config.seed,
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        config_path=str(resolved.config_path),
        config_sha256=sha256_file(resolved.config_path),
        dataset_fingerprint=actual_fingerprint,
        cache_signature=cache_signature,
    )


def _load_config(
    project_root: Path,
    data_root: Path,
    output_root: Path,
) -> ExperimentConfig:
    layout = ProjectLayout.discover(project_root)
    config_path = layout.profile_config_dir("local_6gb") / "exp06.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = dict(payload.get("parameters", {}))
    parameters.update(
        {
            "resume": False,
            "smoke_test": False,
            "run_name": "B2/seed_42",
            "variant_id": "B2",
            "release_budget_gate": False,
        }
    )
    payload.update(
        {
            "profile": "local_6gb",
            "seed": 42,
            "data_root": str(data_root),
            "output_root": str(output_root),
            "parameters": parameters,
        }
    )
    config = ExperimentConfig.from_mapping(payload)
    config.validate()
    return config


def _prepare_runtime(
    config: ExperimentConfig,
) -> tuple[MaterializedDataset, MMDetResolvedConfig, Path]:
    records = load_manifest(
        config.data_root / "metadata/split_manifest.csv",
        config.data_root,
    )
    class_mapping = load_class_mapping(
        config.data_root / "metadata/class_mapping.json"
    )
    experiment_directory = (
        config.output_root
        / "experiments/EXP-06/local_6gb/seed_42"
    )
    _, morphology_prior = ensure_d1_morphology_prior(
        records,
        class_mapping,
        experiment_directory,
        seed=config.seed,
    )
    materialized = ensure_coco_dataset(
        records,
        class_mapping,
        config.output_root / "derived",
        ("D1", "D2", "D3"),
    )
    resolved = build_morphoroute_config(
        config,
        materialized,
        variant_id="B2",
        morphology_prior_path=morphology_prior,
        tau=float(config.parameters.get("routing_tau", 0.75)),
    )
    checkpoint = (
        resolved.work_dir / "best_coco_segm_mAP_iter_200000.pth"
    )
    return materialized, resolved, checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover EXP-06/B2 postprocessing without B2 training."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = _load_config(
            arguments.project_root,
            arguments.dataset_root,
            arguments.output_root,
        )
        materialized, resolved, default_checkpoint = _prepare_runtime(config)
        checkpoint = arguments.checkpoint or default_checkpoint
        context = validate_b2_recovery_context(
            config,
            materialized,
            resolved,
            checkpoint,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(context), ensure_ascii=False, indent=2))
    if arguments.dry_run:
        return 0
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        result = postprocess_trained_mmdet_checkpoint(
            config,
            materialized,
            resolved,
            checkpoint,
            variant_id="B2",
            cache_signature=context.cache_signature,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 3
    audit_path = resolved.work_dir / "recovery_audit.json"
    atomic_write_json(
        audit_path,
        {
            **asdict(context),
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "b2_training_invoked": False,
            "d2_image_count": len(result.validation_predictions.images),
            "d3_image_count": len(result.predictions.images),
            "mask_ap": result.evaluation.mask_ap,
            "mask_ap50": result.evaluation.mask_ap50,
            "status": "completed",
        },
    )
    print(f"B2 postprocess recovery completed: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
