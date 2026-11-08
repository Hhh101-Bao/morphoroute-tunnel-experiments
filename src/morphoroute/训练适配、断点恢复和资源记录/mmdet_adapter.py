from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json, sha256_file
from morphoroute.data.materialize import MaterializedDataset
from morphoroute.evaluation.pipeline import (
    EvaluationOutputPaths,
    EvaluationResult,
    evaluate_bundle,
    write_prediction_bundle,
    write_evaluation_outputs,
)
from morphoroute.evaluation.io import read_prediction_bundle
from morphoroute.evaluation.schema import PredictionBundle
from morphoroute.mmdet.config_builder import (
    MMDetResolvedConfig,
    build_mask2former_config,
    build_mask_rcnn_config,
    build_morphoroute_config,
)

from .base import TrainingAdapter, TrainingPlan, plan_common
from .acceleration import training_cache_signature


class MMDetBackend(Protocol):
    version: str

    def train(
        self,
        resolved: MMDetResolvedConfig,
        *,
        resume: bool,
    ) -> Path: ...

    def predict(
        self,
        checkpoint: Path,
        resolved: MMDetResolvedConfig,
        materialized: MaterializedDataset,
        *,
        split: str,
    ) -> PredictionBundle: ...


@dataclass(frozen=True)
class MMDetRunResult:
    resolved_config: MMDetResolvedConfig
    best_checkpoint: Path
    validation_predictions: PredictionBundle
    predictions: PredictionBundle
    evaluation: EvaluationResult
    evaluation_outputs: EvaluationOutputPaths


def _training_cache_directory(
    config: ExperimentConfig,
    signature: str,
) -> Path:
    return config.output_root / "training_cache" / "mmdet" / signature


def _load_completed_cache(
    cache_directory: Path,
    *,
    signature: str,
    experiment_id: str,
    resolved: MMDetResolvedConfig,
) -> MMDetRunResult | None:
    record_path = cache_directory / "completed_run.json"
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            return None
        if record.get("signature") != signature:
            return None
        checkpoint = Path(str(record["best_checkpoint"]))
        validation_path = cache_directory / "predictions_D2.json"
        prediction_path = cache_directory / "predictions_D3.json"
        source_work_dir = Path(str(record["source_work_dir"]))
        source_config_path = Path(str(record["resolved_config_path"]))
        output_payload = record["evaluation_outputs"]
        output_paths = EvaluationOutputPaths(
            metrics_csv=Path(str(output_payload["metrics_csv"])),
            metrics_xlsx=Path(str(output_payload["metrics_xlsx"])),
            summary_json=Path(str(output_payload["summary_json"])),
            overlay_png=Path(str(output_payload["overlay_png"])),
            overlay_pdf=Path(str(output_payload["overlay_pdf"])),
        )
        required_paths = (
            checkpoint,
            validation_path,
            prediction_path,
            source_config_path,
            output_paths.metrics_csv,
            output_paths.metrics_xlsx,
            output_paths.summary_json,
        )
        if not all(path.is_file() for path in required_paths):
            return None
        validation = read_prediction_bundle(validation_path)
        predictions = read_prediction_bundle(prediction_path)
        cache_metadata = {
            "reused_training_cache": True,
            "training_cache_signature": signature,
            "training_cache_source": str(source_work_dir),
        }
        validation = replace(
            validation,
            experiment_id=experiment_id,
            metadata={**dict(validation.metadata), **cache_metadata},
        )
        predictions = replace(
            predictions,
            experiment_id=experiment_id,
            metadata={**dict(predictions.metadata), **cache_metadata},
        )
        evaluation = EvaluationResult(**record["evaluation"])
        evaluation = replace(evaluation, experiment_id=experiment_id)
        cached_resolved = replace(
            resolved,
            work_dir=source_work_dir,
            config_path=source_config_path,
        )
        print(
            "Reusing completed MMDetection cache: "
            f"experiment={experiment_id}, "
            f"model_kind={resolved.model_kind}, "
            f"source={source_work_dir}",
            flush=True,
        )
        return MMDetRunResult(
            resolved_config=cached_resolved,
            best_checkpoint=checkpoint,
            validation_predictions=validation,
            predictions=predictions,
            evaluation=evaluation,
            evaluation_outputs=output_paths,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_completed_cache(
    cache_directory: Path,
    *,
    signature: str,
    result: MMDetRunResult,
) -> None:
    cache_directory.mkdir(parents=True, exist_ok=True)
    validation_path = cache_directory / "predictions_D2.json"
    prediction_path = cache_directory / "predictions_D3.json"
    write_prediction_bundle(result.validation_predictions, validation_path)
    write_prediction_bundle(result.predictions, prediction_path)
    source_config_path = result.resolved_config.config_path
    if not source_config_path.is_file():
        source_config_path = cache_directory / "resolved_config_snapshot.json"
        atomic_write_json(source_config_path, result.resolved_config.config)
    atomic_write_json(
        cache_directory / "completed_run.json",
        {
            "schema_version": 1,
            "status": "completed",
            "signature": signature,
            "best_checkpoint": str(result.best_checkpoint),
            "source_work_dir": str(result.resolved_config.work_dir),
            "resolved_config_path": str(source_config_path),
            "evaluation": asdict(result.evaluation),
            "evaluation_outputs": {
                key: str(value)
                for key, value in asdict(result.evaluation_outputs).items()
            },
        },
    )


def write_completed_mmdet_cache(
    cache_directory: Path,
    *,
    signature: str,
    result: MMDetRunResult,
) -> None:
    _write_completed_cache(
        cache_directory,
        signature=signature,
        result=result,
    )


def postprocess_trained_mmdet_checkpoint(
    config: ExperimentConfig,
    materialized: MaterializedDataset,
    resolved: MMDetResolvedConfig,
    checkpoint: Path,
    *,
    variant_id: str | None,
    cache_signature: str,
    backend: MMDetBackend | None = None,
) -> MMDetRunResult:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"trained checkpoint does not exist: {checkpoint}")
    active_backend = backend or OfficialMMDetBackend()
    validation_predictions = active_backend.predict(
        checkpoint,
        resolved,
        materialized,
        split="D2",
    )
    if validation_predictions.split != "D2":
        if isinstance(active_backend, OfficialMMDetBackend):
            raise ValueError(
                "MMDetection backend returned a non-D2 validation bundle"
            )
        validation_predictions = replace(
            validation_predictions,
            split="D2",
            metadata={
                **dict(validation_predictions.metadata),
                "test_backend_split_normalized": True,
            },
        )
    validation_predictions = replace(
        validation_predictions,
        metadata={
            **dict(validation_predictions.metadata),
            "seed": config.seed,
            "profile": config.profile,
            "smoke_test": bool(config.parameters.get("smoke_test", False)),
            "variant_id": variant_id,
            "recovered_from_existing_checkpoint": True,
        },
    )
    predictions = active_backend.predict(
        checkpoint,
        resolved,
        materialized,
        split="D3",
    )
    if predictions.split != "D3":
        raise ValueError("MMDetection backend returned a non-D3 prediction bundle")
    predictions = replace(
        predictions,
        metadata={
            **dict(predictions.metadata),
            "seed": config.seed,
            "profile": config.profile,
            "smoke_test": bool(config.parameters.get("smoke_test", False)),
            "variant_id": variant_id,
            "recovered_from_existing_checkpoint": True,
        },
    )
    evaluation = evaluate_bundle(predictions)
    outputs = write_evaluation_outputs(
        evaluation,
        predictions,
        resolved.work_dir / "evaluation_D3",
    )
    result = MMDetRunResult(
        resolved_config=resolved,
        best_checkpoint=checkpoint,
        validation_predictions=validation_predictions,
        predictions=predictions,
        evaluation=evaluation,
        evaluation_outputs=outputs,
    )
    write_completed_mmdet_cache(
        _training_cache_directory(config, cache_signature),
        signature=cache_signature,
        result=result,
    )
    return result


class MMDetectionAdapter(TrainingAdapter):
    def build_plan(
        self,
        config: ExperimentConfig,
        derived_dataset_root: str | Path,
    ) -> TrainingPlan:
        common = plan_common(config)
        root = Path(derived_dataset_root)
        dataset_config = root / "instances_D1_D2_D3.json"
        output_dir = (
            config.output_root
            / "experiments"
            / config.experiment_id
            / config.profile
            / f"seed_{config.seed}"
        )
        resolved_config = output_dir / "resolved_mmdet_config.py"
        framework_config = {
            "model_name": common["model"],
            "train_dataloader": {
                "batch_size": common["physical_batch"],
                "dataset_split": "D1",
            },
            "val_dataloader": {"dataset_split": "D2"},
            "test_dataloader": {"dataset_split": "D3"},
            "train_cfg": {
                "type": "IterBasedTrainLoop",
                "max_iters": (
                    common["max_updates"]
                    * common["gradient_accumulation"]
                ),
                "val_interval": (
                    min(1000, common["max_updates"])
                    * common["gradient_accumulation"]
                ),
            },
            "optim_wrapper": {
                "type": "AmpOptimWrapper" if common["amp"] else "OptimWrapper",
                "accumulative_counts": common["gradient_accumulation"],
                "clip_grad": {"max_norm": 1.0, "norm_type": 2},
            },
            "randomness": {"seed": config.seed, "deterministic": True},
            "input_size": common["input_size"],
            "dataset_config": str(dataset_config),
        }
        command = (
            "python",
            "-m",
            "mmdet.tools.train",
            str(resolved_config),
            "--work-dir",
            str(output_dir),
        )
        return TrainingPlan(
            backend="mmdetection",
            experiment_id=config.experiment_id,
            model=common["model"],
            seed=config.seed,
            train_split="D1",
            validation_split="D2",
            training_inputs=("D1", "D2"),
            input_size=common["input_size"],
            max_updates=common["max_updates"],
            physical_batch=common["physical_batch"],
            effective_batch=common["effective_batch"],
            gradient_accumulation=common["gradient_accumulation"],
            amp=common["amp"],
            dataset_config=dataset_config,
            output_dir=output_dir,
            command=command,
            framework_config=framework_config,
        )

    def train(self, plan: TrainingPlan, *, resume: bool = False) -> Any:
        try:
            from mmengine.config import Config
            from mmengine.runner import Runner
        except ImportError as error:
            raise RuntimeError(
                "MMEngine/MMDetection is unavailable. Activate ENV-MAIN in WSL."
            ) from error
        configuration = Config(plan.framework_config)
        configuration.work_dir = str(plan.output_dir)
        configuration.resume = resume
        runner = Runner.from_cfg(configuration)
        return runner.train()


def _resolve_base_config_path(base_config: str | Path) -> Path:
    """Resolve MMEngine package config URIs to installed config files."""
    value = str(base_config)
    if "::" not in value:
        return Path(value)

    package_name, relative_name = value.split("::", 1)
    if not package_name or not relative_name:
        raise ValueError(f"invalid package config URI: {value}")
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        raise FileNotFoundError(
            f"cannot resolve {value}: package {package_name!r} is not installed"
        )

    package_roots: list[Path] = []
    locations = getattr(spec, "submodule_search_locations", None)
    if locations:
        package_roots.extend(Path(location) for location in locations)
    origin = getattr(spec, "origin", None)
    if origin:
        package_roots.append(Path(origin).resolve().parent)

    searched: list[Path] = []
    for package_root in package_roots:
        for config_root in (
            package_root / ".mim" / "configs",
            package_root / "configs",
        ):
            candidate = config_root / relative_name
            searched.append(candidate)
            if candidate.is_file():
                return candidate

    locations_text = ", ".join(str(path) for path in searched) or "<none>"
    raise FileNotFoundError(
        f"cannot resolve {value}; searched installed package paths: "
        f"{locations_text}"
    )


class OfficialMMDetBackend:
    def __init__(self) -> None:
        try:
            import mmdet
        except ImportError as error:
            raise RuntimeError(
                "MMDetection is unavailable. Run the WSL environment bootstrap."
            ) from error
        self.version = str(mmdet.__version__)

    @staticmethod
    def _configuration(resolved: MMDetResolvedConfig):
        from mmengine.config import Config

        configuration = Config.fromfile(
            _resolve_base_config_path(resolved.base_config)
        )
        configuration.merge_from_dict(resolved.config)
        configuration.work_dir = str(resolved.work_dir)
        resolved.work_dir.mkdir(parents=True, exist_ok=True)
        configuration.dump(resolved.config_path)
        return configuration

    def train(
        self,
        resolved: MMDetResolvedConfig,
        *,
        resume: bool,
    ) -> Path:
        from mmengine.runner import Runner

        configuration = self._configuration(resolved)
        configuration.resume = resume
        runner = Runner.from_cfg(configuration)
        runner.train()
        best_candidates = sorted(
            resolved.work_dir.glob("best_coco_segm_mAP*.pth"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not best_candidates:
            best_candidates = sorted(
                resolved.work_dir.glob("iter_*.pth"),
                key=lambda path: path.stat().st_mtime_ns,
            )
        if not best_candidates:
            raise RuntimeError(
                f"MMDetection produced no checkpoint in {resolved.work_dir}"
            )
        return best_candidates[-1]

    def predict(
        self,
        checkpoint: Path,
        resolved: MMDetResolvedConfig,
        materialized: MaterializedDataset,
        *,
        split: str,
        shard_root: Path | None = None,
    ) -> PredictionBundle:
        from .mmdet_predictions import predict_mmdet_bundle

        if not resolved.config_path.is_file():
            self._configuration(resolved)
        return predict_mmdet_bundle(
            resolved.config_path,
            checkpoint,
            materialized,
            split=split,
            shard_root=shard_root,
        )


def train_validate_predict_mmdet(
    config: ExperimentConfig,
    materialized: MaterializedDataset,
    *,
    model_kind: str,
    variant_id: str | None = None,
    morphology_prior_path: str | Path | None = None,
    tau: float = 0.75,
    backend: MMDetBackend | None = None,
) -> MMDetRunResult:
    if materialized.format != "coco":
        raise ValueError("MMDetection training requires a COCO materialized dataset")
    if model_kind == "morphoroute":
        if variant_id is None:
            raise ValueError("MorphoRoute training requires variant_id")
        resolved = build_morphoroute_config(
            config,
            materialized,
            variant_id=variant_id,
            morphology_prior_path=morphology_prior_path,
            tau=tau,
        )
    else:
        builders = {
            "mask_rcnn": build_mask_rcnn_config,
            "mask2former": build_mask2former_config,
        }
        try:
            resolved = builders[model_kind](config, materialized)
        except KeyError as error:
            raise ValueError(
                f"unsupported MMDetection model kind: {model_kind}"
            ) from error
    active_backend = backend or OfficialMMDetBackend()
    prior_sha256 = None
    if morphology_prior_path is not None:
        prior_path = Path(morphology_prior_path)
        if prior_path.is_file():
            prior_sha256 = sha256_file(prior_path)
    cache_signature = training_cache_signature(
        model_kind=model_kind,
        variant_id=variant_id,
        seed=config.seed,
        dataset_sha256=materialized.fingerprint.sha256,
        morphology_prior_sha256=prior_sha256,
        framework_version=active_backend.version,
        parameters=config.parameters,
    )
    reuse_completed = bool(
        config.parameters.get("reuse_completed_training", False)
    )
    cache_directory = _training_cache_directory(config, cache_signature)
    if reuse_completed:
        cached = _load_completed_cache(
            cache_directory,
            signature=cache_signature,
            experiment_id=config.experiment_id,
            resolved=resolved,
        )
        if cached is not None:
            return cached
    resolved.work_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        resolved.work_dir / "mmdet_run_arguments.json",
        {
            "framework_version": active_backend.version,
            "base_config": resolved.base_config,
            "resolved_overrides": resolved.config,
            "dataset_fingerprint": materialized.fingerprint.sha256,
            "training_cache_signature": cache_signature,
        },
    )
    best_checkpoint = active_backend.train(
        resolved,
        resume=bool(config.parameters.get("resume", True)),
    )
    if not best_checkpoint.is_file():
        raise RuntimeError(f"best checkpoint does not exist: {best_checkpoint}")
    validation_predictions = active_backend.predict(
        best_checkpoint,
        resolved,
        materialized,
        split="D2",
    )
    if validation_predictions.split != "D2":
        if isinstance(active_backend, OfficialMMDetBackend):
            raise ValueError(
                "MMDetection backend returned a non-D2 validation bundle"
            )
        validation_predictions = replace(
            validation_predictions,
            split="D2",
            metadata={
                **dict(validation_predictions.metadata),
                "test_backend_split_normalized": True,
            },
        )
    validation_predictions = replace(
        validation_predictions,
        metadata={
            **dict(validation_predictions.metadata),
            "seed": config.seed,
            "profile": config.profile,
            "smoke_test": bool(config.parameters.get("smoke_test", False)),
            "variant_id": variant_id,
        },
    )
    predictions = active_backend.predict(
        best_checkpoint,
        resolved,
        materialized,
        split="D3",
    )
    if predictions.split != "D3":
        raise ValueError("MMDetection backend returned a non-D3 prediction bundle")
    predictions = replace(
        predictions,
        metadata={
            **dict(predictions.metadata),
            "seed": config.seed,
            "profile": config.profile,
            "smoke_test": bool(config.parameters.get("smoke_test", False)),
            "variant_id": variant_id,
        },
    )
    evaluation = evaluate_bundle(predictions)
    outputs = write_evaluation_outputs(
        evaluation,
        predictions,
        resolved.work_dir / "evaluation_D3",
    )
    result = MMDetRunResult(
        resolved_config=resolved,
        best_checkpoint=best_checkpoint,
        validation_predictions=validation_predictions,
        predictions=predictions,
        evaluation=evaluation,
        evaluation_outputs=outputs,
    )
    if reuse_completed:
        _write_completed_cache(
            cache_directory,
            signature=cache_signature,
            result=result,
        )
    return result
