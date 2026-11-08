from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json
from morphoroute.data.materialize import MaterializedDataset
from morphoroute.evaluation.pipeline import (
    EvaluationOutputPaths,
    EvaluationResult,
    evaluate_bundle,
    write_evaluation_outputs,
)
from morphoroute.evaluation.schema import PredictionBundle

from .base import TrainingAdapter, TrainingPlan, plan_common


class UltralyticsBackend(Protocol):
    version: str

    def train(
        self,
        plan: TrainingPlan,
        *,
        epochs: int,
        resume: bool,
    ) -> Path: ...

    def predict(
        self,
        checkpoint: Path,
        materialized: MaterializedDataset,
        *,
        split: str,
    ) -> PredictionBundle: ...


@dataclass(frozen=True)
class UltralyticsRunResult:
    training_plan: TrainingPlan
    epochs: int
    best_checkpoint: Path
    predictions: PredictionBundle
    evaluation: EvaluationResult
    evaluation_outputs: EvaluationOutputPaths


class UltralyticsAdapter(TrainingAdapter):
    def build_plan(
        self,
        config: ExperimentConfig,
        derived_dataset_root: str | Path,
    ) -> TrainingPlan:
        common = plan_common(config)
        root = Path(derived_dataset_root)
        dataset_config = root / "data.yaml"
        output_dir = (
            config.output_root
            / "experiments"
            / config.experiment_id
            / config.profile
            / f"seed_{config.seed}"
            / "runs"
        )
        command = (
            "yolo",
            "segment",
            "train",
            f"model={common['model']}",
            f"data={dataset_config}",
            f"imgsz={common['input_size']}",
            f"batch={common['physical_batch']}",
            f"seed={config.seed}",
            f"amp={str(common['amp']).lower()}",
            f"project={output_dir}",
        )
        return TrainingPlan(
            backend="ultralytics",
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
            framework_config={
                "max_updates": common["max_updates"],
                "accumulate": common["gradient_accumulation"],
                "checkpoint_selection_split": "D2",
                "evaluation_splits": ["D2", "D3"],
            },
        )

    def train(self, plan: TrainingPlan, *, resume: bool = False) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is unavailable. Activate ENV-YOLO in WSL."
            ) from error
        model = YOLO(plan.model)
        epochs = int(plan.framework_config.get("epochs", 100))
        return model.train(
            data=str(plan.dataset_config),
            imgsz=plan.input_size,
            batch=plan.physical_batch,
            seed=plan.seed,
            amp=plan.amp,
            project=str(plan.output_dir),
            epochs=epochs,
            resume=resume,
        )


class OfficialUltralyticsBackend:
    def __init__(self) -> None:
        try:
            import ultralytics
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is unavailable. Run the WSL environment bootstrap."
            ) from error
        self._ultralytics = ultralytics
        self.version = str(ultralytics.__version__)

    def train(
        self,
        plan: TrainingPlan,
        *,
        epochs: int,
        resume: bool,
    ) -> Path:
        last_checkpoint = (
            plan.output_dir / "train" / "weights" / "last.pt"
        )
        resume_from_checkpoint = bool(resume and last_checkpoint.is_file())
        model_source = (
            last_checkpoint if resume_from_checkpoint else plan.model
        )
        model = self._ultralytics.YOLO(str(model_source))
        result = model.train(
            data=str(plan.dataset_config),
            imgsz=plan.input_size,
            batch=plan.physical_batch,
            nbs=plan.effective_batch,
            epochs=epochs,
            seed=plan.seed,
            amp=plan.amp,
            project=str(plan.output_dir),
            name="train",
            exist_ok=True,
            device=0,
            workers=2,
            patience=0,
            save=True,
            plots=True,
            resume=resume_from_checkpoint,
        )
        save_dir = Path(getattr(result, "save_dir", plan.output_dir / "train"))
        best = save_dir / "weights" / "best.pt"
        if not best.is_file():
            last = save_dir / "weights" / "last.pt"
            if not last.is_file():
                raise RuntimeError(
                    f"Ultralytics did not produce best.pt or last.pt in {save_dir}"
                )
            best = last
        return best

    def predict(
        self,
        checkpoint: Path,
        materialized: MaterializedDataset,
        *,
        split: str,
    ) -> PredictionBundle:
        from .ultralytics_predictions import convert_ultralytics_results

        split_directory = {
            "D2": "val",
            "D3": "test",
            "D4": "external",
        }[split]
        model = self._ultralytics.YOLO(str(checkpoint))
        results = model.predict(
            source=str(materialized.root / "images" / split_directory),
            imgsz=640,
            device=0,
            save=False,
            stream=False,
            retina_masks=True,
            verbose=False,
        )
        return convert_ultralytics_results(
            results,
            materialized,
            split=split,
            experiment_id=checkpoint.parents[3].name
            if len(checkpoint.parents) >= 4
            else "unknown",
        )


def _training_epochs(plan: TrainingPlan, materialized: MaterializedDataset) -> int:
    train_directory = materialized.root / "images" / "train"
    train_count = sum(path.is_file() for path in train_directory.iterdir())
    if train_count <= 0:
        raise ValueError("materialized YOLO D1 training split is empty")
    updates_per_epoch = max(1, math.ceil(train_count / plan.effective_batch))
    return max(1, math.ceil(plan.max_updates / updates_per_epoch))


def train_validate_predict_yolo(
    config: ExperimentConfig,
    materialized: MaterializedDataset,
    *,
    backend: UltralyticsBackend | None = None,
) -> UltralyticsRunResult:
    if materialized.format != "yolo" or materialized.data_yaml is None:
        raise ValueError("YOLO training requires a YOLO materialized dataset")
    plan = UltralyticsAdapter().build_plan(config, materialized.root)
    epochs = _training_epochs(plan, materialized)
    active_backend = backend or OfficialUltralyticsBackend()
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        plan.output_dir / "ultralytics_run_arguments.json",
        {
            "framework_version": active_backend.version,
            "epochs": epochs,
            "optimizer_updates": plan.max_updates,
            "training_plan": plan.to_dict(),
            "dataset_fingerprint": materialized.fingerprint.sha256,
        },
    )
    best_checkpoint = active_backend.train(
        plan,
        epochs=epochs,
        resume=bool(config.parameters.get("resume", True)),
    )
    if not best_checkpoint.is_file():
        raise RuntimeError(f"best checkpoint does not exist: {best_checkpoint}")
    predictions = active_backend.predict(
        best_checkpoint,
        materialized,
        split="D3",
    )
    if predictions.split != "D3":
        raise ValueError("Ultralytics backend returned predictions from a non-D3 split")
    evaluation = evaluate_bundle(predictions)
    outputs = write_evaluation_outputs(
        evaluation,
        predictions,
        plan.output_dir / "evaluation_D3",
    )
    return UltralyticsRunResult(
        training_plan=plan,
        epochs=epochs,
        best_checkpoint=best_checkpoint,
        predictions=predictions,
        evaluation=evaluation,
        evaluation_outputs=outputs,
    )
