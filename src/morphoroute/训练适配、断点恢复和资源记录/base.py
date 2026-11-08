from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from morphoroute.core.config import ExperimentConfig


@dataclass(frozen=True)
class TrainingPlan:
    backend: str
    experiment_id: str
    model: str
    seed: int
    train_split: str
    validation_split: str
    training_inputs: tuple[str, ...]
    input_size: int
    max_updates: int
    physical_batch: int
    effective_batch: int
    gradient_accumulation: int
    amp: bool
    dataset_config: Path
    output_dir: Path
    command: tuple[str, ...]
    framework_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_config"] = str(self.dataset_config)
        payload["output_dir"] = str(self.output_dir)
        payload["command"] = list(self.command)
        payload["training_inputs"] = list(self.training_inputs)
        return payload


def plan_common(
    config: ExperimentConfig,
) -> dict[str, Any]:
    parameters = config.parameters
    physical_batch = int(parameters.get("physical_batch", 1))
    effective_batch = int(parameters.get("effective_batch", 8))
    if physical_batch <= 0 or effective_batch <= 0:
        raise ValueError("batch sizes must be positive")
    if effective_batch % physical_batch != 0:
        raise ValueError("effective_batch must be divisible by physical_batch")
    return {
        "model": str(parameters["model"]),
        "input_size": int(parameters.get("input_size", 1024)),
        "max_updates": int(parameters.get("max_updates", 50_000)),
        "physical_batch": physical_batch,
        "effective_batch": effective_batch,
        "gradient_accumulation": effective_batch // physical_batch,
        "amp": bool(parameters.get("amp", True)),
    }


class TrainingAdapter(ABC):
    @abstractmethod
    def build_plan(
        self,
        config: ExperimentConfig,
        derived_dataset_root: str | Path,
    ) -> TrainingPlan:
        raise NotImplementedError

    @abstractmethod
    def train(self, plan: TrainingPlan, *, resume: bool = False) -> Any:
        raise NotImplementedError
