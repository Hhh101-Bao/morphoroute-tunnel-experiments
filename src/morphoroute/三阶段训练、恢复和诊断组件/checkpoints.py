from __future__ import annotations

import hashlib
import os
import pickle
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .stages import TrainingStage


def _rng_hash(
    python_rng: object,
    numpy_rng: tuple,
    torch_rng: torch.Tensor,
    cuda_rng: list[torch.Tensor],
) -> str:
    return hashlib.sha256(
        pickle.dumps(
            (
                python_rng,
                numpy_rng,
                torch_rng.cpu().numpy(),
                [state.cpu().numpy() for state in cuda_rng],
            ),
            protocol=5,
        )
    ).hexdigest()


@dataclass(frozen=True)
class TrainingState:
    update: int
    stage: TrainingStage
    best_d2_score: float | None
    python_rng: object
    numpy_rng: tuple
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor]
    rng_hash: str

    @classmethod
    def capture(
        cls,
        *,
        update: int,
        stage: TrainingStage,
        best_d2_score: float | None,
    ) -> "TrainingState":
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.random.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        )
        return cls(
            update=int(update),
            stage=stage,
            best_d2_score=best_d2_score,
            python_rng=python_rng,
            numpy_rng=numpy_rng,
            torch_rng=torch_rng,
            cuda_rng=cuda_rng,
            rng_hash=_rng_hash(
                python_rng,
                numpy_rng,
                torch_rng,
                cuda_rng,
            ),
        )

    def restore_rng(self) -> None:
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.random.set_rng_state(self.torch_rng)
        if self.cuda_rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_rng)


def save_training_checkpoint(
    path: str | Path,
    *,
    state: TrainingState,
    model,
    optimizer,
    scaler,
    scheduler=None,
    extra: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "training_state": state,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "extra": dict(extra or {}),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scaler,
    scheduler=None,
) -> TrainingState:
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported training checkpoint schema")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    state = payload["training_state"]
    if not isinstance(state, TrainingState):
        raise TypeError("checkpoint training_state has an invalid type")
    calculated_hash = _rng_hash(
        state.python_rng,
        state.numpy_rng,
        state.torch_rng,
        state.cuda_rng,
    )
    if calculated_hash != state.rng_hash:
        raise ValueError("checkpoint RNG state hash mismatch")
    state.restore_rng()
    return state
