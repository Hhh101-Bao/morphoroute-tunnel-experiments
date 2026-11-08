from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from morphoroute.core.io import sha256_json


@dataclass(frozen=True)
class BudgetGatePolicy:
    enabled: bool
    start_update: int
    metric: str
    patience: int
    min_delta: float

    def __post_init__(self) -> None:
        if self.start_update <= 0:
            raise ValueError("budget_gate.start_update must be positive")
        if self.patience <= 0:
            raise ValueError("budget_gate.patience must be positive")
        if self.min_delta < 0:
            raise ValueError("budget_gate.min_delta must be non-negative")
        if self.metric != "coco/segm_mAP":
            raise ValueError(
                "budget gate must use the pre-registered D2 coco/segm_mAP metric"
            )


@dataclass(frozen=True)
class BudgetGateDecision:
    update: int
    value: float
    best_value: float
    bad_validations: int
    should_stop: bool


class BudgetGateMonitor:
    """State machine used by the MMEngine hook and by unit tests."""

    def __init__(self, policy: BudgetGatePolicy) -> None:
        self.policy = policy
        self.best_value = float("-inf")
        self.bad_validations = 0

    def observe(self, *, update: int, value: float) -> BudgetGateDecision:
        update = int(update)
        value = float(value)
        improved = value > self.best_value + self.policy.min_delta
        if improved:
            self.best_value = value
            self.bad_validations = 0
        elif update >= self.policy.start_update:
            self.bad_validations += 1
        else:
            # Pre-gate validation establishes the historical best only. It
            # must not consume patience that is reserved for gated checks.
            self.bad_validations = 0
        should_stop = (
            self.policy.enabled
            and update >= self.policy.start_update
            and self.bad_validations >= self.policy.patience
        )
        return BudgetGateDecision(
            update=update,
            value=value,
            best_value=self.best_value,
            bad_validations=self.bad_validations,
            should_stop=should_stop,
        )


def budget_gate_policy(parameters: Mapping[str, Any]) -> BudgetGatePolicy | None:
    payload = parameters.get("budget_gate")
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("budget_gate must be an object")
    policy = BudgetGatePolicy(
        enabled=bool(payload.get("enabled", True)),
        start_update=int(payload.get("start_update", 30_000)),
        metric=str(payload.get("metric", "coco/segm_mAP")),
        patience=int(payload.get("patience", 2)),
        min_delta=float(payload.get("min_delta", 0.001)),
    )
    if bool(parameters.get("release_budget_gate", False)):
        return None
    return policy if policy.enabled else None


_NON_TRAINING_PARAMETER_KEYS = frozenset(
    {
        "run_name",
        "variant_id",
        "model",
        "result_status",
        "verify_hashes",
        "execute_training",
        "screening_seeds",
        "confirmation_seeds",
        "confirmation_variants",
        "seeds",
        "reuse_completed_training",
        "release_budget_gate_variants",
    }
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def training_cache_signature(
    *,
    model_kind: str,
    variant_id: str | None,
    seed: int,
    dataset_sha256: str,
    morphology_prior_sha256: str | None,
    framework_version: str,
    parameters: Mapping[str, Any],
) -> str:
    training_parameters = {
        str(key): _json_value(value)
        for key, value in parameters.items()
        if key not in _NON_TRAINING_PARAMETER_KEYS
    }
    return sha256_json(
        {
            "schema_version": 1,
            "model_kind": str(model_kind),
            "variant_id": variant_id,
            "seed": int(seed),
            "dataset_sha256": str(dataset_sha256),
            "morphology_prior_sha256": morphology_prior_sha256,
            "framework_version": str(framework_version),
            "parameters": training_parameters,
        }
    )
