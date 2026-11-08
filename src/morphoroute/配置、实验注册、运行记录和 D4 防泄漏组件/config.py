from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


VALID_PROFILES = frozenset(
    {
        "sop_full",
        "local_6gb",
        "local_6gb_fallback",
        "autodl_5090_32gb",
    }
)
VALID_SPLITS = frozenset({"D1", "D2", "D3", "D4"})


def _canonical_path(path: Path) -> str:
    return path.as_posix()


def _contains(parent: Path, child: Path) -> bool:
    parent_text = os.path.normcase(str(parent.resolve(strict=False)))
    child_text = os.path.normcase(str(child.resolve(strict=False)))
    try:
        return os.path.commonpath([parent_text, child_text]) == parent_text
    except ValueError:
        return False


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    profile: str
    data_root: Path
    output_root: Path
    seed: int
    allowed_splits: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        required = {
            "experiment_id",
            "profile",
            "data_root",
            "output_root",
            "seed",
            "allowed_splits",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"missing configuration fields: {', '.join(missing)}")

        profile = str(value["profile"])
        if profile not in VALID_PROFILES:
            raise ValueError(
                f"profile must be one of {sorted(VALID_PROFILES)}, got {profile!r}"
            )

        allowed_splits = tuple(str(item) for item in value["allowed_splits"])
        invalid_splits = sorted(set(allowed_splits).difference(VALID_SPLITS))
        if invalid_splits:
            raise ValueError(
                f"allowed_splits contains unsupported values: {invalid_splits}"
            )

        seed = int(value["seed"])
        if seed < 0:
            raise ValueError("seed must be non-negative")

        parameters = value.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("parameters must be a mapping")

        return cls(
            experiment_id=str(value["experiment_id"]),
            profile=profile,
            data_root=Path(str(value["data_root"])),
            output_root=Path(str(value["output_root"])),
            seed=seed,
            allowed_splits=allowed_splits,
            parameters=dict(parameters),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("configuration JSON root must be an object")
        return cls.from_mapping(payload)

    def validate(self) -> None:
        if _contains(self.data_root, self.output_root):
            raise ValueError("output_root must not be inside the read-only data_root")
        if not self.experiment_id.startswith("EXP-"):
            raise ValueError("experiment_id must begin with EXP-")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "profile": self.profile,
            "data_root": _canonical_path(self.data_root),
            "output_root": _canonical_path(self.output_root),
            "seed": self.seed,
            "allowed_splits": list(self.allowed_splits),
            "parameters": dict(self.parameters),
        }
