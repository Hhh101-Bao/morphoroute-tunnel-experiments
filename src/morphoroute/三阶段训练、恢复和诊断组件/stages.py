from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrainingStage(str, Enum):
    A = "A_dense_warmup"
    B = "B_route_supervision"
    C = "C_sparse_finetune"


@dataclass(frozen=True)
class StageSchedule:
    warmup_end: int
    sparse_start: int
    max_updates: int

    def __post_init__(self) -> None:
        if not 0 <= self.warmup_end <= self.sparse_start <= self.max_updates:
            raise ValueError(
                "stage boundaries must satisfy "
                "0 <= warmup_end <= sparse_start <= max_updates"
            )

    def stage_at(self, update: int) -> TrainingStage:
        if not 0 <= update <= self.max_updates:
            raise ValueError(
                f"update must be in [0, {self.max_updates}], got {update}"
            )
        if update < self.warmup_end:
            return TrainingStage.A
        if update < self.sparse_start:
            return TrainingStage.B
        return TrainingStage.C

    def controls_at(self, update: int) -> dict[str, bool]:
        stage = self.stage_at(update)
        return {
            "route_supervision": stage in {TrainingStage.B, TrainingStage.C},
            "experts_enabled": stage in {TrainingStage.B, TrainingStage.C},
            "sparse_routing": stage is TrainingStage.C,
        }
