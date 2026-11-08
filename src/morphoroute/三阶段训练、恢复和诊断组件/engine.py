from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .stages import StageSchedule, TrainingStage


@dataclass(frozen=True)
class UpdateResult:
    update: int
    stage: TrainingStage
    losses: dict[str, float]


class TrainingEngine:
    def __init__(
        self,
        schedule: StageSchedule,
        *,
        apply_controls: Callable[[TrainingStage, dict[str, bool]], None],
    ) -> None:
        self.schedule = schedule
        self.apply_controls = apply_controls

    def run(
        self,
        batches: Iterable[Any],
        *,
        start_update: int,
        update_step: Callable[[Any, int, TrainingStage], dict[str, float]],
        on_update: Callable[[UpdateResult], None] | None = None,
    ) -> list[UpdateResult]:
        results = []
        previous_stage: TrainingStage | None = None
        for offset, batch in enumerate(batches):
            update = start_update + offset
            if update >= self.schedule.max_updates:
                break
            stage = self.schedule.stage_at(update)
            if stage is not previous_stage:
                self.apply_controls(stage, self.schedule.controls_at(update))
                previous_stage = stage
            result = UpdateResult(
                update=update,
                stage=stage,
                losses=update_step(batch, update, stage),
            )
            results.append(result)
            if on_update is not None:
                on_update(result)
        return results
