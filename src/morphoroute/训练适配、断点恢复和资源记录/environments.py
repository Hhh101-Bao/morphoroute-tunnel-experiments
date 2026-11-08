from __future__ import annotations

from enum import Enum


class EnvironmentKind(str, Enum):
    YOLO = "yolo"
    MAIN = "main"
    STATS = "stats"


YOLO_EXPERIMENTS = frozenset({"EXP-02", "EXP-03"})
MAIN_EXPERIMENTS = frozenset(
    {
        "EXP-04",
        "EXP-05",
        "EXP-06",
        "EXP-07",
        "EXP-08A",
        "EXP-08B",
        "EXP-08D",
        "EXP-09",
        "EXP-10",
        "EXP-11",
        "EXP-14",
    }
)
STATS_EXPERIMENTS = frozenset(
    {
        "EXP-00",
        "EXP-01",
        "EXP-08C",
        "EXP-12",
        "EXP-13",
        "EXP-15",
    }
)


class EnvironmentResolver:
    @staticmethod
    def for_experiment(experiment_id: str) -> EnvironmentKind:
        normalized = experiment_id.split(":", 1)[0].upper()
        if normalized in YOLO_EXPERIMENTS:
            return EnvironmentKind.YOLO
        if normalized in MAIN_EXPERIMENTS:
            return EnvironmentKind.MAIN
        if normalized in STATS_EXPERIMENTS:
            return EnvironmentKind.STATS
        raise ValueError(f"unknown experiment: {experiment_id}")
