from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .registry import EXPERIMENTS


@dataclass(frozen=True)
class ArtifactRequirement:
    producer: str
    state: str = "complete"


@dataclass(frozen=True)
class ExecutionNode:
    node_id: str
    requirements: tuple[ArtifactRequirement, ...]


class ExecutionGraph:
    def __init__(
        self,
        nodes: Mapping[str, ExecutionNode],
        *,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._nodes = dict(nodes)
        self._aliases = dict(aliases or {})

    @classmethod
    def from_registry(cls) -> "ExecutionGraph":
        requirements: dict[str, tuple[str, ...]] = {
            experiment_id: spec.prerequisites
            for experiment_id, spec in EXPERIMENTS.items()
            if experiment_id != "EXP-12"
        }
        requirements.update(
            {
                "EXP-10": (
                    "EXP-03",
                    "EXP-04",
                    "EXP-05",
                    "EXP-06",
                    "EXP-07",
                    "EXP-09",
                    "EXP-12:prefreeze",
                ),
                "EXP-11": ("EXP-10",),
                "EXP-12:prefreeze": ("EXP-09",),
                "EXP-12:external": ("EXP-12:prefreeze", "EXP-11"),
                "EXP-15": (
                    "EXP-11",
                    "EXP-12:external",
                    "EXP-13",
                    "EXP-14",
                ),
            }
        )
        nodes = {
            node_id: ExecutionNode(
                node_id=node_id,
                requirements=tuple(
                    ArtifactRequirement(producer)
                    for producer in producers
                ),
            )
            for node_id, producers in requirements.items()
        }
        return cls(nodes, aliases={"EXP-12": "EXP-12:external"})

    def resolve(self, node_id: str) -> str:
        return self._aliases.get(node_id, node_id)

    def requirements(
        self,
        node_id: str,
    ) -> tuple[ArtifactRequirement, ...]:
        resolved = self.resolve(node_id)
        try:
            return self._nodes[resolved].requirements
        except KeyError as error:
            raise ValueError(f"unknown execution node: {node_id}") from error

    def plan(self, target: str) -> list[str]:
        resolved_target = self.resolve(target)
        ordered: list[str] = []
        permanent: set[str] = set()
        active: list[str] = []

        def visit(node_id: str) -> None:
            resolved = self.resolve(node_id)
            if resolved in permanent:
                return
            if resolved in active:
                cycle = " -> ".join((*active, resolved))
                raise ValueError(f"cyclic experiment dependency: {cycle}")
            if resolved not in self._nodes:
                raise ValueError(f"unknown execution node: {resolved}")
            active.append(resolved)
            for requirement in self._nodes[resolved].requirements:
                visit(requirement.producer)
            active.pop()
            permanent.add(resolved)
            ordered.append(resolved)

        visit(resolved_target)
        return ordered


def manifest_path(output_root: str | Path, node_id: str) -> Path:
    root = Path(output_root) / "experiments"
    if ":" in node_id:
        experiment_id, stage = node_id.split(":", 1)
        return root / experiment_id / f"{stage}_manifest.json"
    return root / node_id / "run_manifest.json"


def is_complete(output_root: str | Path, node_id: str) -> bool:
    path = manifest_path(output_root, node_id)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    status = str(payload.get("status", ""))
    return (
        status == "completed"
        or status.startswith("completed_")
        or status.endswith("_completed")
    )


def ensure_prerequisites(
    target: str,
    output_root: str | Path,
    *,
    executor: Callable[[str], object],
    graph: ExecutionGraph | None = None,
) -> None:
    active_graph = graph or ExecutionGraph.from_registry()
    plan = active_graph.plan(target)
    for node_id in plan[:-1]:
        if not is_complete(output_root, node_id):
            executor(node_id)
            if not is_complete(output_root, node_id):
                raise RuntimeError(
                    f"prerequisite executor did not complete {node_id}"
                )
