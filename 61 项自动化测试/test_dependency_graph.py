from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class DependencyGraphTests(unittest.TestCase):
    def test_every_execution_target_has_an_acyclic_unique_plan(self):
        from morphoroute.core.dependencies import ExecutionGraph
        from morphoroute.core.registry import EXPERIMENTS

        graph = ExecutionGraph.from_registry()
        targets = tuple(EXPERIMENTS) + (
            "EXP-12:prefreeze",
            "EXP-12:external",
        )
        for target in targets:
            with self.subTest(target=target):
                plan = graph.plan(target)
                self.assertEqual(len(plan), len(set(plan)))
                expected_last = (
                    "EXP-12:external" if target == "EXP-12" else target
                )
                self.assertEqual(plan[-1], expected_last)

    def test_exp11_requires_freeze_not_ablation_directly(self):
        from morphoroute.core.dependencies import (
            ArtifactRequirement,
            ExecutionGraph,
        )

        graph = ExecutionGraph.from_registry()

        self.assertEqual(
            graph.requirements("EXP-11"),
            (ArtifactRequirement("EXP-10", "complete"),),
        )

    def test_exp12_stages_break_the_old_cycle(self):
        from morphoroute.core.dependencies import ExecutionGraph

        graph = ExecutionGraph.from_registry()

        self.assertEqual(
            graph.plan("EXP-12:prefreeze")[-1],
            "EXP-12:prefreeze",
        )
        self.assertEqual(
            graph.plan("EXP-12:external")[-2:],
            ["EXP-11", "EXP-12:external"],
        )

    def test_auto_prerequisite_executor_skips_completed_manifests(self):
        from morphoroute.core.dependencies import (
            ExecutionGraph,
            ensure_prerequisites,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            completed = output_root / "experiments" / "EXP-00"
            completed.mkdir(parents=True)
            (completed / "run_manifest.json").write_text(
                '{"status": "completed"}',
                encoding="utf-8",
            )
            executed = []

            ensure_prerequisites(
                "EXP-01",
                output_root,
                executor=executed.append,
                graph=ExecutionGraph.from_registry(),
            )

            self.assertEqual(executed, [])


if __name__ == "__main__":
    unittest.main()
