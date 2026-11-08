from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from morphoroute.core.io import atomic_write_json


def choose_routing_threshold(
    candidate_metrics: Iterable[Mapping[str, object]],
    *,
    metric: str,
) -> float:
    """Select maximum D2 score, preferring sparser lower tau on ties."""
    rows = [dict(row) for row in candidate_metrics]
    if not rows:
        raise ValueError("routing-threshold selection requires candidates")
    for row in rows:
        if "threshold" not in row or metric not in row:
            raise ValueError(
                f"each threshold candidate requires threshold and {metric}"
            )
    selected = sorted(
        rows,
        key=lambda row: (
            -float(row[metric]),
            float(row["threshold"]),
        ),
    )[0]
    return float(selected["threshold"])


try:
    from mmengine.hooks import Hook
    from mmengine.model import is_model_wrapper
    from mmdet.registry import HOOKS
except ImportError:
    Hook = None
    HOOKS = None


if HOOKS is not None:

    @HOOKS.register_module(force=True)
    class RoutingThresholdSelectionHook(Hook):
        """Freeze the low-confidence routing threshold using D2 only."""

        priority = "VERY_HIGH"

        def __init__(
            self,
            *,
            trigger_iteration: int,
            candidates=(0.50, 0.60, 0.70, 0.75, 0.80, 0.90),
            metric: str = "coco/segm_mAP",
            output_filename: str = "routing_threshold_selection_D2.json",
            select_threshold: bool = True,
        ) -> None:
            self.trigger_iteration = int(trigger_iteration)
            self.candidates = tuple(float(value) for value in candidates)
            self.metric = str(metric)
            self.output_filename = str(output_filename)
            self.select_threshold = bool(select_threshold)
            self._completed = False

        @staticmethod
        def _model(runner):
            return (
                runner.model.module
                if is_model_wrapper(runner.model)
                else runner.model
            )

        @staticmethod
        def _reset_best_checkpoint_tracking(runner) -> None:
            hooks = getattr(
                runner,
                "hooks",
                getattr(runner, "_hooks", ()),
            )
            for hook in hooks:
                indicators = getattr(hook, "key_indicators", ())
                rules = getattr(hook, "rules", ())
                initial_values = getattr(hook, "init_value_map", {})
                for index, (indicator, rule) in enumerate(
                    zip(indicators, rules)
                ):
                    key = (
                        "best_score"
                        if len(indicators) == 1
                        else f"best_score_{indicator}"
                    )
                    runner.message_hub.update_info(
                        key,
                        initial_values[rule],
                    )

        def _apply_existing(self, runner, path: Path) -> bool:
            if not path.is_file():
                return False
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "selected_threshold" in payload:
                selected = float(payload["selected_threshold"])
                self._model(
                    runner
                ).panoptic_head.sparse_decoder.threshold = selected
            if int(runner.iter) == self.trigger_iteration:
                self._reset_best_checkpoint_tracking(runner)
            self._completed = True
            return True

        def before_train_iter(
            self,
            runner,
            batch_idx: int,
            data_batch=None,
        ) -> None:
            del batch_idx, data_batch
            if self._completed or int(runner.iter) < self.trigger_iteration:
                return
            output_path = Path(runner.work_dir) / self.output_filename
            if self._apply_existing(runner, output_path):
                return
            if int(runner.iter) > self.trigger_iteration:
                raise RuntimeError(
                    "training passed the sparse-stage boundary without a "
                    "frozen D2 routing-threshold artifact"
                )
            model = self._model(runner)
            decoder = model.panoptic_head.sparse_decoder
            if not self.select_threshold:
                self._reset_best_checkpoint_tracking(runner)
                atomic_write_json(
                    output_path,
                    {
                        "schema_version": 1,
                        "trigger_iteration": self.trigger_iteration,
                        "selected_threshold": decoder.threshold,
                        "selection_split": "pre_registered_fixed",
                        "status": "stage_C_checkpoint_selection_started",
                    },
                )
                self._completed = True
                return
            rows = []
            for threshold in self.candidates:
                decoder.threshold = threshold
                metrics = runner.val_loop.run()
                if self.metric not in metrics:
                    raise KeyError(
                        f"D2 evaluator did not return {self.metric}; "
                        f"available keys: {sorted(metrics)}"
                    )
                rows.append(
                    {
                        "threshold": threshold,
                        self.metric: float(metrics[self.metric]),
                    }
                )
            selected = choose_routing_threshold(rows, metric=self.metric)
            decoder.threshold = selected
            model.train()
            self._reset_best_checkpoint_tracking(runner)
            atomic_write_json(
                output_path,
                {
                    "schema_version": 1,
                    "selection_split": "D2",
                    "trigger_iteration": self.trigger_iteration,
                    "metric": self.metric,
                    "tie_break": "lower_threshold_for_sparser_inference",
                    "candidates": rows,
                    "selected_threshold": selected,
                    "status": "frozen_for_stage_C",
                },
            )
            self._completed = True
