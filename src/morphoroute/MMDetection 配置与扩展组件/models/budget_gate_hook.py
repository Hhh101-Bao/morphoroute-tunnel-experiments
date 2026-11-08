from __future__ import annotations

from pathlib import Path
from typing import Mapping

from morphoroute.core.io import atomic_write_json
from morphoroute.runtime.acceleration import BudgetGateMonitor, BudgetGatePolicy


try:
    from mmengine.hooks import Hook
    from mmdet.registry import HOOKS
except ImportError:
    Hook = None
    HOOKS = None


if HOOKS is not None:

    @HOOKS.register_module(force=True)
    class D2BudgetGateHook(Hook):
        """Stop a plateaued run after a D2-only optimizer-update gate."""

        priority = "LOW"

        def __init__(
            self,
            *,
            start_iteration: int,
            update_divisor: int,
            metric: str = "coco/segm_mAP",
            patience: int = 2,
            min_delta: float = 0.001,
            output_filename: str = "budget_gate_D2.json",
        ) -> None:
            if update_divisor <= 0:
                raise ValueError("update_divisor must be positive")
            self.start_iteration = int(start_iteration)
            self.update_divisor = int(update_divisor)
            self.metric = str(metric)
            self.output_filename = str(output_filename)
            self.monitor = BudgetGateMonitor(
                BudgetGatePolicy(
                    enabled=True,
                    start_update=max(1, self.start_iteration // self.update_divisor),
                    metric=self.metric,
                    patience=int(patience),
                    min_delta=float(min_delta),
                )
            )
            self.history: list[dict[str, object]] = []

        def after_val_epoch(
            self,
            runner,
            metrics: Mapping[str, float] | None = None,
        ) -> None:
            values = dict(metrics or {})
            if self.metric not in values:
                raise KeyError(
                    f"D2 budget gate requires {self.metric}; "
                    f"available keys: {sorted(values)}"
                )
            iteration = int(runner.iter) + 1
            update = max(1, iteration // self.update_divisor)
            decision = self.monitor.observe(
                update=update,
                value=float(values[self.metric]),
            )
            self.history.append(
                {
                    "optimizer_update": decision.update,
                    self.metric: decision.value,
                    "best_value": decision.best_value,
                    "bad_validations": decision.bad_validations,
                }
            )
            atomic_write_json(
                Path(runner.work_dir) / self.output_filename,
                {
                    "schema_version": 1,
                    "selection_split": "D2",
                    "metric": self.metric,
                    "start_update": self.monitor.policy.start_update,
                    "patience": self.monitor.policy.patience,
                    "min_delta": self.monitor.policy.min_delta,
                    "history": self.history,
                    "status": (
                        "stopped_at_budget_gate"
                        if decision.should_stop
                        else "monitoring"
                    ),
                    "resume_instruction": (
                        "set release_budget_gate=true and rerun with resume=true"
                        if decision.should_stop
                        else None
                    ),
                },
            )
            if decision.should_stop:
                runner.train_loop.stop_training = True
                runner.logger.info(
                    "D2 budget gate stopped training at optimizer update %d",
                    decision.update,
                )
