from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TrainingEngineTests(unittest.TestCase):
    def test_schedule_transitions_a_to_b_to_c(self):
        from morphoroute.training.stages import StageSchedule, TrainingStage

        schedule = StageSchedule(
            warmup_end=1000,
            sparse_start=40000,
            max_updates=50000,
        )

        self.assertIs(schedule.stage_at(0), TrainingStage.A)
        self.assertIs(schedule.stage_at(999), TrainingStage.A)
        self.assertIs(schedule.stage_at(1000), TrainingStage.B)
        self.assertIs(schedule.stage_at(40000), TrainingStage.C)
        with self.assertRaises(ValueError):
            schedule.stage_at(50001)

    def test_resume_roundtrip_restores_update_stage_and_rng(self):
        from morphoroute.training.checkpoints import (
            TrainingState,
            load_training_checkpoint,
            save_training_checkpoint,
        )
        from morphoroute.training.stages import TrainingStage

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            original = TrainingState.capture(
                update=1234,
                stage=TrainingStage.B,
                best_d2_score=0.51,
            )

            save_training_checkpoint(
                path,
                state=original,
                model=model,
                optimizer=optimizer,
                scaler=None,
            )
            restored = load_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=None,
            )

            self.assertEqual(restored.update, 1234)
            self.assertIs(restored.stage, TrainingStage.B)
            self.assertEqual(restored.rng_hash, original.rng_hash)


if __name__ == "__main__":
    unittest.main()
