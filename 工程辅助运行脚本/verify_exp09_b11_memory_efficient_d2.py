from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner

import morphoroute.mmdet.models  # noqa: F401
from morphoroute.runtime.pcgrad_fp32 import (
    POLICY_REVISION,
    apply_b11_true_fp32_config,
)
from morphoroute.runtime.strict_fp32 import configure_torch_strict_fp32


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: verify_exp09_b11_memory_efficient_d2.py "
            "CONFIG CHECKPOINT"
        )

    config_path = Path(sys.argv[1]).resolve()
    checkpoint_path = Path(sys.argv[2]).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    audit = configure_torch_strict_fp32(torch)
    if audit["cuda_matmul_allow_tf32"] or audit["cudnn_allow_tf32"]:
        raise RuntimeError(f"strict FP32 configuration failed: {audit}")

    cfg = Config.fromfile(str(config_path))
    apply_b11_true_fp32_config(cfg, variant_id="B11")

    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    cfg.test_cfg = dict(type="TestLoop")
    cfg.train_cfg = None
    cfg.train_dataloader = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None
    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.custom_hooks = []
    cfg.load_from = str(checkpoint_path)
    cfg.resume = False
    cfg.work_dir = str(
        checkpoint_path.parent / "memory_efficient_d2_verification"
    )

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    print(f"EXP09_B11_MEMORY_EFFICIENT_D2_METRICS={metrics}", flush=True)
    print(
        "EXP09_B11_MEMORY_EFFICIENT_D2=PASSED "
        f"revision={POLICY_REVISION}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
