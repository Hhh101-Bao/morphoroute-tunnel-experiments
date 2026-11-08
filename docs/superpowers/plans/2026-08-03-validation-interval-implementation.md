# MMDetection Validation Interval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make formal MMDetection runs validate and checkpoint every 5000 optimizer updates while preserving short smoke-test validation and checkpoint resume.

**Architecture:** Keep optimizer-update semantics in the shared MMDetection config builder. Convert the configured update interval to MMEngine micro-iterations exactly once by multiplying by gradient accumulation, and reuse that value for validation and checkpoint hooks.

**Tech Stack:** Python 3.10, unittest, MMEngine/MMDetection configuration dictionaries.

## Global Constraints

- Do not delete or rewrite existing `outputs` artifacts or checkpoints.
- Do not change training input size, optimizer, scheduler, loss, batch size, or gradient accumulation.
- Formal default is 5000 optimizer updates; `max_updates < 100` defaults to one update.
- Explicit `validation_update_interval` overrides the default and must be positive.
- This directory has no Git metadata, so commit steps are replaced by recorded file diffs and test evidence.

---

### Task 1: Add failing interval behavior tests

**Files:**
- Modify: `61 项自动化测试/test_mmdet_end_to_end.py`
- Test: `61 项自动化测试/test_mmdet_end_to_end.py`

**Interfaces:**
- Consumes: `build_mask2former_config(config, dataset)`.
- Produces: regression coverage for default, explicit override, and smoke-test interval behavior.

- [ ] **Step 1: Make the test config helper accept parameter overrides**

Change `_config(root)` to `_config(root, **parameter_overrides)` and merge the overrides after its literal default parameters.

- [ ] **Step 2: Add a failing formal-default test**

Build a config with `max_updates=50000`, physical batch 1, effective batch 8 and assert literal values:

```python
self.assertEqual(cfg["train_cfg"]["val_interval"], 40000)
self.assertEqual(cfg["default_hooks"]["checkpoint"]["interval"], 40000)
```

- [ ] **Step 3: Add a failing explicit-override test**

Build with `max_updates=50000` and `validation_update_interval=2500`; assert both intervals are `20000` micro-iterations.

- [ ] **Step 4: Protect smoke behavior**

In the existing ten-update configuration test, assert both intervals are `8`, representing one optimizer update with accumulation 8.

- [ ] **Step 5: Run the focused tests and verify RED**

Run:

```powershell
wsl.exe -d Ubuntu -- /home/baoju/.morphoroute/venvs/main/bin/python "/mnt/e/桌面/实验代码/铁路隧道多病害实例分割实验代码/railway_tunnel_experiments/61 项自动化测试/test_mmdet_end_to_end.py" -v
```

Expected: formal-default and explicit-override tests fail because the old builder uses 1000 optimizer updates and ignores the new parameter; unrelated tests pass.

### Task 2: Implement optimizer-update validation interval

**Files:**
- Modify: `src/morphoroute/MMDetection 配置与扩展组件/config_builder.py`
- Test: `61 项自动化测试/test_mmdet_end_to_end.py`

**Interfaces:**
- Consumes: optional integer `config.parameters["validation_update_interval"]`.
- Produces: `train_cfg.val_interval` and `default_hooks.checkpoint.interval` in micro-iterations.

- [ ] **Step 1: Derive the update interval once**

Replace the fixed heuristic with:

```python
default_validation_update_interval = 1 if max_updates < 100 else 5000
validation_update_interval = int(
    config.parameters.get(
        "validation_update_interval",
        default_validation_update_interval,
    )
)
if validation_update_interval <= 0:
    raise ValueError("validation_update_interval must be positive")
validation_update_interval = min(validation_update_interval, max_updates)
validation_interval = validation_update_interval * gradient_accumulation
```

- [ ] **Step 2: Keep validation and checkpoint hooks synchronized**

Continue using the single derived `validation_interval` for both `train_cfg.val_interval` and the checkpoint hook interval.

- [ ] **Step 3: Run focused tests and verify GREEN**

Run the WSL command from Task 1. Expected: all tests pass.

- [ ] **Step 4: Record the diff**

Use a read-only comparison of the edited source and test sections; no commit is possible because `.git` is absent.

### Task 3: Verify the current EXP-04 resume configuration

**Files:**
- Read: `outputs/experiments/EXP-04/local_6gb/seed_42/runs/last_checkpoint`
- Read: generated dry configuration from the current EXP-04 config.

**Interfaces:**
- Consumes: existing EXP-04 local profile and latest complete checkpoint.
- Produces: evidence that the new run resolves to 40000 micro-iterations between validations and resumes rather than restarting.

- [ ] **Step 1: Run the MMDetection configuration unit suite**

Run the focused WSL test file and confirm zero failures.

- [ ] **Step 2: Inspect the last checkpoint without modifying it**

Confirm `last_checkpoint` points to an existing `iter_*.pth` file.

- [ ] **Step 3: Generate or inspect a non-training resolved configuration**

Build the current EXP-04 configuration without starting `Runner.train()` and assert:

```text
max_iters = 400000
val_interval = 40000
checkpoint interval = 40000
resume = true
```

- [ ] **Step 4: Run the broader non-training test subset**

Run `test_adapters.py`, `test_checkpoint_resume.py`, and `test_mmdet_end_to_end.py` in WSL. Expected: zero failures.

- [ ] **Step 5: Hand off the safe resume command**

Tell the user to stop the old in-memory run only after its active validation completes, then rerun the unchanged EXP-04 PowerShell command. Do not automatically launch the multi-day training process.
