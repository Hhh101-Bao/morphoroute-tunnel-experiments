# Controlled Convergence and RTX 5090 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make early stopping scientifically controlled, reduce queries to a reversible 64-query setting, and add a directly selectable RTX 5090/32GB AutoDL profile.

**Architecture:** Keep the existing budget gate and configuration generator as the control points. Correct gate state semantics in the runtime unit, express experiment-specific start updates in generated configuration, and add a fourth profile through the existing config/layout/CLI abstractions. Extend the existing dataloader builder with audited transfer-performance options.

**Tech Stack:** Python 3.10+, unittest, MMEngine/MMDetection configuration dictionaries, JSON experiment cards.

## Global Constraints

- Do not change data splits, model topology, losses, optimizer, effective batch, D2-only selection, or the 50,000-update ceiling.
- Do not modify EXP-04 configuration or any existing output/checkpoint.
- Keep `sop_full` unchanged as the original protocol reference.
- Every accelerated setting must have a one-parameter rollback.

---

### Task 1: Correct budget-gate patience semantics

**Files:**
- Modify: `61 项自动化测试/test_reversible_acceleration.py`
- Modify: `src/morphoroute/训练适配、断点恢复和资源记录/acceleration.py`

**Interfaces:**
- Consumes: `BudgetGateMonitor.observe(update: int, value: float)`
- Produces: pre-gate observations update `best_value` but leave `bad_validations=0`; patience begins at the gate.

- [ ] Add a failing test asserting that a plateau before 30k does not stop at 30k and stops only on the second gated non-improvement at 35k.
- [ ] Run the targeted test and confirm the old implementation fails because it stops at 30k.
- [ ] Reset `bad_validations` to zero for observations before `start_update` while retaining the historical best value.
- [ ] Run the targeted test suite and confirm it passes.

### Task 2: Add controlled profile policies and 5090 profile

**Files:**
- Modify: `61 项自动化测试/test_cli.py`
- Modify: `61 项自动化测试/test_standalone_entries.py`
- Modify: `61 项自动化测试/test_reversible_acceleration.py`
- Modify: `工程辅助运行脚本/三套实验配置生成器.py`
- Modify: `src/morphoroute/配置、实验注册、运行记录和 D4 防泄漏组件/config.py`
- Modify: `src/morphoroute/配置、实验注册、运行记录和 D4 防泄漏组件/layout.py`
- Modify: `src/morphoroute/配置、实验注册、运行记录和 D4 防泄漏组件/standalone.py`
- Modify: `工程辅助运行脚本/全部实验空运行检查工具.py`

**Interfaces:**
- Consumes: profile strings and `generate_profile_configs(output_root)`.
- Produces: selectable `autodl_5090_32gb` and 76 generated cards with explicit controlled budgets.

- [ ] Add failing tests for four profiles/76 cards, 5090 parameter values, 64-query accelerated configs, experiment-specific gate starts, and no default core gate release.
- [ ] Run the tests and confirm failures against the three-profile generator.
- [ ] Add the profile to validation, directory layout and CLI choices.
- [ ] Refactor the generator to share the accelerated policy between `local_6gb` and `autodl_5090_32gb`, with 20k starts for EXP-05/06 and 30k for EXP-07/08B/09.
- [ ] Remove default `release_budget_gate=true` and `release_budget_gate_variants` promotion behavior.
- [ ] Run the profile tests and confirm they pass.

### Task 3: Add safe dataloader transfer acceleration

**Files:**
- Modify: `61 项自动化测试/test_reversible_acceleration.py`
- Modify: `src/morphoroute/MMDetection 配置与扩展组件/config_builder.py`

**Interfaces:**
- Consumes: `performance_mode`, `dataloader_workers`, `pin_memory`, `prefetch_factor` parameters.
- Produces: MMEngine dataloader dictionaries and `acceleration_audit` fields.

- [ ] Add failing tests that performance mode enables pinned memory, preserves persistent workers, applies prefetch only with workers, and records values in the audit.
- [ ] Run the tests and confirm the old builder lacks these fields.
- [ ] Extend `_dataloader` and `_common_config` with validated settings; omit `prefetch_factor` when workers are zero.
- [ ] Run MMDetection config tests and confirm they pass.

### Task 4: Generate, audit and copy final artifacts

**Files:**
- Generate: `三种运行档位的实验配置/RTX 5090 32GB AutoDL 加速正式实验配置/exp00.json` through `exp15.json`
- Regenerate: accelerated `local_6gb` JSON cards
- Preserve: all `完整 SOP 正式实验配置` and EXP-04 files byte-for-byte

**Interfaces:**
- Consumes: the updated generator.
- Produces: runnable JSON cards and a hash-audited change set.

- [ ] Run the generator in a temporary directory and validate every JSON through `ExperimentConfig`.
- [ ] Generate the project configuration directories and verify 76 files.
- [ ] Run targeted tests, config dry runs and the full automated suite.
- [ ] Compare source hashes before copying and copy only the reviewed files back to the user project.
- [ ] Re-run verification from the actual E: project and report exact pass/fail counts.

