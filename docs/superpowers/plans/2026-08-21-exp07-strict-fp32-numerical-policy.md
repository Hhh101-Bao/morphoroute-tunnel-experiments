# EXP-07 Strict FP32 Numerical Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an auditable strict-FP32 EXP-07 runner that prevents non-finite Hungarian inputs from silently corrupting training and supports a 40,000-to-60,000 probe followed by a clean B4/B5/B6 restart.

**Architecture:** A reusable runtime module owns FP32 configuration transformation, finite-loss/gradient guards, Hungarian evidence capture, and checkpoint verification. A project-owned outer runner installs that policy before delegating to the existing efficient EXP-07 runner, preserving compact evaluation and sequential variant execution. A packaging script creates an AutoDL-ready archive without modifying unrelated project files.

**Tech Stack:** Python 3.10, PyTorch 2.1.2, MMEngine 0.10.7, MMDetection 3.3.0, NumPy 1.26.4, unittest, tar/gzip, SHA-256.

**Spec:** `docs/superpowers/specs/2026-08-21-exp07-strict-fp32-numerical-policy-design.md`

## Global Constraints

- B4, B5, and B6 must use identical strict IEEE-FP32 execution.
- AMP, GradScaler, BF16, CUDA matmul TF32, and cuDNN TF32 must be disabled.
- Do not change model architecture, dataset, splits, seed, optimizer hyperparameters, schedule, gradient accumulation, augmentations, budget gate, or evaluation definitions.
- Never clamp, replace, or silently skip non-finite losses, gradients, parameters, or Hungarian costs.
- The 40,000-to-60,000 checkpoint path is diagnostic only; formal training starts from zero with `--no-resume`.
- Failed AMP artifacts cannot enter formal metrics or paper results.

---

### Task 1: Strict FP32 configuration contract

**Files:**
- Create: `61 项自动化测试/test_exp07_strict_fp32_policy.py`
- Create: `src/morphoroute/训练适配、断点恢复和资源记录/strict_fp32.py`

**Interfaces:**
- Produces: `apply_strict_fp32_config(configuration, *, variant_id: str, mode: str) -> object`
- Produces: `configure_torch_strict_fp32(torch_module) -> dict[str, object]`
- Produces: constant `POLICY_REVISION = "morphoroute_strict_fp32_v1"`

- [ ] **Step 1: Write the failing configuration tests**

Create tests that pass a plain dictionary containing an `AmpOptimWrapper`, `loss_scale`, `dtype`, schedule, optimizer, and dataloader. Assert that the wished-for function changes only optimizer-wrapper type and precision keys, preserves all scientific fields byte-for-byte by deep comparison, adds the policy audit, and applies identically to B4/B5/B6.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python "61 项自动化测试/test_exp07_strict_fp32_policy.py"
```

Expected: import failure because `morphoroute.runtime.strict_fp32` does not exist.

- [ ] **Step 3: Implement the minimal configuration policy**

Implement dictionary/MMEngine-Config-compatible access without importing MMEngine at module import time. Change `optim_wrapper.type` to `StrictFiniteOptimWrapper`, remove `loss_scale` and `dtype`, add `clip_grad.error_if_nonfinite=True`, and write an exact numerical audit. Configure PyTorch TF32 flags and highest FP32 matmul precision.

- [ ] **Step 4: Run the configuration tests to verify GREEN**

Run the same unittest command. Expected: all tests pass.

### Task 2: Finite loss, gradient, and Hungarian guards

**Files:**
- Modify: `61 项自动化测试/test_exp07_strict_fp32_policy.py`
- Modify: `src/morphoroute/训练适配、断点恢复和资源记录/strict_fp32.py`

**Interfaces:**
- Produces: `install_strict_finite_optim_wrapper(artifact_dir: Path) -> type`
- Produces: `install_hungarian_finite_guard(artifact_dir: Path) -> None`
- Produces: `NonFiniteTrainingError(RuntimeError)`

- [ ] **Step 1: Add failing real-tensor tests**

Use CPU PyTorch tensors and a real SGD optimizer. Test one finite update, rejection of a NaN scalar loss before backward, rejection of a manually injected Inf gradient before optimizer step, preservation of parameters after rejection, and JSON/NPY evidence for an invalid Hungarian matrix.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: missing guard interfaces or assertions fail for non-finite inputs.

- [ ] **Step 3: Implement minimal guards**

Register `StrictFiniteOptimWrapper` lazily through `mmengine.registry.OPTIM_WRAPPERS` so local pure-policy tests do not require MMEngine. Override backward and step, check tensors before any state-changing call, use `error_if_nonfinite=True`, and write atomic diagnostic JSON. Wrap MMDetection's imported Hungarian function without modifying costs.

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: every guard test passes and rejected updates leave parameters unchanged.

### Task 3: Strict runner and checkpoint verifier

**Files:**
- Create: `工程辅助运行脚本/run_exp07_strict_fp32.py`
- Create: `工程辅助运行脚本/verify_exp07_strict_fp32_checkpoint.py`
- Create: `61 项自动化测试/test_exp07_strict_fp32_runner.py`

**Interfaces:**
- Runner invocation: `python run_exp07_strict_fp32.py INNER_RUNNER ENTRY [entry arguments...]`
- Environment: `EXP07_FP32_MODE=formal|probe`
- Environment: `EXP07_FP32_STOP_ITER=60000` for probe mode
- Environment: `EXP07_NUMERICAL_ARTIFACT_DIR=/absolute/path`
- Verifier invocation: `python verify_exp07_strict_fp32_checkpoint.py CHECKPOINT`

- [ ] **Step 1: Write failing runner source-contract tests**

Assert argument validation, valid mode enforcement, final configuration monkeypatch ordering, strict marker emission, probe max-iteration enforcement, and verifier rejection of checkpoints containing a `loss_scaler` or non-finite tensors.

- [ ] **Step 2: Run and verify RED**

Expected: missing runner/verifier files.

- [ ] **Step 3: Implement the runner**

Set FP32 flags, register guards, wrap `OfficialMMDetBackend._configuration`, apply the strict policy after the source configuration is built, dump the resolved configuration, and then execute the existing efficient runner through `runpy.run_path`. In probe mode, cap only B4 at 60,000 and exit with `EXP07_STRICT_FP32_PROBE_REACHED_LIMIT` after training returns.

- [ ] **Step 4: Implement the checkpoint verifier**

Load to CPU, scan every floating/complex state tensor, reject optimizer state containing `loss_scaler`, print machine-readable counts, and return nonzero on failure.

- [ ] **Step 5: Run focused tests and syntax compilation**

Expected: runner and verifier tests pass; `compileall` returns zero.

### Task 4: AutoDL recovery and formal-start workflow

**Files:**
- Create: `工程辅助运行脚本/install_and_launch_exp07_strict_fp32.sh`
- Create: `docs/EXP07_STRICT_FP32_AUTODL.md`
- Create: `61 项自动化测试/test_exp07_strict_fp32_shell_contract.py`

**Interfaces:**
- Shell modes: `probe`, `formal`, `status`, `verify-probe`, `verify-formal`
- Pointer files: `outputs/autodl_logs/EXP-07_strict_fp32.current_log`, `.pid`, and `.artifacts`

- [ ] **Step 1: Write failing shell-contract tests**

Assert exact project/data paths are arguments rather than hidden defaults, process guards prevent duplicate launch, probe requires finite `iter_40000.pth`, formal mode requires a successful probe marker, formal mode moves the failed directory to a timestamped backup, and only the two D1 morphology files are restored.

- [ ] **Step 2: Run and verify RED**

Expected: missing shell workflow.

- [ ] **Step 3: Implement the shell workflow**

Use one Bash process end-to-end, quote every path, avoid recursive deletion, preserve failed runs by `mv`, launch with `nohup`, and record PID/log/artifact pointers. Formal launch must pass `--no-resume` and refuse any checkpoint in the new B4/B5/B6 directories.

- [ ] **Step 4: Run shell syntax and contract tests**

Run `bash -n` where Bash is available and the Python contract test everywhere. Expected: pass.

### Task 5: Regression suite and distributable archive

**Files:**
- Create: `docs/EXP07_STRICT_FP32_PATCH_FILES.txt`
- Create: `exp07_strict_fp32_recovery_20260821.tar.gz`
- Create: `exp07_strict_fp32_recovery_20260821.sha256`

**Interfaces:**
- Archive paths are project-relative and safe to extract at project root.

- [ ] **Step 1: Run the new focused tests**

Run all three new Python test files and the shell-contract test. Expected: zero failures.

- [ ] **Step 2: Run existing relevant regressions**

Run Mask2Former AMP-loss precision, MorphoRoute numerical-policy, registry, variant, experiment 00-07, MMDetection route diagnostics, and MMDetection end-to-end tests when their dependencies are present. Record explicit skips only for dependencies absent on the Windows packaging host; AutoDL must run the full set.

- [ ] **Step 3: Compile all shipped Python**

Run `python -m compileall -q` on the runtime module, runner, verifier, and tests. Expected: exit zero.

- [ ] **Step 4: Build and verify the archive**

Create the archive from `EXP07_STRICT_FP32_PATCH_FILES.txt`, compute SHA-256, extract into a new temporary directory, compare every extracted SHA-256 with the source, and list the archive. Expected: exact match and no absolute or parent-traversal paths.

- [ ] **Step 5: Deliver AutoDL commands**

Provide upload verification, extraction with backup, AutoDL test commands, probe launch/status/verification, formal launch/status, and final completion checks. Do not claim the long-run bug fixed until the user's 60,000-iteration probe produces the required marker and finite checkpoint.

