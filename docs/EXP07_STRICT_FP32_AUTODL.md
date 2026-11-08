# EXP-07 strict FP32 AutoDL recovery

This package implements the approved numerical policy for EXP-07 B4/B5/B6.
It changes numerical execution only: AMP, GradScaler, BF16 and TF32 are
disabled; the model, data, split, seed, optimizer hyperparameters, schedule,
gradient accumulation, training budget and evaluation definitions are kept.

The old `iter_40000.pth` is used only for a 40,000-to-60,000 diagnostic probe.
It is never used for formal metrics. After a successful probe, formal B4/B5/B6
training starts from zero. The failed tree is moved to a timestamped backup and
is not deleted.

## 1. Install and test

Assume the archive and checksum were uploaded to
`/root/autodl-tmp/Untitled Folder`.

```bash
PROJECT_ROOT="/root/autodl-tmp/railway_tunnel_experiments"
UPLOAD_DIR="/root/autodl-tmp/Untitled Folder"
ARCHIVE="$UPLOAD_DIR/exp07_strict_fp32_recovery_20260821.tar.gz"

cd "$UPLOAD_DIR"
sha256sum -c exp07_strict_fp32_recovery_20260821.sha256
tar -tzf "$ARCHIVE"

BACKUP_DIR="$PROJECT_ROOT/outputs/source_backups/exp07_strict_fp32_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
tar -xOf "$ARCHIVE" docs/EXP07_STRICT_FP32_PATCH_FILES.txt >/tmp/EXP07_STRICT_FP32_PATCH_FILES.txt

while IFS= read -r relative; do
    if [ -f "$PROJECT_ROOT/$relative" ]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$relative")"
        cp -a "$PROJECT_ROOT/$relative" "$BACKUP_DIR/$relative"
    fi
done </tmp/EXP07_STRICT_FP32_PATCH_FILES.txt

tar -xzf "$ARCHIVE" -C "$PROJECT_ROOT"
chmod +x "$PROJECT_ROOT/工程辅助运行脚本/install_and_launch_exp07_strict_fp32.sh"
echo "source backup=$BACKUP_DIR"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python "61 项自动化测试/test_exp07_strict_fp32_policy.py"
python "61 项自动化测试/test_exp07_strict_fp32_runner.py"
python "61 项自动化测试/test_exp07_strict_fp32_shell_contract.py"
python "61 项自动化测试/test_registry.py"
python "61 项自动化测试/test_variants.py"
python "61 项自动化测试/test_variant_factory_runtime.py"
python "61 项自动化测试/test_morphoroute_head.py"
python "61 项自动化测试/test_mmdet_route_diagnostics.py"
python "61 项自动化测试/test_mmdet_end_to_end.py"
python "61 项自动化测试/test_experiments_00_07.py"
bash -n "工程辅助运行脚本/install_and_launch_exp07_strict_fp32.sh"
python -m compileall -q \
  "src/morphoroute/训练适配、断点恢复和资源记录/strict_fp32.py" \
  "工程辅助运行脚本/run_exp07_strict_fp32.py" \
  "工程辅助运行脚本/verify_exp07_strict_fp32_checkpoint.py"
```

Do not proceed unless all unittest commands report `OK`, `bash -n`
returns silently, and compileall exits with code zero.

## 2. Run the diagnostic probe

The existing verified compact evaluator is reused as the inner runner:

```bash
PROJECT_ROOT="/root/autodl-tmp/railway_tunnel_experiments"
DATA_ROOT="/root/autodl-tmp/railway_dataset_v1/铁路隧道多病害实例分割_原始影像严格划分_v1"
WORKFLOW="$PROJECT_ROOT/工程辅助运行脚本/install_and_launch_exp07_strict_fp32.sh"
INNER_RUNNER="$PROJECT_ROOT/outputs/autodl_logs/run_exp07_efficient.py"

bash "$WORKFLOW" probe "$PROJECT_ROOT" "$DATA_ROOT" "$INNER_RUNNER"
```

Monitor without attaching to the process:

```bash
bash "$WORKFLOW" status "$PROJECT_ROOT" "$DATA_ROOT" "$INNER_RUNNER"
tail -f "$(cat "$PROJECT_ROOT/outputs/autodl_logs/EXP-07_strict_fp32_probe.current_log")"
```

`Ctrl+C` only exits `tail`; the nohup job continues. The probe is successful
only when the process has stopped and this command passes:

```bash
bash "$WORKFLOW" verify-probe "$PROJECT_ROOT" "$DATA_ROOT" "$INNER_RUNNER"
```

Required final marker:

```text
EXP07_STRICT_FP32_PROBE_VERIFICATION=PASSED
```

The probe must also produce a finite `iter_60000.pth` without GradScaler state.
If it fails, do not launch formal training; preserve the generated numerical
artifacts and log for diagnosis.

## 3. Start clean formal B4/B5/B6 training

Only after `verify-probe` passes:

```bash
bash "$WORKFLOW" formal "$PROJECT_ROOT" "$DATA_ROOT" "$INNER_RUNNER"
```

This command performs a reversible `mv` of the entire failed EXP-07 tree,
restores only `morphology_features_D1.csv` and `gmm_D1.json`, verifies that no
checkpoint entered the new tree, then launches with `--no-resume`. B4, B5 and
B6 run sequentially under the same strict policy.

Monitor with:

```bash
bash "$WORKFLOW" status "$PROJECT_ROOT" "$DATA_ROOT" "$INNER_RUNNER"
tail -f "$(cat "$PROJECT_ROOT/outputs/autodl_logs/EXP-07_strict_fp32_formal.current_log")"
```

## 4. Final verification

After the process exits:

```bash
bash "$WORKFLOW" verify-formal "$PROJECT_ROOT" "$DATA_ROOT" "$INNER_RUNNER"
```

Required final marker:

```text
EXP07_STRICT_FP32_FORMAL_VERIFICATION=PASSED
```

The workflow rejects a completed claim if the manifest is absent/incomplete,
any of B4/B5/B6 lacks the formal strict-FP32 policy marker, the log contains a
NaN/Inf/Hungarian/Python error, or an EXP-07 process is still running.

## Scientific interpretation

- The 40,000-to-60,000 probe is diagnostic evidence only.
- The old AMP run, the probe checkpoint and their metrics must not be mixed
  with formal results.
- Only the clean strict-FP32 B4/B5/B6 run is eligible for tables and paper
  claims.
- Any non-finite loss/gradient/Hungarian cost stops the run and writes evidence;
  it is never clamped, replaced or silently skipped.
