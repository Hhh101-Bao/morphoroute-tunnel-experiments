#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${1:-/root/autodl-tmp/railway_tunnel_experiments}"
DATA_ROOT="${2:-/root/autodl-tmp/railway_dataset_v1/铁路隧道多病害实例分割_原始影像严格划分_v1}"
OUTPUT_ROOT="$PROJECT_ROOT/outputs"
LOG_DIR="$OUTPUT_ROOT/autodl_logs"
ENTRY="$PROJECT_ROOT/19 个独立实验运行入口/exp09_ablation.py"
EFFICIENT_RUNNER="$LOG_DIR/run_exp07_efficient.py"
COMPAT_RUNNER="$LOG_DIR/exp09_pcgrad_checkpoint_compat.py"
FP32_RUNNER="$PROJECT_ROOT/工程辅助运行脚本/run_exp09_b11_true_fp32.py"
VERIFY="$PROJECT_ROOT/工程辅助运行脚本/verify_exp09_b11_checkpoint.py"
D2_VERIFY="$PROJECT_ROOT/工程辅助运行脚本/verify_exp09_b11_memory_efficient_d2.py"
CONFIG="$LOG_DIR/exp09_local6gb_strict_fp32.json"
CHAIN_SCRIPT="$LOG_DIR/autostart_until_oct02.sh"
B11_DIR="$OUTPUT_ROOT/experiments/EXP-09/local_6gb/seed_42/runs/B11/seed_42"
CHECKPOINT="$B11_DIR/iter_80000.pth"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export MORPHOROUTE_PREDICTION_WORKERS=8
export MORPHOROUTE_EVALUATION_WORKERS=8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 2

for required in \
    "$ENTRY" \
    "$EFFICIENT_RUNNER" \
    "$COMPAT_RUNNER" \
    "$FP32_RUNNER" \
    "$VERIFY" \
    "$D2_VERIFY" \
    "$CONFIG" \
    "$CHECKPOINT"; do
    if [ ! -f "$required" ]; then
        echo "ERROR: required file is missing: $required"
        exit 3
    fi
done

if pgrep -f '[e]xp09_ablation.py' >/dev/null; then
    echo "ERROR: EXP-09 process already exists; refusing duplicate launch"
    pgrep -af '[e]xp09_ablation.py'
    exit 4
fi

OLD_CHAIN_PIDS="$(pgrep -f '[a]utostart_until_oct02.sh' || true)"
if [ -n "$OLD_CHAIN_PIDS" ]; then
    echo "Stopping obsolete downstream wait process before recovery..."
    for old_pid in $OLD_CHAIN_PIDS; do
        kill -TERM "$old_pid" 2>/dev/null || true
    done
fi

echo "===== Verify healthy B11 seed42 checkpoint ====="
python "$VERIFY" "$CHECKPOINT" || exit 6

echo "===== Run focused policy tests ====="
python "61 项自动化测试/test_morphology.py" || exit 7
python "61 项自动化测试/test_exp09_b11_true_fp32.py" || exit 7
python "61 项自动化测试/test_exp09_b11_pcgrad_fp32_runtime.py" || exit 7
python "61 项自动化测试/test_exp09_b11_memory_efficient_postprocess.py" || exit 7
python -m compileall -q \
    "src/morphoroute/MorphoRoute 模型、专家、路由器、损失和 PCGrad/morphology.py" \
    "src/morphoroute/MMDetection 配置与扩展组件/models/morphology_targets.py" \
    "src/morphoroute/MMDetection 配置与扩展组件/models/morphoroute_head.py" \
    "src/morphoroute/MMDetection 配置与扩展组件/models/pcgrad_runtime.py" \
    "src/morphoroute/训练适配、断点恢复和资源记录/pcgrad_fp32.py" \
    "src/morphoroute/MMDetection 配置与扩展组件/models/safe_batch_pad.py" \
    "src/morphoroute/MMDetection 配置与扩展组件/models/memory_efficient_fusion.py" \
    "工程辅助运行脚本/run_exp09_b11_true_fp32.py" \
    "工程辅助运行脚本/verify_exp09_b11_checkpoint.py" \
    "工程辅助运行脚本/verify_exp09_b11_memory_efficient_d2.py" || exit 8

STAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="$LOG_DIR/EXP09_B11_true_fp32_artifacts_$STAMP"
RECOVERY_LOG="$LOG_DIR/EXP-09_B11_true_fp32_recovery_$STAMP.log"
VERIFY_LOG="$LOG_DIR/EXP-09_B11_memory_efficient_D2_verify_$STAMP.log"
mkdir -p "$ARTIFACT_DIR"

echo "===== Verify complete D2 evaluation with bounded-memory postprocess ====="
python "$D2_VERIFY" "$B11_DIR/resolved_mmdet_config.py" \
    "$CHECKPOINT" 2>&1 | tee "$VERIFY_LOG"
VERIFY_CODE=${PIPESTATUS[0]}
if [ "$VERIFY_CODE" -ne 0 ]; then
    echo "ERROR: bounded-memory D2 verification failed"
    tail -100 "$VERIFY_LOG"
    exit 9
fi
if ! grep -q 'EXP09_B11_MEMORY_EFFICIENT_D2=PASSED' "$VERIFY_LOG"; then
    echo "ERROR: bounded-memory D2 verification marker is missing"
    tail -100 "$VERIFY_LOG"
    exit 9
fi
printf '%s\n' "$VERIFY_LOG" \
    > "$LOG_DIR/EXP-09_B11_memory_efficient_D2.current_log"
echo "D2 verification passed: $VERIFY_LOG"

nohup env \
    EXP09_B11_NUMERICAL_ARTIFACT_DIR="$ARTIFACT_DIR" \
    python "$COMPAT_RUNNER" \
    "$FP32_RUNNER" \
    "$EFFICIENT_RUNNER" \
    "$ENTRY" \
    --profile local_6gb \
    --dataset-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --config "$CONFIG" \
    --no-smoke-test \
    --no-auto-run-prerequisites \
    --resume \
    > "$RECOVERY_LOG" 2>&1 &

EXP09_PID=$!
printf '%s\n' "$EXP09_PID" > "$LOG_DIR/EXP-09.pid"
printf '%s\n' "$RECOVERY_LOG" > "$LOG_DIR/EXP-09.current_log"
printf '%s\n' "$ARTIFACT_DIR" \
    > "$LOG_DIR/EXP-09_B11_true_fp32.current_artifact_dir"

sleep 5
if ! kill -0 "$EXP09_PID" 2>/dev/null; then
    echo "ERROR: EXP-09 recovery exited during startup"
    tail -100 "$RECOVERY_LOG"
    exit 10
fi

echo "EXP-09 true-FP32 recovery PID=$EXP09_PID"
echo "EXP-09 log=$RECOVERY_LOG"
echo "Numerical artifacts=$ARTIFACT_DIR"

if [ -f "$CHAIN_SCRIPT" ]; then
    bash -n "$CHAIN_SCRIPT" || {
        echo "ERROR: downstream chain syntax check failed"
        exit 11
    }
    CHAIN_LOG="$LOG_DIR/AUTOCHAIN_UNTIL_1002_restart_$STAMP.log"
    nohup bash "$CHAIN_SCRIPT" > "$CHAIN_LOG" 2>&1 &
    CHAIN_PID=$!
    printf '%s\n' "$CHAIN_PID" > "$LOG_DIR/AUTOCHAIN_UNTIL_1002.pid"
    printf '%s\n' "$CHAIN_LOG" \
        > "$LOG_DIR/AUTOCHAIN_UNTIL_1002.current_log"
    echo "Downstream chain PID=$CHAIN_PID"
    echo "Downstream chain log=$CHAIN_LOG"
else
    echo "WARNING: downstream chain script not found; EXP-09 still launched"
fi

echo "===== Startup log ====="
tail -60 "$RECOVERY_LOG"
