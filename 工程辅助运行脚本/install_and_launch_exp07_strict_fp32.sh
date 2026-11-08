#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
PROJECT_ROOT="${2:-/root/autodl-tmp/railway_tunnel_experiments}"
DATA_ROOT="${3:-/root/autodl-tmp/railway_dataset_v1/铁路隧道多病害实例分割_原始影像严格划分_v1}"
OUTPUT_ROOT="$PROJECT_ROOT/outputs"
LOG_DIR="$OUTPUT_ROOT/autodl_logs"
INNER_RUNNER="${4:-$LOG_DIR/run_exp07_efficient.py}"
ENTRY="$PROJECT_ROOT/19 个独立实验运行入口/exp07_routing_controls.py"
STRICT_RUNNER="$PROJECT_ROOT/工程辅助运行脚本/run_exp07_strict_fp32.py"
VERIFIER="$PROJECT_ROOT/工程辅助运行脚本/verify_exp07_strict_fp32_checkpoint.py"
EXP07_ROOT="$OUTPUT_ROOT/experiments/EXP-07"
SEED_DIR="$EXP07_ROOT/local_6gb/seed_42"
B4_DIR="$SEED_DIR/runs/B4/seed_42"
AMP_CHECKPOINT="$B4_DIR/iter_40000.pth"
PROBE_CHECKPOINT="$B4_DIR/iter_60000.pth"
PROBE_LOG_POINTER="$LOG_DIR/EXP-07_strict_fp32_probe.current_log"
PROBE_PID_POINTER="$LOG_DIR/EXP-07_strict_fp32_probe.pid"
FORMAL_LOG_POINTER="$LOG_DIR/EXP-07_strict_fp32_formal.current_log"
FORMAL_PID_POINTER="$LOG_DIR/EXP-07_strict_fp32_formal.pid"

mkdir -p "$LOG_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export MORPHOROUTE_PREDICTION_WORKERS="${MORPHOROUTE_PREDICTION_WORKERS:-8}"
export MORPHOROUTE_EVALUATION_WORKERS="${MORPHOROUTE_EVALUATION_WORKERS:-8}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required file does not exist: $1" >&2
        exit 2
    fi
}

ensure_no_exp07_process() {
    if pgrep -f '[r]un_exp07_strict_fp32.py|[e]xp07_routing_controls.py' >/dev/null; then
        echo "ERROR: EXP-07 process already exists" >&2
        pgrep -af '[r]un_exp07_strict_fp32.py|[e]xp07_routing_controls.py' >&2
        exit 3
    fi
}

scan_faults() {
    local log_file="$1"
    grep -nEi \
      'NONFINITE_(FORWARD|LOSS|GRAD|GRAD_NORM|PARAMETER)|cost matrix is infeasible|matrix contains invalid numeric entries|Traceback \(most recent call last\)|loss: nan|loss_mask: nan|grad_norm: (nan|inf)|EXP-07_EXIT_CODE=[^0]' \
      "$log_file" || true
}

launch_job() {
    local run_mode="$1"
    local log_file="$2"
    local pointer_file="$3"
    local pid_file="$4"
    shift 4
    nohup "$@" >"$log_file" 2>&1 &
    local launched_pid=$!
    printf '%s\n' "$launched_pid" >"$pid_file"
    printf '%s\n' "$log_file" >"$pointer_file"
    printf '%s\n' "$launched_pid" >"$LOG_DIR/EXP-07.pid"
    printf '%s\n' "$log_file" >"$LOG_DIR/EXP-07.current_log"
    echo "mode=$run_mode pid=$launched_pid"
    echo "log=$log_file"
}

case "$MODE" in
probe)
    ensure_no_exp07_process
    require_file "$STRICT_RUNNER"
    require_file "$VERIFIER"
    require_file "$INNER_RUNNER"
    require_file "$ENTRY"
    require_file "$AMP_CHECKPOINT"
    python "$VERIFIER" --allow-amp-diagnostic "$AMP_CHECKPOINT"
    echo "CHECKPOINT_NUMERICS=PASSED checkpoint=$AMP_CHECKPOINT"
    PROBE_LOG="$LOG_DIR/EXP-07_strict_fp32_probe_$(date +%Y%m%d_%H%M%S).log"
    launch_job probe "$PROBE_LOG" "$PROBE_LOG_POINTER" "$PROBE_PID_POINTER" \
      env EXP07_FP32_MODE=probe \
      EXP07_FP32_STOP_ITER=60000 \
      EXP07_NUMERICAL_ARTIFACT_DIR="$LOG_DIR/EXP07_strict_fp32_artifacts_probe" \
      python "$STRICT_RUNNER" "$INNER_RUNNER" "$ENTRY" \
        --profile local_6gb \
        --dataset-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --no-smoke-test \
        --no-auto-run-prerequisites \
        --resume
    ;;

verify-probe)
    require_file "$VERIFIER"
    require_file "$PROBE_LOG_POINTER"
    PROBE_LOG="$(cat "$PROBE_LOG_POINTER")"
    require_file "$PROBE_LOG"
    require_file "$PROBE_CHECKPOINT"
    grep -q 'EXP07_STRICT_FP32_PROBE_REACHED_LIMIT' "$PROBE_LOG"
    if [ -n "$(scan_faults "$PROBE_LOG")" ]; then
        echo "ERROR: strict-FP32 probe contains a numerical/runtime fault" >&2
        scan_faults "$PROBE_LOG" >&2
        exit 4
    fi
    python "$VERIFIER" "$PROBE_CHECKPOINT"
    echo "EXP07_STRICT_FP32_PROBE_VERIFICATION=PASSED"
    ;;

formal)
    ensure_no_exp07_process
    require_file "$STRICT_RUNNER"
    require_file "$VERIFIER"
    require_file "$INNER_RUNNER"
    require_file "$ENTRY"
    require_file "$PROBE_LOG_POINTER"
    PROBE_LOG="$(cat "$PROBE_LOG_POINTER")"
    require_file "$PROBE_LOG"
    require_file "$PROBE_CHECKPOINT"
    grep -q 'EXP07_STRICT_FP32_PROBE_REACHED_LIMIT' "$PROBE_LOG"
    python "$VERIFIER" "$PROBE_CHECKPOINT"

    if [ ! -d "$EXP07_ROOT" ]; then
        echo "ERROR: EXP-07 source tree for reversible backup is absent" >&2
        exit 5
    fi
    AMP_FAILED_BACKUP="${EXP07_ROOT}_amp_failed_$(date +%Y%m%d_%H%M%S)"
    mv "$EXP07_ROOT" "$AMP_FAILED_BACKUP"
    printf '%s\n' "$AMP_FAILED_BACKUP" >"$LOG_DIR/EXP-07_amp_failed_backup.current"

    mkdir -p "$SEED_DIR"
    for immutable_cache in morphology_features_D1.csv gmm_D1.json; do
        cache_source="$AMP_FAILED_BACKUP/local_6gb/seed_42/$immutable_cache"
        if [ -f "$cache_source" ]; then
            cp -a "$cache_source" "$SEED_DIR/$immutable_cache"
        fi
    done
    if find "$EXP07_ROOT" -type f -name '*.pth' -print -quit | grep -q .; then
        echo "REFUSE_CHECKPOINT_REUSE: formal tree contains a checkpoint" >&2
        exit 6
    fi
    echo "REFUSE_CHECKPOINT_REUSE check=PASSED backup=$AMP_FAILED_BACKUP"

    FORMAL_LOG="$LOG_DIR/EXP-07_strict_fp32_formal_$(date +%Y%m%d_%H%M%S).log"
    launch_job formal "$FORMAL_LOG" "$FORMAL_LOG_POINTER" "$FORMAL_PID_POINTER" \
      env EXP07_FP32_MODE=formal \
      EXP07_FP32_STOP_ITER= \
      EXP07_NUMERICAL_ARTIFACT_DIR="$LOG_DIR/EXP07_strict_fp32_artifacts_formal" \
      python "$STRICT_RUNNER" "$INNER_RUNNER" "$ENTRY" \
        --profile local_6gb \
        --dataset-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --no-smoke-test \
        --no-auto-run-prerequisites \
        --no-resume
    ;;

status)
    if [ -f "$FORMAL_LOG_POINTER" ]; then
        ACTIVE_LOG="$(cat "$FORMAL_LOG_POINTER")"
        ACTIVE_PID="$(cat "$FORMAL_PID_POINTER" 2>/dev/null || true)"
        ACTIVE_MODE=formal
    elif [ -f "$PROBE_LOG_POINTER" ]; then
        ACTIVE_LOG="$(cat "$PROBE_LOG_POINTER")"
        ACTIVE_PID="$(cat "$PROBE_PID_POINTER" 2>/dev/null || true)"
        ACTIVE_MODE=probe
    else
        echo "No strict-FP32 EXP-07 job has been launched"
        exit 0
    fi
    echo "mode=$ACTIVE_MODE pid=${ACTIVE_PID:-none} log=$ACTIVE_LOG"
    if [ -n "${ACTIVE_PID:-}" ] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        ps -p "$ACTIVE_PID" -o pid,etime,time,%cpu,%mem,rss,stat,cmd
    else
        echo "process=stopped"
    fi
    tail -30 "$ACTIVE_LOG"
    ;;

verify-formal)
    require_file "$FORMAL_LOG_POINTER"
    FORMAL_LOG="$(cat "$FORMAL_LOG_POINTER")"
    require_file "$FORMAL_LOG"
    MANIFEST="$SEED_DIR/run_manifest.json"
    require_file "$MANIFEST"
    grep -q '"status": "completed"' "$MANIFEST"
    grep -q 'EXP07_STRICT_FP32_POLICY.*variant=B4.*mode=formal' "$FORMAL_LOG"
    grep -q 'EXP07_STRICT_FP32_POLICY.*variant=B5.*mode=formal' "$FORMAL_LOG"
    grep -q 'EXP07_STRICT_FP32_POLICY.*variant=B6.*mode=formal' "$FORMAL_LOG"
    if [ -n "$(scan_faults "$FORMAL_LOG")" ]; then
        echo "ERROR: formal strict-FP32 log contains a fault" >&2
        scan_faults "$FORMAL_LOG" >&2
        exit 7
    fi
    if pgrep -f '[r]un_exp07_strict_fp32.py|[e]xp07_routing_controls.py' >/dev/null; then
        echo "ERROR: formal process is still running" >&2
        exit 8
    fi
    echo "EXP07_STRICT_FP32_FORMAL_VERIFICATION=PASSED"
    ;;

*)
    echo "usage: $0 {probe|verify-probe|formal|status|verify-formal} [PROJECT_ROOT] [DATA_ROOT] [INNER_RUNNER]" >&2
    exit 64
    ;;
esac
