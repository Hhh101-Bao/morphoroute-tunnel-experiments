#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/railway_tunnel_experiments"
DATA_ROOT="/root/autodl-tmp/railway_dataset_v1/铁路隧道多病害实例分割_原始影像严格划分_v1"
OUTPUT_ROOT="$PROJECT_ROOT/outputs"
LOG_DIR="$OUTPUT_ROOT/autodl_logs"
ENTRY_DIR="$PROJECT_ROOT/19 个独立实验运行入口"
RUNNER="$LOG_DIR/run_exp07_efficient.py"
STATE_FILE="$LOG_DIR/SLIM_EXPERIMENT_CHAIN.state"
STEP_LOG_POINTER="$LOG_DIR/SLIM_EXPERIMENT_CHAIN.current_step_log"
PYTHON_BIN="/root/miniconda3/bin/python3.10"

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

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

set_state() {
    printf '%s\n' "$1" > "$STATE_FILE"
}

fail() {
    set_state "FAILED stage=${CURRENT_STAGE:-preflight} time=$(date -Is) reason=$1"
    echo "ERROR：$1" >&2
    exit 2
}

is_completed() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
accepted = set(sys.argv[2].split(","))
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if str(payload.get("status")) in accepted else 1)
PY
}

run_step() {
    local stage="$1"
    shift
    local log_file="$LOG_DIR/${stage}_$(date +%Y%m%d_%H%M%S).log"
    CURRENT_STAGE="$stage"
    set_state "RUNNING stage=$stage log=$log_file time=$(date -Is)"
    printf '%s\n' "$log_file" > "$STEP_LOG_POINTER"
    echo "===== 开始 $stage ====="
    echo "日志：$log_file"
    set +e
    "$@" > "$log_file" 2>&1
    local code=$?
    set -e
    if (( code != 0 )); then
        tail -80 "$log_file" >&2 || true
        fail "$stage 退出码=$code"
    fi
    if grep -qEi \
        '(^|[[:space:]])ERROR:|Traceback \(most recent call last\)|loss: nan|loss_mask: nan|grad_norm: (nan|inf)|matrix contains invalid|cost matrix is infeasible|cannot reshape|CUDA out of memory|Out of memory|Killed' \
        "$log_file"; then
        grep -nEi \
            '(^|[[:space:]])ERROR:|Traceback \(most recent call last\)|loss: nan|loss_mask: nan|grad_norm: (nan|inf)|matrix contains invalid|cost matrix is infeasible|cannot reshape|CUDA out of memory|Out of memory|Killed' \
            "$log_file" | tail -30 >&2 || true
        fail "$stage 日志出现故障标志"
    fi
    echo "===== 完成 $stage ====="
}

CURRENT_STAGE="preflight"
set_state "PREFLIGHT time=$(date -Is)"

[[ -x "$PYTHON_BIN" ]] || fail "Python不存在：$PYTHON_BIN"
[[ -f "$RUNNER" ]] || fail "高效运行包装器不存在：$RUNNER"
[[ -d "$DATA_ROOT" ]] || fail "数据集不存在：$DATA_ROOT"

if pgrep -f \
    '[e]xp09_ablation.py|[e]xp08b_performance_diagnosis.py|[e]xp10_freeze.py|[e]xp11_external_test.py|[e]xp12_reliability.py|[e]xp14_sequence_efficiency.py' \
    >/dev/null; then
    pgrep -af \
        '[e]xp09_ablation.py|[e]xp08b_performance_diagnosis.py|[e]xp10_freeze.py|[e]xp11_external_test.py|[e]xp12_reliability.py|[e]xp14_sequence_efficiency.py' >&2
    fail "检测到冲突实验进程，未启动精简链"
fi

AVAILABLE_KB="$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')"
(( AVAILABLE_KB >= 15 * 1024 * 1024 )) || \
    fail "可用磁盘不足15GiB"

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY' || exit 2
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
exp09 = root / "experiments" / "EXP-09"
manifest = exp09 / "run_manifest.json"
if not manifest.is_file():
    raise SystemExit("ERROR：缺少EXP-09汇总清单")
payload = json.loads(manifest.read_text(encoding="utf-8"))
if payload.get("status") != "completed":
    raise SystemExit(f"ERROR：EXP-09尚未完成：{payload.get('status')}")

def latest(name):
    paths = list(exp09.rglob(name))
    if not paths:
        raise SystemExit(f"ERROR：EXP-09缺少{name}")
    return max(paths, key=lambda p: p.stat().st_mtime_ns)

table_path = latest("ablation_D3.csv")
table = pd.read_csv(table_path)
required_columns = {
    "variant", "seed", "best_checkpoint", "resolved_config", "predictions"
}
missing = required_columns.difference(table.columns)
if missing:
    raise SystemExit(f"ERROR：ablation_D3.csv缺少列：{sorted(missing)}")
for variant in ("B0", "B4", "B5", "B9", "B11"):
    rows = table[
        (table["variant"].astype(str) == variant)
        & (table["seed"].astype(int) == 42)
    ]
    if rows.empty:
        raise SystemExit(f"ERROR：缺少{variant} seed42结果")
    row = rows.iloc[-1]
    for column in ("best_checkpoint", "resolved_config", "predictions"):
        path = Path(str(row[column]))
        if not path.is_file():
            raise SystemExit(f"ERROR：{variant}的{column}不存在：{path}")

reliability = latest("reliability_logits_D2_D3.csv")
rel = pd.read_csv(reliability)
required_rel = {"split", "logits_json", "label", "high_risk"}
missing_rel = required_rel.difference(rel.columns)
if missing_rel:
    raise SystemExit(f"ERROR：可靠性表缺少列：{sorted(missing_rel)}")
if set(rel["split"].astype(str)) != {"D2", "D3"}:
    raise SystemExit("ERROR：可靠性表必须同时包含D2和D3")
latest("gmm_D1.json")
print("SLIM_PREFLIGHT=PASSED")
print(f"ablation_table={table_path}")
print(f"reliability_table={reliability}")
PY

# EXP-08B精简缓存审计：直接从已完成的EXP-09固定结果派生，绝不调用训练。
EXP08B_MANIFEST="$OUTPUT_ROOT/experiments/EXP-08B/run_manifest.json"
if ! is_completed "$EXP08B_MANIFEST" "completed"; then
    CURRENT_STAGE="EXP-08B-cache-audit"
    set_state "RUNNING stage=$CURRENT_STAGE time=$(date -Is)"
    "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY' || exit 2
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
exp09 = root / "experiments" / "EXP-09"
tables = list(exp09.rglob("ablation_D3.csv"))
source = max(tables, key=lambda p: p.stat().st_mtime_ns)
table = pd.read_csv(source)
mapping = {
    "B0": "shared",
    "B4": "independent_line_area",
    "B5": "hard_morphology_split",
    "B11": "shared_pcgrad",
}
selected = []
for variant, name in mapping.items():
    rows = table[
        (table["variant"].astype(str) == variant)
        & (table["seed"].astype(int) == 42)
    ]
    if rows.empty:
        raise SystemExit(f"ERROR：无法派生EXP-08B，缺少{variant} seed42")
    row = rows.iloc[-1].to_dict()
    row["diagnostic_model"] = name
    row["evidence_source"] = "EXP-09 completed fixed-model result"
    selected.append(row)

directory = root / "experiments" / "EXP-08B" / "local_6gb" / "seed_42"
directory.mkdir(parents=True, exist_ok=True)
pd.DataFrame(selected).to_csv(directory / "M2_performance.csv", index=False)
pd.DataFrame([
    {
        "diagnostic_model": row["diagnostic_model"],
        "variant": row["variant"],
        "best_checkpoint": row["best_checkpoint"],
        "resource_protocol": "same_completed_EXP09_protocol",
    }
    for row in selected
]).to_csv(directory / "M2_resources.csv", index=False)
pd.DataFrame([
    {
        "diagnostic_model": row["diagnostic_model"],
        "stratum": "all",
        "mask_ap50": row["mask_ap50"],
    }
    for row in selected
]).to_csv(directory / "M2_morphology_strata.csv", index=False)
(directory / "diagnostic_models.json").write_text(
    json.dumps({
        "models": list(mapping.values()),
        "variants": list(mapping),
        "selection_split": "D2",
        "evaluation_split": "D3",
        "training_performed": False,
        "source_table": str(source),
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
now = datetime.now(timezone.utc).isoformat()
manifest = {
    "schema_version": 1,
    "experiment_id": "EXP-08B",
    "profile": "local_6gb",
    "seed": 42,
    "status": "completed",
    "started_at_utc": now,
    "finished_at_utc": now,
    "produced_outputs": [
        "M2_performance.csv",
        "M2_resources.csv",
        "M2_morphology_strata.csv",
        "diagnostic_models.json",
    ],
    "notes": [
        "Slim-plan cache audit derived only from completed EXP-09 fixed-model outputs.",
        "No training, checkpoint update, or D4 access was performed.",
    ],
    "source_table": str(source),
}
for path in (directory / "run_manifest.json", root / "experiments" / "EXP-08B" / "run_manifest.json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("EXP08B_SLIM_CACHE_AUDIT=COMPLETED")
PY
fi

COMMON_ARGS=(
    --profile local_6gb
    --dataset-root "$DATA_ROOT"
    --output-root "$OUTPUT_ROOT"
    --no-smoke-test
    --no-auto-run-prerequisites
    --resume
)

EXP12_PREFREEZE="$OUTPUT_ROOT/experiments/EXP-12/prefreeze_manifest.json"
if ! is_completed "$EXP12_PREFREEZE" "completed"; then
    run_step "EXP-12_prefreeze" \
        "$PYTHON_BIN" "$RUNNER" "$ENTRY_DIR/exp12_reliability.py" \
        "${COMMON_ARGS[@]}" --stage prefreeze
fi
is_completed "$EXP12_PREFREEZE" "completed" || \
    fail "EXP-12预冻结清单未完成"

EXP10_MANIFEST="$OUTPUT_ROOT/experiments/EXP-10/run_manifest.json"
if ! is_completed "$EXP10_MANIFEST" "completed"; then
    run_step "EXP-10_freeze" \
        "$PYTHON_BIN" "$RUNNER" "$ENTRY_DIR/exp10_freeze.py" \
        "${COMMON_ARGS[@]}"
fi
is_completed "$EXP10_MANIFEST" "completed" || \
    fail "EXP-10冻结清单未完成"

EXP11_MANIFEST="$OUTPUT_ROOT/experiments/EXP-11/run_manifest.json"
if ! is_completed "$EXP11_MANIFEST" "completed_sealed"; then
    EXISTING_D4_STATUS="$(find "$OUTPUT_ROOT/experiments/EXP-11" -type f -name one_shot_status.json -print -quit 2>/dev/null || true)"
    if [[ -n "$EXISTING_D4_STATUS" ]]; then
        fail "发现既有D4一次性状态但未完整封存，禁止自动重跑：$EXISTING_D4_STATUS"
    fi
    run_step "EXP-11_D4_one_shot" \
        env MORPHOROUTE_D4_CONFIRMATION='I UNDERSTAND D4 IS ONE-SHOT' \
        "$PYTHON_BIN" "$RUNNER" "$ENTRY_DIR/exp11_external_test.py" \
        "${COMMON_ARGS[@]}"
fi
is_completed "$EXP11_MANIFEST" "completed_sealed" || \
    fail "EXP-11未达到completed_sealed"

EXP14_MANIFEST="$OUTPUT_ROOT/experiments/EXP-14/run_manifest.json"
if ! is_completed "$EXP14_MANIFEST" "completed"; then
    run_step "EXP-14_efficiency" \
        "$PYTHON_BIN" "$RUNNER" "$ENTRY_DIR/exp14_sequence_efficiency.py" \
        "${COMMON_ARGS[@]}"
fi
is_completed "$EXP14_MANIFEST" "completed" || \
    fail "EXP-14效率验证未完成"

CURRENT_STAGE="final-verification"
"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY' || exit 2
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "experiments"
checks = {
    "EXP-08B": (root / "EXP-08B" / "run_manifest.json", {"completed"}),
    "EXP-12:prefreeze": (root / "EXP-12" / "prefreeze_manifest.json", {"completed"}),
    "EXP-10": (root / "EXP-10" / "run_manifest.json", {"completed"}),
    "EXP-11": (root / "EXP-11" / "run_manifest.json", {"completed_sealed"}),
    "EXP-14": (root / "EXP-14" / "run_manifest.json", {"completed"}),
}
for name, (path, accepted) in checks.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload.get("status"))
    if status not in accepted:
        raise SystemExit(f"ERROR：{name}状态异常：{status}")
    print(f"{name}={status}")
print("SLIM_EXPERIMENT_CHAIN=COMPLETED")
PY

set_state "COMPLETED time=$(date -Is)"
echo "精简版实验链全部完成。"
