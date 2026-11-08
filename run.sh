#!/usr/bin/env bash
# MorphoRoute-Tunnel 一键运行脚本（Linux / WSL2 / AutoDL）
#
# 用法:
#   ./run.sh list
#   ./run.sh dry 05
#   ./run.sh dryall
#   ./run.sh smoke 05
#   ./run.sh run 09
#   ./run.sh test
#   ./run.sh env
#
# 环境变量:
#   MORPHOROUTE_DATA_ROOT    严格划分数据集目录
#   MORPHOROUTE_OUTPUT_ROOT  实验输出目录（默认 ./outputs）
#   MORPHOROUTE_PYTHON       指定 Python 解释器
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY_DIR="$ROOT/19 个独立实验运行入口"
TEST_DIR="$ROOT/61 项自动化测试"

ACTION="${1:-list}"
shift || true
EXPERIMENT="${1:-05}"
shift || true

PROFILE="${MORPHOROUTE_PROFILE:-local_6gb}"
CLI_DATA_ROOT=""
CLI_OUTPUT_ROOT=""
POSITIONAL_PROFILE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="${2:?--profile 需要取值}"; shift 2 ;;
    --data-root) CLI_DATA_ROOT="${2:?--data-root 需要取值}"; shift 2 ;;
    --output-root) CLI_OUTPUT_ROOT="${2:?--output-root 需要取值}"; shift 2 ;;
    -*) echo "未知参数: $1" >&2; exit 2 ;;
    *) POSITIONAL_PROFILE="$1"; shift ;;
  esac
done
if [ -n "$POSITIONAL_PROFILE" ]; then
  PROFILE="$POSITIONAL_PROFILE"
fi

DATA_ROOT="${CLI_DATA_ROOT:-${MORPHOROUTE_DATA_ROOT:-$ROOT/datasets/铁路隧道多病害实例分割_原始影像严格划分_v1}}"
OUTPUT_ROOT="${CLI_OUTPUT_ROOT:-${MORPHOROUTE_OUTPUT_ROOT:-$ROOT/outputs}}"

ALL_KEYS="00 01 02 03 04 05 06 07 08a 08b 08c 08d 09 10 11 12 13 14 15"

normalize() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  value="${value#exp}"
  value="${value#-}"
  value="${value#_}"
  case "$value" in
    [0-9]) value="0${value}" ;;
  esac
  printf '%s' "$value"
}

english_name() {
  case "$1" in
    00) printf '%s' 'exp00_preregister.py' ;;
    01) printf '%s' 'exp01_data_audit.py' ;;
    02) printf '%s' 'exp02_yolo_original.py' ;;
    03) printf '%s' 'exp03_yolo_strict.py' ;;
    04) printf '%s' 'exp04_mask_rcnn.py' ;;
    05) printf '%s' 'exp05_mask2former.py' ;;
    06) printf '%s' 'exp06_resource_matched.py' ;;
    07) printf '%s' 'exp07_routing_controls.py' ;;
    08a) printf '%s' 'exp08a_gradient_conflict.py' ;;
    08b) printf '%s' 'exp08b_performance_diagnosis.py' ;;
    08c) printf '%s' 'exp08c_morphology_association.py' ;;
    08d) printf '%s' 'exp08d_expert_intervention.py' ;;
    09) printf '%s' 'exp09_ablation.py' ;;
    10) printf '%s' 'exp10_freeze.py' ;;
    11) printf '%s' 'exp11_external_test.py' ;;
    12) printf '%s' 'exp12_reliability.py' ;;
    13) printf '%s' 'exp13_quantification.py' ;;
    14) printf '%s' 'exp14_sequence_efficiency.py' ;;
    15) printf '%s' 'exp15_statistics.py' ;;
    *) printf '%s' '' ;;
  esac
}

chinese_name() {
  case "$1" in
    00) printf '%s' '00预注册与实验冻结.py' ;;
    01) printf '%s' '01数据清单、去重与五类完整性审计.py' ;;
    02) printf '%s' '02YOLOv8n-seg 原实验条件复现.py' ;;
    03) printf '%s' '03YOLOv8n-seg 严格划分纠正复现.py' ;;
    04) printf '%s' '04Mask R-CNN 两阶段强基线实验.py' ;;
    05) printf '%s' '05同骨干 Mask2Former 共享主基线实验.py' ;;
    06) printf '%s' '06容量匹配与 FLOPs 匹配共享模型实验.py' ;;
    07) printf '%s' '07普通 MoE、类别硬路由与形态路由对照实验.py' ;;
    08a) printf '%s' '08A梯度冲突诊断 M1.py' ;;
    08b) printf '%s' '08B共享、独立、硬拆分与 PCGrad 性能诊断 M2.py' ;;
    08c) printf '%s' '08C梯度冲突与连续形态关联 M3.py' ;;
    08d) printf '%s' '08D专家交换、关闭与输入交换 M4.py' ;;
    09) printf '%s' '09B0–B11 完整消融实验.py' ;;
    10) printf '%s' '10唯一候选选择与 D4 前冻结.py' ;;
    11) printf '%s' '11D4 一次性外部测试与效能门 E.py' ;;
    12) printf '%s' '12校准、不确定性与选择性人工复核.py' ;;
    13) printf '%s' '13病害长度、正交宽度与面积量化.py' ;;
    14) printf '%s' '14完整序列推理与效率测试.py' ;;
    15) printf '%s' '15五种子统计、配对 Bootstrap 与论文结论锁定.py' ;;
    *) printf '%s' '' ;;
  esac
}

kind_for() {
  case "$1" in
    02|03) printf '%s' 'yolo' ;;
    00|01|08c|12|13|15) printf '%s' 'stats' ;;
    *) printf '%s' 'main' ;;
  esac
}

resolve_python() {
  local experiment_id="$1"
  if [ -n "${MORPHOROUTE_PYTHON:-}" ]; then
    printf '%s' "$MORPHOROUTE_PYTHON"
    return 0
  fi
  local venv="$HOME/.morphoroute/venvs/$(kind_for "$experiment_id")/bin/python"
  if [ -x "$venv" ]; then
    printf '%s' "$venv"
    return 0
  fi
  printf '%s' 'python3'
}

resolve_entry() {
  local id name
  id="$(normalize "$1")"
  name="$(english_name "$id")"
  if [ -z "$name" ]; then
    printf '无法识别的实验编号: %s\n可用编号: EXP-00 … EXP-15（含 EXP-08A/08B/08C/08D）\n' "$1" >&2
    exit 2
  fi
  printf '%s/%s' "$ENTRY_DIR" "$name"
}

show_command() {
  local formatted="" token
  for token in "$@"; do
    case "$token" in
      *[[:space:]]*) formatted="$formatted \"$token\"" ;;
      *) formatted="$formatted $token" ;;
    esac
  done
  printf '>%s\n' "$formatted"
}

echo "MorphoRoute-Tunnel 实验工程"
echo "  工程根目录 : $ROOT"
echo "  档位       : $PROFILE"
echo "  数据目录   : $DATA_ROOT"
echo "  输出目录   : $OUTPUT_ROOT"

case "$ACTION" in
  list)
    echo
    for key in $ALL_KEYS; do
      printf 'EXP-%s  %s   (%s)\n' "$(printf '%s' "$key" | tr '[:lower:]' '[:upper:]')" "$(chinese_name "$key")" "$(english_name "$key")"
    done
    echo
    echo "示例: ./run.sh dry 05     ./run.sh smoke 05     ./run.sh run 09"
    ;;

  dry)
    ID="$(normalize "$EXPERIMENT")"
    ENTRY="$(resolve_entry "$EXPERIMENT")"
    PYTHON="$(resolve_python "$ID")"
    show_command "$PYTHON" "$ENTRY" --profile "$PROFILE" --dataset-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --dry-run --no-auto-run-prerequisites
    "$PYTHON" "$ENTRY" --profile "$PROFILE" --dataset-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --dry-run --no-auto-run-prerequisites
    ;;

  dryall)
    PYTHON="$(resolve_python 05)"
    failures=""
    for key in $ALL_KEYS; do
      entry="$ENTRY_DIR/$(english_name "$key")"
      if output="$("$PYTHON" "$entry" --profile "$PROFILE" --dataset-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --dry-run --no-auto-run-prerequisites 2>&1)"; then
        echo "PASS  EXP-$(printf '%s' "$key" | tr '[:lower:]' '[:upper:]')"
      else
        echo "FAIL  EXP-$(printf '%s' "$key" | tr '[:lower:]' '[:upper:]')  $(printf '%s' "$output" | grep -m 1 '^ERROR' || true)"
        failures="$failures $key"
      fi
    done
    if [ -n "$failures" ]; then
      echo
      echo "空运行失败:$failures"
      exit 1
    fi
    echo
    echo "全部 19 个实验入口空运行通过。"
    ;;

  smoke)
    ID="$(normalize "$EXPERIMENT")"
    ENTRY="$(resolve_entry "$EXPERIMENT")"
    PYTHON="$(resolve_python "$ID")"
    echo "提示: smoke 只验证代码链，结果写入 \$OUTPUT_ROOT/smoke_runs，不能进入论文统计。"
    show_command "$PYTHON" "$ENTRY" --profile "$PROFILE" --dataset-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --smoke-test
    "$PYTHON" "$ENTRY" --profile "$PROFILE" --dataset-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --smoke-test
    ;;

  run)
    ID="$(normalize "$EXPERIMENT")"
    ENTRY="$(resolve_entry "$EXPERIMENT")"
    PYTHON="$(resolve_python "$ID")"
    if [ ! -d "$DATA_ROOT" ]; then
      echo "数据目录不存在: $DATA_ROOT" >&2
      echo "请设置 MORPHOROUTE_DATA_ROOT 或修改脚本中的 DATA_ROOT。" >&2
      exit 2
    fi
    show_command "$PYTHON" "$ENTRY" --profile "$PROFILE" --dataset-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
    "$PYTHON" "$ENTRY" --profile "$PROFILE" --dataset-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
    ;;

  test)
    PYTHON="$(resolve_python 05)"
    show_command "$PYTHON" -m unittest discover -s "$TEST_DIR" -t "$TEST_DIR" -v
    "$PYTHON" -m unittest discover -s "$TEST_DIR" -t "$TEST_DIR" -v
    ;;

  env)
    if grep -qi microsoft /proc/version 2>/dev/null; then
      show_command bash "$ROOT/工程辅助运行脚本/WSL2 实验环境安装脚本.sh" --kind all
      bash "$ROOT/工程辅助运行脚本/WSL2 实验环境安装脚本.sh" --kind all
    else
      echo "env 动作只在 WSL2 Ubuntu 内可用；Linux/AutoDL 请直接安装依赖：" >&2
      echo "  pip install -r \"$ROOT/WSL2 与 Python 依赖文件/requirements-main.txt\"" >&2
      exit 2
    fi
    ;;

  *)
    echo "未知动作: $ACTION" >&2
    echo "可用动作: list | dry | dryall | smoke | run | test | env" >&2
    exit 2
    ;;
esac
