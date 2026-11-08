# EXP-06 B2 无重训恢复与 B3 自动继续

## 结论

该补丁只改变预测掩膜的内存表示、预测分片、评价复用和恢复控制流，不改变模型、权重、D1/D2/D3、训练预算、budget gate、输入尺寸、query 数、seed 或 profile。B2 使用现有 best checkpoint，只重做 D2/D3 后处理；完成后原 EXP-06 入口复用 B2 cache，并从零开始 B3。

## 1. 定义路径并确认旧进程已结束

```bash
PROJECT_ROOT="/root/autodl-tmp/railway_tunnel_experiments"
DATA_ROOT="/root/autodl-tmp/railway_dataset_v1/铁路隧道多病害实例分割_原始影像严格划分_v1"
OUTPUT_ROOT="$PROJECT_ROOT/outputs"
UPLOAD_DIR="/root/autodl-tmp/Untitled Folder"
LOG_DIR="$OUTPUT_ROOT/autodl_logs"
B2_DIR="$OUTPUT_ROOT/experiments/EXP-06/local_6gb/seed_42/runs/B2/seed_42"

mkdir -p "$LOG_DIR"

pgrep -af 'exp06_resource_matched.py|recover_exp06_b2_postprocess.py' \
  || echo "没有残留EXP-06进程"

ls -lh \
  "$B2_DIR/iter_200000.pth" \
  "$B2_DIR/best_coco_segm_mAP_iter_200000.pth"
```

如果仍有未知 EXP-06 进程，不要安装补丁，先查明该 PID。

## 2. 校验上传包

把补丁 `.tar.gz` 和同名 `.sha256` 上传到 `$UPLOAD_DIR` 后执行：

```bash
cd "$UPLOAD_DIR"
sha256sum -c exp06_rle_streaming_recovery_20260819.sha256
tar -tzf exp06_rle_streaming_recovery_20260819.tar.gz
```

校验必须显示 `OK`。归档列表不得包含绝对路径或 `..`。

## 3. 备份被覆盖的旧源码并安装

```bash
PATCH="$UPLOAD_DIR/exp06_rle_streaming_recovery_20260819.tar.gz"
PATCH_LIST="/tmp/EXP06_RLE_RECOVERY_PATCH_FILES.txt"
BACKUP_DIR="$OUTPUT_ROOT/source_backups/exp06_rle_$(date +%Y%m%d_%H%M%S)"

tar -xOf "$PATCH" docs/EXP06_RLE_RECOVERY_PATCH_FILES.txt > "$PATCH_LIST"
mkdir -p "$BACKUP_DIR"

while IFS= read -r relative; do
    source_path="$PROJECT_ROOT/$relative"
    if [ -f "$source_path" ]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$relative")"
        cp -a "$source_path" "$BACKUP_DIR/$relative"
    fi
done < "$PATCH_LIST"

tar -xzf "$PATCH" -C "$PROJECT_ROOT"
echo "源码备份：$BACKUP_DIR"
```

## 4. 运行服务器侧测试

```bash
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

for test_file in \
  "61 项自动化测试/test_rle_mask_codec.py" \
  "61 项自动化测试/test_prediction_bundle_roundtrip.py" \
  "61 项自动化测试/test_evaluation_pipeline.py" \
  "61 项自动化测试/test_prediction_shards.py" \
  "61 项自动化测试/test_parallel_postprocessing.py" \
  "61 项自动化测试/test_exp06_postprocess_recovery.py"
do
    python "$test_file" || exit 1
done

python -m compileall -q src 工程辅助运行脚本
echo "补丁测试和语法检查通过"
```

## 5. 恢复预检

```bash
RECOVERY_SCRIPT="$PROJECT_ROOT/工程辅助运行脚本/recover_exp06_b2_postprocess.py"
BEST="$B2_DIR/best_coco_segm_mAP_iter_200000.pth"

python "$RECOVERY_SCRIPT" \
  --project-root "$PROJECT_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --checkpoint "$BEST" \
  --dry-run
```

输出必须同时包含：

```text
"experiment_id": "EXP-06"
"variant_id": "B2"
"profile": "local_6gb"
"seed": 42
"training_action": "forbidden"
```

任何 hash/fingerprint 错误都必须先停止排查，不能绕过。

## 6. 一键串行运行 B2 后处理和 B3

该方式在 B2 后处理成功后立即启动 B3，避免实例空闲计费；B2 后处理失败时不会启动 B3。

```bash
ENTRY_DIR="$PROJECT_ROOT/19 个独立实验运行入口"
PIPE_LOG="$LOG_DIR/EXP-06_recovery_and_B3_$(date +%Y%m%d_%H%M%S).log"

export PROJECT_ROOT DATA_ROOT OUTPUT_ROOT LOG_DIR B2_DIR BEST RECOVERY_SCRIPT ENTRY_DIR
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

sha256sum \
  "$B2_DIR/iter_200000.pth" \
  "$B2_DIR/best_coco_segm_mAP_iter_200000.pth" \
  > "$LOG_DIR/EXP-06_B2_checkpoints_before_B3.sha256"

nohup bash -c '
python "$RECOVERY_SCRIPT" \
  --project-root "$PROJECT_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --checkpoint "$BEST"
recovery_code=$?
printf "\nEXP06_B2_RECOVERY_EXIT_CODE=%s\n" "$recovery_code"
if [ "$recovery_code" -ne 0 ]; then
    echo "B2后处理失败，不启动B3。"
    exit "$recovery_code"
fi

python "$ENTRY_DIR/exp06_resource_matched.py" \
  --profile local_6gb \
  --dataset-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --no-smoke-test \
  --auto-run-prerequisites \
  --no-resume
exp06_code=$?
printf "\nEXP-06_EXIT_CODE=%s\n" "$exp06_code"
exit "$exp06_code"
' > "$PIPE_LOG" 2>&1 &

PIPE_PID=$!
printf '%s\n' "$PIPE_PID" > "$LOG_DIR/EXP-06_recovery_and_B3.pid"
printf '%s\n' "$PIPE_LOG" > "$LOG_DIR/EXP-06_recovery_and_B3.current_log"

echo "后台PID=$PIPE_PID"
echo "日志=$PIPE_LOG"
```

启动后可以关闭 SSH/JupyterLab 网页，但不能关闭 AutoDL 实例；关闭实例会中断任务。

## 7. 监控

```bash
PIPE_PID="$(cat "$LOG_DIR/EXP-06_recovery_and_B3.pid")"
PIPE_LOG="$(cat "$LOG_DIR/EXP-06_recovery_and_B3.current_log")"

ps -p "$PIPE_PID" -o pid,etime,time,%cpu,%mem,rss,stat,cmd
tail -f "$PIPE_LOG"
```

`tail -f` 中按 `Ctrl+C` 只退出日志查看，不会停止后台任务。

另一个终端可执行：

```bash
watch -n 5 nvidia-smi
```

查看分片进度：

```bash
find "$B2_DIR/prediction_shards" -type f -name 'part-*.json' | wc -l
du -sh "$B2_DIR/prediction_shards" 2>/dev/null
```

正常日志每 25 张图片出现一次：

```text
MMDetection D2 prediction [25/1158] ... RSS ...MiB
MMDetection D3 prediction [25/1160] ... RSS ...MiB
```

## 8. 确认 B2 被复用、B3 正常开始

```bash
PIPE_LOG="$(cat "$LOG_DIR/EXP-06_recovery_and_B3.current_log")"

grep -E \
  'EXP06_B2_RECOVERY_EXIT_CODE|Reusing completed MMDetection cache|morphoroute_B2|runs/B3|Iter\(train\)' \
  "$PIPE_LOG" | tail -40
```

必须先看到：

```text
EXP06_B2_RECOVERY_EXIT_CODE=0
Reusing completed MMDetection cache: ... model_kind=morphoroute_B2
```

随后 B3 应从首次训练 iteration 开始。不能出现新的 B2 `Iter(train)`。

## 9. 最终验收

```bash
PIPE_LOG="$(cat "$LOG_DIR/EXP-06_recovery_and_B3.current_log")"
AUDIT="$B2_DIR/recovery_audit.json"
MANIFEST="$OUTPUT_ROOT/experiments/EXP-06/local_6gb/seed_42/run_manifest.json"

tail -100 "$PIPE_LOG"
sha256sum -c "$LOG_DIR/EXP-06_B2_checkpoints_before_B3.sha256"

python - "$AUDIT" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

print("B2 recovery status =", audit.get("status"))
print("B2 training invoked =", audit.get("b2_training_invoked"))
print("D2 images =", audit.get("d2_image_count"))
print("D3 images =", audit.get("d3_image_count"))
print("EXP-06 status =", manifest.get("status"))

assert audit.get("status") == "completed"
assert audit.get("b2_training_invoked") is False
assert audit.get("d2_image_count") == 1158
assert audit.get("d3_image_count") == 1160
assert manifest.get("status") == "completed"
PY

grep -E 'EXP-06_EXIT_CODE=0' "$PIPE_LOG"
```

只有以上检查全部通过，EXP-06 才算完整结束。
