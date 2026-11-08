# RTX 5090 / 32GB AutoDL 运行说明

## 选择档位

上传工程和数据集后，在任意独立实验入口后增加：

```bash
python "19 个独立实验运行入口/05同骨干 Mask2Former 共享主基线实验.py" \
  --profile autodl_5090_32gb \
  --dataset-root /root/autodl-tmp/铁路隧道多病害实例分割_原始影像严格划分_v1 \
  --output-root /root/autodl-tmp/railway_outputs
```

先执行不训练的检查：

```bash
python "19 个独立实验运行入口/05同骨干 Mask2Former 共享主基线实验.py" \
  --profile autodl_5090_32gb \
  --dataset-root /root/autodl-tmp/铁路隧道多病害实例分割_原始影像严格划分_v1 \
  --output-root /root/autodl-tmp/railway_outputs \
  --dry-run
```

## 固定参数

- 输入：640×640。
- physical batch：4；effective batch：8；梯度累积：2。
- AMP：开启；梯度检查点：关闭。
- queries：64；将 JSON 中 `query_count` 改回 100 可回退。
- EXP-05/06：20k 起判断收敛。
- EXP-07/08B/09：30k 起判断收敛。
- 最多 50k；连续两次 D2 验证无至少 0.001 的改善后正常停止。
- 如需跑满，在对应 JSON 中设置 `release_budget_gate=true` 并保留 `--resume`。

## 显存回退

如果 B11 或资源匹配变体出现 CUDA OOM，只修改：

```json
{
  "physical_batch": 2,
  "effective_batch": 8,
  "gradient_checkpointing": true
}
```

不要修改输入尺寸、损失、数据划分或优化器。该回退仍保持 effective batch=8，但梯度累积会从 2 变为 4。

## RTX 5090 环境要求

RTX 5090 不能使用项目旧锁文件中的 PyTorch 2.1.2 + CUDA 11.8。应选择 AutoDL 的 CUDA 12.8 镜像，并使用支持 Blackwell 的 PyTorch 版本；随后安装与该 PyTorch/CUDA 组合匹配的 MMCV CUDA 扩展、MMEngine 0.10.x 和 MMDetection 3.3.x。运行前必须确认：

```bash
python -c "import torch, mmcv, mmengine, mmdet; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); assert torch.cuda.is_available()"
```

不要在 RTX 5090 实例中调用旧的 WSL2 CUDA 11.8 自动安装脚本。

