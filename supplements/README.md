# 补采包（supplements）

本目录保存论文工程在 AutoDL（RTX 5090 32 GB）服务器上对**已完成实验**做事后审计与补齐的最终代码（v4），以及当时使用的一次性工具。它们**不是**主实验流水线的一部分，也不替代 `19 个独立实验运行入口`。

## 目录

| 目录 | 用途 | 版本说明 |
|---|---|---|
| `exp08ac_fixed_checkpoint_probe_v4_20261010/` | EXP-08A / EXP-08C 固定检查点事后探针：在 B9 已选检查点上采集真实梯度，做 M1 梯度冲突与 M3 分组形态关联 | v4 修正了 7 张合法阴性图被误判为缺失标注的问题；前版本已删除 |
| `exp08b15_supplement_v4_20261010/` | EXP-08B 五模型资源测量 + EXP-15 五种子统计补齐；含压缩 RLE 恢复入口 `recover_statistics.py` | v4 按 COCO 官方压缩 RLE 读取，前版本 v3 只接受整数列表 |
| `exp08b15_supplement_v3_20261010/` | 旧协议与旧代码哈希 | **仅供 v4 的 `recover_statistics.py --old-package` 核对旧 protocol 代码哈希**，不要用它的 `launch_supplement.py` 重新运行 |
| `one_off_tools/` | 一次性服务器工具：`exp07_hungarian_probe.py`（匈牙利匹配数值探针）、`repair_exp08d_memory_and_resume.py`（EXP-08D 内存修复补丁）、`run_slim_experiments_after_exp09.sh`（EXP-09 之后的精简实验链） | 均含服务器绝对路径（`/root/autodl-tmp/...`），仅作历史记录 |

## 运行前提

- 需要原工程目录（`railway_tunnel_experiments`）与其 `outputs/` 历史产物；
- 需要已完成训练得到的检查点、预测与配置；
- 必须使用对应版本的 GPU 环境（v4 的启动器会先跑包内自动化测试与真实 Torch 检查）。

## 启动命令

```bash
# EXP-08A / EXP-08C 事后探针（v4）
python "supplements/exp08ac_fixed_checkpoint_probe_v4_20261010/launch_fixed_checkpoint_probe.py" --mode run

# EXP-08B 资源 + EXP-15 统计补齐（v4）
python -B "supplements/exp08b15_supplement_v4_20261010/recover_statistics.py" --mode run
```

各自目录内的 `README.md` / `EXP08AC_补采启动说明.md` 给出了完整的预检、日志、断点与完成判据。

## 边界

- 只做**事后审计与补齐**：不重建丢失的早期检查点、不重新训练、不修改已完成运行的结果。
- 不读取 D4 图像或 D4 预测，不改变候选选择、冻结清单、E 门控或 EXP-09/08D/12 的既有结果。
- 统计口径与限制写在各自 README 中（例如“阳性且梯度有效图像上的探索性关联”，不能外推为整个 D1 的总体冲突率）。
