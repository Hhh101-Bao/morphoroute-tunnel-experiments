# 运行命令速查（RUN_COMMANDS）

本文件给出所有可直接复制的命令。命令均在仓库根目录执行。

- Windows：PowerShell（`run.ps1` 以 UTF-8 BOM 保存，Windows PowerShell 5.1 可正确读取中文）
- Linux / WSL2 / AutoDL：Bash（`run.sh`）
- 交互式逐步排查：直接调用 `19 个独立实验运行入口` 下的 Python 入口

---

## 0. 环境变量

| 变量 | 作用 | 默认值 |
|---|---|---|
| `MORPHOROUTE_DATA_ROOT` | 严格划分数据集根目录 | `./datasets/铁路隧道多病害实例分割_原始影像严格划分_v1` |
| `MORPHOROUTE_OUTPUT_ROOT` | 实验输出根目录 | `./outputs` |
| `MORPHOROUTE_PYTHON` | 指定 Python 解释器 | Windows：`python`；Linux：`python3`（WSL 下若存在 `~/.morphoroute/venvs/<kind>/bin/python` 则优先使用） |

```powershell
$env:MORPHOROUTE_DATA_ROOT = "D:\data\铁路隧道多病害实例分割_原始影像严格划分_v1"
$env:MORPHOROUTE_OUTPUT_ROOT = "D:\runs\morphoroute"
```

```bash
export MORPHOROUTE_DATA_ROOT="/root/autodl-tmp/铁路隧道多病害实例分割_原始影像严格划分_v1"
export MORPHOROUTE_OUTPUT_ROOT="/root/autodl-tmp/railway_outputs"
```

---

## 1. 安装与自检

```powershell
# Windows: 建立/修复 WSL2 三套隔离环境（main / yolo / stats）
.\run.ps1 env

# 查看 19 个实验入口
.\run.ps1 list

# 只解析配置与依赖，不训练（推荐先跑）
.\run.ps1 dryall

# 单元测试
.\run.ps1 test
```

```bash
# WSL2 内安装三套环境
./run.sh env

# Linux / AutoDL 手动安装
python3 -m venv .venv && . .venv/bin/activate
pip install -r "WSL2 与 Python 依赖文件/requirements-main.txt"
# 再按 CUDA 版本安装 torch / torchvision / mmcv / mmengine / mmdet
python -c "import torch, mmcv, mmengine, mmdet; assert torch.cuda.is_available()"

./run.sh list
./run.sh dryall
./run.sh test
```

---

## 2. 单个实验：三种模式

以 EXP-05（同骨干 Mask2Former 共享主基线）为例，其他实验把编号换成 `00`–`15`（含 `08a/08b/08c/08d`）即可。

```powershell
.\run.ps1 dry   05      # 空运行：解析配置与依赖
.\run.ps1 smoke 05      # 合成数据 + 128×128 + 3 次更新，验证代码链
.\run.ps1 run   05      # 正式实验（Windows 自动转 WSL2）
```

```bash
./run.sh dry   05
./run.sh smoke 05
./run.sh run   05
```

等价的手工命令：

```powershell
python ".\19 个独立实验运行入口\05同骨干 Mask2Former 共享主基线实验.py" --dry-run
python ".\19 个独立实验运行入口\05同骨干 Mask2Former 共享主基线实验.py" --smoke-test
python ".\19 个独立实验运行入口\05同骨干 Mask2Former 共享主基线实验.py" `
  --profile local_6gb `
  --dataset-root "D:\data\铁路隧道多病害实例分割_原始影像严格划分_v1" `
  --output-root "D:\runs\morphoroute"
```

```bash
python "19 个独立实验运行入口/05同骨干 Mask2Former 共享主基线实验.py" --dry-run
python "19 个独立实验运行入口/05同骨干 Mask2Former 共享主基线实验.py" --smoke-test
python "19 个独立实验运行入口/05同骨干 Mask2Former 共享主基线实验.py" \
  --profile autodl_5090_32gb \
  --dataset-root "/root/autodl-tmp/铁路隧道多病害实例分割_原始影像严格划分_v1" \
  --output-root "/root/autodl-tmp/railway_outputs"
```

---

## 3. 按论文顺序运行

前置实验默认自动补齐（`--auto-run-prerequisites`），因此通常只需按下面的顺序手动推进主链：

```powershell
.\run.ps1 run 00    # 预注册与冻结
.\run.ps1 run 01    # 数据清单与审计
.\run.ps1 run 03    # YOLO 严格划分复现（锚点）
.\run.ps1 run 04    # Mask R-CNN 强基线
.\run.ps1 run 05    # Mask2Former 共享主基线（B0）
.\run.ps1 run 06    # 容量/FLOPs 匹配（B2/B3）
.\run.ps1 run 07    # 路由对照（B4–B6）
.\run.ps1 run 09    # B0–B11 完整消融（五种子，耗时最长）
.\run.ps1 run 08a   # M1 梯度冲突
.\run.ps1 run 08c   # M3 形态关联
.\run.ps1 run 08d   # M4 专家干预
.\run.ps1 run 08b   # M2 性能诊断
.\run.ps1 run 12    # 校准与可靠性（自动先跑 prefreeze 阶段）
.\run.ps1 run 10    # 唯一候选选择与 D4 前冻结
.\run.ps1 run 11    # D4 一次性外部测试（需输入确认短语）
.\run.ps1 run 12    # 自动进入 external 阶段
.\run.ps1 run 13    # 长度/宽度/面积量化（需物理尺度元数据）
.\run.ps1 run 14    # 序列推理与效率
.\run.ps1 run 15    # 五种子统计与结论锁定
```

> EXP-11 运行前会校验 EXP-10 冻结清单中的全部 SHA-256，并要求输入：
>
> ```text
> I UNDERSTAND D4 IS ONE-SHOT
> ```
>
> 标记写入后不可重复运行；D4 结果不得用于重新选模、调阈值或改后处理。

---

## 4. 档位与常用参数

```powershell
.\run.ps1 run 09 -Profile local_6gb_fallback   # 512×512、5k updates，仅预实验
.\run.ps1 run 09 -Profile sop_full             # 1024×1024、50k updates 完整 SOP
.\run.ps1 run 09 -Profile autodl_5090_32gb     # 640×640、physical batch=4
```

```bash
./run.sh run 09 --profile local_6gb_fallback
./run.sh run 09 --profile sop_full
./run.sh run 09 --profile autodl_5090_32gb

# 第三个位置参数也可以直接写档位
./run.sh run 09 local_6gb_fallback
```

常用附加参数（直接传给入口）：

```powershell
# 断点恢复（默认开启）
python ".\19 个独立实验运行入口\09B0–B11 完整消融实验.py" --resume

# 关闭断点恢复，从头开始
python ".\19 个独立实验运行入口\09B0–B11 完整消融实验.py" --no-resume

# 只做已有产物的统计/评价，不重新训练
python ".\19 个独立实验运行入口\15五种子统计、配对 Bootstrap 与论文结论锁定.py" --analysis-only

# 指定单个配置
python ".\19 个独立实验运行入口\05同骨干 Mask2Former 共享主基线实验.py" --config ".\三种运行档位的实验配置\RTX 4050 6GB 本机正式实验配置\exp05.json"
```

---

## 5. 辅助工具

```powershell
# 全部实验配置空运行检查（等价于 run.ps1 dryall）
python ".\工程辅助运行脚本\全部实验空运行检查工具.py" --profile local_6gb

# 重新生成四套档位配置（会覆盖三种运行档位的实验配置/*/exp*.json）
python ".\工程辅助运行脚本\三套实验配置生成器.py"

# 生成五类合成测试数据
python ".\工程辅助运行脚本\五类合成测试数据生成器.py"

# 断点恢复（等价于统一运行器 + --resume）
python ".\工程辅助运行脚本\实验断点恢复工具.py" --config <config.json>
```

EXP-06 / EXP-07 / EXP-09 的专用运行与校验脚本（需要完整 GPU 环境）：

```bash
python "工程辅助运行脚本/recover_exp06_b2_postprocess.py" --help
python "工程辅助运行脚本/run_exp07_strict_fp32.py" --help
python "工程辅助运行脚本/run_exp09_b11_true_fp32.py" --help
python "工程辅助运行脚本/verify_exp07_strict_fp32_checkpoint.py" --help
```

---

## 6. 补采包（supplements）

补采包是在 AutoDL 服务器上对已完成实验做事后审计/补齐的独立代码，依赖原工程目录与输出。使用前请阅读对应目录中的 README：

```bash
# EXP-08A / EXP-08C 固定检查点事后探针（v4）
python "supplements/exp08ac_fixed_checkpoint_probe_v4_20261010/launch_fixed_checkpoint_probe.py" --mode run

# EXP-08B 资源 + EXP-15 统计补齐（v4，含压缩 RLE 恢复）
python -B "supplements/exp08b15_supplement_v4_20261010/recover_statistics.py" --mode run
```

---

## 7. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `ModuleNotFoundError: No module named 'pycocotools'` | 数据/评价依赖缺失。安装 `pycocotools==2.0.7`，或使用 `run.ps1 env` 建立完整 WSL2 环境。 |
| `No module named 'torch'` / `mmengine` / `mmdet` | 只做了空运行或环境不全。正式训练需 `main` 环境；见 README 第 4 节。 |
| `数据目录不存在` | 用 `-DataRoot`（Windows）或 `MORPHOROUTE_DATA_ROOT`（Linux）指定数据集。 |
| Windows 下运行直接报 WSL 错误 | 需要 WSL2 + Ubuntu 且 WSL 内 `nvidia-smi` 可用；先执行 `.\run.ps1 env`。 |
| 运行 EXP-11 被拒绝 | 未输入一次性确认短语，或 EXP-10 冻结清单校验失败；失败会保留证据，不自动重试。 |
| 结果与论文数字不一致 | 检查档位（`profile`）、种子、是否使用了 smoke 结果；smoke 产物不能进入统计。 |
