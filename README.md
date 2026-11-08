# MorphoRoute-Tunnel：隧道和桥梁衬砌多病害实例分割实验工程

**MorphoRoute-Tunnel** 是一套可审计、可复现的隧道和桥梁衬砌病害实例分割实验工程：把论文 SOP 拆成 19 个可独立启动的实验入口，共享同一套模型、路由、训练、评价与统计组件，并在数据划分、冻结、一次性外部测试和统计检验上保留完整的证据链。

> An auditable, reproducible experiment toolkit for railway tunnel lining multi-disease instance segmentation. It splits the paper protocol into 19 standalone experiment entries that share one model / routing / training / evaluation codebase, with hash-audited data splits, a one-shot frozen external test, and paired Bootstrap / Holm statistics.

---

## 1. 主要特性

- **19 个独立实验入口**：打开任意一个中文入口文件即可运行；程序自动读取配置、检查前置实验、补齐缺失步骤、进入对应隔离环境并恢复断点。
- **三层环境隔离**：`main`（MMDetection / Mask2Former）、`yolo`（Ultralytics）、`stats`（统计与绘图），互不污染。
- **四种运行档位**：`local_6gb`、`local_6gb_fallback`、`sop_full`、`autodl_5090_32gb`。
- **严格数据划分**：D1 训练、D2 选模/校准、D3 内部确认、D4 一次性外部测试；D4 只允许 EXP-11 读取一次。
- **可审计**：每次运行写 `run_manifest.json`（配置、档位、seed、状态、哈希）；EXP-10 冻结代码、配置、环境、阈值、校准器与评价器的 SHA-256。
- **统一评价**：Box/Mask AP50、AP75、AP@[.50:.95]、逐类 AP、最差类 AP，以及裂缝结构指标（clDice、断裂率、连通分量误差）与面状病害指标（Boundary-F1、HD95、相对面积误差）。
- **附带补采包**：EXP-08A/08C 固定检查点事后探针、EXP-08B 资源测量与 EXP-15 统计补齐的最终版本（v4），见 [`supplements/`](supplements/README.md)。

---

## 2. 仓库结构

| 路径 | 内容 |
|---|---|
| `src/morphoroute/` | 共享代码包：配置注册、数据审计、模型与路由、MMDetection 扩展、三阶段训练、运行时适配、指标、评价、统计、可视化、机制分析 |
| `19 个独立实验运行入口/` | 19 个中文入口 + 19 个同名英文入口 + `_entry.py`（统一启动逻辑） |
| `61 项自动化测试/` | 54 个测试模块与 fixtures（历史目录名，实际用例数以运行结果为准） |
| `三种运行档位的实验配置/` | 4 个档位 × 19 个实验的 JSON 配置 |
| `工程辅助运行脚本/` | 环境安装、配置生成、合成数据、断点恢复、EXP-06/07/09 专用运行与校验脚本 |
| `WSL2 与 Python 依赖文件/` | `requirements-*.txt` 与 `environment-lock-*.json`（真实环境快照） |
| `docs/` | 运行指南、实验—输出对照表、AutoDL 说明、设计与计划文档、数据集说明、发布说明 |
| `verification/` | 早期任务的验收记录（入口契约、环境转交、依赖图等） |
| `已生成的真实数据审计验证结果/` | EXP-00 预注册与 EXP-01 真实数据审计的产物 |
| `supplements/` | 事后补采包（v4）与一次性服务器工具 |
| `run.ps1` / `run.cmd` / `run.sh` | 一键运行命令文件（Windows / Linux） |
| `pyproject.toml` | 工程元数据（不依赖 `pip install` 运行） |

> 目录名沿用论文工程的中文交付命名；`morphoroute.core.layout.ProjectLayout` 同时兼容中英文两套目录名，因此改回 `experiments/`、`tests/`、`scripts/` 等英文名也能运行。

---

## 3. 数据集（不随仓库分发）

五类病害：**裂缝（crack）、渗漏水（water_leakage）、剥落（spalling）、泛碱（efflorescence）、露筋（exposed_rebar）**。

数据划分基于 TACK（TA/TB/TC）与 DACL 原始文件，按组划分，逐字节复制，未做缩放、裁剪、增强或格式转换：

| 逻辑划分 | 图像数 | 用途 | 说明 |
|---|---:|---|---|
| `train`（D1） | 5050 | 训练 | TACK TA/TB + DACL train |
| `val`（D2） | 1158 | 选模、校准 | 与 D1 同源、按组隔离 |
| `test_internal`（D3） | 1160 | 内部确认 | 与 D1/D2 同源、按组隔离 |
| `D4` | 2002 | 一次性外部测试 | TACK TC 未见隧道 + DACL 官方 validation |

合计 9370 张。数据完整性审计结果：缺失/多余图像 0、缺失/多余标注 0、源哈希失败 0、TACK 分组泄漏 0、精确重复泄漏 0，且每个逻辑划分都包含全部五类。

目录结构（`--dataset-root` 指向该根目录）：

```text
铁路隧道多病害实例分割_原始影像严格划分_v1/
├─ metadata/
│  ├─ split_manifest.csv        # 文件、来源、分组、类别、SHA-256 的权威清单
│  ├─ class_mapping.json
│  ├─ split_summary.json
│  └─ README_划分说明.txt
├─ D1_D3_internal/
│  ├─ train/{images, annotations_original}/
│  ├─ val/{images, annotations_original}/
│  └─ test_internal/{images, annotations_original}/
└─ D4_external_frozen/
   ├─ TC_unseen_tunnel/{images, annotations_original}/
   └─ DACL_external/{images, annotations_original}/
```

- TACK 标注保留原始 PNG 语义掩膜；DACL 标注保留原始 JSON 多边形。
- TC 子域本身没有剥落和露筋，D4 的五类完整性由 TC + DACL_external 组合保证，评价时须同时报告两个子域。
- 数据集体积大且受来源许可约束，**不包含在本仓库中**，请按论文的数据可得性声明获取。更多说明见 [`docs/数据集说明.md`](docs/数据集说明.md)。

---

## 4. 环境准备

### 4.1 Windows + WSL2（`local_6gb` / `local_6gb_fallback`）

1. 安装 WSL2 + Ubuntu，并确认 `nvidia-smi` 在 WSL 内可用。
2. 建立三个隔离环境（会在 `~/.morphoroute/venvs` 下创建 `main`/`yolo`/`stats`，并写入 `environment-lock-*.json`）：

```powershell
.\run.ps1 env
# 等价于
python ".\工程辅助运行脚本\实验环境自动检测与安装.py" --kind all
```

Windows 上**非 `--dry-run`** 的运行会由入口自动通过 `wsl.exe` 转入 WSL2 对应虚拟环境执行，Windows 侧只负责调度与前置实验，不需要安装 PyTorch/MMDetection。

### 4.2 Linux / AutoDL（`autodl_5090_32gb`）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r "WSL2 与 Python 依赖文件/requirements-main.txt"
# 再按镜像的 CUDA 版本安装 PyTorch 与匹配的 MMCV / MMEngine / MMDetection
python -c "import torch, mmcv, mmengine, mmdet; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
```

RTX 5090 需要支持 Blackwell 的 PyTorch 与 CUDA 12.8 组合，**不要**在 5090 实例中使用 CUDA 11.8 的旧安装脚本；详见 [`docs/RTX5090_AutoDL_运行说明.md`](docs/RTX5090_AutoDL_运行说明.md)。

### 4.3 依赖分组

| 环境 | 依赖文件 | 说明 |
|---|---|---|
| main | `requirements-main.txt` | numpy/pandas/opencv/pycocotools/mmengine/mmdet/openpyxl（mmcv 由安装脚本按 CUDA 版本单独安装） |
| yolo | `requirements-yolo.txt` | ultralytics 8.3.0 等 |
| stats | `requirements-stats.txt` | scipy/scikit-learn/scikit-image/statsmodels/matplotlib/seaborn/pyarrow 等 |

---

## 5. 快速开始

所有命令都在仓库根目录执行。至少先跑一次空运行（不训练、不读 D4）：

```powershell
# Windows
.\run.ps1 list                 # 查看 19 个实验
.\run.ps1 dryall               # 全部入口空运行检查
.\run.ps1 dry 05               # 单个实验空运行（EXP-05）
.\run.ps1 smoke 05             # 合成数据 128×128、3 次更新的代码链验证
.\run.ps1 run 05               # 正式实验（自动转入 WSL2）
.\run.ps1 test                 # 运行单元测试
```

```bash
# Linux / WSL2 / AutoDL
./run.sh list
./run.sh dryall
./run.sh dry 05
./run.sh smoke 05
./run.sh run 05
./run.sh test
```

指定数据集与输出目录（推荐，避免依赖作者机器上的默认路径）：

```powershell
.\run.ps1 run 05 -DataRoot "D:\data\铁路隧道多病害实例分割_原始影像严格划分_v1" -OutputRoot "D:\runs\morphoroute"
```

```bash
export MORPHOROUTE_DATA_ROOT="/root/autodl-tmp/铁路隧道多病害实例分割_原始影像严格划分_v1"
export MORPHOROUTE_OUTPUT_ROOT="/root/autodl-tmp/railway_outputs"
./run.sh run 09 --profile autodl_5090_32gb
```

### 直接调用入口（不经脚本）

每个入口都是完整的独立程序：

```powershell
python ".\19 个独立实验运行入口\05同骨干 Mask2Former 共享主基线实验.py" --dry-run
python ".\19 个独立实验运行入口\09B0–B11 完整消融实验.py" --profile local_6gb
python ".\19 个独立实验运行入口\05同骨干 Mask2Former 共享主基线实验.py" --smoke-test
```

> 注意：入口文件内置的默认数据路径是作者机器上的 `E:\桌面\铁路隧道多病害实例分割_原始影像严格划分_v1`（见 `morphoroute.core.standalone.AUTHORITATIVE_DATASET_ROOT`）。在别的机器上请始终通过 `--dataset-root`、`-DataRoot` 或 `MORPHOROUTE_DATA_ROOT` 指定数据集；`run.ps1` / `run.sh` 已经替你固定传入该参数。

入口参数（`19 个独立实验运行入口/_entry.py` 定义）：

| 参数 | 含义 |
|---|---|
| `--profile` | `local_6gb`（默认）/ `local_6gb_fallback` / `sop_full` / `autodl_5090_32gb` |
| `--dataset-root` | 严格划分数据集根目录（覆盖入口文件里的默认值） |
| `--output-root` | 输出根目录（默认 `./outputs`） |
| `--config` | 指定单个实验 JSON 配置 |
| `--dry-run` | 只解析配置与依赖，不训练 |
| `--smoke-test` | 五类合成数据、128×128、3 次优化器更新，验证代码链 |
| `--stage` | 仅 EXP-12 使用：`prefreeze` / `external`（通常自动选择） |
| `--[no-]auto-run-prerequisites` | 是否自动执行缺失的前置实验（默认开启） |
| `--[no-]resume` | 是否从兼容检查点恢复（默认开启） |

更多命令示例见 [`RUN_COMMANDS.md`](RUN_COMMANDS.md)。

---

## 6. 实验清单

| 实验 | 中文入口 | 内容 | 前置 | 主要输出 |
|---|---|---|---|---|
| EXP-00 | `00预注册与实验冻结.py` | 预注册与实验冻结 | — | 预注册、冻结注册表、分析计划 |
| EXP-01 | `01数据清单、去重与五类完整性审计.py` | 数据清单、去重、完整性审计 | EXP-00 | 完整性、重复、跨划分交集、哈希审计 |
| EXP-02 | `02YOLOv8n-seg 原实验条件复现.py` | YOLOv8n-seg 原条件复现（独立锚点） | EXP-00 | 原条件指标、复现来源、误差审计 |
| EXP-03 | `03YOLOv8n-seg 严格划分纠正复现.py` | YOLOv8n-seg 严格划分复现 | EXP-01 | 严格划分指标、D3 预测、划分效应 |
| EXP-04 | `04Mask R-CNN 两阶段强基线实验.py` | Mask R-CNN 强基线 | EXP-01 | D3 预测、严格 AP/结构指标、资源 |
| EXP-05 | `05同骨干 Mask2Former 共享主基线实验.py` | 同骨干 Mask2Former 共享主基线（B0） | EXP-01 | B0 预测、指标、共享参数列表 |
| EXP-06 | `06容量匹配与 FLOPs 匹配共享模型实验.py` | 容量/FLOPs 匹配（B2/B3） | EXP-05 | 候选搜索、匹配误差、性能—资源表 |
| EXP-07 | `07普通 MoE、类别硬路由与形态路由对照实验.py` | 普通 MoE、硬路由、形态路由（B4–B6） | EXP-05 | 路由诊断、D1 GMM、指标 |
| EXP-08A | `08A梯度冲突诊断 M1.py` | 梯度冲突诊断（M1） | EXP-09 | 梯度余弦、敏感性 |
| EXP-08B | `08B共享、独立、硬拆分与 PCGrad 性能诊断 M2.py` | 性能诊断（M2） | EXP-05 | 四诊断模型性能、分层、资源 |
| EXP-08C | `08C梯度冲突与连续形态关联 M3.py` | 形态关联（M3） | EXP-08A | 控制类别/面积后的关联 |
| EXP-08D | `08D专家交换、关闭与输入交换 M4.py` | 专家干预（M4） | EXP-09 | 六种冻结干预、配对效应 |
| EXP-09 | `09B0–B11 完整消融实验.py` | B0–B11 完整消融（五种子） | EXP-07 | 消融总表、配置差异、D2/D3 |
| EXP-10 | `10唯一候选选择与 D4 前冻结.py` | 唯一候选选择与 D4 前冻结 | EXP-03/04/05/06/07/09/12:prefreeze | D4 冻结清单（SHA-256） |
| EXP-11 | `11D4 一次性外部测试与效能门 E.py` | D4 一次性外部测试与 E 门 | EXP-10 | D4 合并/分域指标、E 判定 |
| EXP-12 | `12校准、不确定性与选择性人工复核.py` | 校准、不确定性、选择性复核 | EXP-12:prefreeze、EXP-11 | 温度校准、风险—覆盖 |
| EXP-13 | `13病害长度、正交宽度与面积量化.py` | 长度、正交宽度、面积量化 | EXP-10 | 量化清单、预测/参考测量、误差 |
| EXP-14 | `14完整序列推理与效率测试.py` | 完整序列推理与效率 | EXP-10 | 序列时延、分阶段时延、激活分析 |
| EXP-15 | `15五种子统计、配对 Bootstrap 与论文结论锁定.py` | 五种子统计与结论锁定 | EXP-11/12/13/14 | 最终表图、统计结果、结论分支 |

依赖链：`EXP-00 → EXP-01 → 强基线/路由对照 → EXP-09 → EXP-12:prefreeze → EXP-10 → EXP-11 → EXP-12:external → EXP-15`。

完整对照见 [`docs/实验与输出对照表.md`](docs/实验与输出对照表.md) 与 [`docs/实验运行指南.md`](docs/实验运行指南.md)。

---

## 7. 输出与状态约定

```text
outputs/
├─ experiments/<EXP-ID>/<profile>/seed_<n>/      # 正式实验产物（含 run_manifest.json）
├─ derived/                                      # COCO/YOLO 派生数据（不写回原始数据集）
├─ smoke_runs/                                   # smoke 验证产物（永久标记，不进入论文统计）
├─ supplements/                                  # 补采包输出
└─ logs/、autodl_logs/、pids/                    # 运行日志
```

- `completed` / `completed_sealed` / `completed_locked`：成功终态；`completed_smoke_only`：仅代码链验证；`failed_*`：保留失败证据，修正后可恢复。
- 输出目录不得位于只读的 `data_root` 内（`ExperimentConfig.validate()` 会拒绝）。
- 原始数据集只读；派生数据与预测全部写入 `outputs/`。

---

## 8. 测试

```powershell
.\run.ps1 test
# 或
python -m unittest discover -s "61 项自动化测试" -t "61 项自动化测试" -v
```

- 测试覆盖入口契约、布局发现、依赖图、数据物化、RLE 掩膜编解码、评价管线、预测分片、MMDet/YOLO 端到端、可逆加速、严格 FP32 策略、统计与输出格式等。
- 部分用例需要整套 GPU 依赖（torch、mmengine、mmdet、pycocotools）。在仅有 numpy/pandas/openpyxl/Pillow 的裸环境中，其余用例全部通过，失败项均为第三方依赖缺失（`ModuleNotFoundError`），不含断言失败。
- 无 GPU 也能验证的入口检查：`dryall`（见第 5 节）。

---

## 9. 科学性边界（务必阅读）

- **D4 一次性**：EXP-11 前会校验 EXP-10 冻结清单并交互输入 `I UNDERSTAND D4 IS ONE-SHOT`，写入一次性开始标记后不可重复运行；D4 结果不得用于重新选择模型、阈值、校准器或后处理。
- **Smoke 不算结果**：`--smoke-test` 使用合成数据与极小训练量，只验证代码链，带永久标记，不能进入 EXP-15 正式统计。
- **YOLO 锚点**：EXP-02/03 复现保留为独立锚点，不参与 MorphoRoute 创新候选选择。
- **匹配与 PCGrad**：B2/B3 使用与 D2 无关的静态参数量/FLOPs 搜索匹配 B9，并输出候选审计；B11 使用实际接入 MMEngine 更新步骤的 PCGrad。
- **统计口径**：五种子配对使用精确双侧符号翻转检验与 Holm 校正；只有 5 对时最小双侧 p 值为 0.0625，不把小样本不显著写成“显著提升”。
- **补采边界**：`supplements/` 中的 EXP-08A/08C 探针与 EXP-08B/15 补齐只针对已完成实验做事后审计，不重建丢失检查点、不重训、不访问 D4 图像。

---

## 10. 许可证与引用

- 本仓库暂未声明开源许可证；如需再分发或商用，请先联系作者。
- 引用信息（论文标题、作者、期刊、DOI）将在论文录用后补充。

---

## 11. 本次整理说明

本仓库是论文工程的最终代码快照，已完成的整理：

- 仅保留最新版本：剔除 `*.bak`、`*.before_*`、`*.pre_*`、`__pycache__`、`.ipynb_checkpoints` 与旧的重复运行器；
- 剔除模型权重（`yolo11n.pt` 等）、`outputs/`（约 91 GB 实验产物）与数据集，只保留代码、配置、文档与关键审计产物；
- 新增 `run.ps1` / `run.cmd` / `run.sh` / `RUN_COMMANDS.md` / `.gitignore` / `.gitattributes`；
- 修复两处与当前目录布局/架构不一致的测试，修正两个辅助工具的导入路径。

逐项说明见 [`docs/发布说明.md`](docs/发布说明.md)。
