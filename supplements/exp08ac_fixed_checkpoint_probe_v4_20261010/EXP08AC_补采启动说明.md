# EXP-08A / EXP-08C 固定模型事后补采 v4

本包只启动首批 08A/08C 补采，不是“剩余所有实验全部自启动”。
08B 资源补齐、15 最终统计仍是后续步骤；13 缺少物理尺度输入，暂不启动。

## 本次修正

- 服务器审计已确认：同一300张D1样本全部有COCO图像记录，7张无前景。逐项本地原掩膜核对均为TACK单通道全零背景；v3错误地把这些合法阴性当成缺失标注。
- 保留原300张抽样清单及顺序，不删除数据、不补抽。预检核实清单声明阴性、原掩膜哈希/尺寸/全零值，单列7张背景排除证据。缺图、仅crowd、阳性声明却无COCO实例、非零原掩膜、未知哈希变化仍停止。
- 统一预检、试采、断点和统计口径：只对原样本中的293张阳性图片补采三种子，最多879条真实梯度记录。若运行中缩放后实例为空或梯度零范数，继续单列排除；879不是保证的有效结果数量。阴性图不写伪造梯度记录。
- 首批试采用原阳性子集的前5张，不新增样本；统计披露300张抽样、7张背景排除及运行时无效观测。结论限定为“阳性且梯度有效图像上的探索性关联”，不能作为整个D1的总体冲突率。
- 本地复核相同确定性 300 张 D1 样本：299 张图片哈希匹配，1 张方向归一化副本哈希不同；300 份原始标注哈希匹配，图像与 DACL 标注尺寸匹配。该结论只覆盖这 300 张本地样本，不能外推全数据集或替代服务器检查。
- 不再混淆“原始源文件哈希”和“当前探针文件哈希”。只识别下面逐项核对的一个精确副本，并保存两套指纹、理由和尺寸；不修改 split_manifest.csv，不旋转、重编码或替换服务器图片。
- 启动时仍检查服务器真实图片 SHA-256、标注 SHA-256、COCO 坐标尺寸和 EXIF。未知图片变动、标注变动或坐标不一致全部拒绝，不能使用本包跳过完整性检查。每次梯度采集前后再次检查冻结图片和标注指纹。
- 校验文件显式写入 LF，不再让 Linux 将 CR 当作文件名的一部分。
- 根据项目真实配置检查 `model.panoptic_head.variant_id`。不再强制要求不存在的 `model.variant_id`，但若可选顶层字段与 B9 冲突仍拒绝启动。
- COCO与清单的显式record_id冲突时，即使文件路径相同也拒绝；仅没有显式ID的COCO记录可以使用精确路径映射。
- 同时检查 detector/head 类型、seed 与 work_dir，并打印三个种子的真实身份字段供审计。
- 从依赖测试开始就保存日志；预检退出也有完整失败日志，不再要求必须先启动后台进程才能查看日志。
- 独立输出目录为 `EXP08AC_fixed_checkpoint_D1_v4`。旧包、旧失败状态和已有训练结果不覆盖、不删除。

## 已确认的7张原始阴性图片

TACK_TB_Camera6_001685_A、TACK_TB_Camera3_000866_A、TACK_TB_Camera9_000558_A、TACK_TB_Camera9_000074_B、TACK_TB_Camera4_001620_A、TACK_TB_Camera3_000248_G、TACK_TA_Camera5_000064_M。

本地完整300张回归检查：缺失0、阳性293、背景7；300份原标注哈希匹配，299张图片匹配清单原哈希，1张是下述精确核对的方向副本。该检查只在内存中把本地旧WSL COCO路径映射到实际本地路径；没有写数据文件。服务器仍必须对实际配置路径和哈希重新预检。

## 已核对的单图溯源例外

仅适用于 D1 的 DACL_train_dacl10k_v2_train_3520，路径 D1_D3_internal/train/images/DACL__dacl10k_v2_train_3520.jpg。

- 原始源文件：SHA-256 `2a85c12efdce62f89afc93e93984e1cf558a2cb594985fa40d675973d3a31d8f`，4032×3024，EXIF Orientation=6，3,067,997 字节；清单哈希对应此文件。
- 当前本地副本：SHA-256 `739ae79710e05687ea0e94e9930918102409e9a018626a42849327c0f7a6ad0f`，3024×4032，无 EXIF 方向标记，5,505,735 字节。
- 标注 SHA-256 `6f7c51692efae19289d7c837b1b3fde21c3d2bce168e98622ae6b1b6a080841f`，imageWidth=3024，imageHeight=4032；现副本和标注坐标空间一致。
- 原始文件经 Pillow EXIF 转置后与现副本比较：RGB 每通道平均绝对差约 0.0594/255，最大差 4/255。这支持方向归一化并重编码的解释，但不是逐像素/逐字节相同，也不是模型性能等价的证明。
- 服务器必须与该副本的精确 SHA-256 一致，否则停止。不能据此宣称原始数据清单已经全盘通过，或证明训练期间文件从未改变。论文数据处理描述应披露该例外；如需全数据集审计，另行只读核对，不打开/重跑封存 D4 实验。

## 已确认口径

- 使用 B9 seed42/43/44 各自现存、由 D2 选出的 best checkpoint；不根据 D3/D4 选择 checkpoint。
- 默认确定性选取 300 张 D1 图片，三种子共用样本。按清单中的 assignment_group_id / group_key 轮流选取，优先覆盖不同组。此抽样是分组覆盖样本，不是全 D1 图像均匀总体估计。
- 记录真实分组，保存样本、配置、先验、源码和 checkpoint 哈希。断点协议发生变化时拒绝混用结果。
- 固定 640 输入，无随机缩放/翻转；保持模型损失分支，冻结 BatchNorm 统计和 dropout，禁用 AMP/TF32 与梯度 checkpointing。设置持久化 training_update 对应的损失阶段。
- 不调用 optimizer、train_step 或训练 Runner。每张记录落盘前核对模型所有 state_dict 张量的哈希，且检查 checkpoint 文件未改动。这会增加少量审计开销，计时包含该开销；异常退出也不接纳未验证的断点记录。
- 主诊断与现有记录一致：加权 route+cldice 辅助目标对加权 boundary 辅助目标。它不是“裂缝主任务对面状病害主任务”的直接梯度对照。
- 额外记录主分割损失对全部辅助损失的梯度余弦。未使用参数按同一完整参数顺序补零，防止梯度向量错位。
- 形态变量是该图片中实例的均值，取自探针分辨率，不是原图物理尺度测量。控制类占比、log 实例面积、种子指示项，使用真实分组 cluster bootstrap。
- 最佳 checkpoint 迭代不同，因此种子与阶段差异不可分离。结果只能标注“事后固定 checkpoint 机制诊断 / 探索性关联”，不声称复原早中晚训练或证明因果。
- 少于 20 组、设计矩阵秩不足、严重不可识别、非有限梯度时停止，不自动输出“显著”或 M-positive。20 组是程序警戒线，不是功效充分性的证明。
- 零梯度、缩放后无实例图片不伪造观测；配对汇总仅纳入三种子均有效图片，并单列排除数量。
- 不读取 D4 图像/标注/预测，不修改候选选择、冻结文件、E 门控或已有 EXP09/08D/12 结果。

## 在 AutoDL 启动

将压缩包上传到 `/root/autodl-tmp/Untitled Folder`。解压到一个新的独立目录，不覆盖项目源码。

```bash
cd "/root/autodl-tmp/Untitled Folder"
tr -d '\r' < exp08ac_fixed_checkpoint_probe_v4_20261010.sha256 |
sha256sum -c - &&
tar -xzf exp08ac_fixed_checkpoint_probe_v4_20261010.tar.gz &&
python "/root/autodl-tmp/Untitled Folder/exp08ac_fixed_checkpoint_probe_v4_20261010/launch_fixed_checkpoint_probe.py" --mode run
```

启动器先运行自动化测试（包括服务器端真实 Torch 梯度与权重不变测试），再检查真实配置、D1 映射和文件哈希。此预检可能需要几分钟。通过后后台运行，小样本检查失败则不进入长任务。

应看到 `PROBE_D1_FOREGROUND_AUDIT=PASSED sampled=300 positive=293 background_excluded=7 missing=0 max_gradient_records=879`；然后 `SUPPLEMENT_LAUNCHED`、`PROBE_PILOT=PASSED`、`PROBE_PROGRESS`。仅打印预检通过不代表已经补采完成。

进度：

```bash
LOG_DIR="/root/autodl-tmp/railway_tunnel_experiments/outputs/autodl_logs"
cat "/root/autodl-tmp/railway_tunnel_experiments/outputs/supplements/EXP08AC_fixed_checkpoint_D1_v4/state.json"
tail -F "$(cat "$LOG_DIR/EXP08AC_FIXED_PROBE.current_log")"
```

终端关闭不会主动结束后台进程；实例重启会结束进程。再次运行同一启动命令会检查协议并继续已有记录，不删除或覆盖旧记录。已有 GPU 任务或补采进程时拒绝重复启动，不执行 kill。

## 输出与完成判据

所有输出在 `outputs/supplements/EXP08AC_fixed_checkpoint_D1_v4`，不覆盖原 EXP08A/C 清单。

- protocol.json、sample_manifest_D1.csv：固定样本与可复核协议。
- input_fingerprint_audit.json：300 张探针输入的原始清单/当前文件指纹、标注哈希、尺寸、已核对的单图变换例外。
- foreground_eligibility_audit.json：原抽样每张图片的前景资格、COCO映射与计数。
- sample_exclusions_D1.csv：7张背景图片、原标注哈希、全零掩膜证据及排除原因；不伪造梯度。
- pilot_report.json：真实小样本速度，仅估计本阶段采集耗时。
- gradient_records.jsonl：每图、每种子的断点记录。
- gradient_conflict_records.csv、M1_group_summary.csv：完整记录和分组汇总。
- M3_analysis_table.csv、M3_association_coefficients.csv、M3_analysis_audit.json：探索性关联、分组置信区间及限制。
- weight_integrity.json：前后模型权重及持久化缓冲区一致性证据。
- state.json / run_manifest.json：本阶段结果。成功状态是 completed_posthoc_probe，不代表全体实验完成。

本地完整包测试32项：31项通过、1项真实Torch测试因本地未安装而跳过。另已核对真实300张D1数据及每份原标注哈希，并完成独立代码复核。服务器启动自动运行包内31项测试（含必须执行的真实Torch项），之后仍需全样本预检和真实MMDetection模型试采。本地没有Torch、MMDetection和CUDA，不能保证服务器运行绝无错误；遇到问题会停止并保留日志，不更改已完成训练。
