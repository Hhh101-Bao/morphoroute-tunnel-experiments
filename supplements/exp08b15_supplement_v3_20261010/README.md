# EXP-08B资源与EXP-15统计补齐

这是对已完成实验的独立补齐包，不是重新训练，也不等同于原方案所有EXP-08/EXP-15条款都完成。

## 在AutoDL运行

上传压缩包和对应.sha256到`/root/autodl-tmp/Untitled Folder`，在原来的PyTorch训练环境执行：

```bash
cd "/root/autodl-tmp/Untitled Folder" &&
sha256sum -c exp08b15_supplement_v3_20261010.sha256 &&
tar -xzf exp08b15_supplement_v3_20261010.tar.gz &&
python -B "exp08b15_supplement_v3_20261010/launch_supplement.py" --mode run
```

查看状态与实时日志：

```bash
LOG_DIR="/root/autodl-tmp/railway_tunnel_experiments/outputs/autodl_logs"
cat "$LOG_DIR/EXP08B_EXP15.state"
tail -F "$(cat "$LOG_DIR/EXP08B_EXP15.current_log")"
```

启动前会执行测试、GPU占用检查和输入哈希检查。存在其他GPU计算任务或重复补齐控制器时拒绝启动，不会结束任何已有任务。前台哈希检查可能需要几分钟；打印PREFLIGHT_HASH表示仍在核验，不要重复启动。预检成功后转为后台进程，关闭终端不影响运行。机器关机、重启或平台回收仍会中断进程。

v2修正了EXP-09的两层目录口径：权重与配置位于子种子的work_dir，预测可位于矩阵总表seed42下的runs/模型/子种子目录。只接受原代码定义的精确导出路径，不扫描“最新文件”，不移动原结果；在GPU测量前流式核对全部预测文件的D3、模型/种子元数据、checkpoint哈希和五类映射。PREFLIGHT_BUNDLE为该只读检查的进度。只有打印SUPPLEMENT_STARTED后才使用实时日志命令；预检失败时日志指针尚未生成。

v3兼容旧版预测JSON的checkpoint_sha256为空这一实际格式。仅对空值允许替代来源审计：必须唯一对应completed_run记录，模型种子的权重/配置路径一致，八项完成指标一致，缓存与导出文件逐图的预测、标签和group等完整图像payload一致，D3覆盖及类别正确。只忽略允许不同的头部复用元数据，不忽略任何图像payload差异。非空权重哈希冲突仍拒绝；缺缓存、歧义或内容差异均停止。所有比较都是只读，不生成推理、不修改原文件，也不将当前权重哈希填入历史预测。当前权重指纹用于本次资源测量与恢复，不能据此证明训练结束时的权重文件字节未曾变化。该限制保存在独立来源审计中。LEGACY_CACHE_COMPARE为逐图对账进度。

失败时停止整条链并保留诊断；不要删除旧模型、旧结果或绕过检查。同一压缩包、同一数据与环境可重新运行同一个启动命令，以复用已核对的补齐断点。如果哈希、源码或环境变了，程序会拒绝混合恢复。

## 顺序与口径

1. 资源测量依次运行B0、B4、B5、B11、B9的seed42。模型逐个使用独立Python进程，固定D2所选checkpoint，只做推理与图分析。每个checkpoint逐张量核对加载，测量前后核验全部模型状态；零优化器更新。
2. 参数量为实际加载模型全部参数。输入为已审计D1补采固定队列前8张真实图，保持比例缩放和填充到640。FP32，关闭TF32，batch1。CPU保存输入模板，每次只保留一张GPU输入和新建样本对象；8张预热、30次计时。
3. 时延为GPU模型predict含融合后处理，不含磁盘读取、CPU预处理和保存。吞吐为30次串行耗时的每秒图像数，不代表批量并发吞吐。平均数、中位数、P95均输出。该口径不同于EXP-14原尺寸端到端时延，两者不可混为同一指标。
4. FLOPs采用MMEngine图追踪，一次乘加计一FLOP，每张输入各追踪一次，保留算子明细。若存在不支持算子，输出明确标为“已统计算子GFLOPs”，列出遗漏，不把它冒充完整网络精确FLOPs。时延含融合后处理，而FLOPs对应tensor-forward图，二者边界也不同。追踪是固定输入下实际分支的估计，不是所有可能路由分支的穷举。
5. EXP-15复用EXP-09已完成D3整数据集指标，输出各模型逐种子结果的均值与样本标准差，不计算“逐图AP平均”。主比较B9–B0采用seed42–46的5对完整训练结果。精简总表若省略旧45/46，只从源运行目录完全一致的completed_run缓存恢复；缺失或冲突即停止，不选择修改时间最新文件，不重新训练。
6. 五种子配对采用精确双侧符号翻转检验，8个主比较指标使用Holm校正。其随机化解释依赖配对差值符号交换性假设，不能据此声称跨所有数据域的因果效果。只有5对时，双侧最小原始p值为0.0625。B11–B0另以seed42–44的3对作探索性比较，不把小样本不显著写成“显著提升”。
7. B9/B0结构指标从已有D3预测中流式读取，以同类别、按分数排序、mask IoU≥0.5的匹配实例计算。RLE直接比较交并并仅解码匹配前景的联合区域，不缩放原尺寸掩膜；每个种子的均值必须与原D3表一致。按照真实manifest group做2000次共同分组Bootstrap，各种子、两模型使用同一次分组抽样，保留实例总和/数量。该CI仅是已匹配实例上的结构质量差异，不是无条件召回率、不是组级AP显著性，也不是机制因果证据。
8. EXP08A/C保持已完成的固定checkpoint事后补采身份，引用原审计与系数表，不重建丢失的早期checkpoint，不自动判定机制M门。原EXP11封存E判定仅引用其JSON，不再读取D4图像或D4预测，不修改候选模型及外测结果。

## 产物

独立输出到：

`outputs/supplements/EXP08B_EXP15_20261010_v1/`

- `EXP08B_resources_640_FP32.csv`：五模型参数量、已统计GFLOPs、显存、吞吐、平均/中位/P95时延。
- `EXP08B_resources_audit.json`与`resources/`：硬件、算子覆盖、逐次耗时、权重完整性。
- `EXP15_seed_summary_D3.csv`：各模型完整数据集指标的跨种子汇总。
- `EXP15_paired_seed_tests.csv/json`：主比较Holm校正及三种子探索性对比。
- `EXP15_structure_sufficient_statistics.csv`、`EXP15_structure_reconciliation.csv`：逐图配对基础与原结果一致性核验。
- `EXP15_group_bootstrap_D3.csv/json`：匹配实例结构指标的探索性分组CI。
- `EXP15_evidence_scope.json`：封存E结果引用、事后补采引用与不能作出的结论。
- `EXP15_prediction_provenance.json`：逐模型/种子的来源绑定方式，明确哪些旧预测没有记录历史checkpoint哈希。
- `protocol.json`、`state.json`、`run_manifest.json`：固定协议、断点与完成凭证。

只有`run_manifest.json`的状态为`completed_resources_and_statistics_supplement`，才表示本包两项补齐完成。已有实验目录及原完成状态不被覆盖。统计阶段会打印每张图的处理位置，昂贵图像可能耗时较长；日志中的预测分片ETA不适用于本阶段。本包没有长程训练，也不重做5050张D1形态预处理，但统计耗时应以AutoDL实测为准，不能保证固定时刻完成。

## 验证边界

打包电脑没有Torch/MMEngine/CUDA。当地执行了数字配对、Bootstrap、RLE/原项目指标一致性、输入拒绝与断点校验测试；Torch测试必须在AutoDL执行且不得跳过。资源测量实际模型加载、GPU追踪及真实缓存对账只能在服务器验证，任何失败都会停止，不会悄悄切换为AMP、CPU计时、合成结果或启动重新训练。
