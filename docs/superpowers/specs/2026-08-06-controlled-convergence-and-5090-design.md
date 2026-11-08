# 受控收敛与 RTX 5090 加速设计

## 目标

在不改变 Mask2Former、MorphoRoute、双专家、形态软路由、Top-1 稀疏阶段、结构损失和 PCGrad 定义的前提下，减少 EXP-05 之后的无效训练时间。`max_updates=50000` 继续作为可恢复上限，D2 仍是唯一检查点选择与提前停止依据，D3/D4 不参与训练控制。

## 已批准方案

1. EXP-05 和 EXP-06 从 20,000 optimizer updates 起判断收敛；EXP-07、EXP-08B、EXP-09 从 30,000 起判断，保证从 20,000 开始的稀疏阶段至少获得 10,000 updates。
2. 门控的耐心计数只从 `start_update` 开始。门控前的波动可更新历史最佳值，但不得提前消耗两次耐心。因此 30k 门控、5k 验证间隔、patience=2 的最早停止点是 35k。
3. B0、B9、B10、B11 等核心确认变体不再默认解除门控。所有直接比较使用同一 D2 指标、最小改善量和耐心规则；需要跑满时设置 `release_budget_gate=true` 并从检查点恢复。
4. 受控加速档位默认 `query_count=64`。该值高于 D1 单图最大实例数 44，并保留 20 个空查询余量。将配置改回 100 即可完整回退；训练缓存签名包含该参数，64/100 不会错误复用。
5. 新增独立 `autodl_5090_32gb` 档位：640 输入、physical batch=4、effective batch=8、AMP、关闭梯度检查点、8 个数据加载进程、固定形状性能模式。完整 SOP 档位保持 1024/100 queries/50k 的原始设置不变。
6. 性能模式使用 pinned memory、persistent workers 和受控 prefetch；这些只改变数据传输和并行执行，不改变样本、增强、损失或优化器更新语义。

## 不实施的优化

本轮不把形态特征提取直接移动到 `PackDetInputs` 之后的数据加载进程。当前 `area_fraction` 等特征是在 `BatchFixedSizePad` 之后的掩膜上计算，简单前移会改变形态软目标，违反“不改变核心技术”。保留现有精确实现，后续只有在自定义数据预处理器能证明逐实例特征数值完全一致时才启用。

## 回退与审计

- 训练上限始终为 50k；门控正常停止后保留最佳检查点和门控 JSON。
- `release_budget_gate=true` 可恢复到 50k。
- `query_count=100` 可回退查询容量。
- RTX 5090 若显存不足，可仅把 physical batch 改为 2 并把梯度检查点改为 true，effective batch 仍为 8。
- 完整 SOP 档位不修改，作为原始协议对照。

## 验收

- 门控前不累计 patience；30k 后连续两次无改善才停止。
- 生成器输出四个档位、每档 19 张卡片，共 76 个配置。
- `local_6gb` 和 `autodl_5090_32gb` 的相关实验使用 64 queries 和受控门控，核心变体不自动释放。
- 5090 配置解析、独立入口、MMDetection 配置生成和全部现有测试通过。

