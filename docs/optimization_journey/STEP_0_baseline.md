# STEP 0: 基线实验 — 5K 小样本的首次对比

> 时间：2026-03-04  
> 阶段：项目初始搭建 + 第一轮对比实验

---

## 0.1 实验背景

项目目标是通过 MMoE / CGC / PLE 三模型对比，验证 PLE 论文中「逐层参数隔离减少任务冲突」的 claim。
第一轮实验使用了 **5000 条合成样本**，快速跑通 pipeline。

## 0.2 初始模型配置

### PLE 配置（v1 — 来自 `configs/ple_config.json`）

```json
{
    "num_extraction_layers": 3,     // ❌ 3 层 extraction
    "expert_dim": 128,              // ❌ 128 维 expert
    "num_task_experts": 3,          // 每任务 3 个独占专家
    "num_shared_experts": 2,        // 2 个共享专家
    "tower_hidden_dim": 64,
    "initial_temperature": 2.0,     // ❌ 初始温度过高
    "dropout": 0.1
}
```

### MMoE 配置
```
6 shared experts, expert_dim=128, tower_hidden=64
```

### CGC 配置
```
3 task-specific + 2 shared experts per task, expert_dim=128, tower_hidden=64
```

### 训练配置
```
数据量: 5,000 样本
batch_size: 2048 → 实际只有 ~2.4 batches/epoch
epochs: 10
lr: 0.001
temp_decay: 0.995/epoch  // ❌ 衰减太慢
```

## 0.3 第一轮实验结果

| Model | CTR AUC | CVR AUC | **Avg AUC** | 参数量 |
|-------|---------|---------|-------------|--------|
| MMoE  | 0.4971  | 0.4641  | 0.4806      | ~154K  |
| CGC   | **0.5017** | **0.5421** | **0.5219** | ~193K |
| PLE   | 0.4980  | 0.5047  | 0.5013      | ~1.06M |

### 🚨 反常现象：CGC 赢了！

- CGC Avg AUC = **0.5219** 显著领先
- PLE Avg AUC = **0.5013** 几乎和随机一样（0.50）
- MMoE Avg AUC = **0.4806** 甚至低于随机
- 这与 PLE 论文的结论完全矛盾

## 0.4 初步观察

### 数据量问题
- 5K 样本 → 训练集仅 ~3500 条
- CTR 正样本 ~900 条，CVR 正样本 ~260 条
- 对于 1.06M 参数的 PLE 来说，**严重过拟合**

### AUC 异常低
- 所有模型 AUC 都在 0.46-0.54 之间
- 说明模型根本没有学到有效的特征表示
- 5K 样本不足以训练有意义的 Embedding

### PLE vs CGC
- PLE 参数量 1.06M vs CGC 193K — 差 5.5 倍
- PLE 有 3 层 extraction → 梯度路径更长
- 小数据 + 大模型 = 过拟合 + 欠学习

## 0.5 产出物

- 训练日志：`logs/ple/`, `logs/mmoe/`, `logs/cgc/`
- 对比结果：`logs/comparison_results.json`
- 结论：**数据太少，PLE 过参数化，需要进一步分析**

---

## ⏭️ 下一步

进入 [STEP_1: 根因分析](STEP_1_root_cause_analysis.md) — 深入分析 PLE 为什么在小数据上表现差，并定位代码层面的具体问题。
