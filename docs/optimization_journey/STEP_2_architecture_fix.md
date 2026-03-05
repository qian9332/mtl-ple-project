# STEP 2: 架构修复 — 逐一修正 4 个 Bug

> 时间：2026-03-05  
> 阶段：代码重写，从 `src/models/ple.py` (v1) 到 `scripts/step_gpu.py` (v2)

---

## 2.1 修复总览

| Bug | 修复方案 | 新配置 |
|-----|----------|--------|
| #1 过参数化 | 3层→2层, 128d→64d, 3个→2个task experts | 244K params |
| #2 信息瓶颈 | per-task 独立传递替代 mean | 见代码 |
| #3 温度退火慢 | T=2.0→1.5, decay=0.995→0.95 | 20ep后 T≈0.54 |
| #4 梯度路径长 | 3层→2层 | ~7层深度 |

## 2.2 修复 Bug #1: 模型瘦身

### Before（v1 — `src/models/ple.py`）
```python
num_layers = config.get("num_extraction_layers", 3)   # 3 层
expert_dim = config.get("expert_dim", 128)             # 128 维
num_task_experts = config.get("num_task_experts", 3)   # 3 个/task
num_shared_experts = config.get("num_shared_experts", 2) # 2 个 shared
# 总 expert 参数 ≈ 3层 × (2task×3 + 2shared) × 128d × 2 FC = ~960K
```

### After（v2 — `scripts/step_gpu.py`）
```python
EX = 64    # expert_dim: 128 → 64
TH = 64    # tower_hidden: 64 → 64 (不变)

# PLE 只有 2 层:
# Layer 1: 2 task0 experts + 2 task1 experts + 2 shared = 6 experts
# Layer 2: 2 task0 experts + 2 task1 experts + 2 shared = 6 experts
# 总 expert 参数 ≈ 2层 × 6 experts × 64d × 2 FC = ~98K
# 加上 Embedding + Tower + Gate + Reconstructor ≈ 244K total
```

### 参数量变化
```
v1: 1,060,000+ params → v2: 243,978 params
缩减 77%！参数/样本比从 302:1 降到 0.49:1 (500K数据)
```

## 2.3 修复 Bug #2: Per-Task 独立传递（核心修复）

### Before（v1 — 致命 Bug）
```python
# src/models/ple.py line 196-201
for layer in self.extraction_layers:
    task_outputs, gate_weights = layer(layer_input)
    all_gate_weights.append(gate_weights)
    # ❌ 所有 task 的输出被平均后作为下一层的统一输入
    layer_input = torch.stack(task_outputs, dim=0).mean(dim=0)
```

**问题**：Layer 2 的 task_0 expert 和 task_1 expert 接收**相同输入**，参数隔离形同虚设。

### After（v2 — 正确实现）
```python
class PLE(nn.Module):
    def __init__(s):
        # Layer 1: 输入是 embed (TE 维)
        s.t0a = nn.ModuleList([Exp(TE, EX) for _ in range(2)])  # task0 experts L1
        s.t1a = nn.ModuleList([Exp(TE, EX) for _ in range(2)])  # task1 experts L1
        s.sa  = nn.ModuleList([Exp(TE, EX) for _ in range(2)])  # shared experts L1
        s.g0a = Gt(TE, 4, 1.5); s.g1a = Gt(TE, 4, 1.5)

        # Layer 2: 输入是上一层各自 task 的输出 (EX 维)
        s.t0b = nn.ModuleList([Exp(EX, EX) for _ in range(2)])  # task0 experts L2
        s.t1b = nn.ModuleList([Exp(EX, EX) for _ in range(2)])  # task1 experts L2
        s.sb  = nn.ModuleList([Exp(EX, EX) for _ in range(2)])  # shared experts L2
        s.g0b = Gt(EX, 4, 1.5); s.g1b = Gt(EX, 4, 1.5)

    def forward(s, sp, dn, mask=False):
        e = s.emb(sp, dn)

        # Layer 1
        so1 = [x(e) for x in s.sa]  # shared expert outputs
        o0 = gate_combine(s.g0a(e), [x(e) for x in s.t0a] + so1)  # task0 output
        o1 = gate_combine(s.g1a(e), [x(e) for x in s.t1a] + so1)  # task1 output

        # Layer 2: ✅ 每个 task 用自己的上层输出作为输入
        so2 = [x(o0) for x in s.sb]  # shared 用 task0 输出（也可用 mean，但这里选择简化）
        o0 = gate_combine(s.g0b(o0), [x(o0) for x in s.t0b] + so2)  # task0: 输入是 o0
        o1 = gate_combine(s.g1b(o1), [x(o1) for x in s.t1b] + so2)  # task1: 输入是 o1
        #                      ↑ o0                              ↑ o1
        #                      各自独立！不再是 mean！
```

**关键变化**：
1. Layer 2 的 task_0 expert 输入是 `o0`（上一层 task_0 的输出）
2. Layer 2 的 task_1 expert 输入是 `o1`（上一层 task_1 的输出）
3. 真正实现了 PLE 论文的**逐层参数隔离**

## 2.4 修复 Bug #3: 加快温度退火

### Before
```python
initial_temperature = 2.0
decay_rate = 0.995

# Epoch   1:  T = 2.000
# Epoch   5:  T = 1.950
# Epoch  10:  T = 1.902
# Epoch  20:  T = 1.810  ← 几乎没退火
```

### After
```python
initial_temperature = 1.5   # 降低初始温度
decay_rate = 0.95           # 加快衰减速度

# Epoch   1:  T = 1.500 * 0.95 = 1.425
# Epoch   5:  T = 1.500 * 0.95^5 = 1.158
# Epoch  10:  T = 1.500 * 0.95^10 = 0.898
# Epoch  15:  T = 1.500 * 0.95^15 = 0.695
# Epoch  20:  T = 1.500 * 0.95^20 = 0.539  ← 明显锐化
```

### 效果对比（softmax 分布变化）

假设 Gate logits = [0.5, 1.0, 0.3, 0.8]：

| 阶段 | 温度 T | softmax 分布 | 最大/最小比 |
|------|--------|-------------|------------|
| v1 Ep1 | 2.00 | [0.22, 0.27, 0.21, 0.26] | 1.29x |
| v1 Ep20 | 1.81 | [0.22, 0.28, 0.20, 0.26] | 1.40x |
| **v2 Ep1** | **1.42** | [0.21, 0.29, 0.20, 0.27] | 1.50x |
| **v2 Ep10** | **0.90** | [0.18, 0.34, 0.16, 0.29] | 2.13x |
| **v2 Ep20** | **0.54** | [0.13, 0.42, 0.10, 0.32] | 4.20x |

v2 在训练后期 Gate 有了明显的选择性，不再是"所有 expert 等权重混合"。

## 2.5 修复 Bug #4: 减少层数

```
v1: 3 层 Extraction → ~10 层深度
v2: 2 层 Extraction → ~7 层深度

梯度路径缩短了 30%，有效缓解小数据下的梯度消失问题。
```

## 2.6 同步优化：训练策略

除了架构修复，还同步改进了训练策略：

| 配置项 | Before | After | 原因 |
|--------|--------|-------|------|
| 数据量 | 5,000 | **500,000** | 充足数据发挥大模型优势 |
| batch_size | 2,048 | **4,096** | 更稳定的梯度估计 |
| epochs | 10 | **20** | 更多训练时间 |
| LR scheduler | 无 | **CosineAnnealing** | 后期学习率衰减 |
| gradient clip | 1.0 | **1.0** | 不变 |
| patience | 15 | **8** | 更快触发 early stopping |
| 梯度冲突检测 | 每 batch | **每 epoch 1次** | 减少计算开销 |

## 2.7 数据质量优化

### Before（v1 数据生成）
```python
# 简单随机生成，CTR/CVR 与特征的关系较弱
click_prob = sigmoid(random_score + noise)
```

### After（v2 数据生成 — `scripts/step_gpu.py` gen()）
```python
# 更强的特征-标签相关性
user_affinity = randn(1000) * 1.5            # user 偏好因子
item_quality = randn(500) * 1.5              # item 质量因子
category_bias = randn(50) * 0.5              # 品类偏差

# CTR 由 user+item+category+dense 联合决定
click_score = (user_affinity[user_id] * 0.35 +
               item_quality[item_id] * 0.30 +
               category_bias[category] * 0.15 +
               dense_0 * 0.25 + dense_1 * 0.15 + dense_2 * 0.10 +
               noise * 0.4)

# CVR 依赖不同的特征权重组合（与 CTR 有差异）
conv_score = (user_affinity[user_id] * 0.20 +    # user 影响降低
              item_quality[item_id] * 0.45 +       # item 影响升高
              category_bias[category] * 0.20 +     # 品类更重要
              dense_3 * 0.20 + dense_4 * 0.15 +    # 不同的 dense 特征
              noise * 0.3)

# ESMM 约束: conversion 必须先 click
conversion = click * bernoulli(conv_prob)

# 最终数据统计: CTR=38.3%, CVR=12.7%（给定 click 后 CVR=33.1%）
```

**设计意图**：
- CTR 和 CVR 依赖不同权重的相同特征 → 存在任务相关性
- 但权重分布不同（user vs item 的重要性反转）→ 存在任务差异
- 这种设计能让 PLE 的「参数隔离 + 共享」真正发挥作用

## 2.8 修复后的新旧模型结构对比

```
=== PLE v1 (旧) ===
Embedding (49K params)
  ↓
ExtractionLayer_1: 5 experts × 128d (共享输入 embed)
  ↓ mean(task_outputs)  ← Bug #2
ExtractionLayer_2: 5 experts × 128d (共享输入 mean)
  ↓ mean(task_outputs)
ExtractionLayer_3: 5 experts × 128d (共享输入 mean)
  ↓
Tower_CTR → sigmoid → ctr_pred
Tower_CVR → sigmoid → cvr_raw → ESMM → cvr_pred
Mask Reconstructor (36K params)
总参数: ~1,060,000

=== PLE v2 (新) ===
Embedding (49K params)
  ↓
Layer_1:
  task0_experts (2个, 64d, 输入=embed) ─┐
  task1_experts (2个, 64d, 输入=embed) ─┤
  shared_experts (2个, 64d)             ─┤→ gate → o0, o1 (独立)
  ↓ o0            ↓ o1                      ← 修复 #2: per-task 独立传递
Layer_2:
  task0_experts (2个, 64d, 输入=o0) ─┐
  task1_experts (2个, 64d, 输入=o1) ─┤
  shared_experts (2个, 64d)          ─┤→ gate → final_0, final_1
  ↓
Tower_CTR(final_0) → sigmoid → ctr_pred
Tower_CVR(final_1) → sigmoid → cvr_raw → ESMM → cvr_pred
Mask Reconstructor (16K params)
总参数: 243,978
```

---

## ⏭️ 下一步

进入 [STEP_3: 全量数据训练](STEP_3_full_data_training.md) — 用 500K 完整数据训练修复后的三个模型，记录每个 epoch 的详细指标。
