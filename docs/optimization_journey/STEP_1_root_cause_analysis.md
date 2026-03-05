# STEP 1: 根因分析 — 为什么 PLE 输给了 CGC？

> 时间：2026-03-05  
> 阶段：问题诊断，从代码、数据、理论三个维度逐一排查

---

## 1.1 问题定义

第一轮实验（STEP 0）结果：CGC(0.5219) > PLE(0.5013) > MMoE(0.4806)

**核心矛盾**：PLE 论文声称多层 extraction + 参数隔离应该优于单层的 CGC，但实验结果相反。

## 1.2 排查维度一：数据量 vs 模型容量

### 参数量对比

| Model | 总参数量 | Embedding | Expert | Tower | 其他 |
|-------|----------|-----------|--------|-------|------|
| MMoE  | 153,690  | ~49K      | ~89K   | ~15K  | ~1K  |
| CGC   | 192,890  | ~49K      | ~128K  | ~15K  | ~1K  |
| PLE (v1) | **1,060,000+** | ~49K | **~960K** | ~15K | ~36K |

### 发现 Bug #1: 过参数化 🔴

```
PLE v1 配置：3层 × (3+2) experts/task × 128d × 2tasks = ~960K expert 参数
训练数据量：5,000 × 0.7 = 3,500 训练样本

参数/样本比 = 1,060,000 / 3,500 = 302:1 ← 严重过拟合！
CGC 参数/样本比 = 192,890 / 3,500 = 55:1 ← 仍然偏高但可接受
```

**结论**：PLE 的参数量是 CGC 的 5.5 倍，在 5K 小数据上必然过拟合。

## 1.3 排查维度二：PLE 层间信息传递

### 原始代码关键片段（`src/models/ple.py`）

```python
# PLE Extraction 的 forward pass：
layer_input = embed
for layer in self.extraction_layers:
    task_outputs, gate_weights = layer(layer_input)
    all_gate_weights.append(gate_weights)
    # ❌ 问题在这里！
    layer_input = torch.stack(task_outputs, dim=0).mean(dim=0)
```

### 发现 Bug #2: 信息瓶颈 — 层间用 mean 传递 🔴

**问题详解**：
```
Layer 1 输出: task_0_output (batch, 128), task_1_output (batch, 128)
Layer 2 输入: mean(task_0_output, task_1_output) → (batch, 128)
```

这意味着：
- 第 2 层的所有 expert（包括 task-specific 的）都接收**相同的输入**
- Task 0 的独占 expert 和 Task 1 的独占 expert 看到的是同一个向量
- **完全违背了 PLE 论文的核心思想：逐层参数隔离**
- PLE 论文中，每个 task 的输出应该**独立传递给下一层对应 task 的 expert**

**CGC 没有这个问题**：因为 CGC 只有 1 层 extraction，不存在层间传递。

### 对比 PLE 论文的正确做法

```
正确的 PLE 层间传递：
  Layer 1:
    task_0_input = embed → task_0_experts + shared_experts → gate_0 → task_0_output
    task_1_input = embed → task_1_experts + shared_experts → gate_1 → task_1_output
  Layer 2:
    task_0_input = task_0_output → task_0_experts_L2 + shared_experts_L2 → gate_0_L2 → final_0
    task_1_input = task_1_output → task_1_experts_L2 + shared_experts_L2 → gate_1_L2 → final_1
    ↑ 每个 task 的输入来自上一层对应 task 的输出，而非 mean！
```

## 1.4 排查维度三：Gate 温度退火

### 原始配置

```python
initial_temperature = 2.0    # 初始温度
decay_rate = 0.995           # 每 epoch 衰减

# 10 个 epoch 后的温度：
T_10 = 2.0 × 0.995^10 = 1.95  ← 几乎没变！
```

### 发现 Bug #3: 温度退火太慢 🟡

**影响**：
- 温度 T=2.0 时，softmax(logits/T) 非常均匀（趋近 uniform 分布）
- 所有 expert 被几乎等权重混合 → Gate 无法区分哪些 expert 更重要
- 等价于所有 expert 简单平均 → 失去了 Mixture-of-Experts 的意义
- 10 epoch 后温度仍为 1.95，Gate 几乎没有锐化

**CGC 默认温度 = 1.0**，Gate 从一开始就有区分度。

**数学直觉**：
```
T=2.0: softmax([1,2,3,4,5]/2.0) = [0.09, 0.12, 0.16, 0.22, 0.30] → 比较均匀
T=1.0: softmax([1,2,3,4,5]/1.0) = [0.01, 0.04, 0.09, 0.24, 0.62] → 有区分度
T=0.5: softmax([1,2,3,4,5]/0.5) = [0.00, 0.00, 0.01, 0.12, 0.87] → 很尖锐
```

## 1.5 排查维度四：梯度路径长度

### 发现 Bug #4: 梯度消失风险 🟡

```
PLE v1 的前向路径长度：
  Input → Embedding → Expert_L1 → Gate_L1 → Expert_L2 → Gate_L2 → Expert_L3 → Gate_L3 → Tower → Output
  + ESMM: CVR = CTR * CVR_raw → 额外的乘法路径

反向传播需要穿过：
  3 层 Extraction × 每层 2 个 BatchNorm = 6 个 BN
  + Tower 的 3 层 FC
  + ESMM 的乘法梯度
  = 约 10+ 层深度

CGC 的路径：
  Input → Embedding → Expert_L1 → Gate_L1 → Tower → Output
  = 约 5 层深度
```

5K 样本下 + 长路径 → 梯度极容易消失或爆炸，导致深层参数更新不到。

## 1.6 根因总结

| Bug | 严重度 | 描述 | 影响 |
|-----|--------|------|------|
| **#1 过参数化** | 🔴 严重 | 3层×128d=1.06M params，5K样本不够训练 | 过拟合，学不到特征 |
| **#2 信息瓶颈** | 🔴 致命 | 层间用 mean(task_outputs) 传递 | 破坏参数隔离，PLE退化为劣化的MMoE |
| **#3 温度退火太慢** | 🟡 中等 | T=2.0, decay=0.995 → 10ep后T=1.95 | Gate 无区分度，expert 等权重混合 |
| **#4 梯度路径过长** | 🟡 中等 | 3层extraction + ESMM → ~10层深度 | 小数据下梯度消失，深层参数更新困难 |

### 为什么 CGC 没有这些问题？

| Bug | PLE | CGC | MMoE |
|-----|-----|-----|------|
| 过参数化 | 3层expert多 | 只有1层 | 只有shared experts |
| 信息瓶颈 | 存在 | ❌不存在（只有1层） | ❌不存在 |
| 温度退火慢 | 存在 | 默认T=1.0 | 默认T=1.0 |
| 梯度路径长 | 存在 | 路径短 | 路径短 |

**结论**：PLE 的 4 个 Bug 叠加导致它在小数据上退化为一个比 MMoE 还差的模型。CGC 因为结构简单（单层），天然避开了这些问题。

---

## ⏭️ 下一步

进入 [STEP_2: 架构修复](STEP_2_architecture_fix.md) — 逐一修正 4 个 Bug，重新设计 PLE 架构。
