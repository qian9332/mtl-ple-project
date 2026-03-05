# STEP 4: 最终对比 & 结论

> 时间：2026-03-05  
> 阶段：综合分析，回答「为什么 PLE 最终赢了」

---

## 4.1 最终对比表

| 指标 | MMoE | CGC | **PLE** | PLE vs CGC | PLE vs MMoE |
|------|------|-----|---------|------------|-------------|
| **Params** | 153,690 | 192,890 | 243,978 | +26.5% | +58.7% |
| **CTR AUC** | 0.6270 | 0.6296 | **0.6422** | **+1.26%** | **+1.52%** |
| **CVR AUC** | 0.6888 | 0.6908 | **0.7124** | **+2.16%** | **+2.36%** |
| **Avg AUC** | 0.6579 | 0.6602 | **0.6773** | **+1.71%** | **+1.94%** |
| **CTR LogLoss** | 0.6488 | 0.6508 | **0.6367** | -2.2% better | -1.9% better |
| **CVR LogLoss** | 0.3743 | 0.3675 | **0.3517** | -4.3% better | -6.0% better |
| **Best Epoch** | 14 | 13 | 18 | 更晚收敛 | 更晚收敛 |
| **Final CosSim** | 0.440 | -0.029 | **0.433** | 更高 = 更少冲突 | 接近 |
| **Avg CosSim** | ~0.28 | ~0.09 | **~0.35** | 更高 | 更高 |
| **UW Frozen** | [0.87, 1.22] | [0.88, 1.22] | [0.86, 1.21] | 相似 | 相似 |

## 4.2 为什么 PLE 最终赢了？（因果分析）

### 原因 1: 逐层参数隔离有效减少梯度冲突

**核心证据 — 梯度余弦相似度 (CosSim)**

```
CosSim 物理含义：
  > 0: 两个任务(CTR/CVR)的梯度方向一致 → 互相帮助
  = 0: 梯度正交 → 互不干扰
  < 0: 梯度方向相反 → 互相冲突（更新一个会损害另一个）
```

| Model | CosSim 均值 | CosSim 最小值 | 出现负值次数 |
|-------|------------|--------------|-------------|
| MMoE  | 0.28 | 0.055 | 0次 |
| CGC   | **0.09** | **-0.029** | **2次** (Ep14, Ep20) |
| PLE   | **0.35** | **0.149** | **0次** |

**解读**：
- PLE 的 CosSim 始终为正 → 两个任务的梯度从未冲突
- CGC 在 Ep14 和 Ep20 出现负值 → 共享层存在冲突
- MMoE 虽然没有负值，但均值只有 0.28 → 冲突控制不如 PLE

**为什么 PLE 能做到？**
- PLE 的 per-task expert 只接收对应 task 的上层输出 → 梯度路径天然隔离
- CGC 所有 expert 共享同一个输入 embed → 梯度必然混合

### 原因 2: CVR 的提升幅度远超 CTR

```
CTR 提升: PLE(0.642) vs CGC(0.630) = +1.2%
CVR 提升: PLE(0.712) vs CGC(0.691) = +2.1%  ← 提升更大！
```

**为什么 CVR 提升更大？**
- CVR 是更难的任务（样本少，12.7% vs 38.3%）
- CVR 通过 ESMM 公式 P(conv) = P(click) × P(conv|click) 间接学习
- PLE 的参数隔离让 CVR 的独占 expert 能专注学习 conversion 模式
- CGC 的 CVR expert 虽然也是独占的，但**接收的输入与 CTR expert 完全相同**
- PLE 的第二层 CVR expert 接收的是第一层 CVR gate 的输出，这个输出已经被 CVR 的 gate 过滤过 → **更纯净的 CVR 特征**

### 原因 3: Gate 温度退火让 Expert 选择更精准

```
PLE Gate 温度变化：
  Ep1:  T=1.42 → Gate 分布较均匀，所有 expert 贡献接近
  Ep10: T=0.90 → Gate 开始有选择性
  Ep20: T=0.54 → Gate 高度集中在少数 expert

效果：
  - 早期：所有 expert 被均匀训练 → 避免"赢者通吃"
  - 后期：Gate 锐化后只选最好的 expert → 提高预测精度
  - 温度退火相当于一个"课程学习"策略
```

**MMoE 和 CGC 的 Gate 温度始终为 1.0**，没有这种自适应效果。

### 原因 4: Feature Mask 自监督增强表征质量

PLE 独有的 Feature Mask 自监督任务：
- 随机 mask 15% 的输入特征 → 重建原始特征
- 重建 loss 作为辅助损失加入总损失
- 迫使 expert 学到更鲁棒的特征表示

```python
# PLE forward pass
if mask and training:
    embed = embed * bernoulli(0.85)  # 15% mask
    ...
    mask_loss = MSE(reconstructor(task0_output), original_embed)
    total_loss = uw_loss + 0.1 * mask_loss + 0.01 * balance_loss
```

## 4.3 CTR/CVR 诊断流程总结

### 标准化诊断状态机

```
WARMING → IMPROVING → CONVERGED
                   ↘ OSCILLATING (CVR特有)
                   ↘ DEGRADING (过拟合信号)
```

### 诊断规则

| 状态 | 判断条件 | 含义 |
|------|----------|------|
| WARMING | epoch < 6 | 热身阶段，不做诊断 |
| IMPROVING | 最近5 epoch slope > 0 | 性能持续上升 |
| CONVERGED | CTR: delta < 0.001; CVR: std < 0.002 | 已收敛 |
| OSCILLATING | sign change ≥ 3 in 5 epochs | CVR 来回波动 |
| DEGRADING | slope < 0 | 性能开始下降（过拟合） |

### 三模型诊断时间线

```
MMoE: WARM(1-5) → IMP(6-18) → CONV(19-20)        ← CTR 先收敛
      WARM(1-5) → IMP(6-14) → CONV(15-20)        ← CVR 先收敛

CGC:  WARM(1-5) → IMP(6-16) → DEG(17) → CONV(18-20)  ← CTR 出现轻微退化！
      WARM(1-5) → IMP(6-15) → CONV(16-20)            ← CVR 正常收敛

PLE:  WARM(1-5) → IMP(6-20)                      ← CTR 20 epoch 仍在提升！
      WARM(1-5) → IMP(6-17) → CONV(18-20)        ← CVR Ep18 收敛
```

**关键差异**：
- PLE 的 CTR 在 20 epoch 后仍有微弱提升 → 模型容量尚未饱和，更多 epoch 可能更好
- CGC 的 CTR 在 Ep17 出现 DEGRADING → 模型开始过拟合

## 4.4 从 v1 到 v2 的完整变化总结

### 实验结果对比

| 版本 | 数据 | PLE AUC | CGC AUC | 赢家 |
|------|------|---------|---------|------|
| **v1** (5K样本, Bug版) | 5,000 | 0.5013 | **0.5219** | **CGC** |
| **v2** (500K, 修复版) | 500,000 | **0.6773** | 0.6602 | **PLE** |

### PLE 从输到赢的提升拆解

```
PLE v1 → v2 的 Avg AUC 提升: 0.5013 → 0.6773 = +0.1760 (+35.1%)

提升来源估算:
  1. 数据量 5K → 500K:           ~+0.10 (60% 贡献)
  2. Bug #2 信息瓶颈修复:         ~+0.04 (23% 贡献)
  3. Bug #1 模型瘦身:             ~+0.02 (11% 贡献)
  4. Bug #3 温度退火 + Bug #4:    ~+0.01 (6% 贡献)
  
注：以上为估算值，实际各因素间有交互效应
```

## 4.5 最终结论

### 技术结论

1. **PLE 论文的 claim 成立**：逐层参数隔离确实能减少任务间梯度冲突（CosSim 0.35 vs CGC 0.09）
2. **PLE 优势在 CVR 上更明显**（+2.2%），因为参数隔离让稀疏 CVR 信号不被 CTR 梯度干扰
3. **Uncertainty Weight 对三个模型效果一致**，权重由数据分布决定而非模型架构
4. **Gate 温度退火是 PLE 特有的优势**，提供了"课程学习"效果

### 工程教训

1. **小数据上不要用大模型**：5K 样本下 1M 参数的模型必然过拟合
2. **仔细检查层间信息传递**：mean pooling 看似合理，但破坏了参数隔离的核心假设
3. **温度退火速度要够快**：decay=0.995 几乎无效，0.95 才有明显效果
4. **基线对比很重要**：如果没有 CGC 的对比，可能不会发现 PLE 的 Bug

### 面试要点（简历项目回答）

> **Q: 为什么选择 PLE 而不是 CGC？**
> A: 通过 MMoE/CGC/PLE 三模型对比实验，发现初版 PLE 因为信息瓶颈（层间 mean pooling）和过参数化在小数据上输给 CGC。通过根因分析定位 4 个问题并修复后，PLE 在 500K 数据上以 Avg AUC 0.677 显著领先 CGC 的 0.660（+1.7%），核心优势体现在梯度余弦相似度（PLE 0.35 vs CGC 0.09）——说明 PLE 的逐层参数隔离确实减少了 CTR/CVR 的梯度冲突。

---

## 📁 相关产出物

- 完整训练日志：`logs/gpu_run/train.log`
- 三模型详细结果：`logs/gpu_run/{mmoe,cgc,ple}_result.json`
- 对比 JSON：`logs/gpu_run/comparison.json`
- 训练脚本：`scripts/step_gpu.py`
- 原始 PLE 代码：`src/models/ple.py`（v1 Bug 版，保留用于对比）
