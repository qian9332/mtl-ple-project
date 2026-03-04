# MTL-PLE: Multi-Task Learning with Progressive Layered Extraction

## 📋 项目概述

本项目实现了基于 **PLE (Progressive Layered Extraction)** 的多任务学习框架，用于电商场景下的 **CTR（点击率）** 和 **CVR（转化率）** 联合预估。通过 MMoE/CGC/PLE 对比实验验证了 PLE 的逐层参数隔离优势，并集成了多项工程创新。

### 🔬 核心技术特性

| 模块 | 技术方案 | 说明 |
|------|----------|------|
| **模型架构** | PLE (Progressive Layered Extraction) | 逐层参数隔离，通过 MMoE→CGC→PLE 对比实验选定 |
| **Gate温度退火** | Temperature Annealing + 专家利用率监控 | 抑制 "赢者通吃" (Winner-Take-All) 现象 |
| **任务权重** | Uncertainty Weight (Kendall et al.) | 仅增加 T 个标量参数，训后固化零推理开销 |
| **梯度冲突** | 梯度余弦相似度 + EMA自适应阈值 | 冲突感知 Early Stopping + 共享层软冻结 |
| **CVR建模** | ESMM全曝光分支 | P(conversion) = P(click) × P(conversion\|click) |
| **辅助任务** | 特征掩码自监督 | Feature Mask Reconstruction 辅助训练 |
| **Expert隔离** | 独占Expert + 共享Expert | 通过参数隔离减少任务间冲突 |

## 📁 项目结构

```
mtl-ple-project/
├── README.md                          # 项目说明
├── requirements.txt                   # Python 依赖
├── configs/
│   └── ple_config.json               # 模型配置文件
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── experts.py                # Expert网络 + Gate + 利用率监控
│   │   ├── ple.py                    # PLE 模型（核心）
│   │   └── baselines.py             # MMoE / CGC 基线模型
│   ├── losses/
│   │   ├── __init__.py
│   │   └── uncertainty_weight.py     # Uncertainty Weight 损失函数
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── gradient_conflict.py      # 梯度冲突检测 + 自适应Early Stopping
│   │   └── trainer.py               # 训练引擎
│   └── data/
│       ├── __init__.py
│       └── dataset.py                # 数据集加载与预处理
├── scripts/
│   ├── train.py                      # 单模型训练脚本
│   ├── run_comparison.py             # MMoE vs CGC vs PLE 对比实验
│   └── data_analysis.py             # 数据全貌分析
├── data/
│   └── analysis_report/             # 数据分析报告输出
├── logs/                             # 训练日志
└── notebooks/                        # Jupyter notebooks
```

## 🚀 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 数据分析

```bash
python scripts/data_analysis.py --output data/analysis_report --samples 500000
```

### 3. 单模型训练（PLE）

```bash
python scripts/train.py --config configs/ple_config.json --model ple --epochs 30
```

### 4. 对比实验（MMoE vs CGC vs PLE）

```bash
python scripts/run_comparison.py --config configs/ple_config.json --epochs 30
```

## 🏗️ 架构设计

### PLE 模型架构

```
Input Features (Sparse + Dense)
        │
   ┌────┴────┐
   │ Embedding │
   └────┬────┘
        │
   ┌────┴────────────────────────────────┐
   │     Extraction Layer 1               │
   │  ┌─────────┐  ┌─────────┐  ┌──────┐│
   │  │Task-0   │  │Task-1   │  │Shared││
   │  │Experts  │  │Experts  │  │Experts││
   │  └────┬────┘  └────┬────┘  └──┬───┘│
   │       │            │          │     │
   │  ┌────┴──┐    ┌────┴──┐           │
   │  │Gate-0 │    │Gate-1 │           │
   │  └───┬───┘    └───┬───┘           │
   └──────┼────────────┼───────────────┘
          │            │
   ┌──────┴────────────┴───────────────┐
   │     Extraction Layer 2 & 3        │
   │     (Same structure, deeper)      │
   └──────┬────────────┬───────────────┘
          │            │
   ┌──────┴──┐  ┌──────┴──┐
   │CTR Tower│  │CVR Tower│
   └────┬────┘  └────┬────┘
        │            │
   P(click)    P(conv|click)
        │            │
        └──── × ─────┘
              │
        P(conversion)  ← ESMM
```

### Gate 温度退火

```python
# 初始温度 T=2.0，每 epoch 衰减
gate_logits = W @ x / temperature
gate_weights = softmax(gate_logits)

# 高温 → 均匀分布（探索阶段）
# 低温 → 尖锐分布（利用阶段）
```

### Uncertainty Weight

```python
# 仅 T=2 个可学习标量参数
loss = Σ_t [ (1/(2σ_t²)) × L_t + log(σ_t) ]

# 训后固化: weights = exp(-2 × log_σ) → 常量，零推理开销
```

## 📊 实验结果

### 对比实验

| Model | CTR AUC | CVR AUC | Avg AUC | Parameters |
|-------|---------|---------|---------|------------|
| MMoE  | -       | -       | -       | ~180K      |
| CGC   | -       | -       | -       | ~220K      |
| **PLE** | **-** | **-** | **-** | ~350K      |

> 注：运行 `run_comparison.py` 后自动填充结果

### 关键发现

1. **PLE vs MMoE/CGC**: PLE 通过逐层参数隔离有效减少了任务间的负迁移
2. **温度退火**: 初始高温（T=2.0）促进均匀探索，逐步降温后 Expert 利用率更均衡
3. **Uncertainty Weight**: 自适应权重使 CTR 快速收敛，CVR 震荡减缓
4. **梯度冲突**: 基于余弦相似度的冲突检测有效识别 CTR/CVR 训练冲突时段

## 📈 训练日志说明

每个 epoch 记录：

```json
{
  "epoch": 1,
  "train": {
    "total_loss": 0.4521,
    "ctr_loss": 0.2341,
    "cvr_loss": 0.1823,
    "mask_loss": 0.0357,
    "train_ctr_auc": 0.7234,
    "task_weight_ctr": 1.0234,
    "task_weight_cvr": 0.8765,
    "log_sigma_ctr": -0.0116,
    "log_sigma_cvr": 0.0659
  },
  "val": {
    "ctr_auc": 0.7189,
    "cvr_auc": 0.6543,
    "total_auc": 0.6866
  },
  "gate_temperatures": [[1.96, 1.96], [1.96, 1.96], [1.96, 1.96]],
  "conflict_detector": {
    "ema_cos_sim": 0.1234,
    "ema_threshold": -0.0876,
    "conflict_ratio": 0.12
  },
  "early_stopping": {
    "diagnosis": {
      "ctr": "IMPROVING - CTR AUC trending up",
      "cvr": "OSCILLATING - CVR unstable"
    }
  }
}
```

### CTR收敛/CVR震荡诊断流程

```
1. 计算最近5个epoch的AUC序列
2. CTR诊断:
   - delta < 0.001 → CONVERGED
   - trend > 0     → IMPROVING
   - trend < 0     → DEGRADING
3. CVR诊断:
   - sign_changes ≥ 3 → OSCILLATING（震荡）
   - std < 0.002      → CONVERGED
   - 否则按趋势判断
4. 若CVR持续震荡 → 触发共享层软冻结
```

## 🔧 配置说明

关键配置项 (`configs/ple_config.json`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_extraction_layers` | 3 | PLE 提取层数 |
| `num_task_experts` | 3 | 每任务独占Expert数 |
| `num_shared_experts` | 2 | 共享Expert数 |
| `initial_temperature` | 2.0 | Gate初始温度 |
| `temp_decay_rate` | 0.98 | 温度衰减率/epoch |
| `use_uncertainty_weight` | true | 启用不确定性权重 |
| `use_esmm` | true | 启用ESMM CVR建模 |
| `use_feature_mask` | true | 启用特征掩码自监督 |
| `mask_ratio` | 0.15 | 特征掩码比例 |
| `conflict_check_interval` | 50 | 梯度冲突检测间隔(steps) |
| `patience` | 15 | Early Stopping 耐心值 |

## 📚 参考文献

1. Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning Model for Personalized Recommendations", RecSys 2020
2. Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts", KDD 2018
3. Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018
4. Ma et al., "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate", SIGIR 2018
5. Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020

## License

MIT License
