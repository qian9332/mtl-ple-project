# 🔬 PLE 多任务学习 — 完整优化旅程

> 本目录记录了从「CGC 意外胜出」到「PLE 显著领先」的完整优化过程。
> 每一步的发现、分析、修正、代码变更和训练数据都有详细记录。

## 📖 阅读顺序

| 文件 | 内容 | 关键产出 |
|------|------|----------|
| [STEP_0_baseline.md](STEP_0_baseline.md) | 第一轮实验：5K 小样本基线 | 发现 CGC > PLE 的反常现象 |
| [STEP_1_root_cause_analysis.md](STEP_1_root_cause_analysis.md) | 根因分析：为什么 PLE 输了？ | 定位 4 个核心 Bug |
| [STEP_2_architecture_fix.md](STEP_2_architecture_fix.md) | 架构修复：代码层面逐一修正 | 新旧代码 Diff 对比 |
| [STEP_3_full_data_training.md](STEP_3_full_data_training.md) | 500K 全量数据训练 | 完整训练日志 + 逐 epoch 分析 |
| [STEP_4_final_comparison.md](STEP_4_final_comparison.md) | 最终对比 & 结论 | PLE 反超，梯度冲突实证 |
| [APPENDIX_training_data.md](APPENDIX_training_data.md) | 训练数据全貌 | 所有数值、曲线、诊断状态 |

## 🎯 一句话总结

**PLE 在 5K 样本上输给 CGC，是因为 4 个工程 Bug（过参数化 / 信息瓶颈 / 温度退火太慢 / 梯度路径过长）。修复后在 500K 样本上，PLE 以 Avg AUC 0.6773 显著领先 CGC 的 0.6602（+1.7%）和 MMoE 的 0.6579（+1.9%）。**
