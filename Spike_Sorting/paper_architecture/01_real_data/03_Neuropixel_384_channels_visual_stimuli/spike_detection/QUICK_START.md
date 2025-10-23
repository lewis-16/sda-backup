# Spike Classification 快速开始指南

## 代码优化总结

✅ **已完成的优化**：

1. ✨ **性能优化**：将窗口提取操作从循环内移到循环外，一次性提取所有数据
   - 原来：每个clique都要读取完整数据集 2 次（训练+验证）
   - 现在：所有cliques共享 2 次数据读取（训练+验证）
   - **加速比例**：约15倍（假设15个cliques）

2. 🎯 **单Cluster检测**：自动检测只有1个cluster的clique并跳过训练
   - 节省不必要的训练时间
   - 自动创建直接映射关系
   - 保存映射信息供后续使用

## 运行步骤

### 1. 运行Cell 0-8（准备工作）
按顺序运行前8个cells，准备数据和模型定义

### 2. 运行Cell 9（主要训练 - 已优化）

**代码会自动执行三个阶段**：

#### 阶段1：预处理（~1分钟）
- 确定每个clique对应哪些clusters
- 准备spike信息

#### 阶段2：批量提取窗口（~10-20分钟）⭐ **优化重点**
- **一次性**提取所有训练集窗口
- **一次性**提取所有验证集窗口
- 显示进度条，可以看到提取进度

#### 阶段3：训练模型（~数小时，取决于clique数量）
- 对每个clique训练5个模型（trails 1-5）
- 自动跳过单cluster的cliques
- 保存最佳模型和准确率

### 3. 运行Cell 10（保存结果）
查看训练结果摘要和统计信息

## 输出结果

### 训练模型
```
/spike_classification/train_result/810755807/{clique_id}/
  ├── spike_classification_model_1.pth
  ├── spike_classification_model_2.pth
  ├── spike_classification_model_3.pth
  ├── spike_classification_model_4.pth
  └── spike_classification_model_5.pth
```

### 评估结果
```
/spike_classification/eval_result/810755807/
  ├── classification_accuracy_dict.pkl  # 所有cliques的准确率
  ├── single_cluster_mapping.pkl        # 单cluster的映射关系
  └── {clique_id}/
      ├── accuracy_*.pkl                 # 每个trail的准确率
      └── single_cluster_mapping.pkl     # 单cluster的映射（如适用）
```

## 预期输出示例

```bash
================================================================================
开始 Spike Classification 训练
================================================================================

阶段1：预处理clique和cluster信息...
  完成预处理，共 15 个有效cliques

阶段2：一次性提取所有spike窗口...
  总计需要提取 35678 个spikes的窗口
  提取训练集窗口...
  训练集: 100%|██████████| 800/800 [12:34<00:00,  1.06it/s]
  训练集: 提取了 28542 个窗口
  提取验证集窗口...
  验证集: 100%|██████████| 200/200 [03:08<00:00,  1.06it/s]
  验证集: 提取了 7136 个窗口

阶段3：训练分类模型...
================================================================================

处理 Clique 3...
  Cluster数量: 5, Cluster IDs: [10, 25, 37, 42, 56]
  从预提取的窗口中获取训练数据...
  训练集: 2847 个样本
  从预提取的窗口中获取验证数据...
  验证集: 712 个样本
  Trail 1...
    Epoch 0: Best model saved with Accuracy: 0.2345
    Epoch 10: Best model saved with Accuracy: 0.6789
    ...
    Training stopped at epoch 45 with best Accuracy: 0.9234
  Trail 2...
  ...

处理 Clique 5...
  Cluster数量: 1, Cluster IDs: [123]
  ⚠️  只有1个cluster (原始ID: 123)，跳过训练
  📌 已创建映射：Clique 5 的所有spike → Cluster 123

...

================================================================================
Spike Classification 训练完成!
================================================================================
```

## 性能对比

| 项目 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 窗口提取（15 cliques） | ~5小时 | ~20分钟 | **15倍** |
| 单cluster处理 | 仍需训练 | 自动跳过 | **节省训练时间** |
| 代码可读性 | 较差 | 三阶段清晰 | ✅ |

## 注意事项

1. **内存使用**：阶段2会将所有窗口加载到内存中，确保有足够的RAM（建议32GB+）
2. **中断恢复**：如果在阶段3训练过程中中断，需要重新运行整个Cell 9
3. **数据检查**：确保spike_inf.tsv和cluster_inf.csv路径正确

## 故障排除

### 问题：内存不足
**解决**：可以考虑进一步优化，分批处理cliques

### 问题：某个clique没有数据
**原因**：该clique可能距离所有clusters都超过100um
**解决**：自动跳过，不影响其他cliques

### 问题：训练很慢
**原因**：GPU利用率低或batch_size太小
**解决**：检查CUDA是否可用，调整batch_size

## 下一步

训练完成后，可以使用训练好的模型进行推理：
1. 加载模型：`torch.load('spike_classification_model_*.pth')`
2. 对于单cluster的cliques，使用saved的映射直接分配
3. 对于多cluster的cliques，使用训练好的模型进行分类

---
**最后更新**: 2025-01-08



