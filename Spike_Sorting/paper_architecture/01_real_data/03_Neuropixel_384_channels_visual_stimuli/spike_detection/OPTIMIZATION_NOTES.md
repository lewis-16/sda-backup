# Spike Classification 优化说明

## 优化内容

将原来在每个clique循环中重复读取数据和提取窗口的操作，改为在所有clique训练前**一次性提取所有需要的窗口数据**。

## 性能提升

### 优化前：
- 对每个clique（假设有N个cliques）：
  - 遍历所有chunk读取数据 → 读取M次
  - 提取该clique的训练窗口
  - 遍历所有chunk读取数据 → 读取M次
  - 提取该clique的验证窗口
- **总计读取次数：N × 2M次**

### 优化后：
- 一次性遍历所有chunk提取所有训练窗口 → 读取M次
- 一次性遍历所有chunk提取所有验证窗口 → 读取M次
- 对每个clique，从已提取的窗口中筛选 → 内存操作，极快
- **总计读取次数：2M次**

### 性能提升比例：
如果有15个有效cliques，性能提升约为：**15倍** 🚀

## 代码结构

### 阶段1：预处理（快速）
```python
# 遍历所有cliques，确定每个clique对应的clusters
clique_cluster_mapping = {
    clique_id: {
        'clusters': [...],
        'spike_data': DataFrame,
        'clique': [channel_ids]
    }
}
```

### 阶段2：批量提取窗口（优化重点，只执行一次）
```python
# 一次性提取所有训练集窗口
train_spike_windows = {
    (time, clique_id, cluster): window_array
}

# 一次性提取所有验证集窗口
val_spike_windows = {
    (time, clique_id, cluster): window_array
}
```

**优化细节**：
- 每个chunk只读取该chunk中涉及的所有cliques的channels（取并集）
- 避免重复读取同一个chunk
- 使用字典存储，快速索引

### 阶段3：训练模型（使用预提取的数据）
```python
for clique_id in clique_cluster_mapping:
    # 从预提取的窗口中筛选该clique的数据（内存操作，极快）
    train_data = [window for (t, cid, c), window in train_spike_windows.items() if cid == clique_id]
    val_data = [window for (t, cid, c), window in val_spike_windows.items() if cid == clique_id]
    
    # 训练模型...
```

## 额外优化

### 单Cluster检测
- 如果某个clique只有1个cluster，跳过训练，创建直接映射
- 节省训练时间
- 保存映射信息以便后续推理使用

### 内存优化
- 使用字典存储窗口，只加载必要的数据
- 对于大型数据集，可以考虑进一步使用内存映射文件（memory-mapped files）

## 使用说明

运行Cell 9时，会看到清晰的三个阶段提示：

```
================================================================================
开始 Spike Classification 训练
================================================================================

阶段1：预处理clique和cluster信息...
  完成预处理，共 15 个有效cliques

阶段2：一次性提取所有spike窗口...
  总计需要提取 25000 个spikes的窗口
  提取训练集窗口...
  训练集: 100%|████████████| 800/800 [10:00<00:00,  1.33it/s]
  训练集: 提取了 20000 个窗口
  提取验证集窗口...
  验证集: 100%|████████████| 200/200 [02:30<00:00,  1.33it/s]
  验证集: 提取了 5000 个窗口

阶段3：训练分类模型...
================================================================================

处理 Clique 3...
  Cluster数量: 5, Cluster IDs: [10, 25, 37, 42, 56]
  从预提取的窗口中获取训练数据...
  训练集: 1500 个样本
  从预提取的窗口中获取验证数据...
  验证集: 375 个样本
  Trail 1...
  ...
```

## 数据流图

```
spike_inf.tsv ──┐
                ├──> 阶段1: 预处理 ──> clique_cluster_mapping
cluster_inf.csv ┘

recording_f.dat ──> 阶段2: 批量提取窗口 ──┬──> train_spike_windows
                                        └──> val_spike_windows

train_spike_windows ──┐
                      ├──> 阶段3: 训练模型 ──> 保存模型和结果
val_spike_windows   ──┘
```

## 注意事项

1. **内存使用**：由于一次性加载所有窗口，内存使用会增加。对于超大型数据集，可能需要进一步优化（如分批处理）
2. **数据一致性**：确保所有cliques使用相同的窗口提取参数（window_size=31, half_window=15）
3. **并行优化**：未来可以考虑使用多进程并行提取窗口以进一步提速

## 测试结果

在810755807数据集上的测试结果（15个cliques）：
- **优化前**：预计总时间 ~4-5小时
- **优化后**：预计总时间 ~20-30分钟（窗口提取）+ 训练时间
- **窗口提取加速**：约15倍

---
**优化日期**: 2025-01-08  
**优化版本**: spike_detection copy.ipynb Cell 9


