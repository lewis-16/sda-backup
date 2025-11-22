# MLP输入生成过程对比分析

## 文件对比
- `train_spike_pipeline.py` - 训练脚本
- `eval_spike_pipeline.py` - 评估脚本

## 关键差异总结

### 1. Waveform提取方式

#### train_spike_pipeline.py (第574-580行)
```python
# 参数定义
left_sample = 10   # spike前10个采样点
right_sample = 20   # spike后20个采样点
window_size = 30   # 总共30个采样点

# 数据形状
trace0_car: (time_points, n_channels)  # 已转置

# 提取方式
for time_range in np.arange(-left_sample, right_sample, dtype=np.int64):
    indices = (X_spiketrain_time + time_range).astype(np.int64)
    if time_range == -left_sample:
        waveform = trace0_car[indices, :]  # (n_spikes, n_channels)
    else:
        waveform = np.dstack((waveform, trace0_car[indices, :]))

# 最终形状: (n_spikes, n_channels, n_timepoints) = (n_spikes, 30, 30)
# 提取范围: 对于spike时间t，提取[t-10, t+19]，共30个时间点
```

#### eval_spike_pipeline.py (第65-79行, 第1374-1386行)
```python
# extract_windows函数
def extract_windows(data, indices, window_size=61):
    n_channels, time_length = data.shape
    half_window = window_size // 2
    
    for idx in indices:
        window = data[:, idx - half_window:idx + half_window + 1]
        windows.append(window)
    
    return np.array(windows)

# 调用时
window_size = 30
half_window = 15  # window_size // 2
window = data_chunk[:, start:end]  # 在process_data_chunk中

# 问题：extract_windows函数中
# window = data[:, idx - half_window:idx + half_window + 1]
# 当window_size=30, half_window=15时
# 提取的是 [idx-15, idx+16)，即32个点，而不是30个点！
```

### 2. 数据形状差异

| 文件 | 数据形状 | 提取方式 |
|------|---------|---------|
| train_spike_pipeline.py | `(time_points, n_channels)` | `trace0_car[indices, :]` 然后 `dstack` |
| eval_spike_pipeline.py | `(n_channels, time_points)` | `data[:, idx-half:idx+half+1]` 直接提取 |

### 3. 时间窗口提取范围

#### train_spike_pipeline.py
- **提取范围**: `[spike_time - 10, spike_time + 19]`
- **总点数**: 30个时间点
- **分布**: 前10个点 + spike点 + 后19个点

#### eval_spike_pipeline.py (当前实现)
- **提取范围**: `[spike_time - 15, spike_time + 16)` (由于half_window+1)
- **总点数**: **32个时间点** ⚠️ **这是错误的！**
- **应该**: `[spike_time - 15, spike_time + 14]` 或 `[spike_time - 14, spike_time + 15]` 共30个点

### 4. 关键问题

#### ⚠️ Bug: extract_windows函数提取的点数不正确

在`eval_spike_pipeline.py`的`extract_windows`函数中（第75行）：
```python
window = data[:, idx - half_window:idx + half_window + 1]
```

当`window_size=30`, `half_window=15`时：
- 提取范围：`[idx-15, idx+16)` 
- 实际提取了**32个点**，而不是30个点

**正确的实现应该是**：
```python
# 选项1: 对称窗口（前后各15个点，共30个点）
window = data[:, idx - half_window:idx + half_window]  # [idx-15, idx+15)，共30个点

# 选项2: 与train保持一致（前10后20，共30个点）
window = data[:, idx - 10:idx + 20]  # [idx-10, idx+20)，共30个点
```

### 5. 影响分析

这个差异会导致：
1. **输入维度不匹配**: 训练时输入是900维（30×30），评估时可能是960维（30×32）
2. **模型性能下降**: 如果模型期望30×30的输入，但收到30×32的输入，会导致错误
3. **特征提取不一致**: 时间窗口的不对称性不同，可能影响特征表示

### 6. 建议修复

修改`eval_spike_pipeline.py`中的`extract_windows`函数，使其与训练时保持一致：

```python
def extract_windows(data, indices, window_size=30):
    """根据给定的时间点索引提取窗口"""
    n_channels, time_length = data.shape
    left_sample = 10   # 与train保持一致
    right_sample = 20  # 与train保持一致
    
    windows = []
    for idx in indices:
        if idx < left_sample or idx >= time_length - right_sample:
            continue
        window = data[:, idx - left_sample:idx + right_sample]
        windows.append(window)
    
    return np.array(windows)
```

或者使用对称窗口（但需要确保与训练时一致）：
```python
def extract_windows(data, indices, window_size=30):
    n_channels, time_length = data.shape
    half_window = window_size // 2  # 15
    
    windows = []
    for idx in indices:
        if idx < half_window or idx >= time_length - half_window:
            continue
        # 修正：提取30个点，而不是32个点
        window = data[:, idx - half_window:idx + half_window]  # [idx-15, idx+15)，共30个点
        windows.append(window)
    
    return np.array(windows)
```

## 结论

**两个文件在生成MLP输入的过程中存在显著差异**：

1. ✅ **数据形状**: 两者都最终得到`(n_spikes, 30, 30)`的形状（但eval的实现有bug）
2. ❌ **时间窗口范围**: 
   - train: `[t-10, t+19]` (30个点，前10后20)
   - eval: `[t-15, t+16)` (32个点，错误！应该是30个点)
3. ❌ **提取方式**: 
   - train: 循环堆叠每个时间点的所有spike
   - eval: 直接提取每个spike的完整窗口（但点数错误）

**建议**: 修复`eval_spike_pipeline.py`中的`extract_windows`函数，使其与训练时的时间窗口范围完全一致。

