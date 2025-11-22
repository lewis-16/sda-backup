# Autosort vs 用户实现对比分析

## 性能差异
- **用户结果**: Noise Acc: 58.28%, Label Acc: 48.30%
- **Autosort论文结果**: Noise Acc: 89%, Label Acc: 95% (仅使用waveform)

## 关键差异分析

### 1. **输入特征构建（最重要）**

#### Autosort的实现（仅使用waveform时）：
```python
# model.py line 141-147
codes = batch_features  # (batch, ch_num * samplepoints) - 所有通道的waveform展平
codes = torch.cat((codes, single_waveform), axis=1)  # 添加单个通道的waveform (batch, samplepoints)
# 如果只使用waveform，不添加pred_loc
# 最终输入: (batch, ch_num*samplepoints + samplepoints)
```

**特征组成**（仅waveform）：
1. `batch_features`: 所有通道的完整waveform，形状 `(batch, ch_num, samplepoints)` 展平为 `(batch, ch_num * samplepoints)`
   - 从 `self.Img[index, ...]` 获取，即所有通道的30x30矩阵展平为900维
2. `single_waveform`: **检测通道的单个waveform**，形状 `(batch, samplepoints)`
   - 从 `self.Img_single[index, ...]` 获取
   - 提取方式：`datafile[np.arange(n_samples), channel_id, :]` (waveform_loader.py line 76)
   - 这是从**检测到的通道**提取的单个通道waveform，30维

**对于30通道、30时间点的情况**：
- 输入维度 = 30*30 + 30 = **930维**

#### 用户的实现：
```python
# train_spike_pipeline copy.py line 115-116
x_flat = x.reshape(x.size(0), -1)  # (batch, 30, 30) -> (batch, 900)
# 只使用了30x30的waveform矩阵展平，缺少单个通道的waveform
```

**问题**：
1. ❌ **缺少单个通道的waveform**：用户只使用了所有通道的30x30矩阵（900维），而autosort额外添加了检测通道的单个waveform（30维），总共930维
2. ❌ **没有从检测通道提取单个waveform**：用户的waveform提取代码（line 565-570）只提取了所有通道的waveform，没有额外提取检测通道的单个waveform

### 2. **模型架构差异**

#### Autosort的clssimp类：
```python
# model.py line 22-52
class clssimp(nn.Module):
    def __init__(self, ch=2880, num_classes=20):  # ch = (ch_num+1)*samplepoints (仅waveform时)
        self.pool = nn.AdaptiveAvgPool1d(output_size=(ch))  # 有pooling层！
        self.way1 = nn.Sequential(
            nn.Linear(ch, 1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )
        # ... 后续层相同
    
    def forward(self, x):
        x = self.pool(x[None, :])  # 先pooling
        x = x.reshape(x.size(1), -1)
        # ... 后续层
```

**注意**：AdaptiveAvgPool1d的输入需要是3D tensor `(batch, channels, length)`，但这里传入的是2D tensor，所以先做了`x[None, :]`增加维度。

#### 用户的SimpleClassifier：
```python
# train_spike_pipeline copy.py line 65-91
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        # 没有AdaptiveAvgPool1d层！
        self.way1 = nn.Sequential(
            nn.Linear(input_size, 1000, bias=True),
            # ... 后续层相同
```

**问题**：
1. ❌ **缺少AdaptiveAvgPool1d层**：虽然这个pooling层在autosort中的实际作用可能有限（因为输入已经是展平的），但这是架构差异之一
2. ⚠️ **输入维度不匹配**：用户输入是900维，autosort是930维（30*30 + 30）

### 3. **单个通道waveform提取缺失（关键）**

#### Autosort的提取方式：
```python
# waveform_loader.py line 76
self.Img_single = datafile[np.arange(datafile.shape[0]), np.array(channel_id).astype('int'), :]
# datafile形状: (n_samples, n_channels, n_timepoints)
# channel_id形状: (n_samples,) - 每个spike检测到的通道ID
# 结果: (n_samples, n_timepoints) - 每个spike在其检测通道上的waveform
```

**关键点**：
- `channel_id` 是每个spike被检测到的通道ID（从detection阶段获得）
- 从所有通道的waveform中，提取出检测通道的单个通道waveform
- 这个单个通道waveform与所有通道的waveform**拼接**作为输入

#### 用户的实现：
```python
# train_spike_pipeline copy.py line 565-570
for time_range in tqdm(np.arange(-left_sample, right_sample, dtype=np.int64)):
    indices = (X_spiketrain_time + time_range).astype(np.int64)
    if time_range == -left_sample:
        waveform = trace0_car[indices, :]
    else:
        waveform = np.dstack((waveform, trace0_car[indices, :]))
# 结果: (n_spikes, n_channels, n_timepoints) = (n_spikes, 30, 30)
# 只提取了所有通道的waveform，没有额外提取单个通道的waveform
```

**问题**：
- ❌ **没有提取单个通道的waveform**：用户有`Y_spiketrain_id_final`（检测通道ID），但没有用它来提取单个通道的waveform
- ❌ **输入特征不完整**：缺少了autosort中关键的单个通道waveform特征

### 4. **pos_weight计算方式差异**

#### Autosort：
```python
# waveform_loader.py line 88-90
self.pos_weight_noise = torch.tensor([
    -np.sum(self.GT_binary[:,0]-1)/np.sum(self.GT_binary[:,0]),
    -np.sum(self.GT_binary[:,1]-1)/np.sum(self.GT_binary[:,1])
])
# 注意：使用了负号，这可能是为了处理one-hot编码
# GT_binary是one-hot格式：[noise, spike]
```

#### 用户：
```python
# train_spike_pipeline copy.py line 828-831
pos_weight_noise = torch.tensor([
    spike_count / noise_count if noise_count > 0 else 1.0,  # noise类权重
    noise_count / spike_count if spike_count > 0 else 1.0   # spike类权重
])
```

**问题**：
- ⚠️ **计算方式不同**：autosort使用了负号和one-hot的减法操作，用户使用了简单的比例计算
- 这可能导致类别平衡效果不同

### 5. **pos_weight计算方式**

#### Autosort：
```python
# waveform_loader.py line 88-90
self.pos_weight_noise = torch.tensor([
    -np.sum(self.GT_binary[:,0]-1)/np.sum(self.GT_binary[:,0]),
    -np.sum(self.GT_binary[:,1]-1)/np.sum(self.GT_binary[:,1])
])
# 注意：使用了负号，这可能是为了处理one-hot编码
```

#### 用户：
```python
# train_spike_pipeline copy.py line 828-831
pos_weight_noise = torch.tensor([
    spike_count / noise_count if noise_count > 0 else 1.0,
    noise_count / spike_count if spike_count > 0 else 1.0
])
```

**问题**：计算方式不同，可能影响类别平衡。

### 6. **训练epochs**

- **Autosort**: 20 epochs
- **用户**: 210 epochs（但early stopping在3个epochs无改善后停止）

这个差异可能不是主要问题，但autosort只用20个epochs就达到了很好的效果。

## 修复建议

### 优先级1（最重要）：添加单个通道waveform提取

在`preprocess_data`函数中，提取waveform后添加：

```python
# 在提取waveform后（line 572之后）
# waveform形状: (n_spikes, n_channels, n_timepoints) = (n_spikes, 30, 30)

# 提取单个通道的waveform（从检测通道）
single_channel_waveform = waveform[np.arange(len(waveform)), Y_spiketrain_id_final, :]
# 形状: (n_spikes, n_timepoints) = (n_spikes, 30)

print(f"[INFO] Single channel waveform shape: {single_channel_waveform.shape}")
```

### 优先级2：修改输入特征构建

修改`MultiTaskAutoSort.forward`方法：

```python
def forward(self, x, single_waveform=None, mode='train'):
    """
    前向传播
    x: 所有通道的waveform，shape: (batch, 30, 30)
    single_waveform: 单个通道的waveform，shape: (batch, 30)
    """
    # 展平所有通道的waveform: (batch, 30, 30) -> (batch, 900)
    x_flat = x.reshape(x.size(0), -1)
    
    # 拼接单个通道的waveform: (batch, 900) + (batch, 30) -> (batch, 930)
    if single_waveform is not None:
        x_flat = torch.cat([x_flat, single_waveform], dim=1)
    
    # Noise分类器
    noise_output = self.clsfier_noise(x_flat)
    
    # Label分类器
    label_output = self.clsfier_label(x_flat)
    
    return noise_output, label_output
```

同时修改`MultiTaskDataset`：

```python
class MultiTaskDataset(Dataset):
    def __init__(self, waveforms, single_waveforms, noise_labels, cluster_labels, spike_mask, num_classes):
        self.waveforms = torch.FloatTensor(waveforms)
        self.single_waveforms = torch.FloatTensor(single_waveforms)  # 新增
        # ... 其他代码
    
    def __getitem__(self, idx):
        return {
            'waveform': self.waveforms[idx],
            'single_waveform': self.single_waveforms[idx],  # 新增
            'noise_label': self.noise_labels_onehot[idx],
            'cluster_label': self.cluster_labels_onehot[idx],
            'is_spike': self.spike_mask[idx]
        }
```

### 优先级3：修改模型输入维度

修改`SimpleClassifier`的初始化：

```python
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        # 如果使用AdaptiveAvgPool1d，需要调整
        # 但autosort中的pooling可能不是必需的，可以先不加
        self.way1 = nn.Sequential(
            nn.Linear(input_size, 1000, bias=True),  # input_size应该是930而不是900
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )
        # ... 后续层相同
```

在`train_multitask_model`中：

```python
# line 820
input_size = all_waveforms.shape[1] * all_waveforms.shape[2] + all_waveforms.shape[2]  # 30*30 + 30 = 930
# 而不是: input_size = all_waveforms.shape[1] * all_waveforms.shape[2]  # 900
```

### 优先级4：修正pos_weight计算

```python
# 修改pos_weight计算方式，与autosort一致
# line 828-831
noise_count = np.sum(all_noise_labels == 0)
spike_count = np.sum(all_noise_labels == 1)

# 使用autosort的方式（注意负号）
pos_weight_noise = torch.tensor([
    -np.sum(all_noise_labels == 0) / np.sum(all_noise_labels == 0) if noise_count > 0 else 1.0,
    -np.sum(all_noise_labels == 1) / np.sum(all_noise_labels == 1) if spike_count > 0 else 1.0
]).to(device)

# 或者更简单的方式（如果上面的计算有问题）：
pos_weight_noise = torch.tensor([
    spike_count / noise_count if noise_count > 0 else 1.0,
    noise_count / spike_count if spike_count > 0 else 1.0
]).to(device)
```

## 总结

**最关键的差异（仅使用waveform时）**：
1. ❌ **缺少单个通道的waveform** - **这是最关键的差异！**
   - Autosort输入：所有通道展平(900维) + 单个通道(30维) = **930维**
   - 用户输入：所有通道展平(900维) = **900维**
   - 缺少了检测通道的单个waveform特征，这是autosort的关键设计

2. ❌ **没有从检测通道提取单个waveform**
   - 用户有`Y_spiketrain_id_final`（检测通道ID），但没有用它来提取单个通道的waveform
   - 需要在提取waveform后，额外提取：`waveform[np.arange(n), channel_ids, :]`

3. ⚠️ **pos_weight计算方式不同** - 可能影响类别平衡，但不是主要问题

4. ⚠️ **缺少AdaptiveAvgPool1d层** - 这个层在autosort中可能作用有限，但也是架构差异

**结论**：
最可能的原因是**缺少单个通道的waveform特征**。Autosort的设计是同时使用：
- 所有通道的完整信息（30x30矩阵展平）
- 检测通道的单个通道信息（30维向量）

这种设计可以让模型同时关注全局和局部特征，提高分类性能。用户只使用了全局特征，缺少了关键的局部特征。

