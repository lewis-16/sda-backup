# 训练集生成和训练流程对比报告

## 一、训练集生成的差异

### 1.1 关键差异：未匹配 GT Spike 的添加 ❌❌❌ **最重要**

#### AutoSort (detection.py:211-223)
```python
print("### 4.5 add all gt")
mapped_ind = gt_label_array1[np.where(gt_label_array1 > -1)[0]].astype("int")
A = [i for i in np.arange(len(y_unit_id)) if i not in mapped_ind]

# 将未匹配的 GT spike 添加到训练集
X_spiketrain_time_train = list(X_spiketrain_time) + list(
    np.array(spike_train_all)[A]
)
Y_spiketrain_id_train = list(Y_spiketrain_id) + list(np.array(y_unit_id)[A])
Y_spiketrain_id_final_train = list(Y_spiketrain_id_final) + list(
    np.array(gt_ch)[A]
)
```

**流程说明**:
1. 首先处理**检测到的 spike** → 保存到 `test_data/`
2. 然后创建**训练集** (`train_data/`)：
   - 包含所有检测到的 spike（已匹配 + 未匹配）
   - **额外添加未匹配的 GT spike**（检测失败但 GT 中存在的 spike）

#### test_251121.ipynb
```python
# Cell 5: 只有映射和提取波形的步骤
# 没有步骤 4.5: 添加未匹配的 GT spike
# 直接使用检测到的 spike 作为训练数据
```

**流程说明**:
- 只使用检测到的 spike（无论是否匹配 GT）
- **没有添加未匹配的 GT spike**

### 影响分析

| 方面 | AutoSort | test_251121.ipynb |
|------|----------|-------------------|
| 训练样本数 | 检测到的 + 未匹配的 GT | 仅检测到的 |
| 数据完整性 | ✅ 更完整（包含漏检的真实 spike） | ❌ 可能遗漏 |
| 类别平衡 | ✅ 更平衡（补充了真实样本） | ❌ 可能不平衡 |

**这可能是最关键的差异！** AutoSort 通过添加未匹配的 GT spike 来：
1. 增加训练数据的完整性
2. 提高真实 spike 的覆盖率
3. 改善类别不平衡问题

---

### 1.2 数据集分离策略

#### AutoSort
- **训练模式** (`mode='train'`): 分别保存 `train_data/` 和 `test_data/`
  - `test_data/`: 检测到的 spike（用于测试）
  - `train_data/`: 检测到的 spike + 未匹配的 GT spike（用于训练）

#### test_251121.ipynb
- 只保存一个数据集（相当于 AutoSort 的 `test_data/`）
- 没有区分训练和测试数据

---

### 1.3 数据保存格式

#### AutoSort (detection.py:244-283)
```python
# 步骤 6: 保存 test_data (检测到的 spike)
save_obj(waveform, current_save_path + "/X_waveform")
save_obj(Y_spiketrain_id, current_save_path + "/Y_spike_id")
save_obj(Y_spiketrain_id_final, current_save_path + "/Y_spike_id_noise")
save_obj(X_spiketrain_time, current_save_path + "/X_spiketrain_time")

# 步骤 8: 保存 train_data (检测到的 + 未匹配的 GT)
save_obj(waveform, current_save_path + "/X_waveform")  # 训练集波形
save_obj(Y_spiketrain_id_train, current_save_path + "/Y_spike_id")
save_obj(Y_spiketrain_id_final_train, current_save_path + "/Y_spike_id_noise")
save_obj(X_spiketrain_time_train, current_save_path + "/X_spiketrain_time")
```

#### test_251121.ipynb (Cell 6)
```python
# 只保存一次数据
with open(train_data_dir / "X_waveform.pkl", "wb") as f:
    pickle.dump(X_waveform, f)
# ... 其他保存
```

**相同点**: 保存的文件格式和命名相同
**不同点**: AutoSort 会保存两套数据（train/test），test_251121.ipynb 只保存一套

---

## 二、训练流程的差异

### 2.1 Dataset 类的差异

#### AutoSort (waveform_loader.py:48-102)
```python
class waveformLoader(data.Dataset):
    def __init__(self, root, shank_channel, sensor_positions, Keep_id=None):
        # ... 加载数据 ...
        
        # 计算位置信息
        pred_location = location_cal_group(sensor_positions, datafile, channel_id)
        self.pred_location = pred_location
    
    def __getitem__(self, index):
        return (
            self.Img[index, ...],           # 多通道波形
            self.GT[index, ...],            # 单元分类标签
            self.GT_binary[index, ...],     # 噪声/非噪声标签
            self.Img_single[index, ...],    # 单通道波形
            self.pred_location[index, ...]  # 位置信息 ⚠️
        )
```

#### test_251121.ipynb (Cell 7)
```python
class SimpleWaveformLoader(data.Dataset):
    def __init__(self, root, shank_channel, Keep_id=None):
        # ... 加载数据（基本相同）...
        # 不计算位置信息
    
    def __getitem__(self, index):
        return (
            self.Img[index, ...],           # 多通道波形
            self.GT[index, ...],            # 单元分类标签
            self.GT_binary[index, ...],     # 噪声/非噪声标签
            self.Img_single[index, ...]     # 单通道波形
            # 不返回位置信息 ⚠️
        )
```

**差异**: 位置信息的计算和返回（已知差异）

---

### 2.2 模型输入构建的差异

#### AutoSort (model.py:141-147)
```python
def iter_model(self, batch_features, classify_labels, labels,
               single_waveform, pred_loc):
    self.optimizer.zero_grad()
    
    codes = batch_features
    codes = torch.cat((codes, single_waveform), axis=1)
    codes = torch.cat((codes, pred_loc), axis=1)  # 拼接位置信息
    
    cls_output = self.clsfier_noise(codes.float())
    # ...
```

#### test_251121.ipynb (Cell 8)
```python
def iter_model(self, batch_features, classify_labels, labels, single_waveform):
    self.optimizer.zero_grad()
    
    # 拼接 multi-waveform 和 single-waveform
    codes = torch.cat((batch_features, single_waveform), axis=1)
    # 不拼接位置信息
    
    cls_output = self.clsfier_noise(codes.float())
    # ...
```

**差异**: 输入维度
- AutoSort: `(ch_num+1)*samplepoints + loc_dim`
- test_251121.ipynb: `(ch_num+1)*samplepoints`

---

### 2.3 训练循环的差异

#### AutoSort (run.py:90-107)
```python
for batch_features, classify_labels, labels, single_waveform, pred_loc in tqdm(train_loader):
    classify_labels = classify_labels.to(device)
    batch_features = batch_features.view(-1, args.samplepoints*args.ch_num).to(device)
    labels = labels.to(device)
    single_waveform = single_waveform.to(device)
    pred_loc = torch.tensor(pred_loc).to(device)  # 位置信息转换
    
    train_loss1, train_loss2, train_loss3, test = autosort_model.iter_model(
        batch_features, classify_labels, labels, single_waveform, pred_loc
    )
    # ...
```

#### test_251121.ipynb (Cell 9)
```python
for batch_features, classify_labels, labels, single_waveform in tqdm(train_loader):
    batch_features = batch_features.view(-1, samplepoints * ch_num).to(device)
    classify_labels = classify_labels.to(device)
    labels = labels.to(device)
    single_waveform = single_waveform.to(device)
    # 不需要位置信息
    
    loss1, loss2, loss3, test = autosort_model.iter_model(
        batch_features, classify_labels, labels, single_waveform
    )
    # ...
```

**相同点**:
- 训练循环结构相同
- Loss 计算方式相同
- 验证流程相同
- 学习率、batch_size 等超参数相同

**不同点**:
- 位置信息的处理（已知差异）

---

### 2.4 损失函数和评估指标的差异

#### AutoSort vs test_251121.ipynb

| 项目 | AutoSort | test_251121.ipynb | 是否相同 |
|------|----------|-------------------|---------|
| 噪声分类损失 | `BCEWithLogitsLoss(pos_weight=pos_weight_noise)` | ✅ 相同 | ✅ |
| 单元分类损失 | `BCEWithLogitsLoss(pos_weight=pos_weight_label)` | ✅ 相同 | ✅ |
| Loss 权重 | `1000 * loss` | ✅ 相同 | ✅ |
| 验证准确率计算 | `accuracy_score(gt_all, pred_all)` | ✅ 相同 | ✅ |
| 验证 F1 分数 | `f1_score(..., average='micro')` | ✅ 相同 | ✅ |

**结论**: 损失函数和评估指标完全一致

---

### 2.5 数据划分策略

#### AutoSort (run.py:54-58)
```python
train, val = random_split(
    train_notpure_dataset, 
    [int(len(train_notpure_dataset) * 0.8),
     len(train_notpure_dataset) - int(len(train_notpure_dataset) * 0.8)]
)
train_loader = torch.utils.data.DataLoader(train, batch_size=512, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=512, shuffle=False)
```

#### test_251121.ipynb (Cell 9)
```python
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
```

**结论**: 数据划分策略完全相同（80/20 划分）

---

## 三、关键差异总结

### ❌❌❌ **最关键的差异：训练数据增强**

| 差异点 | AutoSort | test_251121.ipynb | 影响程度 |
|--------|----------|-------------------|----------|
| **未匹配 GT Spike 添加** | ✅ 有 | ❌ 无 | 🔴 **极高** |
| 位置信息 | ✅ 有 | ❌ 无 | 🔴 高（已知） |
| Common Reference | `average` | `median` | 🟡 中等 |
| 检测阈值 | `thr_min=3` | `thr_min=3.5` | 🟡 中等 |
| 训练循环 | ✅ 相同 | ✅ 相同 | - |
| 损失函数 | ✅ 相同 | ✅ 相同 | - |
| 评估指标 | ✅ 相同 | ✅ 相同 | - |

---

## 四、建议修改

### 1. **添加未匹配 GT Spike 步骤（最关键）**

在 test_251121.ipynb 的 Cell 5 中，在映射 GT 标注后添加：

```python
print("\n### 4.5 add all gt (训练数据增强)")

# 找出未匹配的 GT spike 索引
mapped_ind = gt_label_array1[np.where(gt_label_array1 > -1)[0]].astype("int")
unmatched_gt_indices = [i for i in np.arange(len(y_unit_id)) if i not in mapped_ind]

print(f"未匹配的 GT spike 数量: {len(unmatched_gt_indices)}")

# 构建增强后的训练数据
X_spiketrain_time_train = list(X_spiketrain_time) + list(
    np.array(spike_train_all)[unmatched_gt_indices]
)
Y_spiketrain_id_train = list(Y_spiketrain_id) + list(
    np.array(y_unit_id, dtype=object)[unmatched_gt_indices]
)
Y_spiketrain_id_final_train = list(Y_spiketrain_id_final) + list(
    np.array(gt_ch)[unmatched_gt_indices]
)

# 更新变量用于后续波形提取
X_spiketrain_time = np.array(X_spiketrain_time_train)
Y_spiketrain_id = np.array(Y_spiketrain_id_train)
Y_spiketrain_id_final = np.array(Y_spiketrain_id_final_train)

print(f"增强后的训练数据总量: {len(X_spiketrain_time)}")
```

### 2. **统一其他参数**
- Common Reference: 改为 `average`
- 检测阈值: 改为 `thr_min=3`

---

## 五、结论

除了位置信息外，**训练集生成的最大差异是未匹配 GT Spike 的添加**。这个差异可能导致：

1. **训练数据不完整**: test_251121.ipynb 可能遗漏了一些真实的 spike
2. **类别不平衡加剧**: 真实 spike 的样本可能不足
3. **模型性能下降**: 训练数据的质量和数量直接影响模型性能

训练流程本身（损失函数、优化器、训练循环等）基本相同，主要差异在于：
- 输入特征是否包含位置信息（已知）
- 训练数据是否包含未匹配的 GT spike（**新发现的关键差异**）

