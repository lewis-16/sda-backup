# AutoSort 与 test_251121.ipynb 流程对比报告

## 差异总结（除位置信息外）

### 1. Common Reference 操作符不同 ⚠️

**test_251121.ipynb (Cell 2)**:
```python
recording_f = spre.common_reference(recording_f, reference="global", operator="median")
```

**AutoSort (auto_sorting.py, sorting.py)**:
```python
recording_cmr = spikeinterface.preprocessing.common_reference(
    recording_f, reference="global", operator="average"
)
```

**影响**: 
- `median` 对异常值更鲁棒，但计算成本更高
- `average` 更标准，计算更快
- 这会影响信号的预处理结果

---

### 2. 检测阈值参数不同 ⚠️

**test_251121.ipynb (Cell 4)**:
```python
spikes = detect_spike(
    trace0_car,
    thr_min=3.5,  # 注意：使用 3.5
    thr_max=30,
    distance=3,
    ch_max_simul_firing=5,
    wlen=5,
    prominence=10,
)
```

**AutoSort (detection.py:168)**:
```python
spikes = detect_spike(
    trace0_car,
    thr_min=3,  # 注意：使用 3
    thr_max=30,
    distance=3,
    ch_max_simul_firing=5,
    wlen=5,
    prominence=10,
)
```

**影响**: 
- `thr_min=3.5` 会导致更严格的检测阈值，可能漏检更多 spike
- `thr_min=3` 会检测到更多候选 spike，但可能包含更多噪声

---

### 3. 训练数据增强策略缺失 ❌

**AutoSort (detection.py:211-223)** - 在训练模式下会添加未匹配的 GT spike:
```python
print("### 4.5 add all gt")
mapped_ind = gt_label_array1[np.where(gt_label_array1 > -1)[0]].astype("int")
A = [i for i in np.arange(len(y_unit_id)) if i not in mapped_ind]

X_spiketrain_time_train = list(X_spiketrain_time) + list(
    np.array(spike_train_all)[A]
)
Y_spiketrain_id_train = list(Y_spiketrain_id) + list(np.array(y_unit_id)[A])
Y_spiketrain_id_final_train = list(Y_spiketrain_id_final) + list(
    np.array(gt_ch)[A]
)
```

**test_251121.ipynb**: 
- 没有这一步，只使用检测到的 spike 进行训练

**影响**:
- AutoSort 通过添加未匹配的 GT spike 来增加训练数据的完整性
- test_251121.ipynb 可能丢失一些真实 spike 的样本

---

### 4. 波形窗口参数注释不一致 ⚠️

**test_251121.ipynb (Cell 3)** - `extract_windows` 函数定义:
```python
def extract_windows(data, indices, window_size=30):
    left_sample = 30   # 与train_spike_pipeline.py保持一致
    right_sample = 30  # 与train_spike_pipeline.py保持一致
```

但实际使用 (Cell 5):
```python
left_sample = 10
right_sample = 20
```

**AutoSort (detection.py:112-113)**:
```python
left_sample,  # 通常为 10
right_sample, # 通常为 20
```

**影响**: 
- 函数定义与实际使用不一致，可能导致混淆
- 实际窗口长度均为 30 个采样点（10 前 + 20 后）

---

### 5. 数据处理流程顺序略有不同

**test_251121.ipynb**:
1. 检测 spike
2. 构建 detect_array
3. 加载 GT
4. 映射 GT 标注
5. 过滤边界附近的 spike
6. 提取波形

**AutoSort (detection.py)**:
1. 检测 spike
2. 构建 detect_array
3. 加载 GT
4. 映射 GT 标注
5. **添加未匹配的 GT spike (仅训练模式)**
6. 过滤边界附近的 spike
7. 提取波形

---

### 6. 模型输入维度不同（已知，位置信息相关）

**AutoSort**:
- 输入维度: `(ch_num+1)*samplepoints + loc_dim`

**test_251121.ipynb**:
- 输入维度: `(ch_num+1)*samplepoints`

---

### 7. 其他细节

**map_gt_annotation 函数实现**:
- test_251121.ipynb 使用了向量化优化版本（更快）
- AutoSort 使用原始循环版本

**数据保存**:
- AutoSort 会分别保存 train_data 和 test_data
- test_251121.ipynb 只保存了一种模式的数据

---

## 建议修改

1. **统一 Common Reference 操作符** - 建议使用 `average`（与 AutoSort 一致）
2. **统一检测阈值** - 建议使用 `thr_min=3`（与 AutoSort 一致）
3. **添加训练数据增强** - 建议添加未匹配 GT spike 的步骤（如果用于训练）
4. **修正窗口参数注释** - 修正 Cell 3 中的注释，使其与实际使用一致

---

## 总结

除了位置信息的引入外，主要差异在于：
1. Common Reference 操作符（`median` vs `average`）
2. 检测阈值参数（`3.5` vs `3`）
3. 训练数据增强策略（是否添加未匹配的 GT spike）
4. 一些代码实现细节的优化（如 map_gt_annotation 的向量化版本）

这些差异可能会影响最终的分类性能和结果。

