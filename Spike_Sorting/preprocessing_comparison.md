# 数据预处理流程对比

## 1. eval_model.ipynb 预处理流程

### 步骤：
1. `unsigned_to_signed(recording)` - 将无符号数据转换为有符号
2. `bandpass_filter(recording_raw, freq_min=300, freq_max=3000)` - 带通滤波
3. `notch_filter(recording_recorded, freq=50)` - 陷波滤波
4. `common_reference(recording_recorded, reference="global", operator="median")` - 公共参考
5. `recording_f.get_traces()` - 获取traces
6. `traces.astype(np.float32)` - 转换为float32

### 用于matching的waveform提取：
- 从 `traces_original` 提取（经过上述预处理，但**未白化**）
- 窗口大小：left 20, right 40 (60 samples)
- 数据范围：用户报告为 -10 到 10

## 2. generate_neuron_inf_phy_template.py 预处理流程

### 步骤：
1. `unsigned_to_signed(recording_raw)` - 将无符号数据转换为有符号
2. `bandpass_filter(recording_raw, freq_min=300, freq_max=3000)` - 带通滤波
3. `notch_filter(recording_f, freq=50)` - 陷波滤波
4. `common_reference(recording_f, reference="global", operator="median")` - 公共参考
5. `recording.get_traces()` - 获取traces
6. `traces.astype(np.float32)` - 转换为float32

### 用于matching的waveform提取：
- 从 `traces` 提取（经过上述预处理，但**未白化**）
- 窗口大小：left 20, right 40 (60 samples)
- 数据范围：用户报告为 -250 到 100

## 3. 关键发现

### 预处理步骤一致：
- ✅ 两者都使用相同的预处理步骤
- ✅ 两者都从预处理后的recording对象获取traces
- ✅ 两者都转换为float32
- ✅ 两者都使用相同的窗口大小（left 20, right 40, 60 samples）

### 数据范围差异：
- ❌ eval_model: -10 到 10
- ❌ generate_neuron_inf: -250 到 100

### 可能的原因：
1. **数据单位不同**：可能是μV vs 原始ADC值
2. **增益设置不同**：recording对象可能有不同的增益设置
3. **数据源不同**：可能使用了不同的recording文件
4. **缩放操作**：可能有隐式的缩放操作（但代码中没有发现）

## 4. 需要确认的问题

1. **recording路径**：
   - eval_model.ipynb 使用：`/media/ubuntu/sda/mouse_test/raw_data/WLF_128chmouse1_natima_RHD_251201_204035`
   - generate_neuron_inf_phy_template.py 使用：需要确认

2. **数据单位**：
   - 两者是否使用相同的数据单位？
   - 是否有增益或缩放差异？

3. **recording对象属性**：
   - 两者使用的recording对象是否有不同的增益设置？

## 5. 建议

为了确保一致性，建议：
1. 确认两个流程使用相同的recording文件
2. 检查recording对象的增益设置
3. 如果数据单位不同，需要统一单位（例如，都转换为μV）




