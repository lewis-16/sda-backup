# SortingAnalyzer recording 属性设置解决方案

## 问题说明

`SortingAnalyzer` 对象的 `recording` 属性是只读的（read-only property），无法直接设置。如果尝试执行 `analyzer.recording = new_recording`，会抛出以下错误：

```
AttributeError: property 'recording' of 'SortingAnalyzer' object has no setter
```

## 解决方案

### 方案 1：创建新的 SortingAnalyzer（推荐）

如果需要使用不同的 recording，必须创建一个新的 `SortingAnalyzer` 对象：

```python
import spikeinterface as si

# 假设你已经有了一个 sorting 对象和新的 recording
# 错误的方式：
# analyzer.recording = new_recording  # ❌ 这会报错

# 正确的方式：
new_analyzer = si.create_sorting_analyzer(
    sorting=sorting_kilosort4,  # 使用相同的 sorting
    recording=new_recording,      # 使用新的 recording
    format='binary_folder',
    folder=output_folder + '/analyzer_new'
)
```

### 方案 2：从已保存的 analyzer 加载并重新创建

如果你已经有一个保存的 analyzer，但想更换 recording：

```python
import spikeinterface as si

# 加载现有的 analyzer（只读取 sorting 信息）
existing_analyzer = si.load_sorting_analyzer(folder=existing_folder)

# 获取 sorting 对象
sorting = existing_analyzer.sorting

# 使用新的 recording 创建新的 analyzer
new_analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=new_recording,
    format='binary_folder',
    folder=new_output_folder
)
```

### 方案 3：如果只是想修改 recording 的某些属性

如果你只是想对 recording 进行预处理（如滤波、重采样等），应该在创建 analyzer 之前就处理好：

```python
import spikeinterface.preprocessing as spre

# 在创建 analyzer 之前进行预处理
recording_filtered = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)
recording_notched = spre.notch_filter(recording_filtered, freq=50)
recording_referenced = spre.common_reference(recording_notched, reference="global")

# 然后使用处理后的 recording 创建 analyzer
analyzer = si.create_sorting_analyzer(
    sorting=sorting_kilosort4,
    recording=recording_referenced,  # 使用处理后的 recording
    format='binary_folder',
    folder=output_folder
)
```

### 方案 4：使用 select_segments 或 select_channels（如果适用）

如果你的需求是选择特定的通道或时间段，可以使用 recording 的方法：

```python
# 选择特定通道
recording_selected = recording.select_channels(channel_ids=['ch1', 'ch2', 'ch3'])

# 选择特定时间段
recording_selected = recording.select_segments(segment_indices=[0, 1])

# 然后创建新的 analyzer
new_analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=recording_selected,
    format='binary_folder',
    folder=output_folder
)
```

## 注意事项

1. **创建新的 analyzer 会重新计算所有扩展（extensions）**：如果你已经计算了 waveforms、quality_metrics 等，需要重新计算。

2. **确保 sorting 和 recording 兼容**：
   - sorting 和 recording 必须有相同的通道数量
   - 采样率应该匹配
   - 时间范围应该兼容

3. **如果只是想访问 recording**：可以直接使用 `analyzer.recording` 来读取，只是不能修改。

## 示例代码

```python
import spikeinterface as si
import spikeinterface.preprocessing as spre

# 假设你已经有了一个 analyzer
# analyzer = si.load_sorting_analyzer(folder='path/to/analyzer')

# 如果你想使用不同的 recording：
# 1. 获取现有的 sorting
sorting = analyzer.sorting

# 2. 准备新的 recording（例如应用不同的预处理）
new_recording = spre.bandpass_filter(original_recording, freq_min=500, freq_max=5000)

# 3. 创建新的 analyzer
new_analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=new_recording,
    format='binary_folder',
    folder='path/to/new_analyzer'
)

# 4. 重新计算需要的扩展
new_analyzer.compute(['waveforms', 'quality_metrics'])
```

## 总结

**核心原则**：`SortingAnalyzer` 的 `recording` 属性是只读的，无法修改。如果需要使用不同的 recording，必须创建一个新的 `SortingAnalyzer` 对象。


