# eval_spike_pipeline.py 使用说明

## 输入路径配置

### 必需输入

1. **模型路径** (`model_paths`)
   - **AutoSort模型**（推荐，自动检测）:
     - `autosort_trail_1_noise_clsfier.pth` - Noise分类器权重（用作detection模型）
     - `autosort_trail_1_label_clsfier.pth` - Label分类器权重（用于提取100维中间层特征）
     - 说明: 如果这两个文件存在，会自动使用AutoSort模型
     - 注意: noise分类器同时承担detection的功能（区分noise和spike）
   
   - **独立模型**（如果AutoSort文件不存在）:
     - `detection_trail_1.pth` - Detection模型
     - `spike_classification_model_1.pth` - Classification模型（100维中间层）
     - 说明: 如果AutoSort文件不存在，会尝试使用独立的detection和classification模型

2. **Neuron信息文件** (`neuron_inf_path`)
   - 路径: `kilosort_spike_sorting/sorting_results/{date}/neuron_inf.pkl`
   - 格式: pickle文件
   - 说明: 包含训练集neuron的位置、waveform等信息，用于建立cluster到neuron的映射关系

3. **待评估的录音数据** (`new_recording_path`)
   - 路径: 新数据的.ns4文件路径
   - 格式: Blackrock NS4格式
   - 说明: 需要进行spike sorting的新录音数据

4. **通道配置** (`channel_indices`, `channel_position`)
   - `channel_indices`: 通道分组索引字典
   - `channel_position`: 通道位置字典（用于计算spike位置）

### 可选输入

5. **Ground Truth数据** (`gt_spike_inf_path`)
   - 路径: `kilosort_spike_sorting/sorting_new/{date}/spike_inf.tsv`
   - 格式: TSV文件，包含列：`time`, `cluster`（可能还有`neuron`或`Neuron`）
   - 说明: 如果提供，会在classification_results中添加ground truth信息，并生成混淆矩阵
   - 注意: 如果文件不存在，程序会自动跳过ground truth匹配

## 输出内容

所有输出文件保存在 `output_dir` 目录中。

### 1. 检测结果文件

#### `all_threshold_spikes.csv`
- **内容**: 所有通过阈值检测的spike时间点
- **列**: 
  - `time`: spike时间点（采样点数）

#### `detection_results.csv`
- **内容**: Detection模型的预测结果
- **列**:
  - `time`: spike时间点
  - `detection_predicted`: 是否被detection模型识别为spike (0/1)
  - `detection_score`: detection模型的输出分数

#### `classification_results.csv`
- **内容**: Classification和KMeans聚类的完整结果
- **列**:
  - `time`: spike时间点
  - `cluster_id`: KMeans预测的cluster ID
  - `neuron_id`: 映射到neuron_inf的neuron ID（预测结果）
  - `gt_cluster_id`: Ground truth cluster ID（如果提供了GT数据）
  - `gt_neuron_id`: Ground truth neuron ID，映射到neuron_inf（如果提供了GT数据）

#### `evaluated_spike_inf.csv`
- **内容**: 最终的spike train，只包含成功映射到neuron的spike
- **列**:
  - `time`: spike时间点
  - `neuron_id`: neuron ID

### 2. 可视化文件

#### `umap_visualization.pdf`
- **内容**: UMAP降维可视化图
- **说明**: 
  - 随机采样最多100,000个样本
  - 使用100维中间层特征进行UMAP降维
  - 包含两个子图：
    1. 按cluster_id着色
    2. 按neuron_id着色（与neuron_inf对应）

### 3. 评估结果文件（如果提供了Ground Truth）

#### `confusion_matrix.csv`
- **内容**: 混淆矩阵数据
- **格式**: CSV文件，使用 `pd.crosstab(neuron_id, gt_neuron_id)` 生成
- **说明**: 
  - 行: 预测的neuron_id
  - 列: Ground truth的neuron_id
  - 值: 匹配的spike数量

#### `confusion_matrix_heatmap.pdf`
- **内容**: 混淆矩阵热力图
- **说明**: 
  - 可视化predicted neuron_id vs ground truth neuron_id的混淆矩阵
  - 包含准确率信息
  - 使用seaborn绘制

## 处理流程

### 阶段1: 校准阶段（前60秒）
1. 阈值检测（std_multiplier=2.4, window_size=10）
2. Spike detection（使用detection模型筛选）
3. Spike classification（提取100维中间层特征）
4. KMeans聚类（使用100维中间层特征）
5. 与trainset neuron建立映射关系（neuron_inf）

### 阶段2: 后续数据处理（每500ms为单位）
1. 阈值检测（std_multiplier=2.4, window_size=10）
2. Spike detection（使用detection模型筛选）
3. Spike classification（提取100维中间层特征）
4. KMeans预测（使用训练好的KMeans模型）
5. 映射到neuron spiketrain

## 配置示例

```python
# 必需配置
pipeline_results_dir = 'path/to/pipeline_results'
model_paths = {
    'detection': os.path.join(pipeline_results_dir, 'detection_trail_1.pth'),
    'classification': os.path.join(pipeline_results_dir, 'spike_classification_model_1.pth')
}

neuron_inf_path = "path/to/neuron_inf.pkl"
output_dir = 'path/to/eval_results'
new_recording_path = 'path/to/new_recording.ns4'

# 可选配置
gt_spike_inf_path = "path/to/spike_inf.tsv"  # 可选，用于生成混淆矩阵

# 通道配置（根据实际probe配置）
channel_indices = {
    "1": [1, 3, 5, 7, 9, 11],
    "2": [13, 15, 17, 19, 21, 23],
    # ...
}

channel_position = {
    0: [650, 0],
    2: [650, 50],
    # ...
}
```

## 注意事项

1. **模型路径**: 确保模型路径与 `train_spike_pipeline.py` 保存的路径一致
2. **Neuron映射**: Ground truth的cluster需要能够映射到neuron_inf中的neuron，否则gt_neuron_id可能为None
3. **时间匹配**: Ground truth匹配使用±1个采样点的时间容差
4. **采样率**: 默认假设采样率为10kHz，如果不同需要修改代码中的sampling_rate参数
5. **内存使用**: 如果数据量很大，可视化时会随机采样100,000个样本以节省内存

