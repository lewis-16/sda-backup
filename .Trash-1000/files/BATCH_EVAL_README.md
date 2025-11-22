# 批量评估脚本使用说明

## 脚本功能

`batch_eval.sh` 脚本用于批量运行spike sorting evaluation，自动处理 `sorting_new` 目录下除了 `021322`（训练集）外的所有月份数据。

## 使用方法

### 直接运行

```bash
cd /media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels
./batch_eval.sh
```

### 脚本功能说明

1. **自动发现日期**: 扫描 `sorting_new` 目录，找到所有日期目录（排除021322）
2. **文件检查**: 自动检查每个日期所需的文件是否存在：
   - `spike_inf.tsv` (ground truth数据)
   - `mouse6_{date}_natural_image_001.ns4` (录音数据)
3. **独立输出**: 每个日期的结果保存在 `eval_results/{date}/` 目录下
4. **进度显示**: 显示处理进度和预计剩余时间

## 配置说明

### 固定配置

- **Neuron信息**: 所有评估都使用 `sorting_results/021322/neuron_inf.pkl`
- **模型路径**: 
  - Detection: `pipeline_results/detection_trail_1.pth`
  - Classification: `pipeline_results/spike_classification_model_1.pth`

### 自动生成的路径

对于每个日期 `{date}`（如 `022522`）：

- **录音文件**: `/media/ubuntu/sda/data/mouse6/ns4/natural_image/mouse6_{date}_natural_image_001.ns4`
- **Ground Truth**: `kilosort_spike_sorting/sorting_new/{date}/spike_inf.tsv`
- **输出目录**: `eval_results/{date}/`

## 输出内容

每个日期的输出目录包含：

1. **检测结果CSV文件**:
   - `all_threshold_spikes.csv`
   - `detection_results.csv`
   - `classification_results.csv`
   - `evaluated_spike_inf.csv`

2. **可视化文件**:
   - `umap_visualization.pdf`

3. **评估文件**（如果提供了GT）:
   - `confusion_matrix.csv`
   - `confusion_matrix_heatmap.pdf`

## 单独运行某个日期

如果需要单独运行某个日期的评估：

```bash
python eval_spike_pipeline.py --date 022522
```

或者指定自定义路径：

```bash
python eval_spike_pipeline.py --date 022522 \
    --base-dir /path/to/base \
    --data-base-dir /path/to/data
```

## 注意事项

1. **文件存在性**: 脚本会自动检查必要文件是否存在，如果缺失会跳过该日期
2. **运行时间**: 每个日期的处理可能需要较长时间，脚本会显示进度和预计剩余时间
3. **错误处理**: 如果某个日期处理失败，脚本会继续处理下一个日期
4. **输出目录**: 每个日期的结果保存在独立的目录中，不会相互覆盖

## 示例输出

```
========================================
批量运行Spike Sorting Evaluation
========================================

找到 12 个日期需要处理:
  - 022223
  - 022522
  - 031722
  - 042422
  - 052422
  - 062422
  - 072322
  - 082322
  - 092422
  - 102122
  - 112022
  - 122022

是否继续? (y/n): y

========================================
[1/12] 处理日期: 022223
========================================
开始运行evaluation...
...
```

