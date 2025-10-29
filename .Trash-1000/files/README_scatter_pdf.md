# 类别散点图PDF生成器

## 项目概述

基于 `/media/ubuntu/sda/visual_stimuli_pattern/dynamic/generate_video.ipynb` 的代码，我们创建了一个PDF生成器，为每个 `class_name` 生成一页散点图。

## 数据信息

- **总数据点**: 22,248 个
- **唯一类别数**: 1,854 个不同的 `class_name`
- **数据文件**:
  - `clustering_results.csv`: 包含聚类结果和类别信息
  - `tsne_features.csv`: 包含t-SNE降维后的二维坐标

## 生成的文件

### 1. 测试版本
- **文件**: `test_generate_class_scatter_pdf.py`
- **输出**: `test_class_scatter_plots.pdf`
- **内容**: 前5个类别的散点图（用于测试）
- **大小**: 1.68 MB

### 2. 最终版本
- **文件**: `generate_class_scatter_pdf_final.py`
- **输出**: `class_scatter_plots_final.pdf`
- **内容**: 所有1,854个类别的散点图
- **状态**: 正在生成中（当前约48MB）

## 散点图特性

每页散点图包含以下特性：

1. **目标类别**: 用红色高亮显示当前类别的所有数据点
2. **其他类别**: 用浅灰色显示所有其他类别的数据点
3. **统计信息**: 显示总点数、当前类别点数和占比
4. **坐标轴**: t-SNE降维后的二维坐标
5. **图例**: 清楚标识不同颜色的含义

## 技术特点

- **内存优化**: 定期清理内存，避免内存溢出
- **进度显示**: 使用tqdm显示生成进度
- **性能优化**: 降低DPI和点大小以提高生成速度
- **错误处理**: 包含数据验证和错误处理
- **字体支持**: 使用英文标签避免中文字体问题

## 使用方法

```bash
# 运行测试版本（生成5个类别的PDF）
python3 test_generate_class_scatter_pdf.py

# 运行完整版本（生成所有1,854个类别的PDF）
python3 generate_class_scatter_pdf_final.py

# 监控生成进度
python3 monitor_pdf_generation.py
```

## 预期结果

- **PDF页数**: 1,854页（每页一个类别）
- **预计文件大小**: 200-500 MB
- **生成时间**: 预计30-60分钟（取决于系统性能）

## 文件结构

```
dynamic/
├── generate_class_scatter_pdf_final.py      # 主生成脚本
├── test_generate_class_scatter_pdf.py       # 测试版本
├── monitor_pdf_generation.py               # 进度监控脚本
├── class_scatter_plots_final.pdf          # 完整PDF文件（生成中）
└── test_class_scatter_plots.pdf            # 测试PDF文件
```

## 注意事项

1. 生成完整PDF需要较长时间，建议在后台运行
2. 确保有足够的磁盘空间（至少1GB）
3. 可以使用监控脚本跟踪生成进度
4. 如果中断，可以重新运行脚本继续生成

