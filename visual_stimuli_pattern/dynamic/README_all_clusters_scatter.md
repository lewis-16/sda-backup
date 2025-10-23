# 所有Cluster散点图PDF生成器

## 项目概述

基于您的需求，我创建了一个notebook来生成包含所有K-means cluster的散点图PDF，并在每个cluster位置标注cluster名称。

## 功能特性

### ✅ 已实现的功能

1. **所有cluster可视化**: 在一个散点图中显示所有20个K-means cluster
2. **颜色区分**: 每个cluster使用不同的颜色进行区分
3. **cluster标注**: 在每个cluster的中心位置标注cluster名称（C0, C1, C2, ...）
4. **PDF保存**: 将图形保存为高质量的PDF文件
5. **统计信息**: 显示总点数、cluster数量等统计信息
6. **图例**: 包含所有cluster的图例，显示每个cluster的点数

## 生成的文件

### 主要文件
- **`all_clusters_scatter.ipynb`** - 主notebook文件
- **`all_clusters_scatter_executed.ipynb`** - 执行后的notebook
- **`all_clusters_scatter_plot.pdf`** - 生成的PDF文件（0.37 MB，1页）

## 数据信息

- **总数据点**: 22,248 个
- **K-means cluster数量**: 20 个
- **Cluster ID**: 0-19
- **最大cluster大小**: 4,593 个点
- **最小cluster大小**: 938 个点
- **平均cluster大小**: 1,112.4 个点

## 图形特性

### 视觉效果
- **图形尺寸**: 16x12 英寸
- **分辨率**: 300 DPI（高质量）
- **颜色方案**: 使用tab20颜色映射，确保20个cluster都有不同颜色
- **透明度**: 0.7 alpha值，便于观察重叠区域

### 标注系统
- **标注格式**: C0, C1, C2, ..., C19
- **标注位置**: 每个cluster的中心位置
- **标注样式**: 白色背景框，黑色粗体文字
- **偏移**: 标注文字稍微偏移，避免遮挡数据点

### 布局优化
- **图例位置**: 图形右侧外部
- **统计信息**: 左上角显示总点数和cluster数量
- **网格**: 半透明网格线，便于读取坐标
- **坐标轴**: 清晰的t-SNE维度标签

## 使用方法

### 在Jupyter Notebook中运行
```python
# 运行所有cell即可生成PDF
# 或者单独运行特定函数：

# 生成PDF文件
generate_all_clusters_scatter_pdf()

# 在notebook中显示图形
visualize_all_clusters_in_notebook()
```

### 查看结果
- PDF文件位置: `/media/ubuntu/sda/visual_stimuli_pattern/dynamic/all_clusters_scatter_plot.pdf`
- 文件大小: 0.37 MB
- 页数: 1页

## 技术特点

1. **内存优化**: 使用plt.close()及时释放内存
2. **高质量输出**: 300 DPI确保打印质量
3. **自动布局**: tight_layout()自动调整布局
4. **颜色管理**: 使用matplotlib内置颜色映射
5. **标注算法**: 自动计算cluster中心位置进行标注

## 扩展功能

### 可自定义参数
- 图形大小: 修改figsize参数
- 点大小: 修改s参数
- 透明度: 修改alpha参数
- 标注样式: 修改annotate参数
- 颜色方案: 修改colors参数

### 可能的改进
1. 添加cluster边界线
2. 显示cluster密度信息
3. 添加交互式功能
4. 支持其他聚类算法结果

## 文件结构

```
dynamic/
├── all_clusters_scatter.ipynb              # 主notebook
├── all_clusters_scatter_executed.ipynb     # 执行后的notebook
├── all_clusters_scatter_plot.pdf          # 生成的PDF文件
├── clustering_results.csv                 # 聚类结果数据
└── tsne_features.csv                      # t-SNE特征数据
```

## 总结

成功创建了一个包含所有20个K-means cluster的散点图PDF，每个cluster都有：
- 独特的颜色标识
- 中心位置标注
- 详细的统计信息
- 高质量的PDF输出

这个可视化工具可以帮助您更好地理解数据在t-SNE空间中的聚类分布情况。
