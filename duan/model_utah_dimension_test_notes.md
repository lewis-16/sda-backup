# Model Utah 维度测试修改说明

## 修改日期
2025年10月10日

## 修改目标
测试不同输入维度（神经元数量）对模型重建性能的影响，通过比较不同维度下的重建质量（余弦相似度），评估模型对神经元数量的敏感性。

## 测试维度
- **原始维度**: 669个神经元
- **测试维度**: [100, 200, 400, 600] 个神经元

## 主要修改内容

### 1. 添加神经元采样函数 `sample_neurons()`

**位置**: 第52-82行

**功能**: 
- 从原始MUA数据中随机采样指定数量的神经元
- 使用固定随机种子（seed=42）确保可重复性
- 返回采样后的数据和被选中的神经元索引

**代码**:
```python
def sample_neurons(mua_data, n_neurons, seed=42):
    np.random.seed(seed)
    total_neurons = mua_data.shape[1]
    
    if n_neurons >= total_neurons:
        return mua_data, np.arange(total_neurons)
    
    selected_indices = np.random.choice(total_neurons, n_neurons, replace=False)
    selected_indices = np.sort(selected_indices)
    sampled_data = mua_data[:, selected_indices]
    
    return sampled_data, selected_indices
```

### 2. 添加维度测试循环

**位置**: 第518-803行

**流程**:

```
for test_dim in [100, 200, 400, 600]:
    ├── 1. 采样神经元到指定维度
    ├── 2. 创建训练/测试数据集
    ├── 3. 构建模型（input_dim=test_dim）
    ├── 4. 训练模型（45个epochs）
    ├── 5. 保存模型（var_utah_monkeyN_dim{test_dim}.pth）
    ├── 6. 评估重建质量
    │   ├── 生成重建图像
    │   ├── 计算余弦相似度（重复5次）
    │   ├── 保存结果（metric_cosine_linear_dim{test_dim}_MonkeyN.npy）
    │   └── 打印统计结果
    └── 7. 清理显存

循环结束后：
└── 总结所有维度的结果并保存
```

### 3. 添加辅助评估函数

**位置**: 第253-380行（在TrainingConfig类之后）

**函数列表**:
- `compute_correlation()`: 计算Pearson相关系数
- `compute_cosine_similarity()`: 计算余弦相似度
- `correct_brightness_contrast_batch()`: 批量校正图像亮度和对比度

这些函数在循环中用于评估重建质量。

### 4. 数据结构调整

**修改前**:
```python
filtered_test_MUA = np.load(...)
```

**修改后**:
```python
filtered_test_MUA_full = np.load(...)  # 保存完整数据
TEST_DIMENSIONS = [100, 200, 400, 600]  # 定义测试维度
all_dimension_results = {}  # 存储所有维度的结果
```

## 输出文件

### 每个维度生成的文件

1. **模型文件**: `var_utah_monkeyN_dim{100|200|400|600}.pth`
   - 每个维度训练后的VAR模型

2. **评估结果**: `metric_cosine_linear_dim{100|200|400|600}_MonkeyN.npy`
   - 格式: numpy数组 (5, n_batches)
   - 5次重复，每次包含所有batch的余弦相似度

### 总结文件

**文件名**: `dimension_test_summary_MonkeyN.npy`

**内容**:
```python
{
    'results': [
        {'dimension': 100, 'mean_cosine': 0.xxx, 'std_cosine': 0.xxx},
        {'dimension': 200, 'mean_cosine': 0.xxx, 'std_cosine': 0.xxx},
        {'dimension': 400, 'mean_cosine': 0.xxx, 'std_cosine': 0.xxx},
        {'dimension': 600, 'mean_cosine': 0.xxx, 'std_cosine': 0.xxx}
    ],
    'dimensions': [100, 200, 400, 600]
}
```

## 训练配置

保持与原始配置相同：
- **Epochs**: 45
- **Batch size**: 8
- **Learning rate**: 1e-4
- **Optimizer**: AdamW
- **Label smoothing**: 0.1
- **Device**: cuda:1

每个维度独立训练，不共享权重。

## 评估指标

**主要指标**: 余弦相似度（Cosine Similarity）

**计算方式**:
1. 对每个batch进行5次重复重建
2. 使用linear方法校正亮度和对比度
3. 计算每张图像的余弦相似度
4. 对每个batch取平均
5. 最终统计所有batch的均值和标准差

## 实验意义

### 科学价值

1. **最小神经元数量**: 确定有效重建所需的最少神经元数量
2. **性能-维度关系**: 量化重建质量与神经元数量的关系
3. **冗余度评估**: 评估神经元群体编码的冗余程度
4. **优化指导**: 为实时解码系统提供神经元选择策略

### 预期结果

假设存在以下几种可能：

1. **线性关系**: 重建质量随神经元数量线性提升
2. **饱和效应**: 某个阈值后增加神经元数量收益递减
3. **阶跃变化**: 存在关键神经元数量阈值
4. **随机采样鲁棒性**: 不同神经元子集表现相似

## 使用方法

### 运行测试

```bash
python model_utah.py
```

程序会自动：
1. 顺序测试所有维度
2. 打印每个维度的进度和结果
3. 保存所有中间和最终结果

### 预计运行时间

假设单个维度训练45个epochs需要约2-3小时：
- **总时间**: 约8-12小时（4个维度）
- **建议**: 使用nohup或screen在后台运行

### 读取结果

```python
import numpy as np

# 读取总结结果
summary = np.load('dimension_test_summary_MonkeyN.npy', allow_pickle=True).item()
print(summary['results'])

# 读取某个维度的详细结果
dim_100_results = np.load('metric_cosine_linear_dim100_MonkeyN.npy')
print(f"维度100的余弦相似度: {dim_100_results.mean():.4f} ± {dim_100_results.std():.4f}")
```

## 后续分析建议

1. **可视化分析**:
   - 绘制维度-性能曲线
   - 比较不同维度的重建图像
   - 分析误差分布

2. **统计检验**:
   - 配对t检验比较不同维度
   - ANOVA分析维度效应
   - 效应量计算（Cohen's d）

3. **神经元重要性**:
   - 分析哪些神经元被选中
   - 评估不同脑区的贡献
   - 测试有目的的神经元选择（vs随机）

4. **模型分析**:
   - 比较不同维度模型的权重
   - 分析模型的注意力模式
   - 评估过拟合程度

## 注意事项

1. **随机性控制**: 
   - 使用固定随机种子（seed=42）
   - 确保不同维度选择的神经元集合一致（100⊂200⊂400⊂600）
   - 实际上当前实现是独立采样，如需嵌套采样需修改

2. **显存管理**:
   - 每个维度测试后清理显存
   - 如遇显存不足，可调整batch_size

3. **中断恢复**:
   - 如中断，可手动修改TEST_DIMENSIONS跳过已完成的维度
   - 已保存的模型和结果不会被覆盖

4. **结果验证**:
   - 检查每个维度的训练曲线
   - 验证重建图像质量
   - 确保评估指标计算正确

## 版本信息

- **原始版本**: 固定669维输入
- **修改版本**: 支持多维度测试循环
- **兼容性**: 保持原有代码结构，只是添加循环包装

## 未来改进方向

1. **更多维度**: 测试更密集的维度点（如每隔50个神经元）
2. **重复实验**: 使用不同随机种子重复整个实验
3. **其他指标**: 添加SSIM、PSNR等评估指标
4. **可视化**: 自动生成重建图像对比PDF
5. **智能采样**: 基于神经元响应特性的智能采样策略


