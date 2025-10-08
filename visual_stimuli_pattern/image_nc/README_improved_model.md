# 改进的视觉重建模型 (model_nc_improved.py)

## 概述

这是针对您的nat60k数据集改进的视觉重建模型。该模型专门适配了FX8_nat60k_2023_05_16数据集的数据结构，用于从神经元响应重建视觉图像。

## 数据结构

### 输入数据格式
- **ss_all**: 形状为 (500, 10, 5804) 的神经元响应数据
  - 500个测试图像
  - 每个图像有10次重复试验
  - 5804个神经元
- **istim_ss**: 形状为 (500,) 的图像索引数组
  - 指向image文件夹中对应的图像文件

### 图像文件
- 图像文件夹: `/media/ubuntu/sda/visual_stimuli_pattern/image_nc/image/`
- 命名格式: `image_XXXXX.png` (例如: image_00000.png)
- 总图像数量: 38000张

## 主要改进

### 1. 新的数据集类 (Nat60kDataset)
```python
class Nat60kDataset(Dataset):
    def __init__(self, neural_data, image_indices, image_folder, transform=None):
        # 适配nat60k数据格式
        # 自动处理神经元响应的重复试验平均
        # 根据图像索引加载对应的图像文件
```

### 2. 数据加载函数
```python
def load_nat60k_data(data_path, image_folder):
    # 专门用于加载.npz格式的nat60k数据集
    # 自动提取ss_all和istim_ss数据
    # 提供详细的数据信息输出
```

### 3. 训练配置优化
- 自动检测神经元数量 (5804个)
- 适配模型输入维度
- 优化的训练参数设置

### 4. 图像重建和可视化
- 自动生成重建结果PDF文件
- 并排显示原始图像和重建图像
- 包含图像索引信息

## 使用方法

### 1. 环境要求
确保安装了以下依赖包：
```bash
pip install torch torchvision numpy scipy matplotlib pillow scikit-learn
```

### 2. 预训练模型
需要以下预训练权重文件：
- `vae_ch160v4096z32.pth` - VAE模型权重
- `var_d16.pth` - VAR模型权重

### 3. 运行训练
```bash
cd /media/ubuntu/sda/visual_stimuli_pattern/image_nc
python model_nc_improved.py
```

### 4. 测试数据加载
```bash
python test_data_only.py
```

## 训练过程

1. **数据加载**: 自动加载FX8_nat60k数据集
2. **数据预处理**: 对重复试验进行平均，加载对应图像
3. **模型构建**: 构建VAE和VAR模型，适配神经元数量
4. **训练**: 使用渐进式训练策略
5. **评估**: 定期评估模型性能
6. **重建**: 生成重建结果可视化

## 输出文件

- `var_FX8_nat60k.pth` - 训练好的VAR模型
- `reconstruction_results_FX8_nat60k.pdf` - 重建结果可视化
- `checkpoints/ckpt_epoch*.pth` - 训练检查点

## 关键特性

### 数据适配
- 自动处理500个测试图像
- 对每个图像的10次重复试验取平均
- 支持5804个神经元的响应数据

### 模型架构
- 使用VQ-VAE进行图像编码
- 使用VAR (Visual Autoregressive) 模型进行重建
- 渐进式训练策略

### 训练优化
- 自适应学习率
- 梯度裁剪
- 混合精度训练
- 定期模型保存

## 性能监控

训练过程中会输出：
- 每个epoch的训练损失和准确率
- 验证集上的损失和准确率
- 训练时间统计
- 梯度范数信息

## 故障排除

### 常见问题
1. **缺少预训练权重**: 确保VAE和VAR权重文件在正确位置
2. **内存不足**: 减少batch_size或使用梯度累积
3. **图像加载失败**: 检查图像文件路径和权限

### 调试建议
1. 首先运行 `test_data_only.py` 验证数据加载
2. 检查CUDA可用性: `torch.cuda.is_available()`
3. 监控GPU内存使用情况

## 扩展使用

该代码框架可以轻松适配其他nat60k数据集：
- L1_A1_nat60k_2023_03_06.npz
- L1_A5_nat60k_2023_02_27.npz
- FX9_nat60k_2023_05_15.npz
- FX10_nat60k_2023_05_16.npz
- FX20_nat60k_2023_09_29.npz

只需修改 `data_path` 变量即可。
