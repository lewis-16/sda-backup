# MUA分类网络使用说明

## 概述
这个项目实现了一个基于筛选后MUA数据的4层ResNet MLP分类网络，用于91类分类任务。

## 文件结构
- `mua_classification.py`: 核心模型和训练函数
- `run_training.py`: 完整的训练脚本
- `filtered_test_MUA_oracle.npy`: 筛选后的MUA数据（需要先运行数据筛选代码生成）
- `filtered_labels.npy`: 筛选后的标签（需要先运行数据筛选代码生成）

## 模型架构
- **输入层**: 线性投影到512维
- **ResNet块**: 4个残差块，维度为 [512, 1024, 1024, 1024]
- **特征层**: 1024维特征提取
- **分类器**: 1024维到91类的线性分类器

## 使用方法

### 1. 数据准备
首先需要运行数据筛选代码，生成以下文件：
- `filtered_test_MUA_oracle.npy`: 筛选后的MUA数据
- `filtered_labels.npy`: 筛选后的标签

### 2. 运行训练
```bash
cd /media/ubuntu/sda/Monkey/scripts
python run_training.py
```

### 3. 训练参数
- **批次大小**: 64
- **学习率**: 1e-3
- **优化器**: AdamW
- **损失函数**: CrossEntropyLoss
- **学习率调度**: ReduceLROnPlateau
- **训练轮数**: 50 epochs

### 4. 输出文件
训练完成后会生成以下文件：
- `best_mua_classifier.pth`: 最佳模型权重
- `training_history.png`: 训练历史图表
- `confusion_matrix.png`: 混淆矩阵
- `test_features.npy`: 测试集特征（1024维）
- `test_predictions.npy`: 测试集预测结果
- `test_targets.npy`: 测试集真实标签

## 模型特点
1. **残差连接**: 使用ResNet风格的残差块，有助于梯度传播
2. **批归一化**: 每个线性层后都有BatchNorm1d
3. **Dropout**: 防止过拟合
4. **梯度裁剪**: 防止梯度爆炸
5. **学习率调度**: 根据验证损失自动调整学习率

## 标签映射
- 原始标签会被重新映射到连续的整数范围 [0, num_classes-1]
- 支持任意数量的类别（默认91类）

## 评估指标
- 训练/验证准确率
- 训练/验证损失
- 混淆矩阵
- 分类报告（精确率、召回率、F1分数）

## 注意事项
1. 确保有足够的GPU内存（推荐8GB+）
2. 数据文件必须存在且格式正确
3. 训练过程中会自动保存最佳模型
4. 可以使用 `torch.load('best_mua_classifier.pth')` 加载训练好的模型

## 依赖库
- torch
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm

