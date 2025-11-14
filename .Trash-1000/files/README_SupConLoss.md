# Supervised Contrastive Loss 修改说明

## 🎯 目标
将原有的基于 CrossEntropy 的分类模型改造为使用 **Supervised Contrastive Loss**，以获得更适合无监督聚类的特征表示。

## ✅ 已完成的修改

### 1. 添加 SupConLoss 类 (Cell 4)
- 实现了 Supervised Contrastive Learning 损失函数
- 参考论文: Khosla et al. "Supervised Contrastive Learning." NeurIPS 2020
- 核心功能:
  - 让同类样本在特征空间中更加聚集
  - 让不同类样本更加分离
  - 使用L2归一化和温度缩放

### 2. 改进模型架构 (Cell 4)
**原始模型:**
```
Input -> FC1 -> ReLU -> FC2 -> ReLU -> FC3 (分类层) -> Output
```

**新模型:**
```
Input -> FC1 -> ReLU -> FC2 -> ReLU -> Features (50维)
                                        ├─> Projection Head (128维) -> 用于对比学习
                                        └─> Classification Head -> 用于分类
```

**关键改进:**
- 添加了投影头（Projection Head）用于对比学习
- 特征层（FC2的输出，50维）用于后续聚类
- 支持两种模式：
  - `mode='train'`: 返回投影特征、logits和原始特征
  - `mode='eval'`: 只返回特征向量（用于聚类）

### 3. 修改训练代码 (Cell 9)
**损失函数:**
- 原来: 单一 CrossEntropyLoss
- 现在: SupConLoss + CrossEntropyLoss 的加权组合
  ```python
  loss = lambda_supcon * supcon_loss + lambda_ce * ce_loss
  ```

**训练流程:**
1. 前向传播获取三个输出：投影特征、logits、原始特征
2. 计算两个损失：
   - SupCon Loss: 优化特征分离性
   - CE Loss: 保证分类准确性
3. 组合损失进行反向传播

**新增监控:**
- 每10个epoch打印详细的损失分解
- 显示 Train Accuracy 和 Test Accuracy

### 4. 更新评估代码 (Cell 32)
- 使用 `model(data, mode='eval')` 直接提取特征
- 简化了代码，移除了手动逐层计算
- 添加了特征维度打印用于验证

## 🎛️ 超参数配置

### 默认值
```python
lambda_supcon = 0.5      # 对比损失权重
lambda_ce = 0.5          # 交叉熵损失权重  
temperature = 0.07       # 温度参数
proj_dim = 128          # 投影空间维度
hidden_size2 = 50       # 特征空间维度（用于聚类）
```

### 调参指南

#### 1. lambda_supcon (对比损失权重)
- **增大到 0.7-0.8**: 如果聚类效果不理想，需要更强的特征分离
- **减小到 0.3**: 如果聚类效果已经很好，但分类准确率下降

#### 2. lambda_ce (交叉熵权重)
- **增大到 0.7-0.8**: 如果分类准确率不够，需要更强的监督信号
- **减小到 0.3**: 如果分类准确率已经足够，想要更强的特征学习

#### 3. temperature (温度参数)
- **0.05-0.07** (较小): 
  - 更严格的区分标准
  - 适合类别数多 (>50) 的情况
  - 特征分离更清晰
- **0.1-0.2** (较大):
  - 更宽松的区分标准
  - 适合类别数少的情况
  - 训练更稳定

#### 4. proj_dim (投影维度)
- **64-256**: 通常选择范围
- **128** (推荐): 平衡效果和计算开销
- 不需要太大，因为只用于辅助对比学习

## 📊 效果验证

### 方法1: UMAP可视化
运行 Cell 42-45，查看特征在2D空间的分布:
- 同类样本应该聚集成团
- 不同类样本应该有明显间隔

### 方法2: 聚类性能
运行 Cell 33-46，查看聚类结果的混淆矩阵:
- 对角线值应该更高（类内一致性）
- 非对角线值应该更低（类间分离性）

### 方法3: 定量指标
可以计算:
- **Silhouette Score**: 衡量聚类质量
- **Davies-Bouldin Index**: 越小越好
- **Calinski-Harabasz Score**: 越大越好

## 🔍 工作原理

### Supervised Contrastive Loss 原理

1. **正样本对**: 相同标签的样本对
2. **负样本对**: 不同标签的样本对

损失函数鼓励:
- 正样本对之间的相似度 ↑
- 负样本对之间的相似度 ↓

数学表达:
```
L = -log[ Σ exp(zi·zj⁺/τ) / Σ exp(zi·zk/τ) ]
```
其中:
- zi, zj⁺ 是同类样本的归一化特征
- zk 是所有其他样本
- τ 是温度参数

### 为什么适合聚类？

1. **类内紧凑性**: 同类样本被拉近
2. **类间分离性**: 不同类样本被推开
3. **泛化性**: 学到的特征表示更具判别性
4. **鲁棒性**: 对噪声和异常值更鲁棒

## 🚀 使用建议

### 训练建议
1. **首次训练**: 使用默认参数
2. **观察输出**: 关注 SupCon Loss 和 CE Loss 的比例
3. **调整权重**: 根据聚类效果调整 lambda_supcon
4. **Fine-tuning**: 如果需要，调整 temperature

### 评估建议
1. **定期检查**: 每隔几个epoch检查特征分布
2. **对比实验**: 与原始 CrossEntropy 模型对比
3. **下游任务**: 最终以聚类质量为准

### 常见问题

**Q: SupCon Loss 很大但 CE Loss 很小，怎么办？**
A: 增大 lambda_ce 或减小 lambda_supcon

**Q: 分类准确率下降了，怎么办？**
A: 增大 lambda_ce，或者增大 temperature 让训练更稳定

**Q: 聚类效果没有明显提升，怎么办？**
A: 
- 增大 lambda_supcon 到 0.7-0.8
- 降低 temperature 到 0.05
- 增加训练 epoch

**Q: 训练不稳定，loss震荡？**
A: 
- 增大 temperature 到 0.1
- 减小学习率
- 增大 batch size

## 📝 代码示例

### 提取特征用于聚类
```python
model.eval()
features_list = []

with torch.no_grad():
    for batch_data, _ in dataloader:
        batch_data = batch_data.to(device)
        # 使用 eval 模式获取特征
        features = model(batch_data, mode='eval')
        features_list.append(features.cpu().numpy())

all_features = np.vstack(features_list)

# 使用KMeans聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=50, random_state=42)
cluster_labels = kmeans.fit_predict(all_features)
```

### 可视化特征分布
```python
from sklearn.decomposition import PCA
import umap

# PCA降维
pca = PCA(n_components=20)
features_pca = pca.fit_transform(all_features)

# UMAP可视化
reducer = umap.UMAP(n_components=2, random_state=42)
features_2d = reducer.fit_transform(features_pca)

# 绘图
plt.figure(figsize=(10, 10))
plt.scatter(features_2d[:, 0], features_2d[:, 1], 
           c=labels, cmap='tab20', s=1, alpha=0.5)
plt.title('Feature Distribution (colored by true labels)')
plt.show()
```

## 📚 参考文献

1. Khosla, P., et al. (2020). "Supervised Contrastive Learning." NeurIPS 2020.
2. Chen, T., et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.
3. Wen, Y., et al. (2016). "A Discriminative Feature Learning Approach for Deep Face Recognition." ECCV 2016.

## 🎉 总结

使用 Supervised Contrastive Loss 后，模型将:
- ✅ 学习到更适合聚类的特征表示
- ✅ 同类样本在特征空间中更紧凑
- ✅ 不同类样本在特征空间中更分离
- ✅ 保持良好的分类性能

这将显著提升后续无监督聚类的效果！

