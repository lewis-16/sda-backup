# Utah模型维度测试 - 快速使用指南

## 🎯 目标
测试不同神经元数量（100, 200, 400, 600）对图像重建质量的影响

## 🚀 运行方法

```bash
# 直接运行（前台）
python model_utah.py

# 后台运行（推荐）
nohup python model_utah.py > dimension_test.log 2>&1 &

# 查看进度
tail -f dimension_test.log
```

## ⏱️ 预计时间
约8-12小时（4个维度 × 45 epochs × 2-3小时）

## 📊 输出文件

### 模型文件
- `var_utah_monkeyN_dim100.pth`
- `var_utah_monkeyN_dim200.pth`
- `var_utah_monkeyN_dim400.pth`
- `var_utah_monkeyN_dim600.pth`

### 评估结果
- `metric_cosine_linear_dim100_MonkeyN.npy`
- `metric_cosine_linear_dim200_MonkeyN.npy`
- `metric_cosine_linear_dim400_MonkeyN.npy`
- `metric_cosine_linear_dim600_MonkeyN.npy`

### 总结文件
- `dimension_test_summary_MonkeyN.npy` - 所有维度的总结结果

## 📈 查看结果

```python
import numpy as np

# 快速查看总结
summary = np.load('dimension_test_summary_MonkeyN.npy', allow_pickle=True).item()

print("维度测试结果:")
for result in summary['results']:
    print(f"维度 {result['dimension']:3d}: "
          f"余弦相似度 = {result['mean_cosine']:.4f} ± {result['std_cosine']:.4f}")
```

## 🔍 测试维度说明

| 维度 | 神经元数量 | 占比 | 说明 |
|------|-----------|------|------|
| 100  | 100       | 15%  | 最小测试集 |
| 200  | 200       | 30%  | 中小规模 |
| 400  | 400       | 60%  | 中大规模 |
| 600  | 600       | 90%  | 接近全集 |
| 669  | 669       | 100% | 完整数据（原始） |

## 🔧 修改测试维度

编辑 `model_utah.py` 第48行：

```python
# 当前设置
TEST_DIMENSIONS = [100, 200, 400, 600]

# 修改示例1: 更密集的测试
TEST_DIMENSIONS = [50, 100, 150, 200, 300, 400, 500, 600]

# 修改示例2: 只测试关键维度
TEST_DIMENSIONS = [100, 300, 500]
```

## ⚠️ 注意事项

1. **显存要求**: 确保GPU有足够显存（建议>16GB）
2. **磁盘空间**: 每个模型约2-3GB，确保有足够空间
3. **中断恢复**: 如中断可手动修改TEST_DIMENSIONS跳过已完成维度
4. **随机种子**: 固定为42，确保结果可重复

## 🐛 常见问题

### Q: 显存不足怎么办？
A: 编辑 `model_utah.py` 第197行，减小batch_size：
```python
batch_size = 4  # 从8改为4
```

### Q: 如何只测试某个维度？
A: 修改第48行：
```python
TEST_DIMENSIONS = [200]  # 只测试200维
```

### Q: 如何查看训练进度？
A: 程序会实时打印训练loss和accuracy，或查看log文件

### Q: 结果文件太大怎么办？
A: 可以删除.pth模型文件（约2-3GB/个），保留.npy结果文件（很小）

## 📚 详细文档

更详细的说明请参考：`model_utah_dimension_test_notes.md`

## ✅ 修改完成

- ✅ 添加神经元采样功能
- ✅ 实现多维度测试循环
- ✅ 自动评估和保存结果
- ✅ 生成总结报告
- ✅ 显存自动清理

## 🎉 就这么简单！

直接运行 `python model_utah.py`，等待结果即可！

