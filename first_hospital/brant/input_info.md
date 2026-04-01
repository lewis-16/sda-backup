# 输入数据维度说明

## 模型概述

本模型是一个基于 Transformer 架构的 EEG 信号处理模型，包含三个主要编码器：
1. **InputEmbedding**：输入嵌入层
2. **TimeEncoder**：时间编码器 (Temporal Encoder)
3. **ChannelEncoder**：通道编码器 (Spatial Encoder)

---

## 输入数据

### 原始 EEG 信号 (data)

| 维度 | 大小 | 说明 |
|------|------|------|
| batch | 批次大小 | 可变 |
| ch_num | 通道数/电极数 | 可变，取决于数据采集设备 |
| seq_len | 15 | 时序片段数量 |
| seg_len | 1500 | 每个片段包含的数据点数 |

**形状**：`data.shape = (batch, ch_num, 15, 1500)`

### 功率谱 (power)

功率谱通过对原始信号进行频谱分析得到：
- 使用 `scipy.signal.periodogram` 计算功率谱密度
- 将频率范围划分为 8 个频段
- 对每个频段的功率求和后取 log10

| 频段索引 | 频率范围 (Hz) |
|----------|---------------|
| 0 | 4 - 8 |
| 1 | 8 - 13 |
| 2 | 13 - 30 |
| 3 | 30 - 50 |
| 4 | 50 - 70 |
| 5 | 70 - 90 |
| 6 | 90 - 110 |
| 7 | 110 - 128 |

| 维度 | 大小 | 说明 |
|------|------|------|
| batch | 批次大小 | 可变 |
| ch_num | 通道数/电极数 | 可变 |
| seq_len | 15 | 时序片段数量 |
| band_num | 8 | 频段数量 |

**形状**：`power.shape = (batch, ch_num, 15, 8)`

---

## 标签数据 (分类任务)

用于分类任务 (main_task=1) 的标签：

| 维度 | 大小 | 说明 |
|------|------|------|
| batch | 批次大小 | 可变 |
| ch_num | 通道数/电极数 | 可变 |
| seq_len | 15 | 时序片段数量 |

**形状**：`y_label.shape = (batch, ch_num, 15)`

**标签值**：二分类任务，标签为 `0` 或 `1`

---

## 数据流与维度变换

### 整体流程

```
原始信号 (data) + 功率谱 (power)
        ↓
   InputEmbedding (输入嵌入层)
        ↓
   TimeEncoder (时间编码器)
        ↓
   ChannelEncoder (通道编码器)
        ↓
   最终嵌入向量
```

### 各步骤维度变换

| 步骤 | 输入形状 | 输出形状 |
|------|----------|----------|
| InputEmbedding | `(batch, ch_num, 15, 1500)` + `(batch, ch_num, 15, 8)` | `(batch×ch_num, 15, d_model)` |
| TimeEncoder | `(batch×ch_num, 15, 2048)` | `(batch×ch_num, 15, 2048)` |
| 维度变换 | - | `(batch×15, ch_num, 2048)` |
| ChannelEncoder | `(batch×15, ch_num, 2048)` | `(batch×15, ch_num, 2048)` |
| 最终变换 | - | `(batch, ch_num, 15, 2048)` |

---

## 模型参数 (从预训练权重推断)

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 2048 | 模型隐藏层维度 |
| band_num | 8 | 频段数量 |
| seq_len | 15 | 序列长度 |
| seg_len | 1500 | 片段长度 |

---

## 预训练权重文件

| 文件 | 包含内容 |
|------|----------|
| `time_encoder.pt` | TimeEncoder (包含 InputEmbedding) + Transformer Encoder |
| `channel_encoder.pt` | ChannelEncoder (包含 Transformer Encoder + proj_out) |
