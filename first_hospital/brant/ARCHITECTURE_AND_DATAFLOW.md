# Brant 模型架构、数据准备与数据流

本文档基于 `brant/Brant_src` 源码整理，描述 Brant（颅内神经信号基础模型）的模型架构、数据准备与预处理方式，以及数据流动（含维度信息）。

---

## 1. 模型架构

Brant 采用**双编码器**结构：**时间编码器（TimeEncoder）** 和 **通道编码器（ChannelEncoder）**，对颅内神经信号进行先时间后通道的两阶段编码。下游任务在此基础上加任务头。

### 1.1 整体架构图

```
原始数据 (data) + 频谱功率 (power)
         │
         ▼
┌─────────────────────────────────────┐
│         InputEmbedding              │
│  (CNN/Linear 投影 + Band/Position)   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       TimeEncoder (Transformer)      │  ← 时间维度建模
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    ChannelEncoder (Transformer)      │  ← 通道维度建模
└─────────────────────────────────────┘
         │
         ▼
    嵌入表示 (emb) → 任务头 (MLP/Linear/pred_head)
```

### 1.2 核心模块

#### 1.2.1 InputEmbedding（输入嵌入）

- **作用**：将原始波形和频谱功率投影为 d_model 维 embedding
- **输入投影**：
  - `cnn`：Conv1d 堆叠（kernel 150/10、MaxPool 等）→ Linear → d_model
  - `linear`：`Linear(in_dim=seg_len, out_dim=d_model)`
- **编码**：
  - Band Encoding：`(band_num, d_model)` 可学习参数
  - Positional Encoding：`(seq_len, d_model)` 可学习参数
  - Mask Encoding：可选（用于预训练遮挡）
- **输出**：`(batch*ch_num, seq_len, d_model)`

#### 1.2.2 TimeEncoder（时间编码器）

- **结构**：`InputEmbedding` + `TransformerEncoder`
- **输入**：`data`, `power`, 以及可选的 mask 参数
- **作用**：在时间维度上建模，输出每个通道在每个时间步的表示
- **输出**：`(batch*ch_num, seq_len, d_model)`

#### 1.2.3 ChannelEncoder（通道编码器）

- **结构**：`TransformerEncoder` + `Linear(d_model → out_dim)`
- **输入**：经 transpose/reshape 后的 `time_z`，形状 `(batch*seq_len, ch_num, d_model)`
- **作用**：在通道维度上建模，融合多通道信息
- **输出**：`ch_z`, `rec`（rec 用于预训练重建）

#### 1.2.4 任务头（下游）

| 任务 | 说明 | 头结构 |
|------|------|--------|
| **Task 1：癫痫检测** | 二分类 | `MLP(in_dim=d_model, out_dim=2)` |
| **Task 2：预测** | 短期/长期/频率-相位 | `pred_head` (Linear + Dropout) |
| **Task 3：填补** | 信号填补 | `Linear(d_model → seg_len)` |

### 1.3 超参数（需在 main 中配置）

- `seg_len`（L）：每个时间段的采样点数。**论文/默认 L=1500**
- `seq_len`（S）：时间步数。**论文/默认 S=15**
- `d_model`（D）：模型维度
- `band_num`：频段数（默认 8）
- `time_ar_layer`, `time_ar_head`：时间 Transformer 层数、注意力头数
- `ch_ar_layer`, `ch_ar_head`：通道 Transformer 层数、注意力头数
- `input_emb_mode`：`'cnn'` 或 `'linear'`

---

## 2. 数据准备与预处理

### 2.1 数据格式

**微调 / 推理**（`utils.load_data`）：

- 每个患者目录包含：
  - `data.npy`：形状 `(ch_num, board_num, seq_len, seg_len)` 或类似 `(ch_num, sample_num, seq_len, seg_len)`
  - `power.npy`：对应频谱功率，`(ch_num, sample_num, seq_len, band_num)`
  - `label.npy`（可选）：任务 1 使用，形状与样本数对应

**预训练**（`pretrain/pre_utils.load_data`）：

- 每个样本目录包含：
  - `data.npy`：原始波形
  - `power.npy`：频谱功率（可预计算）

### 2.2 频谱功率计算（pre_utils.compute_power）

- 方法：`signal.periodogram` 计算功率谱
- 采样率：**256 Hz**（`utils.py` 与 `pretrain/pre_utils.py` 中传入 `compute_power` 的 fs 参数）
- 频段：8 个

```
f_thres = [4, 8, 13, 30, 50, 70, 90, 110, 128] Hz
```

- 每个频段内功率求和后取 log10
- 输出形状：在原始 data 最后一维后增加一维 8，即 `(..., 8)`

### 2.3 Mayo/FNUSA 预处理（mayo_fnusa_preprocess.py）

用于 Mayo 和 FNUSA 数据集：

1. 从 `segments.csv` 读取分段元数据，按 `patient_id`、`category_id` 筛选
2. **插值（interp）**：将每段数据插值到固定长度 `up_win_size`（如 15×1500）
3. **聚合（agg_data）**：将各段合并为 `x_{up_win_size}.npy` 和 `y_{up_win_size}.npy`

### 2.4 数据集类 BoardDataset

- **输入**：`data`, `power`, `y_label`（可选）
- **data 形状**：`(ch_num, board_num, seq_len, seg_len)`
- **索引**：按 `board_idx` 取单个 board 的所有通道数据
- **输出**：`(data[:, board_idx, :, :], power[:, board_idx, :, :], y_label[:, board_idx, :])`

### 2.5 数据划分

- `split_data_trvlts`：按 `src_ratio` / `val_ratio` / `test_ratio` 划分训练/验证/测试
- `split_data_ts`：仅划分测试集
- 划分前对样本做随机打乱

---

## 3. 数据流动（含维度）

### 3.1 符号说明

| 符号 | 含义 |
|------|------|
| B / bat_size | batch size |
| C / ch_num | 通道数 |
| S / seq_len | 时间步数（论文/默认 **S=15**） |
| L / seg_len | 每段时间的采样点数（论文/默认 **L=1500**） |
| D / d_model | 嵌入维度 |
| band_num | 频段数（8） |

### 3.2 get_emb 流程（微调/推理）

```text
输入 x: (B, C, S, L)
输入 power: (B, C, S, band_num)
──────────────────────────────────────────────────────────────
1. TimeEncoder
   x → InputEmbedding → (B*C, S, D)
   power → band_encoding → power_emb (B, C, S, D)
   input_emb = proj(x) + power_emb + positional_encoding
   trans_out: (B*C, S, D)
──────────────────────────────────────────────────────────────
2. Reshape & Transpose
   (B*C, S, D) → (B, C, S, D)
   → transpose(1,2) → (B, S, C, D)
   → reshape → (B*S, C, D)
──────────────────────────────────────────────────────────────
3. ChannelEncoder
   输入: (B*S, C, D)
   输出 emb: (B*S, C, D)
──────────────────────────────────────────────────────────────
4. Reshape
   (B*S, C, D) → (B, S, C, D)
   → transpose(1,2) → (B, C, S, D)
──────────────────────────────────────────────────────────────
输出 emb: (B, C, S, D)
```

### 3.3 各任务数据流

#### 任务 1：癫痫检测（Seizure Detection）

```text
x, power, label
  x: (B, C, S, L)
  power: (B, C, S, 8)
  label: (B, C, S) 或 (B, S, C) 等，需与样本一一对应

→ get_emb(x, power) → emb: (B, C, S, D)
→ reshape → (B*C*S, D)
→ MLP(emb) → logit: (B*C*S, 2)
→ CrossEntropyLoss(logit, label)
```

#### 任务 2：预测（短期/长期/频率-相位）

```text
输入 x: (B, C, his_len + fut_len, L)
历史 his_x: (B, C, his_len, L)，his_len = seq_len
未来 fut_x: (B, C, fut_len, L)

→ get_emb(his_x, his_power) → emb: (B, C, his_len, D)
→ reshape → (B*C, his_len, D)
→ mean(dim=-2) → (B*C, D)
→ pred_head(emb) → 预测值

ph_freq：预测相位 pred_ph 和频率 pred_freq
long/short：预测 fut_x（可能下采样）
```

#### 任务 3：填补（Imputation）

```text
x: (B, C, S, L)
随机 mask，mask_rate 比例置 0
masked_x = x * mask（被 mask 的位置为 0）

→ get_emb(masked_x, power) → emb: (B, C, S, D)
→ Linear(emb) → rec_x: (B, C, S, L)
→ MSE(ori_x[mask==0], rec_x[mask==0])
```

### 3.4 预训练数据流（基于 pre_utils / pre_model）

```text
data: (B, C, S, L)
power: (B, C, S, 8)

可选：随机 mask 部分 (ch, seg) 位置
masked_x = data，被 mask 位置替换为 mask_encoding

→ InputEmbedding (含 mask)
   masked_x → CNN/Linear → (B*C, S, D)
   power → band_encoding → power_emb
   input_emb = proj + power_emb + pos_enc

→ TimeEncoder
   (B*C, S, D)

→ Reshape & ChannelEncoder
   (B*S, C, D) → ChannelEncoder → rec: (B*S, C, L)

→ 损失：重建被 mask 的 segment（MSE 等）
```

### 3.5 维度汇总

| 阶段 | 张量 | 形状 |
|------|------|------|
| 原始数据 | data | (B, C, S, L) |
| 频谱功率 | power | (B, C, S, 8) |
| InputEmbedding 输出 | input_emb | (B×C, S, D) |
| TimeEncoder 输出 | time_z | (B×C, S, D) |
| ChannelEncoder 输入 | time_z | (B×S, C, D) |
| ChannelEncoder 输出 | emb | (B×S, C, D) |
| 最终 embedding | emb | (B, C, S, D) |
| 分类 logit | logit | (B×C×S, 2) |
| 填补重建 | rec_x | (B, C, S, L) |

---

## 4. 文件结构速查

| 文件 | 职责 |
|------|------|
| `pretrain/pre_model.py` | InputEmbedding, TimeEncoder, ChannelEncoder |
| `pretrain/pre_utils.py` | compute_power, 预训练数据加载、mask 生成 |
| `model.py` | MLP 分类头 |
| `utils.py` | get_emb, load_data, split_data, Metrics |
| `dataset.py` | BoardDataset |
| `main.py` | 参数解析、模型构建、训练/测试入口 |
| `mayo_fnusa_preprocess.py` | Mayo/FNUSA 数据预处理 |
