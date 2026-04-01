# BaRISTA 替换为自建 sEEG 数据：必要数据与维度说明

本文档说明若要将 BaRISTA 改为使用你自己的 sEEG 数据，**模型真正依赖的数据**以及**各数据的维度要求**。

---

## 一、模型实际用到的数据流

1. **原始数据** → 预处理（滤波、重参考等）→ **按段切分** → 保存为 `.pt` 片段
2. **训练/验证/测试** 时读取 `.pt` 片段，每个片段包含 `x` 和标签
3. **Tokenizer** 接收 `x`，形状为 `(B, N, D)` = (batch, 时间点数, 通道数)
4. **空间编码** 依赖每个 subject_session 的 **空间分组**（coords / destrieux / lobes）

因此你需要准备：**原始神经数据**、**电极/通道信息**、**空间定位（用于空间分组）**、以及（若做监督任务）**标签与划分信息**。

---

## 二、必要数据及维度（按阶段）

### 1. 原始神经数据（必须）

| 项目 | 要求 |
|------|------|
| **格式** | 每个 trial/session 一个文件。原版为 HDF5（`sub_X_trial00Y.h5`），你可用 `.h5` 或预处理后直接存成 `.pt`。 |
| **单文件内容** | 连续多通道 sEEG，**形状**：`(n_samples, n_channels)`，即 **时间 × 通道**。 |
| **数据类型** | 浮点，单位与预处理一致（如 μV）。 |
| **采样率** | 原版 `samp_frequency: 2048` Hz；若你不同，需在 config 中改 `samp_frequency`，并注意与“片段长度”一致。 |

**H5 原版约定**（若你完全仿照 Brain Treebank）：  
- 键为 `electrode_0`, `electrode_1`, ...，每个键为一维数组长度 `n_samples`。  
- 读入后堆叠转置得到 `(n_samples, n_channels)`。

**维度小结**：单 trial 原始数据 = **`(n_samples, n_channels)`**。

---

### 2. 片段数据（模型直接输入）

预处理 + 切段后，每个片段保存为一个 `.pt` 文件，训练时被直接加载。

| 键 | 类型 | 形状 / 含义 |
|----|------|-------------|
| **`x`** | `torch.Tensor` | **`(segment_len, n_channels)`**。单段神经数据，已 z-score 归一化。 |
| **`timestamps`** | `torch.Tensor` | 可选，与 `x` 时间点对应的索引或时间戳。 |
| **`<experiment>`** | `torch.Tensor` 或 标量 | 该片段标签。二分类时为 0/1；回归/多分类时形状需与下游一致。 |

**片段长度**（时间维）由配置决定：

- `segment_length_s`（秒）× `samp_frequency`（Hz）= **segment_len**
- 原版：`3 × 2048 = 6144`，即 **`(6144, n_channels)`**。

**结论**：  
- 每个 `.pt` 里的 **`x` 必须是 `(segment_len, n_channels)`**，例如 `(6144, n_channels)`。  
- 若你改 `segment_length_s` 或 `samp_frequency`，`segment_len` 要一起改。

---

### 3. Tokenizer / 模型输入维度（必须满足）

在 `barista/models/tokenizer.py` 中，输入为：

- **`x`**：`(B, N, D)` = **(batch_size, num_timepoints, num_channels)**

即：

- **N = segment_len**（如 6144）
- **D = n_channels**（该 session 的通道数）

**时间子段**（temporal subsegment）由 `model.yaml` 决定：

- `temporal_subsegment_len: 512`
- `temporal_subsegment_step: 512`
- `num_seconds: 3`，`samp_frequency: 2048`  
→ 总时间点 = 2048×3 = 6144，子段数 = 6144/512 = **12**。

若你改动 `segment_length_s` 或 `samp_frequency`，需保证：

- `segment_len = samp_frequency * segment_length_s`
- 与 `temporal_subsegment_len/step` 兼容（能被整除或按代码逻辑切分）。

**结论**：  
- 模型必要输入 = **`(B, segment_len, n_channels)`**，例如 **`(B, 6144, n_channels)`**。  
- 同一 batch 内通常来自同一 subject_session，即 **n_channels 固定**。

---

### 4. 空间分组（Spatial encoding，必须）

BaRISTA 用“空间分组”给每个通道一个 group id，用于空间编码。  
原版支持三种：`coords`、`destrieux`、`lobes`。  
**用自建 sEEG 时，最少要实现 `coords`（仅需电极坐标）**。

每个 **subject_session** 需要一份空间分组信息，包含：

- **group_components**：长度为 n_channels 的列表，每个元素为该通道的“分组信息”。
  - **coords**：每个通道一个三元组 `(L, I, P)` 或 `(x, y, z)`，整数坐标。
  - destrieux/lobes：需要脑区/叶标签，依赖定位文件。
- **group_ids**：长度为 n_channels 的列表，每个通道一个整数 group_id。
- **max_elements_for_component**、**padding_indices**：由 `metadata_spatial_groups` 使用，按 coords/destrieux/lobes 的约定填。

**维度小结**：  
- 每个 subject_session：**n_channels** 个通道 → **n_channels** 个 group_components，**n_channels** 个 group_ids。  
- 若只用 **coords**，你需要的是：**每个通道的 (L,I,P) 或 (x,y,z)**，以及据此算出的 group_ids（例如按网格/区域离散化）。

---

### 5. 电极与定位（用于预处理与空间分组）

原版依赖以下文件（你若写自己的数据加载，可只保留“逻辑等价”的信息）：

| 数据 | 用途 | 原版格式 / 等价信息 |
|------|------|----------------------|
| **电极名称列表** | 通道顺序、Laplacian 重参考筛选 | 原版：`electrode_labels.json`，列表长度 = n_channels。 |
| **坏通道/可用通道** | 剔除坏通道、选“干净”通道 | 原版：`corrupted_elec.json`、`clean_laplacian.json`。你可用一份“可用通道列表”替代。 |
| **每个通道的坐标** | 空间分组 `coords`、可选 destrieux/lobes | 原版：`localization/sub_X/depth-wm.csv`，列至少包含电极名、**L, I, P**（或 X,Y,Z）。 |
| **Destrieux/脑叶**（可选） | 若用 `destrieux` 或 `lobes` 分组 | 定位表中对应列（如 Destrieux、DesikanKilliany）。 |

**维度**：  
- 电极列表：**长度 = n_channels**（与数据矩阵的通道维一致）。  
- 定位表：**每行一个通道**，列包含电极名 + 坐标（+ 可选脑区）。

---

### 6. 标签与划分（监督任务）

- **片段级标签**：每个片段一个标签，存于该片段的 `.pt` 中（用 `experiment` 名作键）或存于 metadata 的 `label` 列。  
  - 二分类：0/1。  
  - 多分类/回归：形状与 `model.create_downstream_head(..., output_dim=...)` 一致。
- **Metadata**：至少需要 **subject_session、path、split、label、d_data（即 x 的 shape）、seq_len（= segment_len）** 等，以便 DataLoader 和 split 使用。

---

## 三、维度汇总表（便于对照）

| 数据 | 形状 / 维度 | 说明 |
|------|--------------|------|
| 单 trial 原始神经数据 | `(n_samples, n_channels)` | 连续 sEEG，时间 × 通道 |
| 单片段 `x`（.pt） | `(segment_len, n_channels)` | 例如 (6144, n_channels) |
| 模型 / Tokenizer 输入 `x` | `(B, segment_len, n_channels)` | B=batch_size |
| 片段长度 segment_len | `samp_frequency * segment_length_s` | 如 2048×3=6144 |
| 电极列表 | 长度 `n_channels` | 与数据通道维一一对应 |
| 空间分组 group_components / group_ids | 长度 `n_channels` | 每个通道一个分量、一个 group_id |
| 定位表 | 行数 = n_channels，列含电极名 + L,I,P（及可选脑区） | 用于空间分组与可选过滤 |

---

## 四、你需要做的替换工作（简要）

1. **原始数据**：提供 `(n_samples, n_channels)` 的 sEEG，采样率与 config 中 `samp_frequency` 一致（或修改 config）。
2. **预处理**：可沿用或改写 `BrainTreebankDatasetPreprocessor`（滤波、重参考、z-score），输出仍为 `(n_channels, n_samples)` 再转置为 `(n_samples, n_channels)` 存盘。
3. **切段**：按 `segment_length_s` 和 `samp_frequency` 切成长度 **segment_len** 的段，每段 **`(segment_len, n_channels)`**，存成 `.pt`（含 `x`、可选 `timestamps`、实验名键的标签）。
4. **空间分组**：为每个 subject_session 提供每个通道的坐标（及可选脑区），生成 **coords**（必选）以及可选的 destrieux/lobes 分组，写入 metadata 的 spatial_groups。
5. **Metadata**：生成与现有一致的列（含 path、split、label、d_data、seq_len 等），并保证 `d_data` 与真实 `x.shape` 一致。
6. **数据加载**：实现或改写 `BrainTreebankWrapper` / `BrainTreebankDataset`，使返回的 `x` 为 **`(B, segment_len, n_channels)`**，并传入对应 `subject_sessions` 以匹配空间分组。

按上述准备“必要数据 + 维度”，即可用自建 sEEG 替换 BaRISTA 的 Brain Treebank 数据。
