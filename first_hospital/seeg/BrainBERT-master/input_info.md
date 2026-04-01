# BrainBERT 自建数据输入说明

本文档说明使用 BrainBERT 架构在**自有数据**上做预训练或测试时，需要准备的数据格式、采样率、每个 patch 的时长，以及进入模型前的数据维度。

---

## 1. 数据前提与预处理

- **信号类型**：颅内电极数据（iEEG/SEEG），代码中默认假定已做 **Laplacian 重参考**。
- **通道**：每个样本为**单通道**（单电极）一段连续信号；多通道需按通道/电极分别生成样本。
- **推荐预处理流程（与论文描述一致）**：
  - 对原始信号进行 **0.1 Hz 高通滤波**。
  - 去除 **60 Hz 工频噪声及其谐波**（notch/线噪回归均可，保持一致即可）。
  - 对每个电极做 **Laplacian 重参考**：用同一电极杆（shaft）上相邻两根电极的**均值**作为参考并相减，从而降低电极间交叉相关。
  - **仅纳入可进行 Laplacian 重参考的电极**（即该电极在同一 shaft 上存在相邻电极）。
  - 对所有电极信号进行**人工目视检查**，剔除出现明显损坏/饱和/脱落/严重伪迹等“明显腐败”的记录。

---

## 2. 采样率

- **固定为 2048 Hz**。  
  在 `preprocessors/stft.py`、`data/electrode_subject_data.py`、`data/timestamped_subject_data.py`、`util/mask_utils.py` 等处均为 2048，未通过配置修改。
- 若你的数据采样率不是 2048 Hz，需要先**重采样到 2048 Hz** 再写入下面所述的 `.npy` 或先做时频并缓存。

---

## 3. 每个 patch 的时长与样本长度

- **时长**：你当前设定为 **5 s**（对应配置中的 `duration: 5.0`）。
- **每段样本点数**（每段波形长度）：
  ```text
  n_samples = 2048 × duration
  ```
  在 `duration=5.0` 时：`n_samples = 2048 × 5 = 10240` 点。

每个 `.npy` 文件对应**一段连续波形**，形状为 `(n_samples,)`，即**一维 float32 数组**。

---

## 4. 目录与清单格式（预训练用）

### 4.1 目录结构

```text
/path/to/pretrain_data
├── manifests
│   └── manifest.tsv      # 清单文件（见下）
├── <subject_id>           # 例如 subject_01
│   └── <trial_id>        # 例如 trial001
│       ├── 0.npy
│       ├── 1.npy
│       └── ...
```

- 每个 `.npy`：**单通道、单段**波形，形状 `(n_samples,)`，`n_samples = 2048 * duration`。
- `subject_id`、`trial_id` 仅用于组织路径，清单里写的是**相对路径**。

### 4.2 manifest.tsv 格式

- **第 1 行**：数据根目录的**绝对路径**（一行一个路径），与上面 `/path/to/pretrain_data` 对应。
- **第 2 行起**：每行两列，制表符 `\t` 分隔：
  - 第 1 列：该样本的**相对路径**（相对上述根目录），如 `subject_01/trial001/0.npy`
  - 第 2 列：该样本的**长度**（点数），即 `n_samples`，如 `10240`

示例：

```tsv
/path/to/pretrain_data
subject_01/trial001/0.npy	10240
subject_01/trial001/1.npy	10240
subject_02/trial001/0.npy	10240
```

---

## 5. 进入模型前的数据维度（时频表示）

训练时，代码会先对**原始波形**做 **STFT**（或使用你事先算好的时频缓存），再在时频上做 mask 与预测。因此“最后准备数据的维度”分两种用法说明。

### 5.1 方式一：原始波形 .npy（推荐先按此准备）

- **你只需准备**：上述 `(n_samples,)` 的 float32 波形 `.npy` 和 `manifest.tsv`。
- **STFT 在 DataLoader 里在线计算**，使用默认预处理器配置（如 `conf/preprocessor/stft.yaml`）：
  - `nperseg: 400`, `noverlap: 350` → 帧移 hop = 50
  - `freq_channel_cutoff: 40` → 只保留前 40 个频率 bin（单边谱）
  - `normalizing: zscore` 等按配置

**时频矩阵维度**（由代码自动得到）：

- **时间维**（帧数）：
  ```text
  n_time = (n_samples - nperseg) // (nperseg - noverlap) + 1
         = (n_samples - 400) // 50 + 1
  ```
  例如：`n_samples = 10240`（5 s）→ `n_time = 197`。
- **频率维**：`n_freq = 40`（与 `freq_channel_cutoff` 一致）。

因此，**送入模型的时频张量形状为**：

```text
(n_time, 40)
```

即 **时间步 × 40 维频率**。batch 后为 `(B, n_time, 40)`，其中 `n_time` 因各段长度可能不同会做 padding，对应 `cfg.input_dim = 40`。

### 5.2 方式二：预计算时频缓存

若使用**已预计算好的时频**（例如用 Superlet 等），则：

- 每个缓存 `.npy` 的形状应为 **`(n_time, input_dim)`**。
- `input_dim` 必须与模型配置中的 `input_dim` 一致（默认 STFT 为 **40**）。
- 需要另外配置 `cached_features` 及缓存目录下的 `config.yaml`、缓存用 `manifest.tsv`（格式与代码中 `initialize_cached_features` 约定一致）。

此时“最后准备数据的维度”就是：**每个样本 `(n_time, input_dim)`**，与上面时间/频率含义一致，只是 `input_dim` 可能因预处理器不同而非 40。

---

## 6. 小结表

| 项目           | 要求或公式 |
|----------------|------------|
| 采样率         | **2048 Hz**（需事先重采样） |
| 每段时长       | **5.0 s**（即 `duration: 5.0`） |
| 每段样本点数   | `n_samples = 2048 × duration` |
| 单段波形 .npy  | 形状 `(n_samples,)`，float32 |
| 清单 manifest  | 第 1 行：根目录绝对路径；其余行：`相对路径\t长度` |
| 模型输入维度   | 时频：`(n_time, 40)`，其中 `n_time = (n_samples - 400) // 50 + 1` |
| 模型 cfg       | `input_dim: 40`（与 STFT 的 40 频 bin 一致） |

按上述准备数据后，在训练命令中通过 `+data.data=/path/to/pretrain_data/manifests` 指向清单所在目录（即 `manifest.tsv` 的父目录）即可。
