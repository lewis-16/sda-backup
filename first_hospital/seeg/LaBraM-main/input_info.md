## LaBraM 输入与预处理说明（面向自建数据）

本说明基于 `seeg/LaBraM-main` 代码的真实实现（`engine_for_pretraining.py`、`engine_for_finetuning.py`、`modeling_finetune.py`、`modeling_pretrain.py`、`modeling_vqnsp.py`、`utils.py`）。

### 1. 模型需要的输入张量形状

LaBraM 的主干 Transformer 接收四维 EEG patch 张量：

- **输入形状**：`[B, N, A, T]`
  - `B`：batch size
  - `N`：通道数（电极数）
  - `A`：时间 patch 数（time window，以 1 秒为单位的 patch 个数）
  - `T`：每个 patch 的采样点数，代码里固定为 **200**

代码里经常先用二维形式喂入（把 `A` 和 `T` 展平）：

- **展平输入形状**：`[B, N, A*T]`
- 然后在训练/评估时做：
  - `rearrange(samples, 'B N (A T) -> B N A T', T=200)`

因此你只要保证每条样本的时间长度是 **200 的整数倍**，即可自动得到 `A = (样本长度 / 200)`。

#### A 的范围建议

模型里 `time_embed` 的长度为 **16**（`modeling_finetune.py` 与 `modeling_pretrain.py` 中 `self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))`），因此建议：

- **推荐**：`A <= 16`
- 典型例子：
  - 1 秒：`A=1`，输入 `[B, N, 1, 200]`
  - 4 秒：`A=4`，输入 `[B, N, 4, 200]`
  - 10 秒：`A=10`，输入 `[B, N, 10, 200]`

### 2. 通道顺序与 `input_chans`（非常关键）

LaBraM 的“空间位置编码”不是简单按 `0..N-1`，而是把通道名映射到 `utils.py` 里的 `standard_1020` 列表索引（并为 CLS token 预留一个位置）。

- 代码路径：`utils.get_input_chans(ch_names)`
- 映射规则（简化描述）：
  - `input_chans[0] = 0`（CLS）
  - 对于每个通道名 `ch_name`：取其在 `standard_1020` 中的位置 `idx`，使用 `idx + 1`

这要求你：

- 提供 **通道名列表** `ch_names`（顺序必须与数据矩阵的通道维一致）
- 并且通道名必须能在 `standard_1020` 中找到

在作者提供的 TUAB/TUEV 例子里，通道名会进一步被处理成类似 `FP1`、`F7`、`T3` 这样的形式（见 `run_class_finetuning.py` 中对 `ch_names` 的处理）。

如果你的数据是 sEEG（非 10-20 标准命名），你有两种做法：

- **做法 A（推荐）**：把你的通道映射/归并到一个固定的标准集合（例如你选定的 montage 或自定义标准列表），并在 `standard_1020` 里扩展对应名字（需要你改 `utils.py` 的 `standard_1020`）。
- **做法 B**：不使用 `input_chans`（传 `None`），但这会使通道位置编码退化为“共享同一套 pos embed”，通常会明显影响效果，尤其是跨被试/跨数据集。

### 3. 采样率、单位与数值缩放

代码假设的关键物理量约定：

- **采样率**：**200 Hz**
- **单位**：**µV**
- **训练时的缩放**：在训练/评估循环中会对输入做 `x = x / 100`
  - 预训练：`engine_for_pretraining.py` 第 87 行附近
  - 微调：`engine_for_finetuning.py` 第 64 行附近

因此你应该：

- 预处理后把数据统一到 **µV**
- 保持数值量级合理（例如几十到几百 µV），然后在喂模型前按代码逻辑 `/ 100`

### 4. 官方推荐的预处理流程（与 README 对齐）

作者在 `README.md` 明确给出的预处理设定如下（用于把原始 `.cnt/.edf/.bdf` 等转成训练用数据）：

- 去除无关通道（例如眼电、心电、呼吸等，或数据集特有的非 EEG 通道）
- **带通滤波**：0.1–75 Hz
- **陷波**：50 Hz
- **重采样**：200 Hz
- 单位设置为 **µV**

在提供的示例脚本里也能看到同样的处理：

- `dataset_maker/make_TUAB.py`、`dataset_maker/make_TUEV.py`：
  - `raw.filter(l_freq=0.1, h_freq=75.0)`
  - `raw.notch_filter(50.0)`
  - `raw.resample(200, ...)`
  - `raw.get_data(units='uV')`

### 5. 切窗与 patch 化（你需要决定的部分）

模型内部 patch 长度固定为 `T=200`，也就是 1 秒（在 200 Hz 下）。

你需要决定每条样本包含多少秒，也就是 `A`：

- **预训练（无监督）**：
  - 代码中会按不同通道数选择不同 `A`，目标是让序列长度 `N*A` 大约在 256 左右（见 `run_labram_pretraining.py` 注释）。
  - 例如 `N=64` 时可选 `A=4`（序列长度 256），窗口长度 `A*T = 800` 点（4 秒）。
  - `utils.build_pretraining_dataset(...)` 里窗口长度是 `window_size * 200`，其中 `window_size` 就是你传入的 `A`。

- **下游微调（分类等）**：
  - 只要样本长度是 `200` 的整数倍即可（例如 5 秒=1000 点，10 秒=2000 点）。
  - 示例：TUAB 在制作数据时按 10 秒切片（2000 点），TUEV 的事件片段是 5 秒（1000 点）。

### 6. 你最终应准备的数据格式（最小可用约定）

无论你是做预训练还是微调，最小可用的数据约定如下：

- **每条样本**：一个浮点矩阵 `X`
  - 形状：`[N, L]`
  - `N`：通道数
  - `L`：时间点数，满足 `L % 200 == 0`（因此 `A = L/200`）
  - 单位：µV
- **通道名列表**：`ch_names`（长度为 `N`，顺序与 `X` 的通道维一致）
  - 需要能映射到 `utils.standard_1020`

喂入模型时的形状会变成：

- `X_batch`: `[B, N, L]`
- 模型内部/训练脚本重排后：`[B, N, A, 200]`，其中 `A = L/200`

### 7. 额外：Neural Tokenizer（VQ-NSP）对输入的要求

预训练 LaBraM 之前会训练一个 tokenizer（`modeling_vqnsp.py`）把 EEG patch 编成离散 code：

- tokenizer 输入（在 `VQNSP.forward` 里）也是从 `[B, N, A*T]` 先重排成 `[B, N, A, 200]`
- 之后对每个 1 秒 patch 做 FFT，分别用幅度与相位做重建损失
- tokenizer 产生的离散码用于 LaBraM 的 masked code prediction

对你来说，这意味着：

- tokenizer 与 LaBraM 对 **采样率=200Hz、patch=200点** 的约定是一致的
- 预处理与切窗策略必须前后一致，否则 tokenizer 学到的 code 分布会漂移

