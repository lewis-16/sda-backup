# SEEG 数据整理与预处理说明（Brant 用）

本文档说明如何将 `不同年龄段的SEEG原始数据2026-2-1` 下的 EDF 记录整理、标注并切分为 Brant 可用的滑动窗口数据。

---

## 1. 原始数据来源

- **路径**：`/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1`
- **结构**：每个患者一个目录，目录内为若干 EDF 文件（如 `于涵 男 14y/CZ4441AR.edf`）
- **标注**：EDF 内 TAL 注解由 MNE 解析为 `raw.annotations`（含 onset、duration、description）

---

## 2. 放电期与标签定义

### 2.1 发作期（ictal）

- **规则**：每一个 **「◆发作」到「end」** 的区间均视为一次发作期。
- **说明**：
  - 「◆发作」包含变体：`◆发作`、`◆发作1`、`◆发作？` 等以 `◆发作` 开头的描述。
  - 「end」包含：`end`、`END`、`发作`（作为结束标记时）。
- **区间**：从该次「◆发作」的 onset 起，到**随后第一个**「end」类事件的 onset 止，为该次发作期的时间区间 `[t_start, t_end]`。

### 2.2 间期放电（interictal，亦标为发作）

- **规则**：每一个**间期放电**事件单独视为一次“发作”区间。
- **认定方式**：描述为**纯英文大写字母组合**的注解（如 `AD`、`AH`、`ADFG`、`ADH` 等），每个事件对应其注解的 `[onset, onset+duration]` 区间。
- **说明**：小写或混合（如 `a`、`ad`、`abc`）可按需一并视为间期放电，与实现一致即可。

### 2.3 刺激段（整段排除）

- **规则**：**刺激记录整段不参与切窗与标注**。
- **认定方式**：描述中包含 `Stim Start` / `Stim Stop`（或同义）的注解，每对 `Stim Start` → `Stim Stop` 构成一个刺激区间。
- **处理**：所有**与刺激区间有重叠**的滑动窗口一律**丢弃**，不写入 `data.npy` / `label.npy`。

---

## 3. 通道排除规则

预处理时**删除**以下两类通道，仅保留 SEEG 电极通道：

### 3.1 类型一：DC 通道

- **示例**：`POL DC01`、`POL DC02`、`POL DC03`、`POL DC04`
- **规则**：`POL DC` 后跟数字的通道（如 DC01、DC02、DC03、DC04）

### 3.2 类型二：心电 / 辅助 / 参考通道

- **示例**：
  - `POL ECG-0`、`POL ECG-1`
  - `POL 0`、`POL 0V`
  - `POL X1-0`、`POL -1-0`、`POL X2-0`、`POL -2-0`、…、`POL X6-0`、`POL -6-0`
  - `POL X1-1`、`POL -1-1`、…、`POL X6-1`、`POL -6-1`
  - `POL L7`、`POL L8`、`POL L9`、`POL L10`
  - `POL X7`、`POL -7`、`POL X8`、`POL -8`
- **规则**：
  - `POL ECG` 开头的通道
  - `POL 0`、`POL 0V`
  - `POL X数字-数字`、`POL -数字-数字` 格式
  - `POL L` 后跟数字
  - `POL X数字`、`POL -数字` 格式（非 A–H 电极名）

---

## 4. 滑动窗口与输出形状

### 4.1 参数

| 参数       | 取值   | 说明 |
|------------|--------|------|
| 窗口长度   | 15×6 s | 15 段，每段 6 s，共 90 s |
| 每段点数   | 1500   | 6 s × 250 Hz（若降采样到 250 Hz） |
| 滑动步长   | 3 s    | 相邻窗口起点间隔 3 s |
| 采样率     | 250 Hz | 与论文一致；若原始非 250 Hz 需先重采样 |

### 4.2 单窗口数据形状

- 每个窗口：**（n_channel, 15, 1500）**
  - 即 **（n_channel, 15×1500）** 的物理含义：15 段 × 每段 1500 点。
- 保存时按 Brant 习惯为 4 维：`(n_channel, n_windows, 15, 1500)`。

### 4.3 窗口标签（二分类）

- **标签 1（放电）**：该 90 s 窗口与**任意一个**放电区间（2.1 或 2.2）**有重叠**。
- **标签 0（非放电）**：该窗口与所有放电区间均不重叠，且未被 2.3 排除。

重叠判定：窗口 `[t, t+90]` 与区间 `[a, b]` 重叠 ⟺ `t < b` 且 `t+90 > a`。

---

## 5. 输出格式（Brant 使用）

### 5.1 目录与文件

每个**患者+EDF**或按患者汇总对应一个输出目录，目录内：

| 文件        | 形状 | 说明 |
|-------------|------|------|
| `data.npy`  | (n_channel, n_windows, 15, 1500) | float32 |
| `power.npy` | (n_channel, n_windows, 15, 8)    | 频谱功率，8 频段 |
| `label.npy` | (1, n_windows, 15) 或 (n_windows,) | 二分类 0/1 |

Brant 的 `BoardDataset` 期望 `data` 为 `(ch_num, board_num, seq_len, seg_len)`，此处 **seq_len=15, seg_len=1500**，与 Brant 设置一致。

### 5.2 采样率与功率

- 计算 `power.npy` 时，若数据已重采样到 250 Hz，则 `compute_power(..., fs=250)`；若保持 256 Hz 则 `fs=256`（与当前 Brant 代码一致）。

---

## 6. 流程小结

1. 遍历 `不同年龄段的SEEG原始数据2026-2-1` 下各患者目录及 EDF。
2. 用 MNE 读取 EDF，按 **coordination.md** 与 **通道排除规则**（第 3 节）筛选通道，保留 SEEG 电极通道。
3. 解析 `raw.annotations`（onset, duration, description）。
4. 构建**放电区间**：所有「◆发作」→「end」区间；所有间期放电（如 AD、AH 等）的 `[onset, onset+duration]`。
5. 构建**刺激区间**：所有 `Stim Start` → `Stim Stop`，用于排除。
6. 对每条记录在**非刺激**时间段上做**滑动窗口**（步长 3 s，窗长 90 s），重采样到 250 Hz 后每窗 (n_channel, 15, 1500)。
7. 若窗口与任一放电区间重叠则标 1，否则标 0；与刺激重叠的窗口不输出。
8. 写入 `data.npy`、`power.npy`、`label.npy`。

---

## 7. 脚本与运行

- **脚本**：`data_wrangling/seeg_to_brant_sliding.py`
- **运行示例**：
  ```bash
  cd /media/ubuntu/sda/first_hospital/data_wrangling
  python seeg_to_brant_sliding.py --base "/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1" --out ./brant_seeg_out --fs 250 --power_fs 256 --step 3
  ```
- **输出**：每个 EDF 对应一个子目录 `{out}/{患者名}/{EDF文件名无后缀}/`，内含 `data.npy`、`power.npy`、`label.npy`。Brant 的 `load_data` 可传入这些子目录的列表作为“患者”数据源。

---

## 8. 与 Brant 的对应关系

- **Brant 癫痫检测任务**（Task 1）输入为 `(B, C, S, L)`，**S=15, L=1500**。
- 本流程产出 **S=15, L=1500**，与 Brant 一致。
