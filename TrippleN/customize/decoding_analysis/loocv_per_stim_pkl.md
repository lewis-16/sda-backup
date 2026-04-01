# `loocv_per_stim_original_image_nn_fc6_100d.pkl` 说明

本文件由 `scripts/08_decoding_analysis_overview.ipynb` 末尾导出 cell 生成，依赖 `decoding_results_loocv.pkl` 中的 `alexnet_fc6` 与 `stimuli` 目录下按文件名排序后的前 1000 张 `.bmp`。

## 字段一览

| 字段 | 类型 / 形状 | 说明 |
|------|----------------|------|
| `model` | `str` | 固定为 `alexnet_fc6`。 |
| `space` | `str` | 向量所在空间：`target_reduced` 的 100 维（对原始 fc6 做 PCA 后）。 |
| `neighbor_rule` | `str` | 对每个刺激下标 `i`，在 `j ∈ {0,…,999}` 上最小化欧氏距离 `‖pred[i] − target[j]‖`；`pred` 为 LOOCV 预测，`target` 为 `target_reduced`。 |
| `image_filenames` | `list[str]`，长度 1000 | 与上述 bmp 文件名顺序一致。 |
| `nn_stimulus_idx` | `int32`，`(1000,)` | 刺激 `i` 的最近邻下标 `j`。 |
| `original_images_uint8` | `uint8`，`(1000, H, W, 3)` | 第 `i` 条为原刺激图像（与 `i` 对齐）。 |
| `nn_alexnet_fc6_100d` | `float32`，`(1000, 100)` | 最近邻 `j` 的 100 维 PCA fc6，即 `target_reduced[j]`。 |
| `decode_correct` | `bool`，`(1000,)` | `nn_stimulus_idx[i] == i` 时为真，表示在该 NN 规则下是否匹配到自身。 |
| `alexnet_fc6_pca_mean` | `float32`，`(D,)` | sklearn `PCA.mean_`，`D` 为原始 fc6 维度（常见为 4096）。 |
| `alexnet_fc6_pca_components` | `float32`，`(100, D)` | sklearn `PCA.components_`。 |

## PCA 逆变换（`whiten=False`）

与 `decoding_model_loocv.py` 中默认 PCA 一致时，由 100 维近似还原 fc6：

\[
\hat{x}_{\mathrm{fc6}} = x_{100} \cdot \texttt{alexnet\_fc6\_pca\_components} + \texttt{alexnet\_fc6\_pca\_mean}
\]

Python 示例：

```python
import numpy as np
x_100 = ...  # (..., 100)
x_fc6_hat = x_100 @ data["alexnet_fc6_pca_components"] + data["alexnet_fc6_pca_mean"]
```

## 体积与注意

- `original_images_uint8` 约占用一百多 MB 量级，整体 pkl 较大。
- 若重新运行 LOOCV 或更换 `MODEL_DIM` / PCA 设置，需重新导出本 pkl，且 `pca_mean` / `pca_components` 会与新版 `decoding_results_loocv.pkl` 一致。
- 最近邻与 `decode_correct` 均基于 **100 维 `target_reduced` 空间** 下的欧氏距离，不是原始 fc6 空间。
