## 任务：为 single-unit 图像重建生成与大脑信号对齐的结构化文本描述

### 目标
针对一组训练图像及其对应的 SUA 信号，通过自动搜索最优描述关键词，为每张图像生成结构化的文本描述。这些描述需同时满足：
- **与 fMRI 信号高度对齐**（描述内容反映大脑编码的信息）
- **能引导var高质量重建原图**（描述包含足够视觉细节）

生成的描述将作为后续训练 fMRI 编码器的监督标签。

### 输入
- 训练图像集合 \( \mathcal{Y}^{\text{train}} \)（每张图像 \( \mathbf{Y}_i \)）
- 对应的 fMRI 信号集合 \( \mathcal{X}^{\text{train}} \)（每个信号 \( \mathbf{x}_i \) 为预处理后的向量）
- 图像原始标题（caption）

### 当前 SUA 数据路径
- 图像: `/media/ubuntu/sda/TrippleN/stimuli` （1000 张 .bmp，用作 \(\mathcal{Y}^{\text{train}}\)）
- SUA 信号: `/media/ubuntu/sda/TrippleN/customize/neuron_responses_1000.npy` （形状约为 `(n_units, 1000)`，按列对应图像顺序）
- 图像 caption: `/media/ubuntu/sda/TrippleN/customize/coco_captions_1000x5.pkl` （每张图 5 条 COCO caption）
- 打包 zip: `/media/ubuntu/sda/TrippleN/customize/sua_search_data.zip` （包含上述三类数据，便于迁移和复现实验）

### 输出
- 每张图像 \( \mathbf{Y}_i \) 的一个结构化文本描述 \( D_i^{a^*} \)，格式如下（以两物体为例）：
  ```
  Background color style: [背景描述]
  ## [物体1名称] [位置标签：left/right/top/bottom]
  1. [物体1的第一句描述，包含属性或关系]
  2. [物体1的第二句描述]
  ## [物体2名称] [位置标签]
  1. [物体2的第一句描述]
  2. [物体2的第二句描述]
  ```

### 步骤概述
整个流程分为两个阶段：**关键词搜索阶段**和**最终描述生成阶段**。

---

#### 阶段一：属性-关系关键词搜索

**目标**：从初始关键词集合出发，通过迭代优化找到最优关键词 \( a^* \)，使得基于该关键词生成的描述在 CKA 约束下取得最佳重建得分。

**初始化**：
- 准备一组初始关键词集合 \( \mathcal{A} \)，涵盖常见的视觉关系与属性类型，如：`["spatial layout", "color attribute", "action relation", "part-whole relation", "positional relation", "functional relation"]`。可通过 LLM 为每个类型生成若干同义或相关词，扩大初始池。
- 设定超参数：搜索轮数 \( T \)、探索概率 \( \epsilon \)、每轮候选数 \( k_1 \)、新词生成数 \( k_2 \)、CKA 阈值初始值 \( \beta \)（可设为初始关键词的最小 CKA 值）。

**迭代搜索**（共 \( T \) 轮）：

1. **评估当前关键词集合**：
   - 对于每个关键词 \( a \in \mathcal{A} \)：
     - 构建提示模板，如：`"Given the image, describe the two most important objects and their relationship using {a}. Include background and absolute positions (left/right/top/bottom). Format: ..."`。
     - 对每张训练图像 \( \mathbf{Y}_i \)，调用视觉语言模型（VLM，如 GPT-4o-mini）生成结构化描述 \( D_i^a \)。
     - 使用语言模型（LM，如 Flan-T5）将所有描述编码为嵌入向量 \( \mathbf{k}_i^a \)，形成矩阵 \( \mathbf{K}^a \)。
     - 计算 fMRI 矩阵 \( \mathbf{X} \) 与 \( \mathbf{K}^a \) 的 **CKA 相似度**（采用高斯 RBF 核）。
     - 若 CKA 值 ≤ 当前阈值 \( \beta \)，则剔除该关键词（不满足对齐约束）。
     - 对于通过 CKA 检验的关键词，将生成的描述 \( D_i^a \) 输入预训练的扩散模型（如 Stable Diffusion）重建图像 \( \hat{\mathbf{Y}}_i^a \)。
     - 计算重建图像与原图的 **LPIPS 感知相似度**（越低越好），并转换为得分 \( s_i^a = 1 - \text{LPIPS}(\mathbf{Y}_i, \hat{\mathbf{Y}}_i^a) \)。
     - 对该关键词，计算所有图像上的平均得分 \( \bar{s}^a = \frac{1}{N}\sum_i s_i^a \)。

2. **排序与筛选**：
   - 将所有满足 CKA 约束的关键词按平均得分 \( \bar{s}^a \) 从高到低排序。
   - 选择前 \( k_1 \) 个关键词作为当前轮的高质量候选。

3. **生成新关键词**：
   - 以 \( \epsilon \) 概率随机从 \( \mathcal{A} \) 中抽取 \( k_1 \) 个关键词（探索），否则直接使用前 \( k_1 \) 个关键词（利用）。
   - 将这 \( k_1 \) 个关键词输入一个大语言模型（LLM），要求其生成 \( k_2 \) 个语义相近或衍生出的新关键词（例如：“请基于以下关键词生成 \( k_2 \) 个相关的新关键词：……”，输出逗号分隔列表）。
   - 将生成的新关键词加入集合 \( \mathcal{A} \)。

4. **更新阈值**：
   - 可选：根据当前所有满足 CKA 约束的关键词的最小 CKA 值，重新设定 \( \beta \)（确保只保留与 fMRI 足够对齐的关键词）。

5. **进入下一轮**。

**终止**：完成 \( T \) 轮后，从最终集合 \( \mathcal{A} \) 中选出平均得分最高的关键词作为最优关键词 \( a^* \)。

---

#### 阶段二：最终结构化描述生成

- 使用阶段一得到的最优关键词 \( a^* \) 构建提示模板（同前）。
- 对 **全部训练图像**（或整个数据集）调用 VLM，为每张图像生成符合格式的结构化描述 \( D_i^{a^*} \)。
- 这些描述即为后续训练 fMRI 编码器和微调语言模型的真实文本标签。

### 注意事项
- 搜索阶段可仅使用部分训练图像（如 667 张）以提高效率，但最终描述生成需覆盖所有图像。
- 所有评估指标（CKA、LPIPS）应基于相同的图像子集，以保证公平比较。
- 若图像原本无标题，可先用 VLM 生成简短标题，再用于描述生成。
- 最终描述需严格遵循格式，便于后续解析为物体级别信息。