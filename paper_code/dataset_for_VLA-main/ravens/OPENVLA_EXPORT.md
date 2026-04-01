# OpenVLA 微调数据：采集与格式说明

## 1. 训练指令 (Prompt)

每条轨迹的指令为英文：

```
grasp {xxx} and put it into the basket
```

其中 `xxx` 为当前局随机选中的物体英文名之一：`apple`, `pear`, `banana`, `elephant`, `deer`, `rhino`。  
指令由任务在 `reset` 时写入，并在 `reward()` 的 `info['instruction']` 中随步返回。

## 2. 采集专家轨迹（数据生成脚本用法）

在 `ravens` 目录下执行，使用与预览相同的物体配置（缩放、旋转、随机 xy）自动采集专家轨迹。

**基本用法：**

```bash
cd /path/to/ravens

python scripts/collect_put_basket_demos.py \
  --obj_dir=/path/to/your/OBJ3D \
  --data_dir=./data \
  --mode=train \
  --n_per_object=200
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--obj_dir` | 见脚本 | 放 6 个物体 mesh（.glb/.obj）的目录，需包含 apple, pear, banana, elephant, deer, rhino |
| `--data_dir` | `.` | 数据集根目录，会在此下创建 `put-object-in-basket-{mode}/` |
| `--mode` | `train` | `train` 或 `test` |
| `--n_per_object` | `200` | 每个物体目标成功条数（6 个物体共 6×n_per_object 条） |
| `--assets_root` | 自动 | Ravens 资源根目录，一般不需改 |
| `--disp` | `False` | 设为 `True` 可打开 PyBullet 窗口看仿真 |
| `--video_dir` | 无 | 若指定目录，每个成功 episode 会额外导出一段 MP4 视频（正面视角），便于连贯查看任务 |

**示例：采集训练集并同时导出 MP4 预览：**

```bash
python scripts/collect_put_basket_demos.py \
  --obj_dir=/media/ubuntu/sda/duan/OBJ3D \
  --data_dir=./data \
  --mode=train \
  --n_per_object=200 \
  --video_dir=./data/put-basket-videos
```

**示例：只采集测试集：**

```bash
python scripts/collect_put_basket_demos.py \
  --obj_dir=/path/to/OBJ3D \
  --data_dir=./data \
  --mode=test \
  --n_per_object=50
```

- 数据保存目录：`{data_dir}/put-object-in-basket-train/` 与 `put-object-in-basket-test/`。
- 仅当回合成功（reward > 0.99）且该物体未满 `n_per_object` 条时写入；每条轨迹的每一步 `info` 中都带有 `instruction`。
- 若指定 `--video_dir`，成功写入的每条轨迹会对应一个 `episode_XXXXXX.mp4`（与数据集中的 episode 编号一致），可用播放器连贯观看该局从初始到结束的画面。

## 3. Ravens 原始数据格式

- **路径**：`{data_dir}/put-object-in-basket-{mode}/`
- **每条 episode** 对应一组同名 pkl（`color/`, `depth/`, `action/`, `reward/`, `info/`）：
  - `color`: `(T+1, H, W, 3)` uint8 图像
  - `depth`: `(T+1, H, W)` float32
  - `action`: 长度为 T 的 list，每步为 `dict`: `{'pose0': (pick_pos, pick_quat), 'pose1': (place_pos, place_quat)}`
  - `info`: 长度为 T+1 的 list，每步为 `dict`，其中 `info[i]['instruction']` 为该轨迹的英文指令（整条轨迹相同）

加载示例：

```python
from ravens.dataset import Dataset
data = Dataset('data/put-object-in-basket-train')
episode, seed = data.load(0, images=True)
instruction = episode[0][3].get('instruction')
```

## 4. 转成 OpenVLA 训练格式（两步）

### 第一步：Ravens → 中间导出目录

用 `export_for_openvla.py` 把 Ravens 的 pkl 数据导出成「每 episode 一个文件夹」：

```bash
cd /path/to/ravens

python scripts/export_for_openvla.py \
  --data_dir=/path/to/data/put-object-in-basket-train \
  --out_dir=/path/to/data/put-basket-openvla-train
```

- `--data_dir`: 采集结果目录（含 `color/`, `action/`, `info/` 等 pkl）
- `--out_dir`: 输出目录，不写则默认为 `data_dir-openvla`
- `--max_episodes`: 可选，只导出前 N 条

导出结构：

- `out_dir/episode_000000/`
  - `instruction.txt`: 一行英文指令
  - `image_0000.npy`, `image_0001.npy`, ...: 每步观测图像 (H, W, 3)，正面视角
  - `actions.pkl`: list of dict，每步 `{'pose0': (pos, quat), 'pose1': (pos, quat)}`（Ravens pick/place）

### 第二步：中间目录 → RLDS TFRecord

用 `export_openvla_rlds.py` 把上一步的 `out_dir` 转成 RLDS 风格的 TFRecord，供 OpenVLA 读取：

```bash
python scripts/export_openvla_rlds.py \
  --export_dir=/path/to/data/put-basket-openvla-train \
  --out_dir=/path/to/data/put-basket-rlds-train \
  --split=train
```

- `--export_dir`: 第一步的 `--out_dir`
- `--out_dir`: 输出目录，会生成 `.tfrecord` 和 `dataset_info.json`
- `--split`: `train` 或 `test`，会写进文件名
- `--max_episodes`: 可选，只转换前 N 条

生成文件：

- `put_basket_train-00000-of-00003.tfrecord` 等（每条约 500 个 episode，每条 episode 一个 `SequenceExample`：`context.language_instruction`，`feature_lists.image` / `feature_lists.action`）
- `dataset_info.json`: 简单元信息（name, image_size, action_dim）

在 OpenVLA 里需要：① 把该数据集路径配到 dataloader；② 在 config 里写好 `image_obs_keys`（例如 `primary: "image"`）、`action_encoding`（Ravens 是 EEF pose，6 维 pick_pos + place_pos 已写在 action 里）、语言 key 与 RLDS 的 `language_instruction` 对应。若 OpenVLA 官方用的是标准 RLDS 的 step 结构，你可能要再套一层 step 解析或参考其 conversion 脚本把本脚本的 `SequenceExample` 转成其期望的 step 列表。

## 5. OpenVLA 微调所需格式 (RLDS)

OpenVLA 官方微调使用 **RLDS (Robot Learning Dataset Standard)**，通常为 TFRecord：

- **每条 step** 需包含：图像（如 `image` / `wrist_image`）、机器人状态（如 `state`）、**动作**（如 `action`）、**语言指令**（如 `language_instruction`）。
- **动作**：多为连续控制（如 7 维关节速度 + 夹爪 + 终止），需根据你的机器人或仿真将 Ravens 的 `pose0`/`pose1` 转换为 OpenVLA 的 action 空间（或使用 OpenVLA 提供的转换脚本）。
- **图像**：Ravens 为俯视/固定机位；若 OpenVLA 训练用的是 wrist/多视角，需在转 RLDS 时做 key 映射或注明 observation 来源。

将「导出目录」转为 RLDS 的常见做法：

1. 使用 [RLDS Dataset Builder](https://github.com/google-research/rlds) 或 OpenVLA 仓库中的 conversion 脚本，读入 `out_dir` 下的 episode 文件夹，写出 `dataset_info.json`、`features.json` 及 `.tfrecord`。
2. 在 OpenVLA 的 dataset 配置中注册该 RLDS 路径及使用的 keys（`image_obs_keys`、`state_obs_keys`、`action_encoding`、语言 key 等）。

具体注册方式、action 编码和图像 key 名称以 OpenVLA 当前文档为准：  
<https://github.com/openvla/openvla>

## 6. 小结

| 步骤 | 说明 |
|------|------|
| 指令 | 任务内固定为 `grasp {apple\|pear\|banana\|elephant\|deer\|rhino} and put it into the basket` |
| 采集 | `scripts/collect_put_basket_demos.py` → `data/put-object-in-basket-{train\|test}/` |
| 导出 | `scripts/export_for_openvla.py` → 每 episode 一目录，含 `instruction.txt`、图像、`actions.pkl` |
| 微调 | 将导出目录转为 RLDS TFRecord，并在 OpenVLA 中注册 dataset 与 action/obs 映射 |
