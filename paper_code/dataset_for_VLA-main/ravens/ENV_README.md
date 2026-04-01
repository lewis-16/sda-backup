# Ravens 运行环境配置

本仓库使用 **Python 3.8** 和 **TensorFlow 2.3**，建议用 Conda 创建独立环境。

## 一键配置（推荐）

在 `ravens` 目录下执行：

```bash
cd /path/to/dataset_for_VLA-main/ravens
bash setup_env.sh
```

默认会创建名为 `ravens` 的 conda 环境。若想用其他名字：

```bash
bash setup_env.sh 你的环境名
```

## 手动配置

```bash
conda create -n ravens python=3.8 -y
conda activate ravens
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 使用环境

```bash
conda activate ravens
cd /path/to/ravens
python ravens/demos.py --assets_root=./ravens/environments/assets --data_dir=./data --task=towers-of-hanoi --n=1
```

## 依赖说明

| 依赖 | 版本约束 | 说明 |
|------|----------|------|
| Python | 3.8 | TensorFlow 2.3 仅支持 3.5–3.8 |
| numpy | >=1.16.0,<1.19.0 | 满足 TF 2.3 与 pybullet 兼容 |
| tensorflow | 2.3.0 | 项目代码基于此版本 |
| pybullet | >=3.0.4,<3.2 | 3.2+ 与 numpy 1.18 存在 C-API 不兼容 |
| protobuf | >=3.9.2,<4 | 4.x 与 TF 2.3 不兼容 |

若使用 **Python 3.10+**，需升级 TensorFlow 至 2.10+ 并自行调整代码中的 API 变更。
