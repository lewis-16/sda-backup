# 时间分辨率视频生成器

这个工具用于生成探索生物视觉在时间维度上分辨率的测试视频。

## 功能特点

- **多时间间隔测试**: 支持1ms, 2ms, 4ms, 10ms, 20ms, 33ms, 100ms等不同呈现时间间隔
- **高帧率视频**: 生成1000fps的高帧率视频，确保时间精度
- **随机图片选择**: 每轮随机选择30张自然场景图片
- **精确时间控制**: 每张图片后1秒灰色间隔，每轮后5秒灰色间隔
- **详细记录**: 生成包含刺激时间和图像ID的详细DataFrame

## 文件结构

```
dynamic/
├── temporal_resolution_video_generator.py  # 主生成器类
├── generate_temporal_video.py             # 使用示例
├── test_temporal_resolution.py            # 测试脚本
└── README_temporal_resolution.md         # 说明文档
```

## 使用方法

### 1. 基本使用

```python
from temporal_resolution_video_generator import TemporalResolutionVideoGenerator

# 创建生成器
generator = TemporalResolutionVideoGenerator(
    nature_scene_path="/path/to/nature_scene",
    output_path="/path/to/output.mp4"
)

# 生成视频
video_path, stimulus_df = generator.generate_video()

# 保存刺激记录
generator.save_stimulus_log(stimulus_df)
```

### 2. 运行示例脚本

```bash
cd /media/ubuntu/sda/visual_stimuli_pattern/dynamic
python generate_temporal_video.py
```

## 输出文件

### 视频文件
- **格式**: MP4
- **帧率**: 1000 fps
- **分辨率**: 根据输入图片自动调整
- **时长**: 约250秒（包含所有时间间隔的测试）

### 刺激记录CSV文件
包含以下列：
- `time_interval_ms`: 时间间隔（毫秒）
- `frame_number`: 帧号
- `time_ms`: 时间（毫秒）
- `image_id`: 图像文件名
- `stimulus_type`: 刺激类型（image/gray/round_gray）
- `round`: 轮次编号

## 实验设计

### 时间间隔设置
- 1ms: 极短时间间隔，测试最高时间分辨率
- 2ms: 短时间间隔
- 4ms: 较短时间间隔
- 10ms: 中等时间间隔
- 20ms: 较长时间间隔
- 33ms: 长时间间隔（约30fps）
- 100ms: 很长时间间隔

### 刺激序列
每轮包含：
1. 30张随机选择的自然场景图片
2. 每张图片后1秒灰色间隔
3. 每轮结束后5秒灰色间隔

## 技术参数

- **图片格式**: TIFF
- **图片数量**: 100张（natural_scene_0.tiff 到 natural_scene_99.tiff）
- **每轮图片数**: 30张（随机选择）
- **视频编码**: MP4V
- **颜色空间**: RGB → BGR（OpenCV格式）

## 依赖库

```bash
pip install opencv-python pillow pandas numpy matplotlib
```

注意：需要NumPy < 2.0版本以避免兼容性问题。

## 应用场景

这个工具特别适用于：
- 生物视觉系统的时间分辨率研究
- 视觉感知的时间敏感性测试
- 神经科学实验中的视觉刺激设计
- 时间维度上的视觉阈值测量

## 注意事项

1. 确保nature_scene文件夹中有足够的图片文件
2. 生成的视频文件较大（约500MB），需要足够的存储空间
3. 播放1000fps视频需要支持高帧率的播放器
4. 建议在专业显示设备上播放以确保时间精度
