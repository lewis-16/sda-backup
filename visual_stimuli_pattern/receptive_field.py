import cv2
import numpy as np
import random
from tqdm import tqdm  # 进度条，可选

# ===================== 参数配置 =====================
VIDEO_NAME = "local_sparse_noise_demo.mp4"
FPS = 10                    # 视频帧率 (帧/秒)
TOTAL_FRAMES = 6000          # 总帧数 (10fps × 30秒 = 300帧)
RESOLUTION = (1280, 720)   # 视频分辨率 (宽×高)

GRID_COLS = 4               # 大网格列数
GRID_ROWS = 3               # 大网格行数
SUBGRID_SIZE = 8            # 每个大区块内的虚拟子网格大小 (8×8)

STIM_SIZE = 30              # 小方块的像素大小
BACKGROUND_GRAY = 128       # 背景灰度 (0-255)
ON_COLOR = 255              # ON刺激 (白色)
OFF_COLOR = 0               # OFF刺激 (黑色)

# ===================== 计算布局 =====================
width, height = RESOLUTION
block_width = 240   # 每个大格子的逻辑宽度（用于子网格计算）
block_height = 240  # 每个大格子的固定高度
actual_block_width = width // GRID_COLS  # 实际大格子宽度（填满屏幕）
margin_x = 0  # 无左右边距

# 创建子网格的潜在位置列表 (填满整个大格子)
sub_positions = []
for i in range(SUBGRID_SIZE):
    for j in range(SUBGRID_SIZE):
        # 在每个大区块内均匀分布子位置（基于实际宽度）
        x = int((i + 0.5) * (actual_block_width / SUBGRID_SIZE))
        y = int((j + 0.5) * (block_height / SUBGRID_SIZE))
        sub_positions.append((x, y))

# ===================== 创建视频写入器 =====================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(VIDEO_NAME, fourcc, FPS, RESOLUTION)

print(f"正在生成局部稀疏噪声演示视频...")
print(f"分辨率: {width}x{height}, 总帧数: {TOTAL_FRAMES}")
print(f"网格: {GRID_COLS}x{GRID_ROWS}, 每帧12个随机点")

# ===================== 生成帧 =====================
for frame_idx in tqdm(range(TOTAL_FRAMES)):
    # 创建灰色背景
    frame = np.full((height, width, 3), BACKGROUND_GRAY, dtype=np.uint8)
    
    # 决定本帧是ON还是OFF (交替但随机)
    is_on_frame = random.choice([True, False]) if frame_idx % 2 == 0 else (not is_on_frame)
    color = ON_COLOR if is_on_frame else OFF_COLOR
    
    # 在每个大区块内随机选择一个位置绘制小方块
    for block_row in range(GRID_ROWS):
        for block_col in range(GRID_COLS):
            # 随机选择子位置
            sub_x, sub_y = random.choice(sub_positions)
            
            # 计算实际屏幕坐标 (填满整个宽度)
            x = block_col * actual_block_width + sub_x
            y = block_row * block_height + sub_y
            
            # 绘制小方块 (居中)
            x1 = x - STIM_SIZE // 2
            y1 = y - STIM_SIZE // 2
            x2 = x + STIM_SIZE // 2
            y2 = y + STIM_SIZE // 2
            
            # 确保不超出边界
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(width, x2); y2 = min(height, y2)
            
            # 绘制方块
            cv2.rectangle(frame, (x1, y1), (x2, y2), (color, color, color), -1)
    
    
    # 写入帧
    video.write(frame)

# ===================== 完成 =====================
video.release()
print(f"✅ 视频生成完成: {VIDEO_NAME}")
print(f"📊 参数摘要:")
print(f"   - 每帧刺激: {GRID_COLS}×{GRID_ROWS}={GRID_COLS*GRID_ROWS} 个随机点")
print(f"   - 颜色交替: ON(白色) / OFF(黑色) 帧")
print(f"   - 方块大小: {STIM_SIZE} 像素")
print(f"   - 持续时间: {TOTAL_FRAMES/FPS:.1f} 秒")