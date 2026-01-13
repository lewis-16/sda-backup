#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据CSV文件生成视频
从images_sequence_10000.csv读取图片路径，按照顺序生成视频
每张图片呈现250ms
"""

import os
import csv
import cv2
import numpy as np

# 定义路径
CSV_FILE = "/media/ubuntu/sda/visual_stimuli_pattern/things/images_sequence_10000.csv"
OUTPUT_VIDEO = "/media/ubuntu/sda/visual_stimuli_pattern/things/visual_stimuli_sequence_video.mp4"

# 视频参数
FPS = 30  # 帧率
IMAGE_DURATION_MS = 250  # 每张图片持续时间（毫秒）
FRAMES_PER_IMAGE = int(FPS * IMAGE_DURATION_MS / 1000.0)  # 每张图片的帧数

def load_image_paths_from_csv(csv_file):
    """从CSV文件加载图片路径，按照stimulus_number排序"""
    image_paths = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row['image_path']
            stimulus_number = int(row['stimulus_number'])
            image_paths.append((stimulus_number, image_path))
    
    # 按照stimulus_number排序
    image_paths.sort(key=lambda x: x[0])
    
    # 只返回路径列表
    return [path for _, path in image_paths]

def main():
    print("开始生成视频...")
    
    # 从CSV文件加载图片路径
    print(f"\n正在读取CSV文件: {CSV_FILE}")
    image_paths = load_image_paths_from_csv(CSV_FILE)
    print(f"找到 {len(image_paths)} 张图片")
    
    if len(image_paths) == 0:
        print("错误：CSV文件中没有找到图片路径！")
        return
    
    # 检查第一张图片是否存在并获取尺寸
    first_path = image_paths[0]
    if not os.path.exists(first_path):
        print(f"错误：第一张图片不存在: {first_path}")
        return
    
    first_img = cv2.imread(first_path)
    if first_img is None:
        print(f"错误：无法读取第一张图片: {first_path}")
        return
    
    height, width = first_img.shape[:2]
    print(f"图片尺寸: {width}x{height}")
    
    # 计算视频参数
    total_frames = len(image_paths) * FRAMES_PER_IMAGE
    video_duration = total_frames / FPS
    print(f"\n视频参数:")
    print(f"  帧率: {FPS} fps")
    print(f"  每张图片持续时间: {IMAGE_DURATION_MS} ms ({FRAMES_PER_IMAGE} 帧)")
    print(f"  总帧数: {total_frames}")
    print(f"  视频时长: {video_duration:.2f} 秒 ({video_duration/60:.2f} 分钟)")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))
    
    if not out.isOpened():
        print("错误：无法创建视频文件！")
        return
    
    print(f"\n开始处理图片...")
    processed_images = 0
    missing_images = 0
    
    for i, image_path in enumerate(image_paths, 1):
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"[{i}/{len(image_paths)}] 警告：图片不存在 {image_path}，跳过")
            missing_images += 1
            # 使用黑色图片占位
            img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # 读取图片
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"[{i}/{len(image_paths)}] 警告：无法读取图片 {os.path.basename(image_path)}，跳过")
                missing_images += 1
                # 使用黑色图片占位
                img = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                # 如果图片尺寸不一致，调整大小
                if img.shape[1] != width or img.shape[0] != height:
                    img = cv2.resize(img, (width, height))
        
        # 将图片写入视频，重复FRAMES_PER_IMAGE次
        for _ in range(FRAMES_PER_IMAGE):
            out.write(img)
        
        processed_images += 1
        
        if i % 1000 == 0:
            progress = i / len(image_paths) * 100
            print(f"[{i}/{len(image_paths)}] 进度: {progress:.1f}% - 已处理 {processed_images} 张图片")
    
    # 释放资源
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n完成！")
    print(f"成功处理: {processed_images} 张图片")
    if missing_images > 0:
        print(f"缺失图片: {missing_images} 张（已用黑色占位）")
    print(f"视频已保存到: {OUTPUT_VIDEO}")
    print(f"视频时长: {video_duration:.2f} 秒 ({video_duration/60:.2f} 分钟)")
    
    # 检查文件大小
    if os.path.exists(OUTPUT_VIDEO):
        file_size = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)  # MB
        print(f"视频文件大小: {file_size:.2f} MB")

if __name__ == "__main__":
    main()

