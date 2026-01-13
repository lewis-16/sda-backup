#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从images目录中的1000张图片生成视频
每张图片呈现150ms
"""

import os
import cv2
import numpy as np
from pathlib import Path

# 定义路径
IMAGES_DIR = "/media/ubuntu/sda/visual_stimuli_pattern/things/images"
OUTPUT_VIDEO = "/media/ubuntu/sda/visual_stimuli_pattern/things/visual_stimuli_video.mp4"

# 视频参数
FPS = 30  # 帧率
IMAGE_DURATION_MS = 150  # 每张图片持续时间（毫秒）
FRAMES_PER_IMAGE = int(FPS * IMAGE_DURATION_MS / 1000.0)  # 每张图片的帧数

def get_image_files(images_dir):
    """获取所有图片文件，按文件名排序"""
    image_files = []
    for file in sorted(os.listdir(images_dir)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(os.path.join(images_dir, file))
    return image_files

def get_image_size(image_path):
    """获取图片尺寸"""
    img = cv2.imread(image_path)
    if img is not None:
        return img.shape[1], img.shape[0]  # width, height
    return None, None

def main():
    print("开始生成视频...")
    
    # 获取所有图片文件
    print(f"\n正在扫描图片目录: {IMAGES_DIR}")
    image_files = get_image_files(IMAGES_DIR)
    print(f"找到 {len(image_files)} 张图片")
    
    if len(image_files) == 0:
        print("错误：没有找到图片文件！")
        return
    
    # 获取第一张图片的尺寸
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"错误：无法读取第一张图片: {image_files[0]}")
        return
    
    height, width = first_img.shape[:2]
    print(f"图片尺寸: {width}x{height}")
    
    # 计算视频参数
    total_frames = len(image_files) * FRAMES_PER_IMAGE
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
    
    for i, image_path in enumerate(image_files, 1):
        # 读取图片
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"[{i}/{len(image_files)}] 警告：无法读取图片 {os.path.basename(image_path)}，跳过")
            continue
        
        # 如果图片尺寸不一致，调整大小
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        
        # 将图片写入视频，重复FRAMES_PER_IMAGE次
        for _ in range(FRAMES_PER_IMAGE):
            out.write(img)
        
        processed_images += 1
        
        if i % 100 == 0:
            progress = i / len(image_files) * 100
            print(f"[{i}/{len(image_files)}] 进度: {progress:.1f}% - 已处理 {processed_images} 张图片")
    
    # 释放资源
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n完成！")
    print(f"成功处理: {processed_images} 张图片")
    print(f"视频已保存到: {OUTPUT_VIDEO}")
    print(f"视频时长: {video_duration:.2f} 秒 ({video_duration/60:.2f} 分钟)")
    
    # 检查文件大小
    if os.path.exists(OUTPUT_VIDEO):
        file_size = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)  # MB
        print(f"视频文件大小: {file_size:.2f} MB")

if __name__ == "__main__":
    main()

