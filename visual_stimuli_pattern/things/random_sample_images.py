#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从object_images目录中随机抽取1000个文件夹，每个文件夹随机抽取1张图片，
复制到images目录下
"""

import os
import random
import shutil
from pathlib import Path

# 设置随机种子以便复现
random.seed(42)

# 定义路径
SOURCE_DIR = "/media/ubuntu/sda/visual_stimuli_pattern/things/object_images"
TARGET_DIR = "/media/ubuntu/sda/visual_stimuli_pattern/things/images"
NUM_FOLDERS = 1000

def get_all_folders(source_dir):
    """获取所有文件夹路径"""
    folders = []
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            folders.append(item_path)
    return folders

def get_images_from_folder(folder_path):
    """从文件夹中获取所有图片文件"""
    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            images.append(os.path.join(folder_path, file))
    return images

def main():
    print("开始处理...")
    
    # 确保目标目录存在
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"目标目录: {TARGET_DIR}")
    
    # 获取所有文件夹
    print(f"\n正在扫描源目录: {SOURCE_DIR}")
    all_folders = get_all_folders(SOURCE_DIR)
    print(f"找到 {len(all_folders)} 个文件夹")
    
    if len(all_folders) < NUM_FOLDERS:
        print(f"警告：文件夹数量 ({len(all_folders)}) 少于请求的数量 ({NUM_FOLDERS})")
        print(f"将使用所有 {len(all_folders)} 个文件夹")
        selected_folders = all_folders
    else:
        # 随机选择文件夹
        selected_folders = random.sample(all_folders, NUM_FOLDERS)
        print(f"随机选择了 {NUM_FOLDERS} 个文件夹")
    
    # 从每个文件夹中随机选择1张图片并复制
    copied_count = 0
    skipped_count = 0
    
    print(f"\n开始复制图片...")
    for i, folder_path in enumerate(selected_folders, 1):
        folder_name = os.path.basename(folder_path)
        images = get_images_from_folder(folder_path)
        
        if len(images) == 0:
            print(f"[{i}/{len(selected_folders)}] 跳过 {folder_name}: 没有找到图片文件")
            skipped_count += 1
            continue
        
        # 随机选择一张图片
        selected_image = random.choice(images)
        image_name = os.path.basename(selected_image)
        
        # 生成目标路径（使用文件夹名_图片名来避免重名）
        target_image_name = f"{folder_name}_{image_name}"
        target_path = os.path.join(TARGET_DIR, target_image_name)
        
        # 复制文件
        try:
            shutil.copy2(selected_image, target_path)
            copied_count += 1
            if i % 100 == 0:
                print(f"[{i}/{len(selected_folders)}] 已复制 {copied_count} 张图片...")
        except Exception as e:
            print(f"[{i}/{len(selected_folders)}] 复制失败 {folder_name}/{image_name}: {e}")
            skipped_count += 1
    
    print(f"\n完成！")
    print(f"成功复制: {copied_count} 张图片")
    print(f"跳过: {skipped_count} 个文件夹")
    print(f"目标目录: {TARGET_DIR}")

if __name__ == "__main__":
    main()

