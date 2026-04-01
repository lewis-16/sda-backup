#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用stimuli_new目录中的590张图像生成序列文档
- 随机排列590张图像
- 重复20次，总共11800个trial
- 每一组图片的顺序都不同（每次重新随机打乱）
"""

import os
import random
import pandas as pd
from pathlib import Path

random.seed(42)

IMAGES_DIR = "/media/ubuntu/sda/visual_stimuli_pattern/stimuli_NSD_590"
OUTPUT_CSV = "/media/ubuntu/sda/visual_stimuli_pattern/stimuli_sequence_11800.csv"
NUM_REPEATS = 20

def collect_images(images_dir):
    image_files = []
    for file in sorted(os.listdir(images_dir)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(images_dir, file)
            image_files.append(img_path)
    return image_files

def generate_sequence(images, num_repeats):
    """
    生成序列：重复num_repeats次，每次顺序都不同
    """
    sequence = []
    
    for repeat_idx in range(num_repeats):
        # 每次重复都重新随机打乱
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)
        
        # 添加到序列中
        for img_path in shuffled_images:
            sequence.append({
                'image_path': img_path,
                'repeat_group': repeat_idx + 1,  # 从1开始编号
                'image_name': os.path.basename(img_path)
            })
        
        print(f"第 {repeat_idx + 1} 组完成，已生成 {len(sequence)} 个trials")
    
    return sequence

def save_sequence_to_csv(sequence, output_path):
    """将序列保存为CSV文件"""
    data = {
        'stimulus_number': list(range(1, len(sequence) + 1)),
        'repeat_group': [trial['repeat_group'] for trial in sequence],
        'image_path': [trial['image_path'] for trial in sequence],
        'image_name': [trial['image_name'] for trial in sequence]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n序列已保存到: {output_path}")
    print(f"总刺激次数: {len(sequence)}")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  图片总数: {len(set(df['image_path']))}")
    print(f"  重复组数: {NUM_REPEATS}")
    print(f"  每组图片数: {len(sequence) // NUM_REPEATS}")
    print(f"  总trial数: {len(sequence)}")
    
    # 检查每组是否包含所有图片
    unique_images = set(df['image_path'].unique())
    for group in range(1, NUM_REPEATS + 1):
        group_images = set(df[df['repeat_group'] == group]['image_path'].unique())
        if len(group_images) != len(unique_images):
            print(f"  警告：第{group}组图片数量不完整 ({len(group_images)}/{len(unique_images)})")
        else:
            print(f"  第{group}组: 包含所有 {len(group_images)} 张图片")
    
    return df

def main():
    print("开始收集图片...")
    images = collect_images(IMAGES_DIR)
    
    print(f"\n找到 {len(images)} 张图片")

    if len(images) == 0:
        print("错误：没有找到图片文件！")
        return
    
    if len(images) != 590:
        print(f"警告：图片数量为 {len(images)}，不是预期的590张")
    
    print(f"\n生成序列（重复 {NUM_REPEATS} 次，每次顺序不同）...")
    sequence = generate_sequence(images, NUM_REPEATS)
    
    print(f"\n保存CSV文件...")
    df = save_sequence_to_csv(sequence, OUTPUT_CSV)
    
    print(f"\n前10行预览:")
    print(df.head(10))
    
    print(f"\n后10行预览:")
    print(df.tail(10))
    
    # 验证每组顺序是否不同
    print(f"\n验证每组顺序...")
    first_group_order = list(df[df['repeat_group'] == 1]['image_path'])
    all_different = True
    for group in range(2, NUM_REPEATS + 1):
        group_order = list(df[df['repeat_group'] == group]['image_path'])
        if group_order == first_group_order:
            print(f"  警告：第{group}组与第1组顺序相同！")
            all_different = False
    
    if all_different:
        print(f"  ✓ 所有组的顺序都不同")

if __name__ == "__main__":
    main()

