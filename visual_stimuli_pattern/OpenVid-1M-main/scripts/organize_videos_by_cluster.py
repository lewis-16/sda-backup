#!/usr/bin/env python3
"""
根据 CSV 文件中的 cluster_labels，将视频文件从 part_* 文件夹复制到 filtred_video 对应的 cluster_labels 文件夹下。
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def find_video_in_parts(video_name, base_dir, part_numbers):
    """在所有 part_* 文件夹中查找视频文件"""
    for part_num in part_numbers:
        part_dir = os.path.join(base_dir, f"part_{part_num}")
        video_path = os.path.join(part_dir, video_name)
        if os.path.isfile(video_path):
            return video_path
    return None

def main():
    # 配置路径
    base_dir = "/media/ubuntu/sda/visual_stimuli_pattern/OpenVid-1M-main"
    csv_path = os.path.join(base_dir, "OpenVidHD_filtered.csv")
    output_base_dir = os.path.join(base_dir, "filtred_video")
    
    # 读取 CSV 文件
    print("正在读取 CSV 文件...")
    df = pd.read_csv(csv_path)
    print(f"CSV 中共有 {len(df)} 条记录")
    
    # 获取所有 part_* 文件夹编号（假设从 part_41 开始）
    # 动态检测存在的 part_* 文件夹
    part_numbers = []
    for i in range(1, 100):  # 假设最多到 part_99
        part_dir = os.path.join(base_dir, f"part_{i}")
        if os.path.isdir(part_dir):
            part_numbers.append(i)
    
    print(f"找到以下 part 文件夹: {part_numbers}")
    
    # 统计信息
    copied_count = 0
    not_found_count = 0
    skipped_count = 0
    not_found_videos = []
    
    # 确保输出基础目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 遍历 CSV 中的每一行
    print("\n开始复制视频文件...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理视频"):
        video_name = str(row['video']).strip()
        cluster_label = str(row['cluster_labels']).strip()
        
        # 跳过无效数据
        if pd.isna(row['video']) or pd.isna(row['cluster_labels']):
            skipped_count += 1
            continue
        
        # 创建目标文件夹（根据 cluster_labels）
        target_dir = os.path.join(output_base_dir, cluster_label)
        os.makedirs(target_dir, exist_ok=True)
        
        # 目标文件路径
        target_path = os.path.join(target_dir, video_name)
        
        # 如果目标文件已存在，跳过
        if os.path.isfile(target_path):
            skipped_count += 1
            continue
        
        # 在所有 part_* 文件夹中查找视频文件
        source_path = find_video_in_parts(video_name, base_dir, part_numbers)
        
        if source_path:
            try:
                # 复制文件
                shutil.copy2(source_path, target_path)
                copied_count += 1
            except Exception as e:
                print(f"\n复制失败: {video_name} -> {target_path}")
                print(f"错误: {e}")
                not_found_count += 1
                not_found_videos.append(video_name)
        else:
            not_found_count += 1
            not_found_videos.append(video_name)
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"成功复制: {copied_count} 个视频")
    print(f"跳过（已存在）: {skipped_count} 个视频")
    print(f"未找到: {not_found_count} 个视频")
    print(f"输出目录: {output_base_dir}")
    
    if not_found_videos:
        print(f"\n未找到的视频文件示例（前10个）:")
        for video in not_found_videos[:10]:
            print(f"  - {video}")
        if len(not_found_videos) > 10:
            print(f"  ... 还有 {len(not_found_videos) - 10} 个未找到的视频")

if __name__ == "__main__":
    main()

