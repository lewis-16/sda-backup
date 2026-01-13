#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据pkl文件进行有选择的采样：
- 对于cluster 1/5/6：选取所有class，每个class选取1-2张，总共选取650张
- 对于剩下的cluster：每个class选取1张，总共选取350张
"""

import os
import random
import pickle
import shutil
from collections import defaultdict

# 设置随机种子以便复现
random.seed(42)

# 定义路径
PKL_FILE = "/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyN/merged_cluster_class_counts.pkl"
SOURCE_DIR = "/media/ubuntu/sda/visual_stimuli_pattern/things/object_images"
TARGET_DIR = "/media/ubuntu/sda/visual_stimuli_pattern/things/images"

# 目标数量
TARGET_CLUSTER_156 = 650  # cluster 1/5/6 总共650张
TARGET_OTHER = 350  # 其他cluster总共350张

def load_cluster_data(pkl_file):
    """加载pkl文件"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def get_images_from_folder(folder_path):
    """从文件夹中获取所有图片文件"""
    images = []
    if not os.path.exists(folder_path):
        return images
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            images.append(os.path.join(folder_path, file))
    return images

def allocate_images_for_cluster_156(classes, target_total):
    """
    为cluster 1/5/6分配图片数量
    每个class选1-2张，总共target_total张
    """
    num_classes = len(classes)
    if num_classes == 0:
        return {}
    
    # 计算需要多少class选2张才能达到目标
    # 如果所有class都选1张，总共num_classes张
    # 需要额外 target_total - num_classes 张
    # 每个额外的class（选2张）贡献1张额外
    extra_needed = target_total - num_classes
    
    if extra_needed <= 0:
        # 如果目标数量小于等于class数量，所有class都选1张
        return {cls: 1 for cls in classes}
    
    if extra_needed >= num_classes:
        # 如果目标数量大于等于2倍class数量，所有class都选2张
        return {cls: 2 for cls in classes}
    
    # 随机选择extra_needed个class选2张，其他选1张
    classes_list = list(classes)
    random.shuffle(classes_list)
    allocation = {}
    for i, cls in enumerate(classes_list):
        if i < extra_needed:
            allocation[cls] = 2
        else:
            allocation[cls] = 1
    
    return allocation

def sample_from_class(class_name, num_images, source_dir):
    """从指定class中采样指定数量的图片"""
    folder_path = os.path.join(source_dir, class_name)
    images = get_images_from_folder(folder_path)
    
    if len(images) == 0:
        return []
    
    # 如果需要的图片数大于可用图片数，返回所有图片
    num_to_select = min(num_images, len(images))
    selected = random.sample(images, num_to_select)
    return selected

def main():
    print("开始处理...")
    
    # 加载pkl文件
    print(f"\n正在加载pkl文件: {PKL_FILE}")
    cluster_data = load_cluster_data(PKL_FILE)
    print(f"找到 {len(cluster_data)} 个clusters")
    
    # 打印每个cluster的class数量
    for k in sorted(cluster_data.keys()):
        print(f"  Cluster {k}: {len(cluster_data[k])} classes")
    
    # 分离cluster 1/5/6和其他
    cluster_156_classes = []
    for cluster_id in [1, 5, 6]:
        cluster_156_classes.extend(cluster_data[cluster_id])
    
    other_classes = []
    for cluster_id in [0, 2, 3, 4]:
        other_classes.extend(cluster_data[cluster_id])
    
    print(f"\nCluster 1/5/6 总共有 {len(cluster_156_classes)} 个classes")
    print(f"其他cluster总共有 {len(other_classes)} 个classes")
    
    # 为cluster 1/5/6分配图片数量
    print(f"\n为cluster 1/5/6分配图片（目标：{TARGET_CLUSTER_156}张）...")
    allocation_156 = allocate_images_for_cluster_156(cluster_156_classes, TARGET_CLUSTER_156)
    total_allocated_156 = sum(allocation_156.values())
    print(f"分配完成：总共 {total_allocated_156} 张图片")
    print(f"  选1张的class: {sum(1 for v in allocation_156.values() if v == 1)} 个")
    print(f"  选2张的class: {sum(1 for v in allocation_156.values() if v == 2)} 个")
    
    # 为其他cluster随机选择350个classes，每个选1张
    print(f"\n为其他cluster选择classes（目标：{TARGET_OTHER}张）...")
    if len(other_classes) < TARGET_OTHER:
        print(f"警告：其他cluster只有 {len(other_classes)} 个classes，少于目标 {TARGET_OTHER}")
        selected_other_classes = other_classes
    else:
        selected_other_classes = random.sample(other_classes, TARGET_OTHER)
    allocation_other = {cls: 1 for cls in selected_other_classes}
    print(f"选择了 {len(selected_other_classes)} 个classes，每个选1张")
    
    # 确保目标目录存在
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"\n目标目录: {TARGET_DIR}")
    
    # 开始采样和复制
    print(f"\n开始采样和复制图片...")
    copied_count = 0
    skipped_count = 0
    total_to_process = len(allocation_156) + len(allocation_other)
    processed = 0
    
    # 处理cluster 1/5/6
    for class_name, num_images in allocation_156.items():
        processed += 1
        selected_images = sample_from_class(class_name, num_images, SOURCE_DIR)
        
        if len(selected_images) == 0:
            print(f"[{processed}/{total_to_process}] 跳过 {class_name}: 文件夹不存在或没有图片")
            skipped_count += 1
            continue
        
        # 复制选中的图片
        for img_path in selected_images:
            img_name = os.path.basename(img_path)
            target_img_name = f"{class_name}_{img_name}"
            target_path = os.path.join(TARGET_DIR, target_img_name)
            
            try:
                shutil.copy2(img_path, target_path)
                copied_count += 1
            except Exception as e:
                print(f"[{processed}/{total_to_process}] 复制失败 {class_name}/{img_name}: {e}")
                skipped_count += 1
        
        if processed % 50 == 0:
            print(f"[{processed}/{total_to_process}] 已处理，已复制 {copied_count} 张图片...")
    
    # 处理其他cluster
    for class_name in selected_other_classes:
        processed += 1
        selected_images = sample_from_class(class_name, 1, SOURCE_DIR)
        
        if len(selected_images) == 0:
            print(f"[{processed}/{total_to_process}] 跳过 {class_name}: 文件夹不存在或没有图片")
            skipped_count += 1
            continue
        
        # 复制选中的图片
        img_path = selected_images[0]
        img_name = os.path.basename(img_path)
        target_img_name = f"{class_name}_{img_name}"
        target_path = os.path.join(TARGET_DIR, target_img_name)
        
        try:
            shutil.copy2(img_path, target_path)
            copied_count += 1
        except Exception as e:
            print(f"[{processed}/{total_to_process}] 复制失败 {class_name}/{img_name}: {e}")
            skipped_count += 1
        
        if processed % 50 == 0:
            print(f"[{processed}/{total_to_process}] 已处理，已复制 {copied_count} 张图片...")
    
    print(f"\n完成！")
    print(f"成功复制: {copied_count} 张图片")
    print(f"跳过: {skipped_count} 个classes")
    print(f"目标目录: {TARGET_DIR}")
    
    # 统计信息
    cluster_156_count = sum(allocation_156.values())
    other_count = len(allocation_other)
    print(f"\n统计信息:")
    print(f"  Cluster 1/5/6: 计划 {cluster_156_count} 张")
    print(f"  其他cluster: 计划 {other_count} 张")
    print(f"  总计: 计划 {cluster_156_count + other_count} 张，实际复制 {copied_count} 张")

if __name__ == "__main__":
    main()

