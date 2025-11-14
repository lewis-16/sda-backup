#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成视觉刺激序列
要求：
- object_images_sample10: 1854类，每类10张，每张呈现1次
- test: 100张，每张呈现30次
- test的trial混杂在object_images中，在前10000次刺激中完成
- 相同图片不在相邻两次刺激中出现
- 同一类别的图像不在相邻几次中出现
"""

import os
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 设置随机种子以便复现
random.seed(42)

# 定义路径
OBJECT_IMAGES_DIR = "/media/ubuntu/sda/visual_stimuli_pattern/things/object_images_sample10"
TEST_DIR = "/media/ubuntu/sda/visual_stimuli_pattern/things/test"
OUTPUT_CSV = "/media/ubuntu/sda/Spike_Sorting/visual_stimuli_sequence.csv"

def collect_object_images():
    """收集object_images的所有图片，按类别分组"""
    category_images = defaultdict(list)
    
    for category in sorted(os.listdir(OBJECT_IMAGES_DIR)):
        category_path = os.path.join(OBJECT_IMAGES_DIR, category)
        if os.path.isdir(category_path):
            for img_file in sorted(os.listdir(category_path)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(category_path, img_file)
                    category_images[category].append(img_path)
    
    return category_images

def collect_test_images():
    """收集test目录的所有图片"""
    test_images = []
    for img_file in sorted(os.listdir(TEST_DIR)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(TEST_DIR, img_file)
            test_images.append(img_path)
    return test_images

def create_trial_list(category_images, test_images):
    """创建所有的trial"""
    # 创建object_images的trials (每张图片呈现1次)
    object_trials = []
    for category, images in category_images.items():
        for img_path in images:
            object_trials.append({
                'image_path': img_path,
                'category': category,
                'is_train': 1,
                'is_test': 0
            })
    
    # 创建test的trials (每张图片呈现30次)
    test_trials = []
    for img_path in test_images:
        for _ in range(30):
            test_trials.append({
                'image_path': img_path,
                'category': 'test',
                'is_train': 0,
                'is_test': 1
            })
    
    print(f"Object images trials: {len(object_trials)}")
    print(f"Test trials: {len(test_trials)}")
    print(f"Total trials: {len(object_trials) + len(test_trials)}")
    
    return object_trials, test_trials

def is_valid_placement(sequence, new_trial, position):
    """
    检查新trial放置在指定位置是否合法
    - 相同图片不能相邻
    - 相同类别不能相邻（对object_images）
    """
    # 检查前一个位置
    if position > 0:
        prev_trial = sequence[position - 1]
        # 不能是相同图片
        if prev_trial['image_path'] == new_trial['image_path']:
            return False
        # 对于object_images，不能是相同类别
        if (prev_trial['is_train'] == 1 and new_trial['is_train'] == 1 and 
            prev_trial['category'] == new_trial['category']):
            return False
    
    # 检查后一个位置（如果存在）
    if position < len(sequence):
        next_trial = sequence[position]
        # 不能是相同图片
        if next_trial['image_path'] == new_trial['image_path']:
            return False
        # 对于object_images，不能是相同类别
        if (next_trial['is_train'] == 1 and new_trial['is_train'] == 1 and 
            next_trial['category'] == new_trial['category']):
            return False
    
    return True

def generate_sequence_greedy(object_trials, test_trials):
    """
    使用贪心策略生成序列
    确保test的3000次在前10000次刺激中完成
    """
    # 打乱所有trials
    random.shuffle(object_trials)
    random.shuffle(test_trials)
    
    # 计算需要在前10000次中插入多少test trials
    test_in_first_10k = len(test_trials)  # 3000次
    object_in_first_10k = 10000 - test_in_first_10k  # 7000次
    
    # 分配object_trials
    object_first_10k = object_trials[:object_in_first_10k]
    object_after_10k = object_trials[object_in_first_10k:]
    
    # 在前10000次中混合test和object trials
    first_10k_pool = test_trials + object_first_10k
    random.shuffle(first_10k_pool)
    
    sequence = []
    used_trials = set()
    
    # 尝试多次来生成合法序列
    max_attempts = 10
    for attempt in range(max_attempts):
        sequence = []
        random.shuffle(first_10k_pool)
        remaining_pool = first_10k_pool.copy()
        
        # 生成前10000次
        attempts_count = 0
        max_local_attempts = 100000
        
        while len(remaining_pool) > 0 and attempts_count < max_local_attempts:
            attempts_count += 1
            
            # 随机选择一个trial
            trial_idx = random.randint(0, len(remaining_pool) - 1)
            trial = remaining_pool[trial_idx]
            
            # 检查是否可以添加到序列末尾
            if len(sequence) == 0 or is_valid_placement(sequence, trial, len(sequence)):
                sequence.append(trial)
                remaining_pool.pop(trial_idx)
                attempts_count = 0  # 重置计数器
            elif attempts_count > 1000:
                # 如果尝试太多次，尝试找到任何可以放置的trial
                found = False
                for i, t in enumerate(remaining_pool):
                    if is_valid_placement(sequence, t, len(sequence)):
                        sequence.append(t)
                        remaining_pool.pop(i)
                        attempts_count = 0
                        found = True
                        break
                if not found:
                    attempts_count += 1
        
        if len(remaining_pool) == 0:
            print(f"成功在第{attempt + 1}次尝试中生成前10000次序列")
            break
        else:
            print(f"第{attempt + 1}次尝试失败，剩余{len(remaining_pool)}个trials，重试...")
    
    if len(remaining_pool) > 0:
        print(f"警告：无法完美安排前10000次，剩余{len(remaining_pool)}个trials")
        # 将剩余的添加到object_after_10k
        object_after_10k.extend(remaining_pool)
    
    # 添加剩余的object trials（10000次之后）
    random.shuffle(object_after_10k)
    for trial in object_after_10k:
        attempts = 0
        max_local_attempts = 10000
        placed = False
        
        while attempts < max_local_attempts:
            # 尝试添加到序列末尾
            if is_valid_placement(sequence, trial, len(sequence)):
                sequence.append(trial)
                placed = True
                break
            else:
                # 如果不能直接添加，尝试与最后几个交换位置
                if len(sequence) > 10:
                    swap_idx = random.randint(max(0, len(sequence) - 50), len(sequence) - 1)
                    sequence.insert(swap_idx, trial)
                    if not check_sequence_validity(sequence):
                        sequence.pop(swap_idx)
                    else:
                        placed = True
                        break
            attempts += 1
        
        if not placed:
            # 强制添加
            sequence.append(trial)
    
    return sequence

def check_sequence_validity(sequence):
    """检查整个序列的有效性"""
    for i in range(len(sequence) - 1):
        curr = sequence[i]
        next = sequence[i + 1]
        
        # 检查相同图片
        if curr['image_path'] == next['image_path']:
            return False
        
        # 检查相同类别
        if (curr['is_train'] == 1 and next['is_train'] == 1 and 
            curr['category'] == next['category']):
            return False
    
    return True

def save_sequence_to_csv(sequence, output_path):
    """将序列保存为CSV文件"""
    data = {
        'stimulus_number': list(range(1, len(sequence) + 1)),
        'train': [trial['is_train'] for trial in sequence],
        'test': [trial['is_test'] for trial in sequence],
        'image_path': [trial['image_path'] for trial in sequence]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n序列已保存到: {output_path}")
    print(f"总刺激次数: {len(sequence)}")
    
    # 统计信息
    test_count = sum(trial['is_test'] for trial in sequence)
    train_count = sum(trial['is_train'] for trial in sequence)
    test_in_first_10k = sum(trial['is_test'] for trial in sequence[:10000])
    
    print(f"Test trials: {test_count}")
    print(f"Train trials: {train_count}")
    print(f"前10000次中test的数量: {test_in_first_10k}")
    
    return df

def main():
    print("开始收集图片...")
    category_images = collect_object_images()
    test_images = collect_test_images()
    
    print(f"\n类别数量: {len(category_images)}")
    print(f"Test图片数量: {len(test_images)}")
    
    print("\n创建trials...")
    object_trials, test_trials = create_trial_list(category_images, test_images)
    
    print("\n生成刺激序列...")
    sequence = generate_sequence_greedy(object_trials, test_trials)
    
    print("\n检查序列有效性...")
    if check_sequence_validity(sequence):
        print("✓ 序列有效：没有相邻的相同图片或相同类别")
    else:
        print("✗ 警告：序列可能包含相邻的相同图片或相同类别")
    
    print("\n保存CSV文件...")
    df = save_sequence_to_csv(sequence, OUTPUT_CSV)
    
    print("\n前10行预览:")
    print(df.head(10))

if __name__ == "__main__":
    main()

