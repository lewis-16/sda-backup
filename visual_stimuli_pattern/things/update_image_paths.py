#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修改CSV文件中的image_path格式
从: /media/ubuntu/sda/visual_stimuli_pattern/things/images/rust_rust_09s.jpg
到: /media/ubuntu/sda/visual_stimuli_pattern/things/object_images/rust/rust_09s.jpg
"""

import os
import re
import csv

# 定义路径
CSV_FILE = "/media/ubuntu/sda/visual_stimuli_pattern/things/images_sequence_10000.csv"
OUTPUT_CSV = "/media/ubuntu/sda/visual_stimuli_pattern/things/images_sequence_10000.csv"
OBJECT_IMAGES_DIR = "/media/ubuntu/sda/visual_stimuli_pattern/things/object_images"

def extract_category_from_filename(filename):
    """
    从文件名中提取类别名
    例如: rust_rust_09s.jpg -> rust
          teddy_bear_teddy_bear_07s.jpg -> teddy_bear
    """
    # 去掉扩展名
    name_without_ext = os.path.splitext(filename)[0]
    
    # 查找重复的类别名模式
    # 例如: rust_rust_09s -> rust
    # 例如: teddy_bear_teddy_bear_07s -> teddy_bear
    
    # 尝试匹配 pattern_pattern_rest 格式
    # 使用正则表达式找到重复的部分
    match = re.match(r'^(.+?)_\1_(.+)$', name_without_ext)
    if match:
        category = match.group(1)
        rest = match.group(2)
        return category, f"{category}_{rest}"
    
    # 如果匹配失败，尝试其他模式
    # 有些可能是 category_category_suffix 格式
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        # 尝试找到重复的部分
        for i in range(1, len(parts) // 2 + 1):
            if parts[:i] == parts[i:2*i]:
                category = '_'.join(parts[:i])
                rest = '_'.join(parts[2*i:])
                return category, f"{category}_{rest}"
    
    # 如果都匹配不上，返回None
    return None, None

def convert_image_path(old_path):
    """
    转换image_path
    从: /media/ubuntu/sda/visual_stimuli_pattern/things/images/rust_rust_09s.jpg
    到: /media/ubuntu/sda/visual_stimuli_pattern/things/object_images/rust/rust_09s.jpg
    """
    # 提取文件名
    filename = os.path.basename(old_path)
    
    # 提取类别名和新文件名
    category, new_filename = extract_category_from_filename(filename)
    
    if category is None:
        print(f"警告：无法解析文件名 {filename}")
        return old_path
    
    # 添加扩展名
    ext = os.path.splitext(filename)[1]
    new_filename_with_ext = new_filename + ext
    
    # 构建新路径
    base_dir = "/media/ubuntu/sda/visual_stimuli_pattern/things"
    new_path = os.path.join(base_dir, "object_images", category, new_filename_with_ext)
    
    return new_path

def main():
    print("开始修改image_path...")
    
    # 读取CSV文件
    rows = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for i, row in enumerate(reader, 1):
            old_path = row['image_path']
            new_path = convert_image_path(old_path)
            row['image_path'] = new_path
            rows.append(row)
            
            if i % 1000 == 0:
                print(f"已处理 {i} 行...")
    
    # 写入新CSV文件
    print(f"\n写入新CSV文件...")
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"完成！已更新 {len(rows)} 行")
    print(f"文件已保存到: {OUTPUT_CSV}")
    
    # 显示几个示例
    print(f"\n示例转换（前5行）:")
    for i in range(min(5, len(rows))):
        old_path = rows[i].get('image_path', '')
        print(f"  {i+1}: {os.path.basename(old_path)}")

if __name__ == "__main__":
    main()

