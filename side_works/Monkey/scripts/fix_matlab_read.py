#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复方案：使用 h5py 读取 MATLAB v7.3+ 格式的 .mat 文件
原始代码使用 scipy.io.loadmat 无法读取 MATLAB v7.3 以上的文件
版本 116 (0x74) 表示这是 HDF5 格式的 MATLAB 文件
"""

import h5py
import numpy as np

file_path = "/media/ubuntu/sda/Monkey/TVSD/monkeyF/THINGS_MUA_trials.mat"

def read_mat_file(filepath):
    """
    读取 MATLAB .mat 文件，支持 v7.3+ 版本
    """
    data = {}
    with h5py.File(filepath, 'r') as f:
        # 打印所有顶层变量名
        print("文件中的顶层变量:")
        for key in f.keys():
            print(f"  - {key}")
        
        # 递归读取所有数据
        def read_group(group, prefix=''):
            for name in group:
                full_name = f"{prefix}/{name}" if prefix else name
                item = group[name]
                if isinstance(item, h5py.Dataset):
                    # 读取数据集
                    try:
                        data[full_name] = item[()]
                    except Exception as e:
                        print(f"无法读取 {full_name}: {e}")
                elif isinstance(item, h5py.Group):
                    # 递归处理子组
                    read_group(item, full_name)
        
        read_group(f)
    
    return data

# 读取数据
print("正在读取文件:", file_path)
a = read_mat_file(file_path)

print("\n" + "="*50)
print("数据加载完成!")
print("="*50)
print(f"共读取 {len(a)} 个变量")
print("\n变量列表:")
for i, key in enumerate(a.keys(), 1):
    value = a[key]
    shape = value.shape if hasattr(value, 'shape') else 'unknown'
    print(f"  {i}. {key} (shape: {shape})")
