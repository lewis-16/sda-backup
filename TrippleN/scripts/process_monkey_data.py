#!/usr/bin/env python3
"""
按猴子分组处理PSTH数据，内存优化版本
"""

import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
import gc

# 配置
monkey_names = ['JianJian', 'FaCai', 'TuTu', 'MaoDan', 'ZhuangZhuang']
columns = ['B_SI', 'F_SI', 'O_SI', 'UnitType', 'best_r_time1', 'best_r_time2', 
           'pos', 'reliability_basic', 'reliability_best', 'reliability_find_testset', 
           'snr', 'snrmax']
psth_dir = "/media/ubuntu/sda/TrippleN/psth"
processed_dir = "/media/ubuntu/sda/TrippleN/Processed"
output_dir = "/media/ubuntu/sda/TrippleN/GoodUnit_by_monkey"

os.makedirs(output_dir, exist_ok=True)

psth_files = sorted(os.listdir(psth_dir))
processed_files = sorted(os.listdir(processed_dir))

print("=" * 50)
print("逐只猴子处理数据（内存优化版）...")
print("=" * 50)

for monkey in monkey_names:
    print(f"\n处理猴子: {monkey}")
    print("-" * 30)
    
    # 找出属于当前猴子的session
    monkey_psth_files = []
    monkey_processed_files = []
    
    for i, psth_file in enumerate(psth_files):
        parts = psth_file.replace('.npy', '').split('_')
        if len(parts) >= 4 and parts[3] == monkey:
            monkey_psth_files.append(psth_file)
            monkey_processed_files.append(processed_files[i])
    
    if len(monkey_psth_files) == 0:
        print(f"  没有找到 {monkey} 的数据")
        continue
    
    print(f"  找到 {len(monkey_psth_files)} 个sessions")
    
    # 第一步：先统计总神经元数，确定数组大小
    session_neurons = []
    for psth_file in monkey_psth_files:
        data = np.load(os.path.join(psth_dir, psth_file), mmap_mode='r')
        n_neurons = data.shape[0]
        session_neurons.append(n_neurons)
        del data
        gc.collect()
    
    total_neurons = sum(session_neurons)
    print(f"  总神经元数: {total_neurons}")
    
    # 获取单个session的形状
    data = np.load(os.path.join(psth_dir, monkey_psth_files[0]), mmap_mode='r')
    n_stimuli = data.shape[1] if data.ndim >= 2 else 1
    n_time = data.shape[2] if data.ndim >= 3 else (data.shape[1] if data.ndim == 2 else 1)
    del data
    gc.collect()
    
    print(f"  PSTH形状预分配: ({total_neurons}, {n_stimuli}, {n_time})")
    print(f"  预计内存占用: {total_neurons * n_stimuli * n_time * 4 / 1024**2:.1f} MB (float32)")
    
    # 第二步：预分配float32数组
    monkey_psth = np.zeros((total_neurons, n_stimuli, n_time), dtype=np.float32)
    monkey_df_list = []
    
    # 第三步：逐个session处理
    current_idx = 0
    for j, (psth_file, proc_file) in enumerate(zip(monkey_psth_files, monkey_processed_files)):
        print(f"    Session {j+1}/{len(monkey_psth_files)}: {psth_file} ({session_neurons[j]} neurons)")
        
        # memmap加载
        data = np.load(os.path.join(psth_dir, psth_file), mmap_mode='r')
        n_neurons = data.shape[0]
        monkey_psth[current_idx:current_idx+n_neurons] = data.astype(np.float32)
        del data
        gc.collect()
        
        # 处理processed数据
        proc_data = loadmat(os.path.join(processed_dir, proc_file))
        
        session_dict = {}
        for col in columns:
            if col in proc_data:
                data = proc_data[col]
                if data.ndim == 2 and data.shape[0] == 1:
                    session_dict[col] = data.flatten()
                else:
                    session_dict[col] = data.flatten() if data.ndim > 1 else data
        
        session_df = pd.DataFrame(session_dict)
        session_df['session_file'] = psth_file.replace('.npy', '')
        session_df['monkey'] = monkey
        
        # 维度检查
        n_df = len(session_df)
        if n_neurons != n_df:
            min_n = min(n_neurons, n_df)
            monkey_psth[current_idx:current_idx+min_n] = monkey_psth[current_idx:current_idx+n_neurons][:min_n]
            session_df = session_df.iloc[:min_n]
            n_neurons = min_n
        
        monkey_df_list.append(session_df)
        current_idx += n_neurons
        
        del proc_data, session_dict, session_df
        gc.collect()
    
    # 合并并保存
    print(f"  合并DataFrame...")
    monkey_df = pd.concat(monkey_df_list, ignore_index=True)
    del monkey_df_list
    gc.collect()
    
    print(f"  保存数据...")
    psth_path = os.path.join(output_dir, f"psth_{monkey}.npy")
    df_path = os.path.join(output_dir, f"processed_{monkey}.csv")
    
    np.save(psth_path, monkey_psth)
    monkey_df.to_csv(df_path, index=False)
    
    mem_usage = monkey_psth.nbytes / 1024**2
    print(f"  已保存: {psth_path}")
    print(f"  已保存: {df_path}")
    print(f"  形状: PSTH {monkey_psth.shape}, DataFrame {monkey_df.shape}")
    print(f"  实际内存占用: {mem_usage:.1f} MB")
    
    # 释放内存
    del monkey_psth, monkey_df
    gc.collect()
    print(f"  内存已释放")

print("\n" + "=" * 50)
print("所有猴子数据处理完成！")
print("=" * 50)
