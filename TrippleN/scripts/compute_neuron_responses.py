#!/usr/bin/env python3
"""
计算神经元对图像的平均放电率
读取PSTH数据，根据best_r_time1和best_r_time2确定响应窗口
输出形状: (n_neurons, n_stimuli)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


NUM_WORKERS = 30  # 并行进程数


def _compute_neuron_batch(args):
    """计算一批神经元的响应值（用于并行处理）"""
    psth_data_path, neuron_indices, start_time_arr, end_time_arr, n_stimuli = args
    
    psth_data = np.load(psth_data_path, mmap_mode='r')
    n_neurons = len(neuron_indices)
    n_time = psth_data.shape[2]
    
    batch_responses = np.zeros((n_neurons, n_stimuli), dtype=np.float32)
    
    for i, neuron_idx in enumerate(neuron_indices):
        start_time = int(start_time_arr[neuron_idx])
        end_time = int(end_time_arr[neuron_idx])
        
        # 处理时间窗口越界情况
        if start_time < 0:
            start_time = 0
        if end_time > n_time:
            end_time = n_time
        if end_time <= start_time:
            # 时间窗口无效，使用整个时间范围的中点作为默认窗口
            mid_time = n_time // 2
            start_time = max(0, mid_time - 5)
            end_time = min(n_time, mid_time + 5)
        
        psth_neuron = psth_data[neuron_idx, :n_stimuli, start_time:end_time]
        batch_responses[i] = np.mean(psth_neuron, axis=1)
    
    return batch_responses


def calculate_neuron_responses(psth_path, unit_info_path, n_stimuli=1000):
    """计算所有神经元对图像的平均放电率（30进程并行）
    
    Parameters:
    -----------
    psth_path : str
        PSTH数据文件路径
    unit_info_path : str
        神经元信息文件路径 (.pkl)
    n_stimuli : int
        使用的图像数量
    
    Returns:
    --------
    neuron_responses : np.ndarray
        形状为 (n_neurons, n_stimuli) 的响应矩阵
    unit_info : pd.DataFrame
        神经元信息DataFrame
    """
    print("="*60)
    print("计算神经元响应值")
    print("="*60)
    
    start_time = datetime.now()
    
    print("\n[1/3] 加载数据...")
    psth_data = np.load(psth_path, mmap_mode='r')
    unit_info = pd.read_pickle(unit_info_path)
    
    n_neurons = psth_data.shape[0]
    psth_data_path = psth_data.filename
    
    print(f"  PSTH数据形状: {psth_data.shape}")
    print(f"  单元信息记录数: {len(unit_info)}")
    
    best_r_time1 = unit_info['best_r_time1'].values
    best_r_time2 = unit_info['best_r_time2'].values
    
    print("\n[2/3] 计算神经元响应...")
    print(f"  使用 {NUM_WORKERS} 个进程并行...")
    
    # 检查NaN情况
    nan_check = np.any(np.isnan(psth_data[:, :n_stimuli, :]))
    print(f"  PSTH数据中是否包含NaN: {nan_check}")
    
    # 检查时间窗口有效性
    invalid_windows = np.sum(best_r_time2 <= best_r_time1)
    if invalid_windows > 0:
        print(f"  警告: 发现 {invalid_windows} 个无效时间窗口（end <= start），将使用默认窗口")
    
    # 准备批次
    neuron_indices = np.arange(n_neurons)
    batch_size = (n_neurons + NUM_WORKERS - 1) // NUM_WORKERS
    batches = []
    
    for i in range(0, n_neurons, batch_size):
        batch_indices = neuron_indices[i:i+batch_size]
        batches.append((psth_data_path, batch_indices, best_r_time1, best_r_time2, n_stimuli))
    
    print(f"  分为 {len(batches)} 个批次")
    
    # 并行处理
    pbar = tqdm(
        total=len(batches),
        desc="  计算响应",
        unit="批次",
        ncols=80
    )
    
    with Pool(processes=NUM_WORKERS) as pool:
        results = []
        for result in pool.imap_unordered(_compute_neuron_batch, batches):
            results.append(result)
            pbar.update(1)
    
    pbar.close()
    
    # 合并结果
    neuron_responses = np.vstack(results)
    
    # 最终检查NaN
    nan_count = np.sum(np.isnan(neuron_responses))
    if nan_count > 0:
        raise ValueError(f"错误: 神经元响应矩阵中发现 {nan_count} 个NaN值！")
    
    print(f"  神经元响应矩阵形状: {neuron_responses.shape}")
    
    # 统计信息
    print(f"  响应值范围: [{neuron_responses.min():.4f}, {neuron_responses.max():.4f}]")
    print(f"  响应值均值: {neuron_responses.mean():.4f}")
    print(f"  响应值标准差: {neuron_responses.std():.4f}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n[3/3] 完成，耗时: {duration}")
    print("="*60)
    
    return neuron_responses, unit_info


def main():
    """主函数"""
    psth_path = '/media/ubuntu/sda/TrippleN/customize/aggregate_response/all_subjects_psth.npy'
    unit_info_path = '/media/ubuntu/sda/TrippleN/customize/aggregate_response/all_subjects_unit_info.pkl'
    output_path = '/media/ubuntu/sda/TrippleN/customize/neuron_responses_1000.npy'
    
    # 计算响应
    neuron_responses, unit_info = calculate_neuron_responses(
        psth_path, unit_info_path, n_stimuli=1000
    )
    
    # 保存
    print(f"\n保存神经元响应到: {output_path}")
    np.save(output_path, neuron_responses)
    print("保存完成!")
    
    return neuron_responses, unit_info


if __name__ == '__main__':
    main()
