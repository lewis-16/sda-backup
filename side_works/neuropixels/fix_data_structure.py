#!/usr/bin/env python3
"""
修复数据结构：每个data样本包含所有session的神经元对同一次image刺激实验的联合反应
"""

import numpy as np
import pandas as pd
import os
import pickle
import torch
from torch.utils.data import Dataset

def safe_spike_binning(spike_times, start_time, end_time, bin_width_ms, n_timesteps):
    """
    安全的spike二值化函数，避免索引越界错误
    """
    # 过滤时间窗口内的spikes
    mask = (spike_times >= start_time) & (spike_times <= end_time)
    filtered_spikes = spike_times[mask]
    
    if len(filtered_spikes) == 0:
        return np.zeros(n_timesteps, dtype=np.int8)
    
    # 转换为相对时间（毫秒）
    relative_spikes = (filtered_spikes - start_time) * 1000
    
    # 转换为bin索引
    spike_bins = (relative_spikes / bin_width_ms).astype(int)
    
    # 确保索引在有效范围内
    spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < n_timesteps)]
    
    # 创建二值化时间序列
    binary_timeline = np.zeros(n_timesteps, dtype=np.int8)
    if len(spike_bins) > 0:
        binary_timeline[spike_bins] = 1
    
    return binary_timeline

def create_correct_visp_dataset(bin_width_ms=1, stimulus_duration_ms=250):
    """
    创建正确的VISp数据集：
    每个data样本 = 所有session的神经元对同一次image刺激实验的联合反应
    """
    
    # 计算时间步数
    n_timesteps = int(stimulus_duration_ms / bin_width_ms)
    print(f"时间步数: {n_timesteps}")
    
    # 获取session列表
    session_folder = os.listdir("/media/ubuntu/sda/neuropixels/output_dir")
    session_list = []
    for session in session_folder:
        if 'session_' in session:
            session_list.append(session)
    
    print(f"找到 {len(session_list)} 个session")
    
    # 第一步：预加载所有session数据
    print("预加载所有session数据...")
    session_data = {}
    
    for session in session_list:
        output_dir = f'/media/ubuntu/sda/neuropixels/output_dir/{session}'
        
        try:
            # 读取数据
            cluster_inf = pd.read_csv(f'{output_dir}/cluster_inf.csv')
            spike_inf = pd.read_csv(f'{output_dir}/spike_inf.csv', index_col=0)
            stimulus_table = pd.read_csv(f'{output_dir}/stimulus_table.csv')
            
            # 预处理
            stimulus_table = stimulus_table[stimulus_table['frame'] != -1.0]
            stimulus_table['frame'] = stimulus_table['frame'].astype(int)
            
            # 过滤低频神经元
            filtered_spikes = spike_inf['id'].value_counts()
            filtered_spikes = filtered_spikes[filtered_spikes > 30000].index
            cluster_inf = cluster_inf[cluster_inf['unit_id'].isin(filtered_spikes)]
            
            # 只保留VISp区域的神经元
            visp_cluster_inf = cluster_inf[cluster_inf['ecephys_structure_acronym'] == 'VISp']
            
            # 预过滤spike数据
            spike_inf_filtered = spike_inf[spike_inf['id'].isin(visp_cluster_inf['unit_id'])]
            
            # 预索引spike数据
            neuron_spikes = {}
            for neuron_id in visp_cluster_inf['unit_id']:
                neuron_data = spike_inf_filtered[spike_inf_filtered['id'] == neuron_id]
                if not neuron_data.empty:
                    neuron_spikes[neuron_id] = neuron_data['time'].values
            
            session_data[session] = {
                'cluster_inf': visp_cluster_inf,
                'stimulus_table': stimulus_table,
                'neuron_spikes': neuron_spikes
            }
            
            print(f"Session {session}: VISp神经元 {len(visp_cluster_inf)}")
            
        except Exception as e:
            print(f"处理session {session}时出错: {e}")
            continue
    
    # 第二步：收集所有VISp神经元信息（跨所有session）
    all_neuron_info = []
    neuron_id_to_index = {}
    
    for session, data in session_data.items():
        for _, row in data['cluster_inf'].iterrows():
            neuron_id = row['unit_id']
            full_neuron_id = f'{neuron_id}_{session}'
            
            if full_neuron_id not in neuron_id_to_index:
                neuron_id_to_index[full_neuron_id] = len(all_neuron_info)
                all_neuron_info.append({
                    'neuron_id': neuron_id,
                    'session': session,
                    'full_id': full_neuron_id,
                    'structure': 'VISp'
                })
    
    n_neurons = len(all_neuron_info)
    print(f"所有session的VISp神经元总数: {n_neurons}")
    
    # 第三步：创建联合实验数据集
    print("创建联合实验数据集...")
    
    all_samples = []
    all_labels = []
    all_metadata = []
    
    # 为每张图像创建联合实验
    for image_id in range(118):
        if image_id % 10 == 0:
            print(f"处理图像 {image_id+1}/118...")
        
        # 收集该图像在所有session中的实验
        image_experiments = []
        
        for session in session_list:
            if session not in session_data:
                continue
                
            data = session_data[session]
            image_stimuli = data['stimulus_table'][data['stimulus_table']['frame'] == image_id]
            
            # 检查列名和是否有数据
            if image_stimuli.empty:
                continue
                
            if 'stop_time' not in image_stimuli.columns:
                print(f"警告: session {session} 的stimulus_table没有'stop_time'列")
                print(f"可用列: {list(image_stimuli.columns)}")
                continue
            
            for _, stimulus_row in image_stimuli.iterrows():
                image_experiments.append({
                    'session': session,
                    'start_time': stimulus_row['start_time'],
                    'end_time': stimulus_row['stop_time']
                })
        
        # 为每个实验创建联合神经元反应矩阵
        for exp_idx, experiment in enumerate(image_experiments):
            session = experiment['session']
            start_time = experiment['start_time']
            end_time = experiment['end_time']
            
            # 创建联合反应矩阵 (n_neurons, n_timesteps)
            joint_matrix = np.zeros((n_neurons, n_timesteps), dtype=np.int8)
            
            # 填充所有session的神经元反应
            for neuron_idx, neuron_info in enumerate(all_neuron_info):
                neuron_id = neuron_info['neuron_id']
                neuron_session = neuron_info['session']
                
                # 只处理当前实验session的神经元
                if neuron_session == session and neuron_id in session_data[session]['neuron_spikes']:
                    spike_times = session_data[session]['neuron_spikes'][neuron_id]
                    
                    # 使用安全的二值化函数
                    binary_timeline = safe_spike_binning(
                        spike_times, start_time, end_time, bin_width_ms, n_timesteps
                    )
                    joint_matrix[neuron_idx] = binary_timeline
            
            # 检查是否有任何神经元有spike
            if np.sum(joint_matrix) > 0:
                all_samples.append(joint_matrix)
                all_labels.append(image_id)
                all_metadata.append({
                    'image_id': image_id,
                    'session': session,
                    'experiment_index': exp_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'n_spikes': np.sum(joint_matrix),
                    'n_active_neurons': np.sum(np.sum(joint_matrix, axis=1) > 0)
                })
    
    # 转换为numpy数组
    if len(all_samples) > 0:
        data = np.array(all_samples)
        labels = np.array(all_labels)
        
        print(f"联合VISp数据集形状: {data.shape}")
        print(f"总样本数: {len(all_samples)}")
        print(f"每个样本包含 {n_neurons} 个神经元（来自所有session）")
        print(f"每个样本的时间步数: {n_timesteps}")
        
        return data, labels, all_neuron_info, all_metadata
    else:
        print("没有找到有效的样本数据")
        return None, None, None, None

def main():
    """主函数"""
    print("开始创建正确的联合VISp数据集...")
    
    try:
        data, labels, neuron_info, metadata = create_correct_visp_dataset()
        
        if data is not None:
            save_dir = "/media/ubuntu/sda/neuropixels/correct_visp_dataset"
            os.makedirs(save_dir, exist_ok=True)
            
            np.save(os.path.join(save_dir, "correct_visp_data.npy"), data)
            np.save(os.path.join(save_dir, "correct_visp_labels.npy"), labels)
            
            with open(os.path.join(save_dir, "correct_visp_neuron_info.pkl"), 'wb') as f:
                pickle.dump(neuron_info, f)
            
            with open(os.path.join(save_dir, "correct_visp_metadata.pkl"), 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"\n数据集统计:")
            print(f"数据形状: {data.shape}")
            print(f"标签形状: {labels.shape}")
            print(f"神经元信息数量: {len(neuron_info)}")
            print(f"元数据数量: {len(metadata)}")
            
            # 验证数据结构
            print(f"\n数据结构验证:")
            print(f"每个样本的神经元数量: {data.shape[1]}")
            print(f"每个样本的时间步数: {data.shape[2]}")
            print(f"标签范围: {labels.min()} - {labels.max()}")
            
            # 检查样本中活跃神经元分布
            active_neurons_per_sample = np.sum(np.sum(data, axis=2) > 0, axis=1)
            print(f"每样本活跃神经元数 - 平均: {active_neurons_per_sample.mean():.2f}, 范围: {active_neurons_per_sample.min()}-{active_neurons_per_sample.max()}")
            
        else:
            print("数据集创建失败")
            
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
