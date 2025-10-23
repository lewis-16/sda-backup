#!/usr/bin/env python3
"""
Spike Detection PCA Visualization Script
随机选择3个group进行可视化验证，提取模型最后一层特征，进行PCA降维，创建散点图
"""

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import spikeinterface as si
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import sys
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import torch.nn.functional as F
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import time
import pickle
import networkx as nx
from probeinterface import write_probeinterface, read_probeinterface, Probe
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 检查CUDA可用性
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载数据
print("加载数据...")
spike_inf = pd.read_csv("/media/ubuntu/sda/duan/script/spike_sorting/spike_inf.tsv", index_col=0, sep='\t')
cluster_inf = pd.read_csv("/media/ubuntu/sda/duan/script/spike_sorting/cluster_inf.csv", index_col=0)

# 加载probe信息
probe = read_probeinterface("/media/ubuntu/sda/duan/script/spike_sorting/probe.json")

# 加载所有结果
print("加载训练结果...")
with open('/media/ubuntu/sda/duan/script/spike_sorting/all_results/all_results_summary.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"加载了 {len(all_results)} 个通道组合的结果")

# 从notebook中复制必要的函数
def detect_local_maxima_in_window(data, window_size=20, std_multiplier=2):
    """在每个滑动窗口范围内检测局部最大值的索引"""
    local_maxima_indices = []
    for row in data:
        maxima_indices = []
        row_std = np.std(row.astype(np.float32))
        threshold = std_multiplier * row_std
        for start in range(0, len(row), window_size):
            end = min(start + window_size, len(row))
            window = np.abs(row[start:end])
            if len(window) > 0:
                local_max_index = np.argmax(window)
                local_max_value = window[local_max_index]
                if local_max_value > threshold:
                    maxima_indices.append(start + local_max_index)  
        local_maxima_indices.extend(maxima_indices)
        local_maxima_indices = list(set(local_maxima_indices))  
    return local_maxima_indices

def label_array1_based_on_array2(array1, array2, threshold=5):
    """根据 array2 的值对 array1 进行标记"""
    sorted_array2 = np.sort(array2)
    labels = np.zeros(len(array1), dtype=int)
    for i, value in enumerate(array1):
        left = value - threshold
        right = value + threshold
        left_index = np.searchsorted(sorted_array2, left, side='left')
        right_index = np.searchsorted(sorted_array2, right, side='right')
        if right_index > left_index:
            labels[i] = 1
    return labels

class SpikeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Spike_Detection_MLP_with_features(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, n_channels, time_window):
        super(Spike_Detection_MLP_with_features, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()  
        self.n_channels = n_channels
        self.time_window = time_window
        
    def forward(self, x):
        x = x.reshape(-1, self.n_channels * self.time_window)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        features = x
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x, features
    
    def extract_features(self, x):
        """提取最后一层特征"""
        x = x.reshape(-1, self.n_channels * self.time_window)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return x

def create_visualization_for_group(channel_group_id, result_dir, recording_f, spike_inf):
    """为单个group创建可视化"""
    print(f"\n处理group: {channel_group_id}")
    
    # 加载保存的模型
    model_path = os.path.join(result_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    
    # 获取该group的结果信息
    result_info = all_results[channel_group_id]
    channels = result_info['channels']
    cluster_ids = result_info['cluster_ids']
    
    print(f"通道: {channels}")
    print(f"Cluster IDs: {cluster_ids}")
    
    # 重新处理数据以获取测试集
    print("重新处理数据...")
    
    # 获取对应clusters的spike数据
    spike_inf_temp = spike_inf[spike_inf['cluster_id'].isin(cluster_ids)]
    
    # 重新检测spike和提取时间窗（使用较小的数据量进行可视化）
    total_frames = 1200 * 30000
    chunk_size = 120000  
    window_size = 91
    half_window = window_size // 2
    
    # 只处理前几个chunk以节省时间
    max_chunks = 10
    all_valid_indices = []
    all_windows = []
    
    for i, start_frame in enumerate(tqdm(range(0, min(max_chunks * chunk_size, total_frames), chunk_size), desc="处理chunks")):
        end_frame = min(start_frame + chunk_size, total_frames)
        
        data_chunk = recording_f.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=channels
        )
        
        # 检测spike
        threshold_result = detect_local_maxima_in_window(
            data_chunk.T,  
            std_multiplier=1.5,
            window_size=30
        )
        
        # 调整时间戳到全局坐标系
        threshold_result = np.array(threshold_result) + start_frame
        valid_indices = threshold_result[
            (threshold_result >= start_frame + half_window + 1) & 
            (threshold_result < end_frame - half_window)
        ]
        
        # 提取时间窗
        for idx_val in valid_indices:
            rel_idx = idx_val - start_frame
            window = data_chunk.T[:, rel_idx-half_window : rel_idx+half_window+1]
            all_windows.append(window)
        
        all_valid_indices.extend(valid_indices)
    
    all_valid_indices = np.array(all_valid_indices)
    all_windows = np.stack(all_windows) if len(all_windows) > 0 else np.array([])
    
    print(f"检测到spike数量: {len(all_valid_indices)}")
    
    # 计算标签
    labels = label_array1_based_on_array2(all_valid_indices, spike_inf_temp['time'], threshold=5)
    
    # 平衡数据集
    indices_0 = np.where(labels == 0)[0] 
    indices_1 = np.where(labels == 1)[0] 
    
    target_0_count = len(indices_1)
    
    if len(indices_0) > target_0_count:
        sampled_indices_0 = np.random.choice(indices_0, target_0_count, replace=False)
    else:
        sampled_indices_0 = indices_0  
    
    final_indices = np.concatenate([sampled_indices_0, indices_1])
    np.random.shuffle(final_indices)
    
    sampled_windows = all_windows[final_indices]
    sampled_labels = labels[final_indices]
    
    # 创建数据集
    dataset = SpikeDataset(sampled_windows, sampled_labels)
    
    # 使用20%作为测试集
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # 创建模型并加载权重
    input_size = sampled_windows.shape[1] * sampled_windows.shape[2]
    model = Spike_Detection_MLP_with_features(input_size, 256, 64, 1, 
                                            n_channels=sampled_windows.shape[1], 
                                            time_window=sampled_windows.shape[2])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 提取特征和预测
    print("提取特征和预测...")
    all_features = []
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            
            # 提取特征
            features = model.extract_features(batch_data)
            outputs = model(batch_data)[0]  # 只取预测结果
            
            all_features.append(features.cpu().numpy())
            all_predictions.append((outputs > 0.5).float().cpu().numpy())
            all_true_labels.append(batch_labels.numpy())
    
    # 合并所有批次的结果
    features = np.vstack(all_features)
    predictions = np.vstack(all_predictions).flatten()
    true_labels = np.concatenate(all_true_labels)
    
    print(f"特征形状: {features.shape}")
    print(f"预测形状: {predictions.shape}")
    print(f"真实标签形状: {true_labels.shape}")
    
    # PCA降维
    print("进行PCA降维...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
    print(f"累计解释方差比: {np.sum(pca.explained_variance_ratio_)}")
    
    return {
        'channel_group_id': channel_group_id,
        'features_pca': features_pca,
        'true_labels': true_labels,
        'predictions': predictions,
        'pca_explained_variance': pca.explained_variance_ratio_,
        'channels': channels,
        'cluster_ids': cluster_ids
    }

def create_scatter_plots(visualization_data, output_path):
    """创建散点图可视化"""
    
    with PdfPages(output_path) as pdf:
        for i, data in enumerate(visualization_data):
            print(f"\n创建第 {i+1} 个group的可视化: {data['channel_group_id']}")
            
            # 创建图形，每页两个子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 提取数据
            features_pca = data['features_pca']
            true_labels = data['true_labels']
            predictions = data['predictions']
            
            # 子图1: Ground Truth
            ax1.scatter(features_pca[true_labels == 0, 0], 
                       features_pca[true_labels == 0, 1], 
                       c='red', alpha=0.6, s=20, label='Non-spike (0)')
            ax1.scatter(features_pca[true_labels == 1, 0], 
                       features_pca[true_labels == 1, 1], 
                       c='blue', alpha=0.6, s=20, label='Spike (1)')
            
            ax1.set_title(f'Ground Truth - Group {i+1}\n{data["channel_group_id"]}', 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel(f'PC1 ({data["pca_explained_variance"][0]:.1%} variance)', fontsize=10)
            ax1.set_ylabel(f'PC2 ({data["pca_explained_variance"][1]:.1%} variance)', fontsize=10)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # 子图2: Predicted Labels
            ax2.scatter(features_pca[predictions == 0, 0], 
                       features_pca[predictions == 0, 1], 
                       c='red', alpha=0.6, s=20, label='Non-spike (0)')
            ax2.scatter(features_pca[predictions == 1, 0], 
                       features_pca[predictions == 1, 1], 
                       c='blue', alpha=0.6, s=20, label='Spike (1)')
            
            ax2.set_title(f'Predicted Labels - Group {i+1}\n{data["channel_group_id"]}', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel(f'PC1 ({data["pca_explained_variance"][0]:.1%} variance)', fontsize=10)
            ax2.set_ylabel(f'PC2 ({data["pca_explained_variance"][1]:.1%} variance)', fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            total_samples = len(true_labels)
            true_spikes = np.sum(true_labels == 1)
            pred_spikes = np.sum(predictions == 1)
            accuracy = np.mean(true_labels == predictions)
            
            # 在图上添加统计信息
            stats_text = f'Total: {total_samples}\nTrue spikes: {true_spikes}\nPred spikes: {pred_spikes}\nAccuracy: {accuracy:.3f}'
            fig.text(0.02, 0.02, stats_text, fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Group {i+1} 可视化完成")
            print(f"  通道: {data['channels']}")
            print(f"  Cluster IDs: {data['cluster_ids']}")
            print(f"  总样本数: {total_samples}")
            print(f"  真实spike数: {true_spikes}")
            print(f"  预测spike数: {pred_spikes}")
            print(f"  准确率: {accuracy:.3f}")
            print(f"  PCA解释方差比: {data['pca_explained_variance']}")
    
    print(f"\n所有可视化已保存到: {output_path}")

def main():
    """主函数"""
    print("=== Spike Detection PCA Visualization ===")
    
    # 随机选择3个group
    all_channel_groups = list(all_results.keys())
    selected_groups = random.sample(all_channel_groups, min(3, len(all_channel_groups)))
    
    print(f"随机选择的3个group:")
    for i, group in enumerate(selected_groups):
        print(f"{i+1}. {group}")
    
    # 加载recording数据（这里需要根据实际情况调整路径）
    print("\n加载recording数据...")
    try:
        # 尝试加载recording数据
        recording_raw = se.read_intan("/media/ubuntu/sda/duan/raw_data/M190011_250521_141514_merged_130.rhd", stream_id='0')
        recording_raw = spre.unsigned_to_signed(recording_raw)
        recording_recorded = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)
        recording_recorded = spre.notch_filter(recording_recorded, freq=50)
        recording_f = spre.common_reference(recording_recorded, reference="global", operator="median")
        recording_f = recording_f.set_probegroup(probe)
        print("Recording数据加载成功")
    except Exception as e:
        print(f"无法加载recording数据: {e}")
        print("将使用模拟数据进行演示...")
        # 创建模拟数据用于演示
        recording_f = None
    
    # 为所有选中的group创建可视化数据
    visualization_data = []
    
    for group in selected_groups:
        # 构建结果目录路径
        group_clean = group.replace(' ', '').replace('[', '').replace(']', '')
        result_dir = f'/media/ubuntu/sda/duan/script/spike_sorting/all_results/channels_{group_clean}'
        
        if recording_f is not None:
            data = create_visualization_for_group(group, result_dir, recording_f, spike_inf)
            if data is not None:
                visualization_data.append(data)
        else:
            print(f"跳过group {group}，因为无法加载recording数据")
    
    # 创建可视化
    if len(visualization_data) > 0:
        output_path = '/media/ubuntu/sda/duan/figure/spike_detection_pca_visualization.pdf'
        create_scatter_plots(visualization_data, output_path)
        print(f"\n可视化完成！结果保存在: {output_path}")
    else:
        print("没有可用的可视化数据")

if __name__ == "__main__":
    main()
