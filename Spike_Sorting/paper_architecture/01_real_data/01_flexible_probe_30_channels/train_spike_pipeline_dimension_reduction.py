#!/usr/bin/env python
# coding: utf-8
"""
训练Spike Detection模型和计算wPCA/Templates的脚本
- 使用PC特征进行降维
- 计算PC空间的templates用于template matching
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm
import sys
import scipy.signal
import time
import json

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from scipy.signal import find_peaks

# 添加spike_detection路径以导入函数
sys.path.append('/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/spike_detection')
from train_models import (
    detect_local_minimum_in_window as _detect_local_minimum_in_window,
    label_array1_based_on_array2,
    extract_windows
)

# ==================== Dataset Classes ====================
class SimpleDetectionDataset(Dataset):
    """简单的Detection数据集"""
    def __init__(self, waveforms, labels):
        self.waveforms = torch.FloatTensor(waveforms)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        return self.waveforms[idx], self.labels[idx]


class ClassificationDataset(Dataset):
    """Classification数据集"""
    def __init__(self, waveforms, labels):
        self.waveforms = torch.FloatTensor(waveforms)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        return self.waveforms[idx], self.labels[idx]


# ==================== Multi-task Learning Model ====================
class SimpleClassifier(nn.Module):
    """简单的分类器，用于多任务学习（参考autosort的clssimp）"""
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        # 添加AdaptiveAvgPool1d层（参考autosort）
        self.pool = nn.AdaptiveAvgPool1d(output_size=input_size)
        self.way1 = nn.Sequential(
            nn.Linear(input_size, 1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )
        self.way2 = nn.Sequential(
            nn.Linear(1000, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.way3 = nn.Sequential(
            nn.Linear(512, 100, bias=True),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(100, num_classes, bias=True)
    
    def forward(self, x):
        # 参考autosort的forward方式：先pooling再reshape
        x = self.pool(x[None, :])  # (batch, input_size) -> (1, batch, input_size) -> (1, batch, input_size)
        x = x.reshape(x.size(1), -1)  # (batch, input_size)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        logits = self.cls(x)
        return logits
    
    def intermediate_forward(self, x):
        """提取中间层特征（用于UMAP可视化）"""
        x = self.pool(x[None, :])
        x = x.reshape(x.size(1), -1)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        return x


# ==================== Noise Classifier Model ====================
class NoiseClassifier(nn.Module):
    """
    Noise分类器：只用于noise/spike分类
    """
    def __init__(self, input_size, device):
        super(NoiseClassifier, self).__init__()
        self.clsfier_noise = SimpleClassifier(input_size, 2).to(device)  # noise/spike分类
        self.device = device
    
    def forward(self, x):
        """
        前向传播
        
        x: 输入特征，shape: (batch, input_size)
        """
        noise_output = self.clsfier_noise(x)
        return noise_output
    
    def get_intermediate_features(self, x):
        """
        提取中间层特征（用于UMAP可视化）
        x: 输入特征，shape: (batch, input_size)
        返回: noise_features
        """
        noise_features = self.clsfier_noise.intermediate_forward(x)
        return noise_features


class NoiseClassifierDataset(Dataset):
    """
    Noise分类器数据集
    """
    def __init__(self, waveforms, noise_labels):
        """
        参数:
        waveforms: numpy.ndarray, shape (n_samples, ...) - waveform特征
        noise_labels: numpy.ndarray, shape (n_samples,), 0=noise, 1=spike
        """
        self.waveforms = torch.FloatTensor(waveforms)
        
        # 将noise_labels转换为one-hot格式 (n_samples, 2): [noise, spike]
        noise_labels_tensor = torch.LongTensor(noise_labels)
        self.noise_labels_onehot = torch.zeros((len(noise_labels), 2), dtype=torch.float32)
        self.noise_labels_onehot[torch.arange(len(noise_labels)), noise_labels_tensor] = 1.0
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        return {
            'waveform': self.waveforms[idx],
            'noise_label': self.noise_labels_onehot[idx]  # one-hot格式
        }


# ==================== Detection Functions (from train_models.py) ====================
def detect_local_minimum_in_window(data, window_size=20, std_multiplier=2, min_distance=None):
    """
    带进度条的detect_local_minimum_in_window包装函数
    期望输入: (n_channels, time_points) 或 (time_points, n_channels)
    会自动检测并转置为 (n_channels, time_points)
    """
    if min_distance is None:
        min_distance = max(1, window_size // 2)
    
    # 确保数据形状为 (n_channels, time_points)
    # 如果第一个维度大于第二个，说明是 (time_points, n_channels)，需要转置
    if data.ndim > 1 and data.shape[0] > data.shape[1]:
        data = data.T  # 转置为 (n_channels, time_points)
    
    local_minima_indices = []
    n_channels = data.shape[0] if data.ndim > 1 else 1
    
    # 如果只有一个通道，直接调用原函数
    if n_channels == 1:
        return _detect_local_minimum_in_window(data, window_size, std_multiplier, min_distance)
    
    # 对每个通道进行处理，添加进度条
    # data 现在是 (n_channels, time_points)，每个 row 是一个通道的时间序列
    for row_idx, row in enumerate(tqdm(data, desc="Processing channels", unit="ch", leave=False, total=n_channels)):
        row = row.astype(np.float32)
        row_mean = np.mean(row)
        row_std = np.std(row)
        threshold = row_mean - std_multiplier * row_std
        
        # 反转信号以检测最小值（find_peaks 检测最大值）
        inverted_row = -row
        
        # 使用 find_peaks 检测峰值（对应原信号的最小值）
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            inverted_row,
            height=-threshold,  # 反转后的阈值
            distance=min_distance
        )
        
        # 在每个滑动窗口内只保留一个峰值
        windowed_peaks = []
        for start in range(0, len(row), window_size):
            end = min(start + window_size, len(row))
            
            # 找到在当前窗口内的所有峰值
            window_peaks = peaks[(peaks >= start) & (peaks < end)]
            
            if len(window_peaks) > 0:
                # 如果窗口内有多个峰值，选择最显著的（即原信号中最小值最小的）
                if len(window_peaks) > 1:
                    # 找到原信号中值最小的峰值
                    peak_values = row[window_peaks]
                    min_peak_idx = np.argmin(peak_values)
                    selected_peak = window_peaks[min_peak_idx]
                else:
                    selected_peak = window_peaks[0]
                
                # 验证是否满足阈值条件
                if row[selected_peak] < threshold:
                    windowed_peaks.append(int(selected_peak))
        
        local_minima_indices.extend(windowed_peaks)
    
    # 去重并排序
    local_minima_indices = sorted(list(set(local_minima_indices)))

    return local_minima_indices


def detect_spike_from_train_models(trace0_car, std_multiplier=2.4, window_size=10, chunk_size=300000):
    """
    使用train_models.py中的detect_local_minimum_in_window进行spike检测（分chunk处理）
    
    参数:
    trace0_car : numpy.ndarray
        输入数据，形状为 (time_points, n_channels)
    std_multiplier : float
        标准差的倍数，用于筛选局部最小值，默认为2.4
    window_size : int
        滑动窗口的大小，默认为10
    chunk_size : int
        每个chunk的大小（采样点数），默认为300000
    
    返回:
    spike_times : numpy.ndarray
        检测到的spike时间点索引（所有通道合并）
    spike_channels : numpy.ndarray
        每个spike对应的通道号（通过找到最小值所在的通道确定）
    """
    # 确保输入形状正确：(time_points, n_channels)
    if trace0_car.shape[0] < trace0_car.shape[1]:
        trace0_car = trace0_car.T
        print(f"[WARNING] Input shape was (n_channels, time_points), transposed to (time_points, n_channels): {trace0_car.shape}")
    
    total_frames = trace0_car.shape[0]
    all_spike_times = []
    
    # 计算总chunk数用于进度条
    total_chunks = (total_frames + chunk_size - 1) // chunk_size
    print(f"[INFO] Detecting spikes using detect_local_minimum_in_window (chunk-based)...")
    print(f"[INFO] Total frames: {total_frames:,} | Chunk size: {chunk_size:,} | Total chunks: {total_chunks}")
    
    # 分chunk处理，添加进度条
    pbar = tqdm(range(0, total_frames, chunk_size), 
                desc="Detecting spikes", 
                unit="chunk",
                total=total_chunks)
    
    for start_frame in pbar:
        end_frame = min(start_frame + chunk_size, total_frames)
        
        # 提取当前chunk的数据
        data_chunk = trace0_car[start_frame:end_frame, :]  # shape: (chunk_size, n_channels)
        
        # 对当前chunk进行检测
        # 原始函数期望 (n_channels, time_points)，所以需要转置
        # 使用原始函数，确保逻辑一致
        chunk_spike_times = _detect_local_minimum_in_window(
            data_chunk.T,  # 转置为 (n_channels, chunk_size)
            window_size=window_size,
            std_multiplier=std_multiplier
        )
        
        # 更新进度条信息，显示当前chunk检测到的spike数量
        pbar.set_postfix({'spikes_in_chunk': len(chunk_spike_times)})
        
        # 将chunk内的相对时间转换为全局时间
        chunk_spike_times = np.array(chunk_spike_times, dtype=np.int64) + start_frame
        
        # 过滤边界附近的spike（避免跨chunk边界的问题）
        valid_mask = (chunk_spike_times >= start_frame) & (chunk_spike_times < end_frame)
        chunk_spike_times = chunk_spike_times[valid_mask]
        
        all_spike_times.extend(chunk_spike_times.tolist())
    
    # 去重并排序
    all_spike_times = np.array(sorted(list(set(all_spike_times))), dtype=np.int64)
    
    # 为每个检测到的spike找到对应的通道（最小值所在的通道）
    print("[INFO] Determining channels for detected spikes...")
    spike_channels = []
    for t in tqdm(all_spike_times, desc="Finding channels", unit="spike", leave=False):
        # 找到该时间点所有通道中的最小值所在通道
        channel_values = trace0_car[t, :]
        min_channel = np.argmin(channel_values)
        spike_channels.append(min_channel)
    
    spike_channels = np.array(spike_channels, dtype=np.int64)
    
    print(f"[INFO] Detected {len(all_spike_times):,} spikes across all channels")
    
    return all_spike_times, spike_channels


def map_gt_annotation(detect_times, gt_times, time_tolerance=1):
    """
    将检测结果与ground truth进行匹配（只匹配时间，不考虑通道）
    使用最近邻匹配策略，确保每个GT spike最多只匹配一个检测到的spike
    
    参数:
    detect_times : numpy.ndarray
        检测结果的时间点，形状为 (n_detected_spikes,)
    gt_times : numpy.ndarray
        Ground truth的时间点，形状为 (n_gt_spikes,)
    time_tolerance : int
        时间容差（采样点数），默认为1（允许±1的误差）
    
    返回:
    gt_label_array1 : numpy.ndarray
        每个检测结果对应的GT索引（-1表示未匹配）
    """
    n_detect = len(detect_times)
    n_gt = len(gt_times)
    gt_label_array1 = np.full(n_detect, -1, dtype=np.int64)
    
    if n_detect == 0 or n_gt == 0:
        return gt_label_array1
    
    # 确保数据类型为整数
    detect_times = detect_times.astype(np.int64)
    gt_times = gt_times.astype(np.int64)
    
    # 对GT时间点进行排序以便快速查找
    sorted_gt_indices = np.argsort(gt_times)
    sorted_gt_times = gt_times[sorted_gt_indices]
    
    # 记录每个GT spike是否已被匹配
    matched_gt_indices = set()
    
    # 对检测到的spike按时间排序，优先匹配更接近的spike
    sorted_detect_indices = np.argsort(detect_times)
    
    # 对于每个检测到的spike，找到最近的GT spike
    for detect_idx in sorted_detect_indices:
        detect_time = detect_times[detect_idx]
        
        # 使用二分搜索找到最近的GT spike
        nearest_idx = np.searchsorted(sorted_gt_times, detect_time, side='left')
        
        # 检查左右两个候选位置（考虑时间容差）
        candidates = []
        
        # 检查左侧候选
        if nearest_idx > 0:
            left_idx = nearest_idx - 1
            left_time = sorted_gt_times[left_idx]
            distance = abs(left_time - detect_time)
            if distance <= time_tolerance:
                candidates.append((left_idx, left_time, distance))
        
        # 检查右侧候选
        if nearest_idx < len(sorted_gt_times):
            right_idx = nearest_idx
            right_time = sorted_gt_times[right_idx]
            distance = abs(right_time - detect_time)
            if distance <= time_tolerance:
                candidates.append((right_idx, right_time, distance))
        
        # 如果找到候选，选择距离最近的且未被匹配的GT spike
        if candidates:
            # 按距离排序，选择最近的
            candidates.sort(key=lambda x: x[2])
            for gt_sorted_idx, gt_time, distance in candidates:
                # 将排序后的索引转换为原始索引
                gt_original_idx = sorted_gt_indices[gt_sorted_idx]
                
                # 如果该GT spike未被匹配，则匹配
                if gt_original_idx not in matched_gt_indices:
                    gt_label_array1[detect_idx] = gt_original_idx
                    matched_gt_indices.add(gt_original_idx)
                    break  # 找到匹配后退出
    
    return gt_label_array1


# ==================== Spike Classification Model ====================
class Spike_Classification_MLP(nn.Module):
    """
    Spike Classification MLP模型
    包含特征提取层和分类头
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, proj_dim=128):
        super(Spike_Classification_MLP, self).__init__()
        # 特征提取层
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        
        # 投影头：用于contrastive learning（可选）
        self.projection = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, proj_dim)
        )
        
        # 分类头
        self.fc3 = nn.Linear(hidden_size2, num_classes)
    
    def forward(self, x, mode='train'):
        """
        Args:
            x: 输入数据
            mode: 'train' 训练模式返回投影特征和logits
                  'eval' 评估模式只返回特征（fc2的输出）
        """
        x = x.reshape(-1, 31 * 30)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        features = self.relu2(x)  # 这是用于聚类的特征（hidden_size2维）
        
        if mode == 'train':
            # 训练时返回：投影特征（用于对比学习）、logits（用于分类）、原始特征
            proj_features = self.projection(features)
            logits = self.fc3(features)
            return proj_features, logits, features
        else:
            # 评估/推理时只返回特征向量
            return features


def cluster_label_array1_based_on_array2(array1, array2, threshold=5, cluster_column='cluster'):
    """
    根据 array2 的 'time' 和 'cluster' 对 array1 进行标记。
    """
    array2 = np.array((array2['time'], array2[cluster_column])).T
    sorted_indices = np.argsort(array2[:, 0])
    sorted_array2 = array2[sorted_indices]
    
    labels = -np.ones(len(array1), dtype=int)
    
    for i, value in enumerate(array1):
        left = value - threshold
        right = value + threshold
        
        left_index = np.searchsorted(sorted_array2[:, 0], left, side='left')
        right_index = np.searchsorted(sorted_array2[:, 0], right, side='right')
        
        if right_index > left_index:
            labels[i] = sorted_array2[left_index, 1]
    
    return labels


# ==================== wPCA and Templates Computation ====================
def compute_wPCA_from_train_data(df, n_pcs=6, output_dir=None, 
                                  include_noise_ratio=0.0, 
                                  include_unmatched_spikes=True):
    """
    从训练数据中计算wPCA（类似Kilosort的extract_wPCA_wTEMP）
    
    参数:
    df: DataFrame，包含waveform和mapping列
    n_pcs: PC特征数量，默认6
    output_dir: 保存wPCA的目录
    include_noise_ratio: 包含noise的比例（0.0-1.0），默认0.0
    include_unmatched_spikes: 是否包含未匹配但检测到的spike，默认True
    
    返回:
    wPCA: numpy.ndarray, shape (n_pcs, nt) = (6, 30)
    """
    print("\n" + "="*60)
    print("Computing wPCA from Training Data")
    print("="*60)
    print(f"[INFO] Strategy: GT spikes + {'unmatched spikes' if include_unmatched_spikes else 'no unmatched spikes'} + {'noise' if include_noise_ratio > 0 else 'no noise'}")
    
    # 提取所有waveform
    all_waveforms = np.stack(df['waveform'].values)  # (n_samples, 30, 30)
    
    # 分层选择数据
    clips_list = []
    
    # 1. 主要使用GT spike（mapping >= 0）
    gt_spike_mask = df['mapping'] >= 0
    gt_spike_waveforms = all_waveforms[gt_spike_mask]
    print(f"[INFO] GT spikes: {len(gt_spike_waveforms):,}")
    
    # 2. 可选：包含未匹配但检测到的spike
    unmatched_spike_waveforms = None
    if include_unmatched_spikes:
        unmatched_mask = df['mapping'] == -1
        unmatched_waveforms = all_waveforms[unmatched_mask]
        
        if len(unmatched_waveforms) > 0:
            # 筛选：只保留能量较高的候选
            energies = np.sum(unmatched_waveforms**2, axis=(1, 2))
            energy_threshold = np.percentile(energies, 75)
            high_energy_mask = energies >= energy_threshold
            unmatched_spike_waveforms = unmatched_waveforms[high_energy_mask]
            print(f"[INFO] High-quality unmatched spikes: {len(unmatched_spike_waveforms):,} (from {len(unmatched_waveforms):,} total unmatched)")
    
    # 3. 可选：包含少量noise
    noise_waveforms = None
    if include_noise_ratio > 0:
        noise_mask = df['mapping'] == -1
        noise_waveforms_all = all_waveforms[noise_mask]
        
        if include_unmatched_spikes and unmatched_spike_waveforms is not None:
            energies = np.sum(noise_waveforms_all**2, axis=(1, 2))
            low_energy_mask = energies < np.percentile(energies, 75)
            noise_candidates = noise_waveforms_all[low_energy_mask]
            n_noise = int(len(gt_spike_waveforms) * include_noise_ratio)
            if len(noise_candidates) > n_noise:
                noise_indices = np.random.choice(len(noise_candidates), n_noise, replace=False)
                noise_waveforms = noise_candidates[noise_indices]
            else:
                noise_waveforms = noise_candidates
        else:
            n_noise = int(len(gt_spike_waveforms) * include_noise_ratio)
            if len(noise_waveforms_all) > n_noise:
                noise_indices = np.random.choice(len(noise_waveforms_all), n_noise, replace=False)
                noise_waveforms = noise_waveforms_all[noise_indices]
            else:
                noise_waveforms = noise_waveforms_all
        print(f"[INFO] Noise samples: {len(noise_waveforms):,} ({include_noise_ratio*100:.1f}% of GT spikes)")
    
    # 合并所有选中的waveform
    selected_waveforms = [gt_spike_waveforms]
    if unmatched_spike_waveforms is not None:
        selected_waveforms.append(unmatched_spike_waveforms)
    if noise_waveforms is not None:
        selected_waveforms.append(noise_waveforms)
    
    all_selected = np.concatenate(selected_waveforms, axis=0)
    print(f"[INFO] Total selected waveforms: {len(all_selected):,}")
    
    # 提取snippets（对每个通道）
    print("[INFO] Extracting snippets from waveforms...")
    for waveform in tqdm(all_selected, desc="Extracting snippets"):
        for ch in range(30):
            snippet = waveform[ch, :]  # (30,)
            clips_list.append(snippet)
    
    clips = np.array(clips_list)  # (n_clips, 30)
    print(f"[INFO] Extracted {len(clips):,} snippets (channels × waveforms)")
    
    # 过滤全零或接近零的snippets（避免归一化时除以0）
    norms = np.linalg.norm(clips, axis=1)
    valid_mask = norms > 1e-10  # 过滤掉接近零的snippets
    clips = clips[valid_mask]
    print(f"[INFO] Filtered {np.sum(~valid_mask):,} zero/near-zero snippets, remaining: {len(clips):,}")
    
    if len(clips) == 0:
        raise ValueError("No valid snippets after filtering. All snippets are zero or near-zero.")
    
    # 归一化（L2归一化）
    norms = np.linalg.norm(clips, axis=1, keepdims=True)
    clips_norm = clips / norms
    
    # 再次检查NaN和Inf
    nan_mask = np.isnan(clips_norm).any(axis=1) | np.isinf(clips_norm).any(axis=1)
    if np.any(nan_mask):
        print(f"[WARNING] Found {np.sum(nan_mask):,} snippets with NaN/Inf after normalization, filtering them out...")
        clips_norm = clips_norm[~nan_mask]
    
    if len(clips_norm) == 0:
        raise ValueError("No valid snippets after NaN/Inf filtering.")
    
    print(f"[INFO] Final valid snippets for TruncatedSVD: {len(clips_norm):,}")
    
    # TruncatedSVD降维
    print(f"[INFO] Computing TruncatedSVD with {n_pcs} components...")
    model = TruncatedSVD(n_components=n_pcs)
    model.fit(clips_norm)
    wPCA = model.components_  # (n_pcs, nt) = (6, 30)
    
    explained_variance = model.explained_variance_ratio_.sum()
    print(f"[INFO] wPCA computed: shape {wPCA.shape}")
    print(f"[INFO] Explained variance: {explained_variance*100:.2f}%")
    
    # 保存wPCA
    if output_dir is not None:
        wPCA_path = os.path.join(output_dir, 'wPCA.npy')
        np.save(wPCA_path, wPCA)
        
        config = {
            'n_pcs': n_pcs,
            'include_noise_ratio': include_noise_ratio,
            'include_unmatched_spikes': include_unmatched_spikes,
            'n_gt_spikes': len(gt_spike_waveforms),
            'n_unmatched_spikes': len(unmatched_spike_waveforms) if unmatched_spike_waveforms is not None else 0,
            'n_noise': len(noise_waveforms) if noise_waveforms is not None else 0,
            'explained_variance': float(explained_variance)
        }
        config_path = os.path.join(output_dir, 'wPCA_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[INFO] wPCA saved to {wPCA_path}")
        print(f"[INFO] Config saved to {config_path}")
    
    return wPCA


def compute_templates_in_pc_space(df, wPCA, output_dir=None):
    """
    在PC空间中计算templates（类似Kilosort的Wall）
    
    参数:
    df: DataFrame，包含waveform和mapping列
    wPCA: numpy.ndarray, shape (n_pcs, nt) = (6, 30)
    output_dir: 保存templates的目录
    
    返回:
    templates_pc: numpy.ndarray, shape (n_clusters, n_channels, n_pcs) = (n_clusters, 30, 6)
    """
    print("\n" + "="*60)
    print("Computing Templates in PC Space")
    print("="*60)
    
    # 只使用GT spike（mapping >= 0）
    gt_spike_mask = df['mapping'] >= 0
    gt_waveforms = np.stack(df[gt_spike_mask]['waveform'].values)  # (n_spikes, 30, 30)
    gt_clusters = df[gt_spike_mask]['mapping'].values  # (n_spikes,)
    
    unique_clusters = np.unique(gt_clusters)
    unique_clusters = np.sort(unique_clusters)
    n_clusters = len(unique_clusters)
    n_pcs = wPCA.shape[0]  # 6
    n_channels = 30
    
    print(f"[INFO] Computing PC-space templates for {n_clusters} clusters")
    print(f"[INFO] Total GT spikes: {len(gt_waveforms):,}")
    print(f"[INFO] wPCA shape: {wPCA.shape}, Template shape will be: ({n_clusters}, {n_channels}, {n_pcs})")
    
    templates_pc = np.zeros((n_clusters, n_channels, n_pcs), dtype=np.float32)
    
    for i, cluster_id in enumerate(tqdm(unique_clusters, desc="Computing templates")):
        cluster_mask = gt_clusters == cluster_id
        cluster_waveforms = gt_waveforms[cluster_mask]  # (n_spikes_in_cluster, 30, 30)
        
        if len(cluster_waveforms) == 0:
            continue
        
        # 对每个cluster的waveform求平均
        mean_waveform = np.mean(cluster_waveforms, axis=0)  # (30, 30)
        
        # 将平均waveform投影到PC空间
        # 对每个通道：30个时间点 -> 6个PC特征
        for ch in range(n_channels):
            channel_waveform = mean_waveform[ch, :]  # (30,)
            # 投影到PC空间
            pc_features = channel_waveform @ wPCA.T  # (6,)
            templates_pc[i, ch, :] = pc_features
    
    print(f"[INFO] Templates computed in PC space: {templates_pc.shape}")
    
    # 保存templates
    if output_dir is not None:
        template_path = os.path.join(output_dir, 'templates_pc.npy')
        np.save(template_path, templates_pc)
        print(f"[INFO] PC-space templates saved to {template_path}")
        
        # 保存cluster ID映射
        cluster_mapping = {
            'cluster_ids': unique_clusters.tolist(),
            'template_indices': list(range(n_clusters))
        }
        cluster_mapping_path = os.path.join(output_dir, 'template_cluster_mapping.json')
        with open(cluster_mapping_path, 'w') as f:
            json.dump(cluster_mapping, f, indent=2)
        print(f"[INFO] Cluster mapping saved to {cluster_mapping_path}")
    
    return templates_pc


def preprocess_data(recording_f, spike_inf, output_dir):
    """
    预处理函数：生成包含time, channel, waveform, mapping的DataFrame
    
    返回:
    df : pandas.DataFrame
        包含列：time, channel, waveform(30×30), mapping(cluster_id或-1)
    """
    print("\n" + "="*60)
    print("Data Preprocessing")
    print("="*60)
    
    left_sample = 10  # spike前15个采样点
    right_sample = 20  # spike后15个采样点
    window_size = left_sample + right_sample  # 总共30个采样点

    # 获取完整的trace数据（仅前2分钟用于测试）
    print("[INFO] Loading recording trace (first 2 minutes for testing)...")
    load_start = time.time()
    
    # 假设采样率为10kHz，2分钟 = 120秒 = 1,200,000个采样点
    sampling_rate = recording_f.get_sampling_frequency()
    test_duration_seconds = 40 * 60  # 2分钟
    test_duration_samples = int(test_duration_seconds * sampling_rate)
    
    # 获取总长度
    total_samples = int(recording_f.get_total_duration() * sampling_rate)
    actual_test_samples = min(test_duration_samples, total_samples)
    
    print(f"[INFO] Sampling rate: {sampling_rate} Hz")
    print(f"[INFO] Total recording duration: {recording_f.get_total_duration():.2f}s ({total_samples:,} samples)")
    print(f"[INFO] Test duration: {test_duration_seconds}s ({actual_test_samples:,} samples)")
    
    trace0_car = recording_f.get_traces(segment_index=0, start_frame=0, end_frame=actual_test_samples)  # shape: (n_channels, time_points)
    load_time = time.time() - load_start
    print(f"[INFO] Trace loaded (before transpose) in {load_time:.2f}s, shape: {trace0_car.shape}")
    
    # 确保转置为 (time_points, n_channels)
    if trace0_car.shape[0] < trace0_car.shape[1]:  # 如果第一个维度小于第二个，说明是(n_channels, time_points)，需要转置
        trace0_car = trace0_car.T
    print(f"[INFO] Trace shape after transpose: {trace0_car.shape} (expected: (time_points, n_channels))")
    
    # 使用train_models.py中的检测方法
    print("[INFO] Detecting spikes using detect_local_minimum_in_window...")
    start_time = time.time()
    X_spiketrain_time, Y_spiketrain_id_final = detect_spike_from_train_models(
        trace0_car,
        std_multiplier=2.4,
        window_size=10
    )
    detection_time = time.time() - start_time
    trace_duration = trace0_car.shape[0] / sampling_rate  # 转换为秒
    speed_factor = trace_duration / detection_time if detection_time > 0 else 0
    print(f"[INFO] Detection completed in {detection_time:.2f}s")
    print(f"[INFO] Processing speed: {speed_factor:.2f}x real-time (trace duration: {trace_duration:.2f}s)")
    print(f"[INFO] Detected {len(X_spiketrain_time):,} candidate spikes")
    
    # 提取waveform（按照autosort的方式）
    print("[INFO] Extracting waveforms (30×30)...")
    # 过滤边界附近的spike
    valid_mask = X_spiketrain_time < trace0_car.shape[0] - (left_sample + right_sample)
    valid_mask = valid_mask & (X_spiketrain_time >= left_sample)
    
    X_spiketrain_time = X_spiketrain_time[valid_mask].astype(np.int64)
    Y_spiketrain_id_final = Y_spiketrain_id_final[valid_mask].astype(np.int64)
    
    # 提取waveform
    for time_range in tqdm(np.arange(-left_sample, right_sample, dtype=np.int64), desc="Extracting waveforms"):
        indices = (X_spiketrain_time + time_range).astype(np.int64)
        if time_range == -left_sample:
            waveform = trace0_car[indices, :]
        else:
            waveform = np.dstack((waveform, trace0_car[indices, :]))
    
    # waveform形状: (n_spikes, n_channels, n_timepoints) = (n_spikes, 30, 30)
    print(f"[INFO] Waveform shape: {waveform.shape}")
    
    # 提取单个通道的waveform（参考autosort的waveform_loader.py line 76）
    # 从检测通道提取单个通道的waveform
    single_channel_waveform = waveform[np.arange(len(waveform)), Y_spiketrain_id_final, :]
    # 形状: (n_spikes, n_timepoints) = (n_spikes, 30)
    print(f"[INFO] Single channel waveform shape: {single_channel_waveform.shape}")
    
    # GT mapping：获取cluster_id（只使用前2分钟的GT数据）
    print("[INFO] Mapping detected spikes to ground truth...")
    print(f"[INFO] Filtering GT spikes to first {test_duration_seconds}s...")
    
    # 过滤GT数据，只保留前2分钟的spike
    spike_inf_filtered = spike_inf[spike_inf['time'] < actual_test_samples].copy()
    print(f"[INFO] GT spikes: {len(spike_inf):,} total -> {len(spike_inf_filtered):,} in first 2 minutes")
    
    # 重置DataFrame索引以确保连续的位置索引
    spike_inf_filtered = spike_inf_filtered.reset_index(drop=True)
    
    # 获取GT时间点（只使用时间，不考虑通道）
    gt_times = spike_inf_filtered['time'].values.astype(np.int64)
    
    # 获取cluster_ids以匹配过滤后的数据
    if 'cluster' in spike_inf_filtered.columns:
        cluster_ids = spike_inf_filtered['cluster'].values
        print(f"[INFO] Using 'cluster' column from spike_inf, cluster range: [{cluster_ids.min()}, {cluster_ids.max()}]")
    else:
        print("[WARNING] No cluster column in spike_inf, using sequential index as cluster_id")
        # 使用连续的索引作为cluster_id（0, 1, 2, ...）
        cluster_ids = np.arange(len(spike_inf_filtered), dtype=int)
        print(f"[INFO] Cluster IDs range: [0, {len(spike_inf_filtered)-1}]")
    
    # 使用train_spike_pipeline.py风格的GT匹配（只匹配时间，不考虑通道）
    print(f"[INFO] Matching {len(X_spiketrain_time):,} detected spikes to {len(gt_times):,} GT spikes...")
    print(f"[INFO] Detect times range: [{X_spiketrain_time.min()}, {X_spiketrain_time.max()}]")
    print(f"[INFO] GT times range: [{gt_times.min()}, {gt_times.max()}]")
    
    gt_label_array1 = map_gt_annotation(X_spiketrain_time, gt_times, time_tolerance=1)
    
    # 构建mapping列：匹配成功则记录cluster_id，否则为-1
    # 注意：gt_label_array1返回的是gt_times的索引，对应spike_inf_filtered的位置索引（0到len-1）
    mapping = np.full(len(X_spiketrain_time), -1, dtype=int)
    matched_indices = np.where(gt_label_array1 >= 0)[0]
    if len(matched_indices) > 0:
        # gt_label_array1[matched_indices]是gt_times的索引，对应cluster_ids的位置索引
        # 确保索引不越界
        gt_indices = gt_label_array1[matched_indices].astype(int)
        valid_mask = (gt_indices >= 0) & (gt_indices < len(cluster_ids))
        if np.any(~valid_mask):
            print(f"[WARNING] {np.sum(~valid_mask)} matched indices out of range, skipping...")
            matched_indices = matched_indices[valid_mask]
            gt_indices = gt_indices[valid_mask]
        mapping[matched_indices] = cluster_ids[gt_indices]
        
        # 调试信息：检查mapping的分布
        unique_mappings, mapping_counts = np.unique(mapping[mapping >= 0], return_counts=True)
        print(f"[INFO] Matched spikes mapped to {len(unique_mappings)} unique clusters")
        print(f"[INFO] Top 10 clusters by match count:")
        top_indices = np.argsort(mapping_counts)[-10:][::-1]
        for idx in top_indices:
            print(f"       Cluster {unique_mappings[idx]}: {mapping_counts[idx]:,} matches")
    
    matched_count = np.sum(mapping >= 0)
    total_gt_spikes = len(gt_times)
    coverage = matched_count / total_gt_spikes if total_gt_spikes > 0 else 0
    print(f"[INFO] GT mapping summary:")
    print(f"       - Detected spikes: {len(X_spiketrain_time):,}")
    print(f"       - Matched to ground truth: {matched_count:,}")
    print(f"       - Ground truth total spikes: {total_gt_spikes:,}")
    print(f"       - Coverage (matched / ground truth): {coverage * 100:.2f}%")
    
    # 添加未匹配的GT spike（类似autosort的步骤4.5）
    print("[INFO] Adding unmatched GT spikes to training data...")
    mapped_ind = gt_label_array1[np.where(gt_label_array1 >= 0)[0]].astype(int)
    # 注意：这里使用spike_inf_filtered的长度，因为我们已经过滤了GT数据
    unmatched_gt_indices = [i for i in np.arange(len(spike_inf_filtered)) if i not in mapped_ind]
    
    if len(unmatched_gt_indices) > 0:
        print(f"[INFO] Found {len(unmatched_gt_indices):,} unmatched GT spikes")
        unmatched_times = spike_inf_filtered.iloc[unmatched_gt_indices]['time'].values.astype(np.int64)
        unmatched_clusters = cluster_ids[unmatched_gt_indices]
        
        # 获取未匹配GT的通道信息
        if 'best_channel' in spike_inf_filtered.columns:
            unmatched_channels = spike_inf_filtered.iloc[unmatched_gt_indices]['best_channel'].values.astype(np.int64)
        elif 'channel' in spike_inf_filtered.columns:
            unmatched_channels = spike_inf_filtered.iloc[unmatched_gt_indices]['channel'].values.astype(np.int64)
        elif 'ch' in spike_inf_filtered.columns:
            unmatched_channels = spike_inf_filtered.iloc[unmatched_gt_indices]['ch'].values.astype(np.int64)
        else:
            unmatched_channels = np.zeros(len(unmatched_gt_indices), dtype=np.int64)
        
        # 过滤边界附近的spike
        valid_unmatched_mask = (unmatched_times < trace0_car.shape[0] - (left_sample + right_sample)) & \
                              (unmatched_times >= left_sample)
        unmatched_times = unmatched_times[valid_unmatched_mask].astype(np.int64)
        unmatched_clusters = unmatched_clusters[valid_unmatched_mask]
        unmatched_channels = unmatched_channels[valid_unmatched_mask].astype(np.int64)
        
        # 提取未匹配GT的waveform
        if len(unmatched_times) > 0:
            for time_range in tqdm(np.arange(-left_sample, right_sample, dtype=np.int64), desc="Extracting unmatched GT waveforms", leave=False):
                indices = (unmatched_times + time_range).astype(np.int64)
                if time_range == -left_sample:
                    unmatched_waveform = trace0_car[indices, :]
                else:
                    unmatched_waveform = np.dstack((unmatched_waveform, trace0_car[indices, :]))
            
            # 提取未匹配GT的单个通道waveform
            unmatched_single_channel_waveform = unmatched_waveform[np.arange(len(unmatched_waveform)), unmatched_channels, :]
            
            # 合并检测到的spike和未匹配的GT spike
            X_spiketrain_time_train = np.concatenate([X_spiketrain_time, unmatched_times])
            Y_spiketrain_id_final_train = np.concatenate([Y_spiketrain_id_final, unmatched_channels])
            mapping_train = np.concatenate([mapping, unmatched_clusters])
            waveform_train = np.concatenate([waveform, unmatched_waveform], axis=0)
            single_channel_waveform_train = np.concatenate([single_channel_waveform, unmatched_single_channel_waveform], axis=0)
            
            print(f"[INFO] Added {len(unmatched_times):,} unmatched GT spikes")
            print(f"[INFO] Total training samples: {len(X_spiketrain_time_train):,}")
        else:
            X_spiketrain_time_train = X_spiketrain_time
            Y_spiketrain_id_final_train = Y_spiketrain_id_final
            mapping_train = mapping
            waveform_train = waveform
            single_channel_waveform_train = single_channel_waveform
    else:
        print("[INFO] All GT spikes are matched, no unmatched spikes to add")
        X_spiketrain_time_train = X_spiketrain_time
        Y_spiketrain_id_final_train = Y_spiketrain_id_final
        mapping_train = mapping
        waveform_train = waveform
        single_channel_waveform_train = single_channel_waveform
    
    # 构建DataFrame（包含检测到的spike和未匹配的GT spike）
    print("[INFO] Building DataFrame...")
    df_data = {
        'time': X_spiketrain_time_train,
        'channel': Y_spiketrain_id_final_train,
        'mapping': mapping_train
    }
    
    # 将waveform和single_channel_waveform添加到DataFrame（作为列表或numpy数组）
    df = pd.DataFrame(df_data)
    df['waveform'] = [waveform_train[i] for i in range(len(waveform_train))]
    df['single_channel_waveform'] = [single_channel_waveform_train[i] for i in range(len(single_channel_waveform_train))]
    
    print(f"[INFO] DataFrame created with {len(df):,} rows")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    print(f"[INFO] Mapping distribution: {df['mapping'].value_counts().head(10)}")
    
    # 保存DataFrame
    df_path = os.path.join(output_dir, 'preprocessed_data.pkl')
    os.makedirs(output_dir, exist_ok=True)
    df.to_pickle(df_path)
    print(f"[INFO] Preprocessed data saved to {df_path}")
    
    # 计算wPCA
    wPCA = compute_wPCA_from_train_data(
        df, 
        n_pcs=6, 
        output_dir=output_dir,
        include_noise_ratio=0.0,
        include_unmatched_spikes=True
    )
    
    # 在PC空间中计算templates
    templates_pc = compute_templates_in_pc_space(df, wPCA, output_dir=output_dir)
    
    return df




def train_detection_model(df, output_dir, use_pc_features=True):
    """
    训练Spike Detection模型 - 从DataFrame读取数据
    支持使用PC特征进行降维
    
    参数:
    df: DataFrame，包含waveform和mapping列
    output_dir: 输出目录
    use_pc_features: 是否使用PC特征（默认True）
    """
    print("\n" + "="*60)
    print("Training Spike Detection Model")
    if use_pc_features:
        print("Using PC features (30×6 = 180 dim)")
    else:
        print("Using raw waveforms (30×30 = 900 dim)")
    print("="*60)
    
    # 从DataFrame提取所有waveform和标签
    print("[INFO] Loading data from DataFrame...")
    all_windows = np.stack(df['waveform'].values)  # shape: (n_spikes, 30, 30)
    
    # 生成标签：mapping >= 0 为1（spike），mapping == -1 为0（noise）
    labels = (df['mapping'] >= 0).astype(int).values
    
    print(f"[INFO] Total samples: {len(df):,}")
    print(f"[INFO] Waveform shape: {all_windows.shape}")
    
    # 如果使用PC特征，先加载wPCA并转换
    if use_pc_features:
        wPCA_path = os.path.join(output_dir, 'wPCA.npy')
        if not os.path.exists(wPCA_path):
            raise FileNotFoundError(f"wPCA not found at {wPCA_path}. Please run preprocess_data first.")
        
        wPCA = np.load(wPCA_path)  # (6, 30)
        print(f"[INFO] Loaded wPCA: shape {wPCA.shape}")
        
        # 将waveform转换为PC特征
        print("[INFO] Converting waveforms to PC features...")
        pc_features_list = []
        for waveform in tqdm(all_windows, desc="Converting to PC"):
            # waveform: (30, 30)
            # 对每个通道：30个时间点 -> 6个PC特征
            pc_channel_features = waveform @ wPCA.T  # (30, 6)
            pc_features_list.append(pc_channel_features)
        
        all_features = np.array(pc_features_list)  # (n_spikes, 30, 6)
        input_size = 30 * 6  # 180
        print(f"[INFO] PC features shape: {all_features.shape}")
    else:
        all_features = all_windows  # (n_spikes, 30, 30)
        input_size = 30 * 30  # 900
        print(f"[INFO] Raw waveform shape: {all_features.shape}")
    
    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]
    print(f"[INFO] Label distribution -> Noise: {len(indices_0):,}, Spike: {len(indices_1):,}")

    target_0_count = len(indices_1) * 2

    if len(indices_0) > target_0_count:
        sampled_indices_0 = np.random.choice(indices_0, target_0_count, replace=False)
    else:
        sampled_indices_0 = indices_0

    final_indices = np.concatenate([sampled_indices_0, indices_1])
    np.random.shuffle(final_indices)

    sampled_features = all_features[final_indices]
    sampled_labels = labels[final_indices]
    print(f"[INFO] Balanced dataset size: {len(sampled_labels):,} (Positive: {len(indices_1):,}, Negative: {len(sampled_indices_0):,})")

    os.makedirs(output_dir, exist_ok=True)
    # 展平特征以适配NoiseClassifier
    sampled_features_flat = sampled_features.reshape(len(sampled_features), -1)  # (n_samples, input_size)
    dataset = SimpleDetectionDataset(sampled_features_flat, sampled_labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"[INFO] Train size: {train_size:,} | Test size: {test_size:,}")

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Input size: {input_size} (using NoiseClassifier)")

    pos_weight = len(indices_0) / len(indices_1) if len(indices_1) > 0 else 1.0
    pos_weight_tpr = pos_weight * 2.0
    print(f"[INFO] Positive class weight: {pos_weight:.4f} (original) -> {pos_weight_tpr:.4f} (for TPR optimization)")

    accuracy_list = []
    tpr_list = []
    tnr_list = []
    f1_list = []

    for trail in range(1, 6):
        print(f"\n[INFO] === Trail {trail}/5 ===")
        # NoiseClassifier输出2类logits，使用CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

        model = NoiseClassifier(input_size, device)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        num_epochs = 30
        f1_best = 0
        best_metrics = {"accuracy": 0, "tpr": 0, "tnr": 0, "f1": 0, "epoch": 0}
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.long().to(device)  # CrossEntropyLoss需要long类型

                outputs = model(batch_data)  # (batch, 2) - logits
                
                # 应用样本权重
                sample_weights = torch.where(batch_labels == 1, 
                                            torch.tensor(pos_weight_tpr).to(device), 
                                            torch.tensor(1.0).to(device))
                # CrossEntropyLoss的weight参数需要在初始化时设置，这里使用reduction='none'然后手动加权
                loss_per_sample = nn.functional.cross_entropy(outputs, batch_labels, reduction='none')
                loss = (loss_per_sample * sample_weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / max(len(train_loader), 1)

            model.eval()
            correct = 0
            total = 0
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.long().to(device)

                    outputs = model(batch_data)  # (batch, 2) - logits
                    # 使用softmax获取概率，然后取argmax作为预测类别
                    probs = torch.softmax(outputs, dim=1)  # (batch, 2)
                    predicted = torch.argmax(probs, dim=1)  # (batch,)
                    
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    true_positive += ((predicted == 1) & (batch_labels == 1)).sum().item()
                    true_negative += ((predicted == 0) & (batch_labels == 0)).sum().item()
                    false_positive += ((predicted == 1) & (batch_labels == 0)).sum().item()
                    false_negative += ((predicted == 0) & (batch_labels == 1)).sum().item()

            tpr = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            tnr = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
            epoch_accuracy = correct / total if total > 0 else 0
            
            # 计算 Precision 和 F1 分数
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = tpr  # recall 就是 TPR
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"[Trail {trail}] Epoch {epoch + 1:02d}/{num_epochs} | "
                  f"Loss: {avg_train_loss:.4f} | Acc: {epoch_accuracy * 100:.2f}% | "
                  f"TPR: {tpr * 100:.2f}% | TNR: {tnr * 100:.2f}% | F1: {f1 * 100:.2f}%")

            if f1 > f1_best:
                f1_best = f1
                patience_counter = 0
                best_metrics = {"accuracy": epoch_accuracy, "tpr": tpr, "tnr": tnr, "f1": f1, "epoch": epoch + 1}
                # 根据是否使用PC特征添加后缀
                model_suffix = "_pc" if use_pc_features else "_raw"
                model_path = os.path.join(output_dir, f'detection_trail_{trail}{model_suffix}.pth')
                torch.save(model, model_path)
                print(f"[Trail {trail}] ✅ New best model saved (Epoch {epoch + 1}) | F1: {f1_best:.4f}")
            else:
                patience_counter += 1
                if patience_counter == 3:
                    print(f"[Trail {trail}] Early stopping at epoch {epoch + 1} with best F1: {f1_best:.4f}")
                    break

        accuracy_list.append(best_metrics["accuracy"])
        tpr_list.append(best_metrics["tpr"])
        tnr_list.append(best_metrics["tnr"])
        f1_list.append(best_metrics["f1"])
        print(f"[Trail {trail}] Best metrics -> Acc: {best_metrics['accuracy'] * 100:.2f}%, "
              f"TPR: {best_metrics['tpr'] * 100:.2f}%, TNR: {best_metrics['tnr'] * 100:.2f}%, "
              f"F1: {best_metrics['f1'] * 100:.2f}% "
              f"(achieved at epoch {best_metrics['epoch']})")

    print("\n[INFO] Detection Training Summary")
    print(f"Mean Accuracy: {np.mean(accuracy_list) * 100:.2f}% ± {np.std(accuracy_list) * 100:.2f}%")
    print(f"Mean TPR: {np.mean(tpr_list) * 100:.2f}% ± {np.std(tpr_list) * 100:.2f}%")
    print(f"Mean TNR: {np.mean(tnr_list) * 100:.2f}% ± {np.std(tnr_list) * 100:.2f}%")
    print(f"Mean F1: {np.mean(f1_list) * 100:.2f}% ± {np.std(f1_list) * 100:.2f}%")




def main():
    print("[INFO] Starting Spike Detection training pipeline with PC features")
    
    # 配置路径
    recording_path = '/media/ubuntu/sda/data/mouse6/ns4/natural_image/mouse6_021322_natural_image_001.ns4'
    spike_inf_path = "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/kilosort_spike_sorting/sorting_new/021322/spike_inf.tsv"
    output_dir = '/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/pipeline_results'
    
    print(f"[INFO] Loading recording from {recording_path}")
    recording_raw = se.read_blackrock(file_path=recording_path)
    recording_recorded = recording_raw.remove_channels(["98", '31', '32'])

    print("[INFO] Applying bandpass filter and common reference")
    recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
    recording_f = spre.common_reference(recording_f, reference="global", operator="average")
    
    print(f"[INFO] Loading spike annotations from {spike_inf_path}")
    # 读取TSV文件（使用tab分隔符）
    spike_inf = pd.read_csv(spike_inf_path, sep='\t')
    
    # 步骤1: 预处理 - 生成DataFrame，计算wPCA和templates
    df = preprocess_data(recording_f, spike_inf, output_dir)
    
    # 步骤2: 训练Detection模型（使用PC特征）
    train_detection_model(df, output_dir, use_pc_features=True)
    
    print("\n[INFO] Training pipeline completed successfully")


if __name__ == "__main__":
    main()

