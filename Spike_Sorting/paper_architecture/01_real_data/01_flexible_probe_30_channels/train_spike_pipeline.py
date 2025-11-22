#!/usr/bin/env python
# coding: utf-8
"""
AutoSort训练脚本：同时训练两个独立的分类网络
- 两个网络共享相同的架构（SimpleClassifier/clssimp）
- 同时训练，使用组合损失函数
- 但保持独立的参数和输出（不是多任务学习）

参考：autosort_neuron/model.py 中的 AutoSort 类
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
from scipy.signal import find_peaks

# 添加spike_detection路径以导入函数
sys.path.append('/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/spike_detection')
from train_models import (
    detect_local_minimum_in_window as _detect_local_minimum_in_window,
    label_array1_based_on_array2,
    extract_windows,
    Spike_Detection_MLP
)

# 包装函数，添加进度条
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


# ==================== AutoSort Model ====================
class SimpleClassifier(nn.Module):
    """
    简单的分类器（参考autosort的clssimp类）
    用于构建两个独立的分类网络，共享相同的架构但参数完全独立
    """
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
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
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        logits = self.cls(x)
        return logits
    
    def intermediate_forward(self, x):
        """提取中间层特征（用于UMAP可视化）"""
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        return x


class AutoSort(nn.Module):
    """
    AutoSort模型：两个独立的分类网络（参考autosort的AutoSort类）
    - 共享相同的网络架构（SimpleClassifier/clssimp）
    - 同时训练
    - 使用组合损失函数
    - 但保持独立的参数和输出
    
    注意：这不是多任务学习（共享特征提取层），而是两个完全独立的网络
    """
    def __init__(self, input_size, num_classes, device):
        super(AutoSort, self).__init__()
        # 两个独立的分类器，参数完全独立（参考autosort model.py line 98-99）
        self.clsfier_noise = SimpleClassifier(input_size, 2).to(device)  # noise/spike分类
        self.clsfier_label = SimpleClassifier(input_size, num_classes).to(device)  # neuron分类
        self.device = device
    
    def forward(self, x, mode='train'):
        """
        前向传播（参考autosort的iter_model，model.py line 141-167）
        两个独立的网络使用相同的输入，但产生独立的输出
        
        x: 输入waveform，shape: (batch, 30, 30)
        mode: 'train' 或 'eval'
        """
        # 展平waveform: (batch, 30, 30) -> (batch, 900)
        x_flat = x.reshape(x.size(0), -1)
        
        # 两个独立的网络使用相同的输入codes，但参数完全独立
        # Noise分类器：对所有样本进行分类（参考autosort line 149）
        noise_output = self.clsfier_noise(x_flat)
        
        # Label分类器：对所有样本进行分类（在训练时通过mask控制损失计算，参考autosort line 154）
        label_output = self.clsfier_label(x_flat)
        
        return noise_output, label_output
    
    def get_intermediate_features(self, x):
        """
        提取中间层特征（用于UMAP可视化）
        x: 输入waveform，shape: (batch, 30, 30)
        返回: (noise_features, label_features)
        """
        x_flat = x.reshape(x.size(0), -1)
        noise_features = self.clsfier_noise.intermediate_forward(x_flat)
        label_features = self.clsfier_label.intermediate_forward(x_flat)
        return noise_features, label_features


class AutoSortDataset(Dataset):
    """
    AutoSort数据集（参考autosort的waveformLoader）
    用于两个独立网络的训练，提供noise分类和label分类所需的标签
    """
    def __init__(self, waveforms, noise_labels, cluster_labels, spike_mask, num_classes):
        """
        参数:
        waveforms: numpy.ndarray, shape (n_samples, 30, 30)
        noise_labels: numpy.ndarray, shape (n_samples,), 0=noise, 1=spike
        cluster_labels: numpy.ndarray, shape (n_samples,), cluster_id（已映射为连续索引）
        spike_mask: numpy.ndarray, shape (n_samples,), True表示是spike样本
        num_classes: int, cluster的数量
        """
        self.waveforms = torch.FloatTensor(waveforms)
        self.spike_mask = torch.BoolTensor(spike_mask)
        
        # 将noise_labels转换为one-hot格式 (n_samples, 2): [noise, spike]
        noise_labels_tensor = torch.LongTensor(noise_labels)
        self.noise_labels_onehot = torch.zeros((len(noise_labels), 2), dtype=torch.float32)
        self.noise_labels_onehot[torch.arange(len(noise_labels)), noise_labels_tensor] = 1.0
        
        # 将cluster_labels转换为one-hot格式 (n_samples, num_classes)
        cluster_labels_tensor = torch.LongTensor(cluster_labels)
        self.cluster_labels_onehot = torch.zeros((len(cluster_labels), num_classes), dtype=torch.float32)
        # 只对有效的cluster（>=0）设置one-hot
        valid_mask = cluster_labels_tensor >= 0
        if valid_mask.any():
            valid_indices = torch.arange(len(cluster_labels))[valid_mask]
            self.cluster_labels_onehot[valid_indices, cluster_labels_tensor[valid_mask]] = 1.0
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        return {
            'waveform': self.waveforms[idx],
            'noise_label': self.noise_labels_onehot[idx],  # one-hot格式
            'cluster_label': self.cluster_labels_onehot[idx],  # one-hot格式
            'is_spike': self.spike_mask[idx]
        }


# ==================== Detection Functions (from train_models.py) ====================
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
    
    # 使用GT匹配（只匹配时间，不考虑通道）
    print(f"[INFO] Matching {len(X_spiketrain_time):,} detected spikes to {len(gt_times):,} GT spikes...")
    print(f"[INFO] Detect time range: [{X_spiketrain_time.min()}, {X_spiketrain_time.max()}]")
    print(f"[INFO] GT time range: [{gt_times.min()}, {gt_times.max()}]")
    
    gt_label_array1 = map_gt_annotation(X_spiketrain_time, gt_times)
    
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
        
        # 为未匹配GT的spike找到对应的通道（通过找到最小值所在的通道）
        unmatched_channels = []
        for t in unmatched_times:
            if t < trace0_car.shape[0]:
                channel_values = trace0_car[t, :]
                min_channel = np.argmin(channel_values)
                unmatched_channels.append(min_channel)
            else:
                unmatched_channels.append(0)  # 边界情况，使用通道0
        unmatched_channels = np.array(unmatched_channels, dtype=np.int64)
        
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
            
            # 合并检测到的spike和未匹配的GT spike
            X_spiketrain_time_train = np.concatenate([X_spiketrain_time, unmatched_times])
            Y_spiketrain_id_final_train = np.concatenate([Y_spiketrain_id_final, unmatched_channels])
            mapping_train = np.concatenate([mapping, unmatched_clusters])
            waveform_train = np.concatenate([waveform, unmatched_waveform], axis=0)
            
            print(f"[INFO] Added {len(unmatched_times):,} unmatched GT spikes")
            print(f"[INFO] Total training samples: {len(X_spiketrain_time_train):,}")
        else:
            X_spiketrain_time_train = X_spiketrain_time
            Y_spiketrain_id_final_train = Y_spiketrain_id_final
            mapping_train = mapping
            waveform_train = waveform
    else:
        print("[INFO] All GT spikes are matched, no unmatched spikes to add")
        X_spiketrain_time_train = X_spiketrain_time
        Y_spiketrain_id_final_train = Y_spiketrain_id_final
        mapping_train = mapping
        waveform_train = waveform
    
    # 构建DataFrame（包含检测到的spike和未匹配的GT spike）
    print("[INFO] Building DataFrame...")
    df_data = {
        'time': X_spiketrain_time_train,
        'channel': Y_spiketrain_id_final_train,
        'mapping': mapping_train
    }
    
    # 将waveform添加到DataFrame（作为列表或numpy数组）
    df = pd.DataFrame(df_data)
    df['waveform'] = [waveform_train[i] for i in range(len(waveform_train))]
    
    print(f"[INFO] DataFrame created with {len(df):,} rows")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    print(f"[INFO] Mapping distribution: {df['mapping'].value_counts().head(10)}")
    
    # 保存DataFrame
    df_path = os.path.join(output_dir, 'preprocessed_data.pkl')
    os.makedirs(output_dir, exist_ok=True)
    df.to_pickle(df_path)
    print(f"[INFO] Preprocessed data saved to {df_path}")
    
    return df


def train_autosort_model(df, output_dir):
    """
    AutoSort训练：同时训练两个独立的分类网络
    - 两个网络共享相同的架构（SimpleClassifier/clssimp）
    - 同时训练，使用组合损失函数
    - 但保持独立的参数和输出（不是多任务学习）
    """
    print("\n" + "="*60)
    print("Training AutoSort Model (Two Independent Networks)")
    print("="*60)
    print("[INFO] Architecture: Two independent networks with shared architecture")
    print("[INFO] Training: Simultaneous training with combined loss")
    print("[INFO] Parameters: Completely independent (not multi-task learning)")
    
    # 从DataFrame提取数据（向量化操作，避免慢速的np.stack）
    print("[INFO] Loading data from DataFrame...")
    cluster_labels_raw = df['mapping'].values
    noise_labels = (cluster_labels_raw >= 0).astype(int)
    spike_mask = noise_labels == 1
    
    # 处理cluster_id：将非连续的cluster_id映射到连续索引（向量化）
    valid_mask = cluster_labels_raw >= 0
    unique_clusters = np.unique(cluster_labels_raw[valid_mask])
    unique_clusters = np.sort(unique_clusters)
    
    if len(unique_clusters) == 0:
        print("[ERROR] No valid clusters found for classification!")
        return
    
    # 使用向量化操作映射cluster_id（比循环快得多）
    cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    index_to_cluster = {idx: cluster for cluster, idx in cluster_to_index.items()}
    
    # 向量化映射：使用pandas的map方法（比循环快得多）
    cluster_labels_mapped = np.full(len(cluster_labels_raw), -1, dtype=int)
    valid_mask = cluster_labels_raw >= 0
    if valid_mask.any():
        # 使用pandas Series的map方法进行向量化映射
        cluster_series = pd.Series(cluster_labels_raw)
        mapped_series = cluster_series.map(cluster_to_index).fillna(-1).astype(int)
        cluster_labels_mapped = mapped_series.values
    
    print(f"[INFO] Total samples: {len(df):,}")
    print(f"[INFO] Noise samples: {np.sum(noise_labels == 0):,}, Spike samples: {np.sum(noise_labels == 1):,}")
    print(f"[INFO] Unique clusters: {len(unique_clusters)}")
    print(f"[INFO] Original cluster IDs: {unique_clusters[:10]}{'...' if len(unique_clusters) > 10 else ''}")
    
    # 保存cluster映射关系
    cluster_mapping_path = os.path.join(output_dir, 'cluster_id_to_index_mapping.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(cluster_mapping_path, 'w') as f:
        json.dump({
            'cluster_to_index': {str(k): int(v) for k, v in cluster_to_index.items()},
            'index_to_cluster': {str(k): int(v) for k, v in index_to_cluster.items()}
        }, f, indent=2)
    print(f"[INFO] Cluster mapping saved to {cluster_mapping_path}")
    
    # 按照autosort的逻辑：使用所有样本，不进行数据平衡
    print("[INFO] Using all samples (no balancing, following autosort logic)...")
    final_indices = np.arange(len(df), dtype=np.int64)
    
    # 打乱数据（autosort也会在DataLoader中shuffle）
    np.random.shuffle(final_indices)
    
    # 提取所有waveform（延迟加载，避免一次性加载所有数据）
    print("[INFO] Extracting all waveforms...")
    all_noise_labels = noise_labels[final_indices]
    all_cluster_labels = cluster_labels_mapped[final_indices]
    all_spike_mask = spike_mask[final_indices]
    
    # 使用iloc批量提取waveform（比列表推导式快）
    all_waveforms = np.stack(df.iloc[final_indices]['waveform'].values)
    
    print(f"[INFO] Total dataset size: {len(final_indices):,}")
    print(f"[INFO]   - Noise: {np.sum(all_noise_labels == 0):,}")
    print(f"[INFO]   - Spike: {np.sum(all_noise_labels == 1):,}")
    
    # 创建数据集（使用所有样本）
    # 注意：这个数据集用于两个独立网络的训练，不是多任务学习
    dataset = AutoSortDataset(
        all_waveforms,
        all_noise_labels,
        all_cluster_labels,
        all_spike_mask,
        num_classes=len(unique_clusters)
    )
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"[INFO] Train size: {train_size:,} | Test size: {test_size:,}")
    
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    input_size = all_waveforms.shape[1] * all_waveforms.shape[2]  # 30 * 30 = 900
    num_classes = len(unique_clusters)
    
    # 计算类别权重（用于BCE loss）
    noise_count = np.sum(all_noise_labels == 0)
    spike_count = np.sum(all_noise_labels == 1)
    
    # Noise分类的pos_weight: [noise_weight, spike_weight]
    pos_weight_noise = torch.tensor([
        spike_count / noise_count if noise_count > 0 else 1.0,  # noise类权重
        noise_count / spike_count if spike_count > 0 else 1.0   # spike类权重
    ]).to(device)
    
    # Label分类的pos_weight: 对每个cluster计算权重
    pos_weight_label = []
    for cluster_idx in range(len(unique_clusters)):
        cluster_count = np.sum(all_cluster_labels == cluster_idx)
        other_count = len(all_cluster_labels) - cluster_count
        weight = other_count / cluster_count if cluster_count > 0 else 1.0
        pos_weight_label.append(weight)
    pos_weight_label = torch.tensor(pos_weight_label).to(device)
    
    print(f"[INFO] Dataset statistics: Noise={noise_count:,}, Spike={spike_count:,}")
    print(f"[INFO] Noise pos_weight: {pos_weight_noise.cpu().numpy()}")
    print(f"[INFO] Label pos_weight range: [{pos_weight_label.min().item():.2f}, {pos_weight_label.max().item():.2f}]")
    
    for trail in range(1, 6):
        print(f"\n[INFO] === Trail {trail}/5 ===")
        
        # 创建两个独立的网络（参考autosort model.py line 98-99）
        model = AutoSort(input_size, num_classes, device)
        
        # 两个网络的参数在optimizer中分开（参考autosort model.py line 101-104）
        # 这确保了参数完全独立，不是多任务学习
        optimizer = optim.Adam([
            {'params': model.clsfier_noise.parameters()},
            {'params': model.clsfier_label.parameters()}
        ], lr=1e-4)
        
        # 使用BCEWithLogitsLoss，与autosort一致
        criterion_noise = nn.BCEWithLogitsLoss(pos_weight=pos_weight_noise)
        criterion_label = nn.BCEWithLogitsLoss(pos_weight=pos_weight_label)
        
        num_epochs = 210
        best_accuracy = 0
        best_noise_accuracy = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_noise_loss = 0
            total_label_loss = 0
            total_loss = 0
            
            for batch in train_loader:
                waveforms = batch['waveform'].to(device)
                noise_labels = batch['noise_label'].to(device)
                cluster_labels = batch['cluster_label'].to(device)
                is_spike = batch['is_spike']
                
                # 前向传播：两个独立网络使用相同的输入
                noise_output, label_output = model(waveforms)
                
                # 两个独立网络的损失计算（参考autosort model.py line 160-164）
                # Noise分类损失：所有样本（参考autosort line 161）
                noise_loss = criterion_noise(noise_output, noise_labels)
                
                # Label分类损失：只对spike样本（参考autosort line 154-157）
                spike_mask_tensor = is_spike.to(device)
                if spike_mask_tensor.sum() > 0:
                    label_loss = criterion_label(label_output[spike_mask_tensor], cluster_labels[spike_mask_tensor])
                else:
                    label_loss = torch.tensor(0.0).to(device)
                
                # 组合损失函数：两个独立网络的损失相加（参考autosort line 164）
                # 注意：这是两个独立网络的组合损失，不是多任务学习的共享损失
                # 梯度计算时，noise_loss只影响clsfier_noise的参数，label_loss只影响clsfier_label的参数
                total_batch_loss = 1000 * noise_loss + 1000 * label_loss
                
                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()
                
                total_noise_loss += noise_loss.item()
                total_label_loss += label_loss.item() if isinstance(label_loss, torch.Tensor) else label_loss
                total_loss += total_batch_loss.item()
            
            # 验证
            model.eval()
            noise_correct = 0
            noise_total = 0
            label_correct = 0
            label_total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    waveforms = batch['waveform'].to(device)
                    noise_labels = batch['noise_label'].to(device)
                    cluster_labels = batch['cluster_label'].to(device)
                    is_spike = batch['is_spike'].to(device)
                    
                    noise_output, label_output = model(waveforms)
                    
                    # Noise分类准确率（one-hot标签，使用argmax比较）
                    noise_pred = torch.argmax(noise_output, dim=1)
                    noise_gt = torch.argmax(noise_labels, dim=1)
                    noise_correct += (noise_pred == noise_gt).sum().item()
                    noise_total += noise_labels.size(0)
                    
                    # Label分类准确率（只对spike样本，one-hot标签）
                    spike_mask_tensor = is_spike
                    if spike_mask_tensor.sum() > 0:
                        label_pred = torch.argmax(label_output[spike_mask_tensor], dim=1)
                        label_gt = torch.argmax(cluster_labels[spike_mask_tensor], dim=1)
                        label_correct += (label_pred == label_gt).sum().item()
                        label_total += spike_mask_tensor.sum().item()
            
            noise_acc = noise_correct / noise_total if noise_total > 0 else 0
            label_acc = label_correct / label_total if label_total > 0 else 0
            
            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | "
                      f"Noise Loss: {total_noise_loss/len(train_loader):.4f} | "
                      f"Label Loss: {total_label_loss/len(train_loader):.4f} | "
                      f"Noise Acc: {noise_acc*100:.2f}% | "
                      f"Label Acc: {label_acc*100:.2f}%")
            
            # 保存最佳模型（基于label准确率）
            # 参考autosort：分别保存两个独立网络的权重（model.py line 118-120）
            if label_acc > best_accuracy:
                best_accuracy = label_acc
                best_noise_accuracy = noise_acc
                patience_counter = 0
                # 分别保存两个独立网络的权重
                noise_model_path = os.path.join(output_dir, f'autosort_trail_{trail}_noise_clsfier.pth')
                label_model_path = os.path.join(output_dir, f'autosort_trail_{trail}_label_clsfier.pth')
                torch.save(model.clsfier_noise.state_dict(), noise_model_path)
                torch.save(model.clsfier_label.state_dict(), label_model_path)
                print(f"✓ Best models saved (Epoch {epoch+1}) | Label Acc: {best_accuracy*100:.2f}% | Noise Acc: {best_noise_accuracy*100:.2f}%")
                print(f"  - Noise classifier: {noise_model_path}")
                print(f"  - Label classifier: {label_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        print(f"[Trail {trail}] Final - Noise Acc: {best_noise_accuracy*100:.2f}% | Label Acc: {best_accuracy*100:.2f}%")
    
    print("\n[INFO] AutoSort training completed!")
    print("[INFO] Each trail saves two separate model files:")
    print("  - autosort_trail_X_noise_clsfier.pth (noise/spike classification network)")
    print("  - autosort_trail_X_label_clsfier.pth (neuron classification network)")


def train_detection_model(df, output_dir):
    """训练Spike Detection模型 - 从DataFrame读取数据"""
    print("\n" + "="*60)
    print("Training Spike Detection Model")
    print("="*60)
    
    # 从DataFrame提取所有waveform和标签
    print("[INFO] Loading data from DataFrame...")
    all_windows = np.stack(df['waveform'].values)  # shape: (n_spikes, 30, 30)
    
    # 生成标签：mapping >= 0 为1（spike），mapping == -1 为0（noise）
    labels = (df['mapping'] >= 0).astype(int).values
    
    print(f"[INFO] Total samples: {len(df):,}")
    print(f"[INFO] Waveform shape: {all_windows.shape}")
    
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

    sampled_windows = all_windows[final_indices]
    sampled_labels = labels[final_indices]
    print(f"[INFO] Balanced dataset size: {len(sampled_labels):,} (Positive: {len(indices_1):,}, Negative: {len(sampled_indices_0):,})")

    os.makedirs(output_dir, exist_ok=True)
    dataset = SimpleDetectionDataset(sampled_windows, sampled_labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"[INFO] Train size: {train_size:,} | Test size: {test_size:,}")

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = sampled_windows.shape[1] * sampled_windows.shape[2]
    hidden_size1 = 128
    hidden_size2 = 32
    output_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    pos_weight = len(indices_0) / len(indices_1) if len(indices_1) > 0 else 1.0
    pos_weight_tpr = pos_weight * 2.0
    print(f"[INFO] Positive class weight: {pos_weight:.4f} (original) -> {pos_weight_tpr:.4f} (for TPR optimization)")

    accuracy_list = []
    tpr_list = []
    tnr_list = []

    for trail in range(1, 6):
        print(f"\n[INFO] === Trail {trail}/5 ===")
        criterion = nn.BCELoss()

        model = Spike_Detection_MLP(
            input_size, hidden_size1, hidden_size2,
            output_size, n_channels=sampled_windows.shape[1], time_window=sampled_windows.shape[2]
        )
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        num_epochs = 30
        tpr_best = 0
        best_metrics = {"accuracy": 0, "tpr": 0, "tnr": 0, "epoch": 0}
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_data, batch_labels in train_loader:
                batch_labels = batch_labels.float().unsqueeze(1)

                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_data)
                base_loss = criterion(outputs, batch_labels)
                sample_weights = torch.where(batch_labels == 1, 
                                            torch.tensor(pos_weight_tpr).to(device), 
                                            torch.tensor(1.0).to(device))
                loss = (base_loss * sample_weights).mean()

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
                    batch_labels = batch_labels.float().unsqueeze(1)
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_data)
                    predicted = (outputs > 0.5).float()
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    true_positive += ((predicted == 1) & (batch_labels == 1)).sum().item()
                    true_negative += ((predicted == 0) & (batch_labels == 0)).sum().item()
                    false_positive += ((predicted == 1) & (batch_labels == 0)).sum().item()
                    false_negative += ((predicted == 0) & (batch_labels == 1)).sum().item()

            tpr = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            tnr = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
            epoch_accuracy = correct / total if total > 0 else 0

            print(f"[Trail {trail}] Epoch {epoch + 1:02d}/{num_epochs} | "
                  f"Loss: {avg_train_loss:.4f} | Acc: {epoch_accuracy * 100:.2f}% | "
                  f"TPR: {tpr * 100:.2f}% | TNR: {tnr * 100:.2f}%")

            if tpr > tpr_best:
                tpr_best = tpr
                patience_counter = 0
                best_metrics = {"accuracy": epoch_accuracy, "tpr": tpr, "tnr": tnr, "epoch": epoch + 1}
                model_path = os.path.join(output_dir, f'detection_trail_{trail}.pth')
                torch.save(model, model_path)
                print(f"[Trail {trail}] ✅ New best model saved (Epoch {epoch + 1}) | TPR: {tpr_best:.4f}")
            else:
                patience_counter += 1
                if patience_counter == 3:
                    print(f"[Trail {trail}] Early stopping at epoch {epoch + 1} with best TPR: {tpr_best:.4f}")
                    break

        accuracy_list.append(best_metrics["accuracy"])
        tpr_list.append(best_metrics["tpr"])
        tnr_list.append(best_metrics["tnr"])
        print(f"[Trail {trail}] Best metrics -> Acc: {best_metrics['accuracy'] * 100:.2f}%, "
              f"TPR: {best_metrics['tpr'] * 100:.2f}%, TNR: {best_metrics['tnr'] * 100:.2f}% "
              f"(achieved at epoch {best_metrics['epoch']})")

    print("\n[INFO] Detection Training Summary")
    print(f"Mean Accuracy: {np.mean(accuracy_list) * 100:.2f}% ± {np.std(accuracy_list) * 100:.2f}%")
    print(f"Mean TPR: {np.mean(tpr_list) * 100:.2f}% ± {np.std(tpr_list) * 100:.2f}%")
    print(f"Mean TNR: {np.mean(tnr_list) * 100:.2f}% ± {np.std(tnr_list) * 100:.2f}%")


def train_classification_model(df, output_dir):
    """训练Spike Classification模型 - 从DataFrame读取数据"""
    print("\n" + "="*60)
    print("Training Spike Classification Model")
    print("="*60)
    
    # 从DataFrame提取有cluster_id的waveform（mapping >= 0）
    print("[INFO] Loading data from DataFrame...")
    df_classification = df[df['mapping'] >= 0].copy()  # 只使用匹配成功的spike
    
    if len(df_classification) == 0:
        print("[ERROR] No matched spikes found for classification training!")
        return
    
    all_windows = np.stack(df_classification['waveform'].values)  # shape: (n_spikes, 30, 30)
    cluster_labels = df_classification['mapping'].values  # cluster_id（可能非连续）

    # 创建cluster映射：将非连续的cluster_id映射到连续的索引（0, 1, 2, ...）
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = np.sort(unique_clusters)  # 确保排序
    cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    index_to_cluster = {idx: cluster for cluster, idx in cluster_to_index.items()}  # 反向映射
    cluster_labels_mapped = np.array([cluster_to_index[cluster] for cluster in cluster_labels])

    print(f"[INFO] Total spikes: {len(all_windows):,}, Unique clusters: {len(unique_clusters)}")
    print(f"[INFO] Original cluster IDs (may be non-continuous): {unique_clusters[:10]}{'...' if len(unique_clusters) > 10 else ''}")
    print(f"[INFO] Mapped to continuous indices: 0 to {len(unique_clusters)-1}")
    
    # 保存cluster映射关系
    cluster_mapping_path = os.path.join(output_dir, 'cluster_id_to_index_mapping.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(cluster_mapping_path, 'w') as f:
        json.dump({
            'cluster_to_index': {str(k): int(v) for k, v in cluster_to_index.items()},
            'index_to_cluster': {str(k): int(v) for k, v in index_to_cluster.items()}
        }, f, indent=2)
    print(f"[INFO] Cluster mapping saved to {cluster_mapping_path}")

    # 平衡数据集
    balanced_indices = []
    for cluster in np.unique(cluster_labels_mapped):
        cluster_indices = np.where(cluster_labels_mapped == cluster)[0]
        if len(cluster_indices) > 8000:
            sampled_indices = np.random.choice(cluster_indices, 8000, replace=False)
        else:
            sampled_indices = cluster_indices
        balanced_indices.extend(sampled_indices)

    np.random.shuffle(balanced_indices)

    balanced_data = all_windows[balanced_indices]
    balanced_labels = cluster_labels_mapped[balanced_indices]

    dataset = ClassificationDataset(balanced_data, balanced_labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = balanced_data.shape[1] * balanced_data.shape[2]
    hidden_size1 = 64
    hidden_size2 = 100  # 100维中间层用于KMeans聚类
    num_classes = len(unique_clusters)
    proj_dim = 128

    for trail in range(1, 6):
        print(f"\n[INFO] === Trail {trail}/5 ===")
        
        model = Spike_Classification_MLP(input_size, hidden_size1, hidden_size2, num_classes, proj_dim)
        model = model.to(device)

        criterion_ce = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001)

        num_epochs = 210
        accuracy_best = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            all_labels = []
            all_predictions = []
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                _, logits, _ = model(batch_data, mode='train')
                ce_loss = criterion_ce(logits, batch_labels)

                optimizer.zero_grad()
                ce_loss.backward()
                optimizer.step()

                predicted = torch.argmax(logits, dim=1)
                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            train_accuracy = accuracy_score(all_labels, all_predictions)

            model.eval()
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    _, logits, _ = model(batch_data, mode='train')
                    predicted = torch.argmax(logits, dim=1)

                    all_labels.extend(batch_labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)

            accuracy = accuracy_score(all_labels, all_predictions)
            
            if epoch % 10 == 0:
                print(f"Trail {trail} - Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Train Acc: {train_accuracy:.4f}, Test Acc: {accuracy:.4f}")
            
            if accuracy > accuracy_best:
                accuracy_best = accuracy
                patience_counter = 0
                # 保存模型，使用与eval一致的命名格式
                model_path = os.path.join(output_dir, f'spike_classification_model_{trail}.pth')
                torch.save(model, model_path)
                print(f"✓ Best model saved with Accuracy: {accuracy_best:.4f}")
            else:
                patience_counter += 1
                if patience_counter == 3:
                    print(f"Early stopping after {epoch+1} epochs with best Accuracy: {accuracy_best:.4f}")
                    break


def main():
    print("[INFO] Starting AutoSort training pipeline (two independent networks)")
    
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
    
    # 步骤1: 预处理 - 生成DataFrame（包含检测到的spike和未匹配的GT spike）
    df = preprocess_data(recording_f, spike_inf, output_dir)
    
    # 步骤2: 训练Detection模型
    train_detection_model(df, output_dir)
    
    # 步骤3: 训练Classification模型（使用100维中间层）
    train_classification_model(df, output_dir)
    
    # 步骤4: AutoSort训练（同时训练两个独立的分类网络）
    # 注意：这是两个独立的网络，共享架构但参数完全独立，不是多任务学习
    train_autosort_model(df, output_dir)
    
    print("\n[INFO] Training pipeline completed successfully")


if __name__ == "__main__":
    main()

