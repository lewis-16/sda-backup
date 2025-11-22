#!/usr/bin/env python
# coding: utf-8
"""
AutoSort评估脚本：对新数据进行完整的spike sorting流程
- 前60秒：校准阶段，建立cluster到neuron的映射关系
- 后续数据：每500ms为单位处理，生成neuron spiketrain
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm
import sys
import time
import json
import pickle
import random
import re

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import umap

# ==================== 从notebook中复制的必要函数 ====================

def detect_local_minimum_in_window(data, window_size=20, std_multiplier=2):
    """在每个滑动窗口范围内检测局部最小值的索引"""
    local_minima_indices = []

    for row in data:
        minima_indices = []
        row = row.astype(np.float32)
        row_mean = np.mean(row)
        row_std = np.std(row)
        threshold = row_mean - std_multiplier * row_std

        for start in range(0, len(row), window_size):
            end = min(start + window_size, len(row))
            window = row[start:end]
            
            if len(window) > 0:
                local_min_index = np.argmin(window)
                local_min_value = window[local_min_index]
                
                if local_min_value < threshold:
                    minima_indices.append(start + local_min_index)  
        
        local_minima_indices.extend(minima_indices)
        local_minima_indices = list(set(local_minima_indices))  

    return local_minima_indices

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


def extract_windows(data, indices, window_size=30):
    """
    根据给定的时间点索引提取窗口
    与train_spike_pipeline.py保持一致：
    - left_sample = 10 (spike前10个采样点)
    - right_sample = 20 (spike后20个采样点)
    - 总共30个采样点: [spike_time - 10, spike_time + 19]
    """
    n_channels, time_length = data.shape
    left_sample = 10   # 与train_spike_pipeline.py保持一致
    right_sample = 20  # 与train_spike_pipeline.py保持一致
    
    # 验证边界
    if np.any(indices < left_sample) or np.any(indices >= time_length - right_sample):
        raise ValueError("Some indices are out of bounds for the given window size.")

    windows = []
    for idx in indices:
        # 提取 [idx - 10, idx + 20)，共30个时间点
        window = data[:, idx - left_sample:idx + right_sample]
        windows.append(window)

    windows = np.array(windows)
    return windows

def compute_cluster_average(sample_data, potent_spike_inf, cluster_column='cluster_predicted'):
    """计算每个cluster对应的sample_data的平均值"""
    cluster_averages = {}
    unique_clusters = potent_spike_inf[cluster_column].unique()
    
    for cluster in unique_clusters:
        cluster_indices = potent_spike_inf[potent_spike_inf[cluster_column] == cluster].index
        cluster_average = sample_data[cluster_indices].mean(axis=0) 
        cluster_averages[cluster] = cluster_average
    
    return cluster_averages

def process_cluster_averages(cluster_averages, channel_indices):
    """对cluster_averages进行处理，保留对应的6个通道"""
    processed_averages = {}
    
    for cluster, avg_matrix in cluster_averages.items():
        max_channel = np.argmax(avg_matrix.max(axis=1))  
        
        for key, indices in channel_indices.items():
            if max_channel in indices:
                selected_channels = avg_matrix[indices, :]
                new_key = f"{cluster}_{key}"
                processed_averages[new_key] = selected_channels
                break
    
    return processed_averages

def compute_spike_shape_and_energy(window: np.ndarray) -> tuple:
    """
    从已经提取的window计算spike的shape和energy（使用所有30个通道）
    
    Args:
        window: (30, 30) 波形数据（所有通道）
    
    Returns:
        shape: (30, 30) 形状模板
        energy: (30,) 能量向量
    """
    # 确保window是(30, 30)的形状
    if window.ndim == 3:
        # 如果是(n, 30, 30)，取第一个
        if window.shape[0] == 1:
            window = window[0]
        else:
            raise ValueError(f"Expected window shape (30, 30) or (1, 30, 30), got {window.shape}")
    elif window.ndim != 2:
        raise ValueError(f"Expected window shape (30, 30), got {window.shape}")
    
    if window.shape != (30, 30):
        raise ValueError(f"Expected window shape (30, 30), got {window.shape}")
    
    # 确保window是float32类型
    snippet = window.astype(np.float32)  # (30, 30)
    
    # 检查snippet是否全为0
    if np.allclose(snippet, 0, atol=1e-10):
        print(f"[WARNING] Window is all zeros!")
        # 如果全为0，返回全0的shape和energy
        shape = np.zeros((30, 30), dtype=np.float32)
        energy = np.zeros(30, dtype=np.float32)
        return shape, energy
    
    # 计算形状：对每条通道内的波形做L2归一化
    normalized_snippet = np.zeros((30, 30), dtype=np.float32)
    for ch_idx in range(30):
        channel_waveform = snippet[ch_idx, :]  # (30,)
        norm = np.linalg.norm(channel_waveform)
        if norm > 1e-10:
            normalized_snippet[ch_idx, :] = channel_waveform / norm
        else:
            # 如果norm太小，保持原值（可能是0）
            normalized_snippet[ch_idx, :] = channel_waveform
    
    # shape就是归一化后的snippet
    shape = normalized_snippet  # (30, 30)
    
    # 计算能量：E = np.sum(snippet**2, axis=1)
    energy = np.sum(snippet**2, axis=1).astype(np.float32)  # (30,)
    
    return shape, energy


def compute_template_score(
    shape1: np.ndarray,
    energy1: np.ndarray,
    shape2: np.ndarray,
    energy2: np.ndarray,
) -> tuple:
    """
    计算两个模板之间的匹配度score
    
    Args:
        shape1: (30, 30) 形状模板1
        energy1: (30,) 能量向量1
        shape2: (30, 30) 形状模板2
        energy2: (30,) 能量向量2
    
    Returns:
        shape_score: 形状匹配度（余弦相似度）
        energy_score: 能量匹配度（余弦相似度）
    """
    # 展平形状模板为向量
    shape1_flat = shape1.flatten()
    shape2_flat = shape2.flatten()
    
    # 计算余弦相似度
    shape_dot = np.dot(shape1_flat, shape2_flat)
    shape_norm1 = np.linalg.norm(shape1_flat)
    shape_norm2 = np.linalg.norm(shape2_flat)
    
    if shape_norm1 > 1e-10 and shape_norm2 > 1e-10:
        shape_score = shape_dot / (shape_norm1 * shape_norm2)
    else:
        shape_score = 0.0
    
    # 计算能量向量的余弦相似度
    energy_dot = np.dot(energy1, energy2)
    energy_norm1 = np.linalg.norm(energy1)
    energy_norm2 = np.linalg.norm(energy2)
    
    if energy_norm1 > 1e-10 and energy_norm2 > 1e-10:
        energy_score = energy_dot / (energy_norm1 * energy_norm2)
    else:
        energy_score = 0.0
    
    return shape_score, energy_score


def normalize_waveform(waveform):
    """
    归一化waveform（L2归一化）
    
    参数:
    waveform: numpy数组，可以是1D或2D
    
    返回:
    normalized_waveform: 归一化后的waveform
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        norm = np.linalg.norm(waveform)
        if norm > 0:
            return waveform / norm
        else:
            return waveform
    else:
        # 如果是2D，对每一行进行归一化
        norms = np.linalg.norm(waveform, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # 避免除零
        return waveform / norms

# ==================== 模型定义 ====================

# 从train_spike_pipeline导入AutoSort相关类
try:
    sys.path.append('/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels')
    from train_spike_pipeline import AutoSort, SimpleClassifier
except ImportError:
    # 如果导入失败，在这里定义（从train_spike_pipeline.py复制）
    class SimpleClassifier(nn.Module):
        """简单的分类器（参考autosort的clssimp类）"""
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
            """提取中间层特征（100维）"""
            x = self.way1(x)
            x = self.way2(x)
            x = self.way3(x)
            return x
    
    class AutoSort(nn.Module):
        """AutoSort模型：两个独立的分类网络"""
        def __init__(self, input_size, num_classes, device):
            super(AutoSort, self).__init__()
            self.clsfier_noise = SimpleClassifier(input_size, 2).to(device)
            self.clsfier_label = SimpleClassifier(input_size, num_classes).to(device)
            self.device = device
        
        def forward(self, x, mode='train'):
            x_flat = x.reshape(x.size(0), -1)
            noise_output = self.clsfier_noise(x_flat)
            label_output = self.clsfier_label(x_flat)
            return noise_output, label_output
        
        def get_intermediate_features(self, x):
            x_flat = x.reshape(x.size(0), -1)
            noise_features = self.clsfier_noise.intermediate_forward(x_flat)
            label_features = self.clsfier_label.intermediate_forward(x_flat)
            return noise_features, label_features

class Spike_Detection_MLP(nn.Module):
    """Spike Detection MLP模型"""
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, n_channels, time_window):
        super(Spike_Detection_MLP, self).__init__()
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
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


# ==================== 主要评估类 ====================

class SpikeSortingEvaluator:
    """Spike Sorting评估器"""
    
    def __init__(self, model_paths, neuron_inf_path, channel_indices, channel_position, gt_spike_inf_path=None, num_classes=None, enable_cluster_gt_validation=False):
        """
        初始化评估器（只支持AutoSort模型）
        
        参数:
        model_paths: 包含模型路径的字典，必须包含'autosort_noise'和'autosort_label'
        neuron_inf_path: neuron_inf.pkl文件路径
        channel_indices: 通道索引字典
        channel_position: 通道位置字典
        gt_spike_inf_path: ground truth spike_inf.tsv文件路径（可选）
        num_classes: AutoSort模型的类别数（可选，会从模型权重中推断）
        enable_cluster_gt_validation: 是否启用cluster到GT neuron的验证功能（可选，仅用于分析，不影响正式流程）
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.channel_indices = channel_indices
        self.channel_position = channel_position
        
        # 加载模型
        print("[INFO] Loading AutoSort models...")
        
        input_size = 30 * 30  # waveform size (30, 30)
        
        # 加载权重文件以推断类别数
        if 'autosort_noise' not in model_paths or 'autosort_label' not in model_paths:
            raise ValueError("model_paths must contain 'autosort_noise' and 'autosort_label'")
        
        noise_state = torch.load(model_paths['autosort_noise'], map_location=self.device)
        label_state = torch.load(model_paths['autosort_label'], map_location=self.device)
        
        # 从label分类器的权重推断类别数
        if isinstance(label_state, dict):
            # 如果是state_dict，从cls.weight或cls.bias推断类别数
            if 'cls.weight' in label_state:
                inferred_num_classes = label_state['cls.weight'].shape[0]
            elif 'cls.bias' in label_state:
                inferred_num_classes = label_state['cls.bias'].shape[0]
            else:
                # 尝试从其他层推断
                if num_classes is None:
                    raise ValueError("Cannot infer num_classes from state_dict. Please provide num_classes.")
                inferred_num_classes = num_classes
        else:
            # 如果是完整模型，从模型的cls层推断
            try:
                inferred_num_classes = label_state.cls.weight.shape[0]
            except:
                if num_classes is None:
                    raise ValueError("Cannot infer num_classes from model. Please provide num_classes.")
                inferred_num_classes = num_classes
        
        print(f"[INFO] Inferred num_classes from model weights: {inferred_num_classes}")
        
        # 使用推断的类别数创建模型
        self.autosort_model = AutoSort(input_size, inferred_num_classes, self.device)
        
        # 处理state_dict（可能是完整模型或state_dict）
        if isinstance(noise_state, dict):
            self.autosort_model.clsfier_noise.load_state_dict(noise_state)
        else:
            # 如果是完整模型，尝试提取state_dict
            try:
                self.autosort_model.clsfier_noise.load_state_dict(noise_state.state_dict())
            except:
                self.autosort_model.clsfier_noise = noise_state
        
        if isinstance(label_state, dict):
            self.autosort_model.clsfier_label.load_state_dict(label_state)
        else:
            try:
                self.autosort_model.clsfier_label.load_state_dict(label_state.state_dict())
            except:
                self.autosort_model.clsfier_label = label_state
        
        self.autosort_model.eval()
        # noise分类器作为detection模型
        self.detection_model = self.autosort_model.clsfier_noise
        
        # 加载neuron信息
        print("[INFO] Loading neuron information...")
        with open(neuron_inf_path, 'rb') as f:
            self.neuron_inf = pickle.load(f)
        
        # 加载021322的template（从neuron_inf_path推断路径）
        print("[INFO] Loading 021322 templates...")
        neuron_inf_dir = os.path.dirname(neuron_inf_path)
        # 假设neuron_inf_path在sorting_new/日期/目录下，需要找到021322的目录
        base_dir = os.path.dirname(os.path.dirname(neuron_inf_dir))  # 回到kilosort_spike_sorting目录
        template_dir_021322 = os.path.join(base_dir, 'sorting_new', '021322')
        shape_template_path = os.path.join(template_dir_021322, 'shape_templates.npy')
        energy_template_path = os.path.join(template_dir_021322, 'energy_templates.npy')
        
        if not os.path.exists(shape_template_path) or not os.path.exists(energy_template_path):
            raise FileNotFoundError(f"Template files not found. Expected:\n  {shape_template_path}\n  {energy_template_path}")
        
        self.shape_templates_021322 = np.load(shape_template_path)  # (n_neurons, 30, 30)
        self.energy_templates_021322 = np.load(energy_template_path)  # (n_neurons, 30)
        print(f"[INFO] Loaded 021322 templates: shape {self.shape_templates_021322.shape}, energy {self.energy_templates_021322.shape}")
        
        # 预处理021322 templates：筛选energy最大的两个通道，其余通道在shape中置0
        print("[INFO] Preprocessing 021322 templates: keeping only top 2 energy channels...")
        n_neurons = self.shape_templates_021322.shape[0]
        for neuron_idx in range(n_neurons):
            energy_template = self.energy_templates_021322[neuron_idx]  # (30,)
            # 找到energy最大的两个通道索引
            top2_channel_indices = np.argsort(energy_template)[-2:][::-1]  # 降序排列，取前2个
            # 创建mask：只有top2通道保留，其余置0
            channel_mask = np.zeros(30, dtype=bool)
            channel_mask[top2_channel_indices] = True
            # 在shape template中，只保留top2通道，其余置0
            shape_template = self.shape_templates_021322[neuron_idx].copy()  # (30, 30)
            shape_template[~channel_mask, :] = 0  # 非top2通道全部置0
            self.shape_templates_021322[neuron_idx] = shape_template
        print(f"[INFO] Preprocessing completed: each neuron's shape template now has only 2 non-zero channels")
        
        # 加载GT template（022522）
        # 尝试从多个可能的路径加载
        gt_template_paths_to_try = []
        
        # 方法1: 从gt_spike_inf_path推断
        if gt_spike_inf_path is not None and os.path.exists(gt_spike_inf_path):
            gt_spike_inf_dir = os.path.dirname(gt_spike_inf_path)
            gt_template_dir = gt_spike_inf_dir
            gt_shape_template_path = os.path.join(gt_template_dir, 'shape_templates.npy')
            gt_energy_template_path = os.path.join(gt_template_dir, 'energy_templates.npy')
            gt_template_paths_to_try.append((gt_shape_template_path, gt_energy_template_path, "from gt_spike_inf_path"))
        
        # 方法2: 从neuron_inf_path推断，找到022522目录
        neuron_inf_dir = os.path.dirname(neuron_inf_path)
        # 假设neuron_inf_path在sorting_new/日期/目录下，需要找到022522目录
        base_dir = os.path.dirname(os.path.dirname(neuron_inf_dir))  # 回到kilosort_spike_sorting目录
        gt_template_dir_022522 = os.path.join(base_dir, 'sorting_new', '022522')
        gt_shape_template_path_022522 = os.path.join(gt_template_dir_022522, 'shape_templates.npy')
        gt_energy_template_path_022522 = os.path.join(gt_template_dir_022522, 'energy_templates.npy')
        gt_template_paths_to_try.append((gt_shape_template_path_022522, gt_energy_template_path_022522, "from 022522 directory"))
        
        # 方法3: 使用绝对路径（如果base_dir推断失败）
        absolute_gt_template_dir = '/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/kilosort_spike_sorting/sorting_new/022522'
        absolute_gt_shape_path = os.path.join(absolute_gt_template_dir, 'shape_templates.npy')
        absolute_gt_energy_path = os.path.join(absolute_gt_template_dir, 'energy_templates.npy')
        gt_template_paths_to_try.append((absolute_gt_shape_path, absolute_gt_energy_path, "absolute path"))
        
        # 尝试加载
        gt_templates_loaded = False
        for gt_shape_path, gt_energy_path, method in gt_template_paths_to_try:
            if os.path.exists(gt_shape_path) and os.path.exists(gt_energy_path):
                print(f"[INFO] Loading GT templates (022522) from {method}...")
                print(f"[INFO] GT template paths:\n  {gt_shape_path}\n  {gt_energy_path}")
                self.shape_templates_gt = np.load(gt_shape_path)  # (n_gt_neurons, 30, 30)
                self.energy_templates_gt = np.load(gt_energy_path)  # (n_gt_neurons, 30)
                print(f"[INFO] Loaded GT templates: shape {self.shape_templates_gt.shape}, energy {self.energy_templates_gt.shape}")
                gt_templates_loaded = True
                break
        
        if not gt_templates_loaded:
            print(f"[WARNING] GT template files not found. Tried:")
            for gt_shape_path, gt_energy_path, method in gt_template_paths_to_try:
                print(f"  Method {method}:")
                print(f"    shape: {gt_shape_path} (exists: {os.path.exists(gt_shape_path)})")
                print(f"    energy: {gt_energy_path} (exists: {os.path.exists(gt_energy_path)})")
            # 即使前面的方法失败，也尝试直接使用绝对路径
            absolute_gt_template_dir = '/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/kilosort_spike_sorting/sorting_new/022522'
            absolute_gt_shape_path = os.path.join(absolute_gt_template_dir, 'shape_templates.npy')
            absolute_gt_energy_path = os.path.join(absolute_gt_template_dir, 'energy_templates.npy')
            if os.path.exists(absolute_gt_shape_path) and os.path.exists(absolute_gt_energy_path):
                print(f"[INFO] Loading GT templates using absolute path...")
                self.shape_templates_gt = np.load(absolute_gt_shape_path)
                self.energy_templates_gt = np.load(absolute_gt_energy_path)
                print(f"[INFO] Loaded GT templates: shape {self.shape_templates_gt.shape}, energy {self.energy_templates_gt.shape}")
            else:
                self.shape_templates_gt = None
                self.energy_templates_gt = None
        
        # neuron_inf的waveform已经在generate_neuron_inf_phy.py中计算为30维（与train/eval一致）
        # 检查neuron_inf中的waveform是否已经归一化
        if len(self.neuron_inf) > 0:
            sample_waveform = self.neuron_inf.iloc[0]['position_waveform']
            if isinstance(sample_waveform, np.ndarray):
                sample_norm = np.linalg.norm(sample_waveform)
                if abs(sample_norm - 1.0) < 1e-5:
                    print("[INFO] neuron_inf中的waveform已经归一化（L2 norm ≈ 1.0）")
                    self.neuron_waveform_normalized = True
                else:
                    print(f"[INFO] neuron_inf中的waveform未归一化（L2 norm = {sample_norm:.4f}），将在匹配时归一化")
                    self.neuron_waveform_normalized = False
            else:
                self.neuron_waveform_normalized = False
        else:
            self.neuron_waveform_normalized = False
        
        # 加载ground truth数据（如果提供）
        self.gt_spike_inf = None
        self.gt_spike_inf_sorted = None
        self.gt_times = None
        self.gt_cluster_to_neuron_mapping = {}
        self.gt_neuron_to_train_neuron_mapping = {}  # GT neuron到训练集neuron的映射
        self.gt_spike_inf_path = gt_spike_inf_path  # 保存路径以便后续使用
        if gt_spike_inf_path is not None and os.path.exists(gt_spike_inf_path):
            print(f"[INFO] Loading ground truth spike data from {gt_spike_inf_path}")
            self.gt_spike_inf = pd.read_csv(gt_spike_inf_path, sep='\t')
            print(f"[INFO] Loaded {len(self.gt_spike_inf):,} ground truth spikes")
            # 预处理：按时间排序以便快速匹配
            self.gt_spike_inf_sorted = self.gt_spike_inf.sort_values('time').reset_index(drop=True)
            self.gt_times = self.gt_spike_inf_sorted['time'].values
            # GT cluster到neuron的映射将在calibration阶段建立（需要recording数据）
            # GT neuron到训练集neuron的映射也将在calibration阶段建立
        
        # 校准阶段变量
        self.calibration_complete = False
        
        # GT template（022522）到021322 neuron的映射（在__init__中计算）
        self.gt_neuron_to_021322_neuron_mapping = {}  # GT neuron索引到021322 neuron索引的映射，-1表示unmatched
        
        # 在__init__阶段计算GT template到021322 template的映射
        if self.shape_templates_gt is not None and self.energy_templates_gt is not None:
            print("[INFO] Computing GT template to 021322 neuron mapping in __init__...")
            n_gt_neurons = self.shape_templates_gt.shape[0]
            n_neurons_021322 = self.shape_templates_021322.shape[0]
            
            gt_shape_scores = np.zeros((n_gt_neurons, n_neurons_021322), dtype=np.float32)
            gt_energy_scores = np.zeros((n_gt_neurons, n_neurons_021322), dtype=np.float32)
            
            for gt_idx in range(n_gt_neurons):
                gt_shape = self.shape_templates_gt[gt_idx]  # (30, 30)
                gt_energy = self.energy_templates_gt[gt_idx]  # (30,)
                
                for template_idx in range(n_neurons_021322):
                    shape_score, energy_score = compute_template_score(
                        gt_shape,
                        gt_energy,
                        self.shape_templates_021322[template_idx],
                        self.energy_templates_021322[template_idx],
                    )
                    gt_shape_scores[gt_idx, template_idx] = shape_score
                    gt_energy_scores[gt_idx, template_idx] = energy_score
            
            # 应用阈值：shape < 0.9 记为0，energy < 0.9 记为0
            gt_shape_scores_filtered = gt_shape_scores.copy()
            gt_energy_scores_filtered = gt_energy_scores.copy()
            gt_shape_scores_filtered[gt_shape_scores_filtered < 0.9] = 0
            gt_energy_scores_filtered[gt_energy_scores_filtered < 0.9] = 0
            
            # 计算sum_scores并找到最佳匹配
            gt_sum_scores = gt_shape_scores_filtered + gt_energy_scores_filtered
            for gt_idx in range(n_gt_neurons):
                if np.sum(gt_sum_scores[gt_idx]) == 0:
                    # 所有score都是0，unmatch
                    self.gt_neuron_to_021322_neuron_mapping[gt_idx] = -1  # -1表示unmatch
                else:
                    # 取argmax
                    best_021322_idx = np.argmax(gt_sum_scores[gt_idx])
                    self.gt_neuron_to_021322_neuron_mapping[gt_idx] = best_021322_idx
            
            matched_count = sum(1 for v in self.gt_neuron_to_021322_neuron_mapping.values() if v != -1)
            print(f"[INFO] GT template mapping completed: {matched_count}/{n_gt_neurons} matched to 021322 neurons")
        else:
            print("[WARNING] GT templates not loaded, skipping GT template to 021322 mapping in __init__")
        
        # 验证功能开关（仅用于分析，不影响正式流程）
        self.enable_cluster_gt_validation = enable_cluster_gt_validation
        
    def calibrate_first_10min(self, recording, sampling_rate=10000, output_dir=None):
        """
        前100秒校准阶段：
        1. 阈值检测（std_multiplier=2.4, window_size=10）
        2. spike_detection（使用detection模型筛选）
        3. 对通过detection的spike计算shape和energy（使用所有30个通道）
        4. 与021322的template计算匹配score
        5. 保存结果用于分析
        
        参数:
        recording: 录音数据
        sampling_rate: 采样率（默认10000）
        output_dir: 输出目录（可选），如果提供，将保存shape/energy和匹配score
        """
        print("\n" + "="*60)
        print("Calibration Phase")
        print("="*60)
        
        calibration_duration = 100  # 秒（10分钟）
        calibration_frames = calibration_duration * sampling_rate
        
        # 提取前10分钟数据
        print(f"[INFO] Extracting first {calibration_duration} seconds of data for calibration...")
        calibration_data = recording.get_traces(
            start_frame=0,
            end_frame=calibration_frames
        ).T
        print(f"[INFO] Calibration data shape: {calibration_data.shape}")
        
        # 1. 阈值检测（std_multiplier=2.4, window_size=10）
        print("[INFO] Performing threshold detection...")
        window_size = 30  # 与训练时一致 (30, 30)
        left_sample = 10   # 与train_spike_pipeline.py保持一致
        right_sample = 20  # 与train_spike_pipeline.py保持一致
        
        threshold_result = detect_local_minimum_in_window(
            calibration_data,
            std_multiplier=2.4,
            window_size=10
        )
        threshold_result = np.array(threshold_result, dtype=np.int64)
        
        # 过滤边界附近的spike
        valid_threshold_indices = threshold_result[
            (threshold_result >= left_sample) & 
            (threshold_result < calibration_frames - right_sample)
        ]
        
        print(f"[INFO] Threshold detection found {len(valid_threshold_indices):,} potential spikes")
        
        if len(valid_threshold_indices) == 0:
            raise RuntimeError("No spikes detected by threshold detection in calibration window.")
        
        # 提取窗口 (30, 30)
        calibration_windows = extract_windows(
            calibration_data,
            valid_threshold_indices,
            window_size=window_size
        )
        print(f"[INFO] Extracted {len(calibration_windows):,} windows from threshold detection")
        
        # 统计有多少GT spike包含在阈值检测提取出来的里面
        if self.gt_spike_inf_sorted is not None and self.gt_times is not None:
            gt_times = self.gt_times
            gt_mask = (
                (gt_times >= left_sample) &
                (gt_times < calibration_frames - right_sample)
            )
            gt_times_in_calibration = gt_times[gt_mask]
            
            # 使用map_gt_annotation函数匹配，看有多少GT spike被检测到
            gt_match_indices_temp = map_gt_annotation(
                valid_threshold_indices,
                gt_times_in_calibration,
                time_tolerance=1
            )
            matched_gt_count = np.sum(gt_match_indices_temp >= 0)
            total_gt_count = len(gt_times_in_calibration)
            print(f"[INFO] GT spikes in threshold detection: {matched_gt_count:,}/{total_gt_count:,} ({matched_gt_count/total_gt_count*100:.2f}%)")
        else:
            print("[WARNING] GT spike_inf not available, cannot compute GT spike coverage")
        
        if len(calibration_windows) == 0:
            raise RuntimeError("No windows extracted from threshold detection results.")
        
        # 保存原始的所有windows和indices（用于后续分析/可视化）
        all_calibration_windows = calibration_windows.copy()
        all_valid_calibration_indices = valid_threshold_indices.copy()
        
        # 2. spike_detection（使用detection模型筛选）
        print("[INFO] Applying spike_detection model to filter potential spikes...")
        detection_scores = []
        detection_keep_mask = []
        detection_100d_features = None
        
        with torch.no_grad():
            batch_size = 4096
            for i in range(0, len(calibration_windows), batch_size):
                batch = calibration_windows[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)  # (batch, 30, 30)
                
                # 使用AutoSort的noise分类器（期望30x30输入）
                # noise分类器输出2类logits，取spike类（索引1）的概率
                noise_output = self.detection_model(batch_tensor.reshape(batch_tensor.size(0), -1))
                # 使用softmax获取spike类的概率
                probs = torch.softmax(noise_output, dim=1)
                outputs = probs[:, 1]  # spike类的概率
        
                scores = outputs.cpu().numpy()
                detection_scores.append(scores)
                detection_keep_mask.append((scores > 0.1).astype(bool))
        
        detection_scores = np.concatenate(detection_scores)
        detection_keep_mask = np.concatenate(detection_keep_mask)
        
        print(f"[INFO] Detection model filtered: {detection_keep_mask.sum():,}/{len(detection_keep_mask):,} spikes passed ({(detection_keep_mask.sum()/len(detection_keep_mask)*100):.2f}%)")
        
        # 应用detection筛选
        calibration_windows = calibration_windows[detection_keep_mask]
        valid_calibration_indices = valid_threshold_indices[detection_keep_mask]
        detection_scores_filtered = detection_scores[detection_keep_mask]
        
        print(f"[INFO] After detection filtering: {len(calibration_windows):,} spikes remaining")
        
        if len(calibration_windows) == 0:
            raise RuntimeError("No spikes passed detection model filtering.")
        
        # 3. 匹配检测到的potential spike到GT spike_inf（只匹配match neuron的GT spike）
        print("[INFO] Matching detected spikes to GT spike_inf...")
        gt_match_indices = None
        
        if self.gt_spike_inf_sorted is not None and self.gt_times is not None:
            # 先加载GT neuron映射信息，用于筛选match neuron
            gt_neuron_name_to_idx_temp = {}
            gt_neuron_inf_path = None
            if self.gt_spike_inf_path is not None:
                gt_spike_inf_dir = os.path.dirname(self.gt_spike_inf_path)
                gt_neuron_inf_path = os.path.join(gt_spike_inf_dir, 'neuron_inf.pkl')
            
            if gt_neuron_inf_path is None or not os.path.exists(gt_neuron_inf_path):
                absolute_gt_neuron_inf_path = '/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/kilosort_spike_sorting/sorting_new/022522/neuron_inf.pkl'
                if os.path.exists(absolute_gt_neuron_inf_path):
                    gt_neuron_inf_path = absolute_gt_neuron_inf_path
            
            if gt_neuron_inf_path is not None and os.path.exists(gt_neuron_inf_path):
                try:
                    with open(gt_neuron_inf_path, 'rb') as f:
                        gt_neuron_inf = pickle.load(f)
                    for idx, (_, row) in enumerate(gt_neuron_inf.iterrows()):
                        neuron_name = row['Neuron']
                        gt_neuron_name_to_idx_temp[neuron_name] = idx
                except Exception as e:
                    gt_neuron_name_to_idx_temp = {}
            
            # 如果从neuron_inf加载失败，回退到从spike_inf推断
            if not gt_neuron_name_to_idx_temp and self.gt_spike_inf_sorted is not None:
                has_neuron_col = 'neuron' in self.gt_spike_inf_sorted.columns or 'Neuron' in self.gt_spike_inf_sorted.columns
                neuron_col = 'neuron' if 'neuron' in self.gt_spike_inf_sorted.columns else 'Neuron'
                if has_neuron_col:
                    unique_gt_neurons = sorted(self.gt_spike_inf_sorted[neuron_col].dropna().unique())
                    for idx, neuron_name in enumerate(unique_gt_neurons):
                        gt_neuron_name_to_idx_temp[neuron_name] = idx
            
            # 筛选出match neuron的GT spike
            gt_times = self.gt_times
            gt_mask = (
                (gt_times >= left_sample) &
                (gt_times < calibration_frames - right_sample)
            )
            gt_spike_inf_calibration = self.gt_spike_inf_sorted[gt_mask].reset_index(drop=True)
            gt_times_in_calibration = gt_times[gt_mask]
            
            if len(gt_neuron_name_to_idx_temp) > 0 and len(self.gt_neuron_to_021322_neuron_mapping) > 0:
                has_neuron_col = 'neuron' in gt_spike_inf_calibration.columns or 'Neuron' in gt_spike_inf_calibration.columns
                neuron_col = 'neuron' if 'neuron' in gt_spike_inf_calibration.columns else 'Neuron'
                
                if has_neuron_col:
                    # 筛选出match neuron的GT spike
                    match_neuron_gt_mask = np.zeros(len(gt_spike_inf_calibration), dtype=bool)
                    for i, row in gt_spike_inf_calibration.iterrows():
                        gt_neuron_name = row[neuron_col]
                        if not pd.isna(gt_neuron_name):
                            gt_idx = gt_neuron_name_to_idx_temp.get(gt_neuron_name, -1)
                            if gt_idx != -1:
                                mapped_021322 = self.gt_neuron_to_021322_neuron_mapping.get(gt_idx, -1)
                                if mapped_021322 != -1:
                                    match_neuron_gt_mask[i] = True
                    
                    # 只使用match neuron的GT spike进行匹配
                    match_neuron_gt_times = gt_times_in_calibration[match_neuron_gt_mask]
                    match_neuron_gt_spike_inf = gt_spike_inf_calibration[match_neuron_gt_mask].reset_index(drop=True)
                    match_neuron_gt_count = np.sum(match_neuron_gt_mask)
                    total_gt_count = len(gt_times_in_calibration)
                    
                    print(f"[INFO] Match neuron GT spikes: {match_neuron_gt_count:,}/{total_gt_count:,} ({match_neuron_gt_count/total_gt_count*100:.2f}%)")
                    
                    # 匹配detected spike到match neuron的GT spike
                    gt_match_indices = map_gt_annotation(
                        valid_calibration_indices,
                        match_neuron_gt_times,
                        time_tolerance=1
                    )
                    
                    # 统计匹配数量（总数是match neuron GT spike的数量）
                    # matched_count是匹配到的match neuron GT spike数量（因为map_gt_annotation确保每个GT spike最多只匹配一个detected spike）
                    matched_count = np.sum(gt_match_indices >= 0)
                    unmatched_gt_count = match_neuron_gt_count - matched_count
                    unmatched_detected_count = np.sum(gt_match_indices < 0)
                    print(f"[INFO] Matched {matched_count:,}/{match_neuron_gt_count:,} match neuron GT spikes ({matched_count/match_neuron_gt_count*100:.2f}%)")
                    print(f"[INFO] Unmatched match neuron GT spikes: {unmatched_gt_count:,} ({unmatched_gt_count/match_neuron_gt_count*100:.2f}%)")
                    print(f"[INFO] Detected spikes not matched to match neuron GT: {unmatched_detected_count:,} ({unmatched_detected_count/len(valid_calibration_indices)*100:.2f}%)")
                    
                    # 保存原始索引映射（用于后续处理）
                    # gt_match_indices现在对应match_neuron_gt_spike_inf的索引
                    # 需要转换为原始gt_spike_inf_calibration的索引
                    original_gt_match_indices = np.full(len(valid_calibration_indices), -1, dtype=np.int64)
                    match_neuron_original_indices = np.where(match_neuron_gt_mask)[0]
                    for i in range(len(gt_match_indices)):
                        if gt_match_indices[i] >= 0:
                            original_gt_match_indices[i] = match_neuron_original_indices[gt_match_indices[i]]
                    gt_match_indices = original_gt_match_indices
                else:
                    # 没有neuron列，使用所有GT spike匹配
                    gt_match_indices = map_gt_annotation(
                        valid_calibration_indices,
                        gt_times_in_calibration,
                        time_tolerance=1
                    )
                    matched_count = np.sum(gt_match_indices >= 0)
                    print(f"[INFO] No neuron column found, matched {matched_count:,}/{len(gt_times_in_calibration):,} GT spikes")
            else:
                # 没有映射信息，使用所有GT spike匹配
                gt_match_indices = map_gt_annotation(
                    valid_calibration_indices,
                    gt_times_in_calibration,
                    time_tolerance=1
                )
                matched_count = np.sum(gt_match_indices >= 0)
                print(f"[INFO] No GT neuron mapping available, matched {matched_count:,}/{len(gt_times_in_calibration):,} GT spikes")
        else:
            print("[WARNING] GT spike_inf not available, skipping GT matching")
            gt_match_indices = np.full(len(valid_calibration_indices), -1, dtype=np.int64)
        
        # 如果gt_match_indices还是None，初始化为-1
        if gt_match_indices is None:
            gt_match_indices = np.full(len(valid_calibration_indices), -1, dtype=np.int64)
        
        # 计算每个spike的shape和energy（使用所有30个通道）
        print("[INFO] Computing shape and energy for each spike...")
        n_spikes = len(calibration_windows)
        n_neurons = self.shape_templates_021322.shape[0]
        
        all_spike_shapes = []
        all_spike_energies = []
        
        for i in range(n_spikes):
            window = calibration_windows[i]  # (30, 30)
            shape, energy = compute_spike_shape_and_energy(window)
            all_spike_shapes.append(shape)
            all_spike_energies.append(energy)
        
        all_spike_shapes = np.array(all_spike_shapes)  # (n_spikes, 30, 30)
        all_spike_energies = np.array(all_spike_energies)  # (n_spikes, 30)
        print(f"[INFO] Computed shapes: {all_spike_shapes.shape}, energies: {all_spike_energies.shape}")
        
        # 与021322的template计算匹配score
        print(f"[INFO] Computing matching scores with {n_neurons} templates from 021322...")
        shape_scores = np.zeros((n_spikes, n_neurons), dtype=np.float32)
        energy_scores = np.zeros((n_spikes, n_neurons), dtype=np.float32)
        
        for spike_idx in range(n_spikes):
            spike_shape = all_spike_shapes[spike_idx]
            spike_energy = all_spike_energies[spike_idx]
            
            for template_idx in range(n_neurons):
                shape_score, energy_score = compute_template_score(
                    spike_shape,
                    spike_energy,
                    self.shape_templates_021322[template_idx],
                    self.energy_templates_021322[template_idx],
                )
                shape_scores[spike_idx, template_idx] = shape_score
                energy_scores[spike_idx, template_idx] = energy_score
            
        
        print(f"[INFO] Computed matching scores: shape {shape_scores.shape}, energy {energy_scores.shape}")
        
        # 1. 创建DataFrame，包含time, gt_match_index, match_gt, match_train, is_match_neuron
        print("[INFO] Creating DataFrame for spike matching results...")
        spike_times = valid_calibration_indices.copy()  # 保存spike时间（检测到的spike）
        matching_df = pd.DataFrame({
            'time': spike_times,
            'gt_match_index': gt_match_indices,  # GT spike的索引（-1表示未匹配）
            'match_gt': [-1] * n_spikes,  # 初始化为-1（unmatched_time），后续填入021322 neuron id
            'match_train': [-1] * n_spikes,  # 初始化为-1（unmatched），后续填入template matching结果
            'is_match_neuron': [False] * n_spikes  # 标记GT neuron是否match到021322
        })
        
        # 2. 通过时间匹配GT spike_inf，通过映射填入021322 neuron id
        # 首先建立GT neuron名称到GT template索引的映射
        # 优先从GT neuron_inf.pkl文件中获取正确的顺序
        gt_neuron_name_to_idx = {}
        
        # 无论GT templates是否加载，都需要建立GT neuron名称到索引的映射
        # 尝试从GT neuron_inf.pkl加载（如果存在）
        gt_neuron_inf_path = None
        if self.gt_spike_inf_path is not None:
            gt_spike_inf_dir = os.path.dirname(self.gt_spike_inf_path)
            gt_neuron_inf_path = os.path.join(gt_spike_inf_dir, 'neuron_inf.pkl')
        
        # 如果从gt_spike_inf_path推断失败，尝试绝对路径
        if gt_neuron_inf_path is None or not os.path.exists(gt_neuron_inf_path):
            absolute_gt_neuron_inf_path = '/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/kilosort_spike_sorting/sorting_new/022522/neuron_inf.pkl'
            if os.path.exists(absolute_gt_neuron_inf_path):
                gt_neuron_inf_path = absolute_gt_neuron_inf_path
        
        if gt_neuron_inf_path is not None and os.path.exists(gt_neuron_inf_path):
            print(f"[INFO] Loading GT neuron_inf from {gt_neuron_inf_path}...")
            try:
                with open(gt_neuron_inf_path, 'rb') as f:
                    gt_neuron_inf = pickle.load(f)
                # GT template的索引顺序应该与GT neuron_inf中的neuron顺序一致
                for idx, (_, row) in enumerate(gt_neuron_inf.iterrows()):
                    neuron_name = row['Neuron']
                    gt_neuron_name_to_idx[neuron_name] = idx
                print(f"[INFO] Built GT neuron name to index mapping from neuron_inf: {len(gt_neuron_name_to_idx)} neurons")

                if self.shape_templates_gt is not None:
                    if len(gt_neuron_name_to_idx) != self.shape_templates_gt.shape[0]:
                        print(f"[WARNING] GT neuron count ({len(gt_neuron_name_to_idx)}) != GT template count ({self.shape_templates_gt.shape[0]})")
            except Exception as e:
                print(f"[WARNING] Failed to load GT neuron_inf: {e}")
                gt_neuron_name_to_idx = {}
        
        # 如果从neuron_inf加载失败，回退到从spike_inf推断
        if not gt_neuron_name_to_idx and self.gt_spike_inf_sorted is not None:
            has_neuron_col = 'neuron' in self.gt_spike_inf_sorted.columns or 'Neuron' in self.gt_spike_inf_sorted.columns
            neuron_col = 'neuron' if 'neuron' in self.gt_spike_inf_sorted.columns else 'Neuron'
            if has_neuron_col:
                # 获取所有唯一的GT neuron名称，并按字母顺序排序（这样索引就是稳定的）
                unique_gt_neurons = sorted(self.gt_spike_inf_sorted[neuron_col].dropna().unique())
                # 假设GT template的索引顺序与sorted unique neurons的顺序一致
                for idx, neuron_name in enumerate(unique_gt_neurons):
                    gt_neuron_name_to_idx[neuron_name] = idx
                print(f"[INFO] Built GT neuron name to index mapping from spike_inf (sorted): {len(gt_neuron_name_to_idx)} neurons")
                print(f"[INFO] GT neuron names (first 10): {unique_gt_neurons[:10]}")
                if self.shape_templates_gt is not None:
                    print(f"[INFO] GT template shape: {self.shape_templates_gt.shape[0]} templates")
                    if len(gt_neuron_name_to_idx) != self.shape_templates_gt.shape[0]:
                        print(f"[WARNING] GT neuron count ({len(gt_neuron_name_to_idx)}) != GT template count ({self.shape_templates_gt.shape[0]})")
            else:
                print(f"[WARNING] No neuron column found in GT spike_inf. Available columns: {self.gt_spike_inf_sorted.columns.tolist()}")
        
        # 2. 通过gt_match_indices获取GT neuron信息，填入match_gt和is_match_neuron
        if self.gt_spike_inf_sorted is not None and len(self.gt_spike_inf_sorted) > 0:
            print("[INFO] Mapping detected spikes to GT neurons using gt_match_indices...")
            
            has_neuron_col = 'neuron' in self.gt_spike_inf_sorted.columns or 'Neuron' in self.gt_spike_inf_sorted.columns
            neuron_col = 'neuron' if 'neuron' in self.gt_spike_inf_sorted.columns else 'Neuron'
            
            if has_neuron_col:
                # 获取校准时间范围内的GT spike_inf
                gt_mask = (
                    (self.gt_times >= left_sample) &
                    (self.gt_times < calibration_frames - right_sample)
                )
                gt_spike_inf_calibration = self.gt_spike_inf_sorted[gt_mask].reset_index(drop=True)
                
                matched_count = 0
                for i in range(len(matching_df)):
                    gt_match_idx = matching_df.loc[i, 'gt_match_index']
                    
                    if gt_match_idx >= 0 and gt_match_idx < len(gt_spike_inf_calibration):
                        # 找到匹配的GT spike
                        gt_match = gt_spike_inf_calibration.iloc[gt_match_idx]
                        
                        # 获取GT neuron名称并转换为索引，然后通过映射转换为021322 neuron id
                        gt_neuron_name = gt_match[neuron_col]
                        if not pd.isna(gt_neuron_name):
                            gt_idx = gt_neuron_name_to_idx.get(gt_neuron_name, -1)
                            if gt_idx != -1:
                                # 通过映射转换为021322 neuron id
                                mapped_021322 = self.gt_neuron_to_021322_neuron_mapping.get(gt_idx, -1)
                                matching_df.loc[i, 'match_gt'] = mapped_021322
                                # 标记是否为match neuron（mapped_021322 != -1 表示match）
                                matching_df.loc[i, 'is_match_neuron'] = (mapped_021322 != -1)
                                if mapped_021322 != -1:
                                    matched_count += 1
                            else:
                                # GT neuron名称未找到，保持-1（unmatched_time），标记为unmatch
                                matching_df.loc[i, 'match_gt'] = -1
                                matching_df.loc[i, 'is_match_neuron'] = False
                        else:
                            # GT spike没有neuron信息，保持-1（unmatched_time），标记为unmatch
                            matching_df.loc[i, 'match_gt'] = -1
                            matching_df.loc[i, 'is_match_neuron'] = False
                    else:
                        # 没有匹配的GT spike（gt_match_idx == -1），标记为unmatch
                        matching_df.loc[i, 'match_gt'] = -1
                        matching_df.loc[i, 'is_match_neuron'] = False
                
                match_neuron_count = matching_df['is_match_neuron'].sum()
                unmatch_neuron_count = (~matching_df['is_match_neuron']).sum()
                print(f"[INFO] GT neuron mapping: {matched_count}/{n_spikes} spikes matched to 021322 neurons via GT mapping")
                print(f"[INFO] Match neuron spikes: {match_neuron_count:,}, Unmatch neuron spikes: {unmatch_neuron_count:,}")
            else:
                print("[WARNING] No neuron column found in GT spike_inf")
                matching_df['is_match_neuron'] = False
        else:
            print("[WARNING] GT spike_inf not available, all match_gt will be -1 (unmatched_time)")
            # 如果没有GT数据，所有spike都标记为unmatch
            matching_df['is_match_neuron'] = False
        
        # 3. 计算spike到021322的映射（应用阈值），填入match_train
        print("[INFO] Computing spike to 021322 neuron mapping based on template scores...")
        # 应用阈值：shape < 0 记为0，energy < 0.8 记为0
        shape_scores_filtered = shape_scores.copy()
        energy_scores_filtered = energy_scores.copy()
        shape_scores_filtered[shape_scores_filtered < 0.1] = 0
        energy_scores_filtered[energy_scores_filtered < 0.6] = 0
        
        # 计算sum_scores并找到最佳匹配
        # sum_scores[i, j] = shape_scores_filtered[i, j] + energy_scores_filtered[i, j]
        # 如果某个template的shape和energy都是0，则sum为0
        sum_scores = shape_scores_filtered + energy_scores_filtered
        for spike_idx in range(n_spikes):
            if np.all(shape_scores_filtered[spike_idx] == 0) or np.all(energy_scores_filtered[spike_idx] == 0):
                matching_df.loc[spike_idx, 'match_train'] = -1  # -1表示unmatch
            else:
                # 否则取argmax（sum_score最大的那个template）
                best_021322_idx = np.argmax(sum_scores[spike_idx])
                matching_df.loc[spike_idx, 'match_train'] = best_021322_idx
        
        matched_spikes = np.sum(matching_df['match_train'] != -1)
        unmatched_spikes = np.sum(matching_df['match_train'] == -1)
        print(f"[INFO] Template matching: {matched_spikes}/{n_spikes} spikes matched to 021322 neurons")
        print(f"[INFO] Unmatched spikes: {unmatched_spikes}/{n_spikes} (all template sums are 0)")
        
        # 4. 基于match_gt和match_train计算confusion matrix
        print("[INFO] Building confusion matrix from DataFrame...")
        # 获取所有021322 neuron的索引（包括-1表示unmatch）
        all_021322_indices = set(matching_df['match_train'].values)
        all_021322_indices.update(matching_df['match_gt'].values)
        all_021322_indices = sorted([idx for idx in all_021322_indices if idx != -1])  # 排除-1，稍后单独处理
        
        # 创建confusion matrix（行：match_train (predicted)，列：match_gt (GT)）
        n_unique_021322 = len(all_021322_indices)
        confusion_matrix = np.zeros((n_unique_021322 + 1, n_unique_021322 + 1), dtype=np.int32)  # +1 for unmatched
        
        # 创建索引映射字典以提高效率
        idx_to_row = {idx: i for i, idx in enumerate(all_021322_indices)}
        idx_to_row[-1] = n_unique_021322  # unmatched映射到最后一行/列
        
        for i in range(len(matching_df)):
            pred_idx = matching_df.loc[i, 'match_train']
            gt_idx = matching_df.loc[i, 'match_gt']
            
            # 使用字典映射
            pred_row = idx_to_row.get(pred_idx, n_unique_021322)
            gt_col = idx_to_row.get(gt_idx, n_unique_021322)
            
            confusion_matrix[pred_row, gt_col] += 1
        
        print(f"[INFO] Confusion matrix shape: {confusion_matrix.shape}")
        print(f"[INFO] Total spikes in confusion matrix: {np.sum(confusion_matrix)}")
        
        # 计算准确率（对角线元素，排除unmatched行/列）
        if n_unique_021322 > 0:
            diagonal_elements = confusion_matrix[:-1, :-1].diagonal()
            correct = diagonal_elements.sum()
            total = np.sum(confusion_matrix[:-1, :-1])
            accuracy = (correct / total * 100) if total > 0 else 0.0
            print(f"[INFO] Confusion matrix statistics:")
            print(f"  Total spikes: {len(matching_df):,}")
            print(f"  Correctly classified (diagonal): {correct:,}")
            print(f"  Overall accuracy: {accuracy:.2f}%")
        
        # 调试信息：检查mapping的分布
        unique_pred, counts_pred = np.unique(matching_df['match_train'].values, return_counts=True)
        for pred_idx, count in zip(unique_pred[:10], counts_pred[:10]):
            if pred_idx == -1:
                print(f"  Unmatched: {count}")
            else:
                print(f"  Neuron_{pred_idx+1}: {count}")
        if len(unique_pred) > 10:
            print(f"  ... (total {len(unique_pred)} unique predictions)")
        
        unique_gt, counts_gt = np.unique(matching_df['match_gt'].values, return_counts=True)
        for gt_idx, count in zip(unique_gt[:10], counts_gt[:10]):
            if gt_idx == -1:
                print(f"  Unmatched_time: {count}")
            else:
                print(f"  Neuron_{gt_idx+1}: {count}")
        if len(unique_gt) > 10:
            print(f"  ... (total {len(unique_gt)} unique GT labels)")
        
        # 检查是否有匹配的情况
        matching_count = np.sum((matching_df['match_train'] != -1) & (matching_df['match_gt'] != -1))
        
        # 5. 保存结果和绘制heatmap
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存shape和energy
            np.save(os.path.join(output_dir, 'calibration_spike_shapes.npy'), all_spike_shapes)
            np.save(os.path.join(output_dir, 'calibration_spike_energies.npy'), all_spike_energies)
            print(f"[INFO] Saved spike shapes and energies to {output_dir}")
            
            # 保存匹配scores
            np.save(os.path.join(output_dir, 'calibration_shape_scores.npy'), shape_scores)
            np.save(os.path.join(output_dir, 'calibration_energy_scores.npy'), energy_scores)
            print(f"[INFO] Saved matching scores to {output_dir}")
            
            # 保存spike时间索引
            np.save(os.path.join(output_dir, 'calibration_spike_indices.npy'), valid_calibration_indices)
            print(f"[INFO] Saved spike indices to {output_dir}")
            
            # 保存matching DataFrame
            matching_df.to_csv(os.path.join(output_dir, 'calibration_matching_results.csv'), index=False)
            print(f"[INFO] Saved matching DataFrame to {output_dir}")
            
            # 保存mapping结果
            if self.gt_neuron_to_021322_neuron_mapping:
                with open(os.path.join(output_dir, 'gt_neuron_to_021322_neuron_mapping.pkl'), 'wb') as f:
                    pickle.dump(self.gt_neuron_to_021322_neuron_mapping, f)
            print(f"[INFO] Saved mapping results to {output_dir}")
            
            # 保存confusion matrix
            np.save(os.path.join(output_dir, 'calibration_confusion_matrix.npy'), confusion_matrix)
            print(f"[INFO] Saved confusion matrix to {output_dir}")
            
            # 绘制confusion matrix heatmap
            self._plot_calibration_confusion_matrix_heatmap(
                confusion_matrix, all_021322_indices, output_dir
            )
            
            # 计算并保存spike_detection的metrics
            if detection_keep_mask is not None:
                self._compute_and_save_detection_metrics(
                    all_valid_calibration_indices,
                    detection_keep_mask, 
                    recording,
                    sampling_rate,
                    output_dir,
                    matching_df=matching_df  # 传入matching_df以使用match/unmatch标签
                )
            else:
                print("[INFO] Detection metrics skipped (detection step was bypassed).")
            
            # 生成spike_detection的UMAP可视化（使用100维中间层特征）
            if detection_100d_features is not None and detection_keep_mask is not None:
                self._plot_spike_detection_umap(
                    detection_100d_features,
                    detection_keep_mask,
                    all_calibration_windows,
                    all_valid_calibration_indices,
                    recording,
                    sampling_rate,
                    output_dir
                )
            else:
                print("[INFO] Detection UMAP skipped (detection step was bypassed).")
        
        self.calibration_complete = True
        print("[INFO] Calibration phase completed successfully!")
    
    
    def _build_cluster_neuron_mapping(self, calibration_windows, calibration_clusters):
        """
        建立cluster到neuron的映射关系（仿照notebook的方法）
        1. 先计算每个cluster的平均waveform（使用所有30个通道）
        2. 找到最大幅度通道，确定probe_group，保留对应的6个通道
        3. 为每个cluster计算位置和position_waveform
        4. 然后与neuron_inf中的neuron进行匹配
        
        返回:
        cluster_info_df: 包含已匹配cluster的position_1, position_2, position_waveform的DataFrame
        """
        unique_clusters = np.unique(calibration_clusters)
        print(f"[INFO] Total clusters: {len(unique_clusters)}")
        
        # 第一步：计算每个cluster的平均waveform（使用所有30个通道）
        print("[INFO] Computing cluster average waveforms...")
        cluster_averages = compute_cluster_average(
            calibration_windows,
            pd.DataFrame({'cluster_predicted': calibration_clusters}),
            cluster_column='cluster_predicted'
        )
        
        # 第二步：处理cluster_averages，找到最大值所在的通道，并保留对应的6个通道
        print("[INFO] Processing cluster averages to find probe groups...")
        processed_cluster_waveforms = process_cluster_averages(
            cluster_averages,
            self.channel_indices
        )
        
        print(f"[INFO] Processed cluster waveforms: {len(processed_cluster_waveforms)}")
        
        # 第三步：为每个cluster计算位置和position_waveform
        print("[INFO] Computing cluster positions and position waveforms...")
        cluster_info_list = []
        
        for cluster_key, waveform in processed_cluster_waveforms.items():
            cluster_id, probe_group = cluster_key.split('_')
            cluster_id = int(cluster_id)
            
            # 计算位置
            channels = self.channel_indices[probe_group]
            a_squared = [np.sum(waveform[j, :]**2) for j in range(len(channels))]
            
            sum_x_a = 0
            sum_y_a = 0
            sum_a = 0
            
            for j, channel in enumerate(channels):
                x_i, y_i = self.channel_position.get(channel, [0, 0])
                a_i_sq = a_squared[j]
                sum_x_a += x_i * a_i_sq
                sum_y_a += y_i * a_i_sq
                sum_a += a_i_sq
            
            if sum_a == 0:
                continue
            
            x_hat = sum_x_a / sum_a
            y_hat = sum_y_a / sum_a
            
            # 计算position_waveform
            distances = []
            for channel in channels:
                x_channel, y_channel = self.channel_position.get(channel, [np.nan, np.nan])
                if not (np.isnan(x_channel) or np.isnan(y_channel)):
                    distance = np.sqrt((x_hat - x_channel)**2 + (y_hat - y_channel)**2)
                    distances.append(distance)
                else:
                    distances.append(np.inf)
            
            if not distances or all(d == np.inf for d in distances):
                continue
            
            distances = np.array(distances, dtype=np.float32)
            
            # IDW插值
            weights = 1.0 / (np.power(distances, 2, dtype=np.float32) + 1e-10)
            if np.any(distances == 0):
                zero_idx = np.where(distances == 0)[0][0]
                position_waveform = waveform[zero_idx, :].astype(np.float32)
            else:
                weights /= weights.sum()
                # 与train/eval一致：window_size=30
                position_waveform = np.zeros(30, dtype=np.float32)
                for t in range(30):
                    position_waveform[t] = float(np.dot(waveform[:, t], weights))
            
            cluster_info_list.append({
                'cluster_predicted': cluster_id,
                'probe_group': int(probe_group),
                'position_1': x_hat,
                'position_2': y_hat,
                'position_waveform': position_waveform,  # 保存未归一化的waveform用于匹配
                'waveform': waveform
            })
        
        cluster_info_df = pd.DataFrame(cluster_info_list)
        print(f"[INFO] Generated cluster info: {len(cluster_info_df)} clusters")
        
        # 第四步：与neuron_inf中的neuron进行匹配
        # 流程：对于每个neuron，遍历所有cluster，使用neuron的channel_id重新计算cluster的特征并匹配
        print("[INFO] Matching clusters to neurons: for each neuron, recompute all cluster features using neuron's channel_id...")
        from scipy.stats import pearsonr
        
        # 初始化映射：每个cluster记录匹配的neuron和相关性
        cluster_matches = {}  # {cluster_id: {'neuron': neuron_label, 'corr': corr, 'pval': pval, 'cluster_waveform': ..., 'neuron_waveform': ..., 'all_matches': [...]}}
        for cluster_id in unique_clusters:
            cluster_matches[cluster_id] = {
                'neuron': None, 
                'corr': -1, 
                'pval': None, 
                'cluster_waveform': None, 
                'neuron_waveform': None,
                'all_matches': []  # 记录所有满足条件的匹配（用于统计）
            }
        
        # 保存所有位置匹配成功的记录（用于返回DataFrame）
        position_matched_records = []  # 列表，每个元素是 {cluster_id, neuron, cluster_waveform, neuron_waveform, pval, corr}
        
        position_threshold = 10
        waveform_threshold = 0.95
        
        # 统计信息
        total_neuron_cluster_pairs = 0
        position_passed_count = 0
        waveform_passed_count = 0
        
        # 外层循环：遍历neuron_inf中的每个neuron
        for neuron_idx, neuron_row in self.neuron_inf.iterrows():
            neuron_label = neuron_row['Neuron']
            neuron_channel_id = neuron_row.get('channel_id')
            neuron_waveform = neuron_row['position_waveform']
            neuron_x = float(neuron_row['position_1'])
            neuron_y = float(neuron_row['position_2'])
            
            # 确保neuron_waveform是numpy数组
            if not isinstance(neuron_waveform, np.ndarray):
                neuron_waveform = np.array(neuron_waveform)
            
            # 如果没有channel_id，跳过该neuron（无法重新计算）
            if neuron_channel_id is None:
                continue
            
            # 确保channel_id是列表
            if not isinstance(neuron_channel_id, (list, np.ndarray)):
                if isinstance(neuron_channel_id, str):
                    import ast
                    try:
                        neuron_channel_id = ast.literal_eval(neuron_channel_id)
                    except:
                        continue
                else:
                    continue
            
            neuron_channel_id = list(neuron_channel_id)
            
            # 统计该neuron的匹配情况
            neuron_position_passed = 0
            neuron_waveform_passed = 0
            
            # 内层循环：遍历所有cluster
            for cluster_id in unique_clusters:
                total_neuron_cluster_pairs += 1
                # 获取该cluster的所有spike
                cluster_mask = calibration_clusters == cluster_id
                cluster_spikes = calibration_windows[cluster_mask]  # (n_spikes, 30, 30)
                
                if len(cluster_spikes) == 0:
                    continue
                
                # 使用该neuron的channel_id提取cluster的对应通道
                cluster_spikes_channels = cluster_spikes[:, neuron_channel_id, :]  # (n_spikes, n_channels, 30)
                
                # 计算每个spike的位置和waveform，然后取平均
                cluster_positions_x = []
                cluster_positions_y = []
                cluster_waveforms = []
                
                for spike_idx in range(len(cluster_spikes_channels)):
                    spike_waveform_channels = cluster_spikes_channels[spike_idx]  # (n_channels, 30)
                    
                    # 计算位置（基于该neuron的channels）
                    a_squared = [np.sum(spike_waveform_channels[j, :]**2) for j in range(len(neuron_channel_id))]
                    
                    sum_x_a = 0
                    sum_y_a = 0
                    sum_a = 0
                    
                    for j, channel in enumerate(neuron_channel_id):
                        x_i, y_i = self.channel_position.get(channel, [0, 0])
                        a_i_sq = a_squared[j]
                        sum_x_a += x_i * a_i_sq
                        sum_y_a += y_i * a_i_sq
                        sum_a += a_i_sq
                    
                    if sum_a == 0:
                        continue
                    
                    spike_x = sum_x_a / sum_a
                    spike_y = sum_y_a / sum_a
                    cluster_positions_x.append(spike_x)
                    cluster_positions_y.append(spike_y)
                    
                    # 计算position_waveform（基于该neuron的channels）
                    distances = []
                    for channel in neuron_channel_id:
                        x_channel, y_channel = self.channel_position.get(channel, [np.nan, np.nan])
                        if not (np.isnan(x_channel) or np.isnan(y_channel)):
                            distance = np.sqrt((spike_x - x_channel)**2 + (spike_y - y_channel)**2)
                            distances.append(distance)
                        else:
                            distances.append(np.inf)
                    
                    if not distances or all(d == np.inf for d in distances):
                        continue
                    
                    distances = np.array(distances, dtype=np.float32)
                    
                    # IDW插值计算position_waveform
                    weights = 1.0 / (np.power(distances, 2, dtype=np.float32) + 1e-10)
                    if np.any(distances == 0):
                        zero_idx = np.where(distances == 0)[0][0]
                        spike_position_waveform = spike_waveform_channels[zero_idx, :].astype(np.float32)
                    else:
                        weights /= weights.sum()
                        spike_position_waveform = np.zeros(30, dtype=np.float32)
                        for t in range(30):
                            spike_position_waveform[t] = float(np.dot(spike_waveform_channels[:, t], weights))
                    
                    cluster_waveforms.append(spike_position_waveform)
                
                if len(cluster_waveforms) == 0:
                    continue
                
                # 计算平均位置和waveform
                cluster_x = np.mean(cluster_positions_x)
                cluster_y = np.mean(cluster_positions_y)
                cluster_avg_waveform = np.mean(cluster_waveforms, axis=0)  # (30,)
                
                # 检查位置是否接近neuron的位置
                position_diff_x = abs(cluster_x - neuron_x)
                position_diff_y = abs(cluster_y - neuron_y)
                
                if position_diff_x > position_threshold or position_diff_y > position_threshold:
                    # 位置不匹配
                    continue
                
                # 位置匹配通过
                position_passed_count += 1
                neuron_position_passed += 1
                
                # 与neuron的waveform比较（归一化后）
                # 注意：neuron_inf中的waveform现在应该是30维（与train/eval一致）
                # 但为了兼容旧的31维数据，保留对齐逻辑
                if len(cluster_avg_waveform) == 30:
                    # 如果neuron_waveform是31维（旧数据），去掉第一个点对齐到30维
                    if len(neuron_waveform) == 31:
                        neuron_waveform_aligned = neuron_waveform[1:31]  # 去掉第一个点，保留后30个点
                    elif len(neuron_waveform) == 30:
                        neuron_waveform_aligned = neuron_waveform
                    else:
                        continue
                    
                    if len(neuron_waveform_aligned) == 30:
                        # 归一化waveform
                        cluster_waveform_norm = normalize_waveform(cluster_avg_waveform)
                        # 如果neuron_inf中的waveform已经归一化，则不需要再次归一化
                        if self.neuron_waveform_normalized:
                            neuron_waveform_norm = normalize_waveform(neuron_waveform_aligned)
                        else:
                            neuron_waveform_norm = normalize_waveform(neuron_waveform_aligned)
                    
                    corr, pval = pearsonr(cluster_waveform_norm, neuron_waveform_norm)
                    
                    # 保存位置匹配成功的记录（无论waveform相关性是否足够）
                    position_matched_records.append({
                        'kmeans_cluster': cluster_id,
                        'neuron': neuron_label,
                        'kmeans_cluster_waveform': cluster_waveform_norm.copy(),
                        'neuron_inf_waveform': neuron_waveform_norm.copy(),
                        'pval': pval,
                        'corr': corr  # 也保存corr以便后续分析
                    })
                    
                    if corr > waveform_threshold:
                        # 波形相关性足够
                        waveform_passed_count += 1
                        neuron_waveform_passed += 1
                        
                        # 记录所有满足条件的匹配
                        cluster_matches[cluster_id]['all_matches'].append({
                            'neuron': neuron_label,
                            'corr': corr,
                            'pval': pval
                        })
                        
                        # 如果相关性更高，更新该cluster的匹配
                        if corr > cluster_matches[cluster_id]['corr']:
                            cluster_matches[cluster_id]['neuron'] = neuron_label
                            cluster_matches[cluster_id]['corr'] = corr
                            cluster_matches[cluster_id]['pval'] = pval
                            cluster_matches[cluster_id]['cluster_waveform'] = cluster_waveform_norm.copy()
                            cluster_matches[cluster_id]['neuron_waveform'] = neuron_waveform_norm.copy()
                    else:
                        # 波形相关性不够
                        pass
                else:
                    # waveform长度不匹配
                    pass
            
        # 输出统计信息
        print("\n" + "="*60)
        print("[INFO] Matching Statistics:")
        print("="*60)
        print(f"Total neuron-cluster pairs checked: {total_neuron_cluster_pairs}")
        print(f"Position passed: {position_passed_count} ({position_passed_count/total_neuron_cluster_pairs*100:.2f}%)")
        print(f"Waveform passed (corr > {waveform_threshold}): {waveform_passed_count} ({waveform_passed_count/total_neuron_cluster_pairs*100:.2f}%)")
        print(f"Position failed: {total_neuron_cluster_pairs - position_passed_count}")
        print(f"Waveform failed (after position passed): {position_passed_count - waveform_passed_count}")
        print("="*60 + "\n")
        
        # 建立最终映射关系，并统计多匹配情况
        multi_match_clusters = []  # 记录匹配到多个neuron的cluster
        for cluster_id, match_info in cluster_matches.items():
            self.cluster_to_neuron_mapping[cluster_id] = match_info['neuron']
            
            # 保存所有匹配信息（用于后续分析）
            if len(match_info['all_matches']) > 0:
                sorted_matches = sorted(match_info['all_matches'], key=lambda x: x['corr'], reverse=True)
                self.cluster_multi_matches[cluster_id] = sorted_matches
            
            # 统计匹配到多个neuron的cluster
            if len(match_info['all_matches']) > 1:
                multi_match_clusters.append({
                    'cluster_id': cluster_id,
                    'selected_neuron': match_info['neuron'],
                    'selected_corr': match_info['corr'],
                    'all_matches': sorted_matches
                })
        
        # 输出多匹配统计信息
        if len(multi_match_clusters) > 0:
            print(f"\n[INFO] Clusters matched to multiple neurons: {len(multi_match_clusters)}")
            print("[INFO] These clusters matched multiple neurons (showing top 10):")
            for i, info in enumerate(multi_match_clusters[:10]):
                print(f"  Cluster {info['cluster_id']}: selected {info['selected_neuron']} (corr={info['selected_corr']:.3f})")
                matches_str = ', '.join([f"{m['neuron']}({m['corr']:.3f})" for m in info['all_matches']])
                print(f"    All matches: {matches_str}")
            if len(multi_match_clusters) > 10:
                print(f"  ... and {len(multi_match_clusters) - 10} more clusters with multiple matches")
        # 归一化waveform并更新cluster_info_df
        for idx, cluster_row in cluster_info_df.iterrows():
            cluster_info_df.at[idx, 'position_waveform'] = normalize_waveform(cluster_row['position_waveform'])
        
        # 输出映射结果
        matched_count = sum(1 for v in self.cluster_to_neuron_mapping.values() if v is not None)
        unmatched_count = sum(1 for v in self.cluster_to_neuron_mapping.values() if v is None)
        print(f"[INFO] Cluster to neuron mapping: {matched_count} matched, {unmatched_count} unmatched")
        print(f"[INFO] Total neurons in neuron_inf: {len(self.neuron_inf)}")
        print(f"[INFO] Total clusters: {len(unique_clusters)}")
        
        # 构建匹配结果的DataFrame（包含所有位置匹配成功的记录）
        # 注意：这里保存的是所有位置匹配成功的记录，不仅仅是最终waveform相关性也足够的
        match_results_df = pd.DataFrame(position_matched_records)
        
        # 如果DataFrame不为空，移除corr列（只保留要求的5列）
        if len(match_results_df) > 0:
            match_results_df = match_results_df[['kmeans_cluster', 'neuron', 'kmeans_cluster_waveform', 'neuron_inf_waveform', 'pval']].copy()
        
        # 返回cluster_info_df以便保存（只保留必要的列）
        cluster_info_df_output = cluster_info_df[['cluster_predicted', 'position_1', 'position_2', 'position_waveform']].copy()
        
        # 返回两个DataFrame：cluster_info_df和match_results_df
        return cluster_info_df_output, match_results_df
    
    def _build_gt_cluster_to_neuron_mapping(self, recording, sampling_rate=10000):
        """
        建立ground truth cluster到neuron的映射关系
        使用与calibration阶段相同的逻辑：对每个neuron，使用其channel_id来计算GT cluster的特征
        
        参数:
        recording: 录音数据（用于提取waveform）
        sampling_rate: 采样率（默认10000）
        """
        if self.gt_spike_inf is None:
            return
        
        # 检查是否有cluster列
        if 'cluster' not in self.gt_spike_inf.columns:
            print("[WARNING] Ground truth spike_inf does not have 'cluster' column, skipping GT cluster mapping")
            return
        
        # 如果GT数据中已经有neuron列，需要建立GT neuron到训练集neuron的映射
        if 'neuron' in self.gt_spike_inf.columns or 'Neuron' in self.gt_spike_inf.columns:
            neuron_col = 'neuron' if 'neuron' in self.gt_spike_inf.columns else 'Neuron'
            print(f"[INFO] Ground truth spike_inf has '{neuron_col}' column")
            # 检查neuron列是否有缺失值
            neuron_count = self.gt_spike_inf[neuron_col].notna().sum()
            print(f"[INFO] {neuron_count}/{len(self.gt_spike_inf)} spikes have neuron labels")
            
            # 尝试加载当前日期的neuron_inf（用于建立GT neuron到训练集neuron的映射）
            # 从self.gt_spike_inf_path推断当前日期的neuron_inf路径
            if self.gt_spike_inf_path and isinstance(self.gt_spike_inf_path, str):
                gt_spike_inf_dir = os.path.dirname(self.gt_spike_inf_path)
            else:
                gt_spike_inf_dir = None
            
            if gt_spike_inf_dir:
                current_date_neuron_inf_path = os.path.join(gt_spike_inf_dir, 'neuron_inf.pkl')
                if os.path.exists(current_date_neuron_inf_path):
                    print(f"[INFO] Loading current date neuron_inf from {current_date_neuron_inf_path}")
                    with open(current_date_neuron_inf_path, 'rb') as f:
                        current_date_neuron_inf = pickle.load(f)
                    print(f"[INFO] Loaded {len(current_date_neuron_inf)} neurons from current date")
                    
                    # 建立GT neuron到训练集neuron的映射
                    print("[INFO] Building GT neuron to training neuron mapping...")
                    self._build_gt_neuron_to_train_neuron_mapping(current_date_neuron_inf)
                else:
                    print(f"[WARNING] Current date neuron_inf not found at {current_date_neuron_inf_path}, cannot map GT neurons to training neurons")
            else:
                print("[WARNING] Cannot determine current date neuron_inf path, cannot map GT neurons to training neurons")
            
            # 不需要建立cluster到neuron的映射，因为可以直接从spike_inf中获取neuron
            return
        
        # 如果没有直接的neuron映射，使用与calibration相同的逻辑：从recording提取waveform并匹配
        print("[INFO] Building GT cluster to neuron mapping using waveform matching...")
        
        # 获取recording数据
        waveform_matrix = recording.get_traces().astype("float32")
        window_size = 30  # 与训练时一致 (30, 30)
        left_sample = 10   # 与train_spike_pipeline.py保持一致
        right_sample = 20  # 与train_spike_pipeline.py保持一致
        
        # 过滤边界附近的spike
        valid_gt_spikes = self.gt_spike_inf[
            (self.gt_spike_inf['time'] >= left_sample) & 
            (self.gt_spike_inf['time'] < waveform_matrix.shape[0] - right_sample)
        ].copy()
        
        if len(valid_gt_spikes) == 0:
            print("[WARNING] No valid GT spikes after boundary filtering")
            return
        
        # 获取所有唯一的ground truth cluster
        unique_gt_clusters = valid_gt_spikes['cluster'].unique()
        print(f"[INFO] Found {len(unique_gt_clusters)} unique ground truth clusters")
        
        # 建立映射关系：对每个neuron，使用其channel_id计算GT cluster的特征
        position_threshold = 20
        waveform_threshold = 0.9
        
        # 初始化映射：每个GT cluster记录匹配的neuron和相关性
        gt_cluster_matches = {}  # {cluster_id: {'neuron': neuron_label, 'corr': corr}}
        for cluster_id in unique_gt_clusters:
            gt_cluster_matches[cluster_id] = {'neuron': None, 'corr': -1}
        
        # 为每个neuron计算所有GT cluster的特征并匹配
        for neuron_idx, neuron_row in self.neuron_inf.iterrows():
            neuron_label = neuron_row['Neuron']
            neuron_channel_id = neuron_row.get('channel_id')
            
            if neuron_channel_id is None:
                continue
            
            # 确保channel_id是列表
            if not isinstance(neuron_channel_id, (list, np.ndarray)):
                if isinstance(neuron_channel_id, str):
                    import ast
                    try:
                        neuron_channel_id = ast.literal_eval(neuron_channel_id)
                    except:
                        continue
                else:
                    continue
            
            neuron_channel_id = list(neuron_channel_id)
            neuron_x = float(neuron_row['position_1'])
            neuron_y = float(neuron_row['position_2'])
            neuron_waveform = neuron_row['position_waveform']
            if not isinstance(neuron_waveform, np.ndarray):
                neuron_waveform = np.array(neuron_waveform)
            
            # 对每个GT cluster，使用该neuron的channel_id计算位置和waveform
            for cluster_id in unique_gt_clusters:
                # 获取该cluster的所有spike
                cluster_spikes = valid_gt_spikes[valid_gt_spikes['cluster'] == cluster_id]
                
                if len(cluster_spikes) == 0:
                    continue
                
                # 提取该cluster的所有spike waveform
                cluster_waveforms = []
                cluster_positions_x = []
                cluster_positions_y = []
                
                for _, spike_row in cluster_spikes.iterrows():
                    spike_time = int(spike_row['time'])
                    # 与train_spike_pipeline.py保持一致：[spike_time - 10, spike_time + 19]，共30个时间点
                    start = spike_time - left_sample   # spike_time - 10
                    end = spike_time + right_sample    # spike_time + 20
                    
                    if start < 0 or end > waveform_matrix.shape[0]:
                        continue
                    if end - start != window_size:
                        continue
                    
                    # 提取对应channels的waveform
                    # waveform_matrix形状: (time_points, n_channels)
                    spike_waveform_all = waveform_matrix[start:end, :].T  # (30, 30)
                    spike_waveform_channels = spike_waveform_all[neuron_channel_id, :]  # (n_channels, 30)
                    
                    # 计算位置（基于该neuron的channels）
                    a_squared = [np.sum(spike_waveform_channels[j, :]**2) for j in range(len(neuron_channel_id))]
                    
                    sum_x_a = 0
                    sum_y_a = 0
                    sum_a = 0
                    
                    for j, channel in enumerate(neuron_channel_id):
                        x_i, y_i = self.channel_position.get(channel, [0, 0])
                        a_i_sq = a_squared[j]
                        sum_x_a += x_i * a_i_sq
                        sum_y_a += y_i * a_i_sq
                        sum_a += a_i_sq
                    
                    if sum_a == 0:
                        continue
                    
                    spike_x = sum_x_a / sum_a
                    spike_y = sum_y_a / sum_a
                    cluster_positions_x.append(spike_x)
                    cluster_positions_y.append(spike_y)
                    
                    # 计算position_waveform（基于该neuron的channels）
                    distances = []
                    for channel in neuron_channel_id:
                        x_channel, y_channel = self.channel_position.get(channel, [np.nan, np.nan])
                        if not (np.isnan(x_channel) or np.isnan(y_channel)):
                            distance = np.sqrt((spike_x - x_channel)**2 + (spike_y - y_channel)**2)
                            distances.append(distance)
                        else:
                            distances.append(np.inf)
                    
                    if not distances or all(d == np.inf for d in distances):
                        continue
                    
                    distances = np.array(distances, dtype=np.float32)
                    weights = 1.0 / (np.power(distances, 2, dtype=np.float32) + 1e-10)
                    if np.any(distances == 0):
                        zero_idx = np.where(distances == 0)[0][0]
                        spike_position_waveform = spike_waveform_channels[zero_idx, :].astype(np.float32)
                    else:
                        weights /= weights.sum()
                        spike_position_waveform = np.zeros(window_size, dtype=np.float32)
                        for t in range(window_size):
                            spike_position_waveform[t] = float(np.dot(spike_waveform_channels[:, t], weights))
                    
                    cluster_waveforms.append(spike_position_waveform)
                
                if len(cluster_waveforms) == 0:
                    continue
                
                # 计算平均位置和waveform
                cluster_x = np.mean(cluster_positions_x)
                cluster_y = np.mean(cluster_positions_y)
                cluster_avg_waveform = np.mean(cluster_waveforms, axis=0)  # (30,)
                
                # 检查位置是否接近neuron的位置
                if abs(cluster_x - neuron_x) > position_threshold or abs(cluster_y - neuron_y) > position_threshold:
                    continue
                
                # 与neuron的waveform比较（归一化后）
                # 注意：neuron_inf中的waveform可能是31维，需要对齐到30维
                if len(cluster_avg_waveform) == 30:
                    # 如果neuron_waveform是31维，取中间30个点对齐（去掉第一个或最后一个点）
                    if len(neuron_waveform) == 31:
                        neuron_waveform_aligned = neuron_waveform[1:31]  # 去掉第一个点，保留后30个点
                    elif len(neuron_waveform) == 30:
                        neuron_waveform_aligned = neuron_waveform
                    else:
                        continue
                    
                    if len(neuron_waveform_aligned) == 30:
                        # 归一化waveform
                        cluster_waveform_norm = normalize_waveform(cluster_avg_waveform)
                        # 如果neuron_inf中的waveform已经归一化，则不需要再次归一化
                        if self.neuron_waveform_normalized:
                            neuron_waveform_norm = normalize_waveform(neuron_waveform_aligned)
                        else:
                            neuron_waveform_norm = normalize_waveform(neuron_waveform_aligned)
                    
                        corr, _ = pearsonr(cluster_waveform_norm, neuron_waveform_norm)
                        
                        # 如果相关性更高，更新该cluster的匹配
                        if corr > waveform_threshold and corr > gt_cluster_matches[cluster_id]['corr']:
                            gt_cluster_matches[cluster_id]['neuron'] = neuron_label
                            gt_cluster_matches[cluster_id]['corr'] = corr
        
        # 建立最终映射关系
        for cluster_id, match_info in gt_cluster_matches.items():
            self.gt_cluster_to_neuron_mapping[cluster_id] = match_info['neuron']
            if match_info['neuron'] is not None:
                print(f"[INFO] GT Cluster {cluster_id} -> {match_info['neuron']} (corr: {match_info['corr']:.3f})")
        
        matched_gt = sum(1 for v in self.gt_cluster_to_neuron_mapping.values() if v is not None)
        print(f"[INFO] Ground truth cluster to neuron mapping: {matched_gt}/{len(unique_gt_clusters)} matched")
    
    def _build_gt_neuron_to_train_neuron_mapping(self, current_date_neuron_inf):
        """
        建立GT neuron（当前日期）到训练集neuron（021322）的映射关系
        基于位置和waveform相似性
        
        参数:
        current_date_neuron_inf: 当前日期的neuron_inf DataFrame
        """
        if current_date_neuron_inf.empty or self.neuron_inf.empty:
            print("[WARNING] Cannot build mapping: empty neuron_inf")
            return
        
        position_threshold = 10.0
        waveform_threshold = 0.95
        
        print(f"[INFO] Mapping {len(current_date_neuron_inf)} GT neurons to {len(self.neuron_inf)} training neurons...")
        
        # 对每个GT neuron，找到最匹配的训练集neuron
        for gt_idx, gt_row in current_date_neuron_inf.iterrows():
            gt_neuron = gt_row['Neuron']
            gt_pos_x = float(gt_row['position_1'])
            gt_pos_y = float(gt_row['position_2'])
            gt_waveform = gt_row['position_waveform']
            
            if not isinstance(gt_waveform, np.ndarray):
                gt_waveform = np.array(gt_waveform, dtype=np.float32)
            else:
                gt_waveform = gt_waveform.astype(np.float32)
            
            best_match = None
            best_corr = -1
            
            # 遍历训练集的所有neuron
            for train_idx, train_row in self.neuron_inf.iterrows():
                train_neuron = train_row['Neuron']
                train_pos_x = float(train_row['position_1'])
                train_pos_y = float(train_row['position_2'])
                train_waveform = train_row['position_waveform']
                
                if not isinstance(train_waveform, np.ndarray):
                    train_waveform = np.array(train_waveform, dtype=np.float32)
                else:
                    train_waveform = train_waveform.astype(np.float32)
                
                # 检查位置是否接近
                pos_diff_x = abs(gt_pos_x - train_pos_x)
                pos_diff_y = abs(gt_pos_y - train_pos_y)
                if pos_diff_x > position_threshold or pos_diff_y > position_threshold:
                    continue
                
                # 检查waveform相似性
                # 现在应该都是30维（与train/eval一致），但保留31维的兼容性检查
                if len(gt_waveform) == 30 and len(train_waveform) == 30:
                    # 归一化waveform
                    gt_waveform_norm = normalize_waveform(gt_waveform)
                    if self.neuron_waveform_normalized:
                        train_waveform_norm = train_waveform
                    else:
                        train_waveform_norm = normalize_waveform(train_waveform)
                    
                    corr, _ = pearsonr(gt_waveform_norm, train_waveform_norm)
                    
                    if corr > waveform_threshold and corr > best_corr:
                        best_corr = corr
                        best_match = train_neuron
            
            if best_match is not None:
                self.gt_neuron_to_train_neuron_mapping[gt_neuron] = best_match
                print(f"[INFO] GT {gt_neuron} -> Training {best_match} (corr: {best_corr:.3f})")
        
        matched_count = len(self.gt_neuron_to_train_neuron_mapping)
        print(f"[INFO] GT neuron to training neuron mapping: {matched_count}/{len(current_date_neuron_inf)} matched")
    
    def _plot_umap_calibration_features(self, calibration_windows, calibration_clusters, calibration_embeddings,
                                       valid_calibration_indices, recording, sampling_rate, output_dir):
        """
        生成UMAP散点图，展示kmeans mapping的neuron和GT neuron的PCA降维后的特征分布
        两张图使用相同的50000个样本
        注意：使用PCA降维后的30维特征（与KMeans聚类时一致）
        
        参数:
        calibration_windows: 校准阶段的spike windows
        calibration_clusters: 校准阶段的kmeans cluster标签
        calibration_embeddings: 校准阶段的100维特征（way3输出），会通过PCA降维到30维
        valid_calibration_indices: 校准阶段的有效时间索引
        recording: 录音数据（用于提取GT spike特征）
        sampling_rate: 采样率
        output_dir: 输出目录
        """
        try:
            import umap
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError as e:
            print(f"[WARNING] Cannot generate UMAP plot: {e}")
            return
        
        print("[INFO] Generating UMAP visualization from PCA-reduced features...")
        
        # 注意：calibration_embeddings传入的是100维特征，但为了与KMeans一致，需要使用PCA降维后的30维特征
        # 应用PCA降维（与KMeans聚类时一致）
        if self.pca is None:
            raise RuntimeError("PCA model not fitted. Calibration phase must be completed first.")
        calibration_embeddings_pca = self.pca.transform(calibration_embeddings)
        
        # 1. 从calibration数据中采样50000个样本（确保两张图使用相同的样本）
        n_samples = min(50000, len(calibration_embeddings_pca))
        sample_indices = np.random.choice(len(calibration_embeddings_pca), n_samples, replace=False)
        sample_indices = np.sort(sample_indices)  # 排序以便后续匹配
        
        # 获取采样后的特征和cluster标签（使用PCA降维后的30维特征）
        kmeans_features = calibration_embeddings_pca[sample_indices]  # (n_samples, 30)
        kmeans_cluster_labels = calibration_clusters[sample_indices]  # (n_samples,)
        sampled_calibration_indices = valid_calibration_indices[sample_indices]  # 对应的时间索引
        
        # 2. 为这些采样样本匹配GT neuron（通过时间匹配）
        gt_neuron_labels = []
        if self.gt_spike_inf is not None:
            # 获取校准阶段的时间范围
            calibration_duration = 100  # 100秒
            calibration_frames = calibration_duration * sampling_rate
            
            # 过滤GT spike（在校准时间范围内）
            left_sample = 10
            right_sample = 20
            gt_spikes_calibration = self.gt_spike_inf[
                (self.gt_spike_inf['time'] >= left_sample) & 
                (self.gt_spike_inf['time'] < calibration_frames - right_sample)
            ].copy()
            
            if len(gt_spikes_calibration) > 0:
                # 对每个采样样本，通过时间匹配找到对应的GT neuron
                time_tolerance = 1  # 1个采样点
                for sampled_time in sampled_calibration_indices:
                    # 在GT spike中查找匹配的时间
                    matched_gt = gt_spikes_calibration[
                        (gt_spikes_calibration['time'] >= sampled_time - time_tolerance) &
                        (gt_spikes_calibration['time'] <= sampled_time + time_tolerance)
                    ]
                    
                    if len(matched_gt) > 0:
                        # 取最近的匹配
                        distances = np.abs(matched_gt['time'].values - sampled_time)
                        nearest_idx = np.argmin(distances)
                        gt_neuron = matched_gt.iloc[nearest_idx].get('neuron', matched_gt.iloc[nearest_idx].get('Neuron', None))
                        
                        # 将GT neuron映射到trainset neuron（如果存在映射）
                        if gt_neuron is not None and not pd.isna(gt_neuron):
                            mapped_train_neuron = self.gt_neuron_to_train_neuron_mapping.get(gt_neuron, gt_neuron)
                            gt_neuron_labels.append(mapped_train_neuron)
                        else:
                            gt_neuron_labels.append(None)
                    else:
                        gt_neuron_labels.append(None)
            else:
                gt_neuron_labels = [None] * len(sampled_calibration_indices)
        else:
            gt_neuron_labels = [None] * len(sampled_calibration_indices)
        
        # 3. 使用相同的特征进行UMAP降维
        all_features = kmeans_features
        print(f"[INFO] Running UMAP on {len(all_features)} samples...")
        
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_features)-1))
        embedding = reducer.fit_transform(all_features)
        
        # 两张图使用相同的embedding
        kmeans_embedding = embedding
        gt_embedding = embedding  # 使用相同的embedding
        
        # 4. 将KMeans cluster映射到neuron
        kmeans_neuron_labels = []
        for cluster_id in kmeans_cluster_labels:
            neuron = self.cluster_to_neuron_mapping.get(cluster_id, None)
            kmeans_neuron_labels.append(neuron)
        
        # 5. 创建统一的neuron颜色映射（确保KMeans和GT使用相同颜色）
        def extract_neuron_number(neuron_name):
            """从Neuron_xx中提取数字用于排序"""
            if isinstance(neuron_name, str) and neuron_name.startswith('Neuron_'):
                match = re.search(r'Neuron_(\d+)', neuron_name)
                if match:
                    return int(match.group(1))
            return float('inf')  # 如果无法提取数字，放在最后
        
        all_unique_neurons = set()
        for neuron in kmeans_neuron_labels:
            if neuron is not None:
                all_unique_neurons.add(neuron)
        for neuron in gt_neuron_labels:
            if neuron is not None:
                all_unique_neurons.add(neuron)
        
        # 按照Neuron_xx的数字顺序排序
        all_unique_neurons = sorted(list(all_unique_neurons), key=extract_neuron_number)
        neuron_to_color = {neuron: i for i, neuron in enumerate(all_unique_neurons)}
        
        # 6. 创建PDF（两张图使用相同的样本和embedding）
        pdf_path = os.path.join(output_dir, 'calibration_match_results_umap.pdf')
        with PdfPages(pdf_path) as pdf:
            # 图1: KMeans mapping得到的neuron
            fig, ax = plt.subplots(figsize=(8, 8))
            kmeans_colors = [neuron_to_color.get(neuron, -1) if neuron is not None else -1 for neuron in kmeans_neuron_labels]
            # 过滤掉未映射的点（-1）
            valid_mask = np.array([c >= 0 for c in kmeans_colors])
            if np.sum(valid_mask) > 0:
                scatter = ax.scatter(kmeans_embedding[valid_mask, 0], kmeans_embedding[valid_mask, 1], 
                                   c=np.array(kmeans_colors)[valid_mask], cmap='tab20', 
                                   s=10, alpha=0.6, edgecolors='none', vmin=0, vmax=len(all_unique_neurons)-1)
            # 绘制未映射的点（灰色）
            unmapped_mask = ~valid_mask
            if np.sum(unmapped_mask) > 0:
                ax.scatter(kmeans_embedding[unmapped_mask, 0], kmeans_embedding[unmapped_mask, 1], 
                          c='#d3d3d3', s=10, alpha=0.3, edgecolors='none', label='Unmapped')
            ax.set_title(f'UMAP: KMeans Mapped Neuron (n={len(kmeans_features)})', 
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')  # xy比为1:1
            ax.set_xticks([])  # 去除xticks
            ax.set_yticks([])  # 去除yticks
            ax.grid(False)  # 去除grid
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # 图2: GT Neuron（使用相同的样本和embedding，trainset mapping的neuron）
            fig, ax = plt.subplots(figsize=(8, 8))
            gt_colors = [neuron_to_color.get(label, -1) if label is not None else -1 for label in gt_neuron_labels]
            # 过滤掉未映射的点（-1）
            valid_mask = np.array([c >= 0 for c in gt_colors])
            if np.sum(valid_mask) > 0:
                scatter = ax.scatter(gt_embedding[valid_mask, 0], gt_embedding[valid_mask, 1], 
                                   c=np.array(gt_colors)[valid_mask], cmap='tab20', 
                                   s=10, alpha=0.6, edgecolors='none', vmin=0, vmax=len(all_unique_neurons)-1)
            # 绘制未映射的点（灰色）
            unmapped_mask = ~valid_mask
            if np.sum(unmapped_mask) > 0:
                ax.scatter(gt_embedding[unmapped_mask, 0], gt_embedding[unmapped_mask, 1], 
                          c='#d3d3d3', s=10, alpha=0.3, edgecolors='none', label='Unmapped')
            ax.set_title(f'UMAP: GT Neuron (Trainset Mapped, n={len(gt_neuron_labels)})', 
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')  # xy比为1:1
            ax.set_xticks([])  # 去除xticks
            ax.set_yticks([])  # 去除yticks
            ax.grid(False)  # 去除grid
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"[INFO] Saved UMAP visualization to {pdf_path}")
    
    def _compute_and_save_detection_metrics(self, valid_calibration_indices, detection_keep_mask,
                                            recording, sampling_rate, output_dir, matching_df=None):
        """
        计算并保存spike_detection的accuracy, TPR, TNR
        
        参数:
        valid_calibration_indices: 校准阶段的有效时间索引
        detection_keep_mask: detection预测结果（True=spike, False=noise）
        recording: 录音数据
        sampling_rate: 采样率
        output_dir: 输出目录
        matching_df: 包含is_match_neuron列的DataFrame（可选），如果提供，将使用match/unmatch标签
        """
        print("[INFO] Computing spike_detection metrics...")
        
        # 获取GT labels（通过时间匹配）
        gt_labels = []
        if self.gt_spike_inf is not None:
            calibration_duration = 100  # 100秒
            calibration_frames = calibration_duration * sampling_rate
            
            # 过滤GT spike（在校准时间范围内）
            left_sample = 10
            right_sample = 20
            gt_spikes_calibration = self.gt_spike_inf[
                (self.gt_spike_inf['time'] >= left_sample) & 
                (self.gt_spike_inf['time'] < calibration_frames - right_sample)
            ].copy()
            
            if len(gt_spikes_calibration) > 0:
                # 使用完全向量化的方法快速匹配GT labels
                time_tolerance = 1  # 1个采样点
                gt_times = gt_spikes_calibration['time'].values  # numpy array
                valid_calibration_indices_array = np.array(valid_calibration_indices)
                
                # 使用numpy的searchsorted进行批量匹配（比循环快得多）
                # 对GT times排序以便使用searchsorted
                gt_times_sorted = np.sort(gt_times)
                
                # 批量查找所有calibration indices的匹配范围
                # 找到每个sampled_time在[sampled_time - tolerance, sampled_time + tolerance]范围内的GT spike
                left_indices = np.searchsorted(gt_times_sorted, valid_calibration_indices_array - time_tolerance, side='left')
                right_indices = np.searchsorted(gt_times_sorted, valid_calibration_indices_array + time_tolerance + 1, side='right')
                
                # 如果right_idx > left_idx，说明找到了匹配
                gt_labels = (right_indices > left_indices).astype(int)
                
                # 如果提供了matching_df，使用is_match_neuron标签修改GT labels
                # 将unmatch的GT spike视为noise（0）而不是spike（1）
                if matching_df is not None and 'is_match_neuron' in matching_df.columns:
                    print("[INFO] Using match/unmatch labels to modify GT labels for spike_detection metrics...")
                    # 确保matching_df和valid_calibration_indices的顺序一致
                    # 通过time列匹配matching_df和valid_calibration_indices
                    if len(matching_df) == len(valid_calibration_indices):
                        # 将unmatch neuron的GT spike标记为noise
                        is_match_mask = matching_df['is_match_neuron'].values
                        # 只有match neuron的spike才被认为是真正的spike
                        # gt_labels已经是匹配到的GT spike（1）或未匹配（0）
                        # 现在进一步：只有match neuron的GT spike才被认为是spike
                        gt_labels = (gt_labels & is_match_mask).astype(int)
                        match_spike_count = np.sum(gt_labels == 1)
                        # 计算unmatch neuron的GT spike数量（这些会被视为noise）
                        found_gt_mask = (right_indices > left_indices)
                        unmatch_spike_count = np.sum(found_gt_mask & (~is_match_mask))
                        print(f"[INFO] Match neuron GT spikes (treated as spike): {match_spike_count:,}")
                        print(f"[INFO] Unmatch neuron GT spikes (treated as noise): {unmatch_spike_count:,}")
                    else:
                        print(f"[WARNING] matching_df length ({len(matching_df)}) != valid_calibration_indices length ({len(valid_calibration_indices)}), skipping match/unmatch modification")
            else:
                gt_labels = np.zeros(len(valid_calibration_indices), dtype=int)  # 没有GT数据，全部设为noise
        else:
            gt_labels = np.zeros(len(valid_calibration_indices), dtype=int)  # 没有GT数据，全部设为noise
        
        # gt_labels已经是numpy array了（如果没有GT数据，上面已经创建为array）
        if not isinstance(gt_labels, np.ndarray):
            gt_labels = np.array(gt_labels)
        predicted_labels = detection_keep_mask.astype(int)  # 1=spike, 0=noise
        
        # 计算metrics
        # TP: predicted=spike, GT=spike
        # TN: predicted=noise, GT=noise
        # FP: predicted=spike, GT=noise
        # FN: predicted=noise, GT=spike
        tp = np.sum((predicted_labels == 1) & (gt_labels == 1))
        tn = np.sum((predicted_labels == 0) & (gt_labels == 0))
        fp = np.sum((predicted_labels == 1) & (gt_labels == 0))
        fn = np.sum((predicted_labels == 0) & (gt_labels == 1))
        
        total = len(predicted_labels)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity/Recall)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
        
        print(f"[INFO] Spike Detection Metrics:")
        print(f"  Total samples: {total:,}")
        print(f"  TP: {tp:,}, TN: {tn:,}, FP: {fp:,}, FN: {fn:,}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  TPR (Sensitivity): {tpr*100:.2f}%")
        print(f"  TNR (Specificity): {tnr*100:.2f}%")
        
        # 保存metrics到文件
        metrics_path = os.path.join(output_dir, 'calibration_detection_metrics.json')
        metrics = {
            'total_samples': int(total),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'accuracy': float(accuracy),
            'tpr': float(tpr),
            'tnr': float(tnr)
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved detection metrics to {metrics_path}")
    
    def _plot_calibration_confusion_matrix_heatmap(self, confusion_matrix, all_021322_indices, output_dir):
        """
        绘制calibration阶段的confusion matrix heatmap
        
        参数:
        confusion_matrix: (n_neurons+1, n_neurons+1) 的矩阵，最后一行/列是unmatched
        all_021322_indices: 所有021322 neuron的索引列表
        output_dir: 输出目录
        """
        print("[INFO] Plotting calibration confusion matrix heatmap...")
        
        # 创建标签
        labels = [f"Neuron_{idx+1}" for idx in all_021322_indices] + ["Unmatched"]
        
        # 绘制热图
        plt.figure(figsize=(14, 12))
        
        # 转换为数值矩阵
        cm_values = confusion_matrix.astype(float)
        cm_values = np.nan_to_num(cm_values, nan=0.0)
        cm_values_int = cm_values.astype(int)
        
        sns.heatmap(
            cm_values_int,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title('Calibration Confusion Matrix: Predicted 021322 Neuron vs GT 021322 Neuron', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Ground Truth 021322 Neuron', fontsize=12)
        plt.ylabel('Predicted 021322 Neuron', fontsize=12)
        plt.tight_layout()
        
        # 保存heatmap
        heatmap_path = os.path.join(output_dir, 'calibration_template_matching_confusion_matrix_heatmap.pdf')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"[INFO] Saved confusion matrix heatmap to {heatmap_path}")
        
        # 计算并打印统计信息
        total = np.sum(confusion_matrix)
        diagonal_sum = np.trace(confusion_matrix[:-1, :-1])  # 排除unmatched行/列的对角线
        accuracy = diagonal_sum / total if total > 0 else 0
        
        print(f"[INFO] Confusion matrix statistics:")
        print(f"  Total spikes: {total:,}")
        print(f"  Correctly classified (diagonal): {diagonal_sum:,}")
        print(f"  Overall accuracy: {accuracy * 100:.2f}%")
        
        # 计算每个neuron的precision和recall
        print(f"\n[INFO] Per-neuron statistics:")
        for i, idx in enumerate(all_021322_indices):
            neuron_name = f"Neuron_{idx+1}"
            row_sum = np.sum(confusion_matrix[i, :])
            col_sum = np.sum(confusion_matrix[:, i])
            diagonal = confusion_matrix[i, i]
            
            precision = diagonal / row_sum if row_sum > 0 else 0
            recall = diagonal / col_sum if col_sum > 0 else 0
            
            print(f"  {neuron_name}: precision={precision*100:.1f}%, recall={recall*100:.1f}%, "
                  f"TP={diagonal}, predicted={row_sum}, GT={col_sum}")
    
    def _compute_and_save_calibration_confusion_matrix(self, valid_calibration_indices, calibration_clusters,
                                                       recording, sampling_rate, output_dir):
        """
        计算并保存mapping之后predicted和GT的confusion matrix和热图可视化
        
        参数:
        valid_calibration_indices: 校准阶段的有效时间索引（已经过detection筛选）
        calibration_clusters: 校准阶段的kmeans cluster标签
        recording: 录音数据
        sampling_rate: 采样率
        output_dir: 输出目录
        """
        print("[INFO] Computing calibration confusion matrix (predicted vs GT neuron)...")
        
        # 1. 将cluster映射到neuron
        predicted_neuron_labels = []
        for cluster_id in calibration_clusters:
            neuron = self.cluster_to_neuron_mapping.get(cluster_id, None)
            predicted_neuron_labels.append(neuron)
        
        # 2. 获取GT neuron labels（通过时间匹配）
        gt_neuron_labels = []
        if self.gt_spike_inf is not None:
            calibration_duration = 100  # 100秒
            calibration_frames = calibration_duration * sampling_rate
            
            # 过滤GT spike（在校准时间范围内）
            left_sample = 10
            right_sample = 20
            gt_spikes_calibration = self.gt_spike_inf[
                (self.gt_spike_inf['time'] >= left_sample) & 
                (self.gt_spike_inf['time'] < calibration_frames - right_sample)
            ].copy()
            
            if len(gt_spikes_calibration) > 0:
                # 使用向量化方法快速匹配GT neuron
                time_tolerance = 1  # 1个采样点
                gt_times = gt_spikes_calibration['time'].values
                gt_neurons = gt_spikes_calibration.get('neuron', gt_spikes_calibration.get('Neuron', None))
                if gt_neurons is None:
                    gt_neurons = [None] * len(gt_spikes_calibration)
                else:
                    gt_neurons = gt_neurons.values
                
                valid_calibration_indices_array = np.array(valid_calibration_indices)
                gt_times_sorted = np.sort(gt_times)
                gt_times_argsort = np.argsort(gt_times)
                
                # 批量查找匹配范围
                left_indices = np.searchsorted(gt_times_sorted, valid_calibration_indices_array - time_tolerance, side='left')
                right_indices = np.searchsorted(gt_times_sorted, valid_calibration_indices_array + time_tolerance + 1, side='right')
                
                # 对每个calibration index，找到最近的GT spike
                gt_neuron_labels = []
                for i, sampled_time in enumerate(valid_calibration_indices_array):
                    left_idx = left_indices[i]
                    right_idx = right_indices[i]
                    
                    if right_idx > left_idx:
                        # 找到了匹配的GT spike，取最近的
                        matched_indices = gt_times_argsort[left_idx:right_idx]
                        matched_times = gt_times[matched_indices]
                        distances = np.abs(matched_times - sampled_time)
                        nearest_matched_idx = matched_indices[np.argmin(distances)]
                        gt_neuron = gt_neurons[nearest_matched_idx]
                        
                        # 将GT neuron映射到trainset neuron（如果存在映射）
                        if gt_neuron is not None and not pd.isna(gt_neuron):
                            mapped_train_neuron = self.gt_neuron_to_train_neuron_mapping.get(gt_neuron, gt_neuron)
                            gt_neuron_labels.append(mapped_train_neuron)
                        else:
                            gt_neuron_labels.append(None)
                    else:
                        gt_neuron_labels.append(None)
            else:
                gt_neuron_labels = [None] * len(valid_calibration_indices)
        else:
            gt_neuron_labels = [None] * len(valid_calibration_indices)
        
        # 3. 过滤掉未匹配的样本（predicted或GT为None）
        valid_mask = np.array([
            (pred is not None) and (gt is not None) 
            for pred, gt in zip(predicted_neuron_labels, gt_neuron_labels)
        ])
        
        if np.sum(valid_mask) == 0:
            print("[WARNING] No valid neuron pairs found for confusion matrix")
            return
        
        predicted_neuron_valid = [predicted_neuron_labels[i] for i in range(len(predicted_neuron_labels)) if valid_mask[i]]
        gt_neuron_valid = [gt_neuron_labels[i] for i in range(len(gt_neuron_labels)) if valid_mask[i]]
        
        print(f"[INFO] Valid neuron pairs: {len(predicted_neuron_valid):,} / {len(predicted_neuron_labels):,}")
        
        # 4. 创建confusion matrix
        confusion_matrix = pd.crosstab(
            pd.Series(predicted_neuron_valid, name='predicted_neuron'),
            pd.Series(gt_neuron_valid, name='gt_neuron'),
            margins=False
        )
        
        # 保存confusion matrix为CSV
        confusion_matrix_path = os.path.join(output_dir, 'calibration_confusion_matrix.csv')
        confusion_matrix.to_csv(confusion_matrix_path)
        print(f"[INFO] Saved calibration confusion matrix to {confusion_matrix_path}")
        
        # 计算准确率：只计算predicted_neuron == gt_neuron的情况
        # 由于confusion matrix可能不是方阵，需要找到predicted和GT neuron名称相同的对角线元素
        diagonal_sum = 0
        for pred_neuron in confusion_matrix.index:
            if pred_neuron in confusion_matrix.columns:
                diagonal_sum += confusion_matrix.loc[pred_neuron, pred_neuron]
        
        total_sum = confusion_matrix.values.sum()
        accuracy = diagonal_sum / total_sum if total_sum > 0 else 0
        print(f"[INFO] Calibration neuron classification accuracy: {accuracy * 100:.2f}%")
        print(f"[INFO] Diagonal sum (correct matches): {diagonal_sum:,}, Total: {total_sum:,}")
        
        # 5. 按照Neuron_xx的数字顺序排序confusion matrix
        def extract_neuron_number(neuron_name):
            """从Neuron_xx中提取数字用于排序"""
            if isinstance(neuron_name, str) and neuron_name.startswith('Neuron_'):
                match = re.search(r'Neuron_(\d+)', neuron_name)
                if match:
                    return int(match.group(1))
            return float('inf')  # 如果无法提取数字，放在最后
        
        # 对行和列进行排序
        sorted_rows = sorted(confusion_matrix.index, key=extract_neuron_number)
        sorted_cols = sorted(confusion_matrix.columns, key=extract_neuron_number)
        
        # 重新排列confusion matrix
        confusion_matrix_sorted = confusion_matrix.reindex(index=sorted_rows, columns=sorted_cols, fill_value=0)
        
        # 生成热图
        plt.figure(figsize=(12, 10))
        
        # 转换为数值矩阵（处理可能的非数值类型）
        cm_values = confusion_matrix_sorted.values.astype(float)
        cm_values = np.nan_to_num(cm_values, nan=0.0)
        cm_values_int = cm_values.astype(int)
        
        # 使用seaborn绘制heatmap
        sns.heatmap(
            cm_values_int,
            annot=True,
            fmt='d',  # 整数格式
            cmap='Blues',
            cbar_kws={'label': 'Count'},
            xticklabels=confusion_matrix_sorted.columns,
            yticklabels=confusion_matrix_sorted.index,
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title(f'Calibration Confusion Matrix: Predicted Neuron vs GT Neuron\n(Accuracy: {accuracy * 100:.2f}%)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Ground Truth Neuron (Trainset Mapped)', fontsize=12)
        plt.ylabel('Predicted Neuron', fontsize=12)
        plt.tight_layout()
        
        # 保存heatmap为PDF
        heatmap_path = os.path.join(output_dir, 'calibration_confusion_matrix_heatmap.pdf')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"[INFO] Saved calibration confusion matrix heatmap to {heatmap_path}")
        
        plt.close()
        
        # 打印统计信息
        print("\n[INFO] Calibration Confusion Matrix Statistics:")
        print(f"  Total matched spikes: {len(predicted_neuron_valid):,}")
        print(f"  Unique predicted neurons: {len(set(predicted_neuron_valid))}")
        print(f"  Unique GT neurons: {len(set(gt_neuron_valid))}")
        print(f"  Correctly classified (predicted == GT): {diagonal_sum:,}")
        print(f"  Misclassified: {total_sum - diagonal_sum:,}")
        print(f"  Overall accuracy: {accuracy * 100:.2f}%")
        
        # 打印每个正确匹配的neuron的详细信息
        print("\n[INFO] Correct neuron matches (predicted == GT):")
        for pred_neuron in sorted(confusion_matrix.index):
            if pred_neuron in confusion_matrix.columns:
                count = confusion_matrix.loc[pred_neuron, pred_neuron]
                total_for_pred = confusion_matrix.loc[pred_neuron].sum()
                precision = count / total_for_pred if total_for_pred > 0 else 0
                print(f"  {pred_neuron}: {count:,} / {total_for_pred:,} ({precision*100:.1f}% precision)")
        
        # 分析：一个predicted neuron对应多个GT neuron的情况
        print("\n[INFO] Analyzing predicted neurons that map to multiple GT neurons:")
        multi_gt_neurons = []
        for pred_neuron in confusion_matrix.index:
            gt_counts = confusion_matrix.loc[pred_neuron]
            gt_counts_nonzero = gt_counts[gt_counts > 0]
            if len(gt_counts_nonzero) > 1:
                # 这个predicted neuron对应多个GT neuron
                total_spikes = gt_counts.sum()
                main_gt = gt_counts_nonzero.idxmax()
                main_gt_count = gt_counts_nonzero[main_gt]
                other_gts = {gt: int(count) for gt, count in gt_counts_nonzero.items() if gt != main_gt}
                multi_gt_neurons.append({
                    'predicted_neuron': pred_neuron,
                    'total_spikes': int(total_spikes),
                    'main_gt_neuron': main_gt,
                    'main_gt_count': int(main_gt_count),
                    'other_gt_neurons': other_gts
                })
        
        if len(multi_gt_neurons) > 0:
            print(f"  Found {len(multi_gt_neurons)} predicted neurons mapping to multiple GT neurons")
            print("\n  Top cases (showing first 10):")
            for i, info in enumerate(multi_gt_neurons[:10]):
                print(f"    {info['predicted_neuron']}: {info['total_spikes']:,} spikes")
                print(f"      Main GT: {info['main_gt_neuron']} ({info['main_gt_count']:,} spikes, {info['main_gt_count']/info['total_spikes']*100:.1f}%)")
                print(f"      Other GTs: {', '.join([f'{gt}({count:,})' for gt, count in info['other_gt_neurons'].items()])}")
            
            if len(multi_gt_neurons) > 10:
                print(f"    ... and {len(multi_gt_neurons) - 10} more cases")
            
            # 进一步分析：这些predicted neuron对应的cluster是否在匹配时也匹配到了多个neuron
            print("\n  Checking if these predicted neurons' clusters matched multiple trainset neurons:")
            for info in multi_gt_neurons[:5]:  # 只检查前5个
                pred_neuron = info['predicted_neuron']
                # 找到对应的cluster（通过cluster_to_neuron_mapping反向查找）
                matching_clusters = [c for c, n in self.cluster_to_neuron_mapping.items() if n == pred_neuron]
                if len(matching_clusters) > 0:
                    cluster_id = matching_clusters[0]  # 通常一个neuron对应一个cluster
                    # 检查这个cluster在匹配时是否有多个候选
                    if cluster_id in self.cluster_multi_matches:
                        all_matches = self.cluster_multi_matches[cluster_id]
                        if len(all_matches) > 1:
                            print(f"    {pred_neuron} (cluster {cluster_id}): matched {len(all_matches)} trainset neurons")
                            print(f"      Selected: {all_matches[0]['neuron']} (corr={all_matches[0]['corr']:.3f})")
                            other_candidates_str = ', '.join([f"{m['neuron']}({m['corr']:.3f})" for m in all_matches[1:]])
                            print(f"      Other candidates: {other_candidates_str}")
                        else:
                            print(f"    {pred_neuron} (cluster {cluster_id}): matched only 1 trainset neuron ({all_matches[0]['neuron']})")
                    else:
                        print(f"    {pred_neuron} (cluster {cluster_id}): no match info available")
        else:
            print("  No predicted neurons mapping to multiple GT neurons found")
    
    def _plot_spike_detection_umap(self, detection_100d_features, detection_keep_mask, 
                                   all_calibration_windows, all_valid_calibration_indices,
                                   recording, sampling_rate, output_dir):
        """
        生成spike_detection的UMAP可视化（使用100维中间层特征）
        
        参数:
        detection_100d_features: numpy.ndarray, shape (n_all_windows, 100) - 所有windows的100维特征
        detection_keep_mask: numpy.ndarray, shape (n_all_windows,) - detection预测结果（True=spike, False=noise）
        all_calibration_windows: numpy.ndarray - 所有calibration windows
        all_valid_calibration_indices: numpy.ndarray - 所有calibration indices
        recording: 录音数据（用于提取GT spike特征）
        sampling_rate: 采样率
        output_dir: 输出目录
        """
        try:
            import umap
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError as e:
            print(f"[WARNING] Cannot generate spike_detection UMAP plot: {e}")
            return
        
        print("[INFO] Generating spike_detection UMAP visualization from 100-dimensional features...")
        
        # 1. 从所有windows中采样50000个样本（确保两张图使用相同的样本）
        n_samples = min(50000, len(detection_100d_features))
        sample_indices = np.random.choice(len(detection_100d_features), n_samples, replace=False)
        sample_indices = np.sort(sample_indices)  # 排序以便后续匹配
        
        # 获取采样后的特征和predicted labels
        sampled_features = detection_100d_features[sample_indices]  # (n_samples, 100)
        sampled_predicted_labels = detection_keep_mask[sample_indices].astype(int)  # 1=spike, 0=noise
        sampled_calibration_indices = all_valid_calibration_indices[sample_indices]  # 对应的时间索引
        
        # 2. 为这些采样样本匹配GT labels（通过时间匹配）
        sampled_gt_labels = []
        if self.gt_spike_inf is not None:
            # 获取校准阶段的时间范围
            calibration_duration = 100  # 100秒
            calibration_frames = calibration_duration * sampling_rate
            
            # 过滤GT spike（在校准时间范围内）
            left_sample = 10
            right_sample = 20
            gt_spikes_calibration = self.gt_spike_inf[
                (self.gt_spike_inf['time'] >= left_sample) & 
                (self.gt_spike_inf['time'] < calibration_frames - right_sample)
            ].copy()
            
            if len(gt_spikes_calibration) > 0:
                # 对每个采样样本，通过时间匹配找到对应的GT label
                time_tolerance = 1  # 1个采样点
                for sampled_time in sampled_calibration_indices:
                    # 在GT spike中查找匹配的时间
                    matched_gt = gt_spikes_calibration[
                        (gt_spikes_calibration['time'] >= sampled_time - time_tolerance) &
                        (gt_spikes_calibration['time'] <= sampled_time + time_tolerance)
                    ]
                    
                    if len(matched_gt) > 0:
                        # 如果匹配到GT spike，则为spike（1）
                        sampled_gt_labels.append(1)
                    else:
                        # 如果没有匹配到，则为noise（0）
                        sampled_gt_labels.append(0)
            else:
                sampled_gt_labels = [0] * len(sampled_calibration_indices)  # 没有GT数据，全部设为noise
        else:
            sampled_gt_labels = [0] * len(sampled_calibration_indices)  # 没有GT数据，全部设为noise
        
        sampled_gt_labels = np.array(sampled_gt_labels)
        
        # 3. 使用相同的特征进行UMAP降维
        all_features = sampled_features
        print(f"[INFO] Running UMAP on {len(all_features)} samples...")
        
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_features)-1))
        embedding = reducer.fit_transform(all_features)
        
        # 两张图使用相同的embedding
        predicted_embedding = embedding
        gt_embedding = embedding  # 使用相同的embedding
        predicted_labels_plot = sampled_predicted_labels
        gt_labels_plot = sampled_gt_labels
        
        # 4. 创建PDF
        pdf_path = os.path.join(output_dir, 'spike_detection_umap.pdf')
        with PdfPages(pdf_path) as pdf:
            # 图1: Predicted (spike=橙色, noise=灰色)
            fig, ax = plt.subplots(figsize=(8, 8))
            # spike为橙色，noise为灰色
            spike_color = '#FF8C00'  # 橙色
            noise_color = '#808080'  # 灰色
            colors = [spike_color if label == 1 else noise_color for label in predicted_labels_plot]
            ax.scatter(predicted_embedding[:, 0], predicted_embedding[:, 1], 
                      c=colors, s=10, alpha=0.6, edgecolors='none')
            ax.set_title(f'UMAP: Predicted Spike Detection (n={len(predicted_embedding)})', 
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')  # xy比为1:1
            ax.set_xticks([])  # 去除xticks
            ax.set_yticks([])  # 去除yticks
            ax.grid(False)  # 去除grid
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # 图2: GT (spike=橙色, noise=灰色，使用相同的样本和embedding)
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = [spike_color if label == 1 else noise_color for label in gt_labels_plot]
            ax.scatter(gt_embedding[:, 0], gt_embedding[:, 1], 
                      c=colors, s=10, alpha=0.6, edgecolors='none')
            ax.set_title(f'UMAP: GT Spike Detection (n={len(gt_embedding)})', 
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')  # xy比为1:1
            ax.set_xticks([])  # 去除xticks
            ax.set_yticks([])  # 去除yticks
            ax.grid(False)  # 去除grid
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"[INFO] Saved spike_detection UMAP visualization to {pdf_path}")
    
    def _validate_cluster_gt_mapping(self, valid_calibration_indices, calibration_clusters,
                                     recording, sampling_rate, output_dir):
        """
        验证功能：分析无GT映射的cluster中，有多少spike是GT的noise
        
        注意：此功能仅用于验证和分析，不影响正式流程
        
        参数:
        valid_calibration_indices: 校准阶段的有效时间索引（已经过detection筛选）
        calibration_clusters: 校准阶段的kmeans cluster标签
        recording: 录音数据
        sampling_rate: 采样率
        output_dir: 输出目录（可选）
        """
        print("[INFO] Validating: checking if unmapped clusters contain GT noise...")
        
        # 1. 为每个cluster收集其对应的GT neuron（通过时间匹配）
        calibration_duration = 100  # 100秒
        calibration_frames = calibration_duration * sampling_rate
        
        # 过滤GT spike（在校准时间范围内）
        left_sample = 10
        right_sample = 20
        gt_spikes_calibration = self.gt_spike_inf[
            (self.gt_spike_inf['time'] >= left_sample) & 
            (self.gt_spike_inf['time'] < calibration_frames - right_sample)
        ].copy()
        
        # 获取所有GT spike的时间点（用于判断某个时间点是否是GT spike）
        gt_times_set = set(gt_spikes_calibration['time'].values) if len(gt_spikes_calibration) > 0 else set()
        
        # 为每个cluster建立到GT neuron的映射
        cluster_to_gt_neuron = {}  # {cluster_id: {gt_neuron: count}}
        cluster_spike_indices = {}  # {cluster_id: [indices]}
        
        if len(gt_spikes_calibration) > 0:
            # 使用向量化方法快速匹配
            time_tolerance = 1  # 1个采样点
            gt_times = gt_spikes_calibration['time'].values
            gt_neurons = gt_spikes_calibration.get('neuron', gt_spikes_calibration.get('Neuron', None))
            if gt_neurons is None:
                gt_neurons = [None] * len(gt_spikes_calibration)
            else:
                gt_neurons = gt_neurons.values
            
            valid_calibration_indices_array = np.array(valid_calibration_indices)
            gt_times_sorted = np.sort(gt_times)
            gt_times_argsort = np.argsort(gt_times)
            
            # 批量查找匹配范围
            left_indices = np.searchsorted(gt_times_sorted, valid_calibration_indices_array - time_tolerance, side='left')
            right_indices = np.searchsorted(gt_times_sorted, valid_calibration_indices_array + time_tolerance + 1, side='right')
            
            # 对每个calibration index，找到对应的GT neuron
            for i, (sampled_time, cluster_id) in enumerate(zip(valid_calibration_indices_array, calibration_clusters)):
                # 初始化cluster统计
                if cluster_id not in cluster_spike_indices:
                    cluster_spike_indices[cluster_id] = []
                    cluster_to_gt_neuron[cluster_id] = {}
                
                cluster_spike_indices[cluster_id].append(i)
                
                left_idx = left_indices[i]
                right_idx = right_indices[i]
                
                if right_idx > left_idx:
                    # 找到了匹配的GT spike，取最近的
                    matched_indices = gt_times_argsort[left_idx:right_idx]
                    matched_times = gt_times[matched_indices]
                    distances = np.abs(matched_times - sampled_time)
                    nearest_matched_idx = matched_indices[np.argmin(distances)]
                    gt_neuron = gt_neurons[nearest_matched_idx]
                    
                    # 将GT neuron映射到trainset neuron（如果存在映射）
                    if gt_neuron is not None and not pd.isna(gt_neuron):
                        mapped_train_neuron = self.gt_neuron_to_train_neuron_mapping.get(gt_neuron, gt_neuron)
                        if mapped_train_neuron not in cluster_to_gt_neuron[cluster_id]:
                            cluster_to_gt_neuron[cluster_id][mapped_train_neuron] = 0
                        cluster_to_gt_neuron[cluster_id][mapped_train_neuron] += 1
        
        # 2. 找出无GT映射的cluster
        unmapped_cluster_ids = [cluster_id for cluster_id, gt_neuron_dict in cluster_to_gt_neuron.items() 
                                if len(gt_neuron_dict) == 0]
        
        # 3. 统计无GT映射的cluster中的spike，有多少是GT的noise
        total_unmapped_spikes = 0
        unmapped_spikes_that_are_gt_noise = 0
        
        for cluster_id in unmapped_cluster_ids:
            spike_indices = cluster_spike_indices[cluster_id]
            total_unmapped_spikes += len(spike_indices)
            
            # 检查这些spike的时间点是否在GT spike中
            for idx in spike_indices:
                spike_time = valid_calibration_indices_array[idx]
                # 如果这个时间点不在GT spike中，说明它是GT的noise
                if spike_time not in gt_times_set:
                    unmapped_spikes_that_are_gt_noise += 1
        
        # 4. 输出结果
        if total_unmapped_spikes > 0:
            noise_ratio = unmapped_spikes_that_are_gt_noise / total_unmapped_spikes
            print(f"\n[INFO] Unmapped clusters analysis:")
            print(f"  Total spikes in unmapped clusters: {total_unmapped_spikes:,}")
            print(f"  Spikes that are GT noise: {unmapped_spikes_that_are_gt_noise:,} ({noise_ratio*100:.1f}%)")
        else:
            print(f"\n[INFO] No unmapped clusters found.")
        
        # 5. 保存结果到文件（如果提供了output_dir）
        if output_dir is not None and total_unmapped_spikes > 0:
            validation_results = {
                'total_unmapped_spikes': int(total_unmapped_spikes),
                'unmapped_spikes_that_are_gt_noise': int(unmapped_spikes_that_are_gt_noise),
                'noise_ratio': float(unmapped_spikes_that_are_gt_noise / total_unmapped_spikes) if total_unmapped_spikes > 0 else 0.0
            }
            
            validation_json_path = os.path.join(output_dir, 'cluster_gt_validation_summary.json')
            with open(validation_json_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            print(f"[INFO] Saved validation summary to {validation_json_path}")
    
    def process_data_chunk(self, data_chunk, start_frame, chunk_duration_ms=500):
        """
        处理数据块（每500ms为单位）：
        1. 阈值检测（std_multiplier=2.4, window_size=10）
        2. spike_detection（使用detection模型筛选）
        3. 对通过detection的spike计算shape和energy（使用所有30个通道）
        4. 与021322的template计算匹配score
        5. 保存结果
        
        返回:
        detailed_results: 包含完整检测结果的字典
        """
        if not self.calibration_complete:
            raise RuntimeError("Calibration phase must be completed first")
        
        sampling_rate = 10000  # 假设采样率为10kHz
        chunk_size_frames = int(chunk_duration_ms * sampling_rate / 1000)
        
        # 检测峰值（阈值检测）
        threshold_result = detect_local_minimum_in_window(
            data_chunk,
            std_multiplier=2.4,
            window_size=10
        )
        threshold_result = np.array(threshold_result) + start_frame
        
        # 过滤边界附近的spike
        window_size = 30  # 与训练时一致 (30, 30)
        left_sample = 10   # 与train_spike_pipeline.py保持一致
        right_sample = 20  # 与train_spike_pipeline.py保持一致
        valid_indices = threshold_result[
            (threshold_result >= start_frame + left_sample) & 
            (threshold_result < start_frame + chunk_size_frames - right_sample)
        ]
        
        if len(valid_indices) == 0:
            return {
                'all_threshold_spikes': pd.DataFrame(columns=['time']),
                'detection_results': pd.DataFrame(columns=['time', 'detection_predicted', 'detection_score']),
                'classification_results': pd.DataFrame(columns=['time', 'cluster_id', 'neuron_id', 'gt_cluster_id', 'gt_neuron_id']),
                'waveforms': np.array([]),
                'embeddings': np.array([])
            }
        
        # 提取窗口 (30, 30)
        # 与train_spike_pipeline.py保持一致：[spike_time - 10, spike_time + 19]，共30个时间点
        windows = []
        valid_indices_for_windows = []  # 跟踪实际成功提取窗口的索引
        for idx in valid_indices:
            rel_idx = idx - start_frame
            start = rel_idx - left_sample   # rel_idx - 10
            end = rel_idx + right_sample    # rel_idx + 20
            if start < 0 or end > data_chunk.shape[1]:
                continue
            window = data_chunk[:, start:end]
            if window.shape[1] != window_size:
                continue
            windows.append(window)
            valid_indices_for_windows.append(idx)  # 记录成功提取窗口的索引
        
        if len(windows) == 0:
            return {
                'all_threshold_spikes': pd.DataFrame(columns=['time']),
                'detection_results': pd.DataFrame(columns=['time', 'detection_predicted', 'detection_score']),
                'classification_results': pd.DataFrame(columns=['time', 'cluster_id', 'neuron_id', 'gt_cluster_id', 'gt_neuron_id']),
                'waveforms': np.array([]),
                'embeddings': np.array([])
            }
        
        windows = np.stack(windows)
        valid_indices_for_windows = np.array(valid_indices_for_windows)  # 转换为numpy数组
        
        # 使用detection模型筛选
        detection_scores = []
        detection_keep_mask = []
        with torch.no_grad():
            batch_size = 4096
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)  # (batch, 30, 30)
                
                # 使用AutoSort的noise分类器（期望30x30输入）
                # noise分类器输出2类logits，取spike类（索引1）的概率
                noise_output = self.detection_model(batch_tensor.reshape(batch_tensor.size(0), -1))
                # 使用softmax获取spike类的概率
                probs = torch.softmax(noise_output, dim=1)
                outputs = probs[:, 1]  # spike类的概率
        
                scores = outputs.cpu().numpy()
                detection_scores.append(scores)
                detection_keep_mask.append((scores > 0.01).astype(bool))
        
        detection_scores = np.concatenate(detection_scores)
        detection_keep_mask = np.concatenate(detection_keep_mask)
        
        # 保存所有阈值检测的spike和detection结果
        # 注意：使用valid_indices_for_windows而不是valid_indices，因为有些索引可能因为边界检查被跳过
        all_threshold_spikes = pd.DataFrame({'time': valid_indices_for_windows})
        detection_results = pd.DataFrame({
            'time': valid_indices_for_windows,
            'detection_predicted': detection_keep_mask.astype(int),
            'detection_score': detection_scores
        })
        
        windows_after_detection = windows[detection_keep_mask]
        valid_indices_after_detection = valid_indices_for_windows[detection_keep_mask]
        
        if len(windows_after_detection) == 0:
            return {
                'all_threshold_spikes': all_threshold_spikes,
                'detection_results': detection_results,
                'classification_results': pd.DataFrame(columns=['time', 'shape_scores', 'energy_scores', 'best_template_idx', 'gt_neuron_id']),
                'waveforms': np.array([]),
                'shapes': np.array([]),
                'energies': np.array([])
            }
        
        # 计算每个spike的shape和energy（使用所有30个通道）
        n_spikes = len(windows_after_detection)
        n_neurons = self.shape_templates_021322.shape[0]
        
        all_spike_shapes = []
        all_spike_energies = []
        
        for i in range(n_spikes):
            window = windows_after_detection[i]  # (30, 30)
            shape, energy = compute_spike_shape_and_energy(window)
            all_spike_shapes.append(shape)
            all_spike_energies.append(energy)
        
        all_spike_shapes = np.array(all_spike_shapes)  # (n_spikes, 30, 30)
        all_spike_energies = np.array(all_spike_energies)  # (n_spikes, 30)
        
        # 与021322的template计算匹配score
        shape_scores = np.zeros((n_spikes, n_neurons), dtype=np.float32)
        energy_scores = np.zeros((n_spikes, n_neurons), dtype=np.float32)
        
        for spike_idx in range(n_spikes):
            spike_shape = all_spike_shapes[spike_idx]
            spike_energy = all_spike_energies[spike_idx]
            
            for template_idx in range(n_neurons):
                shape_score, energy_score = compute_template_score(
                    spike_shape,
                    spike_energy,
                    self.shape_templates_021322[template_idx],
                    self.energy_templates_021322[template_idx],
                )
                shape_scores[spike_idx, template_idx] = shape_score
                energy_scores[spike_idx, template_idx] = energy_score
        
        # 找到最佳匹配的template（使用shape_score + energy_score的和）
        sum_scores = shape_scores + energy_scores
        best_template_indices = np.argmax(sum_scores, axis=1)  # (n_spikes,)
        
        # 匹配ground truth数据（如果可用）
        gt_neuron_ids = []
        if self.gt_spike_inf_sorted is not None:
            # 检查是否有neuron列
            has_neuron_col = 'neuron' in self.gt_spike_inf_sorted.columns or 'Neuron' in self.gt_spike_inf_sorted.columns
            neuron_col = 'neuron' if 'neuron' in self.gt_spike_inf_sorted.columns else 'Neuron' if 'Neuron' in self.gt_spike_inf_sorted.columns else None
            
            if has_neuron_col:
                # 使用时间匹配ground truth
                time_tolerance = 1  # 1个采样点
                
                for idx in valid_indices_after_detection:
                    # 使用二分搜索找到匹配范围
                    left_idx = np.searchsorted(self.gt_times, idx - time_tolerance, side='left')
                    right_idx = np.searchsorted(self.gt_times, idx + time_tolerance, side='right')
                    
                    if right_idx > left_idx:
                        # 找到匹配，取最近的
                        candidates = self.gt_spike_inf_sorted.iloc[left_idx:right_idx]
                        distances = np.abs(candidates['time'].values - idx)
                        nearest_idx = np.argmin(distances)
                        gt_match = candidates.iloc[nearest_idx]
                        
                        gt_neuron = gt_match[neuron_col]
                        if gt_neuron is not None and not pd.isna(gt_neuron):
                            # 将GT neuron映射到训练集的neuron（如果存在映射）
                            mapped_train_neuron = self.gt_neuron_to_train_neuron_mapping.get(gt_neuron, gt_neuron)
                            gt_neuron_ids.append(mapped_train_neuron)
                        else:
                            gt_neuron_ids.append(None)
                    else:
                        gt_neuron_ids.append(None)
            else:
                gt_neuron_ids = [None] * len(valid_indices_after_detection)
        else:
            gt_neuron_ids = [None] * len(valid_indices_after_detection)
        
        # 创建classification结果DataFrame
        # 注意：shape_scores和energy_scores是(n_spikes, n_neurons)的矩阵，保存为列表
        classification_results = pd.DataFrame({
            'time': valid_indices_after_detection,
            'best_template_idx': best_template_indices,
            'gt_neuron_id': gt_neuron_ids
        })
        
        return {
            'all_threshold_spikes': all_threshold_spikes,
            'detection_results': detection_results,
            'classification_results': classification_results,
            'waveforms': windows_after_detection,
            'shapes': all_spike_shapes,
            'energies': all_spike_energies,
            'shape_scores': shape_scores,
            'energy_scores': energy_scores
        }
    
    def process_complete_recording(self, recording, output_dir, chunk_duration_ms=500):
        """
        处理完整录音数据，保存所有检测结果并生成可视化
        """
        if not self.calibration_complete:
            raise RuntimeError("Calibration phase must be completed first")
        
        print("\n" + "="*60)
        print("Processing Complete Recording")
        print("="*60)
        
        sampling_rate = 10000  # 假设采样率为10kHz
        total_frames = int(recording.get_total_duration() * sampling_rate)
        calibration_frames = 60 * sampling_rate  # 前60秒用于校准
        
        # 从校准后开始处理
        start_frame = calibration_frames
        chunk_size_frames = int(chunk_duration_ms * sampling_rate / 1000)
        
        # 收集所有结果
        all_threshold_spikes = []
        all_detection_results = []
        all_classification_results = []
        all_waveforms = []
        all_shapes = []
        all_energies = []
        all_shape_scores = []
        all_energy_scores = []
        
        print("[INFO] Processing data in chunks...")
        for chunk_start in tqdm(range(start_frame, total_frames, chunk_size_frames)):
            chunk_end = min(chunk_start + chunk_size_frames, total_frames)
            
            # 读取数据块
            data_chunk = recording.get_traces(
                start_frame=chunk_start,
                end_frame=chunk_end
            ).T
            
            # 处理数据块
            chunk_results = self.process_data_chunk(data_chunk, chunk_start, chunk_duration_ms)
            
            # 收集结果
            if len(chunk_results['all_threshold_spikes']) > 0:
                all_threshold_spikes.append(chunk_results['all_threshold_spikes'])
            if len(chunk_results['detection_results']) > 0:
                all_detection_results.append(chunk_results['detection_results'])
            if len(chunk_results['classification_results']) > 0:
                all_classification_results.append(chunk_results['classification_results'])
            if len(chunk_results['waveforms']) > 0:
                all_waveforms.append(chunk_results['waveforms'])
            if len(chunk_results['shapes']) > 0:
                all_shapes.append(chunk_results['shapes'])
            if len(chunk_results['energies']) > 0:
                all_energies.append(chunk_results['energies'])
            if 'shape_scores' in chunk_results and len(chunk_results['shape_scores']) > 0:
                all_shape_scores.append(chunk_results['shape_scores'])
            if 'energy_scores' in chunk_results and len(chunk_results['energy_scores']) > 0:
                all_energy_scores.append(chunk_results['energy_scores'])
        
        # 合并所有结果
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存所有阈值检测的spike
        if all_threshold_spikes:
            final_threshold_spikes = pd.concat(all_threshold_spikes, ignore_index=True)
            final_threshold_spikes = final_threshold_spikes.sort_values('time').reset_index(drop=True)
            threshold_path = os.path.join(output_dir, 'all_threshold_spikes.csv')
            final_threshold_spikes.to_csv(threshold_path, index=False)
            print(f"[INFO] All threshold-detected spikes saved to {threshold_path}")
            print(f"[INFO] Total threshold spikes: {len(final_threshold_spikes):,}")
        else:
            final_threshold_spikes = pd.DataFrame(columns=['time'])
        
        # 保存detection结果
        if all_detection_results:
            final_detection_results = pd.concat(all_detection_results, ignore_index=True)
            final_detection_results = final_detection_results.sort_values('time').reset_index(drop=True)
            detection_path = os.path.join(output_dir, 'detection_results.csv')
            final_detection_results.to_csv(detection_path, index=False)
            print(f"[INFO] Detection results saved to {detection_path}")
            print(f"[INFO] Total detection results: {len(final_detection_results):,}")
            print(f"[INFO] Detected as spike: {final_detection_results['detection_predicted'].sum():,}")
        else:
            final_detection_results = pd.DataFrame(columns=['time', 'detection_predicted', 'detection_score'])
        
        # 保存classification结果
        if all_classification_results:
            final_classification_results = pd.concat(all_classification_results, ignore_index=True)
            final_classification_results = final_classification_results.sort_values('time').reset_index(drop=True)
            classification_path = os.path.join(output_dir, 'classification_results.csv')
            final_classification_results.to_csv(classification_path, index=False)
            print(f"[INFO] Classification results saved to {classification_path}")
            print(f"[INFO] Total classification results: {len(final_classification_results):,}")
            
            # 保存shape和energy
            if all_shapes:
                all_shapes_array = np.vstack(all_shapes)
                np.save(os.path.join(output_dir, 'all_spike_shapes.npy'), all_shapes_array)
                print(f"[INFO] Saved all spike shapes: {all_shapes_array.shape}")
            
            if all_energies:
                all_energies_array = np.vstack(all_energies)
                np.save(os.path.join(output_dir, 'all_spike_energies.npy'), all_energies_array)
                print(f"[INFO] Saved all spike energies: {all_energies_array.shape}")
            
            # 保存shape_scores和energy_scores
            if all_shape_scores:
                all_shape_scores_array = np.vstack(all_shape_scores)
                np.save(os.path.join(output_dir, 'all_shape_scores.npy'), all_shape_scores_array)
                print(f"[INFO] Saved all shape scores: {all_shape_scores_array.shape}")
            
            if all_energy_scores:
                all_energy_scores_array = np.vstack(all_energy_scores)
                np.save(os.path.join(output_dir, 'all_energy_scores.npy'), all_energy_scores_array)
                print(f"[INFO] Saved all energy scores: {all_energy_scores_array.shape}")
            
            if 'gt_neuron_id' in final_classification_results.columns:
                print(f"[INFO] Ground truth clusters matched: {final_classification_results['gt_cluster_id'].notna().sum():,}")
                print(f"[INFO] Ground truth neurons matched: {final_classification_results['gt_neuron_id'].notna().sum():,}")
        else:
            final_classification_results = pd.DataFrame(columns=['time', 'cluster_id', 'neuron_id', 'gt_cluster_id', 'gt_neuron_id'])
        
        # 保存最终spike train（只包含成功映射到neuron的spike）
        final_spike_inf = final_classification_results[final_classification_results['neuron_id'].notna()].copy()
        final_spike_inf = final_spike_inf[['time', 'neuron_id']].copy()
        output_path = os.path.join(output_dir, 'evaluated_spike_inf.csv')
        final_spike_inf.to_csv(output_path, index=False)
        print(f"[INFO] Final spike train saved to {output_path}")
        print(f"[INFO] Total spikes in final train: {len(final_spike_inf):,}")
        print(f"[INFO] Unique neurons: {final_spike_inf['neuron_id'].nunique()}")
        
        # 合并waveforms和embeddings用于可视化
        # 保存waveforms
        if all_waveforms:
            all_waveforms_combined = np.vstack(all_waveforms)
            np.save(os.path.join(output_dir, 'all_waveforms.npy'), all_waveforms_combined)
            print(f"[INFO] Saved all waveforms: {all_waveforms_combined.shape}")
        else:
            all_waveforms_combined = np.array([])
        
        # 生成heatmap（混淆矩阵）
        if len(final_classification_results) > 0 and 'gt_neuron_id' in final_classification_results.columns:
            print("\n[INFO] Generating confusion matrix heatmap...")
            self._generate_confusion_matrix_heatmap(
                final_classification_results,
                output_dir
            )
        
        return final_spike_inf
    
    def _generate_umap_visualization(self, embeddings, classification_results, output_dir, max_samples=100000, n_neighbors=15, min_dist=0.1, random_state=42):
        """
        生成UMAP可视化图
        
        参数:
        embeddings: numpy.ndarray, shape (n_samples, 100) - 100维中间层特征
        classification_results: pd.DataFrame - 包含cluster_id和neuron_id的分类结果
        output_dir: str - 输出目录
        max_samples: int - 最大采样数量
        """
        print(f"[INFO] Total embeddings: {len(embeddings):,}")
        
        # 随机采样
        if len(embeddings) > max_samples:
            print(f"[INFO] Randomly sampling {max_samples} samples from {len(embeddings):,} total samples")
            random.seed(random_state)
            np.random.seed(random_state)
            sample_indices = np.random.choice(len(embeddings), max_samples, replace=False)
            sample_indices = np.sort(sample_indices)
            embeddings_sampled = embeddings[sample_indices]
            classification_sampled = classification_results.iloc[sample_indices].copy()
        else:
            embeddings_sampled = embeddings
            classification_sampled = classification_results.copy()
        
        print(f"[INFO] Using {len(embeddings_sampled):,} samples for visualization")
        
        # 准备标签
        cluster_ids = classification_sampled['cluster_id'].values
        neuron_ids = classification_sampled['neuron_id'].values
        
        # 计算UMAP
        print("[INFO] Computing UMAP embedding...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        umap_embedding = reducer.fit_transform(embeddings_sampled)
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('UMAP Visualization of Classification Features', fontsize=16, fontweight='bold')
        
        # 图1: 按cluster_id着色
        ax1 = axes[0]
        unique_clusters = np.unique(cluster_ids)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # 排除-1
        
        if len(unique_clusters) > 20:
            colors_cluster = list(plt.cm.tab20(np.linspace(0, 1, 20))) + \
                           list(plt.cm.tab20b(np.linspace(0, 1, min(20, len(unique_clusters)-20))))
        else:
            colors_cluster = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_ids == cluster_id
            if np.sum(mask) > 0:
                ax1.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                           c=[colors_cluster[i % len(colors_cluster)]], 
                           label=f'Cluster {cluster_id}', s=2, alpha=0.8)
        
        ax1.set_title('UMAP Visualization by Cluster ID', fontsize=12, fontweight='bold')
        ax1.set_xlabel('UMAP 1', fontsize=10)
        ax1.set_ylabel('UMAP 2', fontsize=10)
        if len(unique_clusters) <= 30:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=3, ncol=1)
        ax1.grid(True, alpha=0.3)
        
        # 图2: 按neuron_id着色（只显示成功映射的）
        ax2 = axes[1]
        valid_neuron_mask = ~pd.isna(neuron_ids)
        neuron_ids_valid = neuron_ids[valid_neuron_mask]
        umap_embedding_valid = umap_embedding[valid_neuron_mask]
        
        if len(neuron_ids_valid) > 0:
            unique_neurons = np.unique(neuron_ids_valid)
            unique_neurons = np.sort(unique_neurons)
            
            if len(unique_neurons) > 20:
                colors_neuron = list(plt.cm.tab20(np.linspace(0, 1, 20))) + \
                              list(plt.cm.tab20b(np.linspace(0, 1, min(20, len(unique_neurons)-20))))
            else:
                colors_neuron = plt.cm.tab20(np.linspace(0, 1, len(unique_neurons)))
            
            for i, neuron_id in enumerate(unique_neurons):
                mask = neuron_ids_valid == neuron_id
                if np.sum(mask) > 0:
                    ax2.scatter(umap_embedding_valid[mask, 0], umap_embedding_valid[mask, 1],
                               c=[colors_neuron[i % len(colors_neuron)]],
                               label=f'Neuron {neuron_id}', s=2, alpha=0.8)
        
        # 绘制未映射的点（灰色）
        unmapped_mask = pd.isna(neuron_ids)
        if np.sum(unmapped_mask) > 0:
            ax2.scatter(umap_embedding[unmapped_mask, 0], umap_embedding[unmapped_mask, 1],
                       c='#d3d3d3', label='Unmapped', s=2, alpha=0.5)
        
        ax2.set_title('UMAP Visualization by Neuron ID\n(From neuron_inf mapping)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('UMAP 1', fontsize=10)
        ax2.set_ylabel('UMAP 2', fontsize=10)
        if len(unique_neurons) <= 30:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=3, ncol=1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片为PDF
        output_path = os.path.join(output_dir, 'umap_visualization.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"[INFO] UMAP visualization saved to {output_path}")
        
        plt.close()
        
        # 打印统计信息
        print("\n[INFO] Visualization Statistics:")
        print(f"  Total samples visualized: {len(embeddings_sampled):,}")
        print(f"  Unique clusters: {len(unique_clusters)}")
        print(f"  Mapped to neurons: {np.sum(valid_neuron_mask):,}")
        print(f"  Unique neurons (mapped): {len(unique_neurons) if len(neuron_ids_valid) > 0 else 0}")
        print(f"  Unmapped samples: {np.sum(unmapped_mask):,}")
    
    def _generate_confusion_matrix_heatmap(self, classification_results, output_dir):
        """
        生成混淆矩阵heatmap（predicted neuron_id vs gt_neuron_id）
        
        参数:
        classification_results: pd.DataFrame - 包含neuron_id和gt_neuron_id的分类结果
        output_dir: str - 输出目录
        """
        # 过滤掉未匹配的样本
        valid_mask = classification_results['neuron_id'].notna() & classification_results['gt_neuron_id'].notna()
        valid_results = classification_results[valid_mask].copy()
        
        if len(valid_results) == 0:
            print("[WARNING] No valid neuron_id and gt_neuron_id pairs found, skipping heatmap generation")
            return
        
        print(f"[INFO] Generating confusion matrix for {len(valid_results):,} matched spikes")
        
        # 创建混淆矩阵
        confusion_matrix = pd.crosstab(
            valid_results['neuron_id'],
            valid_results['gt_neuron_id'],
            margins=False
        )
        
        # 保存混淆矩阵为CSV
        confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.csv')
        confusion_matrix.to_csv(confusion_matrix_path)
        print(f"[INFO] Confusion matrix saved to {confusion_matrix_path}")
        
        # 计算准确率
        diagonal_sum = np.trace(confusion_matrix.values)
        total_sum = confusion_matrix.values.sum()
        accuracy = diagonal_sum / total_sum if total_sum > 0 else 0
        print(f"[INFO] Classification accuracy: {accuracy * 100:.2f}%")
        
        # 生成heatmap
        plt.figure(figsize=(12, 10))
        
        # 转换为数值矩阵（处理可能的非数值类型）
        # 混淆矩阵应该是整数计数，所以转换为int
        cm_values = confusion_matrix.values.astype(float)
        # 处理NaN值（如果有的话）
        cm_values = np.nan_to_num(cm_values, nan=0.0)
        # 转换为整数用于显示
        cm_values_int = cm_values.astype(int)
        
        # 使用seaborn绘制heatmap
        sns.heatmap(
            cm_values_int,
            annot=True,
            fmt='d',  # 整数格式
            cmap='Blues',
            cbar_kws={'label': 'Count'},
            xticklabels=confusion_matrix.columns,
            yticklabels=confusion_matrix.index,
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title(f'Confusion Matrix: Predicted Neuron ID vs Ground Truth Neuron ID\n(Accuracy: {accuracy * 100:.2f}%)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Ground Truth Neuron ID', fontsize=12)
        plt.ylabel('Predicted Neuron ID', fontsize=12)
        plt.tight_layout()
        
        # 保存heatmap为PDF
        heatmap_path = os.path.join(output_dir, 'confusion_matrix_heatmap.pdf')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"[INFO] Confusion matrix heatmap saved to {heatmap_path}")
        
        plt.close()
        
        # 打印统计信息
        print("\n[INFO] Confusion Matrix Statistics:")
        print(f"  Total matched spikes: {len(valid_results):,}")
        print(f"  Unique predicted neurons: {valid_results['neuron_id'].nunique()}")
        print(f"  Unique ground truth neurons: {valid_results['gt_neuron_id'].nunique()}")
        print(f"  Correctly classified: {diagonal_sum:,}")
        print(f"  Misclassified: {total_sum - diagonal_sum:,}")
        print(f"  Overall accuracy: {accuracy * 100:.2f}%")

# ==================== 主函数 ====================

def run_evaluation(date_str, base_dir=None, data_base_dir=None):
    """
    运行单个日期的评估
    
    参数:
    date_str: 日期字符串，格式为MMDDYY（如'022522'）
    base_dir: 项目基础目录，默认为None（使用默认路径）
    data_base_dir: 数据基础目录，默认为None（使用默认路径）
    """
    if base_dir is None:
        base_dir = '/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels'
    if data_base_dir is None:
        data_base_dir = '/media/ubuntu/sda/data/mouse6/ns4/natural_image'
    
    # 配置路径（应与train_spike_pipeline.py中保存的模型路径一致）
    pipeline_results_dir = os.path.join(base_dir, 'pipeline_results')
    
    # 使用AutoSort模型（noise分类器作为detection，label分类器作为classification）
    autosort_noise_path = os.path.join(pipeline_results_dir, 'autosort_trail_2_noise_clsfier.pth')
    autosort_label_path = os.path.join(pipeline_results_dir, 'autosort_trail_1_label_clsfier.pth')
    
    if not os.path.exists(autosort_noise_path):
        raise FileNotFoundError(f"AutoSort noise model not found: {autosort_noise_path}")
    if not os.path.exists(autosort_label_path):
        raise FileNotFoundError(f"AutoSort label model not found: {autosort_label_path}")
    
    print("[INFO] Using AutoSort model")
    print("[INFO] - Noise classifier (detection): autosort_trail_1_noise_clsfier.pth")
    print("[INFO] - Label classifier (classification): autosort_trail_1_label_clsfier.pth")
    model_paths = {
        'autosort_noise': autosort_noise_path,
        'autosort_label': autosort_label_path
    }
    
    # 固定使用021322的neuron_inf（训练集的neuron_inf）
    neuron_inf_date = '021322'
    neuron_inf_path = os.path.join(base_dir, 'kilosort_spike_sorting/sorting_new', neuron_inf_date, 'neuron_inf.pkl')
    # 如果新路径不存在，尝试旧的路径
    if not os.path.exists(neuron_inf_path):
        neuron_inf_path = os.path.join(base_dir, 'kilosort_spike_sorting/sorting_results', neuron_inf_date, 'neuron_inf.pkl')
    if not os.path.exists(neuron_inf_path):
        raise FileNotFoundError(f"Training neuron_inf not found at {neuron_inf_path}")
    print(f"[INFO] Using training neuron_inf from {neuron_inf_date}: {neuron_inf_path}")
    
    # 注意：AutoSort的类别数会从模型权重中自动推断，不需要从neuron_inf读取
    # 因为训练时的类别数可能与当前neuron_inf中的neuron数量不同
    num_classes = None  # 将从模型权重中推断
    
    # 为每个日期创建独立的输出目录
    output_dir = os.path.join(base_dir, 'eval_results', date_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Ground truth数据路径
    gt_spike_inf_path = os.path.join(base_dir, 'kilosort_spike_sorting/sorting_new', date_str, 'spike_inf.tsv')
    
    # 新数据路径
    new_recording_path = os.path.join(data_base_dir, f'mouse6_{date_str}_natural_image_001.ns4')
    
    # 通道配置（从notebook中复制）
    channel_indices = {
        "1": [1, 3, 5, 7, 9, 11],
        "2": [13, 15, 17, 19, 21, 23],
        "3": [24, 25, 26, 27, 28, 29],
        "4": [12, 14, 16, 18, 20, 22],
        "5": [0, 2, 4, 6, 8, 10]
    }
    
    channel_position = {
        0: [650, 0], 2: [650, 50], 4: [650, 100], 6: [600, 100], 8: [600, 50], 10: [600, 0],
        1: [0, 0], 3: [0, 50], 5: [0, 100], 7: [50, 100], 9: [50, 50], 11: [50, 0],
        13: [150, 200], 15: [150, 250], 17: [150, 300], 19: [200, 300], 21: [200, 250], 23: [200, 200],
        12: [500, 200], 14: [500, 250], 16: [500, 300], 18: [450, 300], 20: [450, 250], 22: [450, 200],
        24: [350, 400], 26: [350, 450], 28: [350, 500], 25: [300, 400], 27: [300, 450], 29: [300, 500]
    }
    
    print("[INFO] Starting spike sorting evaluation pipeline")
    
    # 加载新数据
    print(f"[INFO] Loading recording from {new_recording_path}")
    recording_raw = se.read_blackrock(file_path=new_recording_path)
    recording_recorded = recording_raw.remove_channels(["98", '31', '32'])
    
    recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
    recording_f = spre.common_reference(recording_f, reference="global", operator="median")
    
    # 创建评估器（传入ground truth数据路径）
    evaluator = SpikeSortingEvaluator(
        model_paths, 
        neuron_inf_path, 
        channel_indices, 
        channel_position,
        gt_spike_inf_path=gt_spike_inf_path if os.path.exists(gt_spike_inf_path) else None,
        num_classes=num_classes
    )
    
    # neuron_inf的waveform已经在generate_neuron_inf_phy.py中计算为30维（与train/eval一致）
    
    # 执行校准阶段（前60秒），传入output_dir以保存calibration cluster信息
    evaluator.calibrate_first_10min(recording_f, output_dir=output_dir)
    
    # 处理完整数据
    final_results = evaluator.process_complete_recording(recording_f, output_dir)
    
    print("\n[INFO] Evaluation pipeline completed successfully!")
    print(f"[INFO] Final results: {len(final_results)} spikes from {final_results['neuron_id'].nunique()} neurons")
    
    return final_results

def main():
    """主函数，支持命令行参数或使用默认值"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run spike sorting evaluation pipeline')
    parser.add_argument('--date', type=str, default=None,
                        help='Date string in MMDDYY format (e.g., 022522). If not provided, uses default date.')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory for the project')
    parser.add_argument('--data-base-dir', type=str, default=None,
                        help='Base directory for data files')
    
    args = parser.parse_args()
    
    if args.date:
        # 使用命令行参数
        run_evaluation(args.date, args.base_dir, args.data_base_dir)
    else:
        # 使用默认值（保持向后兼容）
        date_str = '022522'  # 默认日期
        run_evaluation(date_str, args.base_dir, args.data_base_dir)

if __name__ == "__main__":
    main()
