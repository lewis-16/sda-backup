#!/usr/bin/env python
# coding: utf-8
"""
比较noise分类器在不同数据上的logits分布
用于判断是否是由于数据分布差异导致的准确率下降

对于训练好的pth模型（基于021322数据训练），计算022522下：
1. match neuron的logits分布
2. unmatch neuron的logits分布

定义match neuron的方法参考eval_spike_pipeline_template.py中计算gt mapping的方法
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

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

# 添加spike_detection路径以导入函数
sys.path.append('/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/spike_detection')
from train_models import detect_local_minimum_in_window as _detect_local_minimum_in_window

# ==================== 模型定义 ====================

class SimpleClassifier(nn.Module):
    """
    简单的分类器（参考autosort的clssimp类）
    用于noise/spike分类
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


# ==================== 数据加载和预处理 ====================

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


def extract_waveforms(trace0_car, spike_times, left_sample=10, right_sample=20):
    """
    提取waveform（30×30）
    
    参数:
    trace0_car : numpy.ndarray, shape (time_points, n_channels)
    spike_times : numpy.ndarray, 检测到的spike时间点
    left_sample : int, spike前的采样点数，默认10
    right_sample : int, spike后的采样点数，默认20
    
    返回:
    waveforms : numpy.ndarray, shape (n_spikes, n_channels, n_timepoints) = (n_spikes, 30, 30)
    """
    window_size = left_sample + right_sample  # 总共30个采样点
    
    # 过滤边界附近的spike
    valid_mask = (spike_times >= left_sample) & (spike_times < trace0_car.shape[0] - right_sample)
    valid_spike_times = spike_times[valid_mask]
    
    print(f"[INFO] Extracting waveforms for {len(valid_spike_times):,} spikes...")
    
    # 提取waveform
    waveforms = []
    for time_range in tqdm(np.arange(-left_sample, right_sample, dtype=np.int64), desc="Extracting waveforms", leave=False):
        indices = (valid_spike_times + time_range).astype(np.int64)
        if time_range == -left_sample:
            waveform = trace0_car[indices, :]
        else:
            waveform = np.dstack((waveform, trace0_car[indices, :]))
    
    # waveform形状: (n_spikes, n_channels, n_timepoints) = (n_spikes, 30, 30)
    print(f"[INFO] Waveform shape: {waveform.shape}")
    
    return waveform, valid_spike_times


def extract_waveforms_from_gt_spikes(recording_path, gt_spike_inf, max_duration_seconds=60):
    """
    从recording数据中根据GT spike_inf的time提取waveform
    
    参数:
    recording_path : str, recording文件路径
    gt_spike_inf : pandas.DataFrame, GT spike信息（包含time列）
    max_duration_seconds : float, 最大处理时长（秒），默认60秒
    
    返回:
    waveforms : numpy.ndarray, shape (n_spikes, 30, 30)
    valid_spike_times : numpy.ndarray, 有效的spike时间点
    valid_indices : numpy.ndarray, 有效的spike索引（在gt_spike_inf中的索引）
    """
    print(f"\n[INFO] Loading recording from {recording_path}")
    recording_raw = se.read_blackrock(file_path=recording_path)
    recording_recorded = recording_raw.remove_channels(["98", '31', '32'])
    
    print("[INFO] Applying bandpass filter and common reference")
    recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
    recording_f = spre.common_reference(recording_f, reference="global", operator="average")
    
    # 获取采样率
    sampling_rate = recording_f.get_sampling_frequency()
    max_duration_samples = int(max_duration_seconds * sampling_rate)
    total_samples = int(recording_f.get_total_duration() * sampling_rate)
    actual_samples = min(max_duration_samples, total_samples)
    
    print(f"[INFO] Sampling rate: {sampling_rate} Hz")
    print(f"[INFO] Processing {max_duration_seconds}s ({actual_samples:,} samples) of data")
    
    # 获取trace数据
    trace0_car = recording_f.get_traces(segment_index=0, start_frame=0, end_frame=actual_samples)
    
    # 确保转置为 (time_points, n_channels)
    if trace0_car.shape[0] < trace0_car.shape[1]:
        trace0_car = trace0_car.T
    
    print(f"[INFO] Trace shape: {trace0_car.shape} (expected: (time_points, n_channels))")
    
    # 从GT spike_inf中获取spike时间点
    if 'time' not in gt_spike_inf.columns:
        raise ValueError("GT spike_inf must have 'time' column")
    
    gt_times = gt_spike_inf['time'].values.astype(np.int64)
    
    # 过滤掉超出数据范围的spike
    left_sample = 10
    right_sample = 20
    valid_mask = (
        (gt_times >= left_sample) & 
        (gt_times < actual_samples - right_sample) &
        (gt_times < len(trace0_car) - right_sample)
    )
    valid_spike_times = gt_times[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    print(f"[INFO] Valid GT spikes: {len(valid_spike_times):,}/{len(gt_times):,}")
    
    # 提取waveform
    waveforms, _ = extract_waveforms(trace0_car, valid_spike_times)
    
    return waveforms, valid_spike_times, valid_indices


# ==================== 模型加载和logits计算 ====================

def load_noise_classifier(model_path, device='cuda'):
    """
    加载noise分类器模型
    
    参数:
    model_path : str, 模型文件路径
    device : str, 设备（'cuda'或'cpu'）
    
    返回:
    model : SimpleClassifier, 加载的模型
    """
    print(f"\n[INFO] Loading noise classifier from {model_path}")
    
    input_size = 30 * 30  # waveform size (30, 30)
    num_classes = 2  # noise和spike两类
    
    model = SimpleClassifier(input_size, num_classes)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    
    # 处理state_dict（可能是完整模型或state_dict）
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    else:
        # 如果是完整模型，尝试提取state_dict
        try:
            model.load_state_dict(state_dict.state_dict())
        except:
            model = state_dict
    
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Model loaded successfully on {device}")
    
    return model


def compute_logits(model, waveforms, device='cuda', batch_size=1024):
    """
    计算waveform的logits
    
    参数:
    model : SimpleClassifier, noise分类器模型
    waveforms : numpy.ndarray, shape (n_spikes, 30, 30)
    device : str, 设备
    batch_size : int, 批处理大小
    
    返回:
    logits : numpy.ndarray, shape (n_spikes, 2) - [noise_logit, spike_logit]
    """
    print(f"[INFO] Computing logits for {len(waveforms):,} waveforms...")
    
    all_logits = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(waveforms), batch_size), desc="Computing logits"):
            batch = waveforms[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)  # (batch, 30, 30)
            
            # 展平waveform: (batch, 30, 30) -> (batch, 900)
            x_flat = batch_tensor.reshape(batch_tensor.size(0), -1)
            
            # 前向传播获取logits
            logits = model(x_flat)  # (batch, 2)
            
            all_logits.append(logits.cpu().numpy())
    
    logits = np.vstack(all_logits)
    print(f"[INFO] Logits shape: {logits.shape}")
    
    return logits


# ==================== GT mapping计算 ====================

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


def compute_gt_neuron_to_021322_neuron_mapping(
    shape_templates_gt, 
    energy_templates_gt, 
    shape_templates_021322, 
    energy_templates_021322
):
    """
    计算GT neuron（022522）到021322 neuron的映射
    
    参数:
    shape_templates_gt : numpy.ndarray, shape (n_gt_neurons, 30, 30)
    energy_templates_gt : numpy.ndarray, shape (n_gt_neurons, 30)
    shape_templates_021322 : numpy.ndarray, shape (n_neurons_021322, 30, 30)
    energy_templates_021322 : numpy.ndarray, shape (n_neurons_021322, 30)
    
    返回:
    gt_neuron_to_021322_neuron_mapping : dict, {gt_idx: 021322_idx or -1}
        -1表示unmatch
    """
    print("[INFO] Computing GT template to 021322 neuron mapping...")
    n_gt_neurons = shape_templates_gt.shape[0]
    n_neurons_021322 = shape_templates_021322.shape[0]
    
    gt_shape_scores = np.zeros((n_gt_neurons, n_neurons_021322), dtype=np.float32)
    gt_energy_scores = np.zeros((n_gt_neurons, n_neurons_021322), dtype=np.float32)
    
    for gt_idx in tqdm(range(n_gt_neurons), desc="Computing template scores"):
        gt_shape = shape_templates_gt[gt_idx]  # (30, 30)
        gt_energy = energy_templates_gt[gt_idx]  # (30,)
        
        for template_idx in range(n_neurons_021322):
            shape_score, energy_score = compute_template_score(
                gt_shape,
                gt_energy,
                shape_templates_021322[template_idx],
                energy_templates_021322[template_idx],
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
    gt_neuron_to_021322_neuron_mapping = {}
    
    for gt_idx in range(n_gt_neurons):
        if np.sum(gt_sum_scores[gt_idx]) == 0:
            # 所有score都是0，unmatch
            gt_neuron_to_021322_neuron_mapping[gt_idx] = -1  # -1表示unmatch
        else:
            # 取argmax
            best_021322_idx = np.argmax(gt_sum_scores[gt_idx])
            gt_neuron_to_021322_neuron_mapping[gt_idx] = best_021322_idx
    
    matched_count = sum(1 for v in gt_neuron_to_021322_neuron_mapping.values() if v != -1)
    print(f"[INFO] GT template mapping completed: {matched_count}/{n_gt_neurons} matched to 021322 neurons")
    
    return gt_neuron_to_021322_neuron_mapping


def load_gt_data_and_templates(base_dir, date='022522'):
    """
    加载GT数据和templates
    
    参数:
    base_dir : str, 基础目录
    date : str, 日期，默认'022522'
    
    返回:
    gt_spike_inf : pandas.DataFrame, GT spike信息
    gt_neuron_inf : pandas.DataFrame, GT neuron信息
    shape_templates_gt : numpy.ndarray, GT shape templates
    energy_templates_gt : numpy.ndarray, GT energy templates
    shape_templates_021322 : numpy.ndarray, 021322 shape templates
    energy_templates_021322 : numpy.ndarray, 021322 energy templates
    """
    print(f"\n[INFO] Loading GT data and templates for {date}...")
    
    # 加载GT spike_inf
    kilosort_dir = os.path.join(base_dir, 'kilosort_spike_sorting', 'sorting_new', date)
    gt_spike_inf_path = os.path.join(kilosort_dir, 'spike_inf.tsv')
    
    if not os.path.exists(gt_spike_inf_path):
        raise FileNotFoundError(f"GT spike_inf not found: {gt_spike_inf_path}")
    
    print(f"[INFO] Loading GT spike_inf from {gt_spike_inf_path}")
    gt_spike_inf = pd.read_csv(gt_spike_inf_path, sep='\t')
    print(f"[INFO] Loaded {len(gt_spike_inf):,} GT spikes")
    
    # 加载GT neuron_inf
    gt_neuron_inf_path = os.path.join(kilosort_dir, 'neuron_inf.pkl')
    if os.path.exists(gt_neuron_inf_path):
        print(f"[INFO] Loading GT neuron_inf from {gt_neuron_inf_path}")
        with open(gt_neuron_inf_path, 'rb') as f:
            gt_neuron_inf = pickle.load(f)
        print(f"[INFO] Loaded {len(gt_neuron_inf):,} GT neurons")
    else:
        print(f"[WARNING] GT neuron_inf not found: {gt_neuron_inf_path}")
        gt_neuron_inf = None
    
    # 加载GT templates
    gt_shape_template_path = os.path.join(kilosort_dir, 'shape_templates.npy')
    gt_energy_template_path = os.path.join(kilosort_dir, 'energy_templates.npy')
    
    if not os.path.exists(gt_shape_template_path) or not os.path.exists(gt_energy_template_path):
        raise FileNotFoundError(f"GT templates not found:\n  {gt_shape_template_path}\n  {gt_energy_template_path}")
    
    print(f"[INFO] Loading GT templates from {kilosort_dir}")
    shape_templates_gt = np.load(gt_shape_template_path)  # (n_gt_neurons, 30, 30)
    energy_templates_gt = np.load(gt_energy_template_path)  # (n_gt_neurons, 30)
    print(f"[INFO] Loaded GT templates: shape {shape_templates_gt.shape}, energy {energy_templates_gt.shape}")
    
    # 加载021322 templates
    template_dir_021322 = os.path.join(base_dir, 'kilosort_spike_sorting', 'sorting_new', '021322')
    shape_template_path_021322 = os.path.join(template_dir_021322, 'shape_templates.npy')
    energy_template_path_021322 = os.path.join(template_dir_021322, 'energy_templates.npy')
    
    if not os.path.exists(shape_template_path_021322) or not os.path.exists(energy_template_path_021322):
        raise FileNotFoundError(f"021322 templates not found:\n  {shape_template_path_021322}\n  {energy_template_path_021322}")
    
    print(f"[INFO] Loading 021322 templates from {template_dir_021322}")
    shape_templates_021322 = np.load(shape_template_path_021322)  # (n_neurons, 30, 30)
    energy_templates_021322 = np.load(energy_template_path_021322)  # (n_neurons, 30)
    print(f"[INFO] Loaded 021322 templates: shape {shape_templates_021322.shape}, energy {energy_templates_021322.shape}")
    
    return gt_spike_inf, gt_neuron_inf, shape_templates_gt, energy_templates_gt, shape_templates_021322, energy_templates_021322


def determine_match_unmatch_from_gt_spikes(
    gt_spike_inf, 
    gt_neuron_name_to_idx, 
    gt_neuron_to_021322_neuron_mapping
):
    """
    根据GT spike_inf中的neuron信息，判断每个spike是match还是unmatch
    
    参数:
    gt_spike_inf : pandas.DataFrame, GT spike信息（包含time和neuron列）
    gt_neuron_name_to_idx : dict, GT neuron名称到索引的映射
    gt_neuron_to_021322_neuron_mapping : dict, GT neuron索引到021322 neuron索引的映射
    
    返回:
    is_match : numpy.ndarray, bool数组，True表示match neuron，False表示unmatch neuron
    """
    print(f"[INFO] Determining match/unmatch for {len(gt_spike_inf):,} GT spikes...")
    
    # 确定neuron列名
    neuron_col = 'neuron' if 'neuron' in gt_spike_inf.columns else 'Neuron'
    if neuron_col not in gt_spike_inf.columns:
        raise ValueError(f"GT spike_inf must have 'neuron' or 'Neuron' column")
    
    is_match = np.zeros(len(gt_spike_inf), dtype=bool)
    matched_count = 0
    
    for i, (_, row) in enumerate(tqdm(gt_spike_inf.iterrows(), total=len(gt_spike_inf), desc="Determining match/unmatch", leave=False)):
        gt_neuron_name = row[neuron_col]
        
        if pd.notna(gt_neuron_name):
            # 获取GT neuron索引
            gt_idx = gt_neuron_name_to_idx.get(gt_neuron_name, -1)
            if gt_idx != -1:
                # 通过映射判断是否match到021322 neuron
                mapped_021322 = gt_neuron_to_021322_neuron_mapping.get(gt_idx, -1)
                if mapped_021322 != -1:
                    is_match[i] = True
                    matched_count += 1
    
    print(f"[INFO] Match neuron spikes: {matched_count:,}/{len(gt_spike_inf):,} ({matched_count/len(gt_spike_inf)*100:.2f}%)")
    print(f"[INFO] Unmatch neuron spikes: {len(gt_spike_inf) - matched_count:,} ({(len(gt_spike_inf) - matched_count)/len(gt_spike_inf)*100:.2f}%)")
    
    return is_match


# ==================== 分布分析和可视化 ====================

def analyze_logits_distribution(logits, dataset_name):
    """
    分析logits分布
    
    参数:
    logits : numpy.ndarray, shape (n_spikes, 2) - [noise_logit, spike_logit]
    dataset_name : str, 数据集名称
    
    返回:
    stats : dict, 统计信息
    """
    noise_logits = logits[:, 0]  # noise类的logits
    spike_logits = logits[:, 1]  # spike类的logits
    
    # 计算softmax概率
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    noise_probs = probs[:, 0]
    spike_probs = probs[:, 1]
    
    # 统计信息（转换为Python原生类型以便JSON序列化）
    stats = {
        'dataset_name': dataset_name,
        'n_samples': int(len(logits)),
        'noise_logit_mean': float(np.mean(noise_logits)),
        'noise_logit_std': float(np.std(noise_logits)),
        'noise_logit_min': float(np.min(noise_logits)),
        'noise_logit_max': float(np.max(noise_logits)),
        'spike_logit_mean': float(np.mean(spike_logits)),
        'spike_logit_std': float(np.std(spike_logits)),
        'spike_logit_min': float(np.min(spike_logits)),
        'spike_logit_max': float(np.max(spike_logits)),
        'spike_prob_mean': float(np.mean(spike_probs)),
        'spike_prob_std': float(np.std(spike_probs)),
        'spike_prob_median': float(np.median(spike_probs)),
        'predicted_spike_ratio': float(np.mean(spike_probs > 0.05)),  # 预测为spike的比例
    }
    
    print(f"\n[INFO] === {dataset_name} Logits Distribution Statistics ===")
    print(f"  Number of samples: {stats['n_samples']:,}")
    print(f"  Noise logit: mean={stats['noise_logit_mean']:.4f}, std={stats['noise_logit_std']:.4f}, "
          f"range=[{stats['noise_logit_min']:.4f}, {stats['noise_logit_max']:.4f}]")
    print(f"  Spike logit: mean={stats['spike_logit_mean']:.4f}, std={stats['spike_logit_std']:.4f}, "
          f"range=[{stats['spike_logit_min']:.4f}, {stats['spike_logit_max']:.4f}]")
    print(f"  Spike probability: mean={stats['spike_prob_mean']:.4f}, std={stats['spike_prob_std']:.4f}, "
          f"median={stats['spike_prob_median']:.4f}")
    print(f"  Predicted spike ratio (prob > 0.5): {stats['predicted_spike_ratio']:.4f} ({stats['predicted_spike_ratio']*100:.2f}%)")
    
    return stats, noise_logits, spike_logits, spike_probs


def plot_logits_comparison(logits_match, logits_unmatch, output_dir):
    """
    绘制match和unmatch neuron的logits分布比较图
    
    参数:
    logits_match : numpy.ndarray, match neuron的logits
    logits_unmatch : numpy.ndarray, unmatch neuron的logits
    output_dir : str, 输出目录
    """
    print(f"\n[INFO] Plotting logits distribution comparison (Match vs Unmatch)...")
    
    # 提取logits
    noise_logits_match = logits_match[:, 0]
    spike_logits_match = logits_match[:, 1]
    noise_logits_unmatch = logits_unmatch[:, 0]
    spike_logits_unmatch = logits_unmatch[:, 1]
    
    # 计算概率
    probs_match = np.exp(logits_match) / np.sum(np.exp(logits_match), axis=1, keepdims=True)
    spike_probs_match = probs_match[:, 1]
    probs_unmatch = np.exp(logits_unmatch) / np.sum(np.exp(logits_unmatch), axis=1, keepdims=True)
    spike_probs_unmatch = probs_unmatch[:, 1]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Logits Distribution Comparison: Match vs Unmatch Neurons (022522)', fontsize=16, fontweight='bold')
    
    # 1. Noise logits分布
    ax = axes[0, 0]
    ax.hist(noise_logits_match, bins=100, alpha=0.6, label=f'Match (n={len(logits_match):,})', density=True, color='green')
    ax.hist(noise_logits_unmatch, bins=100, alpha=0.6, label=f'Unmatch (n={len(logits_unmatch):,})', density=True, color='orange')
    ax.set_xlabel('Noise Logit', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Noise Logits Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Spike logits分布
    ax = axes[0, 1]
    ax.hist(spike_logits_match, bins=100, alpha=0.6, label=f'Match (n={len(logits_match):,})', density=True, color='green')
    ax.hist(spike_logits_unmatch, bins=100, alpha=0.6, label=f'Unmatch (n={len(logits_unmatch):,})', density=True, color='orange')
    ax.set_xlabel('Spike Logit', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Spike Logits Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Spike概率分布
    ax = axes[1, 0]
    ax.hist(spike_probs_match, bins=100, alpha=0.6, label=f'Match (n={len(logits_match):,})', density=True, color='green')
    ax.hist(spike_probs_unmatch, bins=100, alpha=0.6, label=f'Unmatch (n={len(logits_unmatch):,})', density=True, color='orange')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    ax.set_xlabel('Spike Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Spike Probability Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Logits散点图（noise vs spike）
    ax = axes[1, 1]
    # 采样以减少点数量（如果数据太多）
    n_samples_plot = min(10000, len(logits_match))
    if len(logits_match) > 0:
        indices_match = np.random.choice(len(logits_match), min(n_samples_plot, len(logits_match)), replace=False)
        ax.scatter(noise_logits_match[indices_match], spike_logits_match[indices_match], 
                   alpha=0.3, s=1, label=f'Match (n={len(logits_match):,})', color='green')
    
    n_samples_plot_unmatch = min(10000, len(logits_unmatch))
    if len(logits_unmatch) > 0:
        indices_unmatch = np.random.choice(len(logits_unmatch), min(n_samples_plot_unmatch, len(logits_unmatch)), replace=False)
        ax.scatter(noise_logits_unmatch[indices_unmatch], spike_logits_unmatch[indices_unmatch], 
                   alpha=0.3, s=1, label=f'Unmatch (n={len(logits_unmatch):,})', color='orange')
    
    ax.set_xlabel('Noise Logit', fontsize=12)
    ax.set_ylabel('Spike Logit', fontsize=12)
    ax.set_title('Logits Scatter Plot (Noise vs Spike)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join(output_dir, 'logits_distribution_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Figure saved to {output_path}")
    
    plt.close()


def plot_statistics_comparison(stats_match, stats_unmatch, output_dir):
    """
    绘制统计信息比较图
    
    参数:
    stats_match : dict, match neuron的统计信息
    stats_unmatch : dict, unmatch neuron的统计信息
    output_dir : str, 输出目录
    """
    print(f"\n[INFO] Plotting statistics comparison...")
    
    # 准备数据
    metrics = [
        ('noise_logit_mean', 'Noise Logit Mean'),
        ('noise_logit_std', 'Noise Logit Std'),
        ('spike_logit_mean', 'Spike Logit Mean'),
        ('spike_logit_std', 'Spike Logit Std'),
        ('spike_prob_mean', 'Spike Probability Mean'),
        ('spike_prob_median', 'Spike Probability Median'),
        ('predicted_spike_ratio', 'Predicted Spike Ratio'),
    ]
    
    values_match = [stats_match[m[0]] for m in metrics]
    values_unmatch = [stats_unmatch[m[0]] for m in metrics]
    labels = [m[1] for m in metrics]
    
    # 创建条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values_match, width, label='Match Neurons', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, values_unmatch, width, label='Unmatch Neurons', color='orange', alpha=0.7)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Logits Distribution Statistics Comparison: Match vs Unmatch (022522)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join(output_dir, 'logits_statistics_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Figure saved to {output_path}")
    
    plt.close()


# ==================== 主函数 ====================

def main():
    print("="*80)
    print("Logits Distribution Comparison: Match vs Unmatch Neurons (022522)")
    print("="*80)
    
    # 配置路径
    base_dir = '/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels'
    data_base_dir = '/media/ubuntu/sda/data/mouse6/ns4/natural_image'
    pipeline_results_dir = os.path.join(base_dir, 'pipeline_results')
    output_dir = os.path.join(base_dir, 'logits_comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 模型路径（使用训练好的noise分类器）
    noise_model_path = os.path.join(pipeline_results_dir, 'autosort_trail_1_noise_clsfier.pth')
    
    if not os.path.exists(noise_model_path):
        raise FileNotFoundError(f"Noise classifier model not found: {noise_model_path}")
    
    # 数据路径
    recording_022522_path = os.path.join(data_base_dir, 'mouse6_022522_natural_image_001.ns4')
    
    if not os.path.exists(recording_022522_path):
        raise FileNotFoundError(f"Recording 022522 not found: {recording_022522_path}")
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    # 加载GT数据和templates
    print("\n" + "="*80)
    print("Loading GT data and templates")
    print("="*80)
    gt_spike_inf, gt_neuron_inf, shape_templates_gt, energy_templates_gt, shape_templates_021322, energy_templates_021322 = load_gt_data_and_templates(
        base_dir, date='022522'
    )
    
    # 计算GT neuron到021322 neuron的映射
    print("\n" + "="*80)
    print("Computing GT neuron to 021322 neuron mapping")
    print("="*80)
    gt_neuron_to_021322_neuron_mapping = compute_gt_neuron_to_021322_neuron_mapping(
        shape_templates_gt,
        energy_templates_gt,
        shape_templates_021322,
        energy_templates_021322
    )
    
    # 构建GT neuron名称到索引的映射
    print("\n[INFO] Building GT neuron name to index mapping...")
    gt_neuron_name_to_idx = {}
    if gt_neuron_inf is not None and len(gt_neuron_inf) > 0:
        neuron_col = 'neuron' if 'neuron' in gt_neuron_inf.columns else 'Neuron'
        if neuron_col in gt_neuron_inf.columns:
            # GT template的索引顺序应该与GT neuron_inf中的neuron顺序一致
            for idx, (_, row) in enumerate(gt_neuron_inf.iterrows()):
                neuron_name = row[neuron_col]
                if pd.notna(neuron_name):
                    gt_neuron_name_to_idx[neuron_name] = idx
            print(f"[INFO] Built GT neuron name to index mapping from neuron_inf: {len(gt_neuron_name_to_idx)} neurons")
            if shape_templates_gt is not None:
                if len(gt_neuron_name_to_idx) != shape_templates_gt.shape[0]:
                    print(f"[WARNING] GT neuron count ({len(gt_neuron_name_to_idx)}) != GT template count ({shape_templates_gt.shape[0]})")
    else:
        # 如果没有neuron_inf，从spike_inf中构建
        neuron_col = 'neuron' if 'neuron' in gt_spike_inf.columns else 'Neuron'
        if neuron_col in gt_spike_inf.columns:
            # 获取所有唯一的GT neuron名称，并按字母顺序排序（这样索引就是稳定的）
            unique_gt_neurons = sorted(gt_spike_inf[neuron_col].dropna().unique())
            # 假设GT template的索引顺序与sorted unique neurons的顺序一致
            for idx, neuron_name in enumerate(unique_gt_neurons):
                gt_neuron_name_to_idx[neuron_name] = idx
            print(f"[INFO] Built GT neuron name to index mapping from spike_inf (sorted): {len(gt_neuron_name_to_idx)} neurons")
            if shape_templates_gt is not None:
                if len(gt_neuron_name_to_idx) != shape_templates_gt.shape[0]:
                    print(f"[WARNING] GT neuron count ({len(gt_neuron_name_to_idx)}) != GT template count ({shape_templates_gt.shape[0]})")
    
    # 加载模型
    print("\n" + "="*80)
    print("Loading noise classifier model")
    print("="*80)
    model = load_noise_classifier(noise_model_path, device)
    
    # 从GT spike_inf中提取waveform（只使用前60秒数据以加快处理速度）
    print("\n" + "="*80)
    print("Extracting waveforms from GT spikes")
    print("="*80)
    
    # 过滤GT spike_inf，只保留前60秒的数据
    if 'time' not in gt_spike_inf.columns:
        raise ValueError("GT spike_inf must have 'time' column")
    
    sampling_rate = 10000  # 假设采样率为10000 Hz
    max_duration_samples = 60 * sampling_rate
    gt_spike_inf_filtered = gt_spike_inf[gt_spike_inf['time'] < max_duration_samples].copy()
    print(f"[INFO] Filtered GT spikes (first 60s): {len(gt_spike_inf_filtered):,}/{len(gt_spike_inf):,}")
    
    waveforms_022522, valid_spike_times, valid_indices = extract_waveforms_from_gt_spikes(
        recording_022522_path, 
        gt_spike_inf_filtered,
        max_duration_seconds=60
    )
    
    # 计算logits
    print("\n" + "="*80)
    print("Computing logits")
    print("="*80)
    logits_022522 = compute_logits(model, waveforms_022522, device)
    
    # 判断match/unmatch（基于GT spike_inf中的neuron信息）
    print("\n" + "="*80)
    print("Determining match/unmatch from GT neurons")
    print("="*80)
    # 只对有效的spike判断match/unmatch
    gt_spike_inf_valid = gt_spike_inf_filtered.iloc[valid_indices].reset_index(drop=True)
    is_match = determine_match_unmatch_from_gt_spikes(
        gt_spike_inf_valid,
        gt_neuron_name_to_idx,
        gt_neuron_to_021322_neuron_mapping
    )
    
    # 分离match和unmatch的logits
    logits_match = logits_022522[is_match]
    logits_unmatch = logits_022522[~is_match]
    
    print(f"\n[INFO] Match neuron spikes: {len(logits_match):,}")
    print(f"[INFO] Unmatch neuron spikes: {len(logits_unmatch):,}")
    
    # 分析分布
    print("\n" + "="*80)
    print("Analyzing distributions")
    print("="*80)
    if len(logits_match) > 0:
        stats_match, noise_logits_match, spike_logits_match, spike_probs_match = analyze_logits_distribution(
            logits_match, 'Match Neurons (022522)'
        )
    else:
        print("[WARNING] No match neuron spikes found!")
        stats_match = {}
    
    if len(logits_unmatch) > 0:
        stats_unmatch, noise_logits_unmatch, spike_logits_unmatch, spike_probs_unmatch = analyze_logits_distribution(
            logits_unmatch, 'Unmatch Neurons (022522)'
        )
    else:
        print("[WARNING] No unmatch neuron spikes found!")
        stats_unmatch = {}
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, 'logits_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'match': stats_match,
            'unmatch': stats_unmatch,
            'gt_neuron_to_021322_neuron_mapping': {str(k): v for k, v in gt_neuron_to_021322_neuron_mapping.items()}
        }, f, indent=2)
    print(f"\n[INFO] Statistics saved to {stats_path}")
    
    # 保存logits和spike_times
    print(f"\n[INFO] Saving logits and spike_times...")
    logits_match_path = os.path.join(output_dir, 'logits_match.npy')
    logits_unmatch_path = os.path.join(output_dir, 'logits_unmatch.npy')
    spike_times_022522_path = os.path.join(output_dir, 'spike_times_022522.npy')
    is_match_path = os.path.join(output_dir, 'is_match.npy')
    
    np.save(logits_match_path, logits_match)
    np.save(logits_unmatch_path, logits_unmatch)
    np.save(spike_times_022522_path, valid_spike_times)
    np.save(is_match_path, is_match)
    print(f"[INFO] Logits saved to {logits_match_path} and {logits_unmatch_path}")
    print(f"[INFO] Spike times saved to {spike_times_022522_path}")
    print(f"[INFO] Match mask saved to {is_match_path}")
    
    # 可视化
    print("\n" + "="*80)
    print("Generating visualizations")
    print("="*80)
    if len(logits_match) > 0 and len(logits_unmatch) > 0:
        plot_logits_comparison(logits_match, logits_unmatch, output_dir)
        plot_statistics_comparison(stats_match, stats_unmatch, output_dir)
    else:
        print("[WARNING] Cannot generate visualizations: missing match or unmatch data")
    
    # 打印总结
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\n[INFO] Results saved to: {output_dir}")
    if len(logits_match) > 0 and len(logits_unmatch) > 0:
        print(f"\n[INFO] Key differences (Match vs Unmatch):")
        print(f"  - Spike logit mean: {stats_match['spike_logit_mean']:.4f} (Match) vs {stats_unmatch['spike_logit_mean']:.4f} (Unmatch)")
        print(f"    Difference: {stats_unmatch['spike_logit_mean'] - stats_match['spike_logit_mean']:.4f}")
        print(f"  - Spike probability mean: {stats_match['spike_prob_mean']:.4f} (Match) vs {stats_unmatch['spike_prob_mean']:.4f} (Unmatch)")
        print(f"    Difference: {stats_unmatch['spike_prob_mean'] - stats_match['spike_prob_mean']:.4f}")
        print(f"  - Predicted spike ratio: {stats_match['predicted_spike_ratio']:.4f} (Match) vs {stats_unmatch['predicted_spike_ratio']:.4f} (Unmatch)")
        print(f"    Difference: {stats_unmatch['predicted_spike_ratio'] - stats_match['predicted_spike_ratio']:.4f}")
    
    print("\n[INFO] Analysis completed!")


if __name__ == "__main__":
    main()

