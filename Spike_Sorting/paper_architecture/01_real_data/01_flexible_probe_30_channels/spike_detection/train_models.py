#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import spikeinterface.qualitymetrics as sqm
import json
import probeinterface

from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe, plot_probegroup
from probeinterface import generate_dummy_probe, generate_linear_probe
from probeinterface import write_probeinterface, read_probeinterface
from probeinterface import write_prb, read_prb
from torch.nn.functional import max_pool1d


import torch.nn.functional as F
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import Subset
from scipy.signal import find_peaks

# from function.Function import *


# In[6]:


def count_array2_in_range_of_array1(array1, array2, threshold=5):

    sorted_array1 = np.sort(array1)
    
    lefts = array2 - threshold
    rights = array2 + threshold
    
    left_indices = np.searchsorted(sorted_array1, lefts, side='left')
    
    right_indices = np.searchsorted(sorted_array1, rights, side='right')
    
    has_within_range = right_indices > left_indices
    
    count = np.sum(has_within_range)
    
    return count


def detect_local_minimum_in_window(data, window_size=20, std_multiplier=2, min_distance=None):

    """
    使用 scipy.signal.find_peaks 在每个滑动窗口范围内检测局部最小值的索引，
    并确保最小值低于 mean - std_multiplier * std。每个窗口内只保留一个检测值。

    参数:
    data : numpy.ndarray
        输入数据，形状为 (n_rows, n_columns)。
    window_size : int
        滑动窗口的大小，用于定义局部范围，默认为 20。
    std_multiplier : float
        标准差的倍数，用于筛选局部最小值，默认为 2。
    min_distance : int, optional
        峰值之间的最小距离（采样点数）。如果为 None，则使用 window_size // 2。

    返回:
    local_minima_indices : list
        所有通道局部最小值的索引列表（已去重）。
    """
    if min_distance is None:
        min_distance = max(1, window_size // 2)
    
    local_minima_indices = []

    for row in data:
        row = row.astype(np.float32)
        row_mean = np.mean(row)
        row_std = np.std(row)
        threshold = row_mean - std_multiplier * row_std
        
        # 反转信号以检测最小值（find_peaks 检测最大值）
        inverted_row = -row
        
        # 使用 find_peaks 检测峰值（对应原信号的最小值）
        # height: 峰值必须高于此值（对于反转信号，即原信号必须低于 -threshold）
        # distance: 峰值之间的最小距离
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


def cluster_label_array1_based_on_array2(array1, array2, threshold=5):

    """
    根据 array2 的 'time' 和 'cluster' 对 array1 进行标记。
    如果 array1 中的某个值在 threshold 范围内存在于 array2 的 'time' 中，则标记为对应的 'cluster' 值，否则为 0。
    
    参数:
    array1 : numpy.ndarray
        要标记的数组。
    array2 : numpy.ndarray
        包含 'time' 和 'cluster' 的二维数组。
        第一列为 'time'，第二列为 'cluster'。
    threshold : int
        判断范围的阈值。
    
    返回:
    labels : numpy.ndarray
        长度为 len(array1) 的标签数组，值为 array2 中的 'cluster' 或 0。
    """

    array2 = np.array(array2.iloc[:, [5, 1]])
    sorted_indices = np.argsort(array2[:, 0])
    sorted_array2 = array2[sorted_indices]
    
    labels = np.zeros(len(array1), dtype=int)
    
    # 遍历 array1 中的每个元素
    for i, value in enumerate(array1):
        # 计算当前值的范围
        left = value - threshold
        right = value + threshold
        
        left_index = np.searchsorted(sorted_array2[:, 0], left, side='left')
        right_index = np.searchsorted(sorted_array2[:, 0], right, side='right')
        
        # 如果范围内存在值，则标记为对应的 'cluster'
        if right_index > left_index:
            # 获取范围内的第一个匹配值的 'cluster'
            labels[i] = sorted_array2[left_index, 1]
    
    return labels


def label_array1_based_on_array2(array1, array2, threshold=5, use_nearest_neighbor=True):

    """
    根据 array2 的值对 array1 进行标记。
    如果 array1 中的某个值在 threshold 范围内存在于 array2 中，则标记为 1，否则为 0。
    
    参数:
    array1 : numpy.ndarray
        要标记的数组（检测到的spike时间点）。
    array2 : numpy.ndarray
        用于判断的数组（ground truth spike时间点）。
    threshold : int
        判断范围的阈值（采样点数）。建议使用较小的值（0-2）以避免相邻waveform被重复标记。
    use_nearest_neighbor : bool
        如果为 True，使用最近邻匹配策略，确保每个ground truth spike最多只匹配一个检测spike。
        这可以避免相邻waveform被重复标记的问题。默认为 True。
    
    返回:
    labels : numpy.ndarray
        长度为 len(array1) 的标签数组，值为 0 或 1。
    """
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    # 初始化标签数组，默认值为 0
    labels = np.zeros(len(array1), dtype=int)
    
    if len(array2) == 0:
        return labels
    
    if use_nearest_neighbor:
        # 使用最近邻匹配策略：确保每个ground truth spike最多只匹配一个检测spike
        # 对 array2 进行排序以加速搜索
        sorted_array2 = np.sort(array2)
        
        # 记录每个ground truth spike是否已被匹配
        matched_gt_indices = set()
        
        # 对 array1 按值排序，以便优先匹配更接近的spike
        sorted_indices = np.argsort(array1)
        
        # 对于每个检测到的spike，找到最近的ground truth spike
        for idx in sorted_indices:
            value = array1[idx]
            
            # 找到最近的ground truth spike
            nearest_idx = np.searchsorted(sorted_array2, value, side='left')
            
            # 检查左右两个候选位置
            candidates = []
            if nearest_idx > 0:
                candidates.append((nearest_idx - 1, sorted_array2[nearest_idx - 1]))
            if nearest_idx < len(sorted_array2):
                candidates.append((nearest_idx, sorted_array2[nearest_idx]))
            
            # 找到距离最近的候选
            if candidates:
                best_idx, best_value = min(candidates, key=lambda x: abs(x[1] - value))
                distance = abs(best_value - value)
                
                # 如果距离在阈值内且该ground truth spike未被匹配
                if distance <= threshold and best_idx not in matched_gt_indices:
                    labels[idx] = 1
                    matched_gt_indices.add(best_idx)
    else:
        # 原始策略：简单的范围匹配（可能一个ground truth匹配多个检测spike）
        sorted_array2 = np.sort(array2)
        
        for i, value in enumerate(array1):
            left = value - threshold
            right = value + threshold
            
            left_index = np.searchsorted(sorted_array2, left, side='left')
            right_index = np.searchsorted(sorted_array2, right, side='right')
            
            if right_index > left_index:
                labels[i] = 1
    
    return labels


def extract_windows(data, indices, window_size=61):
    """
    根据给定的时间点索引提取窗口。
    
    参数:
    data : numpy.ndarray
        输入数据，形状为 (n_channels, time)
    indices : numpy.ndarray
        时间点索引数组，用于指定需要提取窗口的中心点
    window_size : int
        窗口长度，默认为61（对应time-30到time+31）
    
    返回:
    windows : numpy.ndarray
        提取的窗口数据，形状为 (len(indices), n_channels, window_size)
    """
    n_channels, time_length = data.shape
    half_window = window_size // 2

    if np.any(indices < half_window) or np.any(indices >= time_length - half_window):
        raise ValueError("Some indices are out of bounds for the given window size.")

    windows = []
    for idx in indices:
        window = data[:, idx - half_window:idx + half_window + 1]
        windows.append(window)

    windows = np.array(windows)
    return windows


# In[7]:


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].astype(np.float32), self.labels[idx]
    
class Spike_Detection_MLP(nn.Module):
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


# In[8]:


def main():
    print("[INFO] Starting spike detection training pipeline")
    recording_path = '/media/ubuntu/sda/data/mouse6/ns4/natural_image/mouse6_021322_natural_image_001.ns4'
    print(f"[INFO] Loading recording from {recording_path}")
    recording_raw = se.read_blackrock(file_path=recording_path)
    print(f"[INFO] Raw recording duration: {recording_raw.get_total_duration():.2f}s | Channels: {recording_raw.get_num_channels()}")

    recording_recorded = recording_raw.remove_channels(["98", '31', '32'])

    print("[INFO] Applying bandpass filter and common reference")
    recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
    recording_f = spre.common_reference(recording_f, reference="global", operator="median")
    spike_inf_path = "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/kilosort_spike_sorting/sorting_results/021322/spike_inf.csv"
    print(f"[INFO] Loading spike annotations from {spike_inf_path}")
    spike_inf = pd.read_csv(spike_inf_path)

    total_frames = int(recording_f.get_total_duration() * 10000)
    chunk_size = 300000
    window_size = 31
    half_window = window_size // 2

    print(f"[INFO] Total frames: {total_frames:,} | Chunk size: {chunk_size} | Window size: {window_size}")
    all_valid_indices = []
    all_windows = []

    # 计算总chunk数用于进度条
    total_chunks = (total_frames + chunk_size - 1) // chunk_size
    print(f"[INFO] Processing {total_chunks} chunks...")
    
    for start_frame in tqdm(range(0, total_frames, chunk_size), 
                             desc="Detecting spikes", 
                             unit="chunk",
                             total=total_chunks):
        end_frame = min(start_frame + chunk_size, total_frames)

        data_chunk = recording_f.get_traces(
            start_frame=start_frame,
            end_frame=end_frame
        )  # shape: (n_channels, chunk_size)

        threshold_result = detect_local_minimum_in_window(
            data_chunk.T,
            std_multiplier=2.4,
            window_size=10
        )

        threshold_result = np.array(threshold_result) + start_frame
        valid_indices = threshold_result[
            (threshold_result >= start_frame + half_window + 1) &
            (threshold_result < end_frame - half_window)
        ]

        for idx in valid_indices:
            rel_idx = idx - start_frame
            window = data_chunk.T[:, rel_idx - half_window: rel_idx + half_window + 1]
            all_windows.append(window)

        all_valid_indices.extend(valid_indices)

    all_valid_indices = np.array(all_valid_indices)
    all_windows = np.stack(all_windows)
    print(f"[INFO] Extracted {len(all_valid_indices):,} candidate spikes")

    labels = label_array1_based_on_array2(all_valid_indices, spike_inf['time'], threshold=1)
    labels = np.array(labels)
    detected_spike_count = len(all_valid_indices)
    matched_spike_count = int(np.sum(labels == 1))
    total_gt_spikes = len(spike_inf)
    coverage = matched_spike_count / total_gt_spikes if total_gt_spikes > 0 else 0
    print(f"[INFO] Spike detection summary:")
    print(f"       - Detected spikes: {detected_spike_count:,}")
    print(f"       - Matched to ground truth: {matched_spike_count:,}")
    print(f"       - Ground truth total spikes: {total_gt_spikes:,}")
    print(f"       - Coverage (matched / ground truth): {coverage * 100:.2f}%")

    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]
    print(f"[INFO] Label distribution -> Negative: {len(indices_0):,}, Positive: {len(indices_1):,}")

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

    dataset = CustomDataset(sampled_windows, sampled_labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"[INFO] Train size: {train_size:,} | Test size: {test_size:,}")

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    accuracy_list = []
    tpr_list = []
    tnr_list = []

    input_size = sampled_windows.shape[1] * sampled_windows.shape[2]
    hidden_size1 = 128
    hidden_size2 = 32
    output_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

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
                loss = criterion(outputs, batch_labels)

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
                torch.save(model, f'/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels/spike_detection/train_results/trail_{trail}.pth')
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

    print("\n[INFO] Training summary across trails")
    for idx, (acc, tpr, tnr) in enumerate(zip(accuracy_list, tpr_list, tnr_list), start=1):
        print(f" Trail {idx}: Acc={acc * 100:.2f}%, TPR={tpr * 100:.2f}%, TNR={tnr * 100:.2f}%")

    print(f"\n[INFO] Mean Accuracy: {np.mean(accuracy_list) * 100:.2f}% ± {np.std(accuracy_list) * 100:.2f}%")
    print(f"[INFO] Mean TPR: {np.mean(tpr_list) * 100:.2f}% ± {np.std(tpr_list) * 100:.2f}%")
    print(f"[INFO] Mean TNR: {np.mean(tnr_list) * 100:.2f}% ± {np.std(tnr_list) * 100:.2f}%")
    print("[INFO] Spike detection training pipeline completed successfully")


if __name__ == "__main__":
    main()
