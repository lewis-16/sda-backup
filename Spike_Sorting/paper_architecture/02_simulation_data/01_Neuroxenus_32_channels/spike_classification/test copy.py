import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm

from sklearn.decomposition import PCA
import umap
import random

import sys
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

import matplotlib.pyplot as plt
import json

from probeinterface import write_prb, read_prb

import torch.nn.functional as F
from pathlib import Path

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import os
torch.set_num_threads(2)  # 限制 PyTorch 使用的 CPU 线程数

def count_array2_in_range_of_array1(array1, array2, threshold=5):

    sorted_array1 = np.sort(array1)
    
    lefts = array2 - threshold
    rights = array2 + threshold
    
    left_indices = np.searchsorted(sorted_array1, lefts, side='left')
    
    right_indices = np.searchsorted(sorted_array1, rights, side='right')
    
    has_within_range = right_indices > left_indices
    
    count = np.sum(has_within_range)
    
    return count


def detect_local_maxima_in_window(data, window_size=20, std_multiplier=2):

    """
    在每个滑动窗口范围内检测局部最大值的索引，并确保最大值大于两倍的标准差。

    参数:
    data : numpy.ndarray
        输入数据，形状为 (n_rows, n_columns)。
    window_size : int
        滑动窗口的大小，用于定义局部范围，默认为 20。
    std_multiplier : float
        标准差的倍数，用于筛选局部最大值，默认为 2。

    返回:
    local_maxima_indices : list of numpy.ndarray
        每行局部最大值的索引列表，每个元素是对应行局部最大值的索引数组。
    """
    local_maxima_indices = []

    for row in data:
        maxima_indices = []
        row_std = np.std(row)
        threshold = std_multiplier * row_std

        for start in range(0, len(row), window_size):
            end = min(start + window_size, len(row))
            window = row[start:end]
            
            if len(window) > 0:
                local_max_index = np.argmax(window)
                local_max_value = window[local_max_index]
                
                if local_max_value > threshold:
                    maxima_indices.append(start + local_max_index)  
        
        local_maxima_indices.extend(maxima_indices)
        local_maxima_indices = list(set(local_maxima_indices))  

    return local_maxima_indices


def cluster_label_array1_based_on_array2(array1, array2, threshold=5, 
                                         cluster_column='cluster'):

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

    array2 = np.array((array2['time'], array2[cluster_column])).T
    sorted_indices = np.argsort(array2[:, 0])
    sorted_array2 = array2[sorted_indices]
    
    labels = -np.ones(len(array1), dtype=int)
    
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


def label_array1_based_on_array2(array1, array2, threshold=5):

    """
    根据 array2 的值对 array1 进行标记。
    如果 array1 中的某个值在 threshold 范围内存在于 array2 中，则标记为 1，否则为 0。
    
    参数:
    array1 : numpy.ndarray
        要标记的数组。
    array2 : numpy.ndarray
        用于判断的数组。
    threshold : int
        判断范围的阈值。
    
    返回:
    labels : numpy.ndarray
        长度为 len(array1) 的标签数组，值为 0 或 1。
    """
    # 对 array2 进行排序以加速搜索
    sorted_array2 = np.sort(array2)
    
    # 初始化标签数组，默认值为 0
    labels = np.zeros(len(array1), dtype=int)
    
    # 遍历 array1 中的每个元素
    for i, value in enumerate(array1):
        # 计算当前值的范围
        left = value - threshold
        right = value + threshold
        
        # 使用二分搜索判断范围内是否存在值
        left_index = np.searchsorted(sorted_array2, left, side='left')
        right_index = np.searchsorted(sorted_array2, right, side='right')
        
        # 如果范围内存在值，则标记为 1
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
def calculate_position(row):
    probe_group = str(row['probe_group'])
    channels = channel_indices[probe_group]
    waveform = row['waveform'] 
    
    a_squared = [np.sum(waveform[:, j]**2) for j in range(len(channels))]
    
    sum_x_a = 0
    sum_y_a = 0
    sum_a = 0
    
    for j, channel in enumerate(channels):
        x_i, y_i = channel_position.get(channel, [0, 0])  
        a_i_sq = a_squared[j]
        
        sum_x_a += x_i * a_i_sq
        sum_y_a += y_i * a_i_sq
        sum_a += a_i_sq
    
    if sum_a == 0:
        return pd.Series({'position_1': 0, 'position_2': 0})
    
    x_hat = sum_x_a / sum_a
    y_hat = sum_y_a / sum_a
    return pd.Series({'position_1': x_hat, 'position_2': y_hat})

def calculate_position_waveform(row, channel_position, channel_indices, power=2):
    x_target = row['position_1']
    y_target = row['position_2']
    probe_group = str(row['probe_group'])
    channels = channel_indices[probe_group]  
    waveforms = row['waveform']  
    
    distances = []
    for channel in channels:
        x_channel, y_channel = channel_position.get(channel, [np.nan, np.nan])
        if np.isnan(x_channel):  
            continue
        distance = np.sqrt((x_target - x_channel)**2 + (y_target - y_channel)**2)
        distances.append(distance)
    
    if not distances:  
        return np.zeros(31)
    
    #IDW
    weights = 1 / (np.array(distances) ** power)
    if np.any(distances == 0):
        zero_idx = np.argwhere(distances == 0).flatten()
        return waveforms[:, zero_idx[0]]
    
    weights /= np.sum(weights)
    
    synthesized_waveform = np.zeros(31)
    for t in range(31): 
        weighted_sum = np.dot(waveforms[t, :], weights)
        synthesized_waveform[t] = weighted_sum
    
    return synthesized_waveform


def compute_cluster_average(sample_data, potent_spike_inf, cluster_column='cluster_predicted'):
    """
    计算 potent_spike_inf 中每个 cluster_predicted 对应的 sample_data 的平均值。
    
    参数:
    - sample_data: np.ndarray, 输入的 (n, 30, 61) 矩阵。
    - potent_spike_inf: pd.DataFrame, 包含 cluster_predicted 信息的 DataFrame。
    - cluster_column: str, cluster 信息所在的列名。
    
    返回:
    - cluster_averages: dict, 每个 cluster 对应的平均值矩阵 (30, 61)。
    """
    cluster_averages = {}
    unique_clusters = potent_spike_inf[cluster_column].unique()
    
    for cluster in unique_clusters:
        cluster_indices = potent_spike_inf[potent_spike_inf[cluster_column] == cluster].index
        cluster_average = sample_data[cluster_indices].mean(axis=0) 
        cluster_averages[cluster] = cluster_average
    
    return cluster_averages

def judge_cluster_reality(row, neuron_inf):
    from scipy.stats import pearsonr

    position_threshold =10
    position_condition = (
        (abs(neuron_inf['position_1'] - row['position_1']) <= position_threshold) &
        (abs(neuron_inf['position_2'] - row['position_2']) <= position_threshold)
    )

    candidate_neurons = neuron_inf[position_condition]

    if candidate_neurons.empty:
        return None

    waveform_threshold = 0.95
    row_waveform = row['position_waveform']
    best_match = None
    best_corr = -1 

    for _, candidate in candidate_neurons.iterrows():
        neuron_inf_waveform = candidate['position_waveform'][15:-15]
        corr, _ = pearsonr(row_waveform, neuron_inf_waveform)

        if corr > waveform_threshold and corr > best_corr:
            best_corr = corr
            best_match = candidate['cluster']

    return best_match if best_match is not None else None


def process_cluster_averages(cluster_averages, channel_indices):
    """
    对 cluster_averages 中的每个 item，找到最大值所在的通道，
    并根据 channel_indices 保留对应的 6 个通道。
    
    参数:
    - cluster_averages: dict, 每个 cluster 的平均值 (30, 61)。
    - channel_indices: dict, 通道索引字典。
    
    返回:
    - processed_averages: dict, 处理后的字典，键为 cluster_channelindices，值为 (6, 61) 的数组。
    """
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

def predict_new(feature, kmeans):
    dists = pairwise_distances(feature, kmeans.cluster_centers_ )
    return np.argmin(dists, axis=1)


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        return self.data[idx].astype(np.float32), self.labels[idx]

class Spike_Classification_MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Spike_Classification_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)  

    def forward(self, x):
        x = x.reshape(-1, 31 * 32)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)  
        return x


# file = 'Neuronexus_32_50_cell_recordings'
# print(f'Processing {file}...')
# file = file.split(".")[0]
# recording_raw = se.MEArecRecordingExtractor(file_path=f'/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/data_generation/setting_1/{file}.h5')
# recording_f = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)
# recording_f = spre.common_reference(recording_f, reference="global", operator="median")
# spike_inf = pd.read_csv(f"/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/data_generation/setting_1/spike_inf.csv")

# unique_clusters = np.unique(spike_inf['Neuron'])
# cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
# spike_inf['Neuron'] = np.array([cluster_to_index[cluster] for cluster in spike_inf['Neuron']])

# spike_inf_train = spike_inf[spike_inf['time'] < 1000 * 10000]
# spike_inf_val = spike_inf[spike_inf['time'] > 1000 * 10000]

# total_frames = int(1000 * 10000)
# chunk_size = 100000  
# window_size = 31
# half_window = window_size // 2

# all_valid_indices = []
# all_windows = []
# potent_spike_inf = pd.DataFrame()

# for start_frame in range(0, total_frames, chunk_size):
#     end_frame = min(start_frame + chunk_size, total_frames)

#     spike_inf_temp = spike_inf_train[(spike_inf_train['time'] > start_frame + 15) & (spike_inf_train['time'] < end_frame - 16)]
    
#     data_chunk = recording_f.get_traces(
#         start_frame=start_frame,
#         end_frame=end_frame
#     )  # shape: (n_channels, chunk_size)  
    
#     valid_indices = spike_inf_temp['time'].values
#     for idx in valid_indices:
#         rel_idx = idx - start_frame
#         window = data_chunk.T[:, rel_idx-half_window : rel_idx+half_window+1]
#         all_windows.append(window)
    
#     all_valid_indices.extend(valid_indices)

#     potent_spike_inf = pd.concat((potent_spike_inf, pd.DataFrame((valid_indices, spike_inf_temp['Neuron'])).T), axis=0)

# all_valid_indices = np.array(all_valid_indices)
# all_windows = np.stack(all_windows)

# potent_spike_inf.columns = ['time', 'cluster']

# indices = potent_spike_inf['time'].values

# labels = potent_spike_inf['cluster'].values

# balanced_indices = []
# for cluster in potent_spike_inf['cluster'].unique():
#     cluster_indices = np.where(labels == cluster)[0]
#     if len(cluster_indices) > 8000:
#         sampled_indices = np.random.choice(cluster_indices, 8000, replace=False)
#     else:
#         sampled_indices = cluster_indices
#     balanced_indices.extend(sampled_indices)

# np.random.shuffle(balanced_indices)

# balanced_data = all_windows[balanced_indices]
# balanced_labels = labels[balanced_indices]

# dataset = CustomDataset(balanced_data, balanced_labels)

# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# batch_size = 1024
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 0)
# #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 0)

# device = 'cuda'
# input_size = balanced_data.shape[1] * balanced_data.shape[2]
# hidden_size1 = 64
# hidden_size2 = 50
# num_classes = potent_spike_inf['cluster'].nunique()

# total_frames = int(1000 * 10000)
# chunk_size = 100000  
# window_size = 31
# half_window = window_size // 2

# all_valid_indices = []
# all_windows = []
# potent_spike_inf_val = pd.DataFrame()

# for start_frame in range(1000 * 10000, 1600 * 10000, chunk_size):
#     end_frame = min(start_frame + chunk_size, 1600 * 10000)

#     spike_inf_temp = spike_inf_val[(spike_inf_val['time'] > start_frame + 15) & (spike_inf_val['time'] < end_frame - 16)]
    
#     data_chunk = recording_f.get_traces(
#         start_frame=start_frame,
#         end_frame=end_frame
#     )  # shape: (n_channels, chunk_size)
    
#     valid_indices = spike_inf_temp['time'].values
#     for idx in valid_indices:
#         rel_idx = idx - start_frame
#         window = data_chunk.T[:, rel_idx-half_window : rel_idx+half_window+1]
#         all_windows.append(window)
    
#     all_valid_indices.extend(valid_indices)

#     potent_spike_inf_val = pd.concat((potent_spike_inf_val, pd.DataFrame((valid_indices, spike_inf_temp['Neuron'])).T), axis=0)

# all_valid_indices = np.array(all_valid_indices)
# all_windows = np.stack(all_windows)
# potent_spike_inf_val.columns = ['time', 'cluster']
# indices = potent_spike_inf_val['time'].values
# labels = potent_spike_inf_val['cluster'].values

# dataset = CustomDataset(all_windows, labels)

# val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# accuracy_list = []
# os.makedirs(f"/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/spike_classification/train_results/setting_1/{file}", exist_ok=True)
# for trail in range(1, 6):
#     model = Spike_Classification_MLP(input_size, hidden_size1, hidden_size2, num_classes)
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()  
#     optimizer = optim.Adam(model.parameters(), lr=0.00001)

#     num_epochs = 210
#     accuracy_best = 0
#     i = 0
#     for epoch in range(num_epochs):
#         all_labels = []
#         all_predictions = []
#         model.train()
#         total_loss = 0
#         for batch_data, batch_labels in train_loader:
#             batch_data = batch_data.to(device)
#             batch_labels = batch_labels.to(device)


#             outputs = model(batch_data)
#             predicted = torch.argmax(outputs, dim=1)  

#             loss = criterion(outputs, batch_labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             all_labels.extend(batch_labels.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())

#         accuracy = accuracy_score(all_labels, all_predictions)
#         # print(f"Train Accuracy: {accuracy * 100:.2f}%")
#         # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

#         model.eval()
#         all_labels = []
#         all_predictions = []
#         with torch.no_grad():
#             for batch_data, batch_labels in val_loader:
#                 batch_data = batch_data.to(device)
#                 batch_labels = batch_labels.to(device)

#                 outputs = model(batch_data)
#                 predicted = torch.argmax(outputs, dim=1)  

#                 all_labels.extend(batch_labels.cpu().numpy())
#                 all_predictions.extend(predicted.cpu().numpy())

#         all_labels = np.array(all_labels)
#         all_predictions = np.array(all_predictions)

#         accuracy = accuracy_score(all_labels, all_predictions)
#         if accuracy > accuracy_best:
#             accuracy_best = accuracy
#             i = 0
#             torch.save(model, f'/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/spike_classification/train_results/setting_1/{file}/spike_classification_model_{trail}.pth')
#             print(f"Epoch {epoch}: Best model saved with TPR: {accuracy_best:.4f}")
#             #print("_" * 60)
                    
#         else:
#             i += 1
#             if i == 3:
#                 print(f"Training stopped after {epoch+1} epochs with best TPR: {accuracy_best:.4f}")
#                 accuracy_list.append(accuracy_best)
#                 print("_" * 60)
#                 break
            
#     with open(f"/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/spike_classification/eval_results/setting_1/{file}_accuracy_{trail}.pkl", 'wb') as f:
#         pickle.dump(accuracy_list, f)


for file in [#'Neuronexus_32_50_cell_noise_level_10_recording',
             'Neuronexus_32_50_cell_noise_level_30_recording', 
             'Neuronexus_32_50_cell_noise_level_40_recording']:
    print(f'Processing {file}...')
    file = file.split(".")[0]
    recording_raw = se.MEArecRecordingExtractor(file_path=f'/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/data_generation/setting_5_noise_level/recordings/{file}.h5')
    recording_f = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)
    recording_f = spre.common_reference(recording_f, reference="global", operator="median")
    spike_inf = pd.read_csv(f"/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/data_generation/setting_5_noise_level/spike_inf/{file}_spike_inf.csv")

    unique_clusters = np.unique(spike_inf['Neuron'])
    cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    spike_inf['Neuron'] = np.array([cluster_to_index[cluster] for cluster in spike_inf['Neuron']])

    spike_inf_train = spike_inf[spike_inf['time'] < 1000 * 10000]
    spike_inf_val = spike_inf[spike_inf['time'] > 1000 * 10000]

    total_frames = int(1000 * 10000)
    chunk_size = 100000  
    window_size = 31
    half_window = window_size // 2

    all_valid_indices = []
    all_windows = []
    potent_spike_inf = pd.DataFrame()

    for start_frame in range(0, total_frames, chunk_size):
        end_frame = min(start_frame + chunk_size, total_frames)

        spike_inf_temp = spike_inf_train[(spike_inf_train['time'] > start_frame + 15) & (spike_inf_train['time'] < end_frame - 16)]
        
        data_chunk = recording_f.get_traces(
            start_frame=start_frame,
            end_frame=end_frame
        )  # shape: (n_channels, chunk_size)
        
        valid_indices = spike_inf_temp['time'].values
        for idx in valid_indices:
            rel_idx = idx - start_frame
            window = data_chunk.T[:, rel_idx-half_window : rel_idx+half_window+1]
            all_windows.append(window)
        
        all_valid_indices.extend(valid_indices)

        potent_spike_inf = pd.concat((potent_spike_inf, pd.DataFrame((valid_indices, spike_inf_temp['Neuron'])).T), axis=0)

    all_valid_indices = np.array(all_valid_indices)
    all_windows = np.stack(all_windows)

    potent_spike_inf.columns = ['time', 'cluster']

    indices = potent_spike_inf['time'].values

    labels = potent_spike_inf['cluster'].values

    balanced_indices = []
    for cluster in potent_spike_inf['cluster'].unique():
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) > 8000:
            sampled_indices = np.random.choice(cluster_indices, 8000, replace=False)
        else:
            sampled_indices = cluster_indices
        balanced_indices.extend(sampled_indices)

    np.random.shuffle(balanced_indices)

    balanced_data = all_windows[balanced_indices]
    balanced_labels = labels[balanced_indices]

    dataset = CustomDataset(balanced_data, balanced_labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 0)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 0)

    device = 'cuda'
    input_size = balanced_data.shape[1] * balanced_data.shape[2]
    hidden_size1 = 64
    hidden_size2 = 50
    num_classes = potent_spike_inf['cluster'].nunique()

    total_frames = int(1000 * 10000)
    chunk_size = 100000  
    window_size = 31
    half_window = window_size // 2

    all_valid_indices = []
    all_windows = []
    potent_spike_inf_val = pd.DataFrame()

    for start_frame in range(1000 * 10000, 1600 * 10000, chunk_size):
        end_frame = min(start_frame + chunk_size, 1600 * 10000)

        spike_inf_temp = spike_inf_val[(spike_inf_val['time'] > start_frame + 15) & (spike_inf_val['time'] < end_frame - 16)]
        
        data_chunk = recording_f.get_traces(
            start_frame=start_frame,
            end_frame=end_frame
        )  # shape: (n_channels, chunk_size)
        
        valid_indices = spike_inf_temp['time'].values
        for idx in valid_indices:
            rel_idx = idx - start_frame
            window = data_chunk.T[:, rel_idx-half_window : rel_idx+half_window+1]
            all_windows.append(window)
        
        all_valid_indices.extend(valid_indices)

        potent_spike_inf_val = pd.concat((potent_spike_inf_val, pd.DataFrame((valid_indices, spike_inf_temp['Neuron'])).T), axis=0)

    all_valid_indices = np.array(all_valid_indices)
    all_windows = np.stack(all_windows)
    potent_spike_inf_val.columns = ['time', 'cluster']
    indices = potent_spike_inf_val['time'].values
    labels = potent_spike_inf_val['cluster'].values

    dataset = CustomDataset(all_windows, labels)

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    accuracy_list = []
    os.makedirs(f"/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/spike_classification/train_results/setting_5/{file}", exist_ok=True)
    for trail in [1,2,3]:
        model = Spike_Classification_MLP(input_size, hidden_size1, hidden_size2, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.Adam(model.parameters(), lr=0.00001)

        num_epochs = 210
        accuracy_best = 0
        i = 0
        for epoch in range(num_epochs):
            all_labels = []
            all_predictions = []
            model.train()
            total_loss = 0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)


                outputs = model(batch_data)
                predicted = torch.argmax(outputs, dim=1)  

                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_predictions)
            # print(f"Train Accuracy: {accuracy * 100:.2f}%")
            # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

            model.eval()
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_data)
                    predicted = torch.argmax(outputs, dim=1)  

                    all_labels.extend(batch_labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)

            accuracy = accuracy_score(all_labels, all_predictions)
            if accuracy > accuracy_best:
                accuracy_best = accuracy
                i = 0
                torch.save(model, f'/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/spike_classification/train_results/setting_5/{file}/spike_classification_model_{trail}.pth')
                print(f"Epoch {epoch}: Best model saved with TPR: {accuracy_best:.4f}")
                #print("_" * 60)
                        
            else:
                i += 1
                if i == 3:
                    print(f"Training stopped after {epoch+1} epochs with best TPR: {accuracy_best:.4f}")
                    accuracy_list.append(accuracy_best)
                    print("_" * 60)
                    break
                
        with open(f"/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/01_Neuroxenus_32_channels/spike_classification/eval_results/setting_5/{file}_accuracy_{trail}.pkl", 'wb') as f:
            pickle.dump(accuracy_list, f)