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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import random
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = x.reshape(-1, 61 * 30)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
spike_detection_model = torch.load('ANN_result/spike_detection.pth')
file_list = os.listdir("/media/ubuntu/sda/data/mouse6/ns4/natural_image")
file_list.remove('mouse6_021322_natural_image001.ns4')


device = 'cuda'
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

for file in file_list:
    print("Processing file:", file)
    recording_raw = se.read_blackrock(file_path=f'/media/ubuntu/sda/data/mouse6/ns4/natural_image/{file}')

    file = file.split("_")[1]
    recording_recorded = recording_raw.remove_channels(['31', '32', '98'])
    probe_30channel = read_probeinterface('/media/ubuntu/sda/data/probe.json')
    recording_recorded = recording_recorded.set_probegroup(probe_30channel)
    recording_cmr = recording_recorded
    recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
    recording_cmr = spre.common_reference(recording_f, reference="global", operator="median")
    data = recording_cmr.get_traces().astype("float32").T

    threshold_result = detect_local_maxima_in_window(data)
    threshold_result = np.array(threshold_result)
    valid_indices = threshold_result[(threshold_result > 30)]
    valid_indices = valid_indices[valid_indices < data.shape[1] - 31]
    spike_inf = pd.read_csv(f"/home/ubuntu/Documents/jct/project/code/Spike_Sorting/sorting_results/{file}/spike_inf.csv")
    labels = label_array1_based_on_array2(valid_indices, spike_inf['time'].values, threshold=1)
    sampled_data = extract_windows(data, valid_indices, window_size=61)

    test_dataset = CustomDataset(sampled_data, labels)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    spike_detection_model = spike_detection_model.to(device)
    spike_detection_model.eval()

    all_labels = []
    predicted_labels = []
    latent_value = []

    print("Starting eval...")
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_labels = batch_labels.float().unsqueeze(1)
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = spike_detection_model(batch_data)
            predicted = (outputs > 4).float()  

            batch_data = batch_data.reshape(-1, 61 * 30)
            batch_data = spike_detection_model.fc1(batch_data)
            batch_data = spike_detection_model.relu1(batch_data)
            batch_data = spike_detection_model.fc2(batch_data)
            batch_data = spike_detection_model.relu2(batch_data)
            latent_value.append(batch_data.cpu())  
            
            all_labels.extend(batch_labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    predicted_labels = np.array(predicted_labels)
    latent_value = torch.cat(latent_value, dim=0).numpy()


    num_samples = 100000
    indices = random.sample(range(len(latent_value)), num_samples)
    latent_value_subset = latent_value[indices, :]
    all_labels_subset = np.concatenate(all_labels[indices]).astype(int).astype(str)
    all_predictions_subset = np.concatenate(predicted_labels[indices]).astype(int).astype(str)

    print("UMAP visualization...")
    pca = PCA(n_components=20)
    pca_data = pca.fit_transform(latent_value_subset)

    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_data = umap_reducer.fit_transform(pca_data)

    unique_dates = np.unique(all_labels_subset)
    color_map = {date: plt.cm.tab10(i % 10) for i, date in enumerate(unique_dates)} 

    colors = [color_map[date] for date in all_labels_subset]

    print("Saving UMAP plots...")
    with PdfPages(f'/home/ubuntu/Documents/jct/project/code/Spike_Sorting/figure/{file}_umap_spike_detect.pdf') as pdf:
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(umap_data[:, 0], umap_data[:, 1], c=colors, s=0.1, alpha=0.7)
        plt.title('Ground Truth Labels')
        plt.xticks([])
        plt.yticks([])
        pdf.savefig()
        plt.close()

        unique_dates = np.unique(all_predictions_subset)
        color_map = {date: plt.cm.tab10(i % 10) for i, date in enumerate(unique_dates)} 

        colors = [color_map[date] for date in all_predictions_subset]

        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(umap_data[:, 0], umap_data[:, 1], c=colors, s=0.1, alpha=0.7)
        plt.title('Predected Labels')
        plt.xticks([])
        plt.yticks([])
        pdf.savefig()
        plt.close()


        #all_labels = np.concatenate(all_labels).astype(int).astype(str)
        unique_predicted = np.unique(predicted_labels)
        unique_all = np.unique(all_labels)

        percentages = []
        for pred_class in unique_predicted:
            indices = np.where(predicted_labels == pred_class)[0]
            corresponding_labels = all_labels[indices]
            counts = np.array([np.sum(corresponding_labels == label) for label in unique_all])
            percentages.append(counts / counts.sum() * 100)

        percentages = np.array(percentages).T  

        x = np.arange(len(unique_predicted))  
        width = 0.6  

        plt.figure(figsize=(5, 3))
        bottom = np.zeros(len(unique_predicted))  
        for i, label in enumerate(unique_all):
            plt.bar(x, percentages[i], width, bottom=bottom, label=f'All Label {label}')
            bottom += percentages[i]  

        plt.xlabel('Predicted Labels')
        plt.ylabel('Percentage (%)')
        plt.title('Stacked Bar Chart of All Labels for Each Predicted Label')
        plt.xticks(x, unique_predicted)
        pdf.savefig()  
        plt.close()

    del recording_raw, recording_recorded, recording_f, recording_cmr, data, threshold_result, valid_indices, spike_inf, labels, sampled_data, test_dataset, test_loader, all_labels, predicted_labels, latent_value