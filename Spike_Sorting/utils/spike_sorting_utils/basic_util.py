import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

def calculate_position(row, channel_indices, channel_position):
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

    array2 = np.array((array2['time'], array2['Neuron'])).T
    sorted_indices = np.argsort(array2[:, 0])
    sorted_array2 = array2[sorted_indices]
    
    labels = np.ones(len(array1), dtype=str)
    
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
        return np.zeros(61)
    
    #IDW
    weights = 1 / (np.array(distances) ** power)
    if np.any(distances == 0):
        zero_idx = np.argwhere(distances == 0).flatten()
        return waveforms[:, zero_idx[0]]
    
    weights /= np.sum(weights)
    
    synthesized_waveform = np.zeros(61)
    for t in range(61): 
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

    position_threshold =20
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
        neuron_inf_waveform = candidate['position_waveform']
        corr, _ = pearsonr(row_waveform, neuron_inf_waveform)

        if corr > waveform_threshold and corr > best_corr:
            best_corr = corr
            best_match = candidate['Neuron']

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