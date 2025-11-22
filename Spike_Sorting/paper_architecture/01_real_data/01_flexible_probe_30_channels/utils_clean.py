"""
AutoSort 训练工具函数
包含：阈值检测、数据准备、模型定义和训练函数
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import scipy.signal

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, f1_score


# ==================== 1. 阈值检测 ====================

def detect_spike(
    trace0_car,
    thr_min=3,
    thr_max=30,
    distance=3,
    ch_max_simul_firing=5,
    wlen=5,
    prominence=10,
):
    """
    AutoSort 的阈值检测函数（与 detection.py 完全一致）
    
    参数:
        trace0_car: numpy数组，形状为 (n_timepoints, n_channels)
        thr_min: 最小阈值倍数（相对于噪声标准差），默认3
        thr_max: 最大阈值倍数（用于过滤异常值），默认30
        distance: 峰值之间的最小距离（采样点），默认3
        ch_max_simul_firing: 同时放电的最大通道数，默认5
        wlen: 峰值检测的窗口长度，默认5
        prominence: 峰值的最小突出度，默认10
    
    返回:
        spikes: 二进制矩阵 (n_timepoints, n_channels)，1表示检测到spike
    """
    noise_std_detect = np.median(abs(trace0_car) / 0.6745, axis=0)
    thr = thr_min * noise_std_detect
    thrmax = thr_max * noise_std_detect

    spikes = np.zeros(trace0_car.shape)
    if trace0_car.ndim > 1:
        for i in range(noise_std_detect.shape[0]):
            peaks, props = scipy.signal.find_peaks(
                -trace0_car[:, i],
                height=thr[i],
                distance=distance,
                wlen=wlen,
                prominence=prominence,
            )
            prominences = scipy.signal.peak_prominences(
                -trace0_car[:, i], peaks, wlen=7
            )[0]
            peaks = peaks[props["peak_heights"] > 10]
            prominences = prominences[props["peak_heights"] > 10]
            peaks = peaks[(prominences > 15)]

            spikes[peaks, i] = 1

        # larger value no more than thrmax
        points = trace0_car.shape[0]
        spike_coord = np.argwhere(spikes == 1)
        for i in range(spike_coord.shape[0]):
            near_start = spike_coord[i, 0] - 5
            near_end = spike_coord[i, 0] + 5
            if near_start < 0:
                near_start = 0
            if near_end >= points:
                near_end = points - 1
            if np.any(np.max(trace0_car[near_start:near_end, :], axis=0) >= thrmax):
                spikes[spike_coord[i, 0], spike_coord[i, 1]] = 0

        # no simultanous firing!!!!
        thres_cross = ch_max_simul_firing
        spikes[np.sum(spikes, axis=1) > thres_cross, :] = 0
    return spikes


def map_gt_annotation(detect_array, gt_array):
    """
    AutoSort 的 GT 映射函数（向量化优化版本，逻辑与 detection.py 完全一致）
    
    参数:
        detect_array: numpy数组，形状为 (n_detected, 2)，每行为 [时间点, 通道ID]
        gt_array: numpy数组，形状为 (n_gt, 2)，每行为 [时间点, 通道ID]
    
    返回:
        gt_label_array1: numpy数组，形状为 (n_detected,)，值为对应的 GT 索引或 -1（未匹配）
    """
    n_detected = detect_array.shape[0]
    gt_label_array1 = np.full(n_detected, -1, dtype=np.int64)
    
    if n_detected == 0 or gt_array.shape[0] == 0:
        return gt_label_array1
    
    # 提取检测到的时间和通道
    detect_times = detect_array[:, 0].astype(np.int64)
    detect_channels = detect_array[:, 1].astype(np.int64)
    
    # 提取 GT 的时间和通道
    gt_times = gt_array[:, 0].astype(np.int64)
    gt_channels = gt_array[:, 1].astype(np.int64)
    
    # 使用字典来加速查找：key = (时间, 通道), value = GT索引列表
    gt_dict = defaultdict(list)
    for idx, (t, c) in enumerate(zip(gt_times, gt_channels)):
        gt_dict[(t, c)].append(idx)
    
    # 为每个检测到的 spike 尝试匹配三种时间偏移：0, -1, +1（按优先级）
    time_offsets = [0, -1, 1]
    
    # 向量化匹配：对每个时间偏移，批量处理所有未匹配的检测 spike
    for offset in time_offsets:
        # 找到还未匹配的检测 spike
        unmatched_mask = gt_label_array1 == -1
        if not np.any(unmatched_mask):
            break
        
        # 计算偏移后的时间（只对未匹配的）
        unmatched_indices = np.where(unmatched_mask)[0]
        shifted_times = detect_times[unmatched_indices] + offset
        unmatched_channels = detect_channels[unmatched_indices]
        
        # 向量化查找：使用字典快速匹配（O(1)查找）
        keys = [(shifted_times[i], unmatched_channels[i]) for i in range(len(unmatched_indices))]
        
        # 批量查找匹配（避免逐个循环查找）
        for i, key in enumerate(keys):
            if key in gt_dict and len(gt_dict[key]) > 0:
                # 找到匹配，使用第一个匹配的 GT 索引
                gt_idx = gt_dict[key][0]
                detect_idx = unmatched_indices[i]
                gt_label_array1[detect_idx] = gt_idx
                # 从字典中移除已匹配的项（避免重复匹配）
                gt_dict[key].pop(0)
                if len(gt_dict[key]) == 0:
                    del gt_dict[key]
    
    return gt_label_array1


# ==================== 2. 训练数据准备 ====================

def extract_waveforms(trace0_car, X_spiketrain_time, left_sample=10, right_sample=20):
    """
    提取波形窗口（按照 AutoSort 的方式）
    
    参数:
        trace0_car: numpy数组，形状为 (n_timepoints, n_channels)
        X_spiketrain_time: numpy数组，形状为 (n_spikes,)，spike时间点
        left_sample: spike前的采样点数，默认10
        right_sample: spike后的采样点数，默认20
    
    返回:
        waveform: numpy数组，形状为 (n_spikes, n_channels, window_length)
    """
    # 过滤边界附近的 spike（确保可以提取完整的窗口）
    valid_mask = X_spiketrain_time < trace0_car.shape[0] - (left_sample + right_sample)
    X_spiketrain_time = X_spiketrain_time[valid_mask]
    
    # 按照 AutoSort 的方式提取窗口
    for time_range in tqdm(np.arange(-left_sample, right_sample), desc="提取波形"):
        if time_range == -left_sample:
            # 第一个时间点，初始化 waveform
            waveform = trace0_car[X_spiketrain_time + time_range, :]
        else:
            # 后续时间点，使用 dstack 堆叠
            waveform = np.dstack(
                (waveform, trace0_car[X_spiketrain_time + time_range, :])
            )
    
    # waveform 形状: (n_spikes, n_channels, window_length)
    return waveform, valid_mask


def prepare_training_data(
    recording_f,
    spike_inf,
    neuron_inf,
    save_dir,
    duration_seconds=200,
    thr_min=3.5,
    thr_max=30,
    distance=3,
    ch_max_simul_firing=5,
    wlen=5,
    prominence=10,
    left_sample=10,
    right_sample=20,
):
    """
    准备训练数据（完整流程：检测 -> 匹配 -> 提取波形 -> 保存）
    
    参数:
        recording_f: 预处理后的 recording 对象
        spike_inf: DataFrame，包含 GT spike 信息
        neuron_inf: DataFrame，包含 neuron 信息
        save_dir: 保存目录路径
        duration_seconds: 处理时长（秒），默认200
        thr_min, thr_max, distance, ch_max_simul_firing, wlen, prominence: 检测参数
        left_sample, right_sample: 波形窗口参数
    
    返回:
        train_data_dir: 训练数据保存目录
    """
    print("### 1. 阈值检测")
    
    # 获取recording的采样率和通道数
    sampling_rate = recording_f.get_sampling_frequency()
    n_channels = recording_f.get_num_channels()
    print(f"采样率: {sampling_rate} Hz, 通道数: {n_channels}")
    
    # 计算对应的采样点数
    max_frames = int(duration_seconds * sampling_rate)
    total_frames = recording_f.get_num_frames()
    actual_frames = min(max_frames, total_frames)
    
    print(f"Recording总长度: {total_frames} 采样点 ({total_frames/sampling_rate:.2f} 秒)")
    print(f"将处理前 {actual_frames} 采样点 ({actual_frames/sampling_rate:.2f} 秒)")
    
    # 读取数据
    trace0_car = recording_f.get_traces(start_frame=0, end_frame=actual_frames).astype(np.float32)
    print(f"数据形状: {trace0_car.shape}")
    
    # 使用 AutoSort 的 detect_spike 函数
    spikes = detect_spike(
        trace0_car,
        thr_min=thr_min,
        thr_max=thr_max,
        distance=distance,
        ch_max_simul_firing=ch_max_simul_firing,
        wlen=wlen,
        prominence=prominence,
    )
    
    # 构建 detect_array
    print("构建 detect_array...")
    all_spike_train = []
    spike_loc = []
    for channel_num in range(trace0_car.shape[1]):
        spiketrain_loc = np.where(spikes[:, channel_num])[0]
        all_spike_train += list(spiketrain_loc)
        spike_loc += [channel_num] * len(spiketrain_loc)
    
    X_spiketrain_time = np.array(all_spike_train)
    Y_spiketrain_id_final = np.array(spike_loc)
    detect_array = np.array([X_spiketrain_time, Y_spiketrain_id_final]).T
    
    print(f"检测到的 spike 数量: {len(detect_array)}")
    
    print("\n### 2. 加载 Ground Truth 并匹配")
    
    # 过滤spike_inf，只保留指定时长的数据
    spike_inf_filtered = spike_inf[spike_inf['time'] < max_frames].copy()
    
    # 构建 gt_array
    print("构建 gt_array...")
    spike_train_all = []
    y_unit_id = []
    gt_ch = []
    
    for neuron_idx in range(len(neuron_inf)):
        neuron_name = neuron_inf['Neuron'].iloc[neuron_idx]
        neuron_channel_id = neuron_inf['tract_channel'].iloc[neuron_idx]
        
        neuron_spikes = spike_inf_filtered[spike_inf_filtered['neuron'] == neuron_name]
        if len(neuron_spikes) > 0:
            spike_times = neuron_spikes['time'].values
            spike_train_all += list(spike_times)
            y_unit_id += [neuron_name] * len(spike_times)
            gt_ch += [neuron_channel_id] * len(spike_times)
    
    gt_array = np.array([spike_train_all, gt_ch]).T
    print(f"GT spike 数量: {len(gt_array)}")
    
    # 使用 AutoSort 的 map_gt_annotation 函数
    gt_label_array1 = map_gt_annotation(detect_array, gt_array)
    
    # 计算检测率
    detection_rate = np.where(gt_label_array1 > -1)[0].shape[0] / gt_array.shape[0]
    print(f"---spike detection rate: {detection_rate:.4f}")
    
    # 构建 Y_spiketrain_id
    Y_spiketrain_id = np.full((detect_array.shape[0],), None, dtype=object)
    matched_indices = np.where(gt_label_array1 > -1)[0]
    if len(matched_indices) > 0:
        y_unit_id_array = np.array(y_unit_id, dtype=object)
        Y_spiketrain_id[matched_indices] = y_unit_id_array[
            gt_label_array1[matched_indices].astype("int")
        ]
    
    print(f"匹配到的 spike 数量: {len(matched_indices)}")
    print(f"未匹配的 spike 数量: {len(detect_array) - len(matched_indices)}")
    
    print("\n### 3. 提取波形")
    
    # 提取波形
    waveform, valid_mask = extract_waveforms(
        trace0_car, X_spiketrain_time, left_sample, right_sample
    )
    
    # 应用 valid_mask 过滤
    X_spiketrain_time = X_spiketrain_time[valid_mask]
    Y_spiketrain_id = Y_spiketrain_id[valid_mask]
    Y_spiketrain_id_final = Y_spiketrain_id_final[valid_mask]
    
    print(f"波形提取完成！")
    print(f"waveform 形状: {waveform.shape}")
    
    print("\n### 4. 保存训练数据")
    
    # 创建保存目录
    train_data_dir = Path(save_dir) / "train_data"
    train_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存目录: {train_data_dir}")
    
    # 准备数据
    X_waveform = waveform
    
    # 转换 Y_spike_id
    unique_neurons = np.unique([x for x in Y_spiketrain_id if x is not None])
    neuron_to_id = {neuron: idx for idx, neuron in enumerate(unique_neurons)}
    neuron_to_id[None] = -1
    
    Y_spike_id = np.array([neuron_to_id.get(x, -1) for x in Y_spiketrain_id])
    Y_spike_id_noise = Y_spiketrain_id_final
    
    # 保存 neuron 名称到 ID 的映射（用于后续评估时的 neuron 匹配）
    neuron_mapping = {
        'neuron_to_id': neuron_to_id,
        'id_to_neuron': {idx: neuron for neuron, idx in neuron_to_id.items() if neuron is not None},
        'unique_neurons': list(unique_neurons)
    }
    with open(train_data_dir / "neuron_mapping.pkl", "wb") as f:
        pickle.dump(neuron_mapping, f)
    print(f"  ✓ neuron_mapping.pkl 已保存")
    
    # 保存数据
    print("保存数据...")
    with open(train_data_dir / "X_waveform.pkl", "wb") as f:
        pickle.dump(X_waveform, f)
    print(f"  ✓ X_waveform.pkl 已保存")
    
    with open(train_data_dir / "Y_spike_id.pkl", "wb") as f:
        pickle.dump(Y_spike_id, f)
    print(f"  ✓ Y_spike_id.pkl 已保存")
    
    with open(train_data_dir / "Y_spike_id_noise.pkl", "wb") as f:
        pickle.dump(Y_spike_id_noise, f)
    print(f"  ✓ Y_spike_id_noise.pkl 已保存")
    
    with open(train_data_dir / "X_spiketrain_time.pkl", "wb") as f:
        pickle.dump(X_spiketrain_time, f)
    print(f"  ✓ X_spiketrain_time.pkl 已保存")
    
    print(f"\n所有数据已保存到: {train_data_dir}")
    print(f"数据统计:")
    print(f"  - 总 spike 数量: {len(X_waveform)}")
    print(f"  - 通道数: {X_waveform.shape[1]}")
    print(f"  - 窗口长度: {X_waveform.shape[2]}")
    print(f"  - 唯一单元数: {len(unique_neurons)}")
    print(f"  - 噪声 spike 数量: {np.sum(Y_spike_id == -1)}")
    print(f"  - 有效 spike 数量: {np.sum(Y_spike_id != -1)}")
    
    return train_data_dir


# ==================== 3. 模型定义 ====================

class SimpleClassifier(nn.Module):
    """
    简化的分类器（与 AutoSort 的 clssimp 相同）
    """
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size=(input_dim))
        self.way1 = nn.Sequential(
            nn.Linear(input_dim, 1000, bias=True),
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
        x = self.pool(x[None, :])
        x = x.reshape(x.size(1), -1)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        logits = self.cls(x)
        return logits

    def intermediate_forward(self, x):
        x = self.pool(x[None, :])
        x = x.reshape(x.size(1), -1)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        return x


class SimpleWaveformLoader(data.Dataset):
    """
    简化的 waveform loader，不包含位置信息
    只使用 multi-waveform 和 single-waveform
    """
    def __init__(self, root, shank_channel, Keep_id=None):
        # 加载数据
        with open(root + "X_waveform.pkl", "rb") as f:
            datafile = pickle.load(f)
        try:
            with open(root + "Y_spike_id.pkl", "rb") as f:
                GT = pickle.load(f)
        except FileNotFoundError:
            GT = np.zeros(datafile.shape[0]) - 1
        with open(root + "Y_spike_id_noise.pkl", "rb") as f:
            channel_id = np.array(pickle.load(f))
        
        # 确定要保留的单元ID
        if Keep_id is None:
            Keep_id = np.unique(GT)
            Keep_id = list(Keep_id[Keep_id != -1])
            self.keep_id = Keep_id
        else:
            self.keep_id = Keep_id
        
        # 创建噪声/非噪声标签
        mask = ~np.isin(GT, Keep_id)
        GT = np.array(GT)
        
        GT_binary = np.zeros((GT.shape[0], 2))
        GT_binary[list(mask), 0] = 1  # 噪声
        GT_binary[~mask, 1] = 1       # 非噪声
        
        self.GT_unique = Keep_id + [-1]
        self.GT_binary = GT_binary
        
        # 提取 single waveform（从最大幅度通道）
        self.Img_single = datafile[np.arange(datafile.shape[0]), np.array(channel_id).astype('int'), :]
        
        self.GT_LIST = GT
        
        # 创建单元分类标签（one-hot）
        GT_array = np.zeros((len(GT), len(Keep_id)))
        for idx, unique_id in enumerate(Keep_id):
            rmv_list = np.where(np.array(GT) == unique_id)[0]
            GT_array[rmv_list, idx] = 1
        self.GT = GT_array
        
        self.Img = datafile  # 多通道波形
        
        # 计算类别权重（用于处理不平衡数据）
        self.pos_weight_noise = torch.tensor([
            -np.sum(self.GT_binary[:,0]-1)/np.sum(self.GT_binary[:,0]),
            -np.sum(self.GT_binary[:,1]-1)/np.sum(self.GT_binary[:,1])
        ])
        self.pos_weight_label = torch.tensor([
            -(np.sum(self.GT[:,i]-1)+sum(np.sum(GT_array,axis=1)==0))/np.sum(self.GT[:,i]) 
            for i in range(self.GT.shape[1])
        ])
        
        self.n_classes = len(set(self.GT_unique))
        
        print(f"Dataset 加载完成:")
        print(f"  - 总样本数: {len(self.GT)}")
        print(f"  - 通道数: {self.Img.shape[1]}")
        print(f"  - 窗口长度: {self.Img.shape[2]}")
        print(f"  - 唯一单元数: {len(Keep_id)}")
        print(f"  - 噪声样本数: {np.sum(self.GT_binary[:, 0])}")
        print(f"  - 非噪声样本数: {np.sum(self.GT_binary[:, 1])}")
    
    def __len__(self):
        return len(self.GT)
    
    def __getitem__(self, index):
        # 返回: 多通道波形, 单元分类标签, 噪声/非噪声标签, 单通道波形
        return (
            self.Img[index, ...],      # (n_channels, window_length)
            self.GT[index, ...],       # (n_units,) one-hot
            self.GT_binary[index, ...], # (2,) [noise, spike]
            self.Img_single[index, ...] # (window_length,)
        )


class SimpleAutoSort:
    """
    简化的 AutoSort 模型（不包含位置信息）
    输入: multi-waveform + single-waveform
    与原始 AutoSort 完全一致，只是去掉了位置信息
    """
    def __init__(self, ch_num, samplepoints, device, set_shank_id, save_dir, 
                 pos_weight_noise=None, pos_weight_label=None):
        # 输入维度: (ch_num + 1) * samplepoints（不包含位置信息）
        input_dim = (ch_num + 1) * samplepoints
        
        self.clsfier_noise = SimpleClassifier(input_dim, 2).to(device)
        self.clsfier_label = SimpleClassifier(input_dim, len(set_shank_id)).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.clsfier_noise.parameters()},
            {'params': self.clsfier_label.parameters()},
        ], lr=1e-4)
        
        self.criterion = nn.MSELoss()  # 与原始一致（虽然不使用）
        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_noise)
        self.bceloss_label = nn.BCEWithLogitsLoss(pos_weight=pos_weight_label)
        
        self.save_model_path_2 = save_dir + 'multitask_single_wave_clsfier_noise_clsfier.pth'
        self.save_model_path_3 = save_dir + 'multitask_single_wave_clsfier_label_clsfier.pth'
        
        self.set_shank_id = set_shank_id
        self.device = device
    
    def save_model(self):
        torch.save(self.clsfier_noise.state_dict(), self.save_model_path_2)
        torch.save(self.clsfier_label.state_dict(), self.save_model_path_3)
    
    def load_model(self):
        self.clsfier_noise.load_state_dict(torch.load(self.save_model_path_2))
        self.clsfier_label.load_state_dict(torch.load(self.save_model_path_3))
    
    def train(self):
        self.clsfier_noise.train()
        self.clsfier_label.train()
    
    def eval(self):
        self.clsfier_noise.eval()
        self.clsfier_label.eval()
    
    def iter_model(self, batch_features, classify_labels, labels, single_waveform):
        """
        训练迭代
        """
        self.optimizer.zero_grad()
        
        # 拼接 multi-waveform 和 single-waveform
        codes = torch.cat((batch_features, single_waveform), axis=1)
        
        # 噪声分类
        cls_output = self.clsfier_noise(codes.float())
        
        # 单元分类（只对非噪声样本）
        test = labels[:, 1] == 1
        if sum(test) > 1:
            cls_label_output = self.clsfier_label(codes.float()[test, :])
            train_classification_loss = 1000 * self.bceloss_label(
                cls_label_output, 
                classify_labels[test, :len(self.set_shank_id)]
            )
        else:
            train_classification_loss = torch.tensor(0)
        
        train_detection_loss = 1000 * self.bceloss(cls_output, labels)
        
        train_loss = train_detection_loss + train_classification_loss
        train_loss.backward()
        self.optimizer.step()
        
        return train_detection_loss.item(), train_classification_loss.item(), test
    
    def iter_model_eval(self, batch_features, classify_labels, labels, single_waveform):
        """
        评估迭代
        """
        codes = torch.cat((batch_features, single_waveform), axis=1)
        
        cls_output = self.clsfier_noise(codes.float())
        gt = torch.argmax(labels, axis=1)
        pred = torch.argmax(cls_output, axis=1)
        
        test = labels[:, 1] == 1
        if sum(test) > 1:
            cls_label_output = self.clsfier_label(codes.float()[test, :])
            pred_class = torch.argmax(cls_label_output, axis=1)
            gt_label_class = torch.argmax(classify_labels[test, :len(self.set_shank_id)], axis=1)
            train_classification_loss = 1000 * self.bceloss_label(
                cls_label_output, 
                classify_labels[test, :len(self.set_shank_id)]
            )
        else:
            train_classification_loss = torch.tensor(0)
            gt_label_class = torch.tensor([])
            pred_class = torch.tensor([])
        
        train_detection_loss = 1000 * self.bceloss(cls_output, labels)
        train_loss = train_detection_loss + train_classification_loss
        
        return train_detection_loss.item(), train_classification_loss.item(), gt, pred, gt_label_class, pred_class


# ==================== 4. 训练函数 ====================

def train_autosort_model(
    train_data_dir,
    model_save_dir,
    n_channels,
    left_sample=10,
    right_sample=20,
    epochs=20,
    batch_size=512,
    device=None,
):
    """
    训练 AutoSort 模型
    
    参数:
        train_data_dir: 训练数据目录
        model_save_dir: 模型保存目录
        n_channels: 通道数
        left_sample, right_sample: 窗口参数
        epochs: 训练轮数
        batch_size: batch大小
        device: 设备（如果为None，自动选择）
    
    返回:
        autosort_model: 训练好的模型
        training_log: 训练日志
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型保存目录
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置参数
    samplepoints = left_sample + right_sample
    
    # 创建 dataset
    print("创建 dataset...")
    dataset = SimpleWaveformLoader(
        root=str(train_data_dir) + '/',
        shank_channel=np.arange(n_channels),
        Keep_id=None
    )
    
    set_shank_id = dataset.keep_id
    print(f"模型参数:")
    print(f"  - 通道数: {n_channels}")
    print(f"  - 窗口长度: {samplepoints}")
    print(f"  - 单元数量: {len(set_shank_id)}")
    print(f"  - 输入维度: {(n_channels + 1) * samplepoints}")
    
    # 保存单元ID列表（用于评估时使用）
    keep_id_path = model_save_dir + 'keep_id.pkl'
    with open(keep_id_path, 'wb') as f:
        pickle.dump(set_shank_id, f)
    print(f"单元ID列表已保存到: {keep_id_path}")
    
    # 创建模型
    autosort_model = SimpleAutoSort(
        ch_num=n_channels,
        samplepoints=samplepoints,
        device=device,
        set_shank_id=set_shank_id,
        save_dir=model_save_dir,
        pos_weight_noise=dataset.pos_weight_noise.to(device),
        pos_weight_label=dataset.pos_weight_label.to(device)
    )
    
    # 划分训练集和验证集
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n数据集划分:")
    print(f"  - 训练集: {train_size} 样本")
    print(f"  - 验证集: {val_size} 样本")
    
    # 训练参数
    min_valid_loss = np.inf
    
    # 检查模型是否已存在
    import os
    if os.path.exists(autosort_model.save_model_path_2):
        autosort_model.load_model()
        print("已加载现有模型")
        return autosort_model, None
    
    # 训练日志
    training_log = {'epoch': [],
                    'validation_acc_noise':[],
                    'validation_acc_label':[]}
    
    print(f"\n开始训练（共 {epochs} 个 epoch）...")
    
    for epoch in range(epochs):
        training_log['epoch'].append(epoch + 1)
        print("epoch : {}/{}".format(epoch + 1, epochs))
        
        # 训练阶段
        detection_loss = 0
        classification_loss = 0
        autosort_model.train()
        autosort_model.bceloss.pos_weight = autosort_model.bceloss.pos_weight.to(device)
        autosort_model.bceloss_label.pos_weight = autosort_model.bceloss_label.pos_weight.to(device)
        
        for batch_features, classify_labels, labels, single_waveform in tqdm(train_loader, desc="训练"):
            classify_labels = classify_labels.to(device)
            batch_features = batch_features.view(-1, samplepoints * n_channels).to(device)
            labels = labels.to(device)
            single_waveform = single_waveform.to(device)
            
            train_detection_loss, train_classification_loss, test = autosort_model.iter_model(
                batch_features, classify_labels, labels, single_waveform
            )
            
            detection_loss += train_detection_loss
            if sum(test) > 0:
                classification_loss += train_classification_loss
        
        detection_loss = detection_loss / len(train_loader)
        classification_loss = classification_loss / len(train_loader)
        print("epoch : {}/{}, detection loss = {:.6f}, classification loss = {:.6f}".format(
            epoch + 1, epochs, detection_loss, classification_loss))
        
        # 验证阶段
        valid_detection_loss = 0.0
        valid_classification_loss = 0.0
        
        gt_all = []
        pred_all = []
        gt_class_all = []
        pred_class_all = []
        autosort_model.eval()
        
        with torch.no_grad():
            for batch_features, classify_labels, labels, single_waveform in tqdm(val_loader, desc="验证"):
                classify_labels = classify_labels.to(device)
                batch_features = batch_features.view(-1, samplepoints * n_channels).to(device)
                labels = labels.to(device)
                single_waveform = single_waveform.to(device)
                
                valid_detection_loss_batch, valid_classification_loss_batch, gt, pred, gt_label_class, pred_class = autosort_model.iter_model_eval(
                    batch_features, classify_labels, labels, single_waveform
                )
                
                valid_detection_loss += valid_detection_loss_batch
                valid_classification_loss += valid_classification_loss_batch
                
                gt_all.append(gt.detach().cpu().numpy())
                pred_all.append(pred.detach().cpu().numpy())
                pred_class_all.append(pred_class.detach().cpu().numpy())
                gt_class_all.append(gt_label_class.detach().cpu().numpy())
        
        gt_all = np.concatenate(gt_all, axis=0)
        pred_all = np.concatenate(pred_all, axis=0)
        
        # 过滤空数组
        gt_class_all = [x for x in gt_class_all if len(x) > 0]
        pred_class_all = [x for x in pred_class_all if len(x) > 0]
        if len(gt_class_all) > 0:
            gt_class_all = np.concatenate(gt_class_all, axis=0)
            pred_class_all = np.concatenate(pred_class_all, axis=0)
        else:
            gt_class_all = np.array([])
            pred_class_all = np.array([])
        
        valid_detection_loss = valid_detection_loss / len(val_loader)
        valid_classification_loss = valid_classification_loss / len(val_loader)
        valid_loss = valid_detection_loss + valid_classification_loss
        print("epoch : {}/{}, val detection loss = {:.6f}, classification loss = {:.6f}".format(
            epoch + 1, epochs, valid_detection_loss, valid_classification_loss))
        
        training_log['validation_acc_noise'].append(accuracy_score(gt_all, pred_all))
        if len(gt_class_all) > 0:
            training_log['validation_acc_label'].append(f1_score(gt_class_all, pred_class_all, average='micro'))
        else:
            training_log['validation_acc_label'].append(0.0)
        
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            autosort_model.save_model()
    
    # 保存训练日志
    pd.DataFrame(training_log).to_csv(model_save_dir + 'training_log.csv')
    print(f"\n训练完成！训练日志已保存到: {model_save_dir}training_log.csv")
    
    return autosort_model, training_log


# ==================== 5. Neuron 匹配函数 ====================

def match_neurons(
    train_neuron_inf,
    eval_neuron_inf,
    train_data_dir=None,
    eval_data_dir=None,
    position_threshold=10,
    waveform_similarity_threshold=0.95,
):
    """
    匹配训练数据和评估数据的 neuron
    
    参数:
        train_neuron_inf: 训练数据的 neuron_inf DataFrame（需包含 position_1, position_2, position_waveform 列）
        eval_neuron_inf: 评估数据的 neuron_inf DataFrame（需包含 position_1, position_2, position_waveform 列）
        train_data_dir: 训练数据目录（已废弃，保留以兼容旧代码）
        eval_data_dir: 评估数据目录（已废弃，保留以兼容旧代码）
        position_threshold: 位置距离阈值（欧氏距离，单位：微米），默认10
        waveform_similarity_threshold: 波形相似性阈值（Pearson相关系数），默认0.95
    
    返回:
        eval_neuron_inf_matched: 添加了 neuron_match 列的评估 neuron_inf
    """
    from scipy.stats import pearsonr
    
    print("=" * 50)
    print("Neuron 匹配")
    print("=" * 50)
    
    # 复制评估 neuron_inf
    eval_neuron_inf_matched = eval_neuron_inf.copy()
    eval_neuron_inf_matched['neuron_match'] = 'unmatch'
    
    # 检查必需的列
    required_cols = ['position_1', 'position_2', 'position_waveform']
    for col in required_cols:
        if col not in train_neuron_inf.columns:
            raise ValueError(f"train_neuron_inf 缺少必需的列: {col}")
        if col not in eval_neuron_inf.columns:
            raise ValueError(f"eval_neuron_inf 缺少必需的列: {col}")
    
    # 获取所有训练 neuron 名称
    train_unique_neurons = train_neuron_inf['Neuron'].unique()
    
    # 匹配 neuron
    print("进行 neuron 匹配...")
    matched_count = 0
    for eval_idx, eval_row in eval_neuron_inf_matched.iterrows():
        eval_neuron = eval_row['Neuron']
        
        # 获取评估 neuron 的位置坐标和波形
        eval_pos = np.array([eval_row['position_1'], eval_row['position_2']])
        eval_wf = eval_row['position_waveform']
        
        # 确保波形是 numpy 数组
        if not isinstance(eval_wf, np.ndarray):
            eval_wf = np.array(eval_wf)
        
        if len(eval_wf) == 0:
            continue
        
        best_match = None
        best_similarity = 0
        
        # 遍历所有训练 neuron
        for train_neuron in train_unique_neurons:
            if train_neuron is None:
                continue
            train_rows = train_neuron_inf[train_neuron_inf['Neuron'] == train_neuron]
            if len(train_rows) == 0:
                continue
            train_row = train_rows.iloc[0]
            
            # 获取训练 neuron 的位置坐标和波形
            train_pos = np.array([train_row['position_1'], train_row['position_2']])
            train_wf = train_row['position_waveform']
            
            # 确保波形是 numpy 数组
            if not isinstance(train_wf, np.ndarray):
                train_wf = np.array(train_wf)
            
            if len(train_wf) == 0:
                continue
            
            # 计算位置距离（欧氏距离）
            position_distance = np.linalg.norm(eval_pos - train_pos)
            
            # 计算波形相似性（Pearson 相关系数）
            if len(eval_wf) == len(train_wf):
                similarity, _ = pearsonr(eval_wf, train_wf)
                if np.isnan(similarity):
                    similarity = 0
            else:
                similarity = 0
            
            # 检查是否匹配
            if (position_distance < position_threshold and 
                similarity > waveform_similarity_threshold and
                similarity > best_similarity):
                best_match = train_neuron
                best_similarity = similarity
        
        # 设置匹配结果
        if best_match is not None:
            eval_neuron_inf_matched.loc[eval_idx, 'neuron_match'] = best_match
            matched_count += 1
            train_row_matched = train_neuron_inf[train_neuron_inf['Neuron'] == best_match].iloc[0]
            train_pos_matched = np.array([train_row_matched['position_1'], train_row_matched['position_2']])
            position_distance_final = np.linalg.norm(eval_pos - train_pos_matched)
            print(f"  {eval_neuron} -> {best_match} (相似性: {best_similarity:.4f}, 位置距离: {position_distance_final:.2f})")
    
    print(f"\n匹配完成:")
    print(f"  - 总评估 neuron 数: {len(eval_neuron_inf_matched)}")
    print(f"  - 匹配成功: {matched_count}")
    print(f"  - 未匹配: {len(eval_neuron_inf_matched) - matched_count}")
    
    return eval_neuron_inf_matched


# ==================== 6. 评估函数 ====================

def evaluate_autosort_model(
    train_data_dir,
    model_save_dir,
    n_channels,
    left_sample=10,
    right_sample=20,
    batch_size=512,
    device=None,
    save_results=True,
    results_save_dir=None,
    eval_neuron_inf_matched=None,
    eval_data_dir=None,
):
    """
    评估 AutoSort 模型
    
    参数:
        train_data_dir: 训练数据目录（用于评估）
        model_save_dir: 模型保存目录
        n_channels: 通道数
        left_sample, right_sample: 窗口参数
        batch_size: batch大小
        device: 设备（如果为None，自动选择）
        save_results: 是否保存结果
        results_save_dir: 结果保存目录（如果为None，使用model_save_dir）
        eval_neuron_inf_matched: 评估数据的 neuron_inf（包含 neuron_match 列），如果提供则计算两套结果
        eval_data_dir: 评估数据目录（如果提供 eval_neuron_inf_matched，需要提供此参数）
    
    返回:
        results: 评估结果字典，包含：
            - noise_accuracy: 噪声分类准确率（原始）
            - unit_f1_score: 单元分类F1分数（原始）
            - noise_accuracy_adjusted: 噪声分类准确率（调整后，unmatch视为noise）
            - unit_f1_score_adjusted: 单元分类F1分数（调整后）
            - noise_predictions: 噪声预测结果
            - unit_predictions: 单元预测结果
            - gt_noise: 真实噪声标签（原始）
            - gt_units: 真实单元标签（原始）
            - gt_noise_adjusted: 真实噪声标签（调整后）
            - gt_units_adjusted: 真实单元标签（调整后）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置参数
    samplepoints = left_sample + right_sample
    
    # 加载训练时的单元ID列表（必须使用训练时的单元ID，确保模型维度匹配）
    keep_id_path = model_save_dir + 'keep_id.pkl'
    import os
    if os.path.exists(keep_id_path):
        print(f"从文件加载训练时的单元ID列表: {keep_id_path}")
        with open(keep_id_path, 'rb') as f:
            train_keep_id = pickle.load(f)
        print(f"训练时的单元数量: {len(train_keep_id)}")
    else:
        raise FileNotFoundError(
            f"单元ID列表文件不存在: {keep_id_path}\n"
            "请确保已经运行过训练流程，或者手动创建该文件。"
        )
    
    # 创建 dataset（使用训练时的单元ID列表）
    print("创建 dataset...")
    dataset = SimpleWaveformLoader(
        root=str(train_data_dir) + '/',
        shank_channel=np.arange(n_channels),
        Keep_id=train_keep_id  # 使用训练时的单元ID列表
    )
    
    set_shank_id = dataset.keep_id
    print(f"模型参数:")
    print(f"  - 通道数: {n_channels}")
    print(f"  - 窗口长度: {samplepoints}")
    print(f"  - 单元数量: {len(set_shank_id)}")
    print(f"  - 使用训练时的单元ID列表: {set_shank_id == train_keep_id}")
    
    # 创建模型
    autosort_model = SimpleAutoSort(
        ch_num=n_channels,
        samplepoints=samplepoints,
        device=device,
        set_shank_id=set_shank_id,
        save_dir=model_save_dir,
        pos_weight_noise=dataset.pos_weight_noise.to(device),
        pos_weight_label=dataset.pos_weight_label.to(device)
    )
    
    # 加载模型权重
    import os
    if not os.path.exists(autosort_model.save_model_path_2):
        raise FileNotFoundError(f"模型文件不存在: {autosort_model.save_model_path_2}")
    if not os.path.exists(autosort_model.save_model_path_3):
        raise FileNotFoundError(f"模型文件不存在: {autosort_model.save_model_path_3}")
    
    print("加载模型权重...")
    autosort_model.load_model()
    autosort_model.eval()
    print("模型加载完成")
    
    # 创建数据加载器（使用全部数据或测试集）
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n开始评估（共 {len(dataset)} 样本）...")
    
    # 评估
    all_gt_noise = []
    all_pred_noise = []
    all_gt_units = []
    all_pred_units = []
    all_noise_probs = []
    all_unit_probs = []
    
    with torch.no_grad():
        for batch_features, classify_labels, labels, single_waveform in tqdm(test_loader, desc="评估"):
            classify_labels = classify_labels.to(device)
            batch_features = batch_features.view(-1, samplepoints * n_channels).to(device)
            labels = labels.to(device)
            single_waveform = single_waveform.to(device)
            
            # 前向传播
            codes = torch.cat((batch_features, single_waveform), axis=1)
            
            # 噪声分类
            cls_output = autosort_model.clsfier_noise(codes.float())
            noise_probs = torch.softmax(cls_output, dim=1)
            pred_noise = torch.argmax(cls_output, axis=1)
            gt_noise = torch.argmax(labels, axis=1)
            
            all_gt_noise.append(gt_noise.detach().cpu().numpy())
            all_pred_noise.append(pred_noise.detach().cpu().numpy())
            all_noise_probs.append(noise_probs.detach().cpu().numpy())
            
            # 单元分类（只对非噪声样本）
            test = labels[:, 1] == 1
            if sum(test) > 0:
                cls_label_output = autosort_model.clsfier_label(codes.float()[test, :])
                unit_probs = torch.softmax(cls_label_output, dim=1)
                pred_units = torch.argmax(cls_label_output, axis=1)
                gt_units = torch.argmax(classify_labels[test, :len(set_shank_id)], axis=1)
                
                all_gt_units.append(gt_units.detach().cpu().numpy())
                all_pred_units.append(pred_units.detach().cpu().numpy())
                all_unit_probs.append(unit_probs.detach().cpu().numpy())
    
    # 合并结果
    all_gt_noise = np.concatenate(all_gt_noise, axis=0)
    all_pred_noise = np.concatenate(all_pred_noise, axis=0)
    all_noise_probs = np.concatenate(all_noise_probs, axis=0)
    
    if len(all_gt_units) > 0:
        all_gt_units = np.concatenate(all_gt_units, axis=0)
        all_pred_units = np.concatenate(all_pred_units, axis=0)
        all_unit_probs = np.concatenate(all_unit_probs, axis=0)
    else:
        all_gt_units = np.array([])
        all_pred_units = np.array([])
        all_unit_probs = np.array([])
    
    # 计算指标
    noise_accuracy = accuracy_score(all_gt_noise, all_pred_noise)
    
    if len(all_gt_units) > 0:
        unit_f1_score = f1_score(all_gt_units, all_pred_units, average='micro')
        unit_accuracy = accuracy_score(all_gt_units, all_pred_units)
    else:
        unit_f1_score = 0.0
        unit_accuracy = 0.0
    
    print(f"\n评估结果（原始）:")
    print(f"  - 噪声分类准确率: {noise_accuracy:.4f}")
    if len(all_gt_units) > 0:
        print(f"  - 单元分类准确率: {unit_accuracy:.4f}")
        print(f"  - 单元分类F1分数: {unit_f1_score:.4f}")
        print(f"  - 评估的单元样本数: {len(all_gt_units)}")
    print(f"  - 总样本数: {len(all_gt_noise)}")
    
    # 初始化结果字典
    results = {
        'noise_accuracy': noise_accuracy,
        'unit_accuracy': unit_accuracy,
        'unit_f1_score': unit_f1_score,
        'noise_predictions': all_pred_noise,
        'unit_predictions': all_pred_units,
        'gt_noise': all_gt_noise,
        'gt_units': all_gt_units,
        'noise_probs': all_noise_probs,
        'unit_probs': all_unit_probs,
    }
    
    # 如果提供了 eval_neuron_inf_matched，计算调整后的结果（将 unmatch neuron 视为 noise）
    if eval_neuron_inf_matched is not None and eval_data_dir is not None:
        print(f"\n计算调整后的评估结果（将 unmatch neuron 视为 noise）...")
        
        # 加载评估数据的 neuron 映射
        eval_neuron_mapping_path = Path(eval_data_dir) / "neuron_mapping.pkl"
        if not eval_neuron_mapping_path.exists():
            print(f"  警告: 评估数据的 neuron_mapping.pkl 不存在，跳过调整后的结果计算")
        else:
            with open(eval_neuron_mapping_path, "rb") as f:
                eval_neuron_mapping = pickle.load(f)
            
            # 加载评估数据的 Y_spike_id
            with open(eval_data_dir / "Y_spike_id.pkl", "rb") as f:
                eval_Y_spike_id_full = pickle.load(f)
            
            # 找到所有 unmatch 的 neuron 名称
            unmatch_neurons = eval_neuron_inf_matched[
                eval_neuron_inf_matched['neuron_match'] == 'unmatch'
            ]['Neuron'].values
            
            print(f"  - 未匹配的 neuron 数量: {len(unmatch_neurons)}")
            if len(unmatch_neurons) > 0:
                print(f"  - 未匹配的 neuron: {unmatch_neurons}")
            
            # 建立 unmatch neuron 名称到 ID 的映射
            eval_neuron_to_id = eval_neuron_mapping['neuron_to_id']
            unmatch_neuron_ids = [eval_neuron_to_id.get(neuron, -1) for neuron in unmatch_neurons]
            unmatch_neuron_ids = [nid for nid in unmatch_neuron_ids if nid != -1]
            
            # 创建调整后的 GT 标签
            all_gt_noise_adjusted = all_gt_noise.copy()
            
            # 找到所有属于 unmatch neuron 的样本
            unmatch_sample_mask = np.isin(eval_Y_spike_id_full, unmatch_neuron_ids)
            
            # 将属于 unmatch neuron 的 spike 样本标记为 noise
            # 只调整原本是 spike (gt_noise == 1) 的样本
            spike_mask = all_gt_noise == 1
            unmatch_spike_mask = unmatch_sample_mask & spike_mask
            
            all_gt_noise_adjusted[unmatch_spike_mask] = 0  # 标记为 noise
            adjusted_count = np.sum(unmatch_spike_mask)
            
            print(f"  - 调整的样本数: {adjusted_count}")
            
            # 调整单元标签：保留所有原始单元样本（包括未匹配的neuron样本）
            # 找到非噪声样本的索引（原始）
            non_noise_mask_original = all_gt_noise == 1
            non_noise_indices_original = np.where(non_noise_mask_original)[0]
            
            # 找到调整后仍是非噪声的样本索引（排除未匹配的neuron）
            non_noise_mask_adjusted = all_gt_noise_adjusted == 1
            non_noise_indices_adjusted = np.where(non_noise_mask_adjusted)[0]
            
            # 保留所有原始单元样本（85483个），包括未匹配的neuron样本
            # 对于未匹配的neuron样本：
            # - 它们的GT标签在噪声分类中已被改为噪声（0）
            # - 但在单元分类评估中，我们需要将它们也包含进来
            # - 如果网络将未匹配的neuron样本分类为噪声（all_pred_noise == 0），这是正确的
            # - 如果网络将未匹配的neuron样本分类为某个单元（all_pred_noise == 1），这是错误的
            if len(all_gt_units) > 0:
                # 保留所有原始单元样本（包括未匹配的neuron）
                all_gt_units_adjusted = all_gt_units.copy()  # 保留所有原始GT单元标签（85483个）
                
                # 找到匹配的neuron样本（调整后仍是非噪声的）
                matched_unit_mask = np.isin(non_noise_indices_original, non_noise_indices_adjusted)
                
                # 找到未匹配的neuron样本（在原始单元样本中，但调整后被标记为噪声）
                unmatched_unit_mask = ~matched_unit_mask
                unmatched_unit_indices = non_noise_indices_original[unmatched_unit_mask]
                
                # 统计未匹配neuron样本的误判情况
                unmatch_spike_pred_noise = all_pred_noise[unmatched_unit_indices]
                unmatch_correct_as_noise = np.sum(unmatch_spike_pred_noise == 0)  # 网络正确识别为噪声
                unmatch_misclassified_as_unit = np.sum(unmatch_spike_pred_noise == 1)  # 网络误判为单元
                
                # 计算匹配neuron样本的单元分类准确率
                matched_gt_units = all_gt_units[matched_unit_mask]
                matched_pred_units = all_pred_units[matched_unit_mask]
                
                if len(matched_gt_units) > 0:
                    matched_unit_accuracy = accuracy_score(matched_gt_units, matched_pred_units)
                    matched_unit_f1 = f1_score(matched_gt_units, matched_pred_units, average='micro')
                else:
                    matched_unit_accuracy = 0.0
                    matched_unit_f1 = 0.0
                
                # 计算整体单元分类准确率（包括未匹配的neuron样本）
                # 对于匹配的neuron样本：使用单元分类准确率
                # 对于未匹配的neuron样本：如果网络将它们分类为噪声，则正确；如果分类为单元，则错误
                total_unit_samples = len(all_gt_units)  # 85483
                matched_correct = np.sum(matched_gt_units == matched_pred_units) if len(matched_gt_units) > 0 else 0
                unmatched_correct = unmatch_correct_as_noise
                total_correct = matched_correct + unmatched_correct
                unit_accuracy_adjusted = total_correct / total_unit_samples if total_unit_samples > 0 else 0.0
                
                # 对于F1分数，只计算匹配的neuron样本（因为未匹配的neuron样本没有单元标签）
                unit_f1_score_adjusted = matched_unit_f1
                
                # 保存调整后的预测（用于后续分析）
                all_pred_units_adjusted = all_pred_units.copy()
            else:
                all_gt_units_adjusted = np.array([])
                all_pred_units_adjusted = np.array([])
                unit_accuracy_adjusted = 0.0
                unit_f1_score_adjusted = 0.0
                unmatch_correct_as_noise = 0
                unmatch_misclassified_as_unit = 0
                matched_unit_accuracy = 0.0
            
            # 重新计算调整后的噪声分类准确率
            noise_accuracy_adjusted = accuracy_score(all_gt_noise_adjusted, all_pred_noise)
            
            print(f"\n评估结果（调整后）:")
            print(f"  - 噪声分类准确率: {noise_accuracy_adjusted:.4f}")
            print(f"  - 总样本数: {len(all_gt_noise_adjusted)}")  # 包含所有样本
            if len(all_gt_units_adjusted) > 0:
                print(f"  - 单元分类准确率: {unit_accuracy_adjusted:.4f}")
                print(f"  - 单元分类F1分数: {unit_f1_score_adjusted:.4f}")
                print(f"  - 评估的单元样本数: {len(all_gt_units_adjusted)}")  # 应该等于原始的85483
                if adjusted_count > 0:
                    print(f"    - 其中匹配的neuron样本: {len(matched_gt_units) if len(all_gt_units_adjusted) > 0 else 0}")
                    print(f"    - 其中未匹配的neuron样本: {adjusted_count}")
                    print(f"      - 正确识别为噪声: {unmatch_correct_as_noise} ({unmatch_correct_as_noise/adjusted_count*100:.1f}%)")
                    print(f"      - 误判为单元: {unmatch_misclassified_as_unit} ({unmatch_misclassified_as_unit/adjusted_count*100:.1f}%)")
                    print(f"    - 说明: 未匹配的neuron样本被视为噪声，如果网络正确识别为噪声则计入正确，如果误判为单元则计入错误")
            
            # 添加到结果字典
            results['noise_accuracy_adjusted'] = noise_accuracy_adjusted
            results['unit_accuracy_adjusted'] = unit_accuracy_adjusted
            results['unit_f1_score_adjusted'] = unit_f1_score_adjusted
            results['gt_noise_adjusted'] = all_gt_noise_adjusted
            results['gt_units_adjusted'] = all_gt_units_adjusted
            results['unit_predictions_adjusted'] = all_pred_units_adjusted
    
    if save_results:
        if results_save_dir is None:
            results_save_dir = model_save_dir
        
        Path(results_save_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'gt_noise': all_gt_noise,
            'pred_noise': all_pred_noise,
        })
        if len(all_gt_units) > 0:
            # 创建完整的单元标签数组（噪声样本标记为-1）
            full_gt_units = np.full(len(all_gt_noise), -1, dtype=np.int64)
            full_pred_units = np.full(len(all_gt_noise), -1, dtype=np.int64)
            
            # 找到非噪声样本的索引
            non_noise_mask = all_gt_noise == 1
            non_noise_indices = np.where(non_noise_mask)[0]
            
            if len(non_noise_indices) == len(all_gt_units):
                full_gt_units[non_noise_indices] = all_gt_units
                full_pred_units[non_noise_indices] = all_pred_units
            else:
                # 如果数量不匹配，只填充前N个
                n_units = min(len(non_noise_indices), len(all_gt_units))
                full_gt_units[non_noise_indices[:n_units]] = all_gt_units[:n_units]
                full_pred_units[non_noise_indices[:n_units]] = all_pred_units[:n_units]
            
            results_df['gt_units'] = full_gt_units
            results_df['pred_units'] = full_pred_units
        
        results_df.to_csv(results_save_dir + 'evaluation_results.csv')
        print(f"\n评估结果已保存到: {results_save_dir}evaluation_results.csv")
        
        # 保存指标摘要
        summary_data = {
            'metric': ['noise_accuracy', 'unit_accuracy', 'unit_f1_score'],
            'value': [noise_accuracy, unit_accuracy, unit_f1_score]
        }
        
        # 如果有调整后的结果，也添加到摘要中
        if 'noise_accuracy_adjusted' in results:
            summary_data['metric'].extend(['noise_accuracy_adjusted', 'unit_accuracy_adjusted', 'unit_f1_score_adjusted'])
            summary_data['value'].extend([
                results['noise_accuracy_adjusted'],
                results['unit_accuracy_adjusted'],
                results['unit_f1_score_adjusted']
            ])
        
        summary = pd.DataFrame(summary_data)
        summary.to_csv(results_save_dir + 'evaluation_summary.csv', index=False)
        print(f"评估摘要已保存到: {results_save_dir}evaluation_summary.csv")
    
    return results


# ==================== 6. 优化的分类流程（两阶段） ====================

# 通道位置映射（与generate_neuron_inf_phy_template.py保持一致）
CHANNEL_POSITION = {
    0: (650.0, 0.0),
    2: (650.0, 50.0),
    4: (650.0, 100.0),
    6: (600.0, 100.0),
    8: (600.0, 50.0),
    10: (600.0, 0.0),
    1: (0.0, 0.0),
    3: (0.0, 50.0),
    5: (0.0, 100.0),
    7: (50.0, 100.0),
    9: (50.0, 50.0),
    11: (50.0, 0.0),
    13: (150.0, 200.0),
    15: (150.0, 250.0),
    17: (150.0, 300.0),
    19: (200.0, 300.0),
    21: (200.0, 250.0),
    23: (200.0, 200.0),
    12: (500.0, 200.0),
    14: (500.0, 250.0),
    16: (500.0, 300.0),
    18: (450.0, 300.0),
    20: (450.0, 250.0),
    22: (450.0, 200.0),
    24: (350.0, 400.0),
    26: (350.0, 450.0),
    28: (350.0, 500.0),
    25: (300.0, 400.0),
    27: (300.0, 450.0),
    29: (300.0, 500.0),
}


def compute_cluster_position_waveform(
    snippets: np.ndarray,
    channel_id: list,
    window_size: int = 30,
) -> tuple:
    """
    从snippets计算cluster的位置和position_waveform（参考generate_neuron_inf_phy_template.py）
    
    参数:
        snippets: numpy数组，形状为 (n_spikes, n_channels, window_size)
        channel_id: 通道ID列表
        window_size: 窗口大小，默认30
    
    返回:
        position_1, position_2, position_waveform (30-dim)
    """
    cluster_positions_x = []
    cluster_positions_y = []
    cluster_waveforms = []
    
    for snippet in snippets:  # snippet: (n_channels, window_size)
        # 计算该spike的位置（基于channel_id的通道）
        a_squared = [np.sum(snippet[j, :]**2) for j in range(len(channel_id))]
        
        sum_x_a = 0
        sum_y_a = 0
        sum_a = 0
        
        for j, ch in enumerate(channel_id):
            x_i, y_i = CHANNEL_POSITION.get(ch, (0, 0))
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
        
        # 计算position_waveform（基于该spike的位置和channel_id的通道）
        distances = []
        for ch in channel_id:
            x_channel, y_channel = CHANNEL_POSITION.get(ch, (np.nan, np.nan))
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
            spike_position_waveform = snippet[zero_idx, :].astype(np.float32)
        else:
            weights /= weights.sum()
            spike_position_waveform = np.zeros(window_size, dtype=np.float32)
            for t in range(window_size):
                spike_position_waveform[t] = float(np.dot(snippet[:, t], weights))
        
        cluster_waveforms.append(spike_position_waveform)
    
    if len(cluster_waveforms) == 0:
        return 0.0, 0.0, np.zeros(window_size, dtype=np.float32)
    
    # 计算平均位置和waveform
    cluster_x = np.mean(cluster_positions_x)
    cluster_y = np.mean(cluster_positions_y)
    cluster_avg_waveform = np.mean(cluster_waveforms, axis=0)
    
    return cluster_x, cluster_y, cluster_avg_waveform


def calibration_model(
    recording_f,
    autosort_model: SimpleAutoSort,
    train_neuron_inf: pd.DataFrame,
    calibration_duration_seconds: int = 60,
    n_additional_clusters: int = 5,
    detection_params: dict = None,
    window_params: dict = None,
    position_threshold: float = 10.0,
    waveform_similarity_threshold: float = 0.9,
    eval_neuron_inf: pd.DataFrame = None,
    eval_spike_inf: pd.DataFrame = None,
    device=None,
):
    """
    第一阶段：前60s的calibration阶段
    
    流程：
    1. 阈值检测
    2. 通过noise分类器，认定为spike的
    3. 提取way3层（100维）
    4. PCA降维至30维
    5. K-means聚类（class数 = train neuron数 + n）
    6. 对每个cluster计算位置和波形
    7. 与train neuron匹配，建立映射关系
    
    参数:
        recording_f: 预处理后的recording对象
        autosort_model: 训练好的SimpleAutoSort模型
        train_neuron_inf: 训练数据的neuron_inf DataFrame
        calibration_duration_seconds: calibration时长（秒），默认60
        n_additional_clusters: 额外的cluster数量（n），默认5
        detection_params: 检测参数字典
        window_params: 窗口参数字典
        position_threshold: 位置距离阈值（微米），默认10
        waveform_similarity_threshold: 波形相似性阈值，默认0.9
        device: 设备
    
    返回:
        calibration_results: 字典，包含：
            - kmeans_model: 训练好的K-means模型
            - pca_model: 训练好的PCA模型
            - cluster_to_neuron_mapping: cluster到train neuron的映射
            - cluster_features: 每个cluster的特征（位置、波形等）
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from scipy.stats import pearsonr
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if detection_params is None:
        detection_params = {
            'thr_min': 3.5,
            'thr_max': 30,
            'distance': 3,
            'ch_max_simul_firing': 5,
            'wlen': 5,
            'prominence': 10,
        }
    
    if window_params is None:
        window_params = {
            'left_sample': 10,
            'right_sample': 20,
        }
    
    left_sample = window_params['left_sample']
    right_sample = window_params['right_sample']
    window_size = left_sample + right_sample
    n_channels = recording_f.get_num_channels()
    sampling_frequency = recording_f.get_sampling_frequency()
    
    print("=" * 50)
    print("第一阶段：Calibration (前60秒)")
    print("=" * 50)
    
    # 1. 加载前60s的数据
    max_duration_samples = int(calibration_duration_seconds * sampling_frequency)
    print(f"加载前 {calibration_duration_seconds} 秒的数据...")
    traces = recording_f.get_traces(start_frame=0, end_frame=max_duration_samples)
    if traces.shape[0] > traces.shape[1] and traces.shape[0] > 100:
        traces = traces.T
    traces = traces.astype(np.float32)
    print(f"数据形状: {traces.shape}")
    
    # 2. 阈值检测
    print("\n### 2. 阈值检测")
    trace0_car = traces.T  # (n_timepoints, n_channels)
    spikes = detect_spike(trace0_car, **detection_params)
    spike_coords = np.argwhere(spikes == 1)  # (n_spikes, 2) [time, channel]
    print(f"检测到的spike数量: {len(spike_coords)}")
    
    # 3. 提取波形并过滤边界
    print("\n### 3. 提取波形")
    valid_spikes = []
    waveforms = []
    spike_times = []
    spike_channels = []
    
    for time_idx, channel_idx in spike_coords:
        start = time_idx - left_sample
        end = time_idx + right_sample
        
        if start < 0 or end > trace0_car.shape[0]:
            continue
        if end - start != window_size:
            continue
        
        # 提取波形 (n_channels, window_size)
        waveform = traces[:, start:end]  # (n_channels, window_size)
        waveforms.append(waveform)
        valid_spikes.append((time_idx, channel_idx))
        spike_times.append(time_idx)
        spike_channels.append(channel_idx)
    
    waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size)
    print(f"有效spike数量: {len(waveforms)}")
    
    if len(waveforms) == 0:
        raise ValueError("没有有效的spike用于calibration")
    
    # 4. 通过noise分类器，认定为spike的
    print("\n### 4. Noise分类器过滤")
    autosort_model.eval()
    
    # 准备数据
    batch_size = 512
    n_spikes = len(waveforms)
    spike_indices = []
    way3_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, n_spikes, batch_size), desc="Noise分类"):
            batch_waveforms = waveforms[i:i+batch_size]  # (batch, n_channels, window_size)
            batch_channels = spike_channels[i:i+batch_size]
            
            # 提取single waveform（最大幅度通道）
            batch_single_waveforms = []
            batch_multi_waveforms = []
            
            for j, (wf, ch) in enumerate(zip(batch_waveforms, batch_channels)):
                # multi-waveform: 展平为 (n_channels * window_size,)
                multi_wf = wf.flatten()  # (n_channels * window_size,)
                batch_multi_waveforms.append(multi_wf)
                
                # single-waveform: 最大幅度通道的波形
                single_wf = wf[ch, :]  # (window_size,)
                batch_single_waveforms.append(single_wf)
            
            batch_multi_waveforms = np.array(batch_multi_waveforms)  # (batch, n_channels * window_size)
            batch_single_waveforms = np.array(batch_single_waveforms)  # (batch, window_size)
            
            # 转换为tensor
            batch_multi = torch.from_numpy(batch_multi_waveforms).float().to(device)
            batch_single = torch.from_numpy(batch_single_waveforms).float().to(device)
            
            # 拼接
            codes = torch.cat((batch_multi, batch_single), dim=1)  # (batch, (n_channels+1)*window_size)
            
            # Noise分类
            noise_output = autosort_model.clsfier_noise(codes)
            noise_pred = torch.argmax(noise_output, dim=1)  # 0=noise, 1=spike
            
            # 只保留被分类为spike的样本
            spike_mask = noise_pred == 1
            if spike_mask.sum() > 0:
                batch_indices = np.arange(i, min(i+batch_size, n_spikes))[spike_mask.cpu().numpy()]
                spike_indices.extend(batch_indices)
                
                # 提取way3层特征（只对spike样本）
                codes_spike = codes[spike_mask]
                way3_batch = autosort_model.clsfier_label.intermediate_forward(codes_spike)
                way3_features.append(way3_batch.cpu().numpy())
    
    if len(spike_indices) == 0:
        raise ValueError("没有spike通过noise分类器")
    
    way3_features = np.concatenate(way3_features, axis=0)  # (n_spikes, 100)
    spike_indices = np.array(spike_indices)
    print(f"通过noise分类器的spike数量: {len(spike_indices)}")
    
    # 5. PCA降维至30维
    print("\n### 5. PCA降维")
    pca = PCA(n_components=30)
    way3_pca = pca.fit_transform(way3_features)  # (n_spikes, 30)
    print(f"PCA降维后特征形状: {way3_pca.shape}")
    print(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    
    # 6. K-means聚类
    print("\n### 6. K-means聚类")
    n_train_neurons = len(train_neuron_inf)
    n_clusters = n_train_neurons + n_additional_clusters
    print(f"聚类数: {n_clusters} (训练neuron数: {n_train_neurons}, 额外: {n_additional_clusters})")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(way3_pca)  # (n_spikes,)
    print(f"聚类完成，每个cluster的样本数:")
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} 样本")
    
    # 7. 对每个train neuron和每个cluster计算位置和波形，然后匹配
    print("\n### 7. 计算cluster位置和波形（基于train neuron的channel_id）并匹配")
    cluster_to_neuron_mapping = {}  # {cluster_id: train_neuron_name}
    neuron_to_clusters = defaultdict(list)  # {train_neuron_name: [cluster_ids]}
    cluster_features = {}  # 保存匹配后的cluster特征
    
    # 外层循环：遍历每个train neuron
    for train_idx, train_row in train_neuron_inf.iterrows():
        train_neuron = train_row['Neuron']
        train_pos = np.array([train_row['position_1'], train_row['position_2']])
        train_waveform = np.asarray(train_row['position_waveform'], dtype=np.float32)
        
        # 获取train neuron的channel_id
        train_channel_id = train_row['channel_id']
        if not isinstance(train_channel_id, list):
            if isinstance(train_channel_id, (np.ndarray, tuple)):
                train_channel_id = list(train_channel_id)
            else:
                # 尝试解析字符串
                import ast
                try:
                    train_channel_id = ast.literal_eval(str(train_channel_id))
                    if not isinstance(train_channel_id, list):
                        train_channel_id = [train_channel_id]
                except:
                    print(f"  警告: Neuron {train_neuron} 的channel_id无法解析，跳过")
                    continue
        
        if len(train_channel_id) == 0:
            print(f"  警告: Neuron {train_neuron} 没有有效的channel_id，跳过")
            continue
        
        print(f"\n  处理 Neuron {train_neuron} (channel_id: {train_channel_id})")
        
        # 内层循环：遍历每个kmeans cluster
        for cluster_id in range(n_clusters):
            # 如果该cluster已经匹配到其他neuron，跳过（一个cluster只能匹配一个neuron）
            if cluster_id in cluster_to_neuron_mapping:
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            
            if len(cluster_spike_indices) == 0:
                continue
            
            # 获取该cluster的所有波形
            cluster_waveforms_full = waveforms[cluster_spike_indices]  # (n_spikes, n_channels, window_size)
            
            # 使用train neuron的channel_id提取对应的通道
            # 确保channel_id在有效范围内
            valid_channel_id = [ch for ch in train_channel_id if 0 <= ch < n_channels]
            if len(valid_channel_id) == 0:
                continue
            
            # 从cluster_waveforms中提取train neuron的channel_id对应的通道
            cluster_waveforms = cluster_waveforms_full[:, valid_channel_id, :]  # (n_spikes, n_valid_channels, window_size)
            
            # 计算位置和波形（使用train neuron的channel_id）
            position_1, position_2, position_waveform = compute_cluster_position_waveform(
                cluster_waveforms, valid_channel_id, window_size
            )
            
            # 计算位置距离
            cluster_pos = np.array([position_1, position_2])
            pos_distance = np.linalg.norm(cluster_pos - train_pos)
            if pos_distance >= position_threshold:
                continue
            
            # 计算波形相似性
            min_len = min(len(position_waveform), len(train_waveform))
            if min_len == 0:
                continue
            corr, _ = pearsonr(position_waveform[:min_len], train_waveform[:min_len])
            
            if corr < waveform_similarity_threshold:
                continue
            
            # 计算综合得分（距离越小、相关性越高，得分越高）
            score = corr / (1 + pos_distance / position_threshold)
            
            # 建立映射关系（一个cluster只能匹配一个neuron，选择最优的）
            # 如果该cluster还没有匹配，或者当前匹配的得分更高，则更新
            if cluster_id not in cluster_to_neuron_mapping:
                cluster_to_neuron_mapping[cluster_id] = train_neuron
                neuron_to_clusters[train_neuron].append(cluster_id)
                cluster_features[cluster_id] = {
                    'position_1': position_1,
                    'position_2': position_2,
                    'position_waveform': position_waveform,
                    'n_spikes': len(cluster_spike_indices),
                    'matched_neuron': train_neuron,
                    'score': score,
                    'pos_distance': pos_distance,
                    'waveform_corr': corr,
                }
                print(f"    Cluster {cluster_id} -> {train_neuron} (得分: {score:.4f}, 距离: {pos_distance:.2f}, 相关性: {corr:.4f})")
            else:
                # 如果已经有匹配，比较得分，选择最优的
                existing_neuron = cluster_to_neuron_mapping[cluster_id]
                existing_score = cluster_features[cluster_id]['score']
                if score > existing_score:
                    # 移除旧的映射
                    neuron_to_clusters[existing_neuron].remove(cluster_id)
                    # 建立新的映射
                    cluster_to_neuron_mapping[cluster_id] = train_neuron
                    neuron_to_clusters[train_neuron].append(cluster_id)
                    cluster_features[cluster_id] = {
                        'position_1': position_1,
                        'position_2': position_2,
                        'position_waveform': position_waveform,
                        'n_spikes': len(cluster_spike_indices),
                        'matched_neuron': train_neuron,
                        'score': score,
                        'pos_distance': pos_distance,
                        'waveform_corr': corr,
                    }
                    print(f"    Cluster {cluster_id} -> {train_neuron} (更新匹配，得分: {score:.4f} > {existing_score:.4f})")
    
    # 标记未匹配的cluster
    print("\n  未匹配的cluster:")
    for cluster_id in range(n_clusters):
        if cluster_id not in cluster_to_neuron_mapping:
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            print(f"    Cluster {cluster_id}: {len(cluster_spike_indices)} 个spike")
    
    # 处理冲突：一个cluster只能映射一个neuron，选择最优的
    # 如果一个cluster同时满足多个neuron的条件，选择得分最高的
    # （上面的代码已经处理了这个问题）
    
    # 处理一对多：一个neuron可以映射多个cluster（保留所有映射）
    
    print(f"\n匹配结果:")
    print(f"  - 总cluster数: {n_clusters}")
    print(f"  - 匹配的cluster数: {len(cluster_to_neuron_mapping)}")
    print(f"  - 未匹配的cluster数: {n_clusters - len(cluster_to_neuron_mapping)}")
    print(f"  - 匹配的neuron数: {len(neuron_to_clusters)}")
    
    # 构建结果DataFrame（用于confusion matrix）
    results_df = pd.DataFrame({
        'spike_time': [spike_times[i] for i in spike_indices],
        'spike_channel': [spike_channels[i] for i in spike_indices],
        'predicted_label': [cluster_to_neuron_mapping.get(cluster_labels[i], 'unmatch') for i in range(len(spike_indices))],
    })
    
    # 如果有eval数据，添加GT标签
    if eval_neuron_inf is not None and eval_spike_inf is not None:
        # 建立neuron映射（从eval_neuron_inf到train_neuron）
        if 'neuron_match' in eval_neuron_inf.columns:
            # 建立eval neuron到train neuron的映射
            eval_to_train_mapping = {}
            for _, row in eval_neuron_inf.iterrows():
                eval_neuron = row['Neuron']
                match = row['neuron_match']
                if match != 'unmatch':
                    eval_to_train_mapping[eval_neuron] = match
                else:
                    eval_to_train_mapping[eval_neuron] = 'unmatch'
            
            # 根据spike_time匹配GT标签
            spike_times_array = np.array([spike_times[i] for i in spike_indices])
            spike_inf_sorted = eval_spike_inf.sort_values('time').reset_index(drop=True)
            
            gt_labels = []
            for spike_time in spike_times_array:
                # 在eval_spike_inf中查找对应的spike（允许±1采样点的误差）
                time_diff = (spike_inf_sorted['time'] - spike_time).abs()
                min_diff_idx = time_diff.idxmin()
                min_diff = time_diff.loc[min_diff_idx]
                
                if min_diff <= 1:  # 允许±1采样点的误差
                    eval_neuron = spike_inf_sorted.loc[min_diff_idx, 'neuron']
                    
                    # 映射到train neuron
                    if eval_neuron in eval_to_train_mapping:
                        gt_label = eval_to_train_mapping[eval_neuron]
                    else:
                        gt_label = 'unmatch'
                else:
                    gt_label = 'noise'  # 没有匹配到GT spike，视为noise
                
                gt_labels.append(gt_label)
            
            results_df['gt_label'] = gt_labels
        else:
            print("警告: eval_neuron_inf中没有neuron_match列，无法建立GT标签映射")
            results_df['gt_label'] = 'unknown'
    else:
        results_df['gt_label'] = None
    
    calibration_results = {
        'kmeans_model': kmeans,
        'pca_model': pca,
        'cluster_to_neuron_mapping': cluster_to_neuron_mapping,
        'neuron_to_clusters': dict(neuron_to_clusters),
        'cluster_features': cluster_features,
        'spike_indices': spike_indices,
        'cluster_labels': cluster_labels,
        'results_df': results_df,  # 添加results_df用于confusion matrix
    }
    
    return calibration_results


def real_time_processing(
    recording_f,
    autosort_model: SimpleAutoSort,
    calibration_results: dict,
    start_time_seconds: float = 60.0,
    time_window_seconds: float = 10.0,
    total_duration_seconds: float = None,
    detection_params: dict = None,
    window_params: dict = None,
    eval_neuron_inf: pd.DataFrame = None,
    eval_spike_inf: pd.DataFrame = None,
    device=None,
):
    """
    第二阶段：实时处理（按time_window为单位处理）
    
    流程：
    1. 按time_window加载数据
    2. 阈值检测
    3. 通过noise分类器，认定为spike的
    4. 提取way3层 → PCA降维 → K-means预测 → 映射到train neuron ID
    
    参数:
        recording_f: 预处理后的recording对象
        autosort_model: 训练好的SimpleAutoSort模型
        calibration_results: calibration阶段的结果（包含kmeans_model, pca_model, cluster_to_neuron_mapping）
        start_time_seconds: 开始处理的时间（秒），默认60（calibration之后）
        time_window_seconds: 每个时间窗口的长度（秒），默认10
        total_duration_seconds: 总处理时长（秒），如果为None则处理到recording结束，默认None
        detection_params: 检测参数字典
        window_params: 窗口参数字典
        eval_neuron_inf: 评估数据的neuron_inf（用于生成GT标签）
        eval_spike_inf: 评估数据的spike_inf（用于生成GT标签）
        device: 设备
    
    返回:
        processing_results: 字典，包含：
            - spike_predictions: 每个spike的预测neuron ID
            - spike_times: 每个spike的时间
            - spike_channels: 每个spike的通道
            - results_df: 包含gt_label和predicted_label的DataFrame
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if detection_params is None:
        detection_params = {
            'thr_min': 3.5,
            'thr_max': 30,
            'distance': 3,
            'ch_max_simul_firing': 5,
            'wlen': 5,
            'prominence': 10,
        }
    
    if window_params is None:
        window_params = {
            'left_sample': 10,
            'right_sample': 20,
        }
    
    left_sample = window_params['left_sample']
    right_sample = window_params['right_sample']
    window_size = left_sample + right_sample
    n_channels = recording_f.get_num_channels()
    sampling_frequency = recording_f.get_sampling_frequency()
    
    # 获取calibration阶段的模型和映射
    kmeans_model = calibration_results['kmeans_model']
    pca_model = calibration_results['pca_model']
    cluster_to_neuron_mapping = calibration_results['cluster_to_neuron_mapping']
    
    print("=" * 50)
    print("第二阶段：实时处理")
    print("=" * 50)
    print(f"开始时间: {start_time_seconds} 秒")
    print(f"时间窗口: {time_window_seconds} 秒")
    
    # 计算recording的总时长
    total_duration_samples = recording_f.get_num_samples()
    recording_total_seconds = total_duration_samples / sampling_frequency
    start_frame = int(start_time_seconds * sampling_frequency)
    window_frames = int(time_window_seconds * sampling_frequency)
    
    # 计算结束时间
    if total_duration_seconds is not None:
        end_time_seconds = start_time_seconds + total_duration_seconds
        end_frame = min(int(end_time_seconds * sampling_frequency), total_duration_samples)
        print(f"总处理时长: {total_duration_seconds} 秒 (从 {start_time_seconds}s 到 {end_time_seconds}s)")
    else:
        end_frame = total_duration_samples
        print(f"处理到recording结束 (从 {start_time_seconds}s 到 {recording_total_seconds:.1f}s)")
    
    all_spike_predictions = []
    all_spike_times = []
    all_spike_channels = []
    
    autosort_model.eval()
    
    # 按time_window处理
    current_start_frame = start_frame
    window_idx = 0
    
    while current_start_frame < end_frame:
        window_end_frame = min(current_start_frame + window_frames, total_duration_samples)
        window_duration = (window_end_frame - current_start_frame) / sampling_frequency
        
        print(f"\n处理窗口 {window_idx + 1} ({current_start_frame/sampling_frequency:.1f}s - {window_end_frame/sampling_frequency:.1f}s)")
        
        # 1. 加载当前窗口的数据
        traces = recording_f.get_traces(start_frame=current_start_frame, end_frame=window_end_frame)
        if traces.shape[0] > traces.shape[1] and traces.shape[0] > 100:
            traces = traces.T
        traces = traces.astype(np.float32)
        
        # 2. 阈值检测
        trace0_car = traces.T  # (n_timepoints, n_channels)
        spikes = detect_spike(trace0_car, **detection_params)
        spike_coords = np.argwhere(spikes == 1)  # (n_spikes, 2) [time, channel]
        
        if len(spike_coords) == 0:
            print(f"  窗口 {window_idx + 1}: 未检测到spike")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        # 3. 提取波形并过滤边界
        waveforms = []
        valid_spike_coords = []
        spike_times = []
        spike_channels = []
        
        for time_idx, channel_idx in spike_coords:
            # 转换为全局时间索引
            global_time_idx = current_start_frame + time_idx
            local_start = time_idx - left_sample
            local_end = time_idx + right_sample
            
            if local_start < 0 or local_end > trace0_car.shape[0]:
                continue
            if local_end - local_start != window_size:
                continue
            
            # 提取波形 (n_channels, window_size)
            waveform = traces[:, local_start:local_end]  # (n_channels, window_size)
            waveforms.append(waveform)
            valid_spike_coords.append((time_idx, channel_idx))
            spike_times.append(global_time_idx)
            spike_channels.append(channel_idx)
        
        if len(waveforms) == 0:
            print(f"  窗口 {window_idx + 1}: 没有有效的spike波形")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size)
        
        # 4. 通过noise分类器，认定为spike的
        batch_size = 512
        n_spikes = len(waveforms)
        way3_features_list = []
        way3_spike_indices = []  # 记录每个way3特征对应的原始spike索引
        
        with torch.no_grad():
            for i in range(0, n_spikes, batch_size):
                batch_end = min(i + batch_size, n_spikes)
                batch_waveforms = waveforms[i:batch_end]
                batch_channels = spike_channels[i:batch_end]
                
                # 提取single waveform和multi waveform
                batch_single_waveforms = []
                batch_multi_waveforms = []
                
                for wf, ch in zip(batch_waveforms, batch_channels):
                    multi_wf = wf.flatten()
                    batch_multi_waveforms.append(multi_wf)
                    single_wf = wf[ch, :]
                    batch_single_waveforms.append(single_wf)
                
                batch_multi_waveforms = np.array(batch_multi_waveforms)
                batch_single_waveforms = np.array(batch_single_waveforms)
                
                # 转换为tensor
                batch_multi = torch.from_numpy(batch_multi_waveforms).float().to(device)
                batch_single = torch.from_numpy(batch_single_waveforms).float().to(device)
                
                # 拼接
                codes = torch.cat((batch_multi, batch_single), dim=1)
                
                # Noise分类
                noise_output = autosort_model.clsfier_noise(codes)
                noise_pred = torch.argmax(noise_output, dim=1)
                
                # 只保留被分类为spike的样本
                spike_mask = noise_pred == 1
                if spike_mask.sum() > 0:
                    # 记录通过noise分类器的spike的原始索引
                    batch_spike_indices = np.arange(i, batch_end)[spike_mask.cpu().numpy()]
                    way3_spike_indices.extend(batch_spike_indices.tolist())
                    
                    # 提取way3层特征
                    codes_spike = codes[spike_mask]
                    way3_batch = autosort_model.clsfier_label.intermediate_forward(codes_spike)
                    way3_features_list.append(way3_batch.cpu().numpy())
        
        if len(way3_features_list) == 0:
            print(f"  窗口 {window_idx + 1}: 没有spike通过noise分类器")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        # 合并所有spike的特征
        way3_features = np.concatenate(way3_features_list, axis=0)  # (n_spikes_passed, 100)
        way3_spike_indices = np.array(way3_spike_indices)  # 对应的原始spike索引
        
        # 5. PCA降维
        way3_pca = pca_model.transform(way3_features)  # (n_spikes_passed, 30)
        
        # 6. K-means预测
        cluster_labels = kmeans_model.predict(way3_pca)  # (n_spikes_passed,)
        
        # 7. 映射到train neuron ID
        neuron_predictions = []
        for cluster_id in cluster_labels:
            if cluster_id in cluster_to_neuron_mapping:
                neuron_predictions.append(cluster_to_neuron_mapping[cluster_id])
            else:
                neuron_predictions.append('unmatch')
        
        # 使用way3_spike_indices来获取对应的spike时间和通道
        valid_spike_times = [spike_times[i] for i in way3_spike_indices]
        valid_spike_channels = [spike_channels[i] for i in way3_spike_indices]
        valid_neuron_predictions = neuron_predictions  # 已经是对应通过noise分类器的spike了
        
        all_spike_predictions.extend(valid_neuron_predictions)
        all_spike_times.extend(valid_spike_times)
        all_spike_channels.extend(valid_spike_channels)
        
        print(f"  窗口 {window_idx + 1}: {len(valid_spike_times)} 个spike")
        print(f"    - 匹配的neuron: {sum(1 for p in valid_neuron_predictions if p != 'unmatch')}")
        print(f"    - 未匹配: {sum(1 for p in valid_neuron_predictions if p == 'unmatch')}")
        
        # 移动到下一个窗口
        current_start_frame = window_end_frame
        window_idx += 1
    
    # 构建结果DataFrame
    results_df = pd.DataFrame({
        'spike_time': all_spike_times,
        'spike_channel': all_spike_channels,
        'predicted_label': all_spike_predictions,
    })
    
    # 如果有eval数据，添加GT标签
    if eval_neuron_inf is not None and eval_spike_inf is not None:
        # 建立neuron映射（从eval_neuron_inf到train_neuron）
        # 这里需要根据之前的match_neurons结果来建立映射
        # 假设eval_neuron_inf已经有neuron_match列（从match_neurons函数得到）
        if 'neuron_match' in eval_neuron_inf.columns:
            # 建立eval neuron到train neuron的映射
            eval_to_train_mapping = {}
            for _, row in eval_neuron_inf.iterrows():
                eval_neuron = row['Neuron']
                match = row['neuron_match']
                if match != 'unmatch':
                    eval_to_train_mapping[eval_neuron] = match
                else:
                    eval_to_train_mapping[eval_neuron] = 'unmatch'
            
            # 根据spike_time匹配GT标签
            # 为了提高效率，先建立spike_inf的索引
            spike_inf_sorted = eval_spike_inf.sort_values('time').reset_index(drop=True)
            
            gt_labels = []
            for spike_time in all_spike_times:
                # 在eval_spike_inf中查找对应的spike（允许±1采样点的误差）
                # 使用二分查找提高效率
                time_diff = (spike_inf_sorted['time'] - spike_time).abs()
                min_diff_idx = time_diff.idxmin()
                min_diff = time_diff.loc[min_diff_idx]
                
                if min_diff <= 1:  # 允许±1采样点的误差
                    eval_neuron = spike_inf_sorted.loc[min_diff_idx, 'neuron']
                    
                    # 映射到train neuron
                    if eval_neuron in eval_to_train_mapping:
                        gt_label = eval_to_train_mapping[eval_neuron]
                    else:
                        gt_label = 'unmatch'
                else:
                    gt_label = 'noise'  # 没有匹配到GT spike，视为noise
                
                gt_labels.append(gt_label)
            
            results_df['gt_label'] = gt_labels
        else:
            print("警告: eval_neuron_inf中没有neuron_match列，无法建立GT标签映射")
            results_df['gt_label'] = 'unknown'
    else:
        results_df['gt_label'] = None
    
    processing_results = {
        'spike_predictions': all_spike_predictions,
        'spike_times': all_spike_times,
        'spike_channels': all_spike_channels,
        'results_df': results_df,
    }
    
    print(f"\n处理完成:")
    print(f"  - 总spike数: {len(all_spike_predictions)}")
    print(f"  - 匹配的neuron数: {sum(1 for p in all_spike_predictions if p != 'unmatch')}")
    print(f"  - 未匹配数: {sum(1 for p in all_spike_predictions if p == 'unmatch')}")
    
    return processing_results


def generate_confusion_matrix_df(
    results_df: pd.DataFrame,
    train_neuron_list: list = None,
):
    """
    生成confusion matrix的DataFrame
    
    参数:
        results_df: 包含gt_label和predicted_label的DataFrame
        train_neuron_list: train neuron列表（用于排序）
    
    返回:
        confusion_df: confusion matrix的DataFrame
        summary_df: 包含gt_label和predicted_label的详细DataFrame
    """
    # 确保有gt_label和predicted_label列
    if 'gt_label' not in results_df.columns or 'predicted_label' not in results_df.columns:
        raise ValueError("results_df必须包含'gt_label'和'predicted_label'列")
    
    # 创建summary DataFrame（所有经过noise classifier的spike）
    summary_df = results_df[['gt_label', 'predicted_label']].copy()
    
    # 生成confusion matrix
    confusion_matrix = pd.crosstab(
        summary_df['gt_label'], 
        summary_df['predicted_label'], 
        margins=True
    )
    
    # 如果有train_neuron_list，按照指定顺序排序
    if train_neuron_list is not None:
        # 获取所有唯一的标签
        all_gt_labels = sorted(summary_df['gt_label'].unique())
        all_pred_labels = sorted(summary_df['predicted_label'].unique())
        
        # 按照train_neuron_list排序，然后添加unmatch和noise
        ordered_labels = []
        for label in train_neuron_list:
            if label in all_gt_labels or label in all_pred_labels:
                ordered_labels.append(label)
        
        # 添加unmatch和noise（如果存在）
        for label in ['unmatch', 'noise']:
            if label in all_gt_labels or label in all_pred_labels:
                if label not in ordered_labels:
                    ordered_labels.append(label)
        
        # 重新排序confusion matrix
        confusion_matrix = confusion_matrix.reindex(
            index=ordered_labels + ['All'] if 'All' in confusion_matrix.index else ordered_labels,
            columns=ordered_labels + ['All'] if 'All' in confusion_matrix.columns else ordered_labels,
            fill_value=0
        )
    
    return confusion_matrix, summary_df


def compute_noise_detection_metrics(
    results_df: pd.DataFrame,
    train_neuron_list: list = None,
):
    """
    重新计算noise detection的混淆矩阵和准确率
    
    说明：
    - 在calibration阶段，经过noise分类器后，被分类为spike的样本会进入后续的聚类和匹配
    - 如果GT=noise，但被noise分类器误判为spike，然后经过聚类匹配后，可能被归为unmatch
    - 因此，GT=noise且predicted=unmatch的，应该认为是noise detection的真阴性（TN）
    
    重要说明：
    - 此函数只计算经过noise分类器后被分类为spike的样本（即results_df中的所有样本）
    - 与evaluate_autosort_model中的噪声分类准确率不同：
      * evaluate_autosort_model: 包括所有检测到的spike（包括被分类为noise的），样本数=448248
      * compute_noise_detection_metrics: 只包括被分类为spike的样本，样本数=37503（calibration阶段）
    - 因此这两个准确率不能直接比较，因为计算基础不同
    - 准确率下降是正常的，因为这里只关注被误判为spike的noise样本的后续处理结果
    
    参数:
        results_df: 包含gt_label和predicted_label的DataFrame（只包括经过noise分类器后被分类为spike的样本）
        train_neuron_list: train neuron列表（用于判断哪些是train neuron）
    
    返回:
        noise_detection_metrics: 字典，包含：
            - confusion_matrix: noise detection的混淆矩阵（2x2）
            - TP, TN, FP, FN: 真阳性、真阴性、假阳性、假阴性
            - accuracy: 准确率
            - precision: 精确率
            - recall: 召回率
            - f1_score: F1分数
            - specificity: 特异性
    """
    if 'gt_label' not in results_df.columns or 'predicted_label' not in results_df.columns:
        raise ValueError("results_df必须包含'gt_label'和'predicted_label'列")
    
    # 判断哪些是train neuron
    if train_neuron_list is None:
        # 从results_df中推断train neuron列表
        all_gt_labels = set(results_df['gt_label'].unique())
        all_pred_labels = set(results_df['predicted_label'].unique())
        train_neuron_list = sorted([l for l in (all_gt_labels | all_pred_labels) 
                                   if l not in ['noise', 'unmatch', 'unknown']])
    
    # 将GT标签转换为noise/spike二分类
    # GT=noise -> noise
    # GT=train_neuron或unmatch -> spike（因为这些都是真正的spike，只是可能没有匹配到train neuron）
    def get_gt_noise_label(gt_label):
        if gt_label == 'noise':
            return 'noise'
        elif gt_label in train_neuron_list or gt_label == 'unmatch':
            return 'spike'
        else:
            return 'unknown'
    
    # 将预测标签转换为noise/spike二分类
    # predicted=unmatch -> noise（包括GT=noise被误判为spike后归为unmatch的情况）
    # predicted=train_neuron -> spike
    def get_pred_noise_label(pred_label):
        if pred_label == 'unmatch':
            return 'noise'
        elif pred_label in train_neuron_list:
            return 'spike'
        else:
            return 'unknown'
    
    # 创建noise detection的二分类标签
    noise_detection_df = results_df.copy()
    noise_detection_df['gt_noise'] = noise_detection_df['gt_label'].apply(get_gt_noise_label)
    noise_detection_df['pred_noise'] = noise_detection_df['predicted_label'].apply(get_pred_noise_label)
    
    # 过滤掉unknown的样本
    noise_detection_df = noise_detection_df[
        (noise_detection_df['gt_noise'] != 'unknown') & 
        (noise_detection_df['pred_noise'] != 'unknown')
    ]
    
    # 计算混淆矩阵
    confusion_matrix = pd.crosstab(
        noise_detection_df['gt_noise'],
        noise_detection_df['pred_noise'],
        margins=True
    )
    
    # 确保有noise和spike两行和两列
    for label in ['noise', 'spike']:
        if label not in confusion_matrix.index:
            confusion_matrix.loc[label] = 0
        if label not in confusion_matrix.columns:
            confusion_matrix[label] = 0
    
    # 重新排序
    confusion_matrix = confusion_matrix.reindex(
        index=['noise', 'spike', 'All'] if 'All' in confusion_matrix.index else ['noise', 'spike'],
        columns=['noise', 'spike', 'All'] if 'All' in confusion_matrix.columns else ['noise', 'spike'],
        fill_value=0
    )
    
    # 计算TP, TN, FP, FN
    # TP: GT=spike, Pred=spike
    TP = confusion_matrix.loc['spike', 'spike'] if 'spike' in confusion_matrix.index and 'spike' in confusion_matrix.columns else 0
    
    # TN: GT=noise, Pred=noise（包括predicted=unmatch的情况）
    TN = confusion_matrix.loc['noise', 'noise'] if 'noise' in confusion_matrix.index and 'noise' in confusion_matrix.columns else 0
    
    # FP: GT=noise, Pred=spike（GT=noise被误判为spike，且匹配到了train neuron）
    FP = confusion_matrix.loc['noise', 'spike'] if 'noise' in confusion_matrix.index and 'spike' in confusion_matrix.columns else 0
    
    # FN: GT=spike, Pred=noise（GT=spike但被预测为unmatch）
    FN = confusion_matrix.loc['spike', 'noise'] if 'spike' in confusion_matrix.index and 'noise' in confusion_matrix.columns else 0
    
    # 计算各项指标
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    metrics = {
        'confusion_matrix': confusion_matrix,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'total_samples': total,
    }
    
    return metrics

