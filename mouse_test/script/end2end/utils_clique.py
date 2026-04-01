"""
AutoSort training utility functions
Includes: threshold detection, data preparation, model definition and training functions
"""

import numpy as np
# label helper for PSTH classification models
def create_label_mapping_from_classes(class_labels, unique_classes):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
    mapped_labels = np.array([class_to_idx[class_name] for class_name in class_labels])
    return mapped_labels, class_to_idx

import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import scipy.signal
from scipy.spatial import ConvexHull
from typing import Dict, Iterable, List, Sequence, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, f1_score

import spikeinterface.extractors as se


# ==================== 0. Clique Building ====================

@dataclass
class CliqueInfo:
    """Information about a clique (subset of channels)"""
    clique_id: int
    device_channel_indices: List[int]
    contact_ids: List[str]
    center: Tuple[float, float]


def build_probe_group(probe_template_path: str = None):
    """
    Build probe group from MEArec template or return existing probe group.
    
    Parameters:
        probe_template_path: Path to MEArec recording file for probe template.
                            If None, uses default path.
    
    Returns:
        probegroup: ProbeGroup object with device_channel_indices set
    """
    if probe_template_path is None:
        probe_template_path = '/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/02_Neuropixel_384_channels/data_generation/recording_neuropixels_type1.h5'
    
    print("[INFO] Loading probe template")
    template_recording = se.MEArecRecordingExtractor(file_path=str(probe_template_path))
    probegroup = template_recording.get_probegroup()
    offset = 0
    for probe in probegroup.probes:
        n_contacts = probe.get_contact_count()
        device_indices = np.arange(offset, offset + n_contacts, dtype=int)
        probe.set_device_channel_indices(device_indices)
        offset += n_contacts
    return probegroup


def build_sliding_cliques(
    probe,
    clique_size: int = 50,
    min_size: int = 25,
    min_overlap: int = 16,
    target_groups: int = 11,
) -> List[CliqueInfo]:
    """
    Build sliding cliques from probe.
    
    Parameters:
        probe: Probe or ProbeGroup object
        clique_size: Size of each clique (number of channels), default 50
        min_size: Minimum size for a clique to be valid, default 25
        min_overlap: Minimum overlap between consecutive cliques, default 16
        target_groups: Target number of cliques to generate, default 11
    
    Returns:
        cliques: List of CliqueInfo objects
    """
    df = probe.to_dataframe()
    if "device_channel_indices" in df.columns:
        device_indices = df["device_channel_indices"].astype(int).to_numpy()
    else:
        device_indices = np.arange(len(df), dtype=int)
    positions = df.loc[:, ["x", "y"]].to_numpy()
    contact_ids = df["contact_ids"].astype(str).to_numpy()

    # Sort by y-coordinate (vertical position)
    order = np.argsort(positions[:, 1])
    ordered_device = device_indices[order]
    ordered_contacts = contact_ids[order]
    ordered_positions = positions[order]

    # Calculate step size for sliding window
    step = clique_size - min_overlap
    cliques: List[CliqueInfo] = []

    # Generate start indices for sliding windows
    start_indices = list(range(0, len(ordered_device) - clique_size + 1, step))
    if start_indices[-1] + clique_size < len(ordered_device):
        start_indices.append(len(ordered_device) - clique_size)

    # Build cliques
    for idx, start in enumerate(start_indices[:target_groups]):
        slice_device = ordered_device[start : start + clique_size]
        slice_positions = ordered_positions[start : start + clique_size]
        slice_contacts = ordered_contacts[start : start + clique_size]
        if len(slice_device) < min_size:
            continue
        center = tuple(np.mean(slice_positions, axis=0))
        cliques.append(
            CliqueInfo(
                clique_id=idx,
                device_channel_indices=list(slice_device),
                contact_ids=list(slice_contacts),
                center=center,
            )
        )

    print(f"[INFO] Built {len(cliques)} cliques (target {target_groups})")
    for info in cliques:
        print(
            f"       Clique {info.clique_id:02d}: channels {info.device_channel_indices[0]}-"
            f"{info.device_channel_indices[-1]} ({len(info.device_channel_indices)} channels)"
        )

    return cliques

def build_shank_cliques(probe, shank_boundaries=[250, 750, 1250]):
    """
    根据x坐标划分shank并构建cliques
    
    Parameters:
        probe: Probe对象
        shank_boundaries: shank之间的x坐标边界，默认[250, 750, 1250]
                         将probe划分为4个shank:
                         - shank 0: x < 250
                         - shank 1: 250 <= x < 750
                         - shank 2: 750 <= x < 1250
                         - shank 3: x >= 1250
    
    Returns:
        cliques: List[CliqueInfo] - 每个shank对应一个clique
    """
    from typing import List
    
    df = probe.to_dataframe()
    if "device_channel_indices" in df.columns:
        device_indices = df["device_channel_indices"].astype(int).to_numpy()
    else:
        device_indices = np.arange(len(df), dtype=int)
    positions = df.loc[:, ["x", "y"]].to_numpy()
    contact_ids = df["contact_ids"].astype(str).to_numpy()
    
    # 根据x坐标划分shank
    x_coords = positions[:, 0]
    shank_boundaries_sorted = sorted(shank_boundaries)
    
    cliques: List[CliqueInfo] = []
    
    # 定义shank范围
    shank_ranges = [
        (float('-inf'), shank_boundaries_sorted[0]),  # shank 0: x < 250
        (shank_boundaries_sorted[0], shank_boundaries_sorted[1]),  # shank 1: 250 <= x < 750
        (shank_boundaries_sorted[1], shank_boundaries_sorted[2]),  # shank 2: 750 <= x < 1250
        (shank_boundaries_sorted[2], float('inf')),  # shank 3: x >= 1250
    ]
    
    for shank_id, (x_min, x_max) in enumerate(shank_ranges):
        # 找到属于当前shank的通道
        if x_min == float('-inf'):
            mask = x_coords < x_max
        elif x_max == float('inf'):
            mask = x_coords >= x_min
        else:
            mask = (x_coords >= x_min) & (x_coords < x_max)
        
        shank_device_indices = device_indices[mask]
        shank_contact_ids = contact_ids[mask]
        shank_positions = positions[mask]
        
        if len(shank_device_indices) == 0:
            print(f"[WARNING] Shank {shank_id} has no channels")
            continue
        
        # 计算shank的中心位置
        center = tuple(np.mean(shank_positions, axis=0))
        
        # 创建CliqueInfo对象
        clique = CliqueInfo(
            clique_id=shank_id,
            device_channel_indices=list(shank_device_indices),
            contact_ids=list(shank_contact_ids),
            center=center,
        )
        cliques.append(clique)
        
        print(f"[INFO] Shank {shank_id}: {len(shank_device_indices)} channels "
              f"(x range: {x_min if x_min != float('-inf') else 'min'} to "
              f"{x_max if x_max != float('inf') else 'max'})")
    
    print(f"[INFO] Built {len(cliques)} cliques from {len(shank_boundaries) + 1} shanks")
    return cliques
    
def plot_cliques(probe, cliques: List[CliqueInfo], output_pdf_path: str = 'cliques_visualization.pdf'):
    """
    Plot cliques visualization similar to plot_channel_groups.
    
    Parameters:
    -----------
    probe : Probe object
        Probe object containing channel position information
    cliques : List[CliqueInfo]
        List of CliqueInfo objects to visualize
    output_pdf_path : str
        Path to save the visualization PDF, default 'cliques_visualization.pdf'
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Get channel positions from probe dataframe
    df = probe.to_dataframe()
    if "device_channel_indices" in df.columns:
        device_indices = df["device_channel_indices"].astype(int).to_numpy()
    else:
        device_indices = np.arange(len(df), dtype=int)
    positions = df.loc[:, ["x", "y"]].to_numpy()
    
    channel_x = positions[:, 0]
    channel_y = positions[:, 1]
    
    # Calculate rectangle size based on channel spacing
    if len(channel_x) > 1:
        x_spacing = np.min(np.diff(np.sort(np.unique(channel_x)))) if len(np.unique(channel_x)) > 1 else 20
        y_spacing = np.min(np.diff(np.sort(np.unique(channel_y)))) if len(np.unique(channel_y)) > 1 else 20
        # Electrodes should be horizontal (width > height)
        rect_width = x_spacing * 0.8  # Larger width
        rect_height = y_spacing * 0.6  # Smaller height
    else:
        rect_width = 20
        rect_height = 8
    
    n_groups = len(cliques)
    # Adjust figsize: height greater than width, each subplot width smaller
    fig_width = 3 * n_groups  # Each subplot width 3
    fig_height = 12  # Total height 12
    fig, axes = plt.subplots(1, n_groups, figsize=(fig_width, fig_height))
    
    # Reduce spacing between subplots
    plt.subplots_adjust(left=0.002, right=0.98, top=0.95, bottom=0.05, wspace=0.02)
    
    if n_groups == 1:
        axes = [axes]
    
    # Collect all clique channels
    all_clique_channels = set()
    for clique in cliques:
        all_clique_channels.update(clique.device_channel_indices)
    
    # Calculate axis range (add margin)
    x_min, x_max = channel_x.min(), channel_x.max()
    y_min, y_max = channel_y.min(), channel_y.max()
    # Expand x direction range by 2x
    x_range = x_max - x_min
    x_center = (x_max + x_min) / 2
    x_min_expanded = x_center - x_range
    x_max_expanded = x_center + x_range
    x_margin = 0  # No additional margin needed
    y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 50
    
    for idx, clique in enumerate(cliques):
        ax = axes[idx]
        
        clique_channel_indices = set(clique.device_channel_indices)
        
        red_count = 0
        gray_count = 0
        for ch_idx, (x, y) in enumerate(zip(channel_x, channel_y)):
            # Find the device index for this channel
            device_idx = device_indices[ch_idx] if ch_idx < len(device_indices) else ch_idx
            
            if device_idx in clique_channel_indices:
                color = '#ED7B85'  # Red
                alpha = 0.9
                red_count += 1
            else:
                color = 'lightgrey'
                alpha = 0.4
                gray_count += 1
            
            rect = Rectangle((x - rect_width/2, y - rect_height/2), 
                            rect_width, rect_height,
                            facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        ax.set_xlim(x_min_expanded, x_max_expanded)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Clique {clique.clique_id:02d}\n({len(clique.device_channel_indices)} channels)', 
                    fontsize=10, fontweight='bold', pad=5)
        ax.grid(False)  
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Save as PDF (don't use tight_layout, already adjusted with subplots_adjust)
    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
    
    print(f"\nClique可视化PDF已保存至: {output_pdf_path}")
    plt.close()


def load_clique_info(clique_info_path: str) -> dict:
    """
    Load saved clique information for evaluation.
    
    Parameters:
    -----------
    clique_info_path : str
        Path to the saved clique_info.pkl file
    
    Returns:
    --------
    clique_info : dict
        Dictionary containing:
        - 'cliques': List of CliqueInfo objects
        - 'clique_params': Dictionary of clique building parameters
        - 'probe_df': Probe dataframe
        - 'recording_path': Recording path used during training
    """
    import pickle
    
    with open(clique_info_path, 'rb') as f:
        clique_info = pickle.load(f)
    
    print(f"Loaded clique information from: {clique_info_path}")
    print(f"  - Number of cliques: {len(clique_info['cliques'])}")
    print(f"  - Clique parameters: {clique_info['clique_params']}")
    
    return clique_info


# ==================== 1. Threshold Detection ====================
def detect_spike(
    trace0_car,
    thr_min=3,
    thr_max=30,
    distance=3,
    wlen=5,
    prominence=10,
    valid_channels=None,
    max_firing_channel=None,
):
    """
    AutoSort threshold detection function (identical to detection.py)
    
    Parameters:
        trace0_car: numpy array, shape (n_timepoints, n_channels)
        thr_min: minimum threshold multiplier (relative to noise std), default 3
        thr_max: maximum threshold multiplier (for filtering outliers), default 30
        distance: minimum distance between peaks (samples), default 3
        wlen: window length for peak detection, default 5
        prominence: minimum peak prominence, default 10
        valid_channels: list of valid channel indices to detect on, if None detect on all channels
        max_firing_channel: maximum number of channels that can fire simultaneously, if None no filtering
    
    Returns:
        spikes: binary matrix (n_timepoints, n_channels), 1 indicates detected spike
    """
    noise_std_detect = np.median(abs(trace0_car) / 0.6745, axis=0)
    thr = thr_min * noise_std_detect
    thrmax = thr_max * noise_std_detect

    spikes = np.zeros(trace0_car.shape)
    if trace0_car.ndim > 1:
        # Determine which channels to process
        channels_to_process = range(noise_std_detect.shape[0])
        if valid_channels is not None:
            # Only process channels in valid_channels list
            channels_to_process = [ch for ch in channels_to_process if ch in valid_channels]
        
        for i in channels_to_process:
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

        # Filter simultaneous firing: no more than max_firing_channel channels firing at the same time
        if max_firing_channel is not None:
            # For each time point, if more than max_firing_channel channels have spikes, remove all spikes at that time
            spikes[np.sum(spikes, axis=1) > max_firing_channel, :] = 0

    return spikes

def map_gt_annotation(detect_array, gt_array):
    """
    AutoSort GT mapping function (vectorized optimized version, logic identical to detection.py)
    
    Parameters:
        detect_array: numpy array, shape (n_detected, 2), each row is [time_point, channel_id]
        gt_array: numpy array, shape (n_gt, 2), each row is [time_point, channel_id]
    
    Returns:
        gt_label_array1: numpy array, shape (n_detected,), values are corresponding GT indices or -1 (unmatched)
    """
    n_detected = detect_array.shape[0]
    gt_label_array1 = np.full(n_detected, -1, dtype=np.int64)
    
    if n_detected == 0 or gt_array.shape[0] == 0:
        return gt_label_array1
    
    # Extract detected times and channels
    detect_times = detect_array[:, 0].astype(np.int64)
    detect_channels = detect_array[:, 1].astype(np.int64)
    
    # Extract GT times and channels
    gt_times = gt_array[:, 0].astype(np.int64)
    gt_channels = gt_array[:, 1].astype(np.int64)
    
    # Use dictionary to speed up lookup: key = (time, channel), value = GT index list
    gt_dict = defaultdict(list)
    for idx, (t, c) in enumerate(zip(gt_times, gt_channels)):
        gt_dict[(t, c)].append(idx)
    
    # Try to match each detected spike with three time offsets: 0, -1, +1 (in priority order)
    time_offsets = [0, -1, 1]
    
    # Vectorized matching: for each time offset, batch process all unmatched detected spikes
    for offset in time_offsets:
        # Find unmatched detected spikes
        unmatched_mask = gt_label_array1 == -1
        if not np.any(unmatched_mask):
            break
        
        # Calculate shifted times (only for unmatched ones)
        unmatched_indices = np.where(unmatched_mask)[0]
        shifted_times = detect_times[unmatched_indices] + offset
        unmatched_channels = detect_channels[unmatched_indices]
        
        # Vectorized lookup: use dictionary for fast matching (O(1) lookup)
        keys = [(shifted_times[i], unmatched_channels[i]) for i in range(len(unmatched_indices))]
        
        # Batch lookup matching (avoid loop one by one)
        for i, key in enumerate(keys):
            if key in gt_dict and len(gt_dict[key]) > 0:
                # Found match, use first matching GT index
                gt_idx = gt_dict[key][0]
                detect_idx = unmatched_indices[i]
                gt_label_array1[detect_idx] = gt_idx
                # Remove matched item from dictionary (avoid duplicate matching)
                gt_dict[key].pop(0)
                if len(gt_dict[key]) == 0:
                    del gt_dict[key]
    
    return gt_label_array1


def extract_waveforms(trace0_car, X_spiketrain_time, left_sample=10, right_sample=20):
    """
    Extract waveform window (following AutoSort method)
    
    Parameters:
        trace0_car: numpy array, shape (n_timepoints, n_channels)
        X_spiketrain_time: numpy array, shape (n_spikes,), spike time points
        left_sample: number of samples before spike, default 10
        right_sample: number of samples after spike, default 20
    
    Returns:
        waveform: numpy array, shape (n_spikes, n_channels, window_length)
    """
    # Filter spikes near boundaries (ensure complete window can be extracted)
    valid_mask = X_spiketrain_time < trace0_car.shape[0] - (left_sample + right_sample)
    X_spiketrain_time = X_spiketrain_time[valid_mask]
    
    # Extract window following AutoSort method
    for time_range in tqdm(np.arange(-left_sample, right_sample), desc="Extracting waveforms"):
        if time_range == -left_sample:
            # First time point, initialize waveform
            waveform = trace0_car[X_spiketrain_time + time_range, :]
        else:
            # Subsequent time points, stack using dstack
            waveform = np.dstack(
                (waveform, trace0_car[X_spiketrain_time + time_range, :])
            )
    
    # waveform shape: (n_spikes, n_channels, window_length)
    return waveform, valid_mask


def neuron_inf_dict_to_dataframe(neuron_inf_dict: dict) -> pd.DataFrame:
    """
    将dict格式的neuron_inf转换为DataFrame
    
    Parameters:
        neuron_inf_dict: dict, 键为unit_id，值为包含location_x, location_y, position_waveform, extremum_channel, channel_id, channel_snr的dict
    
    Returns:
        neuron_inf_df: pd.DataFrame, 包含列：Neuron, position_1, position_2, position_waveform, extremum_channel, channel_id, channel_snr
    """
    rows = []
    for unit_id, neuron_data in neuron_inf_dict.items():
        row = {
            'Neuron': unit_id,
            'position_1': neuron_data.get('location_x', 0.0),
            'position_2': neuron_data.get('location_y', 0.0),
            'position_waveform': neuron_data.get('position_waveform', np.array([])),
            'extremum_channel': neuron_data.get('extremum_channel', None)
        }
        # 添加channel_id列（如果存在）
        if 'channel_id' in neuron_data:
            row['channel_id'] = neuron_data['channel_id']
        # 添加channel_snr列（如果存在）
        if 'channel_snr' in neuron_data:
            row['channel_snr'] = neuron_data['channel_snr']
        rows.append(row)
    
    neuron_inf_df = pd.DataFrame(rows)
    return neuron_inf_df


def filter_channels_by_snr(neuron_inf: pd.DataFrame, thr_min: float) -> pd.DataFrame:
    """
    根据SNR阈值过滤每个neuron的channel_id
    
    Parameters:
        neuron_inf: pd.DataFrame, 包含Neuron, channel_id, channel_snr列
        thr_min: float, SNR阈值，只保留SNR > thr_min的channels
    
    Returns:
        neuron_inf_filtered: pd.DataFrame, 更新后的neuron_inf，channel_id只保留SNR > thr_min的channels
    """
    neuron_inf_filtered = neuron_inf.copy()
    
    for idx, row in neuron_inf_filtered.iterrows():
        channel_id = row.get('channel_id', [])
        channel_snr = row.get('channel_snr', {})
        
        # 如果没有channel_snr或channel_snr为空，保留所有原始channel_id
        if not channel_snr or not isinstance(channel_snr, dict) or len(channel_snr) == 0:
            continue
        
        # 如果没有channel_id，跳过
        if not channel_id or len(channel_id) == 0:
            continue
        
        # 过滤channel_id：只保留SNR > thr_min的channels
        filtered_channel_id = []
        for ch in channel_id:
            # 如果channel_id在channel_snr字典中，检查SNR
            if ch in channel_snr:
                snr_value = channel_snr[ch]
                if snr_value > thr_min:
                    filtered_channel_id.append(ch)
            else:
                # 如果channel_id不在channel_snr字典中，默认保留（向后兼容）
                filtered_channel_id.append(ch)
        
        # 更新channel_id
        neuron_inf_filtered.at[idx, 'channel_id'] = filtered_channel_id
    
    return neuron_inf_filtered


def get_recording_clique(recording, clique: CliqueInfo):
    """
    根据clique获取recording_clique
    使用recording的channel_ids（统一命名，如"A-000"）来选择通道
    
    Parameters:
        recording: recording对象
        clique: CliqueInfo对象，包含device_channel_indices
    
    Returns:
        recording_clique: 只包含clique通道的recording对象
    """
    # 获取recording的所有channel IDs（统一命名，如"A-000", "A-001"等）
    all_channel_ids = recording.get_channel_ids()
    
    # 使用device_channel_indices（整数索引）来选择通道
    # device_channel_indices对应recording的channel顺序
    clique_channel_indices = clique.device_channel_indices
    selected_channel_ids = [all_channel_ids[idx] for idx in clique_channel_indices if idx < len(all_channel_ids)]
    recording_clique = recording.select_channels(selected_channel_ids)
    return recording_clique


def filter_neuron_inf_by_clique(neuron_inf: pd.DataFrame, recording_clique) -> pd.DataFrame:
    """
    筛选neuron_inf：extremum_channel在recording_clique.get_channel_ids()中的neuron
    
    Parameters:
        neuron_inf: pd.DataFrame, 包含Neuron和extremum_channel列
        recording_clique: recording对象（clique子集）
    
    Returns:
        neuron_inf_clique: pd.DataFrame, 筛选后的neuron_inf
    """
    if 'extremum_channel' not in neuron_inf.columns:
        raise ValueError("neuron_inf must contain 'extremum_channel' column")
    
    recording_channel_ids = set(recording_clique.get_channel_ids())
    
    # 筛选extremum_channel在recording_clique中的neuron
    mask = neuron_inf['extremum_channel'].isin(recording_channel_ids)
    neuron_inf_clique = neuron_inf[mask].copy()
    
    return neuron_inf_clique


def filter_gt_detect_array_by_clique(gt_detect_array: pd.DataFrame, recording_clique) -> pd.DataFrame:
    """
    筛选gt_detect_array：extremum_channel在recording_clique.get_channel_ids()中的spikes
    
    Parameters:
        gt_detect_array: pd.DataFrame, 包含time, unit_id, extremum_channel列
        recording_clique: recording对象（clique子集）
    
    Returns:
        gt_detect_array_clique: pd.DataFrame, 筛选后的gt_detect_array
    """
    if 'extremum_channel' not in gt_detect_array.columns:
        raise ValueError("gt_detect_array must contain 'extremum_channel' column")
    
    recording_channel_ids = set(recording_clique.get_channel_ids())
    
    # 筛选extremum_channel在recording_clique中的spikes
    mask = gt_detect_array['extremum_channel'].isin(recording_channel_ids)
    gt_detect_array_clique = gt_detect_array[mask].copy()
    
    return gt_detect_array_clique


def deduplicate_spikes_by_neuron_channels(
    detect_array: np.ndarray,
    neuron_inf: pd.DataFrame,
    recording_channel_ids: List[str],
    trace0_car: np.ndarray
) -> np.ndarray:
    """
    对detect_array进行去重：对于每个neuron，如果其extremum_channel和SNR第二大的channel
    在同一个时间点都检测到了spike，保留spike幅值更大的那个channel上的spike
    
    Parameters:
        detect_array: numpy array, shape (n_detected, 2), each row is [time_point, channel_index]
                      channel_index是clique内的列索引（0, 1, 2, ..., n_clique_channels-1）
        neuron_inf: pd.DataFrame, 包含Neuron, extremum_channel, channel_snr列
        recording_channel_ids: list of channel IDs (strings), 用于将channel ID映射到clique索引
        trace0_car: numpy array, shape (n_timepoints, n_channels), 原始trace数据，用于获取spike幅值
    
    Returns:
        deduplicated_detect_array: numpy array, shape (n_dedup, 2), 去重后的detect_array
    """
    if len(detect_array) == 0:
        return detect_array
    
    if 'extremum_channel' not in neuron_inf.columns:
        print("Warning: neuron_inf缺少extremum_channel列，跳过去重")
        return detect_array
    
    if 'channel_snr' not in neuron_inf.columns:
        print("Warning: neuron_inf缺少channel_snr列，跳过去重")
        return detect_array
    
    # 创建channel ID到clique索引的映射
    channel_id_to_clique_idx = {str(ch_id): idx for idx, ch_id in enumerate(recording_channel_ids)}
    
    # 创建要删除的spike索引集合
    spikes_to_remove = set()
    
    # 对每个neuron进行处理
    for _, neuron_row in neuron_inf.iterrows():
        extremum_channel = neuron_row.get('extremum_channel')
        channel_snr = neuron_row.get('channel_snr', {})
        
        # 跳过没有extremum_channel或channel_snr的neuron
        if pd.isna(extremum_channel) or extremum_channel is None:
            continue
        if not isinstance(channel_snr, dict) or len(channel_snr) == 0:
            continue
        
        # 将extremum_channel映射到clique索引
        extremum_channel_str = str(extremum_channel)
        if extremum_channel_str not in channel_id_to_clique_idx:
            continue
        extremum_channel_idx = channel_id_to_clique_idx[extremum_channel_str]
        
        # 找到SNR第二大的channel（在所有channel中，不排除extremum_channel）
        snr_items = [(ch, snr) for ch, snr in channel_snr.items() 
                     if str(ch) in channel_id_to_clique_idx]
        
        if len(snr_items) < 2:
            continue  # 至少需要2个channel才能有"第二大"
        
        # 按SNR排序，取第二大的
        snr_items_sorted = sorted(snr_items, key=lambda x: x[1], reverse=True)
        second_max_snr_channel = snr_items_sorted[1][0]  # 第二大的（索引1）
        second_max_channel_str = str(second_max_snr_channel)
        
        # 如果SNR第二大的channel就是extremum_channel，跳过（不需要去重）
        if second_max_channel_str == extremum_channel_str:
            continue
        
        if second_max_channel_str not in channel_id_to_clique_idx:
            continue
        second_max_channel_idx = channel_id_to_clique_idx[second_max_channel_str]
        
        # 找到在extremum_channel和second_max_channel上都检测到的spike（相同时间点）
        # 使用向量化操作获取两个channel上的所有spike
        extremum_mask = detect_array[:, 1] == extremum_channel_idx
        second_max_mask = detect_array[:, 1] == second_max_channel_idx
        
        if not np.any(extremum_mask) or not np.any(second_max_mask):
            continue
        
        # 获取两个channel上的时间点和索引
        extremum_indices = np.where(extremum_mask)[0]
        extremum_times = detect_array[extremum_indices, 0]
        second_max_indices = np.where(second_max_mask)[0]
        second_max_times = detect_array[second_max_indices, 0]
        
        # 找到相同时间点的spike（使用numpy的intersect1d）
        common_times = np.intersect1d(extremum_times, second_max_times)
        
        if len(common_times) == 0:
            continue
        
        # 向量化获取所有相同时间点的spike索引
        extremum_common_mask = np.isin(extremum_times, common_times)
        extremum_common_indices = extremum_indices[extremum_common_mask]
        extremum_common_times = extremum_times[extremum_common_mask]
        
        second_max_common_mask = np.isin(second_max_times, common_times)
        second_max_common_indices = second_max_indices[second_max_common_mask]
        second_max_common_times = second_max_times[second_max_common_mask]
        
        # 向量化获取所有spike的幅值
        extremum_amplitudes = np.abs(trace0_car[extremum_common_times.astype(int), extremum_channel_idx])
        second_max_amplitudes = np.abs(trace0_car[second_max_common_times.astype(int), second_max_channel_idx])
        
        # 使用pandas DataFrame进行匹配和比较（完全向量化）
        # 创建两个DataFrame，按时间点匹配
        extremum_df = pd.DataFrame({
            'time': extremum_common_times,
            'index': extremum_common_indices,
            'amplitude': extremum_amplitudes
        })
        second_max_df = pd.DataFrame({
            'time': second_max_common_times,
            'index': second_max_common_indices,
            'amplitude': second_max_amplitudes
        })
        
        # 按时间点合并
        merged_df = pd.merge(extremum_df, second_max_df, on='time', suffixes=('_ext', '_sec'))
        
        if len(merged_df) > 0:
            # 向量化比较幅值
            # 如果extremum_amplitude >= second_max_amplitude，删除second_max的；否则删除extremum的
            remove_second_max = merged_df['amplitude_ext'] >= merged_df['amplitude_sec']
            indices_to_remove = np.where(remove_second_max, 
                                        merged_df['index_sec'].values, 
                                        merged_df['index_ext'].values)
            spikes_to_remove.update(indices_to_remove.tolist())
    
    # 删除需要移除的spike
    if len(spikes_to_remove) > 0:
        keep_indices = [i for i in range(len(detect_array)) if i not in spikes_to_remove]
        deduplicated_detect_array = detect_array[keep_indices]
        print(f"去重: 移除了{len(spikes_to_remove)}个spikes（保留幅值更大的channel上的spike）")
        print(f"去重前: {len(detect_array)}个spikes, 去重后: {len(deduplicated_detect_array)}个spikes")
        return deduplicated_detect_array
    else:
        print("去重: 没有找到需要移除的spikes")
        return detect_array




def prepare_training_data(
    recording_f,
    gt_detect_array,
    neuron_inf,
    save_dir,
    duration_seconds=200,
    thr_min=3.5,
    thr_max=30,
    distance=3,
    wlen=5,
    prominence=10,
    left_sample=10,
    right_sample=20,
    max_firing_channel=None,
):
    """
    Prepare training data (complete pipeline: detection -> matching -> waveform extraction -> saving)
    
    This function performs clique-level detection. It receives a recording_clique (subset of channels)
    and performs detection, matching, and waveform extraction on this clique.
    
    Parameters:
        recording_f: preprocessed recording object (should be recording_clique for clique-level processing)
        gt_detect_array: DataFrame containing GT spike information (filtered to neurons in clique), 
                        must have 'time', 'unit_id', and 'extremum_channel' columns.
                        'time' should be in seconds, will be converted to sample indices.
        neuron_inf: DataFrame containing neuron information (filtered to neurons in clique), 
                   must have 'Neuron' and 'extremum_channel' columns.
                   valid_channels will be computed from the union of extremum_channels in this DataFrame.
        save_dir: save directory path
        duration_seconds: processing duration (seconds), default 200
        thr_min, thr_max, distance, wlen, prominence: detection parameters
        left_sample, right_sample: waveform window parameters
    
    Returns:
        train_data_dir: training data save directory
    """
    print("### 1. Threshold Detection")
    
    # Get recording sampling rate and number of channels
    sampling_rate = recording_f.get_sampling_frequency()
    n_channels = recording_f.get_num_channels()
    print(f"Sampling rate: {sampling_rate} Hz, Number of channels: {n_channels}")
    
    # Calculate corresponding number of samples
    max_frames = int(duration_seconds * sampling_rate)
    total_frames = recording_f.get_num_frames()
    actual_frames = min(max_frames, total_frames)
    
    recording_channel_ids = recording_f.get_channel_ids()  # These can be strings (e.g., "B-000") or integers
    probe_to_clique_index = {}  # Map original probe channel ID -> clique column index
    for clique_idx, probe_ch in enumerate(recording_channel_ids):
        # Handle both string and integer channel IDs
        probe_to_clique_index[probe_ch] = clique_idx
    
    # Read data from clique recording
    trace0_car = recording_f.get_traces(start_frame=0, end_frame=actual_frames).astype(np.float32)
    print(f"Data shape: {trace0_car.shape} (clique channels)")
    
    if 'extremum_channel' in neuron_inf.columns and len(neuron_inf) > 0:
        # Extract all unique extremum_channels and convert to clique column indices
        unique_extremum_channels = neuron_inf['extremum_channel'].dropna().unique()
        valid_channels = []
        for ch in unique_extremum_channels:
            if ch in probe_to_clique_index:
                valid_channels.append(probe_to_clique_index[ch])
        valid_channels = sorted(set(valid_channels))  # Remove duplicates and sort
    else:
        valid_channels = None  # Use all channels if neuron_inf is empty or missing extremum_channel column
        print("Using all channels for detection (neuron_inf is empty or missing extremum_channel column)")

    spikes = detect_spike(
        trace0_car,
        thr_min=thr_min,
        thr_max=thr_max,
        distance=distance,
        wlen=wlen,
        prominence=prominence,
        valid_channels=valid_channels,  # Use extremum_channels if specified
        max_firing_channel=max_firing_channel,
    )
    
    # Build detect_array using clique column indices (0, 1, 2, ..., n_clique_channels-1)
    print("Building detect_array...")
    all_spike_train = []
    spike_loc = []
    for channel_idx in range(trace0_car.shape[1]):
        spiketrain_loc = np.where(spikes[:, channel_idx])[0]
        if len(spiketrain_loc) > 0:
            all_spike_train += list(spiketrain_loc)
            spike_loc += [channel_idx] * len(spiketrain_loc)  # Clique column index
    
    X_spiketrain_time = np.array(all_spike_train)
    Y_spiketrain_id_final = np.array(spike_loc)
    detect_array = np.array([X_spiketrain_time, Y_spiketrain_id_final]).T

    print(f"Number of detected spikes: {len(detect_array)}")
    
    # Deduplicate spikes: for each neuron, if both extremum_channel and second-max-SNR channel
    # detect spikes at the same time point, keep the spike with larger amplitude
    if 'extremum_channel' in neuron_inf.columns and 'channel_snr' in neuron_inf.columns:
        recording_channel_ids = list(recording_f.get_channel_ids())
        detect_array = deduplicate_spikes_by_neuron_channels(
            detect_array, neuron_inf, recording_channel_ids, trace0_car
        )
        # Rebuild X_spiketrain_time and Y_spiketrain_id_final from deduplicated detect_array
        X_spiketrain_time = detect_array[:, 0]
        Y_spiketrain_id_final = detect_array[:, 1]
    
    print(f"Number of detected spikes after deduplication: {len(detect_array)}")
    
    print("\n### 2. Load Ground Truth and Match")
    
    # Build gt_array directly from gt_detect_array
    # gt_detect_array has columns: time (sample points, not seconds!), unit_id, extremum_channel
    # Note: gt_detect_array['time'] is already in sample points (not seconds)
    sampling_rate = recording_f.get_sampling_frequency()
    gt_detect_array_filtered = gt_detect_array[gt_detect_array['time'] < max_frames].copy()
    
    # Build mapping from extremum_channel to clique column index
    spike_train_all = []
    y_unit_id = []
    gt_ch = []
    skipped_count = 0
    skipped_reasons = {'na': 0, 'not_in_mapping': 0}
    
    for pos_idx, (_, row) in enumerate(gt_detect_array_filtered.iterrows()):
        extremum_channel = row['extremum_channel']
        unit_id = row['unit_id']
        spike_time_sample_points = row['time']  # Already in sample points
        
        # Time is already in sample points, just convert to int
        spike_time_sample = int(spike_time_sample_points)
        
        # Map extremum_channel (original probe channel ID) to clique column index
        if pd.isna(extremum_channel) or extremum_channel is None:
            skipped_reasons['na'] += 1
            continue
        
        # Convert extremum_channel to string if it's not already, to ensure type matching
        extremum_channel_str = str(extremum_channel)
        
        # Try both the original value and string version
        if extremum_channel not in probe_to_clique_index and extremum_channel_str not in probe_to_clique_index:
            skipped_reasons['not_in_mapping'] += 1
            skipped_count += 1
            continue
        
        # Use whichever version is in the mapping
        if extremum_channel in probe_to_clique_index:
            clique_channel_index = probe_to_clique_index[extremum_channel]
        else:
            clique_channel_index = probe_to_clique_index[extremum_channel_str]
        
        spike_train_all.append(spike_time_sample)
        y_unit_id.append(unit_id)
        gt_ch.append(clique_channel_index)  # 使用clique列索引
    
    gt_array = np.array([spike_train_all, gt_ch]).T if len(spike_train_all) > 0 else np.array([]).reshape(0, 2)
    
    # Use AutoSort's map_gt_annotation function
    gt_label_array1 = map_gt_annotation(detect_array, gt_array)
    
    # Calculate GT matching statistics (same format as calibration_model)
    matched_gt_indices = set(gt_label_array1[gt_label_array1 >= 0])
    matched_gt_count = len(matched_gt_indices)
    gt_total_count = len(gt_array)
    recall_rate = matched_gt_count / gt_total_count if gt_total_count > 0 else 0
    print(f"GT匹配统计: {matched_gt_count}/{gt_total_count} GT spikes被检测到 (召回率: {recall_rate:.4f})")
    
    # 找出未匹配的GT spikes，并将它们添加到detect_array中
    unmatched_gt_indices = set(range(gt_total_count)) - matched_gt_indices
    if len(unmatched_gt_indices) > 0:
        print(f"补全未匹配的GT spikes: {len(unmatched_gt_indices)}个")
        unmatched_gt_array = gt_array[list(unmatched_gt_indices)]
        
        # 检查边界条件，确保可以提取waveform
        left_sample = left_sample
        right_sample = right_sample
        valid_unmatched_mask = (
            (unmatched_gt_array[:, 0] >= left_sample) & 
            (unmatched_gt_array[:, 0] < trace0_car.shape[0] - right_sample)
        )
        valid_unmatched_gt = unmatched_gt_array[valid_unmatched_mask]
        valid_unmatched_indices = np.array(list(unmatched_gt_indices))[valid_unmatched_mask]
        
        if len(valid_unmatched_gt) > 0:
            # 将未匹配的GT spikes添加到detect_array
            detect_array = np.vstack([detect_array, valid_unmatched_gt])
            print(f"  添加了{len(valid_unmatched_gt)}个未匹配的GT spikes到detect_array")
            
            # 更新y_unit_id数组，为未匹配的GT spikes添加对应的unit_id
            unmatched_y_unit_id = [y_unit_id[i] for i in valid_unmatched_indices]
        else:
            print(f"  警告: 所有未匹配的GT spikes都在边界附近，无法提取waveform，跳过")
            unmatched_y_unit_id = []
    else:
        unmatched_y_unit_id = []
    
    # Build Y_spiketrain_id（在添加未匹配GT spikes之后，detect_array已经更新）
    Y_spiketrain_id = np.full((detect_array.shape[0],), None, dtype=object)
    original_detect_array_size = len(gt_label_array1)  # 原始detect_array的长度
    
    # 为已匹配的spikes设置标签
    matched_indices = np.where(gt_label_array1 > -1)[0]
    if len(matched_indices) > 0:
        y_unit_id_array = np.array(y_unit_id, dtype=object)
        Y_spiketrain_id[matched_indices] = y_unit_id_array[
            gt_label_array1[matched_indices].astype("int")
        ]
    
    # 为未匹配的GT spikes设置标签
    if len(unmatched_y_unit_id) > 0:
        unmatched_start_idx = original_detect_array_size
        unmatched_end_idx = unmatched_start_idx + len(unmatched_y_unit_id)
        Y_spiketrain_id[unmatched_start_idx:unmatched_end_idx] = unmatched_y_unit_id
        print(f"  为{len(unmatched_y_unit_id)}个未匹配的GT spikes设置了标签")
    
    # 更新X_spiketrain_time和Y_spiketrain_id_final（detect_array已经包含未匹配的GT spikes）
    X_spiketrain_time = detect_array[:, 0]
    Y_spiketrain_id_final = detect_array[:, 1]
    
    # 输出补全后的统计信息
    if len(unmatched_y_unit_id) > 0:
        final_gt_coverage = (matched_gt_count + len(unmatched_y_unit_id)) / gt_total_count if gt_total_count > 0 else 0
        print(f"补全后GT覆盖率: {matched_gt_count + len(unmatched_y_unit_id)}/{gt_total_count} GT spikes ({final_gt_coverage:.4f})")
        print(f"  - 检测到的GT spikes: {matched_gt_count}")
        print(f"  - 补全的GT spikes: {len(unmatched_y_unit_id)}")
        print(f"  - 总spike数量: {len(detect_array)} (原始: {original_detect_array_size}, 新增: {len(unmatched_y_unit_id)})")
    
    print("\n### 3. Extract Waveforms")
    
    # Extract waveforms from clique recording
    # trace0_car already contains only clique channels, and detect_array uses clique column indices
    waveform, valid_mask = extract_waveforms(
        trace0_car, X_spiketrain_time, left_sample, right_sample
    )
    
    # Apply valid_mask filter
    X_spiketrain_time = X_spiketrain_time[valid_mask]
    Y_spiketrain_id = Y_spiketrain_id[valid_mask]
    Y_spiketrain_id_final = Y_spiketrain_id_final[valid_mask]
    
    print(f"Waveform extraction completed!")
    print(f"waveform shape: {waveform.shape}")
    
    print("\n### 4. Save Training Data")
    
    # Create save directory
    train_data_dir = Path(save_dir) / "train_data"
    train_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {train_data_dir}")
    
    # Prepare data
    X_waveform = waveform
    
    # Convert Y_spike_id
    unique_neurons = np.unique([x for x in Y_spiketrain_id if x is not None])
    neuron_to_id = {neuron: idx for idx, neuron in enumerate(unique_neurons)}
    neuron_to_id[None] = -1
    
    Y_spike_id = np.array([neuron_to_id.get(x, -1) for x in Y_spiketrain_id])
    Y_spike_id_noise = Y_spiketrain_id_final
    
    # Save neuron name to ID mapping (for neuron matching during evaluation)
    neuron_mapping = {
        'neuron_to_id': neuron_to_id,
        'id_to_neuron': {idx: neuron for neuron, idx in neuron_to_id.items() if neuron is not None},
        'unique_neurons': list(unique_neurons)
    }
    with open(train_data_dir / "neuron_mapping.pkl", "wb") as f:
        pickle.dump(neuron_mapping, f)
    print(f"  ✓ neuron_mapping.pkl saved")
    
    # Save data
    print("Saving data...")
    with open(train_data_dir / "X_waveform.pkl", "wb") as f:
        pickle.dump(X_waveform, f)
    print(f"  ✓ X_waveform.pkl saved")
    
    with open(train_data_dir / "Y_spike_id.pkl", "wb") as f:
        pickle.dump(Y_spike_id, f)
    print(f"  ✓ Y_spike_id.pkl saved")
    
    with open(train_data_dir / "Y_spike_id_noise.pkl", "wb") as f:
        pickle.dump(Y_spike_id_noise, f)
    print(f"  ✓ Y_spike_id_noise.pkl saved")
    
    with open(train_data_dir / "X_spiketrain_time.pkl", "wb") as f:
        pickle.dump(X_spiketrain_time, f)
    print(f"  ✓ X_spiketrain_time.pkl saved")
    
    print(f"\nAll data saved to: {train_data_dir}")
    print(f"Data statistics:")
    print(f"  - Total spike count: {len(X_waveform)}")
    print(f"  - Number of channels: {X_waveform.shape[1]}")
    print(f"  - Window length: {X_waveform.shape[2]}")
    print(f"  - Number of unique units: {len(unique_neurons)}")
    print(f"  - Noise spike count: {np.sum(Y_spike_id == -1)}")
    print(f"  - Valid spike count: {np.sum(Y_spike_id != -1)}")
    
    return train_data_dir


# ==================== 3. Model Definition ====================

class SimpleClassifier(nn.Module):
    """
    Simplified classifier (same as AutoSort's clssimp)
    Input: (B, n_channels + 2, window_size) - concatenated multi-waveform + single-waveform + channel_index
    """
    def __init__(self, n_channels, window_size, num_classes):
        super(SimpleClassifier, self).__init__()
        input_dim = (n_channels + 2) * window_size  # Added 1 channel for channel_index
        
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

        self.way4 = nn.Sequential(
            nn.Linear(100, 30, bias=True),
            nn.BatchNorm1d(30),
            nn.ReLU(inplace=True),
        )

        self.cls = nn.Linear(30, num_classes, bias=True)

    def forward(self, x):
        # x: (B, n_channels + 2, window_size)
        # Flatten to (B, (n_channels + 2) * window_size)
        B = x.shape[0]
        x = x.view(B, -1)
        
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        x = self.way4(x)
        logits = self.cls(x)
        return logits

    def intermediate_forward(self, x):
        # x: (B, n_channels + 2, window_size)
        # Flatten to (B, (n_channels + 2) * window_size)
        B = x.shape[0]
        x = x.view(B, -1)
        
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        return self.way4(x)


class SimpleWaveformLoader(data.Dataset):
    """
    Simplified waveform loader without position information
    Only uses multi-waveform and single-waveform
    """
    def __init__(self, root, shank_channel, Keep_id=None):
        # Load data
        with open(root + "X_waveform.pkl", "rb") as f:
            datafile = pickle.load(f)
        try:
            with open(root + "Y_spike_id.pkl", "rb") as f:
                GT = pickle.load(f)
        except FileNotFoundError:
            GT = np.zeros(datafile.shape[0]) - 1
        with open(root + "Y_spike_id_noise.pkl", "rb") as f:
            channel_id = np.array(pickle.load(f))
        
        # Determine which unit IDs to keep
        if Keep_id is None:
            Keep_id = np.unique(GT)
            Keep_id = list(Keep_id[Keep_id != -1])
            self.keep_id = Keep_id
        else:
            self.keep_id = Keep_id
        
        # Create noise/non-noise labels
        mask = ~np.isin(GT, Keep_id)
        GT = np.array(GT)
        
        GT_binary = np.zeros((GT.shape[0], 2))
        GT_binary[list(mask), 0] = 1  # Noise
        GT_binary[~mask, 1] = 1       # Non-noise
        
        self.GT_unique = Keep_id + [-1]
        self.GT_binary = GT_binary
        
        # Extract single waveform (from maximum amplitude channel)
        self.Img_single = datafile[np.arange(datafile.shape[0]), np.array(channel_id).astype('int'), :]
        
        # Store channel indices (detection channel indices, 0-31 for 32-channel clique)
        self.channel_indices = np.array(channel_id).astype('int')
        
        self.GT_LIST = GT
        
        # Create unit classification labels (one-hot)
        GT_array = np.zeros((len(GT), len(Keep_id)))
        for idx, unique_id in enumerate(Keep_id):
            rmv_list = np.where(np.array(GT) == unique_id)[0]
            GT_array[rmv_list, idx] = 1
        self.GT = GT_array
        
        self.Img = datafile  # Multi-channel waveform
        
        # Calculate class weights (for handling imbalanced data)
        self.pos_weight_noise = torch.tensor([
            -np.sum(self.GT_binary[:,0]-1)/np.sum(self.GT_binary[:,0]),
            -np.sum(self.GT_binary[:,1]-1)/np.sum(self.GT_binary[:,1])
        ])
        self.pos_weight_label = torch.tensor([
            -(np.sum(self.GT[:,i]-1)+sum(np.sum(GT_array,axis=1)==0))/np.sum(self.GT[:,i]) 
            for i in range(self.GT.shape[1])
        ])
        
        self.n_classes = len(set(self.GT_unique))
        
        print(f"Dataset loaded:")
        print(f"  - Total samples: {len(self.GT)}")
        print(f"  - Number of channels: {self.Img.shape[1]}")
        print(f"  - Window length: {self.Img.shape[2]}")
        print(f"  - Number of unique units: {len(Keep_id)}")
        print(f"  - Noise samples: {np.sum(self.GT_binary[:, 0])}")
        print(f"  - Non-noise samples: {np.sum(self.GT_binary[:, 1])}")
    
    def __len__(self):
        return len(self.GT)
    
    def __getitem__(self, index):
        # Returns: multi-channel waveform, unit classification labels, noise/non-noise labels, single-channel waveform, channel_index
        return (
            self.Img[index, ...],      # (n_channels, window_length)
            self.GT[index, ...],       # (n_units,) one-hot
            self.GT_binary[index, ...], # (2,) [noise, spike]
            self.Img_single[index, ...], # (window_length,)
            self.channel_indices[index]  # scalar: detection channel index (0-31)
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Parameters:
        alpha: weighting factor for rare class (list or tensor of shape [num_classes])
        gamma: focusing parameter (default 2.0)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # Convert alpha to tensor if it's a list/tuple (will be moved to device in forward)
        if alpha is not None and isinstance(alpha, (list, tuple)):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits from model
            targets: (batch_size, num_classes) one-hot encoded targets
        """
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get the probability of the true class for each sample
        # targets is one-hot, so we need to get the probability of the correct class
        target_probs = (probs * targets).sum(dim=1)  # (batch_size,)
        
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets.argmax(dim=1), reduction='none')
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as inputs
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(inputs.device)
            else:
                alpha = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            # Get alpha for each sample based on true class
            alpha_t = (alpha.unsqueeze(0) * targets).sum(dim=1)  # (batch_size,)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SimpleAutoSort:
    """
    Input: multi-waveform + single-waveform
    Identical to original AutoSort except position information is removed
    Uses SimpleClassifier model
    """
    def __init__(self, ch_num, samplepoints, device, set_shank_id, save_dir, 
                 pos_weight_noise=None, pos_weight_label=None, use_focal_loss=True, focal_alpha=None, focal_gamma=2.0):
        self.ch_num = ch_num
        self.samplepoints = samplepoints
        
        # Input format: (n_channels + 2, window_size) - multi-waveform + single-waveform + channel_index
        self.clsfier_noise = SimpleClassifier(ch_num, samplepoints, 2).to(device)
        # Label classifier (original: n_neurons classes)
        self.clsfier_label = SimpleClassifier(ch_num, samplepoints, len(set_shank_id)).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.clsfier_noise.parameters()},
            {'params': self.clsfier_label.parameters()},
        ], lr=1e-4)
        
        self.criterion = nn.MSELoss()  # Same as original (though not used)
        
        # Use Focal Loss for noise classification if enabled, otherwise use BCE
        if use_focal_loss:
            # Calculate alpha from pos_weight if not provided
            if focal_alpha is None and pos_weight_noise is not None:
                # Convert pos_weight to alpha (normalize to sum to 1)
                # pos_weight_noise[0] = spike_count/noise_count, pos_weight_noise[1] = noise_count/spike_count
                # For focal loss alpha, we want higher weight for rare class (spike)
                # Typical: alpha = [weight_noise, weight_spike] where weight_spike > weight_noise
                if isinstance(pos_weight_noise, torch.Tensor):
                    # Invert and normalize: higher weight for spike class
                    alpha_noise = 1.0 / (1.0 + pos_weight_noise[1].item())  # ~0.1
                    alpha_spike = pos_weight_noise[1].item() / (1.0 + pos_weight_noise[1].item())  # ~0.9
                    focal_alpha = [alpha_noise, alpha_spike]
                else:
                    focal_alpha = [0.25, 0.75]  # Default: favor spike class
            elif focal_alpha is None:
                focal_alpha = [0.25, 0.75]  # Default: favor spike class
            
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
            self.bceloss = None  # Not used for noise classification
        else:
            self.bceloss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_noise)
            self.focal_loss = None
        
        self.bceloss_label = nn.BCEWithLogitsLoss(pos_weight=pos_weight_label)
        self.use_focal_loss = use_focal_loss
        
        import os
        self.save_model_path_2 = os.path.join(save_dir, 'multitask_single_wave_clsfier_noise_clsfier.pth')
        self.save_model_path_3 = os.path.join(save_dir, 'multitask_single_wave_clsfier_label_clsfier.pth')
        
        self.set_shank_id = set_shank_id
        self.device = device
    
    def _prepare_input(self, batch_features, single_waveform, channel_indices):
        """
        Prepare input - unified format
        
        Input format: (B, n_channels + 2, window_size)
        - First n_channels channels: multi-waveform
        - Second-to-last channel: single-waveform
        - Last channel: channel_index (repeated window_size times)
        
        batch_features: (B, n_channels * window_size) - flattened multi-waveform
        single_waveform: (B, window_size)
        channel_indices: (B,) - detection channel indices (0-31 for 32-channel clique)
        """
        batch_size = batch_features.shape[0]
        
        # Reshape batch_features to (B, n_channels, window_size)
        multi_wf = batch_features.view(batch_size, self.ch_num, self.samplepoints)
        
        # Add single_waveform as additional channel: (B, 1, window_size)
        single_wf = single_waveform.view(batch_size, 1, self.samplepoints)
        
        # Convert channel_indices to tensor if needed and expand to (B, 1, window_size)
        if not isinstance(channel_indices, torch.Tensor):
            channel_indices = torch.tensor(channel_indices, dtype=torch.float32, device=batch_features.device)
        # Expand channel_indices: (B,) -> (B, 1, 1) -> (B, 1, window_size)
        channel_idx_channel = channel_indices.view(batch_size, 1, 1).expand(batch_size, 1, self.samplepoints)
        
        # Concatenate along channel dimension: (B, n_channels + 2, window_size)
        codes = torch.cat([multi_wf, single_wf, channel_idx_channel], dim=1)
        
        return codes
    
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
    
    def iter_model(self, batch_features, classify_labels, labels, single_waveform, channel_indices):
        """
        Training iteration
        
        Args:
            classify_labels: (B, n_neurons) one-hot labels for neuron classification
            labels: (B, 2) one-hot labels for noise/spike classification
        """
        self.optimizer.zero_grad()
        
        # Prepare input based on model type
        input_data = self._prepare_input(batch_features, single_waveform, channel_indices)
        
        # Train noise classifier
        cls_output = self.clsfier_noise(input_data.float())
        
        # Use Focal Loss for noise classification if enabled
        if self.use_focal_loss:
            train_detection_loss = 1000 * self.focal_loss(cls_output, labels)
        else:
            train_detection_loss = 1000 * self.bceloss(cls_output, labels)
        
        # Train label classifier (only for samples classified as spike)
        test = labels[:, 1] == 1  # GT spike samples
        train_classification_loss = torch.tensor(0.0, device=self.device)
        
        if sum(test) > 1:
            input_data_spike = input_data[test]
            cls_label_output = self.clsfier_label(input_data_spike.float())
            train_classification_loss = 1000 * self.bceloss_label(
                cls_label_output, 
                classify_labels[test, :len(self.set_shank_id)]
            )
        
        # Combined loss
        train_loss = train_detection_loss + train_classification_loss
        
        train_loss.backward()
        self.optimizer.step()
        
        return train_detection_loss.item(), train_classification_loss.item(), test
    
    def iter_model_eval(self, batch_features, classify_labels, labels, single_waveform, channel_indices):
        """
        Evaluation iteration
        """
        # Prepare input
        input_data = self._prepare_input(batch_features, single_waveform, channel_indices)
        
        cls_output = self.clsfier_noise(input_data.float())
        gt = torch.argmax(labels, axis=1)
        pred = torch.argmax(cls_output, axis=1)
        
        test = labels[:, 1] == 1
        if sum(test) > 1:
            input_data_spike = input_data[test]
            cls_label_output = self.clsfier_label(input_data_spike.float())
            pred_class = torch.argmax(cls_label_output, axis=1)
            gt_label_class = torch.argmax(classify_labels[test, :len(self.set_shank_id)], axis=1)
            train_classification_loss = 1000 * self.bceloss_label(
                cls_label_output, 
                classify_labels[test, :len(self.set_shank_id)]
            )
        else:
            train_classification_loss = torch.tensor(0.0, device=self.device)
            gt_label_class = torch.tensor([], device=self.device, dtype=torch.long)
            pred_class = torch.tensor([], device=self.device, dtype=torch.long)
        
        # Use Focal Loss for noise classification if enabled
        if self.use_focal_loss:
            train_detection_loss = 1000 * self.focal_loss(cls_output, labels)
        else:
            train_detection_loss = 1000 * self.bceloss(cls_output, labels)
        train_loss = train_detection_loss + train_classification_loss
        
        return train_detection_loss.item(), train_classification_loss.item(), gt, pred, gt_label_class, pred_class


# ==================== 4. Training Function ====================

def _train_single_stage(model, train_loader, val_loader, epochs, early_stopping, patience, min_delta, device):
    """
    Helper function to train the model
    
    Returns:
        training_log: dictionary with training metrics
    """
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm
    
    n_channels = model.ch_num
    samplepoints = model.samplepoints
    
    # Training parameters
    min_valid_loss = np.inf
    best_acc_epoch = 0
    patience_counter = 0
    
    # Check if model already exists
    import os
    if os.path.exists(model.save_model_path_2) and os.path.exists(model.save_model_path_3):
        model.load_model()
        print("Loaded existing model")
        return None
    
    # Training log
    training_log = {
        'epoch': [],
        'validation_acc_noise': [],
        'validation_acc_label': []
    }
    
    print(f"\nStarting training (total {epochs} epochs)...")
    if early_stopping:
        print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")
    
    for epoch in range(epochs):
        training_log['epoch'].append(epoch + 1)
        print("epoch : {}/{}".format(epoch + 1, epochs))
        
        # Training phase
        detection_loss = 0
        classification_loss = 0
        model.train()
        # Only set pos_weight for BCE loss (not needed for Focal Loss)
        if not model.use_focal_loss and model.bceloss is not None:
            model.bceloss.pos_weight = model.bceloss.pos_weight.to(device)
        model.bceloss_label.pos_weight = model.bceloss_label.pos_weight.to(device)
        
        for batch_data in tqdm(train_loader, desc="Training"):
            batch_features, classify_labels, labels, single_waveform, channel_indices = batch_data
            classify_labels = classify_labels.to(device)
            # batch_features: (B, n_channels, window_size) from dataset
            # Flatten to (B, n_channels * window_size) for _prepare_input
            batch_size = batch_features.shape[0]
            batch_features = batch_features.view(batch_size, n_channels * samplepoints).to(device)
            labels = labels.to(device)
            single_waveform = single_waveform.to(device)
            channel_indices = channel_indices.to(device) if isinstance(channel_indices, torch.Tensor) else torch.tensor(channel_indices, device=device)
            
            train_detection_loss, train_classification_loss, test = model.iter_model(
                batch_features, classify_labels, labels, single_waveform, channel_indices
            )
            
            detection_loss += train_detection_loss
            if isinstance(test, torch.Tensor) and test.sum() > 0:
                classification_loss += train_classification_loss
            elif not isinstance(test, torch.Tensor) and test:
                classification_loss += train_classification_loss
        
        detection_loss = detection_loss / len(train_loader)
        classification_loss = classification_loss / len(train_loader)
        print("epoch : {}/{}, detection loss = {:.6f}, classification loss = {:.6f}".format(
            epoch + 1, epochs, detection_loss, classification_loss))
        
        # Validation phase
        valid_detection_loss = 0.0
        valid_classification_loss = 0.0
        
        gt_all = []
        pred_all = []
        gt_class_all = []
        pred_class_all = []
        model.eval()
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                batch_features, classify_labels, labels, single_waveform, channel_indices = batch_data
                classify_labels = classify_labels.to(device)
                # batch_features: (B, n_channels, window_size) from dataset
                # Flatten to (B, n_channels * window_size) for _prepare_input
                batch_size = batch_features.shape[0]
                batch_features = batch_features.view(batch_size, n_channels * samplepoints).to(device)
                labels = labels.to(device)
                single_waveform = single_waveform.to(device)
                channel_indices = channel_indices.to(device) if isinstance(channel_indices, torch.Tensor) else torch.tensor(channel_indices, device=device)
                
                valid_detection_loss_batch, valid_classification_loss_batch, gt, pred, gt_label_class, pred_class = model.iter_model_eval(
                    batch_features, classify_labels, labels, single_waveform, channel_indices
                )
                
                valid_detection_loss += valid_detection_loss_batch
                valid_classification_loss += valid_classification_loss_batch
                
                gt_all.append(gt.detach().cpu().numpy())
                pred_all.append(pred.detach().cpu().numpy())
                pred_class_all.append(pred_class.detach().cpu().numpy())
                gt_class_all.append(gt_label_class.detach().cpu().numpy())
        
        gt_all = np.concatenate(gt_all, axis=0)
        pred_all = np.concatenate(pred_all, axis=0)
        
        # Filter empty arrays
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
        
        val_acc_noise = accuracy_score(gt_all, pred_all)
        training_log['validation_acc_noise'].append(val_acc_noise)
        if len(gt_class_all) > 0:
            val_acc_label = f1_score(gt_class_all, pred_class_all, average='micro')
            training_log['validation_acc_label'].append(val_acc_label)
        else:
            val_acc_label = 0.0
            training_log['validation_acc_label'].append(val_acc_label)
        
        print("epoch : {}/{}, val acc noise = {:.4f}, val acc label = {:.4f}".format(
            epoch + 1, epochs, val_acc_noise, val_acc_label))
        
        # Early stopping
        if early_stopping:
            if valid_loss < min_valid_loss - min_delta:
                min_valid_loss = valid_loss
                best_acc_epoch = epoch + 1
                patience_counter = 0
                model.save_model()
                print(f"Model saved (epoch {epoch + 1}, val_loss = {valid_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    print(f"Best model was at epoch {best_acc_epoch} with val_loss = {min_valid_loss:.6f}")
                    model.load_model()  # Load best model
                    break
        else:
            # Save model every epoch if early stopping is disabled
            model.save_model()
    
    return training_log


def train_autosort_model(
    train_data_dir,
    model_save_dir,
    n_channels,
    left_sample=10,
    right_sample=20,
    epochs=20,
    batch_size=512,
    device=None,
    early_stopping=True,
    patience=5,
    min_delta=0.0,
    use_focal_loss=True,
    focal_gamma=2.0,
    keep_id=None,
):
    """
    Train AutoSort model
    
    Parameters:
        train_data_dir: training data directory
        model_save_dir: model save directory
        n_channels: number of channels
        left_sample, right_sample: window parameters
        epochs: number of training epochs
        batch_size: batch size
        device: device (if None, auto-select)
        early_stopping: whether to enable early stopping, default True
        patience: early stopping patience (stop after how many consecutive epochs without improvement), default 5
        min_delta: minimum change for early stopping, default 0.0
        use_focal_loss: whether to use Focal Loss for noise classification, default True
        focal_gamma: Focal Loss gamma parameter, default 2.0
        keep_id: list of unit IDs to keep (if None, auto-extract from data)
    
    Returns:
        autosort_model: trained model
        training_log: training log
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model save directory
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set parameters
    samplepoints = left_sample + right_sample
    
    # Create dataset
    print("Create dataset...")
    if keep_id is not None:
        print(f"Using provided keep_id: {len(keep_id)} units")
        print(f"  keep_id: {keep_id}")
    else:
        print("Auto-extracting keep_id from data")
    
    dataset = SimpleWaveformLoader(
        root=str(train_data_dir) + '/',
        shank_channel=np.arange(n_channels),
        Keep_id=keep_id  # 使用传入的keep_id，如果为None则自动提取
    )
    
    set_shank_id = dataset.keep_id
    print(f"Model parameters:")
    print(f"  - Number of channels: {n_channels}")
    print(f"  - Window length: {samplepoints}")
    print(f"  - Number of units: {len(set_shank_id)}")
    print(f"  - Input dimension: {(n_channels + 1) * samplepoints}")
    
    # Save unit ID list (for evaluation)
    import os
    keep_id_path = os.path.join(model_save_dir, 'keep_id.pkl')
    with open(keep_id_path, 'wb') as f:
        pickle.dump(set_shank_id, f)
    print(f"Unit ID list saved to: {keep_id_path}")
    
    # Save classification mapping (label index -> unit ID)
    # The order of set_shank_id determines the label index: label_index 0 -> set_shank_id[0], etc.
    classification_mapping = {
        'label_to_unit': {idx: unit_id for idx, unit_id in enumerate(set_shank_id)},
        'unit_to_label': {unit_id: idx for idx, unit_id in enumerate(set_shank_id)},
        'label_list': list(set_shank_id)  # Same as keep_id, but saved explicitly for clarity
    }
    classification_mapping_path = os.path.join(model_save_dir, 'classification_mapping.pkl')
    with open(classification_mapping_path, 'wb') as f:
        pickle.dump(classification_mapping, f)
    print(f"Classification mapping saved to: {classification_mapping_path}")
    print(f"  - Label indices: 0 to {len(set_shank_id)-1}")
    print(f"  - Unit IDs: {sorted(set_shank_id)}")
    
    # Calculate pos_weight_label (original, without noise category)
    pos_weight_label = dataset.pos_weight_label.to(device)
    
    # Create model with Focal Loss enabled by default for noise classification
    autosort_model = SimpleAutoSort(
        ch_num=n_channels,
        samplepoints=samplepoints,
        device=device,
        set_shank_id=set_shank_id,
        save_dir=model_save_dir,
        pos_weight_noise=dataset.pos_weight_noise.to(device),
        pos_weight_label=pos_weight_label,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma
    )
    
    # Split training and validation sets
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train model
    training_log = _train_single_stage(
        autosort_model, train_loader, val_loader, epochs,
        early_stopping, patience, min_delta, device
    )
    
    print(f"\nDataset split:")
    print(f"  - Training set: {train_size} samples")
    print(f"  - Validation set: {val_size} samples")
    
    # Save final model
    autosort_model.save_model()
    print("Final model saved")
    
    # Save training log
    import pandas as pd
    if isinstance(training_log, dict) and 'epoch' in training_log:
        log_df = pd.DataFrame(training_log)
        log_path = os.path.join(model_save_dir, 'training_log.csv')
        log_df.to_csv(log_path, index=False)
        print(f"Training log saved to: {log_path}")
    
    return autosort_model, training_log


# ==================== 5. Neuron Matching Function ====================

def match_neurons(
    train_neuron_inf,
    eval_neuron_inf,
    train_data_dir=None,
    eval_data_dir=None,
    position_threshold=10,
    waveform_similarity_threshold=0.95,
):
    """
    Match neurons between training and evaluation data
    
    Parameters:
        train_neuron_inf: Training data neuron_inf DataFrame (must contain position_1, position_2, position_waveform columns）
        eval_neuron_inf: Evaluation data neuron_inf DataFrame (must contain position_1, position_2, position_waveform columns）
        train_data_dir: training data directory (deprecated, kept for backward compatibility)
        eval_data_dir: Evaluation data directory (deprecated, kept for backward compatibility)
        position_threshold: Position distance threshold (Euclidean distance, in microns), default 10
        waveform_similarity_threshold: Waveform similarity threshold (Pearson correlation coefficient), default 0.95
    
    Returns:
        eval_neuron_inf_matched: Evaluation neuron_inf with added neuron_match column
    """
    from scipy.stats import pearsonr
    
    print("=" * 50)
    print("Neuron Matching")
    print("=" * 50)
    
    # Copy evaluation neuron_inf
    eval_neuron_inf_matched = eval_neuron_inf.copy()
    eval_neuron_inf_matched['neuron_match'] = 'unmatch'
    
    # Check required columns
    required_cols = ['position_1', 'position_2', 'position_waveform']
    for col in required_cols:
        if col not in train_neuron_inf.columns:
            raise ValueError(f"train_neuron_inf missing required column: {col}")
        if col not in eval_neuron_inf.columns:
            raise ValueError(f"eval_neuron_inf missing required column: {col}")
    
    # Get all training neuron names
    train_unique_neurons = train_neuron_inf['Neuron'].unique()
    
    # Match neurons
    print("Matching neurons...")
    matched_count = 0
    for eval_idx, eval_row in eval_neuron_inf_matched.iterrows():
        eval_neuron = eval_row['Neuron']
        
        # Get evaluation neuron position coordinates and waveform
        eval_pos = np.array([eval_row['position_1'], eval_row['position_2']])
        eval_wf = eval_row['position_waveform']
        
        # Ensure waveform is numpy array
        if not isinstance(eval_wf, np.ndarray):
            eval_wf = np.array(eval_wf)
        
        if len(eval_wf) == 0:
            continue
        
        best_match = None
        best_similarity = 0
        
        # Iterate through all training neurons
        for train_neuron in train_unique_neurons:
            if train_neuron is None:
                continue
            train_rows = train_neuron_inf[train_neuron_inf['Neuron'] == train_neuron]
            if len(train_rows) == 0:
                continue
            train_row = train_rows.iloc[0]
            
            # Get training neuron position coordinates and waveform
            train_pos = np.array([train_row['position_1'], train_row['position_2']])
            train_wf = train_row['position_waveform']
            
            # Ensure waveform is numpy array
            if not isinstance(train_wf, np.ndarray):
                train_wf = np.array(train_wf)
            
            if len(train_wf) == 0:
                continue
            
            # Calculate position distance (Euclidean distance)
            position_distance = np.linalg.norm(eval_pos - train_pos)
            
            # Calculate waveform similarity (Pearson correlation coefficient)
            if len(eval_wf) == len(train_wf):
                similarity, _ = pearsonr(eval_wf, train_wf)
                if np.isnan(similarity):
                    similarity = 0
            else:
                similarity = 0
            
            # Check if matched
            if (position_distance < position_threshold and 
                similarity > waveform_similarity_threshold and
                similarity > best_similarity):
                best_match = train_neuron
                best_similarity = similarity
        
        # Set matching result
        if best_match is not None:
            eval_neuron_inf_matched.loc[eval_idx, 'neuron_match'] = best_match
            matched_count += 1
            train_row_matched = train_neuron_inf[train_neuron_inf['Neuron'] == best_match].iloc[0]
            train_pos_matched = np.array([train_row_matched['position_1'], train_row_matched['position_2']])
            position_distance_final = np.linalg.norm(eval_pos - train_pos_matched)
            print(f"  {eval_neuron} -> {best_match} (Similarity: {best_similarity:.4f}, Position distance: {position_distance_final:.2f})")
    
    print(f"\nMatching completed:")
    print(f"  - Total evaluation neurons: {len(eval_neuron_inf_matched)}")
    print(f"  - Matched: {matched_count}")
    print(f"  - Unmatched: {len(eval_neuron_inf_matched) - matched_count}")
    
    return eval_neuron_inf_matched




def compute_position_waveform_from_average(
    per_channel_waveform: np.ndarray,
    channel_id: list,
    channel_positions: dict,
    window_size: int = 30,
) -> tuple:
    """
    Compute position and position_waveform from a single average per_channel_waveform
    
    Parameters:
        per_channel_waveform: numpy array, shape (n_timepoints, n_channels) = (window_size, n_channels)
        channel_id: list of channel IDs (clique内的索引，0-based)
        channel_positions: dict, 键为channel索引（clique内的索引），值为(x, y)位置元组
        window_size: window size, default 30
    
    Returns:
        position_1, position_2, position_waveform (window_size-dim)
    """
    # Extract channels corresponding to channel_id
    # per_channel_waveform shape: (window_size, n_channels)
    # Extract columns for channel_id: (window_size, n_valid_channels)
    waveform_subset = per_channel_waveform[:, channel_id]  # (window_size, n_valid_channels)
    # Transpose to (n_valid_channels, window_size) for easier calculation
    snippet = waveform_subset.T  # (n_valid_channels, window_size)
    
    # Calculate position of this average waveform (based on channel_id channels)
    a_squared = [np.sum(snippet[j, :]**2) for j in range(len(channel_id))]
    
    sum_x_a = 0
    sum_y_a = 0
    sum_a = 0
    
    for j, ch_idx in enumerate(channel_id):
        x_i, y_i = channel_positions.get(ch_idx, (0, 0))
        a_i_sq = a_squared[j]
        sum_x_a += x_i * a_i_sq
        sum_y_a += y_i * a_i_sq
        sum_a += a_i_sq
    
    if sum_a == 0:
        return 0.0, 0.0, np.zeros(window_size, dtype=np.float32)
    
    spike_x = sum_x_a / sum_a
    spike_y = sum_y_a / sum_a
    
    # Calculate position_waveform (based on spike position and channel_id channels)
    distances = []
    for ch_idx in channel_id:
        x_channel, y_channel = channel_positions.get(ch_idx, (np.nan, np.nan))
        if not (np.isnan(x_channel) or np.isnan(y_channel)):
            distance = np.sqrt((spike_x - x_channel)**2 + (spike_y - y_channel)**2)
            distances.append(distance)
        else:
            distances.append(np.inf)
    
    if not distances or all(d == np.inf for d in distances):
        return spike_x, spike_y, np.zeros(window_size, dtype=np.float32)
    
    distances = np.array(distances, dtype=np.float32)
    
    # Calculate position_waveform using IDW interpolation
    weights = 1.0 / (np.power(distances, 2, dtype=np.float32) + 1e-10)
    if np.any(distances == 0):
        zero_idx = np.where(distances == 0)[0][0]
        spike_position_waveform = snippet[zero_idx, :].astype(np.float32)
    else:
        weights /= weights.sum()
        spike_position_waveform = np.zeros(window_size, dtype=np.float32)
        for t in range(window_size):
            spike_position_waveform[t] = float(np.dot(snippet[:, t], weights))
    
    return spike_x, spike_y, spike_position_waveform


def compute_cluster_position_waveform(
    snippets: np.ndarray,
    channel_id: list,
    channel_positions: dict,
    window_size: int = 30,
) -> tuple:
    """
    Compute cluster position and position_waveform from snippets (reference: generate_neuron_inf_phy_template.py)
    
    Parameters:
        snippets: numpy array, shape (n_spikes, n_channels, window_size)
        channel_id: list of channel IDs (clique内的索引，0-based)
        channel_positions: dict, 键为channel索引（clique内的索引），值为(x, y)位置元组
        window_size: window size, default 30
    
    Returns:
        position_1, position_2, position_waveform (30-dim)
    """
    cluster_positions_x = []
    cluster_positions_y = []
    cluster_waveforms = []
    
    for snippet in snippets:  # snippet: (n_channels, window_size)
        # Calculate position of this spike (based on channel_id channels)
        a_squared = [np.sum(snippet[j, :]**2) for j in range(len(channel_id))]
        
        sum_x_a = 0
        sum_y_a = 0
        sum_a = 0
        
        for j, ch_idx in enumerate(channel_id):
            x_i, y_i = channel_positions.get(ch_idx, (0, 0))
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
        
        # Calculate position_waveform (based on spike position and channel_id channels)
        distances = []
        for ch_idx in channel_id:
            x_channel, y_channel = channel_positions.get(ch_idx, (np.nan, np.nan))
            if not (np.isnan(x_channel) or np.isnan(y_channel)):
                distance = np.sqrt((spike_x - x_channel)**2 + (spike_y - y_channel)**2)
                distances.append(distance)
            else:
                distances.append(np.inf)
        
        if not distances or all(d == np.inf for d in distances):
            continue
        
        distances = np.array(distances, dtype=np.float32)
        
        # Calculate position_waveform using IDW interpolation
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
    
    # Calculate average position and waveform
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
    gt_detect_array: pd.DataFrame = None,
    match_mode: str = 'per_channel_match',
    device=None,
):
    """
    Stage 1: Calibration stage (first 60s)
    
    Process:
    1. Threshold detection
    2. Pass through noise classifier, classified as spikes
    3. Extract intermediate features (30 dimensions from intermediate_forward)
    4. K-means clustering (number of classes = train neurons + n)
    5. Calculate position and waveform for each cluster
    6. Match with train neurons, establish mapping relationship
    
    Parameters:
        recording_f: preprocessed recording object
        autosort_model: trained SimpleAutoSort model
        train_neuron_inf: training data neuron_inf DataFrame
        calibration_duration_seconds: calibration duration (seconds)，default 60
        n_additional_clusters: number of additional clusters (n)，default 5
        detection_params: detection parameters dictionary
        window_params: window parameters dictionary
        position_threshold: position distance threshold (microns)，default 10
        waveform_similarity_threshold: waveform similarity threshold，default 0.9
        eval_neuron_inf: evaluation data neuron_inf DataFrame (optional, for generating GT labels)
        gt_detect_array: ground truth detect array DataFrame with columns: time, unit_id, extremum_channel (optional, for GT matching and recall calculation)
        match_mode: matching mode, either 'per_channel_match' or 'combined_match'
            - 'per_channel_match': cluster spikes per channel (based on extremum_channel), then match clusters to neurons (default)
            - 'combined_match': perform unified K-means clustering on all spikes (n_train_neuron + 10 clusters), then match clusters to neurons based on channel_id
        device: device
    
    Returns:
        calibration_results: dictionary, containing:
            - kmeans_model: trained K-means model
            - pca_model: trained PCA model
            - cluster_to_neuron_mapping: mapping from cluster to train neuron
            - cluster_features: features for each cluster (position, waveform, etc.)
    """
    from sklearn.cluster import KMeans
    from scipy.stats import pearsonr
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if detection_params is None:
        detection_params = {
            'thr_min': 3.5,
            'thr_max': 30,
            'distance': 3,
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
    
    # 获取channel positions（从recording的probe）
    channel_locations = recording_f.get_channel_locations()  # (n_channels, 2) or (n_channels, 3)
    channel_positions_dict = {}
    for ch_idx in range(n_channels):
        if channel_locations.shape[1] >= 2:
            channel_positions_dict[ch_idx] = (float(channel_locations[ch_idx, 0]), float(channel_locations[ch_idx, 1]))
        else:
            channel_positions_dict[ch_idx] = (0.0, 0.0)
    
    # 计算valid_channels：检查是否使用新的检测方法
    recording_channel_ids = list(recording_f.get_channel_ids())  # clique的channel IDs（如"A-000"）
    # Disabled: use original detection method
    
    print("Using old detection method: extremum_channels")
    valid_channels = None
    if 'extremum_channel' in train_neuron_inf.columns and len(train_neuron_inf) > 0:
        # 提取所有唯一的extremum_channels
        unique_extremum_channels = train_neuron_inf['extremum_channel'].dropna().unique()
        valid_channels = []
        for ch in unique_extremum_channels:
            # 查找extremum_channel在recording_channel_ids中的索引
            try:
                ch_idx = recording_channel_ids.index(str(ch))
                valid_channels.append(ch_idx)
            except ValueError:
                # 如果找不到，跳过
                continue
        valid_channels = sorted(set(valid_channels))  # 去重并排序
        print(f"Using {len(valid_channels)} valid channels from train neuron extremum_channels")
    else:
        print("Using all channels for detection (train_neuron_inf is empty or missing extremum_channel column)")

    # 筛选eval_neuron_inf：只保留extremum_channel在valid_channels中的neuron
    if eval_neuron_inf is not None:
        if valid_channels is not None and 'extremum_channel' in eval_neuron_inf.columns:
            # 筛选：只保留extremum_channel在valid_channels中的neuron
            valid_neuron_mask = []
            # 获取valid_channels对应的channel ID字符串集合
            valid_extremum_channels = set(recording_channel_ids[idx] for idx in valid_channels)
            
            for _, row in eval_neuron_inf.iterrows():
                extremum_channel = row.get('extremum_channel')
                if pd.isna(extremum_channel) or extremum_channel is None:
                    valid_neuron_mask.append(False)
                else:
                    # 检查extremum_channel是否在valid_channels中
                    if str(extremum_channel) in valid_extremum_channels:
                        valid_neuron_mask.append(True)
                    else:
                        valid_neuron_mask.append(False)
            
            eval_neuron_inf_filtered = eval_neuron_inf[valid_neuron_mask].copy()
            filtered_count = len(eval_neuron_inf) - len(eval_neuron_inf_filtered)
            
            if filtered_count > 0:
                print(f"筛选eval_neuron_inf: 从{len(eval_neuron_inf)}个neuron筛选到{len(eval_neuron_inf_filtered)}个（移除了{filtered_count}个不在valid_channels中的neuron）")
            else:
                print(f"eval_neuron_inf筛选: 所有{len(eval_neuron_inf)}个neuron都在valid_channels中")
            
            eval_neuron_inf = eval_neuron_inf_filtered
            
            # 筛选gt_detect_array：只保留属于筛选后的neuron的spikes
            if gt_detect_array is not None:
                valid_neuron_names = set(eval_neuron_inf['Neuron'].unique())
                gt_detect_array_filtered = gt_detect_array[gt_detect_array['unit_id'].isin(valid_neuron_names)].copy()
                filtered_spike_count = len(gt_detect_array) - len(gt_detect_array_filtered)
                
                if filtered_spike_count > 0:
                    print(f"筛选gt_detect_array: 从{len(gt_detect_array)}个spikes筛选到{len(gt_detect_array_filtered)}个（移除了{filtered_spike_count}个不属于valid neurons的spikes）")
                else:
                    print(f"gt_detect_array筛选: 所有{len(gt_detect_array)}个spikes都属于valid neurons")
                
                gt_detect_array = gt_detect_array_filtered
        else:
            print("eval_neuron_inf筛选: 无valid_channels或缺少extremum_channel列，跳过筛选")
    
    print("=" * 50)
    print("Stage 1: Calibration (first 60 seconds)")
    print("=" * 50)
    
    # 1. Load first 60s of data
    max_duration_samples = int(calibration_duration_seconds * sampling_frequency)
    print(f"Loading first {calibration_duration_seconds} seconds of data...")
    traces = recording_f.get_traces(start_frame=0, end_frame=max_duration_samples)
    if traces.shape[0] > traces.shape[1] and traces.shape[0] > 100:
        traces = traces.T
    traces = traces.astype(np.float32)
    print(f"Data shape: {traces.shape}")
    
    # 2. Threshold detection
    print("\n### 2. Threshold detection")
    trace0_car = traces.T  # (n_timepoints, n_channels)
    

    detection_params_with_valid = detection_params.copy() if detection_params else {}
    if valid_channels is not None:
        detection_params_with_valid['valid_channels'] = valid_channels
    
    spikes = detect_spike(trace0_car, **detection_params_with_valid)
    spike_coords = np.argwhere(spikes == 1)  # (n_spikes, 2) [time, channel]
    print(f"Number of detected spikes: {len(spike_coords)}")
    
    # Deduplicate spikes: for each neuron, if both extremum_channel and second-max-SNR channel
    # detect spikes at the same time point, keep the spike with larger amplitude
    if 'extremum_channel' in train_neuron_inf.columns and 'channel_snr' in train_neuron_inf.columns:
        # Convert spike_coords to detect_array format for deduplication
        detect_array_for_dedup = spike_coords.astype(np.int64)
        detect_array_dedup = deduplicate_spikes_by_neuron_channels(
            detect_array_for_dedup, train_neuron_inf, recording_channel_ids, trace0_car
        )
        spike_coords = detect_array_dedup
        print(f"Number of detected spikes after deduplication: {len(spike_coords)}")
    
    gt_label_array = None 
    gt_array_for_mapping = None  # 保存gt_array，用于后续统计
    if gt_detect_array is not None and len(gt_detect_array) > 0:
        # 筛选时间范围：只保留在calibration_duration_seconds内的GT spikes
        max_duration_seconds = calibration_duration_seconds
        gt_detect_array_filtered = gt_detect_array[gt_detect_array['time'] < max_duration_seconds * sampling_frequency].copy()
        
        if len(gt_detect_array_filtered) == 0:
            print("GT匹配统计: 无有效GT spikes数据（时间范围内无数据），跳过匹配统计")
        else:
            # 构建detect_array: (n_detected, 2), 每行是[time_sample, channel_idx]
            detect_array = spike_coords.astype(np.int64)  # (n_detected, 2) [time, channel]
            
            # 构建gt_array: (n_gt, 2), 每行是[time_sample, channel_idx]
            # 从筛选后的gt_detect_array获取时间和extremum_channel
            # Note: gt_detect_array['time'] is already in sample indices, not seconds
            gt_times_samples = gt_detect_array_filtered['time'].values.astype(np.int64)
            
            # 将extremum_channel转换为clique内的索引（extremum_channel是必须的）
            gt_channels = []
            for extremum_channel in gt_detect_array_filtered['extremum_channel'].values:
                ch_idx = recording_channel_ids.index(str(extremum_channel))
                gt_channels.append(ch_idx)
            
            gt_channels = np.array(gt_channels, dtype=np.int64)
            
            # 构建gt_array
            gt_array = np.column_stack([gt_times_samples, gt_channels]).astype(np.int64)
            gt_array_for_mapping = gt_array  # 保存gt_array，用于后续统计
            
            # 使用map_gt_annotation进行匹配（同时考虑时间和通道，允许±1采样点误差）
            gt_label_array = map_gt_annotation(detect_array, gt_array)
            
            # 统计有多少GT spikes被匹配到（召回率）
            matched_gt_indices = set(gt_label_array[gt_label_array >= 0])
            matched_gt_count = len(matched_gt_indices)
            gt_total_count = len(gt_array)
            recall_rate = matched_gt_count / gt_total_count if gt_total_count > 0 else 0
            print(f"GT匹配统计: {matched_gt_count}/{gt_total_count} GT spikes被检测到 (召回率: {recall_rate:.4f})")
    else:
        print("GT匹配统计: 无gt_detect_array数据，跳过匹配统计")
    
    # 3. Extract waveforms and filter boundaries
    print("\n### 3. Extract waveforms")
    valid_spikes = []
    waveforms = []
    spike_times = []
    spike_channels = []
    filtered_gt_labels = []  # 保存过滤后的GT labels，用于后续复用
    
    for spike_idx, (time_idx, channel_idx) in enumerate(spike_coords):
        start = time_idx - left_sample
        end = time_idx + right_sample
        
        if start < 0 or end > trace0_car.shape[0]:
            continue
        if end - start != window_size:
            continue
        
        # Extract waveform (n_channels, window_size)
        waveform = traces[:, start:end]  # (n_channels, window_size)
        waveforms.append(waveform)
        valid_spikes.append((time_idx, channel_idx))
        spike_times.append(time_idx)
        spike_channels.append(channel_idx)
        
        # 同步保存对应的GT label（如果存在）
        if gt_label_array is not None:
            # gt_label_array[spike_idx] >= 0 表示匹配到GT spike，否则为-1（未匹配）
            filtered_gt_labels.append(gt_label_array[spike_idx])
    
    waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size)
    
    # 将过滤后的GT labels转换为numpy数组（如果存在）
    if len(filtered_gt_labels) > 0:
        filtered_gt_labels = np.array(filtered_gt_labels)  # (n_spikes,)
    else:
        filtered_gt_labels = None
    
    # 4. Pass through noise classifier, classified as spikes
    print("\n### 4. Noise classifier filtering")
    autosort_model.eval()
    
    # 提前构建eval_spike_data（如果有gt_detect_array），用于计算GT spike通过noise classifier的比例
    eval_spike_data_for_noise = None
    if gt_detect_array is not None and len(gt_detect_array) > 0:
        max_duration_seconds = calibration_duration_seconds
        gt_detect_array_time_filtered = gt_detect_array[gt_detect_array['time'] < max_duration_seconds].copy()
        if len(gt_detect_array_time_filtered) > 0:
            eval_spike_data_for_noise = pd.DataFrame({
                'time': gt_detect_array_time_filtered['time'].values,
                'neuron': gt_detect_array_time_filtered['unit_id'].values
            })
    
    # Prepare data
    batch_size = 2048
    n_spikes = len(waveforms)
    spike_indices = []
    way3_features = []
    all_noise_pred_labels = []  # 收集所有noise classifier的预测结果
    all_noise_gt_labels = []  # 收集所有GT labels（如果有）
    
    with torch.no_grad():
        for i in tqdm(range(0, n_spikes, batch_size), desc="Noise classification"):
            batch_waveforms = waveforms[i:i+batch_size]  # (batch, n_channels, window_size)
            batch_channels = spike_channels[i:i+batch_size]
            
            # Extract single waveform (maximum amplitude channel)
            batch_single_waveforms = []
            batch_multi_waveforms = []
            
            for j, (wf, ch) in enumerate(zip(batch_waveforms, batch_channels)):
                # multi-waveform: flatten to (n_channels * window_size,)
                multi_wf = wf.flatten()  # (n_channels * window_size,)
                batch_multi_waveforms.append(multi_wf)
                
                # single-waveform: waveform from maximum amplitude channel
                single_wf = wf[ch, :]  # (window_size,)
                batch_single_waveforms.append(single_wf)
            
            batch_multi_waveforms = np.array(batch_multi_waveforms)  # (batch, n_channels * window_size)
            batch_single_waveforms = np.array(batch_single_waveforms)  # (batch, window_size)
            
            # Convert to tensor
            batch_multi = torch.from_numpy(batch_multi_waveforms).float().to(device)
            batch_single = torch.from_numpy(batch_single_waveforms).float().to(device)
            batch_channel_indices = torch.from_numpy(np.array(batch_channels)).long().to(device)
            
            # Prepare input using model's _prepare_input method
            codes = autosort_model._prepare_input(batch_multi, batch_single, batch_channel_indices)
            
            # Noise classification
            noise_output = autosort_model.clsfier_noise(codes)
            noise_pred = torch.argmax(noise_output, dim=1)  # 0=noise, 1=spike
            all_noise_pred_labels.extend(noise_pred.cpu().numpy().tolist())
            
            # Get GT noise labels - 复用map_gt_annotation的结果
            if filtered_gt_labels is not None:
                # 使用map_gt_annotation的结果：>=0表示匹配到GT spike（标记为1），-1表示未匹配（标记为0）
                batch_gt_labels = filtered_gt_labels[i:i+batch_size]
                batch_gt_noise = (batch_gt_labels >= 0).astype(int).tolist()  # >=0 -> 1 (spike), -1 -> 0 (noise)
                all_noise_gt_labels.extend(batch_gt_noise)
            
            # Keep only samples classified as spikes
            spike_mask = noise_pred == 1
            if spike_mask.sum() > 0:
                batch_indices = np.arange(i, min(i+batch_size, n_spikes))[spike_mask.cpu().numpy()]
                spike_indices.extend(batch_indices)
                
                # Extract way3 layer features (only for spike samples)
                codes_spike = codes[spike_mask]
                way3_batch = autosort_model.clsfier_label.intermediate_forward(codes_spike)
                way3_features.append(way3_batch.cpu().numpy())
    
    
    way3_features = np.concatenate(way3_features, axis=0)  # (n_spikes, 30) - intermediate_forward returns 30-dim features
    spike_indices = np.array(spike_indices)
    print(f"Number of spikes passing noise classifier: {len(spike_indices)}")
    
    # Get channels for spikes that passed noise classifier
    # spike_indices are indices into the waveforms array, so we need to get corresponding channels
    spike_channels_filtered = np.array([spike_channels[i] for i in spike_indices])  # (n_spikes_passed,)
    
    # 计算noise classifier的准确率
    noise_classifier_accuracy = None
    if len(all_noise_gt_labels) > 0 and len(all_noise_pred_labels) == len(all_noise_gt_labels):
        all_noise_pred_labels = np.array(all_noise_pred_labels)
        all_noise_gt_labels = np.array(all_noise_gt_labels)
        noise_classifier_accuracy = np.mean(all_noise_pred_labels == all_noise_gt_labels)
        print(f"Noise classifier准确率: {noise_classifier_accuracy:.4f} ({np.sum(all_noise_pred_labels == all_noise_gt_labels)}/{len(all_noise_gt_labels)})")
        
        # 计算GT spike通过noise classifier的比例
        # 使用map_gt_annotation的结果，更准确地统计
        if gt_array_for_mapping is not None and len(gt_array_for_mapping) > 0:
            # 获取实际GT spikes总数
            actual_gt_spikes_total = len(gt_array_for_mapping)
            
            # 使用map_gt_annotation的结果来统计
            # 对于每个GT spike，找到匹配的检测到的spike，检查是否通过了noise classifier
            gt_spike_passed_count = 0
            
            # 创建通过noise classifier的spike索引集合（在waveforms中的索引）
            passed_spike_indices_set = set(spike_indices.tolist())
            
            # 建立spike_coords索引到waveforms索引的映射
            # 由于waveforms是过滤后的，需要找到每个spike_coords索引对应的waveforms索引
            spike_coords_to_waveform_idx = {}
            waveform_idx = 0
            for spike_idx, (time_idx, channel_idx) in enumerate(spike_coords):
                start = time_idx - left_sample
                end = time_idx + right_sample
                if start >= 0 and end <= trace0_car.shape[0] and end - start == window_size:
                    spike_coords_to_waveform_idx[spike_idx] = waveform_idx
                    waveform_idx += 1
            
            # 对于每个GT spike，检查匹配的检测到的spike是否通过了noise classifier
            for gt_idx in range(actual_gt_spikes_total):
                # 找到匹配这个GT spike的检测到的spike（在spike_coords中的索引）
                matched_spike_coords_indices = np.where(gt_label_array == gt_idx)[0]
                
                if len(matched_spike_coords_indices) > 0:
                    # 检查是否有匹配的spike通过了waveform提取和noise classifier
                    for spike_coords_idx in matched_spike_coords_indices:
                        if spike_coords_idx in spike_coords_to_waveform_idx:
                            waveform_idx = spike_coords_to_waveform_idx[spike_coords_idx]
                            if waveform_idx in passed_spike_indices_set:
                                # 找到了一个匹配的spike，且通过了noise classifier
                                gt_spike_passed_count += 1
                                break  # 一个GT spike只需要找到一个匹配的spike即可
            
            gt_spike_pass_rate = gt_spike_passed_count / actual_gt_spikes_total if actual_gt_spikes_total > 0 else 0
            print(f"GT spike通过noise classifier比例: {gt_spike_pass_rate:.4f} ({gt_spike_passed_count}/{actual_gt_spikes_total})")
        else:
            # 如果没有GT数据，使用原来的统计方法（检测到的spikes中标记为GT的数量）
            gt_spike_mask = all_noise_gt_labels == 1  # GT spike的mask
            gt_spike_total = np.sum(gt_spike_mask)  # 检测到的spikes中标记为GT的数量
            if gt_spike_total > 0:
                gt_spike_passed_mask = (all_noise_gt_labels == 1) & (all_noise_pred_labels == 1)  # GT spike且通过noise classifier
                gt_spike_passed_count = np.sum(gt_spike_passed_mask)
                gt_spike_pass_rate = gt_spike_passed_count / gt_spike_total
                print(f"GT spike通过noise classifier比例: {gt_spike_pass_rate:.4f} ({gt_spike_passed_count}/{gt_spike_total}) [基于检测到的spikes]")
            else:
                print("GT spike通过noise classifier比例: 无GT spike数据，跳过计算")
    elif len(all_noise_gt_labels) == 0:
        print("Noise classifier准确率: 无GT数据，跳过计算")
    
    # 5. K-means clustering and matching
    if match_mode == 'per_channel_match':
        print("\n### 5. Per-channel K-means clustering and matching")
    elif match_mode == 'combined_match':
        print("\n### 5. Combined K-means clustering and matching")
    else:
        raise ValueError(f"Unknown match_mode: {match_mode}. Must be 'per_channel_match' or 'combined_match'")
    
    # Group spikes by channel (only for spikes that passed noise classifier)
    spike_channels_array = spike_channels_filtered  # (n_spikes_passed,) - channel indices for spikes that passed noise classifier
    unique_channels = np.unique(spike_channels_array)
    
    # Initialize global results
    cluster_to_neuron_mapping = {}  # {cluster_id: train_neuron_name}
    neuron_to_clusters = defaultdict(list)  # {train_neuron_name: [cluster_ids]}
    cluster_features = {}  # Save matched cluster features
    all_cluster_labels = np.full(len(spike_indices), -1, dtype=int)  # Global cluster labels for all spikes
    noise_clusters = set()  # Set of cluster IDs that are marked as noise
    cluster_noise_spike_indices = []  # List of spike indices marked as noise
    global_cluster_id = 0  # Global cluster ID counter
    kmeans_model = None  # Will be set based on match_mode
    cluster_per_channel_waveforms = {}  # {cluster_id: per_channel_waveform (n_timepoints, n_channels)}
    
    if match_mode == 'combined_match':
        # Combined match mode: unified K-means on all spikes
        print(f"  Performing unified K-means clustering on all {len(spike_indices)} spikes...")
        
        # Calculate number of clusters: n_train_neuron + 10
        n_train_neuron = len(train_neuron_inf)
        n_clusters_combined = n_train_neuron + 30
        print(f"  Number of clusters: {n_clusters_combined} (n_train_neuron={n_train_neuron} + 10)")
        
        # Check if we have enough spikes for clustering
        min_spikes_for_clustering = 30
        if len(spike_indices) < min_spikes_for_clustering:
            print(f"  Warning: Only {len(spike_indices)} spikes < {min_spikes_for_clustering}, marking all as noise")
            cluster_noise_spike_indices.extend(spike_indices.tolist())
            all_cluster_labels.fill(-1)
        else:
            # Perform unified K-means clustering
            combined_kmeans = KMeans(n_clusters=n_clusters_combined, random_state=42, n_init=10)
            all_cluster_labels = combined_kmeans.fit_predict(way3_features)  # (n_spikes_passed,)
            kmeans_model = combined_kmeans
            print(f"  Clustering completed: {len(np.unique(all_cluster_labels))} clusters")
            
            # Map clusters to global cluster IDs (already global in combined mode)
            global_cluster_id = n_clusters_combined
            
            # Match each cluster to neurons based on channel_id
            print(f"  Matching clusters to train neurons...")
            unique_cluster_ids = np.unique(all_cluster_labels)
            
            for cluster_id in unique_cluster_ids:
                # Get spikes in this cluster
                cluster_mask = all_cluster_labels == cluster_id
                cluster_spike_indices_in_waveforms = spike_indices[cluster_mask]  # Indices in waveforms array
                cluster_spike_indices_in_spike_indices = np.where(cluster_mask)[0]  # Positions in spike_indices array
                
                if len(cluster_spike_indices_in_waveforms) == 0:
                    # Empty cluster, mark as noise
                    noise_clusters.add(cluster_id)
                    continue
                
                # Get waveforms for this cluster
                cluster_waveforms_full = waveforms[cluster_spike_indices_in_waveforms]  # (n_spikes, n_channels, window_size)
                
                # Calculate per_channel_waveform for this cluster: (n_timepoints, n_channels)
                # Average across all spikes in the cluster, then transpose from (n_channels, window_size) to (window_size, n_channels)
                cluster_avg_waveform_per_channel = np.mean(cluster_waveforms_full, axis=0)  # (n_channels, window_size)
                cluster_per_channel_waveform = cluster_avg_waveform_per_channel.T  # (window_size, n_channels) = (n_timepoints, n_channels)
                cluster_per_channel_waveforms[cluster_id] = cluster_per_channel_waveform.astype(np.float32)
                
                # Try to match with each train neuron based on channel_id
                best_match = None
                best_score = -1
                best_match_features = None
                
                for neuron_idx, neuron_row in train_neuron_inf.iterrows():
                    train_neuron = neuron_row['Neuron']
                    train_pos = np.array([neuron_row['position_1'], neuron_row['position_2']])
                    train_waveform = np.asarray(neuron_row['position_waveform'], dtype=np.float32)
                    
                    # Get train neuron channel_id
                    train_channel_id = neuron_row.get('channel_id', [])
                    if not train_channel_id or len(train_channel_id) == 0:
                        continue
                    
                    # Convert channel_id to clique indices
                    train_channel_indices = []
                    for ch_item in train_channel_id:
                        try:
                            if isinstance(ch_item, (int, np.integer)):
                                ch_idx = int(ch_item)
                                # Check if it's a valid 0-based index
                                if 0 <= ch_idx < len(recording_channel_ids):
                                    train_channel_indices.append(ch_idx)
                                # If it's a 1-based index, convert to 0-based
                                elif 1 <= ch_idx <= len(recording_channel_ids):
                                    train_channel_indices.append(ch_idx - 1)
                            else:
                                ch_name_str = str(ch_item).strip()
                                recording_channel_ids_str = [str(ch) for ch in recording_channel_ids]
                                try:
                                    ch_idx = recording_channel_ids_str.index(ch_name_str)
                                    train_channel_indices.append(ch_idx)
                                except ValueError:
                                    try:
                                        ch_idx = recording_channel_ids.index(ch_name_str)
                                        train_channel_indices.append(ch_idx)
                                    except ValueError:
                                        continue
                        except:
                            continue
                    
                    if len(train_channel_indices) == 0:
                        continue
                    
                    # Extract channels corresponding to train neuron channel_id
                    valid_channel_id = [ch_idx for ch_idx in train_channel_indices if 0 <= ch_idx < n_channels]
                    if len(valid_channel_id) == 0:
                        continue
                    
                    # Use cluster_per_channel_waveform: (window_size, n_channels)
                    # Extract columns corresponding to valid_channel_id
                    cluster_per_channel_waveform = cluster_per_channel_waveforms[cluster_id]  # (window_size, n_channels)
                    
                    # Calculate position and waveform from average per_channel_waveform
                    position_1, position_2, position_waveform = compute_position_waveform_from_average(
                        cluster_per_channel_waveform, valid_channel_id, channel_positions_dict, window_size
                    )
                    
                    # Calculate position distance
                    cluster_pos = np.array([position_1, position_2])
                    pos_distance = np.linalg.norm(cluster_pos - train_pos)
                    if pos_distance >= position_threshold:
                        continue
                    
                    # Calculate waveform similarity
                    min_len = min(len(position_waveform), len(train_waveform))
                    if min_len == 0:
                        continue
                    corr, _ = pearsonr(position_waveform[:min_len], train_waveform[:min_len])
                    
                    if corr < waveform_similarity_threshold:
                        continue
                    
                    # Calculate comprehensive score
                    score = corr / (1 + pos_distance / position_threshold)
                    
                    if score > best_score:
                        best_score = score
                        best_match = train_neuron
                        best_match_features = {
                            'position_1': position_1,
                            'position_2': position_2,
                            'position_waveform': position_waveform,
                            'n_spikes': len(cluster_spike_indices_in_waveforms),
                            'matched_neuron': train_neuron,
                            'score': score,
                            'pos_distance': pos_distance,
                            'waveform_corr': corr,
                        }
                
                # Establish mapping if best match found
                if best_match is not None:
                    cluster_to_neuron_mapping[cluster_id] = best_match
                    neuron_to_clusters[best_match].append(cluster_id)
                    cluster_features[cluster_id] = best_match_features
                else:
                    # No match found, mark as noise
                    noise_clusters.add(cluster_id)
                    cluster_noise_spike_indices.extend(cluster_spike_indices_in_waveforms.tolist())
            
            print(f"  Matching completed: {len(cluster_to_neuron_mapping)} clusters matched to neurons")
    
    elif match_mode == 'per_channel_match':
        # Per-channel match mode: cluster spikes per channel, then match
        # Process each channel separately
        for channel_idx in unique_channels:
            # Get spikes for this channel
            channel_mask = spike_channels_array == channel_idx
            # channel_mask is boolean mask for spike_indices array
            # channel_spike_indices_in_waveforms are indices into waveforms array
            channel_spike_indices_in_waveforms = spike_indices[channel_mask]  # Indices in waveforms array
            # channel_spike_indices_in_spike_indices are positions in spike_indices array
            channel_spike_indices_in_spike_indices = np.where(channel_mask)[0]  # Positions in spike_indices array
            channel_way3_features = way3_features[channel_mask]  # Features for this channel's spikes
            
            if len(channel_spike_indices_in_waveforms) == 0:
                continue
            
            # Check if channel has enough spikes for clustering (minimum 30 spikes)
            min_spikes_for_clustering = 30
            if len(channel_spike_indices_in_waveforms) < min_spikes_for_clustering:
                # Not enough spikes, mark all as noise
                channel_id_str = recording_channel_ids[channel_idx] if channel_idx < len(recording_channel_ids) else None
                if channel_id_str is not None:
                    print(f"  Channel {channel_id_str}: {len(channel_spike_indices_in_waveforms)} spikes < {min_spikes_for_clustering}, marking as noise")
                cluster_noise_spike_indices.extend(channel_spike_indices_in_waveforms.tolist())
                continue
            
            # 1. K-means clustering (5 classes for this channel)
            n_clusters_per_channel = 10
            channel_kmeans = KMeans(n_clusters=n_clusters_per_channel, random_state=42, n_init=10)
            channel_cluster_labels = channel_kmeans.fit_predict(channel_way3_features)  # (n_channel_spikes,)
            
            # Map local cluster IDs to global cluster IDs
            for local_cluster_id in range(n_clusters_per_channel):
                local_cluster_mask = channel_cluster_labels == local_cluster_id
                local_cluster_spike_indices_in_waveforms = channel_spike_indices_in_waveforms[local_cluster_mask]
                local_cluster_spike_indices_in_spike_indices = channel_spike_indices_in_spike_indices[local_cluster_mask]
                
                if len(local_cluster_spike_indices_in_waveforms) == 0:
                    # Empty cluster, mark as noise
                    global_cluster_id_for_noise = global_cluster_id
                    noise_clusters.add(global_cluster_id_for_noise)
                    global_cluster_id += 1
                    continue
                
                global_cluster_id_current = global_cluster_id
                global_cluster_id += 1
                
                # Update global cluster labels
                # local_cluster_spike_indices_in_spike_indices are positions in spike_indices array
                all_cluster_labels[local_cluster_spike_indices_in_spike_indices] = global_cluster_id_current
            
            # 2. Match clusters to neurons with matching extremum_channel
            # Find neurons with extremum_channel matching this channel
            channel_id_str = recording_channel_ids[channel_idx] if channel_idx < len(recording_channel_ids) else None
            
            if channel_id_str is None:
                continue
            
            # Find neurons with this extremum_channel
            matching_neurons = train_neuron_inf[train_neuron_inf['extremum_channel'] == channel_id_str]
            
            if len(matching_neurons) == 0:
                # No matching neurons for this channel, mark all spikes as noise
                # But still calculate per_channel_waveform for these clusters
                for local_cluster_id in range(n_clusters_per_channel):
                    local_cluster_mask = channel_cluster_labels == local_cluster_id
                    local_cluster_spike_indices_in_waveforms = channel_spike_indices_in_waveforms[local_cluster_mask]
                    if len(local_cluster_spike_indices_in_waveforms) > 0:
                        # Get waveforms for this cluster
                        cluster_waveforms_full_no_match = waveforms[local_cluster_spike_indices_in_waveforms]  # (n_spikes, n_channels, window_size)
                        # Calculate per_channel_waveform
                        cluster_avg_waveform_per_channel = np.mean(cluster_waveforms_full_no_match, axis=0)  # (n_channels, window_size)
                        cluster_per_channel_waveform = cluster_avg_waveform_per_channel.T  # (window_size, n_channels) = (n_timepoints, n_channels)
                        # Get global cluster ID for this local cluster
                        local_cluster_spike_indices_in_spike_indices = channel_spike_indices_in_spike_indices[local_cluster_mask]
                        if len(local_cluster_spike_indices_in_spike_indices) > 0:
                            global_cluster_id_for_no_match = all_cluster_labels[local_cluster_spike_indices_in_spike_indices[0]]
                            if global_cluster_id_for_no_match >= 0:
                                cluster_per_channel_waveforms[global_cluster_id_for_no_match] = cluster_per_channel_waveform.astype(np.float32)
                        cluster_noise_spike_indices.extend(local_cluster_spike_indices_in_waveforms.tolist())
                continue
            
            # For each cluster in this channel, try to match with neurons
            for local_cluster_id in range(n_clusters_per_channel):
                local_cluster_mask = channel_cluster_labels == local_cluster_id
                local_cluster_spike_indices_in_waveforms = channel_spike_indices_in_waveforms[local_cluster_mask]
                local_cluster_spike_indices_in_spike_indices = channel_spike_indices_in_spike_indices[local_cluster_mask]
                
                if len(local_cluster_spike_indices_in_waveforms) == 0:
                    continue
                
                # Get global cluster ID
                # Use the first spike's position in spike_indices array to get global cluster ID
                if len(local_cluster_spike_indices_in_spike_indices) == 0:
                    continue
                global_cluster_id_current = all_cluster_labels[local_cluster_spike_indices_in_spike_indices[0]]
                
                if global_cluster_id_current == -1:
                    continue
                
                # Get waveforms for this cluster
                cluster_waveforms_full = waveforms[local_cluster_spike_indices_in_waveforms]  # (n_spikes, n_channels, window_size)
                
                # Calculate per_channel_waveform for this cluster: (n_timepoints, n_channels)
                # Average across all spikes in the cluster, then transpose from (n_channels, window_size) to (window_size, n_channels)
                cluster_avg_waveform_per_channel = np.mean(cluster_waveforms_full, axis=0)  # (n_channels, window_size)
                cluster_per_channel_waveform = cluster_avg_waveform_per_channel.T  # (window_size, n_channels) = (n_timepoints, n_channels)
                cluster_per_channel_waveforms[global_cluster_id_current] = cluster_per_channel_waveform.astype(np.float32)
                
                # Try to match with each neuron
                best_match = None
                best_score = -1
                best_match_features = None
                
                for neuron_idx, neuron_row in matching_neurons.iterrows():
                    train_neuron = neuron_row['Neuron']
                    train_pos = np.array([neuron_row['position_1'], neuron_row['position_2']])
                    train_waveform = np.asarray(neuron_row['position_waveform'], dtype=np.float32)
                    
                    # Get train neuron channel_id
                    train_channel_id = neuron_row.get('channel_id', [])
                    if not train_channel_id or len(train_channel_id) == 0:
                        continue
                    
                    # Convert channel_id to clique indices
                    train_channel_indices = []
                    for ch_item in train_channel_id:
                        try:
                            if isinstance(ch_item, (int, np.integer)):
                                ch_idx = int(ch_item)
                                if 0 <= ch_idx < len(recording_channel_ids):
                                    train_channel_indices.append(ch_idx)
                            else:
                                ch_name_str = str(ch_item).strip()
                                recording_channel_ids_str = [str(ch) for ch in recording_channel_ids]
                                try:
                                    ch_idx = recording_channel_ids_str.index(ch_name_str)
                                    train_channel_indices.append(ch_idx)
                                except ValueError:
                                    try:
                                        ch_idx = recording_channel_ids.index(ch_name_str)
                                        train_channel_indices.append(ch_idx)
                                    except ValueError:
                                        continue
                        except:
                            continue
                    
                    if len(train_channel_indices) == 0:
                        continue
                    
                    # Extract channels corresponding to train neuron channel_id
                    valid_channel_id = [ch_idx for ch_idx in train_channel_indices if 0 <= ch_idx < n_channels]
                    if len(valid_channel_id) == 0:
                        continue
                    
                    # Use cluster_per_channel_waveform: (window_size, n_channels)
                    # Extract columns corresponding to valid_channel_id
                    cluster_per_channel_waveform = cluster_per_channel_waveforms[global_cluster_id_current]  # (window_size, n_channels)
                    
                    # Calculate position and waveform from average per_channel_waveform
                    position_1, position_2, position_waveform = compute_position_waveform_from_average(
                        cluster_per_channel_waveform, valid_channel_id, channel_positions_dict, window_size
                    )
                    
                    # Calculate position distance
                    cluster_pos = np.array([position_1, position_2])
                    pos_distance = np.linalg.norm(cluster_pos - train_pos)
                    if pos_distance >= position_threshold:
                        continue
                    
                    # Calculate waveform similarity
                    min_len = min(len(position_waveform), len(train_waveform))
                    if min_len == 0:
                        continue
                    corr, _ = pearsonr(position_waveform[:min_len], train_waveform[:min_len])
                    
                    if corr < waveform_similarity_threshold:
                        continue
                    
                    # Calculate comprehensive score
                    score = corr / (1 + pos_distance / position_threshold)
                    
                    if score > best_score:
                        best_score = score
                        best_match = train_neuron
                        best_match_features = {
                            'position_1': position_1,
                            'position_2': position_2,
                            'position_waveform': position_waveform,
                            'n_spikes': len(local_cluster_spike_indices_in_waveforms),
                            'matched_neuron': train_neuron,
                            'score': score,
                            'pos_distance': pos_distance,
                            'waveform_corr': corr,
                        }
                
                # Establish mapping if best match found
                if best_match is not None:
                    cluster_to_neuron_mapping[global_cluster_id_current] = best_match
                    neuron_to_clusters[best_match].append(global_cluster_id_current)
                    cluster_features[global_cluster_id_current] = best_match_features
                else:
                    # No match found, mark as noise
                    noise_clusters.add(global_cluster_id_current)
                    cluster_noise_spike_indices.extend(local_cluster_spike_indices_in_waveforms.tolist())
    
    # Use all_cluster_labels as cluster_labels for compatibility
    cluster_labels = all_cluster_labels
    
    print("\n### 6. Firing rate filtering")
    # Calculate firing rate for each matched neuron
    invalid_neurons = set()  # Neurons with firing rate < 0.5 Hz
    neuron_firing_rates = {}  # {neuron_name: firing_rate}
    
    for neuron_name, cluster_ids in neuron_to_clusters.items():
        # Count spikes for this neuron
        neuron_spike_count = 0
        for cluster_id in cluster_ids:
            if cluster_id in cluster_features:
                neuron_spike_count += cluster_features[cluster_id]['n_spikes']
        
        # Calculate firing rate (spikes per second)
        firing_rate = neuron_spike_count / calibration_duration_seconds if calibration_duration_seconds > 0 else 0
        neuron_firing_rates[neuron_name] = firing_rate
        
        if firing_rate < 0.3:
            invalid_neurons.add(neuron_name)
            print(f"  Neuron {neuron_name}: firing rate {firing_rate:.4f} Hz < 0.3 Hz, marked as invalid")
    
    # Mark clusters matched to invalid neurons as noise
    n_invalid_clusters = 0
    n_invalid_spikes = 0
    for neuron_name in invalid_neurons:
        # Get cluster_ids before deletion
        cluster_ids = list(neuron_to_clusters[neuron_name])  # Create a copy
        for cluster_id in cluster_ids:
            if cluster_id in cluster_to_neuron_mapping:
                # Remove from mapping
                del cluster_to_neuron_mapping[cluster_id]
                noise_clusters.add(cluster_id)
                n_invalid_clusters += 1
                
                # Add spikes from this cluster to noise
                if cluster_id in cluster_features:
                    n_spikes = cluster_features[cluster_id]['n_spikes']
                    # Find spikes in this cluster
                    cluster_mask = cluster_labels == cluster_id
                    cluster_spike_indices_in_waveforms = spike_indices[cluster_mask]
                    cluster_noise_spike_indices.extend(cluster_spike_indices_in_waveforms.tolist())
                    n_invalid_spikes += n_spikes
        
        # Remove from neuron_to_clusters
        del neuron_to_clusters[neuron_name]
    
    if len(invalid_neurons) > 0:
        print(f"  Marked {n_invalid_clusters} clusters and {n_invalid_spikes} spikes as noise due to low firing rate")
    
    print("\n### 7. Summary")
    if match_mode == 'combined_match':
        n_total_clusters = len(np.unique(all_cluster_labels[all_cluster_labels >= 0])) if len(all_cluster_labels) > 0 else 0
    else:
        n_total_clusters = global_cluster_id
    n_noise_spikes = len(cluster_noise_spike_indices)
    n_spikes_matched = len(spike_indices) - n_noise_spikes
    print(f"Matching results:")
    print(f"  - Total clusters: {n_total_clusters}")
    print(f"  - Matched clusters: {len(cluster_to_neuron_mapping)}")
    print(f"  - Matched neurons: {len(neuron_to_clusters)}")
    print(f"  - Invalid neurons (firing rate < 0.5 Hz): {len(invalid_neurons)}")
    print(f"  - Spikes matched to neurons: {n_spikes_matched}")
    print(f"  - Spikes marked as noise: {n_noise_spikes}")
    print(f"  - Total spikes after noise classifier: {len(spike_indices)}")
    
    # Build results DataFrame (for confusion matrix)
    # 创建noise spikes的set以便快速查找
    noise_spike_set = set(cluster_noise_spike_indices)
    
    predicted_labels = []
    for i in range(len(spike_indices)):
        spike_idx = spike_indices[i]
        cluster_id = cluster_labels[i]
        
        # 如果spike来自noise cluster或非主导channel，标记为noise
        if cluster_id in noise_clusters or spike_idx in noise_spike_set:
            predicted_labels.append('noise')
        else:
            # 否则根据cluster匹配结果，如果没有匹配到neuron，也标记为noise
            predicted_labels.append(cluster_to_neuron_mapping.get(cluster_id, 'noise'))
    
    results_df = pd.DataFrame({
        'spike_time': [spike_times[i] for i in spike_indices],
        'spike_channel': [spike_channels[i] for i in spike_indices],
        'predicted_label': predicted_labels,
    })
    
    # If eval data exists, add GT labels
    # 从gt_detect_array构建spike_inf格式的DataFrame
    eval_spike_data = None
    if gt_detect_array is not None and len(gt_detect_array) > 0:
        eval_spike_data = pd.DataFrame({
            'time': gt_detect_array['time'].values,
            'neuron': gt_detect_array['unit_id'].values
        })
    
    if eval_neuron_inf is not None and eval_spike_data is not None:
        # Establish neuron mapping (from eval_neuron_inf to train_neuron)
        if 'neuron_match' in eval_neuron_inf.columns:
            # Establish mapping from eval neuron to train neuron
            # 如果neuron_match是'unmatch'，不加入mapping（后面找不到时会标记为'noise'）
            eval_to_train_mapping = {}
            for _, row in eval_neuron_inf.iterrows():
                eval_neuron = row['Neuron']
                match = row['neuron_match']
                if match != 'unmatch':
                    eval_to_train_mapping[eval_neuron] = match
                # 如果match是'unmatch'，不加入mapping，后面找不到时会标记为'noise'
            
            # Match GT labels based on spike_time
            # spike_times_array是sample indices，eval_spike_data['time']也是sample indices
            spike_times_array = np.array([spike_times[i] for i in spike_indices])
            spike_inf_sorted = eval_spike_data.sort_values('time').reset_index(drop=True)
            
            gt_labels = []
            for spike_time_sample in spike_times_array:
                # Find corresponding spike in gt_detect_array (allow ±1 sample point error)
                time_diff = (spike_inf_sorted['time'] - spike_time_sample).abs()
                min_diff_idx = time_diff.idxmin()
                min_diff = time_diff.loc[min_diff_idx]
                
                # 允许±1采样点的误差
                tolerance_samples = 1
                if min_diff <= tolerance_samples:
                    eval_neuron = spike_inf_sorted.loc[min_diff_idx, 'neuron']
                    
                    # Map to train neuron
                    if eval_neuron in eval_to_train_mapping:
                        gt_label = eval_to_train_mapping[eval_neuron]
                    else:
                        gt_label = 'noise'  # No matching train neuron, treat as noise
                else:
                    gt_label = 'noise'  # No matching GT spike found, treat as noise
                
                gt_labels.append(gt_label)
            
            results_df['gt_label'] = gt_labels
        else:
            print("Warning: eval_neuron_inf has no neuron_match column, cannot establish GT label mapping")
            results_df['gt_label'] = 'unknown'
    else:
        results_df['gt_label'] = None
    
    # Calculate classification accuracy (for train neurons + noise)
    print("\n### Classification Accuracy Calculation")
    if results_df['gt_label'] is not None and not results_df['gt_label'].isna().all():
        # Get all train neuron IDs (convert to string for consistency)
        train_neuron_ids = set(str(neuron_id) for neuron_id in train_neuron_inf['Neuron'].values)
        # Add 'noise' to the label set
        valid_labels = train_neuron_ids | {'noise'}
        
        # Convert GT and predicted labels to string for consistent comparison
        results_df['gt_label_str'] = results_df['gt_label'].astype(str)
        results_df['predicted_label_str'] = results_df['predicted_label'].astype(str)
        
        # Filter results_df to only include valid labels (train neurons + noise)
        # 理论上不应该有'unmatch'了，所有未匹配的都应该是'noise'
        valid_mask_gt = results_df['gt_label_str'].isin(valid_labels)
        valid_mask_pred = results_df['predicted_label_str'].isin(valid_labels)
        valid_mask = valid_mask_gt & valid_mask_pred
        
        # Get valid GT and predicted labels (as strings)
        valid_gt_labels = results_df.loc[valid_mask, 'gt_label_str'].values
        valid_pred_labels = results_df.loc[valid_mask, 'predicted_label_str'].values
        
        # Get all unique labels (train neurons + noise), sorted for consistency
        all_labels = sorted(list(valid_labels))
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
        
        cm = confusion_matrix(valid_gt_labels, valid_pred_labels, labels=all_labels)
        
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(valid_gt_labels, valid_pred_labels)
        
        # Calculate per-class accuracy (recall) for train neurons + noise
        per_class_accuracy = {}
        for i, label in enumerate(all_labels):
            if label in valid_labels:  # Only calculate for train neurons and noise
                true_positives = cm[i, i]
                total_samples = cm[i, :].sum()
                if total_samples > 0:
                    per_class_accuracy[label] = true_positives / total_samples
                else:
                    per_class_accuracy[label] = 0.0
        
        # Calculate accuracy excluding noise class
        # Filter out noise samples from both GT and predicted labels
        non_noise_mask = (valid_gt_labels != 'noise') & (valid_pred_labels != 'noise')
        if non_noise_mask.sum() > 0:
            non_noise_gt = valid_gt_labels[non_noise_mask]
            non_noise_pred = valid_pred_labels[non_noise_mask]
            accuracy_excluding_noise = accuracy_score(non_noise_gt, non_noise_pred)
        else:
            accuracy_excluding_noise = None
        
        # Print results
        print(f"  Total spikes analyzed: {len(valid_gt_labels)}")
        print(f"  Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        if accuracy_excluding_noise is not None:
            print(f"  Accuracy (excluding noise): {accuracy_excluding_noise:.4f} ({accuracy_excluding_noise*100:.2f}%)")
        
        classification_metrics = {
            'overall_accuracy': overall_accuracy,
            'accuracy_excluding_noise': accuracy_excluding_noise,
            'per_class_accuracy': per_class_accuracy,
            'confusion_matrix': cm,
            'confusion_matrix_labels': all_labels,
            'n_samples': len(valid_gt_labels)
        }
    else:
        print("  Warning: GT labels not available, skipping classification accuracy calculation")
        classification_metrics = None
    
    # Save way3 features for visualization
    # For noise detection: need way3 features and noise classification results for all detected spikes
    all_way3_features_noise = []  # Way3 features for all detected spikes (100 dimensions)
    all_noise_gt_labels = []  # GT noise/spike labels
    all_noise_pred_labels = []  # Predicted noise/spike labels
    
    # Reprocess all detected spikes to get way3 features
    # autosort_model.eval()
    # with torch.no_grad():
    #     for i in tqdm(range(0, len(waveforms), batch_size), desc="Extracting way3 features for all spikes"):
    #         batch_waveforms = waveforms[i:i+batch_size]
    #         batch_channels = spike_channels[i:i+batch_size]
            
    #         batch_single_waveforms = []
    #         batch_multi_waveforms = []
    #         for wf, ch in zip(batch_waveforms, batch_channels):
    #             multi_wf = wf.flatten()
    #             batch_multi_waveforms.append(multi_wf)
    #             single_wf = wf[ch, :]
    #             batch_single_waveforms.append(single_wf)
            
    #         batch_multi_waveforms = np.array(batch_multi_waveforms)
    #         batch_single_waveforms = np.array(batch_single_waveforms)
    #         batch_multi = torch.from_numpy(batch_multi_waveforms).float().to(device)
    #         batch_single = torch.from_numpy(batch_single_waveforms).float().to(device)
    #         batch_channel_indices = torch.from_numpy(np.array(batch_channels)).long().to(device)
            
    #         # Prepare input using model's _prepare_input method
    #         codes = autosort_model._prepare_input(batch_multi, batch_single, batch_channel_indices)
            
    #         # Noise classification
    #         noise_output = autosort_model.clsfier_noise(codes)
    #         noise_pred = torch.argmax(noise_output, dim=1)
            
    #         # Extract way3 features (for all spikes, including those classified as noise)
    #         way3_batch = autosort_model.clsfier_label.intermediate_forward(codes)
    #         all_way3_features_noise.append(way3_batch.cpu().numpy())
    #         all_noise_pred_labels.extend(noise_pred.cpu().numpy().tolist())
            
    #         # Get GT noise labels (if eval data exists)
    #         if eval_spike_data is not None:
    #             batch_spike_times = spike_times[i:i+batch_size]
    #             batch_gt_noise = []
    #             for st in batch_spike_times:
    #                 time_diff = (eval_spike_data['time'] - st).abs()
    #                 if time_diff.min() <= 1:
    #                     batch_gt_noise.append(1)  # spike
    #                 else:
    #                     batch_gt_noise.append(0)  # noise
    #             all_noise_gt_labels.extend(batch_gt_noise)
    #         else:
    #             all_noise_gt_labels.extend([-1] * len(batch_waveforms))  # Unknown
    
    # all_way3_features_noise = np.concatenate(all_way3_features_noise, axis=0) if len(all_way3_features_noise) > 0 else np.array([])
    
    # # Calculate neuron comparison statistics (if eval_neuron_inf is provided)
    matched_neuron_count = len(neuron_to_clusters)
    n_matched_neurons = matched_neuron_count
    
    n_overlapping_neurons = None
    n_disappeared_neurons = None
    n_new_neurons = None
    
    if eval_neuron_inf is not None:
        train_neuron_ids = set(train_neuron_inf['Neuron'].unique())
        eval_neuron_ids = set(eval_neuron_inf['Neuron'].unique())
        
        n_overlapping_neurons = len(train_neuron_ids & eval_neuron_ids)  # 重合的神经元数
        n_disappeared_neurons = len(train_neuron_ids - eval_neuron_ids)  # 消失的神经元数
        n_new_neurons = len(eval_neuron_ids - train_neuron_ids)  # 新出现的神经元数
    
    calibration_results = {
        'kmeans_model': kmeans_model,  # K-means model (None for per-channel mode, actual model for combined mode)
        'cluster_to_neuron_mapping': cluster_to_neuron_mapping,
        'neuron_to_clusters': dict(neuron_to_clusters),
        'cluster_features': cluster_features,
        'cluster_per_channel_waveforms': cluster_per_channel_waveforms,  # {cluster_id: per_channel_waveform (n_timepoints, n_channels)}
        'spike_indices': spike_indices,
        'cluster_labels': cluster_labels,
        'noise_spike_indices': np.array(cluster_noise_spike_indices) if len(cluster_noise_spike_indices) > 0 else np.array([], dtype=np.int64),  # Spikes marked as noise
        'noise_clusters': noise_clusters,  # Set of cluster IDs marked as noise
        'results_df': results_df,  # Add results_df for confusion matrix
        'classification_metrics': classification_metrics,  # Classification accuracy metrics (overall accuracy, per-class accuracy, confusion matrix)
        'way3_features_30d': way3_features,  # Features from intermediate_forward (30 dimensions)
        'way3_features_noise_30d': all_way3_features_noise,  # Features from intermediate_forward for all detected spikes (30 dimensions, for noise detection visualization)
        'noise_gt_labels': np.array(all_noise_gt_labels),  # GT noise/spike labels
        'noise_pred_labels': np.array(all_noise_pred_labels),  # Predicted noise/spike labels
        # Additional statistics
        'n_overlapping_neurons': n_overlapping_neurons,  # 重合的神经元数
        'n_disappeared_neurons': n_disappeared_neurons,  # 消失的神经元数
        'n_new_neurons': n_new_neurons,  # 新出现的神经元数
        'noise_classifier_accuracy': noise_classifier_accuracy,  # Noise classifier准确率
        'n_matched_neurons': n_matched_neurons,  # Matched neuron数
        'accuracy_excluding_noise': classification_metrics.get('accuracy_excluding_noise') if classification_metrics else None,  # 不包含noise的分类准确率
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
    batch_size: int = 2048,  # 优化：增大批处理大小
    verbose: bool = True,  # 优化：控制输出
    save_noise_features: bool = False,  # 优化：是否保存noise visualization数据
):
    """
    Stage 2: Real-time processing (process by time_window)
    
    Process:
    1. Load data by time_window
    2. Threshold detection
    3. Pass through noise classifier, classified as spikes
    4. Extract way3 layer → PCA dimensionality reduction → K-means prediction → Map to train neuron ID
    
    Parameters:
        recording_f: preprocessed recording object
        autosort_model: trained SimpleAutoSort model
        calibration_results: results from calibration stage (contains kmeans_model, cluster_to_neuron_mapping)
        start_time_seconds: start time for processing (seconds), default 60 (after calibration)
        time_window_seconds: length of each time window (seconds), default 10
        total_duration_seconds: total processing duration (seconds), if None process until recording ends, default None
        detection_params: detection parameters dictionary
        window_params: window parameters dictionary
        eval_neuron_inf: evaluation data neuron_inf (for generating GT labels)
        eval_spike_inf: evaluation data spike_inf (for generating GT labels)
        device: device
    
    Returns:
        processing_results: dictionary, containing:
            - spike_predictions: predicted neuron ID for each spike
            - spike_times: time for each spike
            - spike_channels: channel for each spike
            - results_df: DataFrame containing gt_label and predicted_label
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if detection_params is None:
        detection_params = {
            'thr_min': 3.5,
            'thr_max': 30,
            'distance': 3,
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
    
    # Get models and mapping from calibration stage
    kmeans_model = calibration_results['kmeans_model']
    cluster_to_neuron_mapping = calibration_results['cluster_to_neuron_mapping']
    
    if verbose:
        print("=" * 50)
        print("Stage 2: Real-time processing")
        print("=" * 50)
        print(f"Start time: {start_time_seconds} seconds")
        print(f"Time window: {time_window_seconds} seconds")
    
    # Calculate total duration of recording
    total_duration_samples = recording_f.get_num_samples()
    recording_total_seconds = total_duration_samples / sampling_frequency
    start_frame = int(start_time_seconds * sampling_frequency)
    window_frames = int(time_window_seconds * sampling_frequency)
    
    # Calculate end time
    if total_duration_seconds is not None:
        end_time_seconds = start_time_seconds + total_duration_seconds
        end_frame = min(int(end_time_seconds * sampling_frequency), total_duration_samples)
        if verbose:
            print(f"Total processing duration: {total_duration_seconds} seconds (from {start_time_seconds}s to {end_time_seconds}s)")
    else:
        end_frame = total_duration_samples
        if verbose:
            print(f"Processing until recording ends (from {start_time_seconds}s to {recording_total_seconds:.1f}s)")
    
    all_spike_predictions = []
    all_spike_times = []
    all_spike_channels = []
    all_way3_features_30d = []  # Features from intermediate_forward (30 dimensions)
    all_noise_way3_features_30d = []  # Features from intermediate_forward for all detected spikes (30 dimensions, for noise detection visualization)
    all_noise_gt_labels_list = []  # GT noise/spike labels
    all_noise_pred_labels_list = []  # Predicted noise/spike labels
    
    autosort_model.eval()
    
    # Process by time_window
    current_start_frame = start_frame
    window_idx = 0
    
    while current_start_frame < end_frame:
        window_end_frame = min(current_start_frame + window_frames, total_duration_samples)
        window_duration = (window_end_frame - current_start_frame) / sampling_frequency
        
        #print(f"\nProcessing window {window_idx + 1} ({current_start_frame/sampling_frequency:.1f}s - {window_end_frame/sampling_frequency:.1f}s)")
        
        # 1. Load current window data
        traces = recording_f.get_traces(start_frame=current_start_frame, end_frame=window_end_frame)
        if traces.shape[0] > traces.shape[1] and traces.shape[0] > 100:
            traces = traces.T
        traces = traces.astype(np.float32)
        
        # 2. Threshold detection
        trace0_car = traces.T  # (n_timepoints, n_channels)
        spikes = detect_spike(trace0_car, **detection_params)
        spike_coords = np.argwhere(spikes == 1)  # (n_spikes, 2) [time, channel]
        
        if len(spike_coords) == 0:
            if verbose:
                print(f"  Window {window_idx + 1}: No spikes detected")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        # 3. Extract waveforms and filter boundaries (优化：向量化提取)
        # 预过滤：只保留有效的spike坐标
        valid_mask = (spike_coords[:, 0] >= left_sample) & (spike_coords[:, 0] + right_sample <= trace0_car.shape[0])
        valid_spike_coords_filtered = spike_coords[valid_mask]
        
        if len(valid_spike_coords_filtered) == 0:
            if verbose:
                print(f"  Window {window_idx + 1}: No valid spikes after boundary filtering")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        # 向量化提取所有waveforms
        time_indices = valid_spike_coords_filtered[:, 0]
        channel_indices = valid_spike_coords_filtered[:, 1]
        
        # 创建索引数组用于提取waveforms
        n_valid = len(valid_spike_coords_filtered)
        waveforms = np.zeros((n_valid, trace0_car.shape[1], window_size), dtype=np.float32)
        
        for i, time_idx in enumerate(time_indices):
            local_start = time_idx - left_sample
            local_end = time_idx + right_sample
            waveforms[i] = trace0_car[local_start:local_end, :].T  # (n_channels, window_size)
        
        spike_times = (current_start_frame + time_indices).tolist()
        spike_channels = channel_indices.tolist()
        
        if len(waveforms) == 0:
            print(f"  Window {window_idx + 1}: No valid spike waveforms")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size)
        
        # 4. Pass through noise classifier, classified as spikes
        n_spikes = len(waveforms)
        way3_features_list = []
        way3_spike_indices = []  # Record original spike indices corresponding to each way3 feature
        
        # Save way3 features for all spikes (for noise detection visualization) - 可选
        window_noise_way3_features = []
        window_noise_gt_labels = []
        window_noise_pred_labels = []
        
        with torch.no_grad():
            for i in range(0, n_spikes, batch_size):
                batch_end = min(i + batch_size, n_spikes)
                batch_waveforms = waveforms[i:batch_end]
                batch_channels = spike_channels[i:batch_end]
                
                # Extract single waveform and multi waveform (优化：向量化操作)
                # batch_waveforms: (batch_size, n_channels, window_size)
                batch_multi_waveforms = batch_waveforms.reshape(batch_end - i, -1)  # (batch_size, n_channels * window_size)
                # 使用advanced indexing提取single waveforms
                batch_single_waveforms = batch_waveforms[np.arange(batch_end - i), batch_channels, :]  # (batch_size, window_size)
                
                # Convert to tensor
                batch_multi = torch.from_numpy(batch_multi_waveforms).float().to(device)
                batch_single = torch.from_numpy(batch_single_waveforms).float().to(device)
                batch_channel_indices = torch.from_numpy(np.array(batch_channels)).long().to(device)
                
                # Prepare input using model's _prepare_input method
                codes = autosort_model._prepare_input(batch_multi, batch_single, batch_channel_indices)
                
                # Noise classification
                noise_output = autosort_model.clsfier_noise(codes)
                noise_pred = torch.argmax(noise_output, dim=1)
                
                # Extract way3 features for all spikes (including those classified as noise) - 可选
                if save_noise_features:
                    way3_batch_all = autosort_model.clsfier_label.intermediate_forward(codes)
                    window_noise_way3_features.append(way3_batch_all.cpu().numpy())
                    window_noise_pred_labels.extend(noise_pred.cpu().numpy().tolist())
                    
                    # Get GT noise labels (if eval data exists)
                    if eval_neuron_inf is not None and eval_spike_inf is not None:
                        batch_spike_times = [spike_times[i+j] for j in range(batch_end - i)]
                        batch_gt_noise = []
                        for st in batch_spike_times:
                            time_diff = (eval_spike_inf['time'] - st).abs()
                            if time_diff.min() <= 1:
                                batch_gt_noise.append(1)  # spike
                            else:
                                batch_gt_noise.append(0)  # noise
                        window_noise_gt_labels.extend(batch_gt_noise)
                    else:
                        window_noise_gt_labels.extend([-1] * (batch_end - i))  # Unknown
                
                # Keep only samples classified as spikes
                spike_mask = noise_pred == 1
                if spike_mask.sum() > 0:
                    # Record original indices of spikes passing noise classifier
                    batch_spike_indices = np.arange(i, batch_end)[spike_mask.cpu().numpy()]
                    way3_spike_indices.extend(batch_spike_indices.tolist())
                    
                    # Extract way3 layer features (only for spike samples)
                    codes_spike = codes[spike_mask]
                    way3_batch = autosort_model.clsfier_label.intermediate_forward(codes_spike)
                    # 优化：只在最后需要时转换到CPU
                    way3_features_list.append(way3_batch)  # 保持在GPU上
        
        if len(way3_features_list) == 0:
            if verbose:
                print(f"  Window {window_idx + 1}: No spikes passed noise classifier")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        # Combine features for all spikes (优化：在GPU上拼接，最后转换)
        way3_features = torch.cat(way3_features_list, dim=0).cpu().numpy()  # (n_spikes_passed, 30)
        way3_spike_indices = np.array(way3_spike_indices)  # Corresponding original spike indices
        
        # Save way3 features
        all_way3_features_30d.append(way3_features)
        
        # Save way3 features for all detected spikes (for noise detection visualization) - 可选
        if save_noise_features and len(window_noise_way3_features) > 0:
            window_noise_way3_all = np.concatenate(window_noise_way3_features, axis=0)
            all_noise_way3_features_30d.append(window_noise_way3_all)
            all_noise_gt_labels_list.extend(window_noise_gt_labels)
            all_noise_pred_labels_list.extend(window_noise_pred_labels)
        
        # 5. K-means prediction (no PCA needed, intermediate_forward already returns 30-dim features)
        if kmeans_model is not None:
            cluster_labels = kmeans_model.predict(way3_features)  # (n_spikes_passed,)
        else:
            # If kmeans_model is None, generate random cluster labels
            # Get available cluster IDs from cluster_to_neuron_mapping
            if cluster_to_neuron_mapping is not None and len(cluster_to_neuron_mapping) > 0:
                available_clusters = list(cluster_to_neuron_mapping.keys())
                n_spikes = way3_features.shape[0]
                cluster_labels = np.random.choice(available_clusters, size=n_spikes)
            else:
                # If no mapping available, just use random integers
                n_spikes = way3_features.shape[0]
                cluster_labels = np.random.randint(0, 10, size=n_spikes)
        
        # 6. Map to train neuron ID
        neuron_predictions = []
        if cluster_to_neuron_mapping is not None:
            for cluster_id in cluster_labels:
                if cluster_id in cluster_to_neuron_mapping:
                    neuron_predictions.append(cluster_to_neuron_mapping[cluster_id])
                else:
                    neuron_predictions.append('unmatch')
        else:
            # If no mapping available, just use 'unmatch' for all
            neuron_predictions = ['unmatch'] * len(cluster_labels)
        
        # Use way3_spike_indices to get corresponding spike times and channels
        valid_spike_times = [spike_times[i] for i in way3_spike_indices]
        valid_spike_channels = [spike_channels[i] for i in way3_spike_indices]
        valid_neuron_predictions = neuron_predictions  # Already corresponds to spikes passing noise classifier
        
        all_spike_predictions.extend(valid_neuron_predictions)
        all_spike_times.extend(valid_spike_times)
        all_spike_channels.extend(valid_spike_channels)
        
        # if verbose:
        #     print(f"  Window {window_idx + 1}: {len(valid_spike_times)} spikes")
        #     print(f"    - Matched neurons: {sum(1 for p in valid_neuron_predictions if p != 'unmatch')}")
        #     print(f"    - Unmatched: {sum(1 for p in valid_neuron_predictions if p == 'unmatch')}")
        
        # Move to next window
        current_start_frame = window_end_frame
        window_idx += 1
    
    # Build results DataFrame
    results_df = pd.DataFrame({
        'spike_time': all_spike_times,
        'spike_channel': all_spike_channels,
        'predicted_label': all_spike_predictions,
    })
    
    # If eval data exists, add GT labels
    if eval_neuron_inf is not None and eval_spike_inf is not None:
        # Establish neuron mapping (from eval_neuron_inf to train_neuron)
        # Need to establish mapping based on previous match_neurons results
        # Assume eval_neuron_inf already has neuron_match column (from match_neurons function)
        if 'neuron_match' in eval_neuron_inf.columns:
            # Establish mapping from eval neuron to train neuron
            # 如果neuron_match是'unmatch'，不加入mapping（后面找不到时会标记为'noise'）
            eval_to_train_mapping = {}
            for _, row in eval_neuron_inf.iterrows():
                eval_neuron = row['Neuron']
                match = row['neuron_match']
                if match != 'unmatch':
                    eval_to_train_mapping[eval_neuron] = match
                # 如果match是'unmatch'，不加入mapping，后面找不到时会标记为'noise'
            
            # Match GT labels based on spike_time
            # For efficiency, first build spike_inf index
            spike_inf_sorted = eval_spike_inf.sort_values('time').reset_index(drop=True)
            
            gt_labels = []
            for spike_time in all_spike_times:
                # Find corresponding spike in eval_spike_inf (allow ±1 sample point error)
                # Use binary search for efficiency
                time_diff = (spike_inf_sorted['time'] - spike_time).abs()
                min_diff_idx = time_diff.idxmin()
                min_diff = time_diff.loc[min_diff_idx]
                
                if min_diff <= 1:  # Allow ±1 sample point error
                    eval_neuron = spike_inf_sorted.loc[min_diff_idx, 'neuron']
                    
                    # Map to train neuron
                    if eval_neuron in eval_to_train_mapping:
                        gt_label = eval_to_train_mapping[eval_neuron]
                    else:
                        gt_label = 'noise'  # No matching train neuron, treat as noise
                else:
                    gt_label = 'noise'  # No matching GT spike found, treat as noise
                
                gt_labels.append(gt_label)
            
            results_df['gt_label'] = gt_labels
        else:
            print("Warning: eval_neuron_inf has no neuron_match column, cannot establish GT label mapping")
            results_df['gt_label'] = 'unknown'
    else:
        results_df['gt_label'] = None
    
    # Combine all way3 features
    if len(all_way3_features_30d) > 0:
        all_way3_features_30d_combined = np.concatenate(all_way3_features_30d, axis=0)
    else:
        all_way3_features_30d_combined = np.array([])
    
    if len(all_noise_way3_features_30d) > 0:
        all_noise_way3_features_30d_combined = np.concatenate(all_noise_way3_features_30d, axis=0)
        all_noise_gt_labels_combined = np.array(all_noise_gt_labels_list)
        all_noise_pred_labels_combined = np.array(all_noise_pred_labels_list)
    else:
        all_noise_way3_features_30d_combined = np.array([])
        all_noise_gt_labels_combined = np.array([])
        all_noise_pred_labels_combined = np.array([])
    
    processing_results = {
        'spike_predictions': all_spike_predictions,
        'spike_times': all_spike_times,
        'spike_channels': all_spike_channels,
        'results_df': results_df,
        'way3_features_30d': all_way3_features_30d_combined,  # Features from intermediate_forward (30 dimensions)
        'way3_features_noise_30d': all_noise_way3_features_30d_combined,  # Features from intermediate_forward for all detected spikes (30 dimensions, for noise detection visualization)
        'noise_gt_labels': all_noise_gt_labels_combined,  # GT noise/spike labels
        'noise_pred_labels': all_noise_pred_labels_combined,  # Predicted noise/spike labels
    }
    
    # print(f"\nProcessing completed:")
    # print(f"  - Total spikes: {len(all_spike_predictions)}")
    # print(f"  - Matched neurons: {sum(1 for p in all_spike_predictions if p != 'unmatch')}")
    # print(f"  - Unmatched: {sum(1 for p in all_spike_predictions if p == 'unmatch')}")
    
    return processing_results


def generate_confusion_matrix_df(
    results_df: pd.DataFrame,
    train_neuron_list: list = None,
):
    """
    Generate confusion matrix DataFrame
    
    Parameters:
        results_df: DataFrame containing gt_label and predicted_label
        train_neuron_list: train neuron list (for sorting)
    
    Returns:
        confusion_df: confusion matrix DataFrame
        summary_df: detailed DataFrame containing gt_label and predicted_label
    """
    # Ensure gt_label and predicted_label columns exist
    if 'gt_label' not in results_df.columns or 'predicted_label' not in results_df.columns:
        raise ValueError("results_df must contain 'gt_label' and 'predicted_label' columns")
    
    # Create summary DataFrame (all spikes passing noise classifier)
    summary_df = results_df[['gt_label', 'predicted_label']].copy()
    
    # Generate confusion matrix
    confusion_matrix = pd.crosstab(
        summary_df['gt_label'], 
        summary_df['predicted_label'], 
        margins=True
    )
    
    # If train_neuron_list exists, sort by specified order
    if train_neuron_list is not None:
        # Get all unique labels
        all_gt_labels = sorted(summary_df['gt_label'].unique())
        all_pred_labels = sorted(summary_df['predicted_label'].unique())
        
        # Sort by train_neuron_list, then add unmatch and noise
        ordered_labels = []
        for label in train_neuron_list:
            if label in all_gt_labels or label in all_pred_labels:
                ordered_labels.append(label)
        
        # Add unmatch and noise (if they exist)
        for label in ['unmatch', 'noise']:
            if label in all_gt_labels or label in all_pred_labels:
                if label not in ordered_labels:
                    ordered_labels.append(label)
        
        # Reorder confusion matrix
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
    Recalculate confusion matrix and accuracy for noise detection
    
    Note:
    - In calibration stage, after noise classifier, samples classified as spikes enter subsequent clustering and matching
    - If GT=noise but misclassified as spike by noise classifier, after clustering matching may be classified as unmatch
    - Therefore, GT=noise and predicted=unmatch should be considered true negative (TN) for noise detection
    
    Important note:
    - This function only calculates samples classified as spikes after noise classifier (i.e., all samples in results_df)
    - Different from noise classification accuracy in evaluate_autosort_model:
      * evaluate_autosort_model: includes all detected spikes (including those classified as noise), sample count=448248
      * compute_noise_detection_metrics: only includes samples classified as spikes, sample count=37503 (calibration stage)
    - Therefore these two accuracies cannot be directly compared due to different calculation bases
    - Accuracy decrease is normal, as this only focuses on subsequent processing results of noise samples misclassified as spikes
    
    Parameters:
        results_df: DataFrame containing gt_label and predicted_label (only includes samples classified as spikes after noise classifier)
        train_neuron_list: train neuron list (for determining which are train neurons)
    
    Returns:
        noise_detection_metrics: dictionary, containing:
            - confusion_matrix: noise detection confusion matrix (2x2)
            - TP, TN, FP, FN: true positive, true negative, false positive, false negative
            - accuracy: accuracy
            - precision: precision
            - recall: recall
            - f1_score: F1 score
            - specificity: specificity
    """
    if 'gt_label' not in results_df.columns or 'predicted_label' not in results_df.columns:
        raise ValueError("results_df must contain 'gt_label' and 'predicted_label' columns")
    
    # Determine which are train neurons
    if train_neuron_list is None:
        # Infer train neuron list from results_df
        all_gt_labels = set(results_df['gt_label'].unique())
        all_pred_labels = set(results_df['predicted_label'].unique())
        train_neuron_list = sorted([l for l in (all_gt_labels | all_pred_labels) 
                                   if l not in ['noise', 'unmatch', 'unknown']])
    
    # Convert GT labels to noise/spike binary classification
    # GT=noise -> noise
    # GT=train_neuron or unmatch -> spike (these are all true spikes, just may not have matched to train neuron)
    def get_gt_noise_label(gt_label):
        if gt_label == 'noise':
            return 'noise'
        elif gt_label in train_neuron_list or gt_label == 'unmatch':
            return 'spike'
        else:
            return 'unknown'
    
    # Convert predicted labels to noise/spike binary classification
    # predicted=unmatch -> noise (including cases where GT=noise misclassified as spike then classified as unmatch)
    # predicted=train_neuron -> spike
    def get_pred_noise_label(pred_label):
        if pred_label == 'unmatch':
            return 'noise'
        elif pred_label in train_neuron_list:
            return 'spike'
        else:
            return 'unknown'
    
    # Create binary classification labels for noise detection
    noise_detection_df = results_df.copy()
    noise_detection_df['gt_noise'] = noise_detection_df['gt_label'].apply(get_gt_noise_label)
    noise_detection_df['pred_noise'] = noise_detection_df['predicted_label'].apply(get_pred_noise_label)
    
    # Filter out unknown samples
    noise_detection_df = noise_detection_df[
        (noise_detection_df['gt_noise'] != 'unknown') & 
        (noise_detection_df['pred_noise'] != 'unknown')
    ]
    
    # Calculate confusion matrix
    confusion_matrix = pd.crosstab(
        noise_detection_df['gt_noise'],
        noise_detection_df['pred_noise'],
        margins=True
    )
    
    # Ensure there are noise and spike rows and columns
    for label in ['noise', 'spike']:
        if label not in confusion_matrix.index:
            confusion_matrix.loc[label] = 0
        if label not in confusion_matrix.columns:
            confusion_matrix[label] = 0
    
    # Reorder
    confusion_matrix = confusion_matrix.reindex(
        index=['noise', 'spike', 'All'] if 'All' in confusion_matrix.index else ['noise', 'spike'],
        columns=['noise', 'spike', 'All'] if 'All' in confusion_matrix.columns else ['noise', 'spike'],
        fill_value=0
    )
    
    # Calculate TP, TN, FP, FN
    # TP: GT=spike, Pred=spike
    TP = confusion_matrix.loc['spike', 'spike'] if 'spike' in confusion_matrix.index and 'spike' in confusion_matrix.columns else 0
    
    # TN: GT=noise, Pred=noise (including predicted=unmatch cases)
    TN = confusion_matrix.loc['noise', 'noise'] if 'noise' in confusion_matrix.index and 'noise' in confusion_matrix.columns else 0
    
    # FP: GT=noise, Pred=spike (GT=noise misclassified as spike and matched to train neuron)
    FP = confusion_matrix.loc['noise', 'spike'] if 'noise' in confusion_matrix.index and 'spike' in confusion_matrix.columns else 0
    
    # FN: GT=spike, Pred=noise (GT=spike but predicted as unmatch)
    FN = confusion_matrix.loc['spike', 'noise'] if 'spike' in confusion_matrix.index and 'noise' in confusion_matrix.columns else 0
    
    # Calculate various metrics
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

