#!/usr/bin/env python3
"""
Generate the consolidated neuron information table (`neuron_inf.pkl`) from
manually curated Kilosort outputs.

The pipeline reads manually curated results from phy_folder_for_kilosort
directory and generates:
- cluster_inf.csv: cluster-level information
- spike_inf.tsv: spike-level information with neuron assignment
- neuron_inf.pkl: neuron-level information (after deduplication)

Position and waveform calculations follow the same logic as train_spike_pipeline.py and eval_spike_pipeline.py:
- Extract spike windows (30 timepoints) from raw recording data: [spike_time - 10, spike_time + 19]
- Use cluster's channel_id to extract corresponding channels
- Calculate position and position_waveform using IDW interpolation
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from probeinterface import read_probeinterface
from scipy.stats import pearsonr
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CHANNEL_POSITION: Dict[int, Tuple[float, float]] = {
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

# 与train_spike_pipeline.py和eval_spike_pipeline.py保持一致
LEFT_SAMPLE = 10   # spike前10个采样点
RIGHT_SAMPLE = 20  # spike后20个采样点
WINDOW_SIZE = 30   # 总共30个采样点: [spike_time - 10, spike_time + 19]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def setup_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def load_recording(raw_file: Path, probe_file: Path) -> any:
    """Load and preprocess recording data"""
    logging.info("Loading raw recording from %s", raw_file)
    recording_raw = se.read_blackrock(file_path=str(raw_file))
    recording_recorded = recording_raw.remove_channels(["98", "31", "32"])
    recording_recorded = recording_recorded.set_probegroup(read_probeinterface(str(probe_file)))
    recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
    recording_cmr = spre.common_reference(recording_f, reference="global", operator="median")
    return recording_cmr


def find_recording_file(session_date: str, recording_root: Path) -> Optional[Path]:
    """
    根据session日期查找对应的recording文件
    
    Args:
        session_date: session目录名（例如 "012123"）
        recording_root: recording文件根目录（例如 "/media/ubuntu/sda/data/mouse6/ns4/natural_image"）
    
    Returns:
        找到的recording文件路径，如果未找到则返回None
    """
    # 支持多种文件扩展名
    extensions = [".ns4", ".ns6"]
    
    for ext in extensions:
        # 尝试匹配格式：mouse6_{date}_natural_image_001.{ext}
        pattern = f"*{session_date}*{ext}"
        matching_files = list(recording_root.glob(pattern))
        
        if matching_files:
            # 如果有多个匹配，选择第一个
            return matching_files[0]
    
    return None


def compute_best_channels_from_templates(phy_dir: Path, probe) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    从templates.npy和template_ind.npy计算每个cluster的best_channels和tract_channel
    
    Returns:
        tuple: (best_channels_dict, tract_channels_dict)
            best_channels_dict: {si_unit_id: [channel_ids]} - 所有有效通道
            tract_channels_dict: {si_unit_id: channel_id} - RMS最大的通道
    """
    templates_path = phy_dir / "templates.npy"
    template_ind_path = phy_dir / "template_ind.npy"
    
    if not all(p.exists() for p in [templates_path, template_ind_path]):
        logging.warning("缺少templates.npy或template_ind.npy文件，无法计算best_channels")
        return {}, {}
    
    templates = np.load(templates_path)  # shape: (n_templates, n_timepoints, n_channels)
    template_ind = np.load(template_ind_path)  # shape: (n_templates, n_channels)
    
    best_channels_dict = {}
    tract_channels_dict = {}
    
    for template_id in range(templates.shape[0]):
        template = templates[template_id]  # shape: (n_timepoints, n_channels)
        template_channels = template_ind[template_id]  # shape: (n_channels,)
        
        # 过滤掉-1的通道
        valid_channels = template_channels[template_channels != -1]
        if len(valid_channels) == 0:
            continue
        
        # 获取有效通道的波形数据
        valid_template = template[:, template_channels != -1]  # shape: (n_timepoints, n_valid_channels)
        
        # 计算每个通道的波形幅度（使用RMS）
        channel_amplitudes = np.sqrt(np.mean(valid_template**2, axis=0))
        
        # best_channels就是所有有效通道
        best_channels_dict[template_id] = valid_channels.tolist()
        
        # tract_channel是RMS最大的那个通道
        max_amplitude_idx = np.argmax(channel_amplitudes)
        tract_channels_dict[template_id] = int(valid_channels[max_amplitude_idx])
    
    return best_channels_dict, tract_channels_dict


def load_cluster_info_phy(phy_dir: Path, probe) -> pd.DataFrame:
    """读取 phy_folder_for_kilosort/cluster_info.tsv 为 DataFrame，并添加best_channels和tract_channel列"""
    cluster_info_path = phy_dir / "cluster_info.tsv"
    
    if cluster_info_path.exists():
        df = pd.read_csv(cluster_info_path, sep='\t')
        if 'cluster_id' not in df.columns:
            raise ValueError(f"{cluster_info_path} 中缺少 cluster_id 列")
    else:
        logging.warning(f"{cluster_info_path} 不存在，尝试从其他文件构建cluster信息")
        
        cluster_group_path = phy_dir / "cluster_group.tsv"
        if cluster_group_path.exists():
            df = pd.read_csv(cluster_group_path, sep='\t')
            if 'cluster_id' not in df.columns:
                raise ValueError(f"{cluster_group_path} 中缺少 cluster_id 列")
        else:
            spike_clusters_path = phy_dir / "spike_clusters.npy"
            if spike_clusters_path.exists():
                spike_clusters = np.load(spike_clusters_path)
                unique_clusters = np.unique(spike_clusters)
                df = pd.DataFrame({
                    'cluster_id': unique_clusters,
                    'group': 'unsorted'
                })
            else:
                raise ValueError(f"无法找到任何cluster信息文件: {phy_dir}")
    
    # 如果cluster_info.tsv中没有si_unit_id，尝试从cluster_si_unit_ids.tsv读取
    if 'si_unit_id' not in df.columns:
        si_unit_id_path = phy_dir / "cluster_si_unit_ids.tsv"
        if si_unit_id_path.exists():
            si_unit_df = pd.read_csv(si_unit_id_path, sep='\t')
            df = df.merge(si_unit_df[['cluster_id', 'si_unit_id']], on='cluster_id', how='left')
        else:
            df['si_unit_id'] = df['cluster_id']
    
    # 如果best_channels或tract_channel列不存在，从templates计算
    if 'best_channels' not in df.columns or 'tract_channel' not in df.columns:
        logging.info("best_channels or tract_channel column not found, computing from templates...")
        best_channels_dict, tract_channels_dict = compute_best_channels_from_templates(phy_dir, probe)
        
        if 'best_channels' not in df.columns:
            df['best_channels'] = None
        if 'tract_channel' not in df.columns:
            df['tract_channel'] = None
            
        for idx, row in df.iterrows():
            si_unit_id = row.get('si_unit_id', row['cluster_id'])
            if pd.isna(si_unit_id):
                si_unit_id = row['cluster_id']
            else:
                si_unit_id = int(si_unit_id)
            
            if si_unit_id in best_channels_dict:
                df.at[idx, 'best_channels'] = str(best_channels_dict[si_unit_id])
            if si_unit_id in tract_channels_dict:
                df.at[idx, 'tract_channel'] = int(tract_channels_dict[si_unit_id])
    
    return df


def load_spike_level_phy(phy_dir: Path) -> pd.DataFrame:
    """读取 spike 层面的 numpy 文件并返回 DataFrame: [cluster_id, time]"""
    spike_clusters_path = phy_dir / "spike_clusters.npy"
    spike_times_path = phy_dir / "spike_times.npy"
    
    spike_clusters = np.load(spike_clusters_path)
    spike_times = np.load(spike_times_path)
    # 展平为一维
    spike_clusters = np.asarray(spike_clusters).reshape(-1)
    spike_times = np.asarray(spike_times).reshape(-1)
    if spike_clusters.shape[0] != spike_times.shape[0]:
        raise ValueError(f"spike_clusters 与 spike_times 行数不一致: {phy_dir}")
    df = pd.DataFrame({
        'cluster_id': spike_clusters.astype(int),
        'time': spike_times.astype(int),
    })
    return df


def parse_best_channels(best_channels_str: str) -> List[int]:
    """解析best_channels字符串为整数列表"""
    import ast
    if pd.isna(best_channels_str) or best_channels_str is None:
        return []
    try:
        if isinstance(best_channels_str, str):
            # 尝试解析字符串形式的列表，例如 "[1, 3, 5]" 或 "1, 3, 5"
            channels = ast.literal_eval(best_channels_str)
        else:
            channels = best_channels_str
        if isinstance(channels, (list, tuple, np.ndarray)):
            return [int(ch) for ch in channels]
        else:
            return [int(channels)]
    except:
        logging.warning(f"Failed to parse best_channels: {best_channels_str}")
        return []


def compute_cluster_features_from_snippets(
    snippets: np.ndarray,
    channel_id: List[int],
) -> Tuple[float, float, np.ndarray]:
    """
    从已经提取的snippets计算cluster的position和position_waveform
    snippets shape: (n_spikes, n_channels, window_size)
    
    Returns:
        position_1, position_2, position_waveform (30-dim)
    """
    window_size = snippets.shape[2] if len(snippets.shape) == 3 else WINDOW_SIZE
    
    cluster_positions_x = []
    cluster_positions_y = []
    cluster_waveforms = []
    
    for snippet in snippets:  # snippet: (n_channels, window_size)
        # 计算该spike的位置（基于channel_id的通道）
        a_squared = [np.sum(snippet[j, :]**2) for j in range(len(channel_id))]
        
        sum_x_a = 0
        sum_y_a = 0
        sum_a = 0
        
        for j, channel in enumerate(channel_id):
            x_i, y_i = CHANNEL_POSITION.get(channel, [0, 0])
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
        for channel in channel_id:
            x_channel, y_channel = CHANNEL_POSITION.get(channel, [np.nan, np.nan])
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


def extract_neuron_index(neuron_label: str) -> int:
    """从neuron label中提取索引"""
    match = re.search(r"(\d+)", str(neuron_label))
    return int(match.group(1)) if match else 0


def deduplicate_neurons(
    neuron_inf: pd.DataFrame,
    position_threshold: float,
    waveform_threshold: float,
) -> pd.DataFrame:
    """去重neuron，合并相似的neuron"""
    if neuron_inf.empty:
        return neuron_inf

    neuron_inf = neuron_inf.copy()
    # 确保cluster列是object类型，以便可以存储列表
    neuron_inf["cluster"] = neuron_inf["cluster"].astype(object)
    
    neuron_inf["neuron_index"] = neuron_inf["Neuron"].apply(extract_neuron_index)
    neuron_inf.sort_values("neuron_index", inplace=True)
    neuron_inf.reset_index(drop=True, inplace=True)

    keep_mask = np.ones(len(neuron_inf), dtype=bool)
    # 使用列表来存储cluster值，避免pandas的索引问题
    cluster_values = neuron_inf["cluster"].tolist()

    for i in range(len(neuron_inf)):
        if not keep_mask[i]:
            continue
        pos_i = neuron_inf.loc[i, ["position_1", "position_2"]].to_numpy(dtype=float)
        waveform_i = neuron_inf.loc[i, "position_waveform"]
        waveform_i = np.asarray(waveform_i, dtype=np.float32)
        clusters_i = cluster_values[i]

        for j in range(i + 1, len(neuron_inf)):
            if not keep_mask[j]:
                continue
            pos_j = neuron_inf.loc[j, ["position_1", "position_2"]].to_numpy(dtype=float)
            dist = float(np.linalg.norm(pos_i - pos_j))
            if dist >= position_threshold:
                continue

            waveform_j = np.asarray(neuron_inf.loc[j, "position_waveform"], dtype=np.float32)
            min_len = min(len(waveform_i), len(waveform_j))
            if min_len == 0:
                continue
            corr, _ = pearsonr(waveform_i[:min_len], waveform_j[:min_len])
            if corr > waveform_threshold:
                # 合并j到i
                keep_mask[j] = False
                clusters_j = cluster_values[j]
                # 合并cluster列表
                if isinstance(clusters_i, list):
                    if isinstance(clusters_j, list):
                        merged_clusters = clusters_i + clusters_j
                    else:
                        merged_clusters = clusters_i + [clusters_j]
                else:
                    if isinstance(clusters_j, list):
                        merged_clusters = [clusters_i] + clusters_j
                    else:
                        merged_clusters = [clusters_i, clusters_j]
                # 更新列表中的值
                cluster_values[i] = merged_clusters
                clusters_i = merged_clusters
    
    # 最后一次性更新DataFrame
    neuron_inf["cluster"] = cluster_values

    removed = np.count_nonzero(~keep_mask)
    if removed:
        logging.info("Deduplicated %d neurons based on position/waveform similarity", removed)
    neuron_inf = neuron_inf[keep_mask].drop(columns=["neuron_index"]).reset_index(drop=True)
    return neuron_inf


# -----------------------------------------------------------------------------
# Template computation functions
# -----------------------------------------------------------------------------

def compute_neuron_templates(
    neuron_inf: pd.DataFrame,
    spike_inf: pd.DataFrame,
    traces: np.ndarray,
    n_spikes_per_neuron: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为每个neuron计算形状模板和能量向量
    
    Args:
        neuron_inf: neuron信息DataFrame，包含Neuron, channel_id, cluster等列
        spike_inf: spike信息DataFrame，包含neuron, cluster, time等列
        traces: 记录数据，shape为(n_channels, n_timepoints)
        n_spikes_per_neuron: 每个neuron使用的spike数量（默认5000）
    
    Returns:
        shape_templates: (n_neurons, 30, 30) 形状模板
        energy_templates: (n_neurons, 30) 能量向量
    """
    logging.info("Computing neuron templates...")
    
    left_sample = LEFT_SAMPLE   # 10
    right_sample = RIGHT_SAMPLE # 20
    window_size = WINDOW_SIZE   # 30
    n_channels, max_frame = traces.shape
    
    shape_templates_list = []
    energy_templates_list = []
    
    for idx, row in neuron_inf.iterrows():
        neuron_name = row['Neuron']
        channel_id = row['channel_id']
        if not isinstance(channel_id, list):
            if isinstance(channel_id, (np.ndarray, tuple)):
                channel_id = list(channel_id)
            else:
                logging.warning(f"Neuron {neuron_name} has invalid channel_id, skipping")
                # 创建空的模板
                shape_template = np.zeros((30, 30), dtype=np.float32)
                energy_template = np.zeros(30, dtype=np.float32)
                shape_templates_list.append(shape_template)
                energy_templates_list.append(energy_template)
                continue
        
        if len(channel_id) == 0:
            logging.warning(f"Neuron {neuron_name} has no valid channels, skipping")
            shape_template = np.zeros((30, 30), dtype=np.float32)
            energy_template = np.zeros(30, dtype=np.float32)
            shape_templates_list.append(shape_template)
            energy_templates_list.append(energy_template)
            continue
        
        # 获取该neuron对应的所有cluster
        clusters = row['cluster']
        if isinstance(clusters, list):
            cluster_ids = clusters
        else:
            cluster_ids = [clusters]
        
        # 从spike_inf中找到该neuron的所有spike
        neuron_spikes = spike_inf[spike_inf['neuron'] == neuron_name].copy()
        if len(neuron_spikes) == 0:
            logging.warning(f"Neuron {neuron_name} has no spikes, skipping")
            shape_template = np.zeros((30, 30), dtype=np.float32)
            energy_template = np.zeros(30, dtype=np.float32)
            shape_templates_list.append(shape_template)
            energy_templates_list.append(energy_template)
            continue
        
        # 过滤边界附近的spike
        spike_times = neuron_spikes['time'].values
        spike_times = spike_times[
            (spike_times >= left_sample) &
            (spike_times < max_frame - right_sample)
        ]
        
        if len(spike_times) == 0:
            logging.warning(f"Neuron {neuron_name} has no valid spikes after boundary filtering, skipping")
            shape_template = np.zeros((30, 30), dtype=np.float32)
            energy_template = np.zeros(30, dtype=np.float32)
            shape_templates_list.append(shape_template)
            energy_templates_list.append(energy_template)
            continue
        
        # 随机选择spike（最多n_spikes_per_neuron个）
        n_spikes_to_use = min(n_spikes_per_neuron, len(spike_times))
        if n_spikes_to_use < len(spike_times):
            selected_indices = np.random.choice(len(spike_times), n_spikes_to_use, replace=False)
            selected_spike_times = spike_times[selected_indices]
        else:
            selected_spike_times = spike_times
        
        # 提取所有选中spike的波形
        snippets = []
        for spike_time in selected_spike_times:
            start = spike_time - left_sample
            end = spike_time + right_sample
            
            if start < 0 or end > max_frame:
                continue
            if end - start != window_size:
                continue
            
            # 提取该spike的窗口数据
            snippet = traces[:, start:end]  # (30, 30)
            # 只保留best_channels的通道
            snippet_selected = snippet[channel_id, :]  # (n_bestchannels, 30)
            snippets.append(snippet_selected)
        
        if len(snippets) == 0:
            logging.warning(f"Neuron {neuron_name} has no valid snippets after extraction, skipping")
            shape_template = np.zeros((30, 30), dtype=np.float32)
            energy_template = np.zeros(30, dtype=np.float32)
            shape_templates_list.append(shape_template)
            energy_templates_list.append(energy_template)
            continue
        
        # 转换为numpy数组: (n_spikes, n_bestchannels, 30)
        snippets = np.array(snippets, dtype=np.float32)
        n_spikes, n_bestchannels, _ = snippets.shape
        
        # 计算形状模板：对每条通道内的波形做L2归一化，然后取中位数
        normalized_snippets = []
        for i in range(n_spikes):
            snippet = snippets[i]  # (n_bestchannels, 30)
            normalized_snippet = np.zeros_like(snippet)
            for ch_idx in range(n_bestchannels):
                channel_waveform = snippet[ch_idx, :]  # (30,)
                norm = np.linalg.norm(channel_waveform)
                if norm > 1e-10:
                    normalized_snippet[ch_idx, :] = channel_waveform / norm
                else:
                    normalized_snippet[ch_idx, :] = channel_waveform
            normalized_snippets.append(normalized_snippet)
        
        normalized_snippets = np.array(normalized_snippets)  # (n_spikes, n_bestchannels, 30)
        
        # 取中位数得到形状模板: (n_bestchannels, 30)
        shape_template_best = np.median(normalized_snippets, axis=0)  # (n_bestchannels, 30)
        
        # 填充到(30, 30)，非best_channel位置置0
        shape_template = np.zeros((30, 30), dtype=np.float32)
        for ch_idx, ch_id in enumerate(channel_id):
            shape_template[ch_id, :] = shape_template_best[ch_idx, :]
        
        # 计算能量向量：使用未归一化的波形
        # 对每个spike计算每个通道的能量: E = np.sum(template_6ch**2, axis=1)
        energy_per_spike = []
        for i in range(n_spikes):
            snippet = snippets[i]  # (n_bestchannels, 30)
            # 计算每个通道的能量（沿时间轴求和）
            channel_energies = np.sum(snippet**2, axis=1)  # (n_bestchannels,)
            energy_per_spike.append(channel_energies)
        
        energy_per_spike = np.array(energy_per_spike)  # (n_spikes, n_bestchannels)
        
        # 取中位数得到能量向量: (n_bestchannels,)
        energy_template_best = np.median(energy_per_spike, axis=0)  # (n_bestchannels,)
        
        # 填充到(30,)，非best_channel位置置0
        energy_template = np.zeros(30, dtype=np.float32)
        for ch_idx, ch_id in enumerate(channel_id):
            energy_template[ch_id] = energy_template_best[ch_idx]
        
        shape_templates_list.append(shape_template)
        energy_templates_list.append(energy_template)
        
        logging.debug(f"Neuron {neuron_name}: computed template from {len(snippets)} spikes, {len(channel_id)} best channels")
    
    # Stack所有neuron的模板
    shape_templates = np.stack(shape_templates_list, axis=0)  # (n_neurons, 30, 30)
    energy_templates = np.stack(energy_templates_list, axis=0)  # (n_neurons, 30)
    
    logging.info(f"Computed templates for {len(neuron_inf)} neurons: shape {shape_templates.shape}, energy {energy_templates.shape}")
    
    return shape_templates, energy_templates


# -----------------------------------------------------------------------------
# Main processing functions
# -----------------------------------------------------------------------------

def process_session(
    session_dir: Path,
    phy_dir: Path,
    recording: any,
    probe: any,
) -> None:
    """处理单个session，生成cluster_inf.csv, spike_inf.tsv, neuron_inf.pkl"""
    
    logging.info("Processing session %s", session_dir.name)
    
    # 1. 加载cluster和spike信息
    cluster_inf = load_cluster_info_phy(phy_dir, probe)
    spike_inf = load_spike_level_phy(phy_dir)
    
    # 检查cluster_inf中是否包含tract_channel列
    if 'tract_channel' in cluster_inf.columns:
        logging.info("cluster_inf contains tract_channel column")
        n_tract_channels = cluster_inf['tract_channel'].notna().sum()
        logging.info(f"Found {n_tract_channels} clusters with tract_channel")
    else:
        logging.warning("cluster_inf does not contain tract_channel column")
    
    # 只保留group == 'good'的cluster
    if 'group' in cluster_inf.columns:
        cluster_inf = cluster_inf[cluster_inf['group'] == 'good'].copy()
        logging.info("Filtered to %d good clusters", len(cluster_inf))
    
    # 过滤spike_inf
    spike_inf = spike_inf[spike_inf['cluster_id'].isin(cluster_inf['cluster_id'].values)].copy()
    
    # 2. 为每个cluster计算position和position_waveform（从recording数据中提取）
    logging.info("Computing cluster features from recording data...")
    cluster_inf['position_1'] = np.nan
    cluster_inf['position_2'] = np.nan
    cluster_inf['position_waveform'] = pd.Series([None] * len(cluster_inf), dtype=object)
    cluster_inf['channel_id'] = None
    
    # 检查是否有best_channels列
    if 'best_channels' not in cluster_inf.columns:
        logging.warning("best_channels column not found in cluster_inf, cannot compute features")
        return
    
    # 一次性加载所有traces（只加载一次，只处理前10分钟）
    logging.info("Loading traces from recording (first 10 minutes)...")
    sampling_frequency = recording.get_sampling_frequency()
    max_duration_seconds = 10 * 60  # 10分钟
    max_duration_samples = int(max_duration_seconds * sampling_frequency)
    
    traces = recording.get_traces(start_frame=0, end_frame=max_duration_samples).astype(np.float32)
    # get_traces返回的shape是(n_channels, n_timepoints)
    # 确保维度正确：如果是(n_timepoints, n_channels)则转置
    if traces.shape[0] > traces.shape[1] and traces.shape[0] > 100:
        # 如果第一个维度远大于第二个，可能是(n_timepoints, n_channels)，需要转置
        traces = traces.T
        logging.info("Transposed traces to (n_channels, n_timepoints)")
    
    n_channels, max_frame = traces.shape
    # 与train_spike_pipeline.py和eval_spike_pipeline.py保持一致
    left_sample = LEFT_SAMPLE   # 10
    right_sample = RIGHT_SAMPLE # 20
    window_size = WINDOW_SIZE   # 30
    
    logging.info(f"Loaded {max_frame} samples ({max_frame/sampling_frequency:.2f} seconds) from recording, {n_channels} channels")
    logging.info(f"Using window: [spike_time - {left_sample}, spike_time + {right_sample}) = {window_size} timepoints")
    
    # 按cluster分组处理
    for idx, row in cluster_inf.iterrows():
        cluster_id = row['cluster_id']
        cluster_spikes = spike_inf[spike_inf['cluster_id'] == cluster_id]
        
        if len(cluster_spikes) == 0:
            continue
        
        # 从best_channels获取channel_id
        best_channels_str = row.get('best_channels', None)
        channel_id = parse_best_channels(best_channels_str)
        
        if len(channel_id) == 0:
            logging.warning(f"Cluster {cluster_id} has no valid best_channels, skipping")
            continue
        
        spike_times = cluster_spikes['time'].values
        
        # 只处理前10分钟的spike，并过滤边界附近的spike（确保可以提取完整的30个时间点窗口）
        # 与train_spike_pipeline.py和eval_spike_pipeline.py保持一致
        max_spike_time = max_duration_samples
        spike_times = spike_times[
            (spike_times >= left_sample) &  # 确保可以提取spike_time - 10
            (spike_times < max_spike_time - right_sample)  # 确保可以提取spike_time + 20
        ]
        
        if len(spike_times) == 0:
            logging.warning(f"Cluster {cluster_id} has no valid spikes in first 10 minutes (after boundary filtering)")
            continue
        
        # 批量提取该cluster所有spike的窗口
        # 与train_spike_pipeline.py和eval_spike_pipeline.py保持一致：[spike_time - 10, spike_time + 19]，共30个时间点
        snippets = []
        valid_spike_times = []
        
        for spike_time in spike_times:
            start = spike_time - left_sample   # spike_time - 10
            end = spike_time + right_sample    # spike_time + 20
            
            # 检查边界
            if start < 0 or end > max_frame:
                continue
            if end - start != window_size:
                continue
            
            # 提取该spike的窗口数据（使用所有30个通道，稍后只取需要的通道）
            snippet = traces[:, start:end]  # (30, 30)
            # 只保留需要的通道
            snippet_selected = snippet[channel_id, :]  # (n_channels, 30)
            snippets.append(snippet_selected)
            valid_spike_times.append(spike_time)
        
        if len(snippets) == 0:
            logging.warning(f"Cluster {cluster_id} has no valid spikes after boundary check")
            continue
        
        # 转换为numpy数组
        snippets = np.array(snippets)  # (n_valid_spikes, n_channels, 30)
        
        # 使用批量snippets计算position和waveform
        position_1, position_2, position_waveform = compute_cluster_features_from_snippets(
            snippets, channel_id
        )
        
        cluster_inf.at[idx, 'position_1'] = position_1
        cluster_inf.at[idx, 'position_2'] = position_2
        cluster_inf.at[idx, 'position_waveform'] = position_waveform
        cluster_inf.at[idx, 'channel_id'] = str(channel_id)  # 存储为字符串以便CSV保存
    
    # 3. 为每个cluster分配Neuron ID（初始分配，每个cluster一个neuron）
    cluster_inf['Neuron'] = cluster_inf['cluster_id'].apply(lambda x: f"Neuron_{x}")
    
    # 4. 构建neuron_inf（去重前）
    neuron_inf_rows = []
    for neuron, temp in cluster_inf.groupby('Neuron'):
        temp = temp.dropna(subset=['position_waveform'])
        if temp.empty:
            continue
        
        # 计算平均position_waveform
        position_waveforms = []
        for pw in temp['position_waveform']:
            if pw is not None:
                position_waveforms.append(np.asarray(pw, dtype=np.float32))
        
        if not position_waveforms:
            continue
        
        position_waveform = np.stack(position_waveforms).mean(axis=0)
        
        # 获取channel_id（使用第一个cluster的）
        channel_id_str = temp['channel_id'].iloc[0]
        # 解析channel_id字符串
        import ast
        try:
            channel_id = ast.literal_eval(channel_id_str) if isinstance(channel_id_str, str) else channel_id_str
        except:
            # 如果解析失败，尝试从best_channels获取
            best_channels_str = temp['best_channels'].iloc[0] if 'best_channels' in temp.columns else None
            channel_id = parse_best_channels(best_channels_str)
            if len(channel_id) == 0:
                logging.warning(f"Failed to get channel_id for neuron {neuron}, skipping")
                continue
        
        # 获取tract_channel（使用第一个cluster的，如果存在）
        tract_channel = None
        if 'tract_channel' in temp.columns:
            tract_channel_values = temp['tract_channel'].dropna()
            if len(tract_channel_values) > 0:
                # 使用第一个非空的tract_channel值
                tract_channel = int(tract_channel_values.iloc[0])
        
        neuron_inf_row = {
            'Neuron': neuron,
            'position_1': float(temp['position_1'].mean()),
            'position_2': float(temp['position_2'].mean()),
            'position_waveform': position_waveform.astype(np.float32),
            'channel_id': channel_id,
            'cluster': temp['cluster_id'].iloc[0],  # 初始只有一个cluster
        }
        
        # 如果tract_channel存在，添加到neuron_inf中
        if tract_channel is not None:
            neuron_inf_row['tract_channel'] = tract_channel
        
        neuron_inf_rows.append(neuron_inf_row)
    
    neuron_inf = pd.DataFrame(neuron_inf_rows)
    
    # 检查neuron_inf中是否包含tract_channel列
    if 'tract_channel' in neuron_inf.columns:
        n_tract_channels = neuron_inf['tract_channel'].notna().sum()
        logging.info(f"neuron_inf contains tract_channel column: {n_tract_channels}/{len(neuron_inf)} neurons have tract_channel")
    else:
        logging.warning("neuron_inf does not contain tract_channel column")
    
    # 5. 去重neuron
    logging.info("Deduplicating neurons...")
    neuron_inf = deduplicate_neurons(
        neuron_inf,
        position_threshold=10.0,
        waveform_threshold=0.95,
    )
    
    # 6. 更新spike_inf：添加neuron列，并将cluster_id改为cluster
    logging.info("Updating spike_inf with neuron assignments...")
    # 建立cluster_id到neuron的映射
    cluster_to_neuron = {}
    for _, row in neuron_inf.iterrows():
        clusters = row['cluster']
        if isinstance(clusters, list):
            for cluster_id in clusters:
                cluster_to_neuron[cluster_id] = row['Neuron']
        else:
            cluster_to_neuron[clusters] = row['Neuron']
    
    # 添加neuron列
    spike_inf['neuron'] = spike_inf['cluster_id'].map(cluster_to_neuron)
    # 将cluster_id改为cluster
    spike_inf = spike_inf.rename(columns={'cluster_id': 'cluster'})
    
    # 7. 保存文件
    cluster_inf_output = session_dir / "cluster_inf.csv"
    spike_inf_output = session_dir / "spike_inf.tsv"
    neuron_inf_output = session_dir / "neuron_inf.pkl"
    
    # 保存cluster_inf（需要处理channel_id和position_waveform列）
    cluster_inf_save = cluster_inf.copy()
    # 不保存position_waveform到CSV（太复杂），只保存基本信息
    if 'position_waveform' in cluster_inf_save.columns:
        cluster_inf_save = cluster_inf_save.drop(columns=['position_waveform'])
    cluster_inf_save.to_csv(cluster_inf_output, index=False)
    logging.info("Saved cluster_inf to %s", cluster_inf_output)
    
    # 保存spike_inf
    spike_inf.to_csv(spike_inf_output, sep='\t', index=False)
    logging.info("Saved spike_inf to %s", spike_inf_output)
    
    # 保存neuron_inf
    with open(neuron_inf_output, "wb") as f:
        pickle.dump(neuron_inf, f)
    logging.info("Saved neuron_inf to %s (%d neurons)", neuron_inf_output, len(neuron_inf))
    
    # 8. 计算并保存neuron模板（形状模板和能量向量）
    logging.info("Computing neuron templates...")
    shape_templates, energy_templates = compute_neuron_templates(
        neuron_inf=neuron_inf,
        spike_inf=spike_inf,
        traces=traces,
        n_spikes_per_neuron=5000,
    )
    
    # 保存模板
    shape_template_output = session_dir / "shape_templates.npy"
    energy_template_output = session_dir / "energy_templates.npy"
    
    np.save(shape_template_output, shape_templates)
    logging.info("Saved shape_templates to %s (shape: %s)", shape_template_output, shape_templates.shape)
    
    np.save(energy_template_output, energy_templates)
    logging.info("Saved energy_templates to %s (shape: %s)", energy_template_output, energy_templates.shape)


def main() -> None:
    setup_logger(verbose=False)

    parser = argparse.ArgumentParser(
        description="Generate cluster_inf.csv, spike_inf.tsv, and neuron_inf.pkl from manually curated Kilosort outputs"
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        help="Path to raw recording file (e.g., .ns6 file)",
    )
    parser.add_argument(
        "--probe-file",
        type=str,
        default="/media/ubuntu/sda/data/probe.json",
        help="Path to probe file",
    )
    parser.add_argument(
        "--recording-root",
        type=str,
        default="/media/ubuntu/sda/data/mouse11/ns4/natural_image",
        help="Root directory containing recording files",
    )
    args = parser.parse_args()

    base_dir = Path(
        "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels"
    )
    sorting_root = base_dir / "other_mouse"
    probe_file = Path(args.probe_file if args.probe_file else "/media/ubuntu/sda/data/probe.json")
    recording_root = Path(args.recording_root)

    # 加载probe信息
    probe = read_probeinterface(str(probe_file))

    session_dirs = sorted(
        p for p in (sorting_root / "mouse11_ni_sorter_output").iterdir() if p.is_dir()
    )
    if not session_dirs:
        raise FileNotFoundError(f"No session directories found under {sorting_root}/sorting_new")

    for session_dir in session_dirs:
        logging.info("=" * 80)
        logging.info("Processing session %s", session_dir.name)

        phy_dir = session_dir / "phy_folder_for_kilosort"
        if not phy_dir.exists():
            logging.warning("phy_folder_for_kilosort not found for session %s, skipping.", session_dir.name)
            continue

        # 查找对应的recording文件
        session_date = session_dir.name  # session目录名就是日期，例如 "012123"
        
        if args.raw_file:
            # 如果提供了raw_file参数，使用它（适用于所有session共享同一个文件的情况）
            raw_file_path = Path(args.raw_file)
            recording = load_recording(raw_file_path, probe_file)
        else:
            # 根据session日期查找对应的recording文件
            raw_file_path = find_recording_file(session_date, recording_root)
            if raw_file_path is None:
                logging.warning(
                    "No recording file found for session %s in %s, skipping.",
                    session_date,
                    recording_root
                )
                continue
            logging.info("Found recording file: %s", raw_file_path)
            recording = load_recording(raw_file_path, probe_file)

        try:
            process_session(session_dir, phy_dir, recording, probe)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to process session %s: %s", session_dir.name, exc)


if __name__ == "__main__":
    main()
