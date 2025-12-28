#!/usr/bin/env python3
"""
Generate the consolidated neuron information table (`neuron_inf.pkl`) from
manually curated Kilosort outputs.

The pipeline reads manually curated results from phy_folder_for_kilosort
directory and generates:
- cluster_inf.csv: cluster-level information with best_channels and tract_channel
- spike_inf.tsv: spike-level information with neuron assignment
- neuron_inf.pkl: neuron-level information (after deduplication)

Key features:
- best_channels: obtained from template_ind.npy (all non-(-1) channels)
- tract_channel: computed from raw recording snippets (channel with max RMS)
- position and position_waveform: computed from raw recording using IDW interpolation
- All waveforms are extracted from raw data, not from templates.npy
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
from probeinterface import read_probeinterface
from scipy.stats import pearsonr
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# 与train_spike_pipeline.py和eval_spike_pipeline.py保持一致
LEFT_SAMPLE = 20   # spike前10个采样点
RIGHT_SAMPLE = 40  # spike后20个采样点
WINDOW_SIZE = 60   # 总共30个采样点: [spike_time - 10, spike_time + 20)


def get_channel_positions_from_probe(probe) -> Dict[int, Tuple[float, float]]:
    """
    从probe对象中提取通道位置信息
    
    Args:
        probe: probeinterface.ProbeGroup或Probe对象
    
    Returns:
        Dict[int, Tuple[float, float]]: 通道ID到(x, y)位置的映射
    """
    channel_positions = {}
    
    # 处理ProbeGroup
    if hasattr(probe, 'probes'):
        for probe_obj in probe.probes:
            contact_positions = probe_obj.contact_positions
            device_channel_indices = probe_obj.device_channel_indices
            if device_channel_indices is not None:
                for i, ch_idx in enumerate(device_channel_indices):
                    if ch_idx is not None and i < len(contact_positions):
                        channel_positions[int(ch_idx)] = tuple(contact_positions[i])
            else:
                # 如果没有device_channel_indices，使用contact_ids
                contact_ids = probe_obj.contact_ids
                for i, contact_id in enumerate(contact_ids):
                    if i < len(contact_positions):
                        try:
                            ch_idx = int(contact_id)
                            channel_positions[ch_idx] = tuple(contact_positions[i])
                        except (ValueError, TypeError):
                            pass
    else:
        # 处理单个Probe对象
        contact_positions = probe.contact_positions
        device_channel_indices = probe.device_channel_indices
        if device_channel_indices is not None:
            for i, ch_idx in enumerate(device_channel_indices):
                if ch_idx is not None and i < len(contact_positions):
                    channel_positions[int(ch_idx)] = tuple(contact_positions[i])
        else:
            # 如果没有device_channel_indices，使用contact_ids
            contact_ids = probe.contact_ids
            for i, contact_id in enumerate(contact_ids):
                if i < len(contact_positions):
                    try:
                        ch_idx = int(contact_id)
                        channel_positions[ch_idx] = tuple(contact_positions[i])
                    except (ValueError, TypeError):
                        pass
    
    return channel_positions


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def setup_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def load_recording(
    raw_file_or_dir: Path, 
    probe_file: Path, 
    channels_to_keep: Optional[List[str]] = None,
    stream_id: Optional[str] = None,
    is_directory: bool = False
) -> any:
    """
    Load and preprocess recording data
    
    Args:
        raw_file_or_dir: 原始记录文件路径或包含多个文件的目录路径
        probe_file: probe配置文件路径
        channels_to_keep: 要保留的通道列表（可选，默认为None，保留所有通道）
                         如果为None，会自动检测：如果存在'A-127'则保留所有A开头的通道，
                         如果存在'B-127'则保留所有B开头的通道
        stream_id: 对于intan文件，指定stream_id（例如'0'或'4'），默认为None
        is_directory: 如果为True，则从目录中读取所有文件并连接（适用于intan的.rhd文件）
    
    Returns:
        预处理后的recording对象
    """
    from spikeinterface.core import concatenate_recordings
    
    if is_directory or raw_file_or_dir.is_dir():
        # 从目录读取多个文件（适用于intan的.rhd文件）
        logging.info("Loading recordings from directory: %s", raw_file_or_dir)
        file_list = os.listdir(raw_file_or_dir)
        if "settings.xml" in file_list:
            file_list.remove("settings.xml")
        
        recording_raw_list = []
        for file in sorted(file_list):  # 排序以确保顺序一致
            file_path = raw_file_or_dir / file
            if file_path.suffix.lower() in ['.rhd', '.rhs']:
                if stream_id is not None:
                    recording_raw_list.append(se.read_intan(file_path=str(file_path), stream_id=stream_id))
                else:
                    recording_raw_list.append(se.read_intan(file_path=str(file_path)))
                logging.debug(f"Loaded file: {file}")
            else:
                logging.warning(f"Skipping file {file} (not .rhd or .rhs)")
        
        if len(recording_raw_list) == 0:
            raise ValueError(f"No valid recording files found in directory: {raw_file_or_dir}")
        
        # 连接所有recording
        recording_raw = concatenate_recordings(recording_list=recording_raw_list)
        logging.info(f"Concatenated {len(recording_raw_list)} recording files")
        # 对于intan文件目录，应用unsigned_to_signed转换
        logging.info("Applying unsigned_to_signed conversion for intan recording")
        recording_raw = spre.unsigned_to_signed(recording_raw)
    else:
        # 读取单个文件
        logging.info("Loading raw recording from %s", raw_file_or_dir)
        file_ext = raw_file_or_dir.suffix.lower()
        
        if file_ext in ['.ns4', '.ns6']:
            recording_raw = se.read_blackrock(file_path=str(raw_file_or_dir))
        elif file_ext in ['.rhd', '.rhs']:
            if stream_id is not None:
                recording_raw = se.read_intan(file_path=str(raw_file_or_dir), stream_id=stream_id)
            else:
                recording_raw = se.read_intan(file_path=str(raw_file_or_dir))
            # 对于intan文件，立即应用unsigned_to_signed转换
            logging.info("Applying unsigned_to_signed conversion for intan recording")
            recording_raw = spre.unsigned_to_signed(recording_raw)
        else:
            # 尝试自动检测
            try:
                recording_raw = se.read_blackrock(file_path=str(raw_file_or_dir))
            except:
                try:
                    if stream_id is not None:
                        recording_raw = se.read_intan(file_path=str(raw_file_or_dir), stream_id=stream_id)
                    else:
                        recording_raw = se.read_intan(file_path=str(raw_file_or_dir))
                    # 如果成功读取为intan，应用unsigned_to_signed转换
                    logging.info("Applying unsigned_to_signed conversion for intan recording")
                    recording_raw = spre.unsigned_to_signed(recording_raw)
                except:
                    raise ValueError(f"Unsupported file format: {file_ext}")
    
    # 自动检测或使用指定的channels_to_keep
    all_channel_ids = recording_raw.get_channel_ids()
    
    if channels_to_keep is None:
        # 自动检测：检查是否存在'A-127'或'B-127'
        has_a127 = 'A-127' in all_channel_ids
        has_b127 = 'B-127' in all_channel_ids
        
        if has_a127:
            # 保留所有A开头的通道
            channels_to_keep = [ch for ch in all_channel_ids if str(ch).startswith('A-')]
            logging.info(f"Auto-detected: Found 'A-127', keeping {len(channels_to_keep)} A-prefixed channels")
        elif has_b127:
            # 保留所有B开头的通道
            channels_to_keep = [ch for ch in all_channel_ids if str(ch).startswith('B-')]
            logging.info(f"Auto-detected: Found 'B-127', keeping {len(channels_to_keep)} B-prefixed channels")
        else:
            # 如果都不存在，保留所有通道
            channels_to_keep = list(all_channel_ids)
            logging.info(f"No 'A-127' or 'B-127' found, keeping all {len(channels_to_keep)} channels")
    
    # 只保留指定的通道
    if channels_to_keep and len(channels_to_keep) < len(all_channel_ids):
        channels_to_remove = [ch for ch in all_channel_ids if ch not in channels_to_keep]
        recording_raw = recording_raw.remove_channels(channels_to_remove)
        logging.info(f"Kept {len(channels_to_keep)} channels, removed {len(channels_to_remove)} channels")
    
    # 设置probe信息
    probe = read_probeinterface(str(probe_file))
    recording_raw = recording_raw.set_probegroup(probe)
    
    # 应用预处理（注意：unsigned_to_signed必须在bandpass_filter之前完成）
    logging.info("Applying preprocessing: bandpass filter (300-3000 Hz), notch filter (50 Hz), common reference")
    #recording_raw = spre.resample(recording_raw, 10000)
    recording_f = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)
    recording_f = spre.notch_filter(recording_f, freq=50)
    recording_cmr = spre.common_reference(recording_f, reference="global", operator="median")
    
    return recording_cmr


def find_recording_file(session_date: str, recording_root: Path) -> Optional[Path]:
    """
    根据session日期查找对应的recording文件或目录
    
    Args:
        session_date: session目录名（例如 "012123" 或 "WLF_128chmouse1_natima_RHD_251129_183351"）
        recording_root: recording文件根目录（例如 "/media/ubuntu/sda/data/mouse6/ns4/natural_image"）
    
    Returns:
        找到的recording文件路径或目录路径，如果未找到则返回None
    """
    # 首先尝试查找目录（适用于intan的.rhd文件）
    pattern_dir = f"*{session_date}*"
    matching_dirs = [p for p in recording_root.glob(pattern_dir) if p.is_dir()]
    
    if matching_dirs:
        # 检查目录中是否有.rhd或.rhs文件
        for dir_path in matching_dirs:
            rhd_files = list(dir_path.glob("*.rhd")) + list(dir_path.glob("*.rhs"))
            if len(rhd_files) > 0:
                logging.info(f"Found recording directory: {dir_path} with {len(rhd_files)} .rhd/.rhs files")
                return dir_path
    
    # 支持多种文件扩展名
    extensions = [".ns4", ".ns6", ".rhd", ".rhs"]
    
    for ext in extensions:
        # 尝试匹配格式：mouse6_{date}_natural_image_001.{ext}
        pattern = f"*{session_date}*{ext}"
        matching_files = list(recording_root.glob(pattern))
        
        if matching_files:
            # 如果有多个匹配，选择第一个
            return matching_files[0]
    
    return None


def parse_cluster_splits_from_output(phy_dir: Path) -> Dict[int, int]:
    """
    从output.txt解析cluster split操作，建立新cluster到旧cluster的映射
    
    注意：此函数目前不再被使用，因为新split的cluster会直接从自己的template_id获取best_channels
    保留此函数以备将来可能需要的情况
    
    Args:
        phy_dir: phy文件夹路径，应该包含output.txt文件
    
    Returns:
        split_mapping: {new_cluster_id: old_cluster_id} - 新split的cluster到原始cluster的映射
    """
    output_path = phy_dir / "output.txt"
    split_mapping = {}
    
    if not output_path.exists():
        logging.info("output.txt文件不存在，跳过cluster split映射解析")
        return split_mapping
    
    logging.info("解析output.txt中的cluster split操作...")
    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 匹配格式: "Split cluster X into clusters Y, Z (deleted: [X])"
            # 使用正则表达式提取split信息（忽略ANSI转义码）
            # 匹配模式：Split cluster <数字> into clusters <数字列表> (deleted: [<数字>])
            match = re.search(r'Split cluster (\d+) into clusters ([\d,\s]+)\s*\(deleted:\s*\[(\d+)\]\)', line)
            if match:
                old_cluster = int(match.group(1))
                new_clusters_str = match.group(2)
                deleted_cluster = int(match.group(3))
                
                # 验证old_cluster和deleted_cluster是否一致
                if old_cluster != deleted_cluster:
                    logging.warning(f"Split操作中old_cluster ({old_cluster}) 与 deleted_cluster ({deleted_cluster}) 不一致")
                
                # 解析新cluster列表（可能包含多个，用逗号分隔）
                new_clusters = [int(x.strip()) for x in new_clusters_str.split(',')]
                
                # 为每个新cluster建立到旧cluster的映射
                for new_cluster in new_clusters:
                    split_mapping[new_cluster] = old_cluster
                    logging.debug(f"映射: cluster {new_cluster} -> 原始cluster {old_cluster}")
    
    if split_mapping:
        logging.info(f"从output.txt解析到 {len(split_mapping)} 个split cluster映射")
    else:
        logging.info("output.txt中未找到cluster split操作")
    
    return split_mapping


def compute_best_channels_from_template_ind(phy_dir: Path) -> Dict[int, List[int]]:
    """
    从template_ind.npy获取每个cluster的best_channels
    不需要templates.npy，只使用template_ind来确定有效通道
    
    注意：spike_templates.npy在manual curation后不会改变，只有spike_clusters.npy会改变
    所以新split的cluster会直接从自己的template_id获取best_channels，不需要特殊处理
    
    Args:
        phy_dir: phy文件夹路径
    
    Returns:
        best_channels_dict: {cluster_id: [channel_ids]} - 所有有效通道（非-1的通道）
    """
    template_ind_path = phy_dir / "template_ind.npy"
    spike_templates_path = phy_dir / "spike_templates.npy"
    spike_clusters_path = phy_dir / "spike_clusters.npy"
    
    if not template_ind_path.exists():
        logging.warning("缺少template_ind.npy文件，无法计算best_channels")
        return {}
    
    template_ind = np.load(template_ind_path)  # shape: (n_templates, n_channels)
    
    # 建立cluster_id到template_id的映射
    cluster_to_template = {}
    if spike_templates_path.exists() and spike_clusters_path.exists():
        spike_templates = np.load(spike_templates_path).flatten()
        spike_clusters = np.load(spike_clusters_path).flatten()
        
        # 对每个cluster，找到它最常用的template（大多数情况下每个cluster只用一个template）
        for cluster_id in np.unique(spike_clusters):
            cluster_mask = spike_clusters == cluster_id
            cluster_template_ids = spike_templates[cluster_mask]
            # 使用最频繁出现的template
            unique_templates, counts = np.unique(cluster_template_ids, return_counts=True)
            most_common_template = unique_templates[np.argmax(counts)]
            cluster_to_template[int(cluster_id)] = int(most_common_template)
        
        logging.info(f"建立了 {len(cluster_to_template)} 个cluster到template的映射")
    else:
        # 如果没有spike_templates和spike_clusters，假设cluster_id == template_id
        logging.warning("未找到spike_templates.npy或spike_clusters.npy，假设cluster_id == template_id")
        for template_id in range(template_ind.shape[0]):
            cluster_to_template[template_id] = template_id
    
    # 从template_ind提取best_channels
    best_channels_dict = {}
    for cluster_id, template_id in cluster_to_template.items():
        if template_id < template_ind.shape[0]:
            template_channels = template_ind[template_id]
            # 过滤掉-1的通道
            valid_channels = template_channels[template_channels != -1]
            if len(valid_channels) > 0:
                best_channels_dict[cluster_id] = valid_channels.tolist()
    
    return best_channels_dict


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
    
    # 如果best_channels列不存在，从template_ind计算
    if 'best_channels' not in df.columns:
        logging.info("best_channels column not found, computing from template_ind...")
        # 计算best_channels
        # 注意：不需要output.txt，因为新split的cluster会直接从自己的template_id获取best_channels
        best_channels_dict = compute_best_channels_from_template_ind(phy_dir)
        
        df['best_channels'] = None
        for idx, row in df.iterrows():
            # 使用cluster_id作为键来查找best_channels
            cluster_id = int(row['cluster_id'])
            
            if cluster_id in best_channels_dict:
                df.at[idx, 'best_channels'] = str(best_channels_dict[cluster_id])
            else:
                logging.warning(f"Cluster {cluster_id} not found in best_channels_dict")
    
    # tract_channel将从raw recording snippets计算，这里初始化为None
    if 'tract_channel' not in df.columns:
        df['tract_channel'] = None
    
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


def _compute_cluster_features_worker(args):
    """Worker function for parallel processing"""
    cluster_id, snippets_list, channel_id, channel_positions = args
    if len(snippets_list) == 0:
        return cluster_id, None
    
    snippets = np.array(snippets_list)  # (n_valid_spikes, n_selected_channels, window_size)
    result = compute_cluster_features_from_snippets(snippets, channel_id, channel_positions)
    return cluster_id, result


def compute_cluster_features_from_snippets(
    snippets: np.ndarray,
    channel_id: List[int],
    channel_positions: Dict[int, Tuple[float, float]],
) -> Tuple[float, float, np.ndarray, int]:
    """
    从已经提取的snippets计算cluster的position、position_waveform和tract_channel
    snippets shape: (n_spikes, n_channels, window_size)
    
    Args:
        snippets: spike波形数据
        channel_id: 通道ID列表
        channel_positions: 通道ID到(x, y)位置的映射字典
    
    Returns:
        position_1, position_2, position_waveform, tract_channel
        tract_channel: RMS最大的通道ID
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
            x_i, y_i = channel_positions.get(channel, (0.0, 0.0))
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
            pos = channel_positions.get(channel, None)
            if pos is not None:
                x_channel, y_channel = pos
                if not (np.isnan(x_channel) or np.isnan(y_channel)):
                    distance = np.sqrt((spike_x - x_channel)**2 + (spike_y - y_channel)**2)
                    distances.append(distance)
                else:
                    distances.append(np.inf)
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
        return 0.0, 0.0, np.zeros(window_size, dtype=np.float32), -1
    
    # 计算平均位置和waveform
    cluster_x = np.mean(cluster_positions_x)
    cluster_y = np.mean(cluster_positions_y)
    cluster_avg_waveform = np.mean(cluster_waveforms, axis=0)
    
    # 计算tract_channel: 对所有snippets计算每个通道的RMS，找到RMS最大的通道
    # snippets shape: (n_spikes, n_channels, window_size)
    channel_rms = np.sqrt(np.mean(snippets**2, axis=(0, 2)))  # 对spikes和时间维度求均值
    max_rms_idx = np.argmax(channel_rms)
    tract_channel = channel_id[max_rms_idx]
    
    return cluster_x, cluster_y, cluster_avg_waveform, tract_channel


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


def plot_neuron_waveforms(neuron_inf: pd.DataFrame, output_path: Path, sampling_frequency: float) -> None:
    """
    绘制所有neuron的waveform，生成PDF文件，一页一个neuron
    
    Args:
        neuron_inf: 包含neuron信息的DataFrame，必须有'Neuron'和'position_waveform'列
        output_path: 输出PDF文件路径
        sampling_frequency: 采样频率（Hz），用于x轴时间刻度
    """
    if neuron_inf.empty:
        logging.warning("neuron_inf is empty, skipping waveform plot")
        return
    
    if 'position_waveform' not in neuron_inf.columns:
        logging.warning("position_waveform column not found, skipping waveform plot")
        return
    
    # 计算统一的scale（所有waveform的最大最小值）
    all_waveforms = []
    valid_neurons = []
    for idx, row in neuron_inf.iterrows():
        waveform = row['position_waveform']
        if waveform is not None:
            waveform_array = np.asarray(waveform, dtype=np.float32)
            if len(waveform_array) > 0:
                all_waveforms.append(waveform_array)
                valid_neurons.append(row['Neuron'])
    
    if len(all_waveforms) == 0:
        logging.warning("No valid waveforms found, skipping waveform plot")
        return
    
    # 计算全局的最大最小值
    all_waveforms_array = np.array(all_waveforms)
    global_min = np.min(all_waveforms_array)
    global_max = np.max(all_waveforms_array)
    # 添加一些边距
    margin = (global_max - global_min) * 0.1
    y_min = global_min - margin
    y_max = global_max + margin
    
    # 计算时间轴（waveform长度对应WINDOW_SIZE个采样点）
    window_size = all_waveforms_array.shape[1]
    # 时间轴：从 -LEFT_SAMPLE 到 RIGHT_SAMPLE（相对于spike时间0）
    time_points = (np.arange(window_size) - LEFT_SAMPLE) / sampling_frequency * 1000  # 转换为毫秒
    
    logging.info(f"Plotting {len(valid_neurons)} neuron waveforms to {output_path}")
    logging.info(f"Global waveform scale: [{y_min:.2f}, {y_max:.2f}]")
    
    # 创建PDF文件
    with PdfPages(output_path) as pdf:
        for idx, row in neuron_inf.iterrows():
            neuron_name = row['Neuron']
            waveform = row['position_waveform']
            
            if waveform is None:
                continue
            
            waveform_array = np.asarray(waveform, dtype=np.float32)
            if len(waveform_array) == 0:
                continue
            
            # 创建新页面
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 绘制waveform
            ax.plot(time_points, waveform_array, 'b-', linewidth=1.5, label='Waveform')
            ax.axvline(x=0, color='r', linestyle='--', linewidth=1, label='Spike time')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # 设置统一的y轴范围
            ax.set_ylim(y_min, y_max)
            
            # 设置标签和标题
            ax.set_xlabel('Time (ms)', fontsize=12)
            ax.set_ylabel('Amplitude (μV)', fontsize=12)
            ax.set_title(f'{neuron_name} - Position Waveform', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # 添加文本信息
            info_text = f"Neuron: {neuron_name}\n"
            if 'position_1' in row and 'position_2' in row:
                if not (pd.isna(row['position_1']) or pd.isna(row['position_2'])):
                    info_text += f"Position: ({row['position_1']:.2f}, {row['position_2']:.2f})\n"
            if 'tract_channel' in row and not pd.isna(row.get('tract_channel')):
                info_text += f"Tract channel: {int(row['tract_channel'])}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    logging.info(f"Saved waveform plots to {output_path}")


def plot_electrode_waveforms(
    neuron_inf: pd.DataFrame,
    channel_positions: Dict[int, Tuple[float, float]],
    output_path: Path,
    sampling_frequency: float,
) -> None:
    """
    绘制电极触点位置和neuron waveform
    首先绘制所有channel的圆形圈，然后在neuron对应的位置绘制waveform
    
    Args:
        neuron_inf: 包含neuron信息的DataFrame，必须有'Neuron', 'position_waveform', 'position_1', 'position_2'列
        channel_positions: 通道ID到(x, y)位置的映射字典
        output_path: 输出PDF文件路径
        sampling_frequency: 采样频率（Hz），用于waveform时间刻度
    """
    if neuron_inf.empty:
        logging.warning("neuron_inf is empty, skipping electrode waveform plot")
        return
    
    if 'position_waveform' not in neuron_inf.columns:
        logging.warning("position_waveform column not found, skipping electrode waveform plot")
        return
    
    # 收集所有有效的waveform用于计算统一的scale
    all_waveforms = []
    valid_neurons_data = []
    for idx, row in neuron_inf.iterrows():
        waveform = row['position_waveform']
        if waveform is not None:
            waveform_array = np.asarray(waveform, dtype=np.float32)
            if len(waveform_array) > 0:
                all_waveforms.append(waveform_array)
                valid_neurons_data.append({
                    'neuron': row['Neuron'],
                    'position_1': row.get('position_1', np.nan),
                    'position_2': row.get('position_2', np.nan),
                    'waveform': waveform_array,
                })
    
    if len(all_waveforms) == 0:
        logging.warning("No valid waveforms found, skipping electrode waveform plot")
        return
    
    # 检查channel_positions
    if not channel_positions:
        logging.warning("channel_positions is empty, cannot plot electrode positions")
        # 如果没有channel位置，尝试从neuron位置推断范围
        all_neuron_x = [d['position_1'] for d in valid_neurons_data if not np.isnan(d['position_1'])]
        all_neuron_y = [d['position_2'] for d in valid_neurons_data if not np.isnan(d['position_2'])]
        if len(all_neuron_x) == 0:
            logging.error("No valid neuron positions found, cannot create plot")
            return
        # 使用neuron位置范围作为参考
        x_range = max(all_neuron_x) - min(all_neuron_x) if len(all_neuron_x) > 1 else 100.0
        y_range = max(all_neuron_y) - min(all_neuron_y) if len(all_neuron_y) > 1 else 100.0
        spatial_range = max(x_range, y_range)
    else:
        all_x = [pos[0] for pos in channel_positions.values()]
        all_y = [pos[1] for pos in channel_positions.values()]
        x_range = max(all_x) - min(all_x) if len(all_x) > 1 else 100.0
        y_range = max(all_y) - min(all_y) if len(all_y) > 1 else 100.0
        spatial_range = max(x_range, y_range)
        logging.info(f"Channel positions range: x=[{min(all_x):.2f}, {max(all_x):.2f}], y=[{min(all_y):.2f}, {max(all_y):.2f}]")
    
    # 计算全局的最大最小值（统一scale）
    all_waveforms_array = np.array(all_waveforms)
    global_min = np.min(all_waveforms_array)
    global_max = np.max(all_waveforms_array)
    margin = (global_max - global_min) * 0.1
    y_min = global_min - margin
    y_max = global_max + margin
    
    # 计算waveform的时间轴
    window_size = all_waveforms_array.shape[1]
    time_points = (np.arange(window_size) - LEFT_SAMPLE) / sampling_frequency * 1000  # 转换为毫秒
    
    # 计算waveform的显示尺寸（在空间坐标中的大小）
    # 使用空间范围来确定waveform的大小，确保与electrode大小协调
    # waveform的显示宽度约为空间范围的5-8%，使其不会太大
    waveform_width = spatial_range * 0.06
    # waveform的高度：根据振幅范围计算，但需要与空间坐标系统协调
    # 将振幅范围映射到空间坐标，高度约为宽度的0.3-0.5倍
    amplitude_range = y_max - y_min
    if amplitude_range > 0:
        # 将振幅范围按比例映射到空间坐标
        waveform_height = waveform_width * 0.4  # 高度约为宽度的40%
    else:
        waveform_height = waveform_width * 0.4
    
    logging.info(f"Plotting electrode positions and waveforms to {output_path}")
    logging.info(f"Found {len(channel_positions) if channel_positions else 0} electrode positions")
    logging.info(f"Found {len(valid_neurons_data)} neurons with valid waveforms")
    logging.info(f"Global waveform scale: [{y_min:.2f}, {y_max:.2f}]")
    logging.info(f"Waveform display size: width={waveform_width:.2f}, height={waveform_height:.2f}")
    logging.info(f"Spatial range: {spatial_range:.2f}")
    
    # 创建PDF文件
    with PdfPages(output_path) as pdf:
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 第一步：绘制所有electrode的位置（圆形）
        circle_radius = None
        if channel_positions and len(channel_positions) > 0:
            # 计算圆形圈的半径（基于空间范围，与waveform大小协调）
            # electrode圆形半径约为空间范围的1.5%，使其大小适中
            circle_radius = spatial_range * 0.015
            
            logging.info(f"Drawing {len(channel_positions)} electrodes with radius {circle_radius:.2f}")
            
            for ch_id, (x, y) in channel_positions.items():
                # 在electrode坐标位置绘制圆形
                circle = plt.Circle((x, y), radius=circle_radius, 
                                  fill=True, facecolor='lightgray', edgecolor='gray', 
                                  linewidth=0.8, alpha=0.7, zorder=1)
                ax.add_patch(circle)
                # 可选：添加channel ID标签（只在channel数量不太多时显示）
                if len(channel_positions) <= 128:
                    ax.text(x, y, str(ch_id), fontsize=4, ha='center', va='center', 
                           alpha=0.7, color='black', zorder=2)
        else:
            logging.warning("No channel positions available, skipping electrode drawing")
        
        # 第二步：在每个neuron的坐标位置绘制waveform（lineplot）
        waveform_count = 0
        for neuron_data in valid_neurons_data:
            pos_x = neuron_data['position_1']  # neuron的x坐标
            pos_y = neuron_data['position_2']  # neuron的y坐标
            waveform = neuron_data['waveform']  # position_waveform数组
            neuron_name = neuron_data['neuron']
            
            # 跳过位置无效的neuron
            if np.isnan(pos_x) or np.isnan(pos_y):
                logging.debug(f"Skipping neuron {neuron_name}: invalid position ({pos_x}, {pos_y})")
                continue
            
            # 将waveform的y值（振幅）映射到空间坐标系统
            # waveform的振幅范围是[y_min, y_max]，需要映射到空间坐标
            # 以neuron的y坐标为中心，上下各占waveform_height/2
            waveform_y_spatial = pos_y + (waveform - y_min) / (y_max - y_min) * waveform_height - waveform_height / 2
            
            # 将时间轴映射到空间坐标
            # waveform以neuron的x坐标为中心，spike时间（time=0）对应neuron的x位置
            # time_points的范围是[time_points[0], time_points[-1]]，需要映射到[pos_x - waveform_width/2, pos_x + waveform_width/2]
            waveform_x_spatial = pos_x + (time_points - time_points[0]) / (time_points[-1] - time_points[0]) * waveform_width - waveform_width / 2
            
            # 绘制waveform的lineplot（蓝色曲线）
            ax.plot(waveform_x_spatial, waveform_y_spatial, 'b-', linewidth=1.2, alpha=0.8, zorder=10)
            
            # 在neuron位置绘制一个小点标记（大小是electrode圆形的1/4）
            # neuron点的直径应该是electrode半径的1/2（这样面积就是1/4）
            # markersize的单位是点（points），需要根据circle_radius计算
            if circle_radius is not None:
                # 如果electrode半径是R，neuron点的半径应该是R/4
                # 由于markersize是直径单位，且单位不同，我们使用经验值
                # 假设在典型显示中，markersize=2-3对应circle_radius的1/4大小
                neuron_markersize = 2.5  # 约为electrode的1/4大小
            else:
                # 如果没有electrode，使用默认值
                neuron_markersize = 2.5
            ax.plot(pos_x, pos_y, 'ko', markersize=neuron_markersize, zorder=11)
            
            waveform_count += 1
        
        logging.info(f"Drew {waveform_count} neuron waveforms")
        
        # 设置坐标轴
        # 收集所有需要显示的点（electrode和neuron位置）
        all_display_x = []
        all_display_y = []
        
        if channel_positions:
            all_display_x.extend([pos[0] for pos in channel_positions.values()])
            all_display_y.extend([pos[1] for pos in channel_positions.values()])
        
        # 添加neuron位置和waveform范围
        for neuron_data in valid_neurons_data:
            pos_x = neuron_data['position_1']
            pos_y = neuron_data['position_2']
            if not (np.isnan(pos_x) or np.isnan(pos_y)):
                all_display_x.append(pos_x)
                all_display_y.append(pos_y)
                # 添加waveform的范围
                all_display_x.extend([pos_x - waveform_width/2, pos_x + waveform_width/2])
                all_display_y.extend([pos_y - waveform_height/2, pos_y + waveform_height/2])
        
        if len(all_display_x) > 0:
            x_margin = (max(all_display_x) - min(all_display_x)) * 0.1 if len(all_display_x) > 1 else 50.0
            y_margin = (max(all_display_y) - min(all_display_y)) * 0.1 if len(all_display_y) > 1 else 50.0
            ax.set_xlim(min(all_display_x) - x_margin, max(all_display_x) + x_margin)
            ax.set_ylim(min(all_display_y) - y_margin, max(all_display_y) + y_margin)
            logging.info(f"Plot limits: x=[{min(all_display_x) - x_margin:.2f}, {max(all_display_x) + x_margin:.2f}], y=[{min(all_display_y) - y_margin:.2f}, {max(all_display_y) + y_margin:.2f}]")
        else:
            logging.warning("No valid positions found for setting plot limits")
        
        ax.set_aspect('equal', adjustable='box')
        
        # 删除坐标轴和ticklabels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    logging.info(f"Saved electrode waveform plots to {output_path}")


# -----------------------------------------------------------------------------
# Main processing functions
# -----------------------------------------------------------------------------

def process_session(
    session_dir: Path,
    phy_dir: Path,
    recording: any,
    probe: any,
    plot_waveforms: bool = True,
    n_jobs: int = 1,
) -> None:
    """处理单个session，生成cluster_inf.csv, spike_inf.tsv, neuron_inf.pkl"""
    
    logging.info("Processing session %s", session_dir.name)
    
    # 从probe获取通道位置信息
    channel_positions = get_channel_positions_from_probe(probe)
    logging.info(f"Loaded {len(channel_positions)} channel positions from probe")
    
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
    
    # 获取recording信息（不一次性加载所有数据）
    logging.info("Preparing for chunked trace loading...")
    sampling_frequency = recording.get_sampling_frequency()
    num_frames = recording.get_num_frames()
    n_channels = recording.get_num_channels()
    
    # 注意：这里使用原始数据（未白化）提取waveform，因为这是用于训练的neuron_inf
    # 白化仅用于阈值检测阶段，提取waveform时使用原始数据
    # 与train_spike_pipeline.py和eval_spike_pipeline.py保持一致
    left_sample = LEFT_SAMPLE   # 20
    right_sample = RIGHT_SAMPLE # 40
    window_size = WINDOW_SIZE   # 60
    
    # 创建recording通道ID到索引的映射
    recording_channel_ids = recording.get_channel_ids()
    channel_id_to_index = {ch_id: idx for idx, ch_id in enumerate(recording_channel_ids)}
    logging.info(f"Created channel ID to index mapping: {len(channel_id_to_index)} channels")
    logging.info(f"Recording channel IDs (first 5): {recording_channel_ids[:5]}")
    
    logging.info(f"Recording info: {num_frames} samples ({num_frames/sampling_frequency:.2f} seconds), {n_channels} channels")
    logging.info(f"Using window: [spike_time - {left_sample}, spike_time + {right_sample}) = {window_size} timepoints")
    
    # 定义chunk大小（每次加载的采样点数，1分钟的数据）
    chunk_duration_seconds = 60.0  # 每次加载1分钟的数据
    chunk_size_samples = int(chunk_duration_seconds * sampling_frequency)
    logging.info(f"Using chunked processing: {chunk_duration_seconds}s per chunk ({chunk_size_samples} samples)")
    logging.info(f"Estimated memory per chunk: ~{n_channels * chunk_size_samples * 4 / 1024 / 1024:.1f} MB (float32)")
    
    # 第一步：为所有cluster建立channel映射
    logging.info("Preparing channel mappings for all clusters...")
    cluster_channel_mapping = {}  # {cluster_id: {'channel_id': [...], 'channel_indices': [...]}}
    
    for idx, row in cluster_inf.iterrows():
        cluster_id = row['cluster_id']
        best_channels_str = row.get('best_channels', None)
        channel_id = parse_best_channels(best_channels_str)
        
        if len(channel_id) == 0:
            continue
        
        # 将channel_id映射到recording的通道索引
        channel_indices = []
        valid_channel_id = []
        
        first_ch_id = channel_id[0] if len(channel_id) > 0 else None
        first_recording_ch_id = recording_channel_ids[0] if len(recording_channel_ids) > 0 else None
        
        if isinstance(first_recording_ch_id, str) and isinstance(first_ch_id, (int, np.integer)):
            for ch_id in channel_id:
                possible_formats = [
                    f"B-{ch_id:03d}", f"C-{ch_id:03d}", f"A-{ch_id:03d}", str(ch_id)
                ]
                found = False
                for ch_id_str in possible_formats:
                    if ch_id_str in channel_id_to_index:
                        channel_indices.append(channel_id_to_index[ch_id_str])
                        valid_channel_id.append(ch_id)
                        found = True
                        break
                if not found and isinstance(ch_id, (int, np.integer)) and 0 <= ch_id < n_channels:
                    channel_indices.append(ch_id)
                    valid_channel_id.append(ch_id)
        else:
            for ch_id in channel_id:
                if ch_id in channel_id_to_index:
                    channel_indices.append(channel_id_to_index[ch_id])
                    valid_channel_id.append(ch_id)
                elif isinstance(ch_id, (int, np.integer)) and 0 <= ch_id < n_channels:
                    channel_indices.append(ch_id)
                    valid_channel_id.append(ch_id)
        
        if len(channel_indices) > 0:
            cluster_channel_mapping[cluster_id] = {
                'channel_id': valid_channel_id,
                'channel_indices': channel_indices
            }
    
    # 第二步：收集所有spike并按时间排序
    logging.info("Collecting all spikes...")
    all_spikes = []  # [(cluster_id, spike_time), ...]
    filtered_spike_rows = []
    
    for cluster_id in cluster_inf['cluster_id'].values:
        if cluster_id not in cluster_channel_mapping:
            continue
        cluster_spikes = spike_inf[spike_inf['cluster_id'] == cluster_id]
        spike_times = cluster_spikes['time'].values
        
        # 过滤边界附近的spike
        valid_spikes = spike_times[
            (spike_times >= left_sample) &
            (spike_times < num_frames - right_sample)
        ]
        
        for t in valid_spikes:
            all_spikes.append((cluster_id, int(t)))
            filtered_spike_rows.append({'cluster_id': cluster_id, 'time': int(t)})
    
    # 按时间排序
    all_spikes.sort(key=lambda x: x[1])
    logging.info(f"Collected {len(all_spikes)} spikes from {len(cluster_channel_mapping)} clusters")
    
    # 第三步：按时间顺序分块处理所有spike
    cluster_snippets = {cluster_id: [] for cluster_id in cluster_channel_mapping.keys()}  # {cluster_id: [snippets]}
    
    if len(all_spikes) > 0:
        min_spike_time = all_spikes[0][1]
        max_spike_time = all_spikes[-1][1]
        chunk_range_start = max(0, min_spike_time - left_sample)
        chunk_range_end = min(num_frames, max_spike_time + right_sample)
        
        # 分块处理
        current_chunk_start = chunk_range_start
        total_chunks = int(np.ceil((chunk_range_end - chunk_range_start) / chunk_size_samples))
        
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            while current_chunk_start < chunk_range_end:
                current_chunk_end = min(current_chunk_start + chunk_size_samples, chunk_range_end)
                
                # 找出这个chunk内的所有spike
                chunk_spikes = [(cid, st) for cid, st in all_spikes 
                               if current_chunk_start <= st < current_chunk_end]
                
                if len(chunk_spikes) > 0:
                    # 加载这个chunk的traces（扩展边界）
                    chunk_load_start = max(0, current_chunk_start - left_sample)
                    chunk_load_end = min(num_frames, current_chunk_end + right_sample)
                    
                    # 加载chunk数据（只加载一次）
                    chunk_traces = recording.get_traces(
                        start_frame=chunk_load_start, 
                        end_frame=chunk_load_end
                    ).astype(np.float32)
                    
                    # 确保维度正确
                    if chunk_traces.shape[0] > chunk_traces.shape[1] and chunk_traces.shape[0] > 100:
                        chunk_traces = chunk_traces.T
                    
                    # 处理这个chunk内的所有spike
                    for cluster_id, spike_time in chunk_spikes:
                        start = spike_time - left_sample
                        end = spike_time + right_sample
                        
                        if start < 0 or end > num_frames or end - start != window_size:
                            continue
                        
                        local_start = start - chunk_load_start
                        local_end = end - chunk_load_start
                        
                        if local_start < 0 or local_end > chunk_traces.shape[1] or local_end - local_start != window_size:
                            continue
                        
                        snippet = chunk_traces[:, local_start:local_end]  # (n_channels, window_size)
                        
                        # 使用该cluster的channel_indices
                        channel_indices = cluster_channel_mapping[cluster_id]['channel_indices']
                        snippet_selected = snippet[channel_indices, :]  # (n_selected_channels, window_size)
                        cluster_snippets[cluster_id].append(snippet_selected)
                
                current_chunk_start = current_chunk_end
                pbar.update(1)
    
    # 第四步：为每个cluster计算features（并行处理）
    logging.info("Computing cluster features...")
    
    # 准备并行处理的任务
    tasks = []
    cluster_idx_map = {}  # {cluster_id: idx} 用于后续更新DataFrame
    
    for idx, row in cluster_inf.iterrows():
        cluster_id = row['cluster_id']
        cluster_idx_map[cluster_id] = idx
        
        if cluster_id not in cluster_channel_mapping:
            continue
        
        snippets_list = cluster_snippets[cluster_id]
        if len(snippets_list) == 0:
            continue
        
        channel_id = cluster_channel_mapping[cluster_id]['channel_id']
        tasks.append((
            cluster_id,
            snippets_list,
            channel_id,
            channel_positions
        ))
    
    # 并行或串行处理
    if n_jobs > 1 and len(tasks) > 1:
        logging.info(f"Using {n_jobs} parallel workers to compute features...")
        results = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_compute_cluster_features_worker, task): task[0] 
                      for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing features"):
                cluster_id, result = future.result()
                results[cluster_id] = result
    else:
        # 串行处理（兼容性更好）
        results = {}
        for task in tqdm(tasks, desc="Computing features"):
            cluster_id, result = _compute_cluster_features_worker(task)
            results[cluster_id] = result
    
    # 更新cluster_inf
    for cluster_id, result in results.items():
        if result is None:
            continue
        
        idx = cluster_idx_map[cluster_id]
        position_1, position_2, position_waveform, tract_channel_computed = result
        
        cluster_inf.at[idx, 'position_1'] = position_1
        cluster_inf.at[idx, 'position_2'] = position_2
        cluster_inf.at[idx, 'position_waveform'] = position_waveform
        
        channel_id = cluster_channel_mapping[cluster_id]['channel_id']
        cluster_inf.at[idx, 'channel_id'] = str(channel_id)  # 存储为字符串以便CSV保存
        
        if tract_channel_computed != -1:
            cluster_inf.at[idx, 'tract_channel'] = tract_channel_computed

    # 用过滤后的 spike_inf 替换（只包含边界有效的spike，SNR过滤将在neuron级别进行）
    if len(filtered_spike_rows) > 0:
        original_count = len(spike_inf)
        spike_inf = pd.DataFrame(filtered_spike_rows, columns=['cluster_id', 'time'])
        filtered_count = len(spike_inf)
        removed_count = original_count - filtered_count
        removal_ratio = removed_count / original_count * 100 if original_count > 0 else 0
        logging.info(f"Boundary filter (valid window): {original_count} -> {filtered_count} spikes kept, {removed_count} removed ({removal_ratio:.2f}%)")
    
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
    
    # 8. 绘制waveform（如果启用）
    if plot_waveforms:
        waveform_plot_output = session_dir / "neuron_waveforms.pdf"
        electrode_plot_output = session_dir / "electrode_waveforms.pdf"
        sampling_frequency = recording.get_sampling_frequency()
        plot_neuron_waveforms(neuron_inf, waveform_plot_output, sampling_frequency)
        plot_electrode_waveforms(neuron_inf, channel_positions, electrode_plot_output, sampling_frequency)


def main() -> None:
    setup_logger(verbose=False)

    parser = argparse.ArgumentParser(
        description="Generate cluster_inf.csv, spike_inf.tsv, and neuron_inf.pkl from manually curated Kilosort outputs"
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        help="Path to raw recording file or directory (e.g., .ns6 file or directory with .rhd files)",
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
        default="/media/ubuntu/sda/data/mouse6/ns4/natural_image",
        help="Root directory containing recording files",
    )
    parser.add_argument(
        "--stream-id",
        type=str,
        default=None,
        help="Stream ID for intan files (e.g., '0' or '4'). Required when loading intan recordings.",
    )
    parser.add_argument(
        "--is-directory",
        action="store_true",
        help="If set, treat --raw-file as a directory and load all .rhd/.rhs files from it",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        required=True,
        help="Path to the session directory (e.g., /path/to/sorted/session_name).",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        default=True,
        help="Disable plotting neuron waveforms (default: plotting is enabled)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers for computing cluster features (default: 1, use -1 for all CPUs)",
    )
    args = parser.parse_args()

    probe_file = Path(args.probe_file if args.probe_file else "/media/ubuntu/sda/data/probe.json")
    recording_root = Path(args.recording_root)

    # 加载probe信息
    probe = read_probeinterface(str(probe_file))

    # 处理单个session目录
    session_dir_path = Path(args.session_dir)
    if not session_dir_path.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir_path}")
    if not session_dir_path.is_dir():
        raise ValueError(f"Session path is not a directory: {session_dir_path}")
    
    logging.info("=" * 80)
    logging.info("Processing session %s", session_dir_path.name)

    phy_dir = session_dir_path / "phy_folder_for_kilosort"
    if not phy_dir.exists():
        raise FileNotFoundError(f"phy_folder_for_kilosort not found for session {session_dir_path.name}")

    # 查找对应的recording文件
    session_date = session_dir_path.name  # session目录名就是日期，例如 "012123"
    
    if args.raw_file:
        # 如果提供了raw_file参数，使用它
        raw_file_path = Path(args.raw_file)
        recording = load_recording(
            raw_file_path, 
            probe_file,
            channels_to_keep=None,  # 自动检测
            stream_id=args.stream_id,
            is_directory=args.is_directory
        )
    else:
        # 根据session日期查找对应的recording文件或目录
        raw_file_path = find_recording_file(session_date, recording_root)
        if raw_file_path is None:
            raise FileNotFoundError(
                f"No recording file found for session {session_date} in {recording_root}"
            )
        logging.info("Found recording file/directory: %s", raw_file_path)
        # 自动检测是否为目录
        is_dir = raw_file_path.is_dir()
        recording = load_recording(
            raw_file_path, 
            probe_file,
            channels_to_keep=None,  # 自动检测
            stream_id=args.stream_id,
            is_directory=is_dir
        )

    # 处理n_jobs参数
    n_jobs = args.n_jobs
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
        logging.info(f"Using all available CPUs: {n_jobs}")
    elif n_jobs < 1:
        n_jobs = 1
    
    try:
        process_session(session_dir_path, phy_dir, recording, probe, 
                       plot_waveforms=args.plot, n_jobs=n_jobs)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to process session %s: %s", session_dir_path.name, exc)
        raise


if __name__ == "__main__":
    main()
