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

import numpy as np
import pandas as pd
from probeinterface import read_probeinterface, ProbeGroup, Probe
from scipy.io import loadmat
from scipy.stats import pearsonr
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# 与train_spike_pipeline.py和eval_spike_pipeline.py保持一致
LEFT_SAMPLE = 10   # spike前10个采样点
RIGHT_SAMPLE = 20  # spike后20个采样点
WINDOW_SIZE = 30   # 总共30个采样点: [spike_time - 10, spike_time + 20)


def get_default_channel_names(num_channels: int = 256) -> List[str]:
    """生成与sorting_test.ipynb一致的通道名称列表（A-000~A-127,B-000~B-127）"""
    half = num_channels // 2
    names_a = [f"A-{i:03d}" for i in range(half)]
    names_b = [f"B-{i:03d}" for i in range(num_channels - half)]
    return names_a + names_b


def build_probe_from_mat_csv(mat_path: Path, csv_path: Path) -> Probe:
    """
    使用 sorting.ipynb 的方式构建当前电极的 Probe 对象
    - mat文件: chanMap_DCX_5mm.mat，包含xcoords/ycoords/chanMap0ind
    - csv文件: ch_map_R.csv，包含probeloc列
    """
    probe_data = loadmat(mat_path)
    probe_x = probe_data["xcoords"]
    probe_y = probe_data["ycoords"]

    probe_position = pd.DataFrame(probe_x)
    probe_position[1] = probe_y
    probe_position["chan_map"] = probe_data["chanMap0ind"].astype(int)

    chan_map = pd.read_csv(csv_path)
    merged = (
        chan_map.merge(probe_position, left_on="probeloc", right_on="chan_map")
        .iloc[chan_map.index]
        .reset_index(drop=True)
    )

    probe = Probe()
    probe.set_contacts(positions=merged.iloc[:, 2:4])
    probe.set_device_channel_indices(range(len(merged)))
    probe.set_contact_ids(list(range(len(merged))))
    return probe


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
    probe_file: Optional[Path],
    probe: Optional[Probe] = None,
    channels_to_remove: Optional[List[str]] = None,
    stream_id: Optional[str] = None,
    is_directory: bool = False,
    binary_sampling_frequency: float = 30000,
    binary_dtype: str = "int16",
    binary_num_channels: int = 256,
) -> any:
    """
    Load and preprocess recording data
    
    Args:
        raw_file_or_dir: 原始记录文件路径或包含多个文件的目录路径
    probe_file: probe配置文件路径（当probe为None时使用）
    probe: 直接传入的Probe/ProbeGroup对象
        channels_to_remove: 要移除的通道列表（可选，默认为None，不移除任何通道）
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
        elif file_ext == '.bin':
            logging.info("Loading binary slice recording from %s", raw_file_or_dir)
            recording_raw = se.read_binary(
                file_paths=str(raw_file_or_dir),
                sampling_frequency=binary_sampling_frequency,
                dtype=binary_dtype,
                num_channels=binary_num_channels,
            )
            # 将通道命名为A-000~B-127，保持与sorting_test.ipynb一致
            channel_names = get_default_channel_names(binary_num_channels)
            recording_raw = recording_raw.rename_channels(channel_names)
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
    
    # 移除指定的通道（如果提供）
    if channels_to_remove:
        recording_raw = recording_raw.remove_channels(channels_to_remove)
        logging.info(f"Removed channels: {channels_to_remove}")
    
    # 设置probe信息
    if probe is None:
        if probe_file is None:
            raise ValueError("必须提供probe对象或probe_file路径")
        probe = read_probeinterface(str(probe_file))
    # Probe转成ProbeGroup以便兼容set_probegroup
    if isinstance(probe, Probe):
        probe_group = ProbeGroup()
        probe_group.add_probe(probe)
    else:
        probe_group = probe
    recording_raw = recording_raw.set_probegroup(probe_group)
    
    # 应用预处理（注意：unsigned_to_signed必须在bandpass_filter之前完成）
    logging.info("Applying preprocessing: bandpass filter (300-3000 Hz), notch filter (50 Hz), common reference")
    recording_raw = spre.resample(recording_raw, 10000)
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


def compute_cluster_features_from_template(
    template_id: int,
    templates: np.ndarray,
    template_ind: Optional[np.ndarray],
    channel_id: List[int],
    channel_positions: Dict[int, Tuple[float, float]],
) -> Tuple[float, float, np.ndarray, int]:
    """
    使用templates.npy中的模板直接计算position、position_waveform和tract_channel
    - 仅使用best_channels对应的通道（与template_ind对齐）
    """
    if template_id < 0 or template_id >= templates.shape[0]:
        return 0.0, 0.0, np.zeros(templates.shape[1], dtype=np.float32), -1

    template = templates[template_id]  # shape: (n_timepoints, n_channels_all)

    # 通过template_ind对齐通道顺序
    if template_ind is not None and template_ind.shape[0] > template_id:
        valid_channels = template_ind[template_id]  # (n_channels_all,)
        mask = valid_channels != -1
        template_channels = valid_channels[mask]  # 设备通道ID
        template_data = template[:, mask]
    else:
        # 无template_ind时，假设channel_id顺序与模板通道顺序一致
        template_channels = np.arange(template.shape[1])
        template_data = template

    # 将best_channels映射到模板通道索引
    indices = []
    mapped_channel_id = []
    for ch in channel_id:
        idx = np.where(template_channels == ch)[0]
        if len(idx) > 0:
            indices.append(int(idx[0]))
            mapped_channel_id.append(int(ch))

    if len(indices) == 0:
        return 0.0, 0.0, np.zeros(template.shape[0], dtype=np.float32), -1

    # 选取对应通道的数据
    selected = template_data[:, indices]  # (n_timepoints, n_selected_channels)

    # 能量/权重
    a_sq = np.sum(selected**2, axis=0)  # 每个通道能量

    sum_x_a = 0.0
    sum_y_a = 0.0
    sum_a = 0.0
    for ch, a_i_sq in zip(mapped_channel_id, a_sq):
        x_i, y_i = channel_positions.get(ch, (0.0, 0.0))
        sum_x_a += x_i * a_i_sq
        sum_y_a += y_i * a_i_sq
        sum_a += a_i_sq

    if sum_a == 0:
        return 0.0, 0.0, np.zeros(template.shape[0], dtype=np.float32), -1

    pos_x = sum_x_a / sum_a
    pos_y = sum_y_a / sum_a

    # position_waveform: 对模板波形在空间上做IDW
    distances = []
    for ch in mapped_channel_id:
        pos = channel_positions.get(ch, None)
        if pos is not None:
            x_channel, y_channel = pos
            if not (np.isnan(x_channel) or np.isnan(y_channel)):
                distance = np.sqrt((pos_x - x_channel) ** 2 + (pos_y - y_channel) ** 2)
                distances.append(distance)
            else:
                distances.append(np.inf)
        else:
            distances.append(np.inf)

    distances = np.array(distances, dtype=np.float32)
    if len(distances) == 0 or np.all(np.isinf(distances)):
        return pos_x, pos_y, np.zeros(template.shape[0], dtype=np.float32), -1

    if np.any(distances == 0):
        zero_idx = np.where(distances == 0)[0][0]
        position_waveform = selected[:, zero_idx].astype(np.float32)
    else:
        weights = 1.0 / (np.power(distances, 2, dtype=np.float32) + 1e-10)
        weights /= weights.sum()
        position_waveform = np.dot(selected, weights).astype(np.float32)

    # tract_channel: RMS最大的通道
    channel_rms = np.sqrt(np.mean(selected**2, axis=0))
    max_rms_idx = int(np.argmax(channel_rms))
    tract_channel = mapped_channel_id[max_rms_idx]

    return pos_x, pos_y, position_waveform, tract_channel


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
    
    # 预载模板相关数据
    templates_path = phy_dir / "templates.npy"
    template_ind_path = phy_dir / "template_ind.npy"
    spike_templates_path = phy_dir / "spike_templates.npy"
    spike_clusters_path = phy_dir / "spike_clusters.npy"

    if not templates_path.exists():
        raise FileNotFoundError(f"templates.npy not found in {phy_dir}")

    templates = np.load(templates_path)
    template_ind = np.load(template_ind_path) if template_ind_path.exists() else None

    # 建立cluster到template的映射（取该cluster最常用的template）
    cluster_to_template: Dict[int, int] = {}
    if spike_templates_path.exists() and spike_clusters_path.exists():
        spike_templates = np.load(spike_templates_path).flatten()
        spike_clusters = np.load(spike_clusters_path).flatten()
        for cluster_id in np.unique(spike_clusters):
            mask = spike_clusters == cluster_id
            tmpl_ids = spike_templates[mask]
            if len(tmpl_ids) == 0:
                continue
            uniq, counts = np.unique(tmpl_ids, return_counts=True)
            cluster_to_template[int(cluster_id)] = int(uniq[np.argmax(counts)])
        logging.info("Built cluster->template map for %d clusters", len(cluster_to_template))
    else:
        logging.warning("spike_templates.npy or spike_clusters.npy missing, fallback cluster_id==template_id")
        for template_id in range(templates.shape[0]):
            cluster_to_template[template_id] = template_id

    # 2. 为每个cluster计算position和position_waveform（直接使用templates）
    logging.info("Computing cluster features from templates...")
    cluster_inf['position_1'] = np.nan
    cluster_inf['position_2'] = np.nan
    cluster_inf['position_waveform'] = pd.Series([None] * len(cluster_inf), dtype=object)
    cluster_inf['channel_id'] = None
    
    # 检查是否有best_channels列
    if 'best_channels' not in cluster_inf.columns:
        logging.warning("best_channels column not found in cluster_inf, cannot compute features")
        return
    
    # 按cluster分组处理
    for idx, row in cluster_inf.iterrows():
        cluster_id = row['cluster_id']
        # 从best_channels获取channel_id
        best_channels_str = row.get('best_channels', None)
        channel_id = parse_best_channels(best_channels_str)

        if len(channel_id) == 0:
            logging.warning(f"Cluster {cluster_id} has no valid best_channels, skipping")
            continue

        # 找到模板ID
        template_id = cluster_to_template.get(int(cluster_id), None)
        if template_id is None:
            logging.warning(f"Cluster {cluster_id} has no mapped template, skipping")
            continue

        position_1, position_2, position_waveform, tract_channel_computed = compute_cluster_features_from_template(
            template_id, templates, template_ind, channel_id, channel_positions
        )
        
        cluster_inf.at[idx, 'position_1'] = position_1
        cluster_inf.at[idx, 'position_2'] = position_2
        cluster_inf.at[idx, 'position_waveform'] = position_waveform
        cluster_inf.at[idx, 'channel_id'] = str(channel_id)  # 存储为字符串以便CSV保存
        if tract_channel_computed != -1:
            cluster_inf.at[idx, 'tract_channel'] = tract_channel_computed
    
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
        default=None,
        help="Path to probe file (if not provided, will build from --probe-mat-path and --probe-csv-path)",
    )
    parser.add_argument(
        "--probe-mat-path",
        type=str,
        default="/media/ubuntu/sda/duan/raw_data/chanMap_DCX_5mm.mat",
        help="Path to chanMap mat file for current electrode",
    )
    parser.add_argument(
        "--probe-csv-path",
        type=str,
        default="/media/ubuntu/sda/duan/raw_data/ch_map_R.csv",
        help="Path to channel map csv (with probeloc column) for current electrode",
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
        default=None,
        help="Path to a specific session directory (e.g., /path/to/sorted/session_name). If specified, only this session will be processed.",
    )
    parser.add_argument(
        "--sorting-root",
        type=str,
        default=None,
        help="Root directory containing session directories (e.g., /path/to/sorted). If not specified, will use default path.",
    )
    args = parser.parse_args()

    probe_file = Path(args.probe_file) if args.probe_file else None
    recording_root = Path(args.recording_root)

    # 加载probe信息（优先使用mat/csv构建当前电极）
    if probe_file and probe_file.exists():
        probe = read_probeinterface(str(probe_file))
        logging.info("Loaded probe from %s", probe_file)
    else:
        probe = build_probe_from_mat_csv(Path(args.probe_mat_path), Path(args.probe_csv_path))
        logging.info("Built probe from mat/csv: %s, %s", args.probe_mat_path, args.probe_csv_path)

    # 确定要处理的session目录
    if args.session_dir:
        # 如果指定了单个session目录，只处理这个
        session_dir_path = Path(args.session_dir)
        if not session_dir_path.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir_path}")
        if not session_dir_path.is_dir():
            raise ValueError(f"Session path is not a directory: {session_dir_path}")
        session_dirs = [session_dir_path]
        logging.info(f"Processing single session: {session_dir_path}")
    else:
        # 从sorting_root读取所有session目录
        if args.sorting_root:
            sorting_root = Path(args.sorting_root)
        else:
            # 使用默认路径
            base_dir = Path(
                "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels"
            )
            sorting_root = base_dir / "kilosort_spike_sorting"
        
        # 检查是否有sorting_new子目录，或者直接使用sorting_root
        potential_sorting_new = sorting_root / "sorting_new"
        if potential_sorting_new.exists():
            search_dir = potential_sorting_new
        else:
            search_dir = sorting_root
        
        session_dirs = sorted(
            p for p in search_dir.iterdir() if p.is_dir()
        )
        if not session_dirs:
            raise FileNotFoundError(f"No session directories found under {search_dir}")
        logging.info(f"Found {len(session_dirs)} session directories in {search_dir}")

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
            # 如果提供了raw_file参数，使用它（适用于所有session共享同一个文件或目录的情况）
            raw_file_path = Path(args.raw_file)
            recording = load_recording(
                raw_file_path, 
                probe_file,
                probe=probe,
                stream_id=args.stream_id,
                is_directory=args.is_directory
            )
        else:
            # 根据session日期查找对应的recording文件或目录
            raw_file_path = find_recording_file(session_date, recording_root)
            if raw_file_path is None:
                logging.warning(
                    "No recording file found for session %s in %s, skipping.",
                    session_date,
                    recording_root
                )
                continue
            logging.info("Found recording file/directory: %s", raw_file_path)
            # 自动检测是否为目录
            is_dir = raw_file_path.is_dir()
            recording = load_recording(
                raw_file_path, 
                probe_file,
                probe=probe,
                stream_id=args.stream_id,
                is_directory=is_dir
            )

        try:
            process_session(session_dir, phy_dir, recording, probe)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to process session %s: %s", session_dir.name, exc)


if __name__ == "__main__":
    main()
