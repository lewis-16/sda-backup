#!/usr/bin/env python3
"""
验证模板匹配的脚本

1. 计算021322和022522两个模板之间的匹配度score
2. 基于022522的spike_inf，对每个neuron采样5000个spike，计算每个spike的shape和energy，
   然后分别与021322和022522的template计算score
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from probeinterface import read_probeinterface
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LEFT_SAMPLE = 10   # spike前10个采样点
RIGHT_SAMPLE = 20  # spike后20个采样点
WINDOW_SIZE = 30   # 总共30个采样点: [spike_time - 10, spike_time + 19]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def setup_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def load_recording(raw_file: Path, probe_file: Path):
    """Load and preprocess recording data"""
    logging.info("Loading raw recording from %s", raw_file)
    recording_raw = se.read_blackrock(file_path=str(raw_file))
    recording_recorded = recording_raw.remove_channels(["98", "31", "32"])
    recording_recorded = recording_recorded.set_probegroup(read_probeinterface(str(probe_file)))
    recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
    recording_cmr = spre.common_reference(recording_f, reference="global", operator="median")
    return recording_cmr


def find_recording_file(session_date: str, recording_root: Path) -> Path | None:
    """根据session日期查找对应的recording文件"""
    extensions = [".ns4", ".ns6"]
    
    for ext in extensions:
        pattern = f"*{session_date}*{ext}"
        matching_files = list(recording_root.glob(pattern))
        
        if matching_files:
            return matching_files[0]
    
    return None


def extract_spike_waveform(
    spike_time: int,
    traces: np.ndarray,
    channel_id: list[int],
) -> np.ndarray | None:
    """
    提取单个spike的波形
    
    Args:
        spike_time: spike时间点
        traces: 记录数据，shape为(n_channels, n_timepoints)
        channel_id: 要提取的通道ID列表
    
    Returns:
        snippet: (n_bestchannels, 30) 或 None（如果边界检查失败）
    """
    start = spike_time - LEFT_SAMPLE
    end = spike_time + RIGHT_SAMPLE
    
    n_channels, max_frame = traces.shape
    
    # 检查边界
    if start < 0 or end > max_frame:
        return None
    if end - start != WINDOW_SIZE:
        return None
    
    # 提取该spike的窗口数据
    snippet = traces[:, start:end]  # (30, 30)
    # 只保留best_channels的通道
    snippet_selected = snippet[channel_id, :]  # (n_bestchannels, 30)
    
    return snippet_selected


def compute_spike_shape_and_energy(
    snippet: np.ndarray,
    channel_id: list[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算单个spike的shape和energy
    
    Args:
        snippet: (n_bestchannels, 30) 波形数据
        channel_id: best_channels列表
    
    Returns:
        shape: (30, 30) 形状模板（填充后）
        energy: (30,) 能量向量（填充后）
    """
    n_bestchannels = snippet.shape[0]
    
    # 计算形状：对每条通道内的波形做L2归一化
    normalized_snippet = np.zeros_like(snippet)
    for ch_idx in range(n_bestchannels):
        channel_waveform = snippet[ch_idx, :]  # (30,)
        norm = np.linalg.norm(channel_waveform)
        if norm > 1e-10:
            normalized_snippet[ch_idx, :] = channel_waveform / norm
        else:
            normalized_snippet[ch_idx, :] = channel_waveform
    
    # 填充到(30, 30)
    shape = np.zeros((30, 30), dtype=np.float32)
    for ch_idx, ch_id in enumerate(channel_id):
        shape[ch_id, :] = normalized_snippet[ch_idx, :]
    
    # 计算能量：E = np.sum(snippet**2, axis=1)
    channel_energies = np.sum(snippet**2, axis=1)  # (n_bestchannels,)
    
    # 填充到(30,)
    energy = np.zeros(30, dtype=np.float32)
    for ch_idx, ch_id in enumerate(channel_id):
        energy[ch_id] = channel_energies[ch_idx]
    
    return shape, energy


def compute_template_score(
    shape1: np.ndarray,
    energy1: np.ndarray,
    shape2: np.ndarray,
    energy2: np.ndarray,
) -> Tuple[float, float]:
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


# -----------------------------------------------------------------------------
# Main validation functions
# -----------------------------------------------------------------------------

def compute_template_to_template_scores(
    shape_templates_021322: np.ndarray,
    energy_templates_021322: np.ndarray,
    shape_templates_022522: np.ndarray,
    energy_templates_022522: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算021322和022522两个模板之间的匹配度score
    
    Args:
        shape_templates_021322: (n_neuron_021322, 30, 30)
        energy_templates_021322: (n_neuron_021322, 30)
        shape_templates_022522: (n_neuron_022522, 30, 30)
        energy_templates_022522: (n_neuron_022522, 30)
    
    Returns:
        shape_scores: (n_neuron_021322, n_neuron_022522)
        energy_scores: (n_neuron_021322, n_neuron_022522)
    """
    n_neuron_021322 = shape_templates_021322.shape[0]
    n_neuron_022522 = shape_templates_022522.shape[0]
    
    shape_scores = np.zeros((n_neuron_021322, n_neuron_022522), dtype=np.float32)
    energy_scores = np.zeros((n_neuron_021322, n_neuron_022522), dtype=np.float32)
    
    logging.info("Computing template-to-template scores...")
    for i in range(n_neuron_021322):
        for j in range(n_neuron_022522):
            shape_score, energy_score = compute_template_score(
                shape_templates_021322[i],
                energy_templates_021322[i],
                shape_templates_022522[j],
                energy_templates_022522[j],
            )
            shape_scores[i, j] = shape_score
            energy_scores[i, j] = energy_score
        
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i + 1}/{n_neuron_021322} neurons from 021322")
    
    return shape_scores, energy_scores


def compute_spike_to_template_scores(
    neuron_inf_022522: pd.DataFrame,
    spike_inf_022522: pd.DataFrame,
    traces_022522: np.ndarray,
    shape_templates_021322: np.ndarray,
    energy_templates_021322: np.ndarray,
    shape_templates_022522: np.ndarray,
    energy_templates_022522: np.ndarray,
    n_spikes_per_neuron: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对022522的每个neuron采样5000个spike，计算每个spike的shape和energy，
    然后分别与021322和022522的template计算score
    
    Args:
        neuron_inf_022522: 022522的neuron信息
        spike_inf_022522: 022522的spike信息
        traces_022522: 022522的记录数据
        shape_templates_021322: (n_neuron_021322, 30, 30)
        energy_templates_021322: (n_neuron_021322, 30)
        shape_templates_022522: (n_neuron_022522, 30, 30)
        energy_templates_022522: (n_neuron_022522, 30)
        n_spikes_per_neuron: 每个neuron使用的spike数量
    
    Returns:
        shape_scores_021322: (n_spike, n_neuron_021322)
        energy_scores_021322: (n_spike, n_neuron_021322)
        shape_scores_022522: (n_spike, n_neuron_022522)
        energy_scores_022522: (n_spike, n_neuron_022522)
        gt_neuron_indices: (n_spike,) 每个spike对应的gt neuron索引（在neuron_inf_022522中的索引）
    """
    n_neuron_021322 = shape_templates_021322.shape[0]
    n_neuron_022522 = shape_templates_022522.shape[0]
    
    n_channels, max_frame = traces_022522.shape
    
    # 收集所有spike的shape和energy
    all_spike_shapes = []
    all_spike_energies = []
    all_spike_neuron_indices = []  # 记录每个spike属于哪个neuron（在neuron_inf中的索引）
    
    logging.info("Extracting spikes and computing shape/energy for 022522...")
    
    for neuron_idx, (_, row) in enumerate(neuron_inf_022522.iterrows()):
        neuron_name = row['Neuron']
        channel_id = row['channel_id']
        
        if not isinstance(channel_id, list):
            if isinstance(channel_id, (np.ndarray, tuple)):
                channel_id = list(channel_id)
            else:
                logging.warning(f"Neuron {neuron_name} has invalid channel_id, skipping")
                continue
        
        if len(channel_id) == 0:
            logging.warning(f"Neuron {neuron_name} has no valid channels, skipping")
            continue
        
        # 获取该neuron的所有spike
        neuron_spikes = spike_inf_022522[spike_inf_022522['neuron'] == neuron_name].copy()
        if len(neuron_spikes) == 0:
            logging.warning(f"Neuron {neuron_name} has no spikes, skipping")
            continue
        
        # 过滤边界附近的spike
        spike_times = neuron_spikes['time'].values
        spike_times = spike_times[
            (spike_times >= LEFT_SAMPLE) &
            (spike_times < max_frame - RIGHT_SAMPLE)
        ]
        
        if len(spike_times) == 0:
            logging.warning(f"Neuron {neuron_name} has no valid spikes after boundary filtering, skipping")
            continue
        
        # 随机选择spike（最多n_spikes_per_neuron个）
        n_spikes_to_use = min(n_spikes_per_neuron, len(spike_times))
        if n_spikes_to_use < len(spike_times):
            selected_indices = np.random.choice(len(spike_times), n_spikes_to_use, replace=False)
            selected_spike_times = spike_times[selected_indices]
        else:
            selected_spike_times = spike_times
        
        # 提取每个spike的波形并计算shape和energy
        for spike_time in selected_spike_times:
            snippet = extract_spike_waveform(spike_time, traces_022522, channel_id)
            if snippet is None:
                continue
            
            shape, energy = compute_spike_shape_and_energy(snippet, channel_id)
            all_spike_shapes.append(shape)
            all_spike_energies.append(energy)
            all_spike_neuron_indices.append(neuron_idx)
        
        logging.debug(f"Neuron {neuron_name}: extracted {len(selected_spike_times)} spikes")
    
    if len(all_spike_shapes) == 0:
        raise ValueError("No valid spikes extracted!")
    
    n_spikes = len(all_spike_shapes)
    logging.info(f"Total extracted {n_spikes} spikes from 022522")
    
    # 计算每个spike与所有模板的score
    logging.info("Computing spike-to-template scores...")
    
    shape_scores_021322 = np.zeros((n_spikes, n_neuron_021322), dtype=np.float32)
    energy_scores_021322 = np.zeros((n_spikes, n_neuron_021322), dtype=np.float32)
    shape_scores_022522 = np.zeros((n_spikes, n_neuron_022522), dtype=np.float32)
    energy_scores_022522 = np.zeros((n_spikes, n_neuron_022522), dtype=np.float32)
    
    for spike_idx in range(n_spikes):
        spike_shape = all_spike_shapes[spike_idx]
        spike_energy = all_spike_energies[spike_idx]
        
        # 与021322的模板计算score
        for template_idx in range(n_neuron_021322):
            shape_score, energy_score = compute_template_score(
                spike_shape,
                spike_energy,
                shape_templates_021322[template_idx],
                energy_templates_021322[template_idx],
            )
            shape_scores_021322[spike_idx, template_idx] = shape_score
            energy_scores_021322[spike_idx, template_idx] = energy_score
        
        # 与022522的模板计算score
        for template_idx in range(n_neuron_022522):
            shape_score, energy_score = compute_template_score(
                spike_shape,
                spike_energy,
                shape_templates_022522[template_idx],
                energy_templates_022522[template_idx],
            )
            shape_scores_022522[spike_idx, template_idx] = shape_score
            energy_scores_022522[spike_idx, template_idx] = energy_score
        
        if (spike_idx + 1) % 1000 == 0:
            logging.info(f"Processed {spike_idx + 1}/{n_spikes} spikes")
    
    # 转换为numpy数组
    gt_neuron_indices = np.array(all_spike_neuron_indices, dtype=np.int32)
    
    return shape_scores_021322, energy_scores_021322, shape_scores_022522, energy_scores_022522, gt_neuron_indices


def compute_matching(
    shape_scores: np.ndarray,
    energy_scores: np.ndarray,
    threshold: float = 0.9,
) -> np.ndarray:
    """
    根据shape_scores和energy_scores计算匹配结果
    
    Args:
        shape_scores: (n_spike, n_neuron) 形状匹配分数
        energy_scores: (n_spike, n_neuron) 能量匹配分数
        threshold: 阈值，低于此值的score设置为0
    
    Returns:
        matched_neuron_indices: (n_spike,) 每个spike匹配到的neuron索引，-1表示unmatched
    """
    n_spikes, n_neurons = shape_scores.shape
    
    # 复制scores
    shape_scores_filtered = shape_scores.copy()
    energy_scores_filtered = energy_scores.copy()
    
    # 将 < threshold 的设置为0
    shape_scores_filtered[shape_scores_filtered < 0] = 0
    energy_scores_filtered[energy_scores_filtered < threshold] = 0
    
    # 计算sum_scores
    sum_scores = shape_scores_filtered + energy_scores_filtered
    
    # 对于每个spike，找到匹配的neuron
    matched_neuron_indices = np.zeros(n_spikes, dtype=np.int32)
    
    for spike_idx in range(n_spikes):
        spike_shape_scores = shape_scores_filtered[spike_idx]
        spike_energy_scores = energy_scores_filtered[spike_idx]
        spike_sum_scores = sum_scores[spike_idx]
        
        # 如果shape_scores或energy_scores的sum是0，则unmatched
        if np.sum(spike_shape_scores) == 0 or np.sum(spike_energy_scores) == 0:
            matched_neuron_indices[spike_idx] = -1
        else:
            # 取sum_scores最大值对应的neuron
            matched_neuron_indices[spike_idx] = np.argmax(spike_sum_scores)
    
    return matched_neuron_indices


def plot_matching_heatmap(
    gt_neuron_indices: np.ndarray,
    matched_neuron_indices: np.ndarray,
    n_neurons: int,
    output_path: Path,
    title: str = "Matching Results",
) -> None:
    """
    绘制匹配结果的热图（confusion matrix）
    
    Args:
        gt_neuron_indices: (n_spike,) 每个spike的真实neuron索引
        matched_neuron_indices: (n_spike,) 每个spike匹配到的neuron索引，-1表示unmatched
        n_neurons: neuron总数
        output_path: 输出路径
        title: 图表标题
    """
    # 创建confusion matrix
    confusion_matrix = np.zeros((n_neurons + 1, n_neurons + 1), dtype=np.int32)
    # 最后一行和最后一列用于unmatched（索引为-1的情况）
    
    for spike_idx in range(len(gt_neuron_indices)):
        gt_idx = gt_neuron_indices[spike_idx]
        matched_idx = matched_neuron_indices[spike_idx]
        
        # 将-1映射到n_neurons（最后一列/行）
        if matched_idx == -1:
            matched_idx = n_neurons
        
        confusion_matrix[gt_idx, matched_idx] += 1
    
    # 绘制热图
    plt.figure(figsize=(12, 10))
    
    # 创建标签
    labels = [f"Neuron {i}" for i in range(n_neurons)] + ["Unmatched"]
    
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Matched Neuron', fontsize=12)
    plt.ylabel('Ground Truth Neuron', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved matching heatmap to {output_path}")


def main() -> None:
    setup_logger(verbose=False)
    
    parser = argparse.ArgumentParser(
        description="Validate template matching between 021322 and 022522"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels",
        help="Base directory",
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: base_dir/kilosort_spike_sorting/validation_results)",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    sorting_root = base_dir / "kilosort_spike_sorting"
    probe_file = Path(args.probe_file)
    recording_root = Path(args.recording_root)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = sorting_root / "validation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义两个日期
    date1 = "021322"
    date2 = "022522"
    
    session_dir1 = sorting_root / "sorting_new" / date1
    session_dir2 = sorting_root / "sorting_new" / date2
    
    # 加载数据
    logging.info("=" * 80)
    logging.info("Loading data for %s and %s", date1, date2)
    
    # 加载模板
    shape_templates_021322 = np.load(session_dir1 / "shape_templates.npy")
    energy_templates_021322 = np.load(session_dir1 / "energy_templates.npy")
    shape_templates_022522 = np.load(session_dir2 / "shape_templates.npy")
    energy_templates_022522 = np.load(session_dir2 / "energy_templates.npy")
    
    logging.info(f"Loaded templates: 021322 shape {shape_templates_021322.shape}, energy {energy_templates_021322.shape}")
    logging.info(f"Loaded templates: 022522 shape {shape_templates_022522.shape}, energy {energy_templates_022522.shape}")
    
    # 加载neuron_inf和spike_inf
    with open(session_dir2 / "neuron_inf.pkl", "rb") as f:
        neuron_inf_022522 = pickle.load(f)
    spike_inf_022522 = pd.read_csv(session_dir2 / "spike_inf.tsv", sep='\t')
    
    logging.info(f"Loaded neuron_inf_022522: {len(neuron_inf_022522)} neurons")
    logging.info(f"Loaded spike_inf_022522: {len(spike_inf_022522)} spikes")
    
    # 加载recording数据（只需要022522的）
    raw_file_022522 = find_recording_file(date2, recording_root)
    if raw_file_022522 is None:
        raise FileNotFoundError(f"Recording file not found for {date2}")
    
    logging.info(f"Loading recording for {date2}: {raw_file_022522}")
    recording_022522 = load_recording(raw_file_022522, probe_file)
    
    # 加载traces（使用全部数据，不限制前10分钟）
    sampling_frequency = recording_022522.get_sampling_frequency()
    logging.info("Loading all traces from recording...")
    traces_022522 = recording_022522.get_traces().astype(np.float32)
    
    # 确保维度正确
    if traces_022522.shape[0] > traces_022522.shape[1] and traces_022522.shape[0] > 100:
        traces_022522 = traces_022522.T
        logging.info("Transposed traces to (n_channels, n_timepoints)")
    
    n_channels, max_frame = traces_022522.shape
    logging.info(f"Loaded {max_frame} samples ({max_frame/sampling_frequency:.2f} seconds) from recording, {n_channels} channels")
    
    # 1. 计算模板之间的匹配度
    logging.info("=" * 80)
    logging.info("Task 1: Computing template-to-template scores")
    shape_scores_template, energy_scores_template = compute_template_to_template_scores(
        shape_templates_021322,
        energy_templates_021322,
        shape_templates_022522,
        energy_templates_022522,
    )
    
    # 保存结果
    np.save(output_dir / "template_to_template_shape_scores.npy", shape_scores_template)
    np.save(output_dir / "template_to_template_energy_scores.npy", energy_scores_template)
    logging.info(f"Saved template-to-template scores: shape {shape_scores_template.shape}, energy {energy_scores_template.shape}")
    
    # 2. 计算spike到模板的匹配度
    logging.info("=" * 80)
    logging.info("Task 2: Computing spike-to-template scores")
    shape_scores_021322, energy_scores_021322, shape_scores_022522, energy_scores_022522, gt_neuron_indices = compute_spike_to_template_scores(
        neuron_inf_022522,
        spike_inf_022522,
        traces_022522,
        shape_templates_021322,
        energy_templates_021322,
        shape_templates_022522,
        energy_templates_022522,
        n_spikes_per_neuron=5000,
    )
    
    # 保存结果
    np.save(output_dir / "spike_to_template_021322_shape_scores.npy", shape_scores_021322)
    np.save(output_dir / "spike_to_template_021322_energy_scores.npy", energy_scores_021322)
    np.save(output_dir / "spike_to_template_022522_shape_scores.npy", shape_scores_022522)
    np.save(output_dir / "spike_to_template_022522_energy_scores.npy", energy_scores_022522)
    np.save(output_dir / "gt_neuron_indices.npy", gt_neuron_indices)
    
    logging.info(f"Saved spike-to-template scores:")
    logging.info(f"  vs 021322: shape {shape_scores_021322.shape}, energy {energy_scores_021322.shape}")
    logging.info(f"  vs 022522: shape {shape_scores_022522.shape}, energy {energy_scores_022522.shape}")
    logging.info(f"  gt_neuron_indices: {gt_neuron_indices.shape}")
    
    # 3. 计算匹配结果（只对022522的匹配）
    logging.info("=" * 80)
    logging.info("Task 3: Computing matching results for 022522")
    matched_neuron_indices = compute_matching(
        shape_scores_022522,
        energy_scores_022522,
        threshold=0.9,
    )
    
    # 保存匹配结果
    np.save(output_dir / "matched_neuron_indices.npy", matched_neuron_indices)
    logging.info(f"Saved matched_neuron_indices: {matched_neuron_indices.shape}")
    
    # 统计匹配结果
    n_unmatched = np.sum(matched_neuron_indices == -1)
    n_matched = len(matched_neuron_indices) - n_unmatched
    logging.info(f"Matching statistics: {n_matched} matched, {n_unmatched} unmatched ({n_unmatched/len(matched_neuron_indices)*100:.2f}%)")
    
    # 4. 绘制匹配热图
    logging.info("=" * 80)
    logging.info("Task 4: Plotting matching heatmap")
    n_neurons_022522 = len(neuron_inf_022522)
    plot_matching_heatmap(
        gt_neuron_indices,
        matched_neuron_indices,
        n_neurons_022522,
        output_dir / "matching_heatmap.pdf",
        title="Spike-to-Template Matching Results (022522)",
    )
    
    logging.info("=" * 80)
    logging.info("Validation completed! Results saved to %s", output_dir)


if __name__ == "__main__":
    main()

