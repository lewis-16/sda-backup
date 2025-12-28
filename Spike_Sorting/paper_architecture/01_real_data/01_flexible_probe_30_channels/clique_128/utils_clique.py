"""
AutoSort training utility functions
Includes: threshold detection, data preparation, model definition and training functions
"""

import numpy as np
import pandas as pd
import pickle
import math
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import scipy.signal
from scipy.spatial import ConvexHull
from typing import Dict, Iterable, List, Sequence, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, f1_score

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from spikeinterface.preprocessing.whiten import compute_whitening_matrix


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

def compute_valid_channels(
    recording_clique,
    neuron_inf_clique,
):
    """
    Compute valid_channels (clique column indices) that have GT neurons.
    
    Parameters:
        recording_clique: Recording object (clique subset)
        neuron_inf_clique: DataFrame containing neuron information for this clique
    
    Returns:
        valid_channels: List of clique column indices that have GT neurons (None if all channels are valid)
    """
    # Create mapping from original probe channel indices to clique column indices
    recording_channel_ids = recording_clique.get_channel_ids()
    probe_to_clique_index = {}
    for clique_idx, probe_ch in enumerate(recording_channel_ids):
        probe_to_clique_index[int(probe_ch)] = clique_idx
    
    # Get all tract_channels from neurons in this clique
    valid_probe_channels = set()
    for neuron_idx in range(len(neuron_inf_clique)):
        if 'tract_channel' in neuron_inf_clique.columns:
            neuron_channel_id_original = neuron_inf_clique['tract_channel'].iloc[neuron_idx]
            if pd.notna(neuron_channel_id_original) and neuron_channel_id_original is not None:
                valid_probe_channels.add(int(neuron_channel_id_original))
        else:
            # Try to get from channel_id
            channel_id = neuron_inf_clique['channel_id'].iloc[neuron_idx]
            if isinstance(channel_id, str):
                import ast
                try:
                    channel_id = ast.literal_eval(channel_id)
                except:
                    channel_id = []
            if isinstance(channel_id, (list, tuple, np.ndarray)) and len(channel_id) > 0:
                valid_probe_channels.add(int(channel_id[0]))
    
    # Convert to clique column indices (valid_channels for detection)
    valid_channels = []
    for probe_ch in valid_probe_channels:
        if probe_ch in probe_to_clique_index:
            valid_channels.append(probe_to_clique_index[probe_ch])
    
    return sorted(valid_channels) if len(valid_channels) > 0 else None


def analyze_gt_data_for_detection_params(
    recording_clique,
    spike_inf_clique,
    neuron_inf_clique,
    duration_seconds=200,
    sampling_rate=None,
):
    """
    Analyze GT data distribution to compute optimal detection parameters for a clique.
    
    Parameters:
        recording_clique: Recording object (clique subset)
        spike_inf_clique: DataFrame containing GT spike information for this clique
        neuron_inf_clique: DataFrame containing neuron information for this clique
        duration_seconds: Processing duration (seconds), default 200
        sampling_rate: Sampling rate (Hz), if None will be read from recording
    
    Returns:
        detection_params: Dictionary with optimized detection parameters:
            - thr_min: minimum threshold multiplier
            - thr_max: maximum threshold multiplier
            - distance: minimum distance between peaks (samples)
            - ch_max_simul_firing: maximum number of simultaneous firing channels
            - wlen: window length for peak detection
            - prominence: minimum peak prominence
        valid_channels: List of clique column indices that have GT neurons (None if all channels are valid)
        stats: Dictionary with analysis statistics
    """
    if sampling_rate is None:
        sampling_rate = recording_clique.get_sampling_frequency()
    
    max_frames = int(duration_seconds * sampling_rate)
    
    # Filter spike_inf to specified duration
    spike_inf_filtered = spike_inf_clique[spike_inf_clique['time'] < max_frames].copy()
    
    # Compute valid_channels first (even if no spikes)
    recording_channel_ids = recording_clique.get_channel_ids()
    probe_to_clique_index = {}
    for clique_idx, probe_ch in enumerate(recording_channel_ids):
        probe_to_clique_index[int(probe_ch)] = clique_idx
    
    valid_probe_channels = set()
    for neuron_idx in range(len(neuron_inf_clique)):
        if 'tract_channel' in neuron_inf_clique.columns:
            neuron_channel_id_original = neuron_inf_clique['tract_channel'].iloc[neuron_idx]
            if pd.notna(neuron_channel_id_original) and neuron_channel_id_original is not None:
                valid_probe_channels.add(int(neuron_channel_id_original))
        else:
            channel_id = neuron_inf_clique['channel_id'].iloc[neuron_idx]
            if isinstance(channel_id, str):
                import ast
                try:
                    channel_id = ast.literal_eval(channel_id)
                except:
                    channel_id = []
            if isinstance(channel_id, (list, tuple, np.ndarray)) and len(channel_id) > 0:
                valid_probe_channels.add(int(channel_id[0]))
    
    valid_channels = []
    for probe_ch in valid_probe_channels:
        if probe_ch in probe_to_clique_index:
            valid_channels.append(probe_to_clique_index[probe_ch])
    valid_channels = sorted(valid_channels) if len(valid_channels) > 0 else None
    
    if len(spike_inf_filtered) == 0:
        # No GT spikes, return default parameters
        return {
            'thr_min': 1.8,
            'thr_max': 15,
            'distance': 3,
            'ch_max_simul_firing': 8,
            'wlen': 10,
            'prominence': 5,
        }, valid_channels, {'n_spikes': 0, 'n_neurons': 0}
    
    # Get recording data for amplitude analysis
    trace0_car = recording_clique.get_traces(start_frame=0, end_frame=max_frames).astype(np.float32)
    
    # 1. Analyze spike frequency/density
    n_spikes = len(spike_inf_filtered)
    n_neurons = len(neuron_inf_clique)
    spike_rate = n_spikes / duration_seconds  # spikes per second
    spike_rate_per_neuron = spike_rate / n_neurons if n_neurons > 0 else 0
    
    # 2. Analyze spike time intervals (ISI - Inter-Spike Interval)
    spike_times = spike_inf_filtered['time'].values
    spike_times_sorted = np.sort(spike_times)
    if len(spike_times_sorted) > 1:
        isi = np.diff(spike_times_sorted)  # Inter-spike intervals
        min_isi = np.min(isi)
        median_isi = np.median(isi)
        p25_isi = np.percentile(isi, 25)
    else:
        min_isi = max_frames
        median_isi = max_frames
        p25_isi = max_frames
    
    # 3. Analyze simultaneous firing (spikes at same time point)
    time_counts = pd.Series(spike_times).value_counts()
    max_simul_firing = time_counts.max() if len(time_counts) > 0 else 1
    mean_simul_firing = time_counts.mean() if len(time_counts) > 0 else 1
    p95_simul_firing = time_counts.quantile(0.95) if len(time_counts) > 0 else 1
    
    # 4. Analyze spike amplitudes (extract waveforms at spike times)
    # Sample a subset of spikes for amplitude analysis (to avoid memory issues)
    n_samples = min(1000, len(spike_times))
    sample_indices = np.random.choice(len(spike_times), n_samples, replace=False)
    sample_times = spike_times[sample_indices]
    
    # Extract spike amplitudes (absolute values at spike times)
    amplitudes = []
    for spike_time in sample_times:
        if 0 <= int(spike_time) < trace0_car.shape[0]:
            # Get amplitude across all channels at this time point
            amp = np.abs(trace0_car[int(spike_time), :])
            amplitudes.append(np.max(amp))  # Maximum amplitude across channels
    
    if len(amplitudes) > 0:
        amplitudes = np.array(amplitudes)
        median_amplitude = np.median(amplitudes)
        p25_amplitude = np.percentile(amplitudes, 25)
        p75_amplitude = np.percentile(amplitudes, 75)
    else:
        median_amplitude = 0
        p25_amplitude = 0
        p75_amplitude = 0
    
    # 5. Analyze noise level (using MAD - Median Absolute Deviation)
    noise_std = np.median(np.abs(trace0_car) / 0.6745, axis=0)
    median_noise_std = np.median(noise_std)
    
    # Compute detection parameters based on analysis
    # thr_min: Based on noise level and spike amplitude (make it more sensitive)
    if median_amplitude > 0 and median_noise_std > 0:
        # Use 25th percentile amplitude to set threshold (more sensitive)
        # Reduce multiplier to make threshold lower (more sensitive)
        thr_min = max(1.5, min(2.5, (p25_amplitude / median_noise_std) * 0.5))
    else:
        thr_min = 1.8  # Default (lower than before for better sensitivity)
    
    # thr_max: Based on maximum amplitude
    if median_amplitude > 0 and median_noise_std > 0:
        thr_max = max(10, min(30, (p75_amplitude / median_noise_std) * 1.5))
    else:
        thr_max = 15  # Default
    
    # distance: Based on minimum ISI (with some margin)
    # Convert ISI from samples to time, then back to samples with margin
    if min_isi > 0:
        # Use 25th percentile ISI as reference, with 0.5x margin
        distance = max(2, int(p25_isi * 0.5))
    else:
        distance = 3  # Default
    
    # ch_max_simul_firing: Based on simultaneous firing statistics
    # Use 95th percentile with some margin
    ch_max_simul_firing = max(5, min(15, int(p95_simul_firing * 1.5)))
    
    # wlen: Based on spike rate (higher rate needs smaller window)
    if spike_rate > 0:
        # Higher spike rate -> smaller window to avoid overlap
        wlen = max(5, min(15, int(1000 / spike_rate * 0.1)))
    else:
        wlen = 10  # Default
    
    # prominence: Based on amplitude distribution
    if median_amplitude > 0 and median_noise_std > 0:
        # Use 25th percentile amplitude relative to noise
        prominence = max(3, min(15, int((p25_amplitude / median_noise_std) * 0.3)))
    else:
        prominence = 5  # Default
    
    detection_params = {
        'thr_min': float(thr_min),
        'thr_max': float(thr_max),
        'distance': int(distance),
        'ch_max_simul_firing': int(ch_max_simul_firing),
        'wlen': int(wlen),
        'prominence': int(prominence),
    }
    
    stats = {
        'n_spikes': n_spikes,
        'n_neurons': n_neurons,
        'spike_rate': spike_rate,
        'spike_rate_per_neuron': spike_rate_per_neuron,
        'min_isi': float(min_isi),
        'median_isi': float(median_isi),
        'p25_isi': float(p25_isi),
        'max_simul_firing': int(max_simul_firing),
        'mean_simul_firing': float(mean_simul_firing),
        'p95_simul_firing': float(p95_simul_firing),
        'median_amplitude': float(median_amplitude),
        'p25_amplitude': float(p25_amplitude),
        'p75_amplitude': float(p75_amplitude),
        'median_noise_std': float(median_noise_std),
    }
    
    return detection_params, valid_channels, stats


def detect_on_channel(data: np.ndarray, *, detect_threshold: float, detect_interval: float, detect_sign: int, margin: int=0):
    """
    MountainSort4 detection method for a single channel.
    This is identical to mountainsort4/ms4alg.py detect_on_channel function.
    
    Parameters:
        data: 1D numpy array, signal for one channel
        detect_threshold: detection threshold value
        detect_interval: minimum interval between detections (samples)
        detect_sign: detection sign (-1 for negative, 0 for absolute, 1 for positive)
        margin: margin to exclude from boundaries (samples)
    
    Returns:
        times: array of detected spike times
    """
    # Adjust the data to accommodate the detect_sign
    # After this adjustment, we only need to look for positive peaks
    if detect_sign < 0:
        data = data * (-1)
    elif detect_sign == 0:
        data = np.abs(data)
    elif detect_sign > 0:
        pass

    data = data.ravel()

    # An event at timepoint t is flagged if the following two criteria are met:
    # 1. The value at t is greater than the detection threshold (detect_threshold)
    # 2. The value at t is greater than the value at any other timepoint within plus or minus <detect_interval> samples

    # First split the data into segments of size detect_interval (don't worry about timepoints left over, we assume we have padding)
    N = len(data)
    S2 = math.floor(N / detect_interval)
    N2 = S2 * detect_interval
    data2 = np.reshape(data[0:N2], (S2, detect_interval))

    # Find the maximum on each segment (these are the initial candidates)
    max_inds2 = np.argmax(data2, axis=1)
    max_inds = max_inds2 + detect_interval * np.arange(0, S2)
    max_vals = data[max_inds]

    # The following two tests compare the values of the candidates with the values of the neighbor candidates
    # If they are too close together, then discard the one that is smaller by setting its value to -1
    # Actually, this doesn't strictly satisfy the above criteria but it is close
    # TODO: fix the subtlety
    max_vals[np.where((max_inds[0:-1] >= max_inds[1:] - detect_interval)
                      & (max_vals[0:-1] < max_vals[1:]))[0]] = -1
    max_vals[1 + np.array(np.where((max_inds[1:] <= max_inds[0:-1] +
                        detect_interval) & (max_vals[1:] <= max_vals[0:-1]))[0])] = -1

    # Finally we use only the candidates that satisfy the detect_threshold condition
    times = max_inds[np.where(max_vals >= detect_threshold)[0]]
    if margin > 0:
        times = times[np.where((times >= margin) & (times < N - margin))[0]]

    return times


def detect_spike(
    trace0_car,
    detect_threshold=None,
    detect_interval=10,
    detect_sign=-1,
    margin=0,
    valid_channels=None,
    thr_min=3,
):
    """
    Spike detection using MountainSort4 method (identical to ms4alg.py).
    
    Parameters:
        trace0_car: numpy array, shape (n_timepoints, n_channels)
        detect_threshold: detection threshold value (if None, calculated from thr_min * noise_std)
        detect_interval: minimum interval between detections (samples), default 10
        detect_sign: detection sign (-1 for negative, 0 for absolute, 1 for positive), default -1
        margin: margin to exclude from boundaries (samples), default 0
        valid_channels: list of valid channel indices to detect on, if None detect on all channels
        thr_min: minimum threshold multiplier (relative to noise std), default 3 (used if detect_threshold is None)
    
    Returns:
        spikes: binary matrix (n_timepoints, n_channels), 1 indicates detected spike
    """
    spikes = np.zeros(trace0_car.shape, dtype=int)
    
    if trace0_car.ndim > 1:
        n_channels = trace0_car.shape[1]
        
        # Calculate detect_threshold if not provided
        if detect_threshold is None:
            noise_std_detect = np.median(abs(trace0_car) / 0.6745, axis=0)
            detect_threshold_per_channel = thr_min * noise_std_detect
        else:
            # If single value provided, use for all channels
            if np.isscalar(detect_threshold):
                detect_threshold_per_channel = np.full(n_channels, detect_threshold)
            else:
                detect_threshold_per_channel = np.array(detect_threshold)
        
        # Determine which channels to process
        channels_to_process = range(n_channels)
        if valid_channels is not None:
            # Only process channels in valid_channels list
            channels_to_process = [ch for ch in channels_to_process if ch in valid_channels]
        
        # Detect spikes on each channel using MS4 method
        for i in channels_to_process:
            # Get signal for this channel
            signal = trace0_car[:, i]
            
            # Detect spikes using MS4 method (identical to mountainsort4)
            detected_times = detect_on_channel(
                signal,
                detect_threshold=detect_threshold_per_channel[i],
                detect_interval=detect_interval,
                detect_sign=detect_sign,
                margin=margin
            )
            
            # Mark detected spikes
            spikes[detected_times, i] = 1
    
    return spikes




def map_gt_annotation(detect_array, gt_array, neuron_to_channel_id=None, time_tolerance=1):
    """
    AutoSort GT mapping function (updated version with channel_id matching)
    
    新逻辑：对于每个GT neuron的spike，如果检测结果中时间offset<=time_tolerance，
    且检测结果所在的channel位于该neuron的channel_id内，则匹配
    
    Parameters:
        detect_array: numpy array, shape (n_detected, 2), each row is [time_point, channel_id]
        gt_array: numpy array, shape (n_gt, 2), each row is [time_point, neuron_name] (new format)
                  or shape (n_gt, 2), each row is [time_point, channel_id] (old format for backward compatibility)
        neuron_to_channel_id: dict, neuron_name -> set of channel_indices (channel_id), optional
                             If None, uses old matching logic (exact channel match)
        time_tolerance: int, time tolerance in samples, default 1
    
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
    
    # Check if using new format (gt_array with neuron names) or old format (gt_array with channel indices)
    if neuron_to_channel_id is not None:
        # New format: gt_array contains [time_point, neuron_name]
        gt_times = gt_array[:, 0].astype(np.int64)
        gt_neurons = gt_array[:, 1]  # neuron names
        
        # Use dictionary to speed up lookup: key = (time, neuron_name), value = GT index list
        gt_dict = defaultdict(list)
        for idx, (t, neuron_name) in enumerate(zip(gt_times, gt_neurons)):
            gt_dict[(t, neuron_name)].append(idx)
        
        # Track which GT spikes have been matched (to avoid duplicate matching)
        matched_gt_set = set()
        
        # Try to match each detected spike with time offsets: 0, -1, +1 (in priority order)
        time_offsets = list(range(-time_tolerance, time_tolerance + 1))
        
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
            
            # For each unmatched detected spike, check if it matches any GT spike
            for i, det_idx in enumerate(unmatched_indices):
                det_time = shifted_times[i]
                det_ch = unmatched_channels[i]
                
                # Check all neurons to see if det_ch is in their channel_id
                for neuron_name, channel_id_set in neuron_to_channel_id.items():
                    if det_ch not in channel_id_set:
                        continue
                    
                    # Check if time matches
                    key = (det_time, neuron_name)
                    if key in gt_dict:
                        # Find first unmatched GT spike in the list
                        for gt_idx in gt_dict[key]:
                            # Check if this GT spike is already matched
                            if gt_idx not in matched_gt_set:
                                gt_label_array1[det_idx] = gt_idx
                                matched_gt_set.add(gt_idx)
                                break
                        if gt_label_array1[det_idx] != -1:
                            break  # Found match, move to next detected spike
    else:
        # Old format: gt_array contains [time_point, channel_id] (backward compatibility)
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


# ==================== 2. Training Data Preparation ====================

def compute_and_apply_whitening(
    recording,
    whitening_duration_seconds=60,
    whitening_matrix_save_path=None,
    random_chunk_kwargs=None,
):
    """
    计算白化矩阵并应用到recording
    
    Parameters:
        recording: Recording对象
        whitening_duration_seconds: 用于计算白化矩阵的数据时长（秒），默认60
        whitening_matrix_save_path: 白化矩阵保存路径（Path对象），如果为None则不保存
        random_chunk_kwargs: 随机采样参数，默认None（使用默认值）
    
    Returns:
        recording_whitened: 白化后的recording对象
        W: 白化矩阵 (n_channels, n_channels)
        M: 均值向量 (1, n_channels) 或 None
    """
    if random_chunk_kwargs is None:
        random_chunk_kwargs = {
            "num_chunks_per_segment": 50,
            "chunk_size": 10000,
            "seed": 42
        }
    
    # 检查是否已有预计算的白化矩阵
    W = None
    M = None
    if whitening_matrix_save_path is not None:
        whitening_matrix_save_path = Path(whitening_matrix_save_path)
        whitening_matrix_save_path.mkdir(parents=True, exist_ok=True)
        W_save_path = whitening_matrix_save_path / "whitening_matrix_W.npy"
        M_save_path = whitening_matrix_save_path / "whitening_matrix_M.npy"
        
        if W_save_path.exists() and M_save_path.exists():
            print("[INFO] 加载已保存的白化矩阵...")
            W = np.load(W_save_path)
            # 加载 M 时允许 pickle（因为可能保存了 None）
            M = np.load(M_save_path, allow_pickle=True)
            # 检查 M 是否为 None
            if isinstance(M, np.ndarray) and M.size == 1 and M.item() is None:
                M = None
            print(f"  白化矩阵形状: {W.shape}")
            print(f"  均值向量形状: {M.shape if M is not None else None}")
    
    # 如果没有预计算的白化矩阵，则计算
    if W is None:
        print(f"[INFO] 从初始 {whitening_duration_seconds} 秒数据计算白化矩阵...")
        sampling_rate = recording.get_sampling_frequency()
        whitening_frames = int(whitening_duration_seconds * sampling_rate)
        
        # 截取初始数据用于计算白化矩阵
        if recording.get_num_segments() == 1:
            recording_for_whitening = recording.frame_slice(start_frame=0, end_frame=whitening_frames)
        else:
            # 如果是多段，先取第一段
            print(f"  警告: recording 有 {recording.get_num_segments()} 段，仅使用第一段计算白化矩阵")
            recording_for_whitening = recording.select_segments([0])
            if whitening_frames < recording_for_whitening.get_num_frames():
                recording_for_whitening = recording_for_whitening.frame_slice(start_frame=0, end_frame=whitening_frames)
        
        # 计算白化矩阵
        # apply_mean=True: 计算并应用均值，M 将是均值向量
        # apply_mean=False: 不计算均值，M 将是 None
        W, M = compute_whitening_matrix(
            recording_for_whitening,
            mode="global",
            random_chunk_kwargs=random_chunk_kwargs,
            apply_mean=True,  # 改为 True，这样 M 会是均值向量而不是 None
            radius_um=None,
            eps=None,
            regularize=False
        )
        
        # 保存白化矩阵
        if whitening_matrix_save_path is not None:
            np.save(W_save_path, W)
            if M is not None:
                np.save(M_save_path, M)
            else:
                # 保存 None 作为占位符
                np.save(M_save_path, np.array(None))
            print(f"  白化矩阵已保存到: {whitening_matrix_save_path}")
        
        print(f"  白化矩阵形状: {W.shape}")
    
    # 应用白化到完整数据
    recording_whitened = spre.whiten(recording, W=W, M=M, dtype="float32")
    print("[INFO] 白化预处理完成")
    
    return recording_whitened, W, M


def whiten_traces(traces, W, M=None):
    """
    对traces数组应用白化矩阵（用于实时处理或已提取的traces）
    
    Parameters:
        traces: numpy array, shape (n_samples, n_channels)
        W: 白化矩阵 (n_channels, n_channels)
        M: 均值向量 (1, n_channels) 或 (n_channels,) 或 None
    
    Returns:
        whitened_traces: 白化后的数据 (n_samples, n_channels)
    """
    traces = traces.astype(np.float32)
    
    if M is not None and not (isinstance(M, np.ndarray) and M.size == 1 and M.item() is None):
        # Ensure M has the correct shape for broadcasting
        # M can be (1, n_channels) or (n_channels,), we need (1, n_channels) for broadcasting
        if M.ndim == 1:
            M = M.reshape(1, -1)  # Reshape (n_channels,) to (1, n_channels)
        elif M.shape[0] != 1:
            # If M is (n_channels, 1) or other shape, reshape to (1, n_channels)
            M = M.flatten().reshape(1, -1)
        
        whiten_traces = (traces - M) @ W
    else:
        whiten_traces = traces @ W
    
    return whiten_traces


def inverse_whiten_traces(whitened_traces, W, M=None):
    """
    对白化后的traces数组进行逆白化，恢复到原始数据
    
    Parameters:
        whitened_traces: numpy array, shape (n_samples, n_channels), 白化后的数据
        W: 白化矩阵 (n_channels, n_channels)
        M: 均值向量 (1, n_channels) 或 None
    
    Returns:
        original_traces: 逆白化后的原始数据 (n_samples, n_channels)
    
    Note:
        白化公式: whitened = (original - M) @ W
        逆白化公式: original = whitened @ W^(-1) + M
        注意：W通常不是正交矩阵，所以W^(-1) ≠ W^T，需要使用np.linalg.inv(W)
    """
    whitened_traces = whitened_traces.astype(np.float32)
    
    # 逆白化: original = whitened @ W^(-1) + M
    # 注意：W通常不是正交矩阵，所以不能简单地使用W^T，需要计算W的逆矩阵
    W_inv = np.linalg.inv(W)  # 计算白化矩阵的逆矩阵
    
    if M is not None and not (isinstance(M, np.ndarray) and M.size == 1 and M.item() is None):
        # Ensure M has the correct shape for broadcasting
        if M.ndim == 1:
            M = M.reshape(1, -1)  # Reshape (n_channels,) to (1, n_channels)
        elif M.shape[0] != 1:
            # If M is (n_channels, 1) or other shape, reshape to (1, n_channels)
            M = M.flatten().reshape(1, -1)
        
        original_traces = whitened_traces @ W_inv + M
    else:
        original_traces = whitened_traces @ W_inv
    
    return original_traces


def extract_waveforms(trace0_car, X_spiketrain_time, left_sample=20, right_sample=40):
    """
    Extract waveform window (no downsampling)
    
    Parameters:
        trace0_car: numpy array, shape (n_timepoints, n_channels) - whitened data
        X_spiketrain_time: numpy array, shape (n_spikes,), spike time points
        left_sample: number of samples before spike to extract, default 20
        right_sample: number of samples after spike to extract, default 40
    
    Returns:
        waveform: numpy array, shape (n_spikes, n_channels, left_sample + right_sample)
        valid_mask: boolean array indicating valid spikes
    """
    # Filter spikes near boundaries (ensure complete window can be extracted)
    window_size = left_sample + right_sample
    valid_mask = (X_spiketrain_time >= left_sample) & (X_spiketrain_time < trace0_car.shape[0] - right_sample)
    X_spiketrain_time = X_spiketrain_time[valid_mask]
    
    # Extract full window (no downsampling)
    waveform = np.zeros((len(X_spiketrain_time), trace0_car.shape[1], window_size), dtype=np.float32)
    
    for i, time_range in enumerate(tqdm(np.arange(-left_sample, right_sample), desc="Extracting waveforms")):
        waveform[:, :, i] = trace0_car[X_spiketrain_time + time_range, :]
    
    # waveform shape: (n_spikes, n_channels, left_sample + right_sample)
    return waveform, valid_mask


def prepare_training_data(
    recording_f,
    spike_inf,
    neuron_inf,
    save_dir,
    duration_seconds=200,
    left_sample=20,  # Changed: now 20 samples (no downsampling)
    right_sample=40,  # Changed: now 40 samples (no downsampling)
    apply_whitening=True,
    detect_threshold=3,  # Mountainsort4: absolute threshold value
    detect_interval=10,  # Mountainsort4: minimum interval between detections (samples)
    detect_sign=-1,  # Mountainsort4: -1 (negative peaks), 0 (both), 1 (positive peaks)
    margin=0,  # Mountainsort4: margin to exclude from boundaries (samples)
    apply_bandpass_filter=True,  # Mountainsort4: whether to apply bandpass filter in preprocessing
    freq_min=300,  # Mountainsort4: bandpass filter minimum frequency
    freq_max=3000,  # Mountainsort4: bandpass filter maximum frequency
    valid_channels=None,  # List of valid channel indices (clique channel indices) to detect on, if None detect on all channels
):
    """
    Prepare training data using Mountainsort4 detection method (complete pipeline: preprocessing -> detection -> matching -> waveform extraction -> saving)
    
    This function performs clique-level detection. It receives a recording_clique (subset of channels)
    and performs preprocessing, detection, matching, and waveform extraction on this clique.
    
    This function follows the exact same preprocessing pipeline as evaluate_mountainsort4_detection.py:
    1. Frame slice (if duration_seconds is specified)
    2. Bandpass filter (if apply_bandpass_filter=True)
    3. Whiten
    
    Parameters:
        recording_f: preprocessed recording object (should be recording_clique for clique-level processing)
        spike_inf: DataFrame containing GT spike information (filtered to neurons in clique)
        neuron_inf: DataFrame containing neuron information (filtered to neurons in clique)
        save_dir: save directory path
        duration_seconds: processing duration (seconds), default 200
        left_sample, right_sample: waveform window parameters
        apply_whitening: bool, whether to apply whitening preprocessing, default True
        detect_threshold: float, Mountainsort4 absolute threshold value (default 3)
        detect_interval: int, Mountainsort4 minimum interval between detections (samples, default 10)
        detect_sign: int, Mountainsort4 detection sign: -1 (negative peaks), 0 (both), 1 (positive peaks), default -1
        margin: int, Mountainsort4 margin to exclude from boundaries (samples, default 0)
        apply_bandpass_filter: bool, whether to apply bandpass filter in preprocessing, default True
        freq_min: float, bandpass filter minimum frequency, default 300
        freq_max: float, bandpass filter maximum frequency, default 3000
        valid_channels: list of int, valid channel indices (clique channel indices) to detect on, if None detect on all channels, default None
    
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
    
    print(f"Recording total length: {total_frames} samples ({total_frames/sampling_rate:.2f} seconds)")
    print(f"Will process first {actual_frames} samples ({actual_frames/sampling_rate:.2f} seconds)")
    

    recording_channel_ids = recording_f.get_channel_ids()  
    probe_to_clique_index = {} 
    for clique_idx, probe_ch in enumerate(recording_channel_ids):
        probe_to_clique_index[int(probe_ch)] = clique_idx
    
    # Apply Mountainsort4 preprocessing pipeline (完全按照evaluate_mountainsort4_detection.py的流程)
    print("\n[INFO] 应用Mountainsort4预处理流程（与evaluate_mountainsort4_detection.py保持一致）")
    
    # 步骤1: 如果指定了处理时长，先截取数据（在预处理前截取）
    if duration_seconds is not None:
        recording_f = recording_f.frame_slice(start_frame=0, end_frame=actual_frames)
        print(f"  截取前 {duration_seconds} 秒数据: {actual_frames} samples")
    
    # 步骤2: Bandpass filter（如果启用）
    if apply_bandpass_filter and freq_min is not None and freq_max is not None:
        print("  应用bandpass filter...")
        print(f"    频率范围: {freq_min} - {freq_max} Hz")
        recording_f = spre.bandpass_filter(
            recording=recording_f,
            freq_min=freq_min,
            freq_max=freq_max
        )
    
    # 步骤3: Whiten（计算并保存白化矩阵，用于检测；waveform 提取使用原始数据）
    # 计算并保存白化矩阵，以便在评估阶段直接使用
    W = None
    M = None
    trace0_car = None        # 白化后的数据，用于检测
    trace0_original = None   # 原始数据，用于 waveform 提取
    if apply_whitening:
        print("  应用whiten预处理（用于阈值检测和waveform提取）...")
        print("    使用默认的random_chunk_kwargs（每个segment采样20个chunks）")
        # 使用compute_and_apply_whitening函数来计算白化矩阵并应用
        # 这样可以在训练阶段保存白化矩阵，供评估阶段使用
        save_dir_path = Path(save_dir)
        whitening_matrices_dir = save_dir_path / "whitening_matrices"
        whitening_matrices_dir.mkdir(parents=True, exist_ok=True)
        whitening_matrix_save_path = whitening_matrices_dir  # 保存到whitening_matrices目录
        
        # 保存原始数据（用于 waveform 提取）
        trace0_original = recording_f.get_traces().astype(np.float32)
        print(f"Original data shape: {trace0_original.shape} (clique channels, 已截取，用于waveform提取)")
        
        recording_f_whitened, W, M = compute_and_apply_whitening(
            recording_f,
            whitening_duration_seconds=duration_seconds,  # 使用全部训练数据计算白化矩阵
            whitening_matrix_save_path=whitening_matrix_save_path,
            random_chunk_kwargs=None  # 使用默认值
        )
        print("    白化完成（用于阈值检测）")
        
        # 获取白化后的数据（用于检测）
        trace0_car = recording_f_whitened.get_traces().astype(np.float32)
        print(f"Whitened data shape: {trace0_car.shape} (clique channels, 已截取，用于检测)")
    else:
        # 如果不白化，使用原始数据（用于检测和waveform提取）
        trace0_car = recording_f.get_traces().astype(np.float32)
        trace0_original = trace0_car.copy()
        print(f"Data shape: {trace0_car.shape} (clique channels, 已截取，未白化，用于检测和waveform提取)")
    
    # Compute valid_channels: union of all channel_id (before detection)
    # If valid_channels is not provided, compute from neuron_inf's channel_id
    if valid_channels is None:
        print("\n### 1.1. Compute valid_channels from channel_id")
        all_channel_id_set = set()  # Union of all channel_id for valid_channels
        
        for neuron_idx in range(len(neuron_inf)):
            # 获取neuron的所有channel_id（channel_id列表）
            channel_id = neuron_inf['channel_id'].iloc[neuron_idx]
            if isinstance(channel_id, str):
                import ast
                try:
                    channel_id = ast.literal_eval(channel_id)
                except:
                    channel_id = []
            if isinstance(channel_id, (list, tuple, np.ndarray)) and len(channel_id) > 0:
                # channel_id是列表，包含所有channels（probe的device_channel_indices）
                channel_id_list = [int(ch) for ch in channel_id]
                
                # Map probe channel indices to clique column indices
                for probe_channel_index in channel_id_list:
                    if probe_channel_index in probe_to_clique_index:
                        clique_channel_index = probe_to_clique_index[probe_channel_index]
                        all_channel_id_set.add(clique_channel_index)
        
        if len(all_channel_id_set) > 0:
            valid_channels = sorted(list(all_channel_id_set))
            print(f"Valid channels (union of all channel_id): {valid_channels} (共 {len(valid_channels)} 个通道)")
        else:
            valid_channels = None
            print("Warning: No valid channels found from channel_id, will detect on all channels")
    else:
        print(f"\n### 1.1. Using provided valid_channels: {valid_channels} (共 {len(valid_channels)} 个通道)")
    
    # 使用 Mountainsort4 检测方法（使用白化数据）
    print(f"\n[INFO] 使用 MountainSort4 检测方法")
    print(f"  阈值: {detect_threshold}, 间隔: {detect_interval}, 检测方向: {detect_sign}, 边界margin: {margin}")
    if valid_channels is not None:
        print(f"  仅在以下通道上进行检测: {valid_channels} (共 {len(valid_channels)} 个通道)")
    else:
        print(f"  在所有通道上进行检测 (共 {n_channels} 个通道)")
    print("  注意：检测使用白化数据，waveform提取使用原始数据")
    spikes = detect_spike(
        trace0_car,
        detect_threshold=detect_threshold,
        detect_interval=detect_interval,
        detect_sign=detect_sign,
        margin=margin,
        valid_channels=valid_channels,
    )
    
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
    
    print("\n### 2. Load Ground Truth and Match")
    
    # Filter spike_inf, keep only data within specified duration
    spike_inf_filtered = spike_inf[spike_inf['time'] < max_frames].copy()
    
    # Build gt_array using new format: [time_point, neuron_name]
    # Also build neuron_to_channel_id mapping
    print("Building gt_array...")
    spike_train_all = []
    y_unit_id = []
    neuron_to_channel_id = {}  # neuron_name -> set of clique_channel_indices (from channel_id)
    
    for neuron_idx in range(len(neuron_inf)):
        neuron_name = neuron_inf['Neuron'].iloc[neuron_idx]
        channel_id_list = []
        
        # 获取neuron的所有channel_id（channel_id列表）
        channel_id = neuron_inf['channel_id'].iloc[neuron_idx]
        if isinstance(channel_id, str):
            import ast
            try:
                channel_id = ast.literal_eval(channel_id)
            except:
                channel_id = []
        if isinstance(channel_id, (list, tuple, np.ndarray)) and len(channel_id) > 0:
            # channel_id是列表，包含所有channels（probe的device_channel_indices）
            channel_id_list = [int(ch) for ch in channel_id]
        
        # 如果没有channel_id或为空，跳过这个neuron（不再使用tract_channel作为fallback）
        if len(channel_id_list) == 0:
            continue
        
        # Map probe channel indices to clique column indices
        channel_id_clique_indices = []
        for probe_channel_index in channel_id_list:
            if probe_channel_index in probe_to_clique_index:
                clique_channel_index = probe_to_clique_index[probe_channel_index]
                channel_id_clique_indices.append(clique_channel_index)
        
        if len(channel_id_clique_indices) == 0:
            # This neuron's channels are not in the clique, skip
            continue
        
        # Store channel_id for this neuron (clique channel indices)
        neuron_to_channel_id[neuron_name] = set(channel_id_clique_indices)
        
        # Get spikes for this neuron
        neuron_spikes = spike_inf_filtered[spike_inf_filtered['neuron'] == neuron_name]
        if len(neuron_spikes) > 0:
            spike_times = neuron_spikes['time'].values
            spike_train_all += list(spike_times)
            y_unit_id += [neuron_name] * len(spike_times)
    
    # Build gt_array: [time_point, neuron_name]
    gt_array = np.array([[t, n] for t, n in zip(spike_train_all, y_unit_id)], dtype=object)
    print(f"GT spike count: {len(gt_array)}")
    print(f"Neurons with channel_id: {len(neuron_to_channel_id)}")
    
    # Use map_gt_annotation function with new matching logic
    # For each GT neuron's spike, match if: time offset <= time_tolerance AND detected channel is in neuron's channel_id
    gt_label_array1 = map_gt_annotation(detect_array, gt_array, neuron_to_channel_id=neuron_to_channel_id, time_tolerance=1)
    
    # Calculate detection rate
    # gt_label_array1 shape is (n_detected,), values are GT indices or -1 (unmatched)
    # To calculate how many GT spikes were matched, we need to count unique matched GT indices
    matched_gt_indices = gt_label_array1[gt_label_array1 >= 0]  # Get all matched GT indices
    n_matched_gt = len(np.unique(matched_gt_indices)) if len(matched_gt_indices) > 0 else 0
    detection_rate = n_matched_gt / gt_array.shape[0] if gt_array.shape[0] > 0 else 0.0
    print(f"---spike detection rate: {detection_rate:.4f}")
    
    # Build Y_spiketrain_id
    Y_spiketrain_id = np.full((detect_array.shape[0],), None, dtype=object)
    matched_indices = np.where(gt_label_array1 > -1)[0]
    if len(matched_indices) > 0:
        y_unit_id_array = np.array(y_unit_id, dtype=object)
        Y_spiketrain_id[matched_indices] = y_unit_id_array[
            gt_label_array1[matched_indices].astype("int")
        ]
    
    print(f"Number of matched spikes: {len(matched_indices)}")
    print(f"Number of unmatched spikes: {len(detect_array) - len(matched_indices)}")
    
    print("\n### 3. Extract Waveforms (using original traces, no downsampling)")
    print("  Extracting 30 samples (left 10, right 20), no downsampling")
    print("  注意：检测使用白化数据，waveform提取使用原始数据")
    
    # Extract waveforms from original traces (no downsampling)
    # trace0_original contains original (non-whitened) data with only clique channels
    # trace0_car contains whitened data (used for detection only)
    # detect_array uses clique column indices
    waveform, valid_mask = extract_waveforms(
        trace0_original, X_spiketrain_time, left_sample=10, right_sample=20
    )
    
    # Apply valid_mask filter (keep only valid spikes)
    # waveform 已经在 extract_waveforms 内部按有效 time 生成，无需再用 valid_mask 过滤
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

        self.way4 = nn.Sequential(
            nn.Linear(100, 30, bias=True),
            nn.BatchNorm1d(30),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(30, num_classes, bias=True)

    def forward(self, x):
        x = self.pool(x[None, :])
        x = x.reshape(x.size(1), -1)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        x = self.way4(x)

        logits = self.cls(x)
        return logits

    def intermediate_forward(self, x):
        x = self.pool(x[None, :])
        x = x.reshape(x.size(1), -1)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        x = self.way4(x)
        return x


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
        
    
    def __len__(self):
        return len(self.GT)
    
    def __getitem__(self, index):
        # Returns: multi-channel waveform, unit classification labels, noise/non-noise labels, single-channel waveform
        return (
            self.Img[index, ...],         # (n_channels, window_length)
            self.GT[index, ...],          # (n_units,) one-hot
            self.GT_binary[index, ...],   # (2,) [noise, spike]
            self.Img_single[index, ...],  # (window_length,)
        )


class SimpleAutoSort:
    """
    Simplified AutoSort model (without position information)
    Input: multi-waveform + single-waveform
    Identical to original AutoSort except position information is removed
    """
    def __init__(self, ch_num, samplepoints, device, set_shank_id, save_dir, 
                 pos_weight_noise=None, pos_weight_label=None):
        # Input dimension: (ch_num + 1) * samplepoints (without top-3 channel ids)
        input_dim = (ch_num + 1) * samplepoints
        
        self.clsfier_noise = SimpleClassifier(input_dim, 2).to(device)
        self.clsfier_label = SimpleClassifier(input_dim, len(set_shank_id)).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.clsfier_noise.parameters()},
            {'params': self.clsfier_label.parameters()},
        ], lr=1e-4)
        
        self.criterion = nn.MSELoss()  # Same as original (though not used)
        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_noise)
        self.bceloss_label = nn.BCEWithLogitsLoss(pos_weight=pos_weight_label)
        
        import os
        self.save_model_path_2 = os.path.join(save_dir, 'multitask_single_wave_clsfier_noise_clsfier.pth')
        self.save_model_path_3 = os.path.join(save_dir, 'multitask_single_wave_clsfier_label_clsfier.pth')
        
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
        Training iteration
        """
        self.optimizer.zero_grad()
        
        # Concatenate multi-waveform and single-waveform
        codes = torch.cat((batch_features, single_waveform), axis=1)
        
        # Noise classification
        cls_output = self.clsfier_noise(codes.float())
        
        # Unit classification (only for non-noise samples)
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
        Evaluation iteration
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
            train_classification_loss = torch.tensor(0.0, device=codes.device)
            gt_label_class = torch.tensor([], dtype=torch.long, device=codes.device)
            pred_class = torch.tensor([], dtype=torch.long, device=codes.device)
        
        train_detection_loss = 1000 * self.bceloss(cls_output, labels)
        train_loss = train_detection_loss + train_classification_loss
        
        return train_detection_loss.item(), train_classification_loss.item(), gt, pred, gt_label_class, pred_class


# ==================== 4. Training Function ====================

def train_autosort_model(
    train_data_dir,
    model_save_dir,
    n_channels,
    left_sample=20,  # Changed: now 20 samples (no downsampling)
    right_sample=40,  # Changed: now 40 samples (no downsampling)
    epochs=20,
    batch_size=512,
    device=None,
    early_stopping=True,
    patience=5,
    min_delta=0.0,
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
    
    Returns:
        autosort_model: trained model
        training_log: training log
    """
    # Import torch at function level to avoid UnboundLocalError
    import torch
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model save directory
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set parameters
    samplepoints = left_sample + right_sample
    
    # Create dataset
    print("Create dataset...")
    dataset = SimpleWaveformLoader(
        root=str(train_data_dir) + '/',
        shank_channel=np.arange(n_channels),
        Keep_id=None
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
    
    # Create model
    autosort_model = SimpleAutoSort(
        ch_num=n_channels,
        samplepoints=samplepoints,
        device=device,
        set_shank_id=set_shank_id,
        save_dir=model_save_dir,
        pos_weight_noise=dataset.pos_weight_noise.to(device),
        pos_weight_label=dataset.pos_weight_label.to(device)
    )
    
    # Split training and validation sets with fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Save train/val split indices for later use in evaluation
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    train_val_split_path = os.path.join(model_save_dir, 'train_val_split_indices.pkl')
    with open(train_val_split_path, 'wb') as f:
        pickle.dump((train_indices, val_indices), f)
    print(f"Saved train/val split indices to: {train_val_split_path}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"\nDataset split:")
    print(f"  - Training set: {train_size} samples")
    print(f"  - Validation set: {val_size} samples")
    
    # Training parameters
    min_valid_loss = np.inf
    min_val_acc = np.inf  # Track minimum validation accuracy (for early stopping)
    best_acc_epoch = 0  # Epoch with minimum accuracy
    patience_counter = 0  # Early stopping counter
    
    # training log
    training_log = {'epoch': [],
                    'validation_acc_noise':[],
                    'validation_acc_label':[]}
    
    print(f"\nStarting training (total {epochs} epochs)...")
    if early_stopping:
        print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")
    
    for epoch in range(epochs):
        training_log['epoch'].append(epoch + 1)
        print("epoch : {}/{}".format(epoch + 1, epochs))
        
        # Training phase
        detection_loss = 0
        classification_loss = 0
        autosort_model.train()
        autosort_model.bceloss.pos_weight = autosort_model.bceloss.pos_weight.to(device)
        autosort_model.bceloss_label.pos_weight = autosort_model.bceloss_label.pos_weight.to(device)
        
        for batch_features, classify_labels, labels, single_waveform in tqdm(train_loader, desc="Training"):
            classify_labels = classify_labels.to(device)
            # batch_features shape: (batch_size, n_channels, window_length)
            # Reshape to (batch_size, n_channels * window_length) while preserving batch_size
            batch_size = batch_features.shape[0]
            batch_features = batch_features.reshape(batch_size, -1).to(device)  # (batch_size, n_channels * window_length)
            labels = labels.to(device)
            # Ensure single_waveform has correct shape: (batch_size, window_length)
            single_waveform = single_waveform.to(device)
            
            # Debug: Check shapes match
            if batch_features.shape[0] != single_waveform.shape[0]:
                raise ValueError(f"Batch size mismatch: batch_features={batch_features.shape[0]}, single_waveform={single_waveform.shape[0]}")
            
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
        
        # Validation phase
        valid_detection_loss = 0.0
        valid_classification_loss = 0.0
        
        gt_all = []
        pred_all = []
        gt_class_all = []
        pred_class_all = []
        autosort_model.eval()
        
        with torch.no_grad():
            for batch_features, classify_labels, labels, single_waveform in tqdm(val_loader, desc="Validation"):
                classify_labels = classify_labels.to(device)
                # batch_features shape: (batch_size, n_channels, window_length)
                # Reshape to (batch_size, n_channels * window_length) while preserving batch_size
                batch_size = batch_features.shape[0]
                batch_features = batch_features.reshape(batch_size, -1).to(device)  # (batch_size, n_channels * window_length)
                labels = labels.to(device)
                # Ensure single_waveform has correct shape: (batch_size, window_length)
                single_waveform = single_waveform.to(device)
                
                # Debug: Check shapes match
                if batch_features.shape[0] != single_waveform.shape[0]:
                    raise ValueError(f"Batch size mismatch: batch_features.shape[0]={batch_features.shape[0]}, single_waveform.shape[0]={single_waveform.shape[0]}")
                
                valid_detection_loss_batch, valid_classification_loss_batch, gt, pred, gt_label_class, pred_class = autosort_model.iter_model_eval(
                    batch_features, classify_labels, labels, single_waveform
                )
                
                valid_detection_loss += valid_detection_loss_batch
                valid_classification_loss += valid_classification_loss_batch
                
                gt_all.append(gt.detach().cpu().numpy())
                pred_all.append(pred.detach().cpu().numpy())
                
                # Handle classification results (may be empty tensors)
                if len(gt_label_class) > 0:
                    gt_class_all.append(gt_label_class.detach().cpu().numpy())
                if len(pred_class) > 0:
                    pred_class_all.append(pred_class.detach().cpu().numpy())
        
        gt_all = np.concatenate(gt_all, axis=0)
        pred_all = np.concatenate(pred_all, axis=0)
        
        # Filter empty arrays and concatenate
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
            training_log['validation_acc_label'].append(0.0)
        
        # Calculate overall accuracy (noise classification accuracy)
        current_val_acc = val_acc_noise
        
        # Save model with minimum loss
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
            min_valid_loss = valid_loss
        
        # Save model with minimum accuracy (for early stopping)
        if current_val_acc < min_val_acc - min_delta:
            print(f'Validation Accuracy Decreased({min_val_acc:.6f}--->{current_val_acc:.6f}) \t Saving The Model (Best Acc Epoch)')
            min_val_acc = current_val_acc
            best_acc_epoch = epoch + 1
            autosort_model.save_model()
            patience_counter = 0  # Reset counter
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= patience:
                print(f'\nEarly stopping triggered: {patience} consecutive epochs without improvement')
                print(f'Best accuracy: {min_val_acc:.6f} (Epoch {best_acc_epoch})')
                break
    
    # Save final model (last epoch) for evaluation consistency
    print(f'\nSaving final model (epoch {epochs}) for evaluation...')
    autosort_model.save_model()
    
    # Save training log
    import os
    training_log_path = os.path.join(model_save_dir, 'training_log.csv')
    pd.DataFrame(training_log).to_csv(training_log_path)
    print(f"\nTraining completed! Training log saved to: {training_log_path}")
    if early_stopping:
        print(f"Best validation accuracy: {min_val_acc:.6f} (Epoch {best_acc_epoch})")
    
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


# ==================== 6. Evaluation Function ====================

def evaluate_autosort_model(
    train_data_dir,
    model_save_dir,
    n_channels,
    left_sample=20,  # Changed: now 20 samples (no downsampling)
    right_sample=40,  # Changed: now 40 samples (no downsampling)
    batch_size=512,
    device=None,
    save_results=True,
    results_save_dir=None,
    eval_neuron_inf_matched=None,
    eval_data_dir=None,
):
    """
    Evaluate AutoSort model
    
    Parameters:
        train_data_dir: training data directory（for evaluation）
        model_save_dir: model save directory
        n_channels: number of channels
        left_sample, right_sample: window parameters
        batch_size: batch size
        device: device (if None, auto-select)
        save_results: whether to save results
        results_save_dir: results save directory (if None, use model_save_dir)
        eval_neuron_inf_matched: evaluation data neuron_inf (contains neuron_match column), if provided compute two sets of results
        eval_data_dir: evaluation data directory (if eval_neuron_inf_matched is provided, this parameter is required)
    
    Returns:
        results: evaluation results dictionary, containing:
            - noise_accuracy: Noise classification accuracy (original)
            - unit_f1_score: Unit classification F1 score (original)
            - noise_accuracy_adjusted: Noise classification accuracy (adjusted, unmatch treated as noise)
            - unit_f1_score_adjusted: Unit classification F1 score (adjusted)
            - noise_predictions: Noise prediction results
            - unit_predictions: Unit prediction results
            - gt_noise: Ground truth noise labels (original)
            - gt_units: Ground truth unit labels (original)
            - gt_noise_adjusted: Ground truth noise labels (adjusted)
            - gt_units_adjusted: Ground truth unit labels (adjusted)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set parameters
    samplepoints = left_sample + right_sample
    
    # Load unit ID list from training (must use training unit IDs to ensure model dimension matching)
    import os
    keep_id_path = os.path.join(model_save_dir, 'keep_id.pkl')
    if os.path.exists(keep_id_path):
        print(f"Loading unit ID list from training file: {keep_id_path}")
        with open(keep_id_path, 'rb') as f:
            train_keep_id = pickle.load(f)
        print(f"Number of units during training: {len(train_keep_id)}")
    else:
        raise FileNotFoundError(
            f"Unit ID list file does not exist: {keep_id_path}\n"
            "Please ensure training pipeline has been run, or manually create this file."
        )
    
    # Create dataset (using training unit ID list)
    print("Create dataset...")
    full_dataset = SimpleWaveformLoader(
        root=str(train_data_dir) + '/',
        shank_channel=np.arange(n_channels),
        Keep_id=train_keep_id  # Use training unit ID list
    )
    
    set_shank_id = full_dataset.keep_id
    print(f"Model parameters:")
    print(f"  - Number of channels: {n_channels}")
    print(f"  - Window length: {samplepoints}")
    print(f"  - Number of units: {len(set_shank_id)}")
    print(f"  - Using training unit ID list: {set_shank_id == train_keep_id}")
    
    # Load train/val split indices to use the same validation set as training
    train_val_split_path = Path(model_save_dir) / "train_val_split_indices.pkl"
    if train_val_split_path.exists():
        print(f"Loading train/val split indices from: {train_val_split_path}")
        with open(train_val_split_path, "rb") as f:
            train_indices, val_indices = pickle.load(f)
        print(f"  - Training set size: {len(train_indices)}")
        print(f"  - Validation set size: {len(val_indices)}")
        # Use validation set for evaluation (same as training log)
        dataset = torch.utils.data.Subset(full_dataset, val_indices)
        print("  - Using validation set for evaluation (consistent with training log)")
    else:
        print(f"Warning: Train/val split indices not found at {train_val_split_path}")
        print("  - Using full dataset for evaluation (may differ from training log)")
        dataset = full_dataset
    
    # Create model
    # Use full_dataset for pos_weight (Subset doesn't have these attributes)
    autosort_model = SimpleAutoSort(
        ch_num=n_channels,
        samplepoints=samplepoints,
        device=device,
        set_shank_id=set_shank_id,
        save_dir=model_save_dir,
        pos_weight_noise=full_dataset.pos_weight_noise.to(device),
        pos_weight_label=full_dataset.pos_weight_label.to(device)
    )
    
    # Load model weights
    import os
    if not os.path.exists(autosort_model.save_model_path_2):
        raise FileNotFoundError(f"Model file does not exist: {autosort_model.save_model_path_2}")
    if not os.path.exists(autosort_model.save_model_path_3):
        raise FileNotFoundError(f"Model file does not exist: {autosort_model.save_model_path_3}")
    
    print("Loading model weights...")
    autosort_model.load_model()
    autosort_model.eval()
    print("Model loaded successfully")
    
    # Create data loader (using all data or test set)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"\nStarting evaluation (total {len(dataset)} samples)...")
    
    # Evaluation
    all_gt_noise = []
    all_pred_noise = []
    all_gt_units = []
    all_pred_units = []
    all_noise_probs = []
    all_unit_probs = []
    
    # Extract way4 features for UMAP visualization (30 dimensions, no PCA needed)
    all_way4_features_30d = []  # Way4 features for all detected spikes (30 dimensions)
    all_way4_features_30d_spike = []  # Way4 features for spike samples only (30 dimensions)
    
    with torch.no_grad():
        for batch_idx, (batch_features, classify_labels, labels, single_waveform) in enumerate(tqdm(test_loader, desc="Evaluating")):
            classify_labels = classify_labels.to(device)
            # batch_features shape: (batch_size, n_channels, window_length)
            # Reshape to (batch_size, n_channels * window_length) while preserving batch_size
            batch_size = batch_features.shape[0]
            batch_features = batch_features.reshape(batch_size, -1).to(device)  # (batch_size, n_channels * window_length)
            labels = labels.to(device)
            # Ensure single_waveform has correct shape: (batch_size, window_length)
            single_waveform = single_waveform.to(device)
            
            # Debug: Check shapes match before concatenation
            if batch_features.shape[0] != single_waveform.shape[0]:
                raise ValueError(f"Batch size mismatch at batch {batch_idx}: batch_features.shape[0]={batch_features.shape[0]}, single_waveform.shape[0]={single_waveform.shape[0]}")
            
            # Forward propagation
            codes = torch.cat((batch_features, single_waveform), axis=1)
            
            # Extract way4 features for all samples (for noise detection visualization)
            # intermediate_forward now returns way4 features (30 dimensions)
            way4_features_all = autosort_model.clsfier_label.intermediate_forward(codes.float())
            all_way4_features_30d.append(way4_features_all.detach().cpu().numpy())
            
            # Noise classification
            cls_output = autosort_model.clsfier_noise(codes.float())
            noise_probs = torch.softmax(cls_output, dim=1)
            pred_noise = torch.argmax(cls_output, axis=1)
            gt_noise = torch.argmax(labels, axis=1)
            
            all_gt_noise.append(gt_noise.detach().cpu().numpy())
            all_pred_noise.append(pred_noise.detach().cpu().numpy())
            all_noise_probs.append(noise_probs.detach().cpu().numpy())
            
            # Unit classification (only for non-noise samples)
            test = labels[:, 1] == 1
            if sum(test) > 0:
                codes_spike = codes[test, :]
                cls_label_output = autosort_model.clsfier_label(codes_spike.float())
                unit_probs = torch.softmax(cls_label_output, dim=1)
                pred_units = torch.argmax(cls_label_output, axis=1)
                gt_units = torch.argmax(classify_labels[test, :len(set_shank_id)], axis=1)
                
                all_gt_units.append(gt_units.detach().cpu().numpy())
                all_pred_units.append(pred_units.detach().cpu().numpy())
                all_unit_probs.append(unit_probs.detach().cpu().numpy())
                
                # Extract way4 features for spikes passing noise classifier (for label classification visualization)
                way4_features_spike = way4_features_all[test, :]
                all_way4_features_30d_spike.append(way4_features_spike.detach().cpu().numpy())
    
    # Combine results
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
    
    # Calculate metrics (consistent with training log)
    noise_accuracy = accuracy_score(all_gt_noise, all_pred_noise)
    
    if len(all_gt_units) > 0:
        # Use F1 score (micro) for unit classification, same as training log
        unit_f1_score = f1_score(all_gt_units, all_pred_units, average='micro')
        unit_accuracy = accuracy_score(all_gt_units, all_pred_units)
    else:
        unit_f1_score = 0.0
        unit_accuracy = 0.0
    
    print(f"\nEvaluation results (on validation set, consistent with training log):")
    print(f"  - Noise classification accuracy: {noise_accuracy:.4f}")
    if len(all_gt_units) > 0:
        print(f"  - Unit classification accuracy: {unit_accuracy:.4f}")
        print(f"  - Unit classification F1 score (micro): {unit_f1_score:.4f}")
        print(f"  - Number of unit samples evaluated: {len(all_gt_units)}")
    print(f"  - Total samples: {len(all_gt_noise)}")
    
    # Combine way4 features (already 30 dimensions, no PCA needed)
    if len(all_way4_features_30d) > 0:
        all_way4_features_30d_combined = np.concatenate(all_way4_features_30d, axis=0)
    else:
        all_way4_features_30d_combined = np.array([])
    
    if len(all_way4_features_30d_spike) > 0:
        all_way4_features_30d_spike_combined = np.concatenate(all_way4_features_30d_spike, axis=0)
    else:
        all_way4_features_30d_spike_combined = np.array([])
    
    # Initialize results dictionary
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
        'way4_features_30d': all_way4_features_30d_combined,  # For noise detection visualization (30 dimensions)
        'way4_features_30d_spike': all_way4_features_30d_spike_combined,  # For label classification visualization (30 dimensions)
    }
    
    # If eval_neuron_inf_matched is provided, compute adjusted results (treat unmatch neuron as noise)
    if eval_neuron_inf_matched is not None and eval_data_dir is not None:
        print(f"\nComputing adjusted evaluation results (treating unmatch neuron as noise)...")
        
        # Load evaluation data neuron mapping
        eval_neuron_mapping_path = Path(eval_data_dir) / "neuron_mapping.pkl"
        if not eval_neuron_mapping_path.exists():
            print(f"  Warning: evaluation data neuron_mapping.pkl does not exist, skipping adjusted results computation")
        else:
            with open(eval_neuron_mapping_path, "rb") as f:
                eval_neuron_mapping = pickle.load(f)
            
            # Load evaluation data Y_spike_id
            with open(eval_data_dir / "Y_spike_id.pkl", "rb") as f:
                eval_Y_spike_id_full = pickle.load(f)
            
            # Find all unmatch neuron names
            unmatch_neurons = eval_neuron_inf_matched[
                eval_neuron_inf_matched['neuron_match'] == 'unmatch'
            ]['Neuron'].values
            
            print(f"  - Number of unmatched neurons: {len(unmatch_neurons)}")
            if len(unmatch_neurons) > 0:
                print(f"  - Unmatched neurons: {unmatch_neurons}")
            
            # Create mapping from unmatch neuron names to IDs
            eval_neuron_to_id = eval_neuron_mapping['neuron_to_id']
            unmatch_neuron_ids = [eval_neuron_to_id.get(neuron, -1) for neuron in unmatch_neurons]
            unmatch_neuron_ids = [nid for nid in unmatch_neuron_ids if nid != -1]
            
            # Create adjusted GT labels
            all_gt_noise_adjusted = all_gt_noise.copy()
            
            # Find all samples belonging to unmatch neurons
            unmatch_sample_mask = np.isin(eval_Y_spike_id_full, unmatch_neuron_ids)
            
            # Mark spike samples belonging to unmatch neurons as noise
            # Only adjust samples that were originally spikes (gt_noise == 1)
            spike_mask = all_gt_noise == 1
            unmatch_spike_mask = unmatch_sample_mask & spike_mask
            
            all_gt_noise_adjusted[unmatch_spike_mask] = 0  # Marked as noise
            adjusted_count = np.sum(unmatch_spike_mask)
            
            print(f"  - Number of adjusted samples: {adjusted_count}")
            
            # Adjust unit labels: keep all original unit samples (including unmatched neuron samples)
            # Find indices of non-noise samples (original)
            non_noise_mask_original = all_gt_noise == 1
            non_noise_indices_original = np.where(non_noise_mask_original)[0]
            
            # Find indices of samples that are still non-noise after adjustment (excluding unmatched neurons)
            non_noise_mask_adjusted = all_gt_noise_adjusted == 1
            non_noise_indices_adjusted = np.where(non_noise_mask_adjusted)[0]
            
            # Keep all original unit samples (85483), including unmatched neuron samples
            # For unmatched neuron samples:
            # - Their GT labels in noise classification have been changed to noise (0)
            # - But in unit classification evaluation, we need to include them
            # - If network classifies unmatched neuron samples as noise (all_pred_noise == 0), this is correct
            # - If network classifies unmatched neuron samples as a unit (all_pred_noise == 1), this is wrong
            if len(all_gt_units) > 0:
                # Keep all original unit samples (including unmatched neurons)
                all_gt_units_adjusted = all_gt_units.copy()  # Keep all original GT unit labels (85483)
                
                # Find matched neuron samples (still non-noise after adjustment)
                matched_unit_mask = np.isin(non_noise_indices_original, non_noise_indices_adjusted)
                
                # Find unmatched neuron samples (in original unit samples, but marked as noise after adjustment)
                unmatched_unit_mask = ~matched_unit_mask
                unmatched_unit_indices = non_noise_indices_original[unmatched_unit_mask]
                
                # Statistics on misclassification of unmatched neuron samples
                unmatch_spike_pred_noise = all_pred_noise[unmatched_unit_indices]
                unmatch_correct_as_noise = np.sum(unmatch_spike_pred_noise == 0)  # Network correctly identified as noise
                unmatch_misclassified_as_unit = np.sum(unmatch_spike_pred_noise == 1)  # Network misclassified as unit
                
                # Calculate unit classification accuracy for matched neuron samples
                matched_gt_units = all_gt_units[matched_unit_mask]
                matched_pred_units = all_pred_units[matched_unit_mask]
                
                if len(matched_gt_units) > 0:
                    matched_unit_accuracy = accuracy_score(matched_gt_units, matched_pred_units)
                    matched_unit_f1 = f1_score(matched_gt_units, matched_pred_units, average='micro')
                else:
                    matched_unit_accuracy = 0.0
                    matched_unit_f1 = 0.0
                
                # Calculate overall unit classification accuracy (including unmatched neuron samples)
                # For matched neuron samples: use unit classification accuracy
                # For unmatched neuron samples: correct if network classifies as noise, wrong if classifies as unit
                total_unit_samples = len(all_gt_units)  # 85483
                matched_correct = np.sum(matched_gt_units == matched_pred_units) if len(matched_gt_units) > 0 else 0
                unmatched_correct = unmatch_correct_as_noise
                total_correct = matched_correct + unmatched_correct
                unit_accuracy_adjusted = total_correct / total_unit_samples if total_unit_samples > 0 else 0.0
                
                # For F1 score, only calculate matched neuron samples (unmatched neuron samples have no unit labels)
                unit_f1_score_adjusted = matched_unit_f1
                
                # Save adjusted predictions (for subsequent analysis)
                all_pred_units_adjusted = all_pred_units.copy()
            else:
                all_gt_units_adjusted = np.array([])
                all_pred_units_adjusted = np.array([])
                unit_accuracy_adjusted = 0.0
                unit_f1_score_adjusted = 0.0
                unmatch_correct_as_noise = 0
                unmatch_misclassified_as_unit = 0
                matched_unit_accuracy = 0.0
            
            # Recalculate adjusted noise classification accuracy
            noise_accuracy_adjusted = accuracy_score(all_gt_noise_adjusted, all_pred_noise)
            
            print(f"\nEvaluation results (adjusted):")
            print(f"  - Noise classification accuracy: {noise_accuracy_adjusted:.4f}")
            print(f"  - Total samples: {len(all_gt_noise_adjusted)}")  # Including all samples
            if len(all_gt_units_adjusted) > 0:
                print(f"  - Unit classification accuracy: {unit_accuracy_adjusted:.4f}")
                print(f"  - Unit classification F1 score: {unit_f1_score_adjusted:.4f}")
                print(f"  - Number of unit samples evaluated: {len(all_gt_units_adjusted)}")  # Should equal original 85483
                if adjusted_count > 0:
                    print(f"    - Matched neuron samples: {len(matched_gt_units) if len(all_gt_units_adjusted) > 0 else 0}")
                    print(f"    - Unmatched neuron samples: {adjusted_count}")
                    print(f"      - Correctly identified as noise: {unmatch_correct_as_noise} ({unmatch_correct_as_noise/adjusted_count*100:.1f}%)")
                    print(f"      - Misclassified as unit: {unmatch_misclassified_as_unit} ({unmatch_misclassified_as_unit/adjusted_count*100:.1f}%)")
                    print(f"    - Note: unmatched neuron samples are treated as noise, correctly identified as noise counts as correct, misclassified as unit counts as error")
            
            # Add to results dictionary
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
        
        # Save prediction results
        results_df = pd.DataFrame({
            'gt_noise': all_gt_noise,
            'pred_noise': all_pred_noise,
        })
        if len(all_gt_units) > 0:
            # Create complete unit label array (noise samples marked as -1)
            full_gt_units = np.full(len(all_gt_noise), -1, dtype=np.int64)
            full_pred_units = np.full(len(all_gt_noise), -1, dtype=np.int64)
            
            # Find indices of non-noise samples
            non_noise_mask = all_gt_noise == 1
            non_noise_indices = np.where(non_noise_mask)[0]
            
            if len(non_noise_indices) == len(all_gt_units):
                full_gt_units[non_noise_indices] = all_gt_units
                full_pred_units[non_noise_indices] = all_pred_units
            else:
                # If counts don't match, only fill first N
                n_units = min(len(non_noise_indices), len(all_gt_units))
                full_gt_units[non_noise_indices[:n_units]] = all_gt_units[:n_units]
                full_pred_units[non_noise_indices[:n_units]] = all_pred_units[:n_units]
            
            results_df['gt_units'] = full_gt_units
            results_df['pred_units'] = full_pred_units
        
        import os
        results_path = os.path.join(results_save_dir, 'evaluation_results.csv')
        results_df.to_csv(results_path)
        print(f"\nEvaluation results saved to: {results_path}")
        
        # Save metrics summary
        summary_data = {
            'metric': ['noise_accuracy', 'unit_accuracy', 'unit_f1_score'],
            'value': [noise_accuracy, unit_accuracy, unit_f1_score]
        }
        
        # If adjusted results exist, also add to summary
        if 'noise_accuracy_adjusted' in results:
            summary_data['metric'].extend(['noise_accuracy_adjusted', 'unit_accuracy_adjusted', 'unit_f1_score_adjusted'])
            summary_data['value'].extend([
                results['noise_accuracy_adjusted'],
                results['unit_accuracy_adjusted'],
                results['unit_f1_score_adjusted']
            ])
        
        summary = pd.DataFrame(summary_data)
        import os
        summary_path = os.path.join(results_save_dir, 'evaluation_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"Evaluation summary saved to: {summary_path}")
    
    return results


# ==================== 6. Optimized Classification Pipeline (Two-Stage) ====================

def compute_cluster_position_waveform(
    snippets: np.ndarray,
    channel_id: list,
    channel_positions: dict,  # {channel_id: (x, y)} - required, must be provided from probe
    window_size: int = 30,
) -> tuple:
    """
    Compute cluster position and position_waveform from snippets (reference: generate_neuron_inf_phy_template.py)
    
    Parameters:
        snippets: numpy array, shape (n_spikes, n_channels, window_size)
        channel_id: list of channel IDs (probe channel indices)
        window_size: window size, default 30
        channel_positions: dict, {channel_id: (x, y)} - required, must be provided from probe
    
    Returns:
        position_1, position_2, position_waveform (30-dim)
    """
    # channel_positions is required and must be provided from probe
    if channel_positions is None:
        raise ValueError("channel_positions must be provided (obtained from probe)")
    
    cluster_positions_x = []
    cluster_positions_y = []
    cluster_waveforms = []
    
    for snippet in snippets:  # snippet: (n_channels, window_size)
        # Calculate position of this spike (based on channel_id channels)
        a_squared = [np.sum(snippet[j, :]**2) for j in range(len(channel_id))]
        
        sum_x_a = 0
        sum_y_a = 0
        sum_a = 0
        
        for j, ch in enumerate(channel_id):
            x_i, y_i = channel_positions.get(ch, (0.0, 0.0))
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
        for ch in channel_id:
            x_channel, y_channel = channel_positions.get(ch, (np.nan, np.nan))
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
    probe,  # Probe object (required, obtained from read_probeinterface)
    calibration_duration_seconds: int = 60,
    n_additional_clusters: int = 5,
    detect_threshold: float = 3.0,  # Mountainsort4: absolute threshold value
    detect_interval: int = 10,  # Mountainsort4: minimum interval between detections (samples)
    detect_sign: int = -1,  # Mountainsort4: -1 (negative peaks), 0 (both), 1 (positive peaks)
    margin: int = 0,  # Mountainsort4: margin to exclude from boundaries (samples)
    window_params: dict = None,
    position_threshold: float = 10.0,
    waveform_similarity_threshold: float = 0.9,
    eval_neuron_inf: pd.DataFrame = None,
    eval_spike_inf: pd.DataFrame = None,
    valid_channels=None,  # List of valid channel indices (clique channel indices) to detect on, if None detect on all channels
    device=None,
    save_eval_results: bool = False,  # Whether to save evaluation results to files
    eval_save_dir: str = None,  # Directory to save evaluation results (e.g., "output/clique_XX/model_save/run_X/eval")
    run_name: str = "run_1",  # Run name for file naming
    date_str: str = None,  # Date string for file naming (e.g., "022522"), if None will use current date
    train_data_dir: str = None,  # Training data directory path (not used for loading whitening matrix, kept for compatibility)
    skip_noise_classifier: bool = False,  # 若为 True，则跳过噪声分类，直接送入 label classifier
):
    """
    Stage 1: Calibration stage (first 60s) using Mountainsort4 detection method
    
    Process:
    1. Load whitening matrix from training data (not recompute)
    2. Apply whitening to calibration data
    3. Threshold detection (Mountainsort4, on whitened data)
    4. Extract waveforms (from whitened data)
    5. Pass through noise classifier, classified as spikes
    6. Extract way4 layer features (30 dimensions, no PCA needed)
    7. K-means clustering directly on way4 features (number of classes = train neurons + n)
    8. Calculate position and waveform for each cluster (inverse whiten waveforms before matching)
    9. Match with train neurons, establish mapping relationship
    
    Parameters:
        recording_f: preprocessed recording object
        autosort_model: trained SimpleAutoSort model
        train_neuron_inf: training data neuron_inf DataFrame
        probe: Probe object (required, obtained from read_probeinterface)
        calibration_duration_seconds: calibration duration (seconds)，default 60
        n_additional_clusters: number of additional clusters (n)，default 5
        detect_threshold: float, Mountainsort4 absolute threshold value (default 3.0)
        detect_interval: int, Mountainsort4 minimum interval between detections (samples, default 10)
        detect_sign: int, Mountainsort4 detection sign: -1 (negative peaks), 0 (both), 1 (positive peaks), default -1
        margin: int, Mountainsort4 margin to exclude from boundaries (samples, default 0)
        window_params: window parameters dictionary
        position_threshold: position distance threshold (microns)，default 10
        waveform_similarity_threshold: waveform similarity threshold，default 0.9
        valid_channels: list of int, valid channel indices (clique channel indices) to detect on, if None detect on all channels, default None
        device: device
    
    Returns:
        calibration_results: dictionary, containing:
            - kmeans_model: trained K-means model (trained on way4 features, 30 dimensions)
            - pca_model: None (no PCA needed, way4 is already 30 dimensions)
            - cluster_to_neuron_mapping: mapping from cluster to train neuron
            - cluster_features: features for each cluster (position, waveform, etc., computed from inverse-whitened waveforms)
            - whitening_matrix_W: whitening matrix loaded from training data
            - whitening_matrix_M: mean vector (or None) for whitening
    """
    from sklearn.cluster import KMeans
    from scipy.stats import pearsonr
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    
    # Window size (60 samples, no downsampling)
    left_sample_extract = window_params['left_sample']  # 20
    right_sample_extract = window_params['right_sample']  # 40
    window_size = left_sample_extract + right_sample_extract  # 60
    window_size_extract = window_size  # Same as window_size (no downsampling)
    
    n_channels = recording_f.get_num_channels()
    sampling_frequency = recording_f.get_sampling_frequency()
    
    # Build mapping from probe device channel indices to clique channel indices
    # recording_f is already a clique recording with renamed channels (device_channel_indices)
    recording_channel_ids = recording_f.get_channel_ids()  # These are the device_channel_indices (probe channels)
    probe_to_clique_index = {}
    clique_to_probe_index = {}  # Reverse mapping: clique index -> probe channel index
    for clique_idx, probe_ch in enumerate(recording_channel_ids):
        probe_to_clique_index[int(probe_ch)] = clique_idx
        clique_to_probe_index[clique_idx] = int(probe_ch)
    
    # Get channel positions from probe (dynamic, from actual probe geometry)
    # This ensures all channels have correct positions, matching generate_neuron_inf_phy_template.py
    # probe is passed as parameter (not from recording_f)
    probe_df = probe.to_dataframe()
    channel_positions = {}  # {probe_channel_id: (x, y)}
    for channel_id in probe_df.index:
        try:
            x = probe_df.loc[channel_id, 'x']
            y = probe_df.loc[channel_id, 'y']
            channel_positions[int(channel_id)] = (float(x), float(y))
        except (KeyError, ValueError, TypeError):
            # If channel not found or invalid, use (0, 0) as fallback
            channel_positions[int(channel_id)] = (0.0, 0.0)

    # Build neuron -> channel_id(set of clique indices) mapping (same logic as training)
    def _build_neuron_to_channel_id(df_neuron):
        mapping = {}
        if df_neuron is None or len(df_neuron) == 0:
            return mapping
        import ast
        for _, row in df_neuron.iterrows():
            neuron_name = row.get('Neuron', None)
            channel_id_list = row.get('channel_id', [])
            if isinstance(channel_id_list, str):
                try:
                    channel_id_list = ast.literal_eval(channel_id_list)
                except Exception:
                    channel_id_list = []
            if not isinstance(channel_id_list, (list, tuple, np.ndarray)):
                channel_id_list = []
            clique_indices = []
            for probe_ch in channel_id_list:
                try:
                    probe_ch_int = int(probe_ch)
                except Exception:
                    continue
                if probe_ch_int in probe_to_clique_index:
                    clique_indices.append(probe_to_clique_index[probe_ch_int])
            if len(clique_indices) == 0 or neuron_name is None:
                continue
            mapping[neuron_name] = set(clique_indices)
        return mapping

    neuron_to_channel_id_eval = _build_neuron_to_channel_id(eval_neuron_inf)
    
    print("=" * 50)
    print("Stage 1: Calibration")
    print("=" * 50)
    
    # 1. Load first 60s of data
    max_duration_samples = int(calibration_duration_seconds * sampling_frequency)
    total_samples = recording_f.get_num_samples()
    actual_samples = min(max_duration_samples, total_samples)
    print(f"Loading first {calibration_duration_seconds} seconds of data...")
    # 1. Load first 60s of data for calibration
    recording_calibration = recording_f.frame_slice(start_frame=0, end_frame=actual_samples)
    
    import os
    from pathlib import Path
    whitening_matrices_dir = Path(train_data_dir).parent / "whitening_matrices"
    W_save_path = whitening_matrices_dir / "whitening_matrix_W.npy"
    M_save_path = whitening_matrices_dir / "whitening_matrix_M.npy"
    
    W = np.load(W_save_path)
    if M_save_path.exists():
        M = np.load(M_save_path, allow_pickle=True)
        if isinstance(M, np.ndarray) and M.size == 1 and M.item() is None:
            M = None
    else:
        M = None

    recording_calibration_whitened = spre.whiten(recording_calibration, W=W, M=M, dtype="float32")
    
    # Get whitened traces
    traces_whitened = recording_calibration_whitened.get_traces()
    if traces_whitened.shape[0] > traces_whitened.shape[1] and traces_whitened.shape[0] > 100:
        traces_whitened = traces_whitened.T
    traces_whitened = traces_whitened.astype(np.float32)
    
    # Ensure traces_whitened is in (n_channels, n_timepoints) format for consistency with detection
    # get_traces returns (n_samples, n_channels) by default, so transpose if needed
    if traces_whitened.shape[0] > traces_whitened.shape[1]:
        traces_whitened = traces_whitened.T  # Transpose to (n_channels, n_timepoints)
    
    # Also get original traces for waveform extraction (not whitened)
    traces_original = recording_calibration.get_traces()
    if traces_original.shape[0] > traces_original.shape[1] and traces_original.shape[0] > 100:
        traces_original = traces_original.T
    traces_original = traces_original.astype(np.float32)
    
    # 2. Threshold detection using Mountainsort4 (on whitened data)
    print("\n### 2. Threshold detection (Mountainsort4)")
    print(f"  阈值: {detect_threshold}, 间隔: {detect_interval}, 检测方向: {detect_sign}, 边界margin: {margin}")
    if valid_channels is not None:
        print(f"共 {len(valid_channels)} 个通道")
    else:
        print(f"  在所有通道上进行检测 (共 {n_channels} 个通道)")
    trace0_car_detect = traces_whitened.T  # (n_timepoints, n_channels) - whitened data for detection
    spikes = detect_spike(
        trace0_car_detect,
        detect_threshold=detect_threshold,
        detect_interval=detect_interval,
        detect_sign=detect_sign,
        margin=margin,
        valid_channels=valid_channels,
    )
    spike_coords = np.argwhere(spikes == 1)  # (n_spikes, 2) [time, channel]
    print(f"Number of detected spikes: {len(spike_coords)}")
    
    # 2.5. Compare detected spikes with GT (if available) — align with training (time<=1 & channel in neuron's channel_id)
    if eval_spike_inf is not None:
        detected_spike_times = spike_coords[:, 0].astype(int)
        detected_spike_channels = spike_coords[:, 1].astype(int)
        detect_array_init = np.stack([detected_spike_times, detected_spike_channels], axis=1)

        calibration_end_frame = actual_samples
        gt_spikes_in_calibration = eval_spike_inf[
            (eval_spike_inf['time'] >= 0) & (eval_spike_inf['time'] < calibration_end_frame)
        ].copy()

        # Filter GT spikes to clique neurons (use eval_neuron_inf already filtered to this clique)
        if eval_neuron_inf is not None and len(gt_spikes_in_calibration) > 0:
            clique_neuron_names = set(eval_neuron_inf['Neuron'].unique())
            if 'neuron' in gt_spikes_in_calibration.columns:
                gt_spikes_in_calibration = gt_spikes_in_calibration[
                    gt_spikes_in_calibration['neuron'].isin(clique_neuron_names)
                ].copy()
            elif 'cluster' in gt_spikes_in_calibration.columns:
                gt_spikes_in_calibration = gt_spikes_in_calibration[
                    gt_spikes_in_calibration['cluster'].isin(clique_neuron_names)
                ].copy()

        if len(gt_spikes_in_calibration) > 0:
            gt_spike_times = gt_spikes_in_calibration['time'].values.astype(int)
            gt_spike_neurons = gt_spikes_in_calibration['neuron'].values if 'neuron' in gt_spikes_in_calibration.columns else gt_spikes_in_calibration['cluster'].values
            gt_array_init = np.stack([gt_spike_times, gt_spike_neurons], axis=1)

            gt_label_array = map_gt_annotation(
                detect_array_init,
                gt_array_init,
                neuron_to_channel_id=neuron_to_channel_id_eval,
                time_tolerance=1
            )
            matched_mask = gt_label_array >= 0
            n_matched = int(matched_mask.sum())
            n_unmatched = int(len(detect_array_init) - n_matched)
            spike_detection_rate = n_matched / len(gt_array_init) if len(gt_array_init) > 0 else 0.0

            print(f"\n---spike detection rate: {spike_detection_rate:.4f}")
            print(f"\nNumber of matched spikes: {n_matched}")
            print(f"Number of unmatched spikes: {n_unmatched}")

            detection_stats = {
                'spike_detection_rate': spike_detection_rate,
                'n_matched': n_matched,
                'n_unmatched': n_unmatched,
            }
        else:
            detection_stats = None
    else:
        detection_stats = None
    
    # 3. Extract waveforms and filter boundaries (using original traces, no downsampling)
    print("\n### 3. Extract waveforms (using original traces)")
    valid_spikes = []
    waveforms = []
    spike_times = []
    spike_channels = []
    
    # Use original traces for waveform extraction (keep same shape as training)
    # trace0_car is (n_timepoints, n_channels) original (non-whitened) data
    
    # Extract waveforms using the same method as training (extract_waveforms) on original traces
    trace0_original = traces_original.T  # (n_timepoints, n_channels) original data for waveform
    spike_time_indices = np.array([time_idx for time_idx, _ in spike_coords])
    spike_channel_indices = np.array([channel_idx for _, channel_idx in spike_coords])
    
    # Filter spikes near boundaries (ensure complete window can be extracted)
    valid_mask = (spike_time_indices >= left_sample_extract) & (spike_time_indices < trace0_original.shape[0] - right_sample_extract)
    spike_time_indices = spike_time_indices[valid_mask]
    spike_channel_indices = spike_channel_indices[valid_mask]
    
    if len(spike_time_indices) == 0:
        raise ValueError("No valid spikes after boundary filtering")
    
    waveform = np.zeros((len(spike_time_indices), trace0_original.shape[1], window_size_extract), dtype=np.float32)
    
    for i, time_range in enumerate(np.arange(-left_sample_extract, right_sample_extract)):
        waveform[:, :, i] = trace0_original[spike_time_indices + time_range, :]
    
    # Store waveforms and metadata
    for idx, (time_idx, channel_idx) in enumerate(zip(spike_time_indices, spike_channel_indices)):
        waveforms.append(waveform[idx])  # (n_channels, window_size_extract=60)
        valid_spikes.append((time_idx, channel_idx))
        spike_times.append(time_idx)
        spike_channels.append(channel_idx)
    
    waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size_extract=60) - whitened waveforms, no downsampling
    print(f"Number of valid spikes: {len(waveforms)}")
    
    if len(waveforms) == 0:
        raise ValueError("No valid spikes for calibration")
    
    # 2.5. Compare detected spikes (after waveform extraction) with GT (align with training)
    if eval_spike_inf is not None:
        detected_spike_times_after_extraction = np.array(spike_times).astype(int)
        detected_spike_channels_after_extraction = np.array(spike_channels).astype(int)
        detect_array_after_ext = np.stack([detected_spike_times_after_extraction, detected_spike_channels_after_extraction], axis=1)

        calibration_end_frame = actual_samples
        gt_spikes_in_calibration = eval_spike_inf[
            (eval_spike_inf['time'] >= 0) & (eval_spike_inf['time'] < calibration_end_frame)
        ].copy()

        print(f"  GT spikes after time filtering: {len(gt_spikes_in_calibration)}")

        if eval_neuron_inf is not None and len(gt_spikes_in_calibration) > 0:
            clique_neuron_names = set(eval_neuron_inf['Neuron'].unique())
            print(f"  Clique neuron names (eval set, full clique): {sorted(clique_neuron_names)}")
            print(f"  GT spike columns: {gt_spikes_in_calibration.columns.tolist()}")

            if 'neuron' in gt_spikes_in_calibration.columns:
                gt_spikes_in_calibration = gt_spikes_in_calibration[
                    gt_spikes_in_calibration['neuron'].isin(clique_neuron_names)
                ].copy()
                print(f"  GT spikes after neuron filtering (eval neurons, no train filtering): {len(gt_spikes_in_calibration)}")
            elif 'cluster' in gt_spikes_in_calibration.columns:
                gt_spikes_in_calibration = gt_spikes_in_calibration[
                    gt_spikes_in_calibration['cluster'].isin(clique_neuron_names)
                ].copy()
                print(f"  GT spikes after neuron filtering (eval clusters, no train filtering): {len(gt_spikes_in_calibration)}")
            else:
                print(f"  Warning: GT spike data has neither 'neuron' nor 'cluster' column, cannot filter by neuron")

        if len(gt_spikes_in_calibration) > 0:
            gt_spike_times = gt_spikes_in_calibration['time'].values.astype(int)
            gt_spike_neurons = gt_spikes_in_calibration['neuron'].values if 'neuron' in gt_spikes_in_calibration.columns else gt_spikes_in_calibration['cluster'].values
            gt_array_after_ext = np.stack([gt_spike_times, gt_spike_neurons], axis=1)

            gt_label_array = map_gt_annotation(
                detect_array_after_ext,
                gt_array_after_ext,
                neuron_to_channel_id=neuron_to_channel_id_eval,
                time_tolerance=1
            )
            matched_mask = gt_label_array >= 0
            n_matched = int(matched_mask.sum())
            n_unmatched = int(len(detect_array_after_ext) - n_matched)
            spike_detection_rate = n_matched / len(gt_array_after_ext) if len(gt_array_after_ext) > 0 else 0.0

            print(f"\n---spike detection rate (after waveform extraction): {spike_detection_rate:.4f}")
            print(f"Number of matched spikes: {n_matched}")
            print(f"Number of unmatched spikes: {n_unmatched}")
        else:
            print(f"\n---spike detection rate (after waveform extraction): N/A (no GT spikes after filtering)")
            print(f"  - GT spikes after time filtering: {len(eval_spike_inf[(eval_spike_inf['time'] >= 0) & (eval_spike_inf['time'] < actual_samples)]) if eval_spike_inf is not None else 0}")
            print(f"  - GT spikes after neuron filtering: 0")
            print(f"Number of matched spikes: 0")
            print(f"Number of unmatched spikes: {len(detect_array_after_ext)}")
    else:
        print(f"\n---spike detection rate (after waveform extraction): N/A (no GT data provided)")
        print(f"Number of matched spikes: 0")
        print(f"Number of unmatched spikes: {len(spike_times)}")
    
    print("\n### 4. Noise classifier filtering")
    autosort_model.eval()
    
    # Prepare data
    batch_size = 4096
    n_spikes = len(waveforms)
    spike_indices = []
    way3_features = []
    
    if skip_noise_classifier:
        print("  Skip noise classifier: all detected spikes will be kept and sent to label classifier directly.")
        with torch.no_grad():
            for i in tqdm(range(0, n_spikes, batch_size), desc="Label feature extraction (skip noise clf)"):
                batch_waveforms = waveforms[i:i+batch_size]  # (batch, n_channels, window_size)
                batch_channels = spike_channels[i:i+batch_size]
                
                batch_single_waveforms = []
                batch_multi_waveforms = []
                for wf, ch in zip(batch_waveforms, batch_channels):
                    multi_wf = wf.flatten()
                    batch_multi_waveforms.append(multi_wf)
                    single_wf = wf[ch, :]
                    batch_single_waveforms.append(single_wf)
                
                batch_multi_waveforms = np.array(batch_multi_waveforms)
                batch_single_waveforms = np.array(batch_single_waveforms)
                
                batch_multi = torch.from_numpy(batch_multi_waveforms).float().to(device)
                batch_single = torch.from_numpy(batch_single_waveforms).float().to(device)
                
                codes = torch.cat((batch_multi, batch_single), dim=1)
                way4_batch = autosort_model.clsfier_label.intermediate_forward(codes)
                way3_features.append(way4_batch.cpu().numpy())
        
        spike_indices = np.arange(n_spikes)
        way4_features = np.concatenate(way3_features, axis=0)
        print(f"Number of spikes passing noise classifier: {len(spike_indices)} (noise classifier skipped)")
    else:
        # Track noise classifier predictions for matched spikes (for debugging)
        matched_spike_indices_set = set()
        if eval_spike_inf is not None:
            # Get matched spike indices from waveform extraction stage
            detected_spike_times_after_extraction = np.array(spike_times).astype(int)
            calibration_end_frame = actual_samples
            gt_spikes_in_calibration = eval_spike_inf[
                (eval_spike_inf['time'] >= 0) & (eval_spike_inf['time'] < calibration_end_frame)
            ].copy()
            
            if eval_neuron_inf is not None and len(gt_spikes_in_calibration) > 0:
                clique_neuron_names = set(train_neuron_inf['Neuron'].unique()) if train_neuron_inf is not None else set()
                if 'neuron_match' in eval_neuron_inf.columns:
                    eval_to_train_mapping = {}
                    for idx, row in eval_neuron_inf.iterrows():
                        eval_neuron = row['Neuron']
                        train_neuron = row['neuron_match']
                        eval_to_train_mapping[eval_neuron] = train_neuron
                    
                    if 'neuron' in gt_spikes_in_calibration.columns:
                        gt_spikes_in_calibration['neuron_mapped'] = gt_spikes_in_calibration['neuron'].map(eval_to_train_mapping)
                        gt_spikes_in_calibration = gt_spikes_in_calibration[
                            (gt_spikes_in_calibration['neuron_mapped'].isin(clique_neuron_names)) &
                            (gt_spikes_in_calibration['neuron_mapped'] != 'unmatch')
                        ].copy()
                        gt_spikes_in_calibration = gt_spikes_in_calibration.drop(columns=['neuron_mapped'])
            
            if len(gt_spikes_in_calibration) > 0:
                gt_spike_times = gt_spikes_in_calibration['time'].values.astype(int)
                tolerance_samples = 1
                
                # Find matched spike indices
                for det_idx, det_time in enumerate(detected_spike_times_after_extraction):
                    time_diffs = np.abs(gt_spike_times - det_time)
                    min_diff = np.min(time_diffs)
                    if min_diff <= tolerance_samples:
                        matched_spike_indices_set.add(det_idx)
        
        # Store noise classifier outputs for matched spikes
        matched_spike_noise_probs = []  # List of (noise_prob, spike_prob) for matched spikes
        
        with torch.no_grad():
            for i in tqdm(range(0, n_spikes, batch_size), desc="Noise classification"):
                batch_waveforms = waveforms[i:i+batch_size]  # (batch, n_channels, window_size)
                batch_channels = spike_channels[i:i+batch_size]
                
                # Extract single waveform (maximum amplitude channel)
                batch_single_waveforms = []
                batch_multi_waveforms = []
                
                for wf, ch in zip(batch_waveforms, batch_channels):
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
                
                # Concatenate
                codes = torch.cat((batch_multi, batch_single), dim=1)  # (batch, (n_channels+1)*window_size)

                noise_output = autosort_model.clsfier_noise(codes)
                noise_probs = torch.softmax(noise_output, dim=1)  # (batch, 2) - [noise_prob, spike_prob]
                noise_pred = torch.argmax(noise_output, dim=1)  # 0=noise, 1=spike
                
                # Track noise classifier predictions for matched spikes
                batch_start_idx = i
                for batch_idx, global_idx in enumerate(range(batch_start_idx, min(batch_start_idx + batch_size, n_spikes))):
                    if global_idx in matched_spike_indices_set:
                        noise_prob = noise_probs[batch_idx, 0].item()  # Probability of noise
                        spike_prob = noise_probs[batch_idx, 1].item()  # Probability of spike
                        pred_label = noise_pred[batch_idx].item()
                        matched_spike_noise_probs.append({
                            'index': global_idx,
                            'noise_prob': noise_prob,
                            'spike_prob': spike_prob,
                            'predicted': 'noise' if pred_label == 0 else 'spike'
                        })
                
                spike_mask = noise_pred == 1
                if spike_mask.sum() > 0:
                    batch_indices = np.arange(i, min(i+batch_size, n_spikes))[spike_mask.cpu().numpy()]
                    spike_indices.extend(batch_indices)
                    
                    # Extract way4 layer features (30 dimensions, no PCA needed)
                    codes_spike = codes[spike_mask]
                    way4_batch = autosort_model.clsfier_label.intermediate_forward(codes_spike)
                    way3_features.append(way4_batch.cpu().numpy())  # Note: variable name kept as way3_features for compatibility
        
        if len(spike_indices) == 0:
            raise ValueError("No spikes passed noise classifier")
        
        way4_features = np.concatenate(way3_features, axis=0)  # (n_spikes, 30) - way4 features are already 30 dimensions
        spike_indices = np.array(spike_indices)
        print(f"Number of spikes passing noise classifier: {len(spike_indices)}")
        
        # Analyze noise classifier performance on matched spikes
        if len(matched_spike_noise_probs) > 0:
            matched_spike_indices_set_filtered = set(spike_indices)  # Indices that passed noise classifier
            matched_spikes_passed = sum(1 for item in matched_spike_noise_probs if item['index'] in matched_spike_indices_set_filtered)
            matched_spikes_rejected = len(matched_spike_noise_probs) - matched_spikes_passed
            
            print(f"\n---Noise classifier analysis on matched spikes:")
            print(f"  Total matched spikes (from waveform extraction): {len(matched_spike_noise_probs)}")
            print(f"  Matched spikes passing noise classifier: {matched_spikes_passed}")
            print(f"  Matched spikes rejected by noise classifier: {matched_spikes_rejected}")
            print(f"  Matched spike retention rate: {matched_spikes_passed / len(matched_spike_noise_probs):.4f}")
            
            # Analyze probability distribution and waveform features
            rejected_matched = [item for item in matched_spike_noise_probs if item['index'] not in matched_spike_indices_set_filtered]
            if len(rejected_matched) > 0:
                noise_probs_rejected = [item['noise_prob'] for item in rejected_matched]
                spike_probs_rejected = [item['spike_prob'] for item in rejected_matched]
                
                # Calculate waveform features for rejected matched spikes
                rejected_indices = [item['index'] for item in rejected_matched]
                rejected_waveforms = waveforms[rejected_indices]  # (n_rejected, n_channels, window_size)
                rejected_channels = [spike_channels[i] for i in rejected_indices]
                
                # Calculate peak-to-peak amplitude (maximum amplitude channel)
                rejected_amplitudes = []
                for wf, ch in zip(rejected_waveforms, rejected_channels):
                    single_wf = wf[ch, :]  # Waveform from maximum amplitude channel
                    peak_to_peak = np.max(single_wf) - np.min(single_wf)
                    rejected_amplitudes.append(peak_to_peak)
                
                # Calculate SNR (signal-to-noise ratio) as peak amplitude / std of baseline
                rejected_snrs = []
                for wf, ch in zip(rejected_waveforms, rejected_channels):
                    single_wf = wf[ch, :]
                    # Use first 10 samples as baseline
                    baseline = single_wf[:10] if len(single_wf) >= 10 else single_wf[:len(single_wf)//2]
                    baseline_std = np.std(baseline) if len(baseline) > 1 else 1.0
                    peak_amplitude = np.max(np.abs(single_wf))
                    snr = peak_amplitude / baseline_std if baseline_std > 0 else 0.0
                    rejected_snrs.append(snr)

            passed_matched = [item for item in matched_spike_noise_probs if item['index'] in matched_spike_indices_set_filtered]
            if len(passed_matched) > 0:
                noise_probs_passed = [item['noise_prob'] for item in passed_matched]
                spike_probs_passed = [item['spike_prob'] for item in passed_matched]
                
                # Calculate waveform features for passed matched spikes
                passed_indices = [item['index'] for item in passed_matched]
                passed_waveforms = waveforms[passed_indices]  # (n_passed, n_channels, window_size)
                passed_channels = [spike_channels[i] for i in passed_indices]
                
                # Calculate peak-to-peak amplitude (maximum amplitude channel)
                passed_amplitudes = []
                for wf, ch in zip(passed_waveforms, passed_channels):
                    single_wf = wf[ch, :]  # Waveform from maximum amplitude channel
                    peak_to_peak = np.max(single_wf) - np.min(single_wf)
                    passed_amplitudes.append(peak_to_peak)
                
                # Calculate SNR (signal-to-noise ratio) as peak amplitude / std of baseline
                passed_snrs = []
                for wf, ch in zip(passed_waveforms, passed_channels):
                    single_wf = wf[ch, :]
                    # Use first 10 samples as baseline
                    baseline = single_wf[:10] if len(single_wf) >= 10 else single_wf[:len(single_wf)//2]
                    baseline_std = np.std(baseline) if len(baseline) > 1 else 1.0
                    peak_amplitude = np.max(np.abs(single_wf))
                    snr = peak_amplitude / baseline_std if baseline_std > 0 else 0.0
                    passed_snrs.append(snr)

    # 2.6. Compare spikes passing noise classifier with GT (if available)
    if eval_spike_inf is not None:
        # Get spike times that passed noise classifier (in samples, relative to calibration start)
        # spike_indices are indices into the waveforms array, which corresponds to spike_times
        spikes_passing_noise_times = np.array([spike_times[i] for i in spike_indices]).astype(int)  # (n_passing_noise,)
        
        # Filter GT spikes: use ALL neurons whose channels are inside this clique (eval set)
        # First, filter by time (within calibration duration)
        calibration_end_frame = actual_samples
        gt_spikes_in_calibration = eval_spike_inf[
            (eval_spike_inf['time'] >= 0) & (eval_spike_inf['time'] < calibration_end_frame)
        ].copy()
        
        print(f"  GT spikes after time filtering: {len(gt_spikes_in_calibration)}")
        
        # Second, filter by neuron (ALL eval neurons in this clique; no train filtering)
        if eval_neuron_inf is not None and len(gt_spikes_in_calibration) > 0:
            clique_neuron_names = set(eval_neuron_inf['Neuron'].unique())
            
            print(f"  Clique neuron names (eval set, full clique): {sorted(clique_neuron_names)}")
            print(f"  GT spike columns: {gt_spikes_in_calibration.columns.tolist()}")
            
            if 'neuron' in gt_spikes_in_calibration.columns:
                gt_spikes_in_calibration = gt_spikes_in_calibration[
                    gt_spikes_in_calibration['neuron'].isin(clique_neuron_names)
                ].copy()
                print(f"  GT spikes after neuron filtering (eval neurons, no train filtering): {len(gt_spikes_in_calibration)}")
            elif 'cluster' in gt_spikes_in_calibration.columns:
                gt_spikes_in_calibration = gt_spikes_in_calibration[
                    gt_spikes_in_calibration['cluster'].isin(clique_neuron_names)
                ].copy()
                print(f"  GT spikes after neuron filtering (eval clusters, no train filtering): {len(gt_spikes_in_calibration)}")
            else:
                print(f"  Warning: GT spike data has neither 'neuron' nor 'cluster' column, cannot filter by neuron")
        
        if len(gt_spikes_in_calibration) > 0:
            gt_spike_times = gt_spikes_in_calibration['time'].values.astype(int)
            gt_spike_neurons = gt_spikes_in_calibration['neuron'].values if 'neuron' in gt_spikes_in_calibration.columns else gt_spikes_in_calibration['cluster'].values
            gt_array_noise = np.stack([gt_spike_times, gt_spike_neurons], axis=1)

            # Channels for spikes passing noise classifier (clique indices)
            spike_channels_passing = [spike_channels[i] for i in spike_indices]
            detect_array_noise = np.stack([spikes_passing_noise_times, spike_channels_passing], axis=1)

            gt_label_array = map_gt_annotation(
                detect_array_noise,
                gt_array_noise,
                neuron_to_channel_id=neuron_to_channel_id_eval,
                time_tolerance=1
            )
            matched_mask = gt_label_array >= 0
            n_matched = int(matched_mask.sum())
            n_unmatched = int(len(detect_array_noise) - n_matched)
            spike_detection_rate = n_matched / len(gt_array_noise) if len(gt_array_noise) > 0 else 0.0

            print(f"\n---spike detection rate (after noise classifier): {spike_detection_rate:.4f}")
            print(f"Number of matched spikes: {n_matched}")
            print(f"Number of unmatched spikes: {n_unmatched}")
        else:
            print(f"\n---spike detection rate (after noise classifier): N/A (no GT spikes after filtering)")
            print(f"  - GT spikes after time filtering: {len(eval_spike_inf[(eval_spike_inf['time'] >= 0) & (eval_spike_inf['time'] < actual_samples)]) if eval_spike_inf is not None else 0}")
            print(f"  - GT spikes after neuron filtering: 0")
            print(f"Number of matched spikes: 0")
            print(f"Number of unmatched spikes: {len(spikes_passing_noise_times)}")
    else:
        print(f"\n---spike detection rate (after noise classifier): N/A (no GT data provided)")
        print(f"Number of matched spikes: 0")
        print(f"Number of unmatched spikes: {len(spike_indices)}")
    
    # Filter spike_times and spike_channels to only include spikes that passed noise classifier
    # spike_indices are indices into the waveforms array (which corresponds to spike_times and spike_channels)
    spike_times_filtered = [spike_times[i] for i in spike_indices]  # Only spikes that passed noise classifier
    spike_channels_filtered = [spike_channels[i] for i in spike_indices]  # Only spikes that passed noise classifier
    
    # Create mapping from spike_indices values to spike_times_filtered indices
    # This is needed because cluster_spike_indices are values from spike_indices array,
    # but we need to index into spike_times_filtered which has the same length as spike_indices
    spike_indices_to_filtered_idx = {val: idx for idx, val in enumerate(spike_indices)}
    
    # 5. K-means clustering (directly on way4 features, no PCA needed)
    print("\n### 5. K-means clustering (using way4 features, 30 dimensions, no PCA)")
    n_train_neurons = len(train_neuron_inf)
    n_clusters = n_train_neurons + n_additional_clusters
    print(f"Number of clusters: {n_clusters} (Training neurons: {n_train_neurons}, additional: {n_additional_clusters})")
    print(f"Feature shape: {way4_features.shape} (way4 features, 30 dimensions)")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(way4_features)  # (n_spikes,) - directly on way4 features

    # 7. Calculate position and waveform for each train neuron and each cluster, then match
    print("\n### 7. Calculate cluster position and waveform (based on train neuron channel_id) and match")
    cluster_to_neuron_mapping = {}  # {cluster_id: train_neuron_name}
    neuron_to_clusters = defaultdict(list)  # {train_neuron_name: [cluster_ids]}
    cluster_features = {}  # Save matched cluster features
    neuron_cluster_comparison = {}  # {train_neuron_name: {cluster_id: {position, waveform, position_distance, waveform_corr}}}
    
    # Prepare traces_original_for_extraction in (n_channels, n_timepoints) format for consistent extraction
    # This ensures consistency with generate_neuron_inf_phy_template.py
    # After the transpose check at line 2684-2685, traces_original should be (n_channels, n_timepoints)
    if traces_original.shape[0] > traces_original.shape[1] and traces_original.shape[0] > 100:
        # If shape is (n_timepoints, n_channels), transpose to (n_channels, n_timepoints)
        traces_original_for_extraction = traces_original.T
    else:
        # Already in (n_channels, n_timepoints) format
        traces_original_for_extraction = traces_original
    
    # Outer loop: iterate through each train neuron
    for train_idx, train_row in train_neuron_inf.iterrows():
        train_neuron = train_row['Neuron']
        train_pos = np.array([train_row['position_1'], train_row['position_2']])
        train_waveform = np.asarray(train_row['position_waveform'], dtype=np.float32)
        
        # Get train neuron channel_id (these are probe device channel indices, need to convert to clique indices)
        train_channel_id_probe = train_row['channel_id']
        if not isinstance(train_channel_id_probe, list):
            if isinstance(train_channel_id_probe, (np.ndarray, tuple)):
                train_channel_id_probe = list(train_channel_id_probe)
            else:
                # Try to parse string
                import ast
                try:
                    train_channel_id_probe = ast.literal_eval(str(train_channel_id_probe))
                    if not isinstance(train_channel_id_probe, list):
                        train_channel_id_probe = [train_channel_id_probe]
                except:
                    print(f"  Warning: Neuron {train_neuron} channel_id cannot be parsed, skipping")
                    continue
        
        if len(train_channel_id_probe) == 0:
            print(f"  Warning: Neuron {train_neuron} has no valid channel_id, skipping")
            continue
        
        # Convert probe channel indices to clique channel indices
        train_channel_id = []
        for probe_ch in train_channel_id_probe:
            if int(probe_ch) in probe_to_clique_index:
                clique_ch = probe_to_clique_index[int(probe_ch)]
                train_channel_id.append(clique_ch)
        
        if len(train_channel_id) == 0:
            print(f"  Warning: Neuron {train_neuron} channel_id ({train_channel_id_probe}) not in clique, skipping")
            continue
        
        # Inner loop: iterate through each kmeans cluster
        for cluster_id in range(n_clusters):
            # If this cluster is already matched to another neuron, skip (one cluster can only match one neuron)
            if cluster_id in cluster_to_neuron_mapping:
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            
            if len(cluster_spike_indices) == 0:
                continue
            
            # Re-extract waveforms from original data (not from whitened waveforms)
            # Get spike times and channels for this cluster
            # Note: cluster_spike_indices are values from spike_indices array, need to map to spike_times_filtered indices
            cluster_spike_times = [spike_times_filtered[spike_indices_to_filtered_idx[i]] for i in cluster_spike_indices]  # Original spike time indices
            cluster_spike_chs = [spike_channels_filtered[spike_indices_to_filtered_idx[i]] for i in cluster_spike_indices]  # Original spike channel indices
            
            # Extract waveforms from original traces (not whitened)
            # Use traces_original_for_extraction which is already in (n_channels, n_timepoints) format
            cluster_waveforms_full = []
            
            for time_idx, channel_idx in zip(cluster_spike_times, cluster_spike_chs):
                start = time_idx - left_sample_extract  # Extract 20 samples before
                end = time_idx + right_sample_extract  # Extract 40 samples after
                
                # Check boundaries using time dimension (second dimension of (n_channels, n_timepoints))
                if start < 0 or end > traces_original_for_extraction.shape[1]:
                    continue
                if end - start != window_size_extract:
                    continue
                
                # Extract waveform from original traces (n_channels, window_size_extract=60)
                # No downsampling - use full 60 samples (left 20, right 40)
                # traces_original_for_extraction is (n_channels, n_timepoints), so use [:, start:end]
                waveform = traces_original_for_extraction[:, start:end]  # (n_channels, window_size_extract=60) - using original data, no downsampling
                
                cluster_waveforms_full.append(waveform)
            
            if len(cluster_waveforms_full) == 0:
                continue
            
            cluster_waveforms_full = np.array(cluster_waveforms_full)  # (n_cluster_spikes, n_channels, window_size_extract=60) - original data, no downsampling
            actual_window_size = cluster_waveforms_full.shape[2]  # Should be window_size_extract (60), not window_size (30)
            
            # Use train neuron channel_id to extract corresponding channels
            # Ensure channel_id is within valid range
            valid_channel_id = [ch for ch in train_channel_id if 0 <= ch < n_channels]
            if len(valid_channel_id) == 0:
                continue
            
            # Extract channels corresponding to train neuron channel_id from cluster_waveforms
            cluster_waveforms = cluster_waveforms_full[:, valid_channel_id, :]  # (n_spikes, n_valid_channels, actual_window_size) - original data
            
            # Get actual window size from cluster_waveforms (should be 60, no downsampling)
            actual_window_size_for_compute = cluster_waveforms.shape[2]
            
            # Convert clique channel indices back to probe channel indices for position lookup
            # channel_positions uses probe channel indices, not clique indices
            valid_probe_channel_id = [clique_to_probe_index[ch] for ch in valid_channel_id if ch in clique_to_probe_index]
            
            # Calculate position and waveform (using train neuron channel_id, on original data)
            # Save all waveforms (n_spikes, n_channels, time_window) instead of mean
            # Note: compute_cluster_position_waveform expects probe channel indices for position lookup
            # Use channel_positions from probe (dynamic, from actual probe geometry)
            position_1, position_2, position_waveform = compute_cluster_position_waveform(
                cluster_waveforms, valid_probe_channel_id, channel_positions, actual_window_size_for_compute
            )
            
            # Store all waveforms (n_spikes, n_channels, time_window) for this cluster
            # This is the raw waveform matrix without mean calculation
            cluster_waveforms_all = cluster_waveforms  # (n_spikes, n_valid_channels, actual_window_size)
            
            # Calculate position distance
            cluster_pos = np.array([position_1, position_2])
            pos_distance = np.linalg.norm(cluster_pos - train_pos)
            
            # Calculate waveform similarity
            min_len = min(len(position_waveform), len(train_waveform))
            if min_len == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(position_waveform[:min_len], train_waveform[:min_len])
            
            # Store comparison information for this train neuron and cluster (regardless of threshold)
            if train_neuron not in neuron_cluster_comparison:
                neuron_cluster_comparison[train_neuron] = {}
            
            neuron_cluster_comparison[train_neuron][cluster_id] = {
                'position': [position_1, position_2],
                'waveform': position_waveform,  # Mean waveform
                'position_distance': pos_distance,
                'waveform_corr': corr,
                'n_spikes': len(cluster_spike_indices),
            }
            
            # Apply thresholds for matching
            if pos_distance >= position_threshold:
                continue
            
            if corr < waveform_similarity_threshold:
                continue
            
            # Calculate comprehensive score (smaller distance, higher correlation, higher score)
            # Note: score is kept for reference, but matching is based on position_distance (smaller is better)
            score = corr / (1 + pos_distance / position_threshold)
            
            # Establish mapping relationship (one cluster can only match one neuron, choose optimal)
            # If this cluster is not yet matched, or current match has smaller distance, update
            if cluster_id not in cluster_to_neuron_mapping:
                cluster_to_neuron_mapping[cluster_id] = train_neuron
                neuron_to_clusters[train_neuron].append(cluster_id)
                cluster_features[cluster_id] = {
                    'position_1': position_1,
                    'position_2': position_2,
                    'position_waveform': position_waveform,  # Mean waveform for compatibility
                    'waveforms': cluster_waveforms_all,  # All waveforms (n_spikes, n_channels, time_window) - no mean
                    'n_spikes': len(cluster_spike_indices),
                    'matched_neuron': train_neuron,
                    'score': score,
                    'pos_distance': pos_distance,
                    'waveform_corr': corr,
                }
            else:
                # If already matched, compare position distances, choose the one with smaller distance
                existing_neuron = cluster_to_neuron_mapping[cluster_id]
                existing_distance = cluster_features[cluster_id]['pos_distance']
                if pos_distance < existing_distance:
                    # Remove old mapping
                    neuron_to_clusters[existing_neuron].remove(cluster_id)
                    # Establish new mapping
                    cluster_to_neuron_mapping[cluster_id] = train_neuron
                    neuron_to_clusters[train_neuron].append(cluster_id)
                    cluster_features[cluster_id] = {
                        'position_1': position_1,
                        'position_2': position_2,
                        'position_waveform': position_waveform,  # Mean waveform for compatibility
                        'waveforms': cluster_waveforms_all,  # All waveforms (n_spikes, n_channels, time_window) - no mean
                        'n_spikes': len(cluster_spike_indices),
                        'matched_neuron': train_neuron,
                        'score': score,
                        'pos_distance': pos_distance,
                        'waveform_corr': corr,
                    }    
    all_train_channel_ids_clique = set()
    for train_idx, train_row in train_neuron_inf.iterrows():
        train_channel_id_probe = train_row.get('channel_id', [])
        if isinstance(train_channel_id_probe, str):
            import ast
            try:
                train_channel_id_probe = ast.literal_eval(train_channel_id_probe)
            except:
                train_channel_id_probe = []
        elif not isinstance(train_channel_id_probe, (list, tuple, np.ndarray)):
            train_channel_id_probe = []
        
        # Convert probe channel indices to clique channel indices
        for probe_ch in train_channel_id_probe:
            if int(probe_ch) in probe_to_clique_index:
                clique_ch = probe_to_clique_index[int(probe_ch)]
                all_train_channel_ids_clique.add(clique_ch)
    
    # Use union of all train neuron channel_ids (clique indices), or fallback to all channels
    default_channel_id = sorted(list(all_train_channel_ids_clique)) if len(all_train_channel_ids_clique) > 0 else list(range(n_channels))
    
    for cluster_id in range(n_clusters):
        if cluster_id not in cluster_to_neuron_mapping:
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            
            if len(cluster_spike_indices) > 0:
                # Get spike times and channels for this cluster
                # Note: cluster_spike_indices are values from spike_indices array, need to map to spike_times_filtered indices
                cluster_spike_times = [spike_times_filtered[spike_indices_to_filtered_idx[i]] for i in cluster_spike_indices]  # Original spike time indices
                cluster_spike_chs = [spike_channels_filtered[spike_indices_to_filtered_idx[i]] for i in cluster_spike_indices]  # Original spike channel indices
                

                cluster_waveforms_full = []
                
                for time_idx, channel_idx in zip(cluster_spike_times, cluster_spike_chs):
                    start = time_idx - left_sample_extract  # Extract 20 samples before
                    end = time_idx + right_sample_extract  # Extract 40 samples after
                    
                    # Check boundaries using time dimension (second dimension of (n_channels, n_timepoints))
                    if start < 0 or end > traces_original_for_extraction.shape[1]:
                        continue
                    if end - start != window_size_extract:
                        continue
                    
                    # Extract waveform from original traces (n_channels, window_size_extract=60)
                    # No downsampling - use full 60 samples (left 20, right 40)
                    # traces_original_for_extraction is (n_channels, n_timepoints), so use [:, start:end]
                    waveform = traces_original_for_extraction[:, start:end]  # (n_channels, window_size_extract=60) - using original data, no downsampling
                    
                    cluster_waveforms_full.append(waveform)
                
                if len(cluster_waveforms_full) > 0:
                    cluster_waveforms_full = np.array(cluster_waveforms_full)  # (n_cluster_spikes, n_channels, window_size_extract=60) - original data, no downsampling
                    actual_window_size = cluster_waveforms_full.shape[2]  # Should be window_size_extract (60), not window_size (30)
                
                valid_channel_id = [ch for ch in default_channel_id if 0 <= ch < n_channels]
                if len(valid_channel_id) > 0:
                    cluster_waveforms = cluster_waveforms_full[:, valid_channel_id, :]  # (n_spikes, n_valid_channels, actual_window_size) - original
                    # Get actual window size from cluster_waveforms (should be 60, not 30)
                    actual_window_size_for_compute = cluster_waveforms.shape[2]
                    # Convert clique channel indices back to probe channel indices for position lookup
                    valid_probe_channel_id = [clique_to_probe_index[ch] for ch in valid_channel_id if ch in clique_to_probe_index]
                    position_1, position_2, position_waveform = compute_cluster_position_waveform(
                        cluster_waveforms, valid_probe_channel_id, channel_positions, actual_window_size_for_compute
                    )
                    # Store all waveforms (n_spikes, n_channels, time_window) for this cluster
                    cluster_waveforms_all = cluster_waveforms  # (n_spikes, n_valid_channels, actual_window_size)
                    cluster_features[cluster_id] = {
                        'position_1': position_1,
                        'position_2': position_2,
                        'position_waveform': position_waveform,  # Mean waveform for compatibility
                        'waveforms': cluster_waveforms_all,  # All waveforms (n_spikes, n_channels, time_window) - no mean
                        'n_spikes': len(cluster_spike_indices),
                        'matched_neuron': 'unmatch',
                        'score': None,
                        'pos_distance': None,
                        'waveform_corr': None,
                    }
    
    # Handle conflicts: one cluster can only map to one neuron, choose optimal
    # If a cluster satisfies conditions for multiple neurons, choose highest score
    # (code above already handles this)
    
    # Handle one-to-many: one neuron can map to multiple clusters (keep all mappings)
    
    print(f"\nMatching results:")
    print(f"  - Total clusters: {n_clusters}")
    print(f"  - Matched clusters: {len(cluster_to_neuron_mapping)}")
    print(f"  - Unmatched clusters: {n_clusters - len(cluster_to_neuron_mapping)}")
    print(f"  - Matched neurons: {len(neuron_to_clusters)}")
    
    # Build results DataFrame (for confusion matrix)
    results_df = pd.DataFrame({
        'spike_time': [spike_times[i] for i in spike_indices],
        'spike_channel': [spike_channels[i] for i in spike_indices],
        'predicted_label': [cluster_to_neuron_mapping.get(cluster_labels[i], 'unmatch') for i in range(len(spike_indices))],
    })
    
    # If eval data exists, add GT labels
    if eval_neuron_inf is not None and eval_spike_inf is not None:
        # Establish neuron mapping (from eval_neuron_inf to train_neuron)
        if 'neuron_match' in eval_neuron_inf.columns:
            # Establish mapping from eval neuron to train neuron
            eval_to_train_mapping = {}
            for _, row in eval_neuron_inf.iterrows():
                eval_neuron = row['Neuron']
                match = row['neuron_match']
                if match != 'unmatch':
                    eval_to_train_mapping[eval_neuron] = match
                else:
                    eval_to_train_mapping[eval_neuron] = 'unmatch'
            
            # Match GT labels based on spike_time
            spike_times_array = np.array([spike_times[i] for i in spike_indices])
            spike_inf_sorted = eval_spike_inf.sort_values('time').reset_index(drop=True)
            
            gt_labels = []
            for spike_time in spike_times_array:
                # Find corresponding spike in eval_spike_inf (allow ±1 sample point error)
                time_diff = (spike_inf_sorted['time'] - spike_time).abs()
                min_diff_idx = time_diff.idxmin()
                min_diff = time_diff.loc[min_diff_idx]
                
                if min_diff <= 1:  # Allow ±1 sample point error
                    eval_neuron = spike_inf_sorted.loc[min_diff_idx, 'neuron']
                    
                    # Map to train neuron
                    if eval_neuron in eval_to_train_mapping:
                        gt_label = eval_to_train_mapping[eval_neuron]
                    else:
                        gt_label = 'unmatch'
                else:
                    gt_label = 'noise'  # No matching GT spike found, treat as noise
                
                gt_labels.append(gt_label)
            
            results_df['gt_label'] = gt_labels
        else:
            print("Warning: eval_neuron_inf has no neuron_match column, cannot establish GT label mapping")
            results_df['gt_label'] = 'unknown'
    else:
        results_df['gt_label'] = None
    
    # Save way3 features for visualization
    # For noise detection: need way3 features and noise classification results for all detected spikes
    # all_way3_features_noise = []  # Way3 features for all detected spikes (100 dimensions)
    # all_noise_gt_labels = []  # GT noise/spike labels
    # all_noise_pred_labels = []  # Predicted noise/spike labels
    
    # # Reprocess all detected spikes to get way3 features
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
    #         codes = torch.cat((batch_multi, batch_single), dim=1)
            
    #         # Noise classification
    #         noise_output = autosort_model.clsfier_noise(codes)
    #         noise_pred = torch.argmax(noise_output, dim=1)
            
    #         # Extract way3 features (for all spikes, including those classified as noise)
    #         way3_batch = autosort_model.clsfier_label.intermediate_forward(codes)
    #         all_way3_features_noise.append(way3_batch.cpu().numpy())
    #         all_noise_pred_labels.extend(noise_pred.cpu().numpy().tolist())
            
    #         # Get GT noise labels (if eval data exists)
    #         if eval_neuron_inf is not None and eval_spike_inf is not None:
    #             batch_spike_times = spike_times[i:i+batch_size]
    #             batch_gt_noise = []
    #             for st in batch_spike_times:
    #                 time_diff = (eval_spike_inf['time'] - st).abs()
    #                 if time_diff.min() <= 1:
    #                     batch_gt_noise.append(1)  # spike
    #                 else:
    #                     batch_gt_noise.append(0)  # noise
    #             all_noise_gt_labels.extend(batch_gt_noise)
    #         else:
    #             all_noise_gt_labels.extend([-1] * len(batch_waveforms))  # Unknown
    
    # all_way3_features_noise = np.concatenate(all_way3_features_noise, axis=0) if len(all_way3_features_noise) > 0 else np.array([])
    
    # ==================== Evaluation Metrics Calculation ====================
    print("\n" + "=" * 80)
    print("Computing evaluation metrics...")
    print("=" * 80)
    
    # Get train neuron list
    train_neuron_list = train_neuron_inf['Neuron'].tolist() if train_neuron_inf is not None else []
    
    # 1. Generate confusion matrix
    print("\n1. Generating confusion matrix...")
    confusion_matrix_df, summary_df = generate_confusion_matrix_df(
        results_df=results_df,
        train_neuron_list=train_neuron_list
    )
    
    # 2. Calculate classification accuracy (all classes, with unmatch+noise treated as one class)
    print("2. Calculating classification accuracy (all classes, merging unmatch & noise)...")
    if 'All' in confusion_matrix_df.index and 'All' in confusion_matrix_df.columns:
        total_samples = confusion_matrix_df.loc['All', 'All']
        correct_predictions = 0
        # Correct train-neuron predictions
        for neuron in train_neuron_list:
            if neuron in confusion_matrix_df.index and neuron in confusion_matrix_df.columns:
                correct_predictions += confusion_matrix_df.loc[neuron, neuron]
        # Treat unmatch + noise as同一类：GT 为 unmatch 或 noise，预测为 unmatch 或 noise 都算正确
        bg_rows = [lbl for lbl in ['unmatch', 'noise'] if lbl in confusion_matrix_df.index]
        bg_cols = [lbl for lbl in ['unmatch', 'noise'] if lbl in confusion_matrix_df.columns]
        if bg_rows and bg_cols:
            correct_predictions += confusion_matrix_df.loc[bg_rows, bg_cols].to_numpy().sum()
        classification_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    else:
        classification_accuracy = 0.0
    print(f"   Classification accuracy (all classes, unmatch+noise merged): {classification_accuracy:.6f}")
    
    # 2.1. Calculate unit classification accuracy (only train neurons, excluding unmatch and noise)
    print("2.1. Calculating unit classification accuracy (train neurons only)...")
    if 'All' in confusion_matrix_df.index and 'All' in confusion_matrix_df.columns:
        # Calculate total samples for train neurons only (GT is train neuron)
        total_unit_samples = 0
        correct_unit_predictions = 0
        for neuron in train_neuron_list:
            if neuron in confusion_matrix_df.index:
                # Total samples with GT = this neuron
                total_unit_samples += confusion_matrix_df.loc[neuron, 'All']
                # Correct predictions (predicted = this neuron)
                if neuron in confusion_matrix_df.columns:
                    correct_unit_predictions += confusion_matrix_df.loc[neuron, neuron]
        unit_classification_accuracy = correct_unit_predictions / total_unit_samples if total_unit_samples > 0 else 0.0
    else:
        unit_classification_accuracy = 0.0
    print(f"   Unit classification accuracy (train neurons only): {unit_classification_accuracy:.6f}")
    print(f"   Total unit samples (GT is train neuron): {total_unit_samples if 'All' in confusion_matrix_df.index and 'All' in confusion_matrix_df.columns else 0}")
    
    # 3. Calculate noise detection metrics
    print("3. Calculating noise detection metrics...")
    noise_detection_metrics = compute_noise_detection_metrics(
        results_df=results_df,
        train_neuron_list=train_neuron_list
    )
    noise_detection_accuracy = noise_detection_metrics['accuracy']
    noise_detection_accuracy_adjusted = noise_detection_metrics.get('accuracy_adjusted', noise_detection_metrics.get('accuracy', 0.0))  # Use adjusted accuracy (treat unmatch as noise in GT)
    print(f"   Noise detection accuracy: {noise_detection_accuracy:.6f}")
    print(f"   Noise detection accuracy (adjusted): {noise_detection_accuracy_adjusted:.6f}")
    
    # # 4. Generate UMAP coordinates for visualization
    # print("4. Generating UMAP coordinates...")
    # import umap
    # from sklearn.decomposition import PCA
    
    # # 4.1 UMAP for noise detection
    # umap_noise_df = None
    # if len(all_way3_features_noise) > 0:
    #     print("   4.1 Noise detection UMAP...")
    #     # PCA to 30 dimensions
    #     pca_noise = PCA(n_components=30)
    #     way3_features_noise_30d = pca_noise.fit_transform(all_way3_features_noise)
        
    #     # UMAP to 2 dimensions
    #     reducer_noise = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    #     noise_umap_coords = reducer_noise.fit_transform(way3_features_noise_30d)
        
    #     # Create DataFrame
    #     noise_gt_labels_str = ['spike' if label == 1 else 'noise' for label in all_noise_gt_labels] if len(all_noise_gt_labels) > 0 else ['unknown'] * len(noise_umap_coords)
    #     noise_pred_labels_str = ['spike' if label == 1 else 'noise' for label in all_noise_pred_labels] if len(all_noise_pred_labels) > 0 else ['unknown'] * len(noise_umap_coords)
        
    #     umap_noise_df = pd.DataFrame({
    #         'UMAP_1': noise_umap_coords[:, 0],
    #         'UMAP_2': noise_umap_coords[:, 1],
    #         'gt_label': noise_gt_labels_str,
    #         'predicted_label': noise_pred_labels_str
    #     })
    #     print(f"      Generated {len(umap_noise_df)} points")
    
    # # 4.2 UMAP for label classification
    # umap_label_df = None
    # if len(way3_pca) > 0 and len(results_df) > 0:
    #     print("   4.2 Label classification UMAP...")
    #     # Filter: only keep points where gt label is not unmatch/noise and predicted label is not unmatch
    #     valid_indices = []
    #     valid_gt_labels = []
    #     valid_pred_labels = []
        
    #     for idx in range(len(way3_pca)):
    #         if idx < len(results_df):
    #             gt_label = results_df.iloc[idx]['gt_label']
    #             pred_label = results_df.iloc[idx]['predicted_label']
                
    #             if (gt_label not in ['unmatch', 'noise', 'unknown', None]) and (pred_label != 'unmatch'):
    #                 valid_indices.append(idx)
    #                 valid_gt_labels.append(gt_label)
    #                 valid_pred_labels.append(pred_label)
        
    #     if len(valid_indices) > 0:
    #         valid_indices = np.array(valid_indices)
    #         way3_features_label_filtered = way3_pca[valid_indices]
            
    #         # UMAP to 2 dimensions
    #         reducer_label = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    #         label_umap_coords = reducer_label.fit_transform(way3_features_label_filtered)
            
    #         # Create DataFrame
    #         umap_label_df = pd.DataFrame({
    #             'UMAP_1': label_umap_coords[:, 0],
    #             'UMAP_2': label_umap_coords[:, 1],
    #             'gt_label': valid_gt_labels,
    #             'predicted_label': valid_pred_labels
    #         })
    #         print(f"      Generated {len(umap_label_df)} points (filtered from {len(way3_pca)})")
    #     else:
    #         print("      No valid points for label classification UMAP")
    
    # # 5. Generate visualization figures
    # print("5. Generating visualization figures...")
    # figs = visualize_umap_features(
    #     way3_features_100d=all_way3_features_noise,
    #     way3_features_30d=way3_pca,
    #     results_df=results_df,
    #     train_neuron_list=train_neuron_list,
    #     noise_gt_labels=np.array(all_noise_gt_labels) if len(all_noise_gt_labels) > 0 else None,
    #     noise_pred_labels=np.array(all_noise_pred_labels) if len(all_noise_pred_labels) > 0 else None,
    #     neuron_inf_color=None,
    #     n_samples=50000,
    #     random_state=42
    # )
    
    # # Generate confusion matrix figure
    # import matplotlib.pyplot as plt
    # fig_confusion = plt.figure(figsize=(12, 10))
    # ax = fig_confusion.add_subplot(111)
    
    # # Plot confusion matrix as heatmap
    # confusion_matrix_plot = confusion_matrix_df.copy()
    # if 'All' in confusion_matrix_plot.index:
    #     confusion_matrix_plot = confusion_matrix_plot.drop('All')
    # if 'All' in confusion_matrix_plot.columns:
    #     confusion_matrix_plot = confusion_matrix_plot.drop('All', axis=1)
    
    # im = ax.imshow(confusion_matrix_plot.values, cmap='Blues', aspect='auto')
    # ax.set_xticks(np.arange(len(confusion_matrix_plot.columns)))
    # ax.set_yticks(np.arange(len(confusion_matrix_plot.index)))
    # ax.set_xticklabels(confusion_matrix_plot.columns, rotation=45, ha='right')
    # ax.set_yticklabels(confusion_matrix_plot.index)
    # ax.set_xlabel('Predicted Label')
    # ax.set_ylabel('GT Label')
    # ax.set_title('Confusion Matrix')
    
    # # Add text annotations
    # for i in range(len(confusion_matrix_plot.index)):
    #     for j in range(len(confusion_matrix_plot.columns)):
    #         text = ax.text(j, i, confusion_matrix_plot.iloc[i, j],
    #                       ha="center", va="center", color="black", fontsize=8)
    
    # plt.tight_layout()
    
    # Add evaluation results to calibration_results
    # Build cluster_inf dictionary for diagnosis (all clusters with position and waveform)
    cluster_inf = {}
    for cluster_id in range(n_clusters):
        if cluster_id in cluster_features:
            cluster_inf[cluster_id] = {
                'position_1': cluster_features[cluster_id]['position_1'],
                'position_2': cluster_features[cluster_id]['position_2'],
                'position_waveform': cluster_features[cluster_id]['position_waveform'],  # Mean waveform for compatibility
                'waveforms': cluster_features[cluster_id].get('waveforms', None),  # All waveforms (n_spikes, n_channels, time_window) - no mean
                'n_spikes': cluster_features[cluster_id]['n_spikes'],
                'matched_neuron': cluster_features[cluster_id].get('matched_neuron', 'unmatch'),
                'score': cluster_features[cluster_id].get('score', None),
                'pos_distance': cluster_features[cluster_id].get('pos_distance', None),
                'waveform_corr': cluster_features[cluster_id].get('waveform_corr', None),
            }
        else:
            # If cluster has no features (empty cluster), still record it
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            cluster_inf[cluster_id] = {
                'position_1': None,
                'position_2': None,
                'position_waveform': None,
                'waveforms': None,  # All waveforms (n_spikes, n_channels, time_window) - no mean
                'n_spikes': len(cluster_spike_indices),
                'matched_neuron': 'unmatch',
                'score': None,
                'pos_distance': None,
                'waveform_corr': None,
            }
    
    calibration_results = {
        'kmeans_model': kmeans,
        'pca_model': None,  # No PCA needed, way4 is already 30 dimensions
        'cluster_to_neuron_mapping': cluster_to_neuron_mapping,
        'neuron_to_clusters': dict(neuron_to_clusters),
        'neuron_cluster_comparison': neuron_cluster_comparison,  # {train_neuron: {cluster_id: {position, waveform, position_distance, waveform_corr, n_spikes}}}
        'cluster_features': cluster_features,
        'cluster_inf': cluster_inf,  # Add cluster_inf for diagnosis
        'spike_indices': spike_indices,
        'cluster_labels': cluster_labels,
        'results_df': results_df,  # Add results_df for confusion matrix
        'way4_features_30d': way4_features,  # Way4 features for spikes passing noise classifier (30 dimensions, no PCA needed)
        'whitening_matrix_W': W,  # Whitening matrix loaded from training data
        'whitening_matrix_M': M,  # Mean vector (or None) for whitening
        # 'way3_features_noise_100d': all_way3_features_noise,  # Way3 features for all detected spikes (100 dimensions, for noise detection visualization)
        # 'noise_gt_labels': np.array(all_noise_gt_labels),  # GT noise/spike labels
        # 'noise_pred_labels': np.array(all_noise_pred_labels),  # Predicted noise/spike labels
        # Detection statistics (detection vs GT)
        'detection_stats': detection_stats,  # Detection statistics: precision, recall, F1, etc.
        # Evaluation metrics
        'confusion_matrix_df': confusion_matrix_df,
        'classification_accuracy': classification_accuracy,  # All classes (including unmatch and noise)
        'unit_classification_accuracy': unit_classification_accuracy,  # Train neurons only (excluding unmatch and noise)
        'noise_detection_accuracy': noise_detection_accuracy,
        'noise_detection_accuracy_adjusted': noise_detection_accuracy_adjusted,
        'noise_detection_metrics': noise_detection_metrics,
        # 'umap_noise_df': umap_noise_df,
        # 'umap_label_df': umap_label_df,
        # 'visualization_figures': figs,  # List of matplotlib figures for UMAP
        # 'confusion_matrix_figure': fig_confusion,  # Matplotlib figure for confusion matrix
    }
    
    
    # Save evaluation results to files if requested
    if save_eval_results and eval_save_dir is not None:
        print("\n" + "=" * 80)
        print("Saving evaluation results to files...")
        print("=" * 80)
        save_calibration_evaluation_results(
            calibration_results=calibration_results,
            save_dir=eval_save_dir,
            run_name=run_name,
            date_str=date_str
        )
    
    # Provide simplified evaluation DataFrame (time, detect_channel, gt/pred noise & neuron labels)
    def _to_noise_label(lbl, train_list):
        if lbl in ['noise', 'unmatch', None, 'unknown']:
            return 'noise'
        if train_list is not None and lbl in train_list:
            return 'spike'
        return 'noise'

    simple_df = results_df.copy()
    train_neuron_list = train_neuron_inf['Neuron'].tolist() if train_neuron_inf is not None else []
    simple_df['gt_noise_label'] = simple_df['gt_label'].apply(lambda x: _to_noise_label(x, train_neuron_list))
    simple_df['predicted_noise_label'] = simple_df['predicted_label'].apply(lambda x: _to_noise_label(x, train_neuron_list))
    simple_df = simple_df.rename(columns={
        'spike_time': 'time',
        'spike_channel': 'detect_channel',
        'gt_label': 'gt_neuron_label',
        'predicted_label': 'predicted_neuron_label'
    })
    # 仅在非 unmatch 时用噪声标签覆盖神经元标签，保留 unmatch 类别
    mask_gt_noise = (simple_df['gt_noise_label'] == 'noise') & (simple_df['gt_neuron_label'] != 'unmatch')
    mask_pred_noise = (simple_df['predicted_noise_label'] == 'noise') & (simple_df['predicted_neuron_label'] != 'unmatch')
    simple_df.loc[mask_gt_noise, 'gt_neuron_label'] = 'noise'
    simple_df.loc[mask_pred_noise, 'predicted_neuron_label'] = 'noise'
    calibration_results['results_df_simple'] = simple_df[[
        'time',
        'detect_channel',
        'gt_noise_label',
        'predicted_noise_label',
        'gt_neuron_label',
        'predicted_neuron_label'
    ]]

    return calibration_results


def real_time_processing(
    recording_f,
    autosort_model: SimpleAutoSort,
    calibration_results: dict,
    start_time_seconds: float = 60.0,
    time_window_seconds: float = 10.0,
    total_duration_seconds: float = None,
    detect_threshold: float = 3.0,  # Mountainsort4: absolute threshold value
    detect_interval: int = 10,  # Mountainsort4: minimum interval between detections (samples)
    detect_sign: int = -1,  # Mountainsort4: -1 (negative peaks), 0 (both), 1 (positive peaks)
    margin: int = 0,  # Mountainsort4: margin to exclude from boundaries (samples)
    window_params: dict = None,
    eval_neuron_inf: pd.DataFrame = None,
    eval_spike_inf: pd.DataFrame = None,
    valid_channels=None,  # List of valid channel indices (clique channel indices) to detect on, if None detect on all channels
    device=None,
):
    """
    Stage 2: Real-time processing (process by time_window) using Mountainsort4 detection method
    
    Process:
    1. Load data by time_window
    2. Threshold detection (Mountainsort4)
    3. Pass through noise classifier, classified as spikes
    4. Extract way3 layer → PCA dimensionality reduction → K-means prediction → Map to train neuron ID
    
    Parameters:
        recording_f: preprocessed recording object
        autosort_model: trained SimpleAutoSort model
        calibration_results: results from calibration stage (contains kmeans_model, pca_model, cluster_to_neuron_mapping)
        start_time_seconds: start time for processing (seconds), default 60 (after calibration)
        time_window_seconds: length of each time window (seconds), default 10
        total_duration_seconds: total processing duration (seconds), if None process until recording ends, default None
        detect_threshold: float, Mountainsort4 absolute threshold value (default 3.0)
        detect_interval: int, Mountainsort4 minimum interval between detections (samples, default 10)
        detect_sign: int, Mountainsort4 detection sign: -1 (negative peaks), 0 (both), 1 (positive peaks), default -1
        margin: int, Mountainsort4 margin to exclude from boundaries (samples, default 0)
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
    
    if window_params is None:
        window_params = {
            'left_sample': 20,  # Window size (no downsampling)
            'right_sample': 40,  # Window size (no downsampling)
        }
    
    left_sample = window_params['left_sample']  # 20
    right_sample = window_params['right_sample']  # 40
    window_size = left_sample + right_sample  # 60
    n_channels = recording_f.get_num_channels()
    sampling_frequency = recording_f.get_sampling_frequency()
    
    # Get models and mapping from calibration stage
    kmeans_model = calibration_results['kmeans_model']
    pca_model = calibration_results.get('pca_model', None)  # May be None if using way4 features directly
    cluster_to_neuron_mapping = calibration_results['cluster_to_neuron_mapping']
    
    # Get whitening matrix from calibration stage (loaded from training data)
    W = calibration_results.get('whitening_matrix_W', None)
    M = calibration_results.get('whitening_matrix_M', None)
    if W is None:
        raise ValueError("Whitening matrix not found in calibration_results. Please ensure calibration_model loaded the whitening matrix from training data.")
    print(f"Using whitening matrix from calibration stage (shape: {W.shape})")
    
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
        print(f"Total processing duration: {total_duration_seconds} seconds (from {start_time_seconds}s to {end_time_seconds}s)")
    else:
        end_frame = total_duration_samples
        print(f"Processing until recording ends (from {start_time_seconds}s to {recording_total_seconds:.1f}s)")
    
    all_spike_predictions = []
    all_spike_times = []
    all_spike_channels = []
    all_way3_features_100d = []  # Way3 features for all spikes passing noise classifier (100 dimensions)
    all_way3_features_30d = []  # Features after PCA dimensionality reduction (30 dimensions)
    all_noise_way3_features_100d = []  # Way3 features for all detected spikes (100 dimensions, for noise detection visualization)
    all_noise_gt_labels_list = []  # GT noise/spike labels
    all_noise_pred_labels_list = []  # Predicted noise/spike labels
    
    autosort_model.eval()
    
    # Process by time_window
    current_start_frame = start_frame
    window_idx = 0
    
    while current_start_frame < end_frame:
        window_end_frame = min(current_start_frame + window_frames, total_duration_samples)
        window_duration = (window_end_frame - current_start_frame) / sampling_frequency
        
        print(f"\nProcessing window {window_idx + 1} ({current_start_frame/sampling_frequency:.1f}s - {window_end_frame/sampling_frequency:.1f}s)")
        
        # 1. Load current window data
        traces = recording_f.get_traces(start_frame=current_start_frame, end_frame=window_end_frame)
        if traces.shape[0] > traces.shape[1] and traces.shape[0] > 100:
            traces = traces.T
        traces = traces.astype(np.float32)
        
        # 1.5. Apply whitening using matrix from calibration stage
        # For real-time processing: use pre-computed whitening matrix from calibration (first 60s)
        # This avoids recomputing the whitening matrix for each small window
        traces_whitened = whiten_traces(traces, W, M)  # (n_samples, n_channels)
        
        # 2. Threshold detection using Mountainsort4 (on whitened data)
        trace0_car = traces_whitened.T  # (n_timepoints, n_channels) - whitened data
        spikes = detect_spike(
            trace0_car,
            detect_threshold=detect_threshold,
            detect_interval=detect_interval,
            detect_sign=detect_sign,
            margin=margin,
            valid_channels=valid_channels,
        )
        spike_coords = np.argwhere(spikes == 1)  # (n_spikes, 2) [time, channel]
        
        if len(spike_coords) == 0:
            print(f"  Window {window_idx + 1}: No spikes detected")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        # 3. Extract waveforms and filter boundaries
        waveforms = []
        valid_spike_coords = []
        spike_times = []
        spike_channels = []
        
        for time_idx, channel_idx in spike_coords:
            # Convert to global time indices
            global_time_idx = current_start_frame + time_idx
            local_start = time_idx - left_sample
            local_end = time_idx + right_sample
            
            if local_start < 0 or local_end > trace0_car.shape[0]:
                continue
            if local_end - local_start != window_size:
                continue
            
            # Extract waveform from whitened data (n_channels, window_size)
            waveform = traces_whitened[:, local_start:local_end]  # (n_channels, window_size) - whitened
            waveforms.append(waveform)
            valid_spike_coords.append((time_idx, channel_idx))
            spike_times.append(global_time_idx)
            spike_channels.append(channel_idx)
        
        if len(waveforms) == 0:
            print(f"  Window {window_idx + 1}: No valid spike waveforms")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size)
        
        # 4. Pass through noise classifier, classified as spikes
        batch_size = 4096
        n_spikes = len(waveforms)
        way3_features_list = []
        way3_spike_indices = []  # Record original spike indices corresponding to each way3 feature
        
        # Save way3 features for all spikes (for noise detection visualization)
        window_noise_way3_features = []
        window_noise_gt_labels = []
        window_noise_pred_labels = []
        
        with torch.no_grad():
            for i in range(0, n_spikes, batch_size):
                batch_end = min(i + batch_size, n_spikes)
                batch_waveforms = waveforms[i:batch_end]
                batch_channels = spike_channels[i:batch_end]
                
                # Extract single waveform and multi waveform
                batch_single_waveforms = []
                batch_multi_waveforms = []
                
                for wf, ch in zip(batch_waveforms, batch_channels):
                    multi_wf = wf.flatten()
                    batch_multi_waveforms.append(multi_wf)
                    single_wf = wf[ch, :]
                    batch_single_waveforms.append(single_wf)
                
                batch_multi_waveforms = np.array(batch_multi_waveforms)
                batch_single_waveforms = np.array(batch_single_waveforms)
                
                # Convert to tensor
                batch_multi = torch.from_numpy(batch_multi_waveforms).float().to(device)
                batch_single = torch.from_numpy(batch_single_waveforms).float().to(device)
                
                # Concatenate
                codes = torch.cat((batch_multi, batch_single), dim=1)
                
                # Noise classification
                noise_output = autosort_model.clsfier_noise(codes)
                noise_pred = torch.argmax(noise_output, dim=1)
                
                # Extract way3 features for all spikes (including those classified as noise)
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
                    way3_features_list.append(way3_batch.cpu().numpy())
        
        if len(way3_features_list) == 0:
            print(f"  Window {window_idx + 1}: No spikes passed noise classifier")
            current_start_frame = window_end_frame
            window_idx += 1
            continue
        
        # Combine features for all spikes
        # Note: intermediate_forward now returns way4 features (30 dimensions), not way3 (100 dimensions)
        way4_features = np.concatenate(way3_features_list, axis=0)  # (n_spikes_passed, 30) - way4 features are already 30 dimensions
        way3_spike_indices = np.array(way3_spike_indices)  # Corresponding original spike indices
        
        # Save way4 features (30 dimensions)
        all_way3_features_30d.append(way4_features)  # Keep variable name for compatibility
        
        # Save way4 features for all detected spikes (for noise detection visualization)
        if len(window_noise_way3_features) > 0:
            window_noise_way4_all = np.concatenate(window_noise_way3_features, axis=0)  # (n_all_spikes, 30) - way4 features
            all_noise_way3_features_100d.append(window_noise_way4_all)  # Keep variable name for compatibility
            all_noise_gt_labels_list.extend(window_noise_gt_labels)
            all_noise_pred_labels_list.extend(window_noise_pred_labels)
        
        # 5. K-means prediction (directly on way4 features, no PCA needed)
        
        # 6. K-means prediction (directly on way4 features, no PCA)
        cluster_labels = kmeans_model.predict(way4_features)  # (n_spikes_passed,)
        
        # 7. Map to train neuron ID
        neuron_predictions = []
        for cluster_id in cluster_labels:
            if cluster_id in cluster_to_neuron_mapping:
                neuron_predictions.append(cluster_to_neuron_mapping[cluster_id])
            else:
                neuron_predictions.append('unmatch')
        
        # Use way3_spike_indices to get corresponding spike times and channels
        valid_spike_times = [spike_times[i] for i in way3_spike_indices]
        valid_spike_channels = [spike_channels[i] for i in way3_spike_indices]
        valid_neuron_predictions = neuron_predictions  # Already corresponds to spikes passing noise classifier
        
        all_spike_predictions.extend(valid_neuron_predictions)
        all_spike_times.extend(valid_spike_times)
        all_spike_channels.extend(valid_spike_channels)
        
        print(f"  Window {window_idx + 1}: {len(valid_spike_times)} spikes")
        print(f"    - Matched neurons: {sum(1 for p in valid_neuron_predictions if p != 'unmatch')}")
        print(f"    - Unmatched: {sum(1 for p in valid_neuron_predictions if p == 'unmatch')}")
        
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
            eval_to_train_mapping = {}
            for _, row in eval_neuron_inf.iterrows():
                eval_neuron = row['Neuron']
                match = row['neuron_match']
                if match != 'unmatch':
                    eval_to_train_mapping[eval_neuron] = match
                else:
                    eval_to_train_mapping[eval_neuron] = 'unmatch'
            
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
                        gt_label = 'unmatch'
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
    if len(all_way3_features_100d) > 0:
        all_way3_features_100d_combined = np.concatenate(all_way3_features_100d, axis=0)
        all_way3_features_30d_combined = np.concatenate(all_way3_features_30d, axis=0)
    else:
        all_way3_features_100d_combined = np.array([])
        all_way3_features_30d_combined = np.array([])
    
    if len(all_noise_way3_features_100d) > 0:
        all_noise_way3_features_100d_combined = np.concatenate(all_noise_way3_features_100d, axis=0)
        all_noise_gt_labels_combined = np.array(all_noise_gt_labels_list)
        all_noise_pred_labels_combined = np.array(all_noise_pred_labels_list)
    else:
        all_noise_way3_features_100d_combined = np.array([])
        all_noise_gt_labels_combined = np.array([])
        all_noise_pred_labels_combined = np.array([])
    
    processing_results = {
        'spike_predictions': all_spike_predictions,
        'spike_times': all_spike_times,
        'spike_channels': all_spike_channels,
        'results_df': results_df,
        'way3_features_100d': all_way3_features_100d_combined,  # Way3 features for spikes passing noise classifier (100 dimensions)
        'way3_features_30d': all_way3_features_30d_combined,  # Features after PCA dimensionality reduction (30 dimensions)
        'way3_features_noise_100d': all_noise_way3_features_100d_combined,  # Way3 features for all detected spikes (100 dimensions, for noise detection visualization)
        'noise_gt_labels': all_noise_gt_labels_combined,  # GT noise/spike labels
        'noise_pred_labels': all_noise_pred_labels_combined,  # Predicted noise/spike labels
    }

    # 提供简化的评估结果 DataFrame：time, detect_channel, gt_noise_label, predicted_noise_label, gt_neuron_label, predicted_neuron_label
    def _to_noise_label(lbl, train_neuron_list):
        if lbl in ['noise', 'unmatch', None, 'unknown']:
            return 'noise'
        if train_neuron_list is not None and lbl in train_neuron_list:
            return 'spike'
        return 'noise'

    simple_df = results_df.copy()
    # Try to get train neuron list from calibration mapping (fall back to empty)
    train_neuron_list = []
    if calibration_results and isinstance(calibration_results, dict):
        cluster_to_neuron_mapping = calibration_results.get('cluster_to_neuron_mapping', {})
        train_neuron_list = list(set(cluster_to_neuron_mapping.values()))
    simple_df['gt_noise_label'] = simple_df['gt_label'].apply(lambda x: _to_noise_label(x, train_neuron_list))
    simple_df['predicted_noise_label'] = simple_df['predicted_label'].apply(lambda x: _to_noise_label(x, train_neuron_list))
    simple_df = simple_df.rename(columns={
        'spike_time': 'time',
        'spike_channel': 'detect_channel',
        'gt_label': 'gt_neuron_label',
        'predicted_label': 'predicted_neuron_label'
    })
    mask_gt_noise = (simple_df['gt_noise_label'] == 'noise') & (simple_df['gt_neuron_label'] != 'unmatch')
    mask_pred_noise = (simple_df['predicted_noise_label'] == 'noise') & (simple_df['predicted_neuron_label'] != 'unmatch')
    simple_df.loc[mask_gt_noise, 'gt_neuron_label'] = 'noise'
    simple_df.loc[mask_pred_noise, 'predicted_neuron_label'] = 'noise'
    processing_results['results_df_simple'] = simple_df[[
        'time',
        'detect_channel',
        'gt_noise_label',
        'predicted_noise_label',
        'gt_neuron_label',
        'predicted_neuron_label'
    ]]
    
    print(f"\nProcessing completed:")
    print(f"  - Total spikes: {len(all_spike_predictions)}")
    print(f"  - Matched neurons: {sum(1 for p in all_spike_predictions if p != 'unmatch')}")
    print(f"  - Unmatched: {sum(1 for p in all_spike_predictions if p == 'unmatch')}")
    
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
    # Original: GT=noise -> noise, GT=train_neuron or unmatch -> spike
    def get_gt_noise_label(gt_label):
        if gt_label == 'noise':
            return 'noise'
        elif gt_label in train_neuron_list or gt_label == 'unmatch':
            return 'spike'
        else:
            return 'unknown'
    
    # Adjusted: GT=noise or unmatch -> noise (treat unmatch as noise)
    def get_gt_noise_label_adjusted(gt_label):
        if gt_label == 'noise' or gt_label == 'unmatch':
            return 'noise'
        elif gt_label in train_neuron_list:
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
    
    # Calculate adjusted metrics (treat unmatch as noise in GT)
    noise_detection_df_adjusted = results_df.copy()
    noise_detection_df_adjusted['gt_noise_adjusted'] = noise_detection_df_adjusted['gt_label'].apply(get_gt_noise_label_adjusted)
    noise_detection_df_adjusted['pred_noise'] = noise_detection_df_adjusted['predicted_label'].apply(get_pred_noise_label)
    
    # Filter out unknown samples
    noise_detection_df_adjusted = noise_detection_df_adjusted[
        (noise_detection_df_adjusted['gt_noise_adjusted'] != 'unknown') & 
        (noise_detection_df_adjusted['pred_noise'] != 'unknown')
    ]
    
    # Calculate confusion matrix (adjusted)
    confusion_matrix_adjusted = pd.crosstab(
        noise_detection_df_adjusted['gt_noise_adjusted'],
        noise_detection_df_adjusted['pred_noise'],
        margins=True
    )
    
    # Ensure there are noise and spike rows and columns
    for label in ['noise', 'spike']:
        if label not in confusion_matrix_adjusted.index:
            confusion_matrix_adjusted.loc[label] = 0
        if label not in confusion_matrix_adjusted.columns:
            confusion_matrix_adjusted[label] = 0
    
    # Reorder
    confusion_matrix_adjusted = confusion_matrix_adjusted.reindex(
        index=['noise', 'spike', 'All'] if 'All' in confusion_matrix_adjusted.index else ['noise', 'spike'],
        columns=['noise', 'spike', 'All'] if 'All' in confusion_matrix_adjusted.columns else ['noise', 'spike'],
        fill_value=0
    )
    
    # Calculate TP, TN, FP, FN (adjusted)
    TP_adj = confusion_matrix_adjusted.loc['spike', 'spike'] if 'spike' in confusion_matrix_adjusted.index and 'spike' in confusion_matrix_adjusted.columns else 0
    TN_adj = confusion_matrix_adjusted.loc['noise', 'noise'] if 'noise' in confusion_matrix_adjusted.index and 'noise' in confusion_matrix_adjusted.columns else 0
    FP_adj = confusion_matrix_adjusted.loc['noise', 'spike'] if 'noise' in confusion_matrix_adjusted.index and 'spike' in confusion_matrix_adjusted.columns else 0
    FN_adj = confusion_matrix_adjusted.loc['spike', 'noise'] if 'spike' in confusion_matrix_adjusted.index and 'noise' in confusion_matrix_adjusted.columns else 0
    
    # Calculate adjusted accuracy
    total_adj = TP_adj + TN_adj + FP_adj + FN_adj
    accuracy_adjusted = (TP_adj + TN_adj) / total_adj if total_adj > 0 else 0.0
    
    metrics = {
        'confusion_matrix': confusion_matrix,
        'confusion_matrix_adjusted': confusion_matrix_adjusted,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TP_adjusted': TP_adj,
        'TN_adjusted': TN_adj,
        'FP_adjusted': FP_adj,
        'FN_adjusted': FN_adj,
        'accuracy': accuracy,
        'accuracy_adjusted': accuracy_adjusted,  # Adjusted accuracy (treat unmatch as noise in GT)
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'total_samples': total,
    }
    
    return metrics


def visualize_umap_features(
    way3_features_100d: np.ndarray,
    way3_features_30d: np.ndarray,
    results_df: pd.DataFrame,
    train_neuron_list: list,
    noise_gt_labels: np.ndarray = None,
    noise_pred_labels: np.ndarray = None,
    neuron_inf_color: dict = None,
    n_samples: int = 50000,
    random_state: int = 42,
):
    """
    Visualize UMAP dimensionality reduction results for way3 features
    
    Parameters:
        way3_features_100d: way3 layer 100-dimensional features (for noise detection visualization)
        way3_features_30d: 30-dimensional features after PCA dimensionality reduction (for label classification visualization)
        results_df: DataFrame containing gt_label and predicted_label (for label classification)
        train_neuron_list: train neuron list
        noise_gt_labels: GT noise/spike label array (0=noise, 1=spike), for noise detection visualization
        noise_pred_labels: predicted noise/spike label array (0=noise, 1=spike), for noise detection visualization
        neuron_inf_color: neuron color mapping dictionary, format {neuron_name: color}
        n_samples: number of randomly sampled samples, default 50000
        random_state: random seed
    
    Returns:
        figs: list of matplotlib figure objects, containing 4 independent figures (each is square):
            - figs[0]: Noise Detection - GT Label
            - figs[1]: Noise Detection - Predicted Label
            - figs[2]: Label Classification - GT Label
            - figs[3]: Label Classification - Predicted Label
            if a figure has no data, corresponding position is None
    """
    import umap
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Set random seed
    np.random.seed(random_state)
    
    # 1. UMAP visualization for Noise Detection
    print("=" * 50)
    print("1. Noise Detection UMAP Visualization")
    print("=" * 50)
    
    # Prepare data for noise detection
    if len(way3_features_100d) == 0:
        print("Warning: no way3 feature data for noise detection visualization")
        noise_umap_coords = None
        noise_gt_labels_plot = None
        noise_pred_labels_plot = None
    else:
        # Random sampling
        n_total = len(way3_features_100d)
        n_sample = min(n_samples, n_total)
        sample_indices = np.random.choice(n_total, n_sample, replace=False)
        way3_features_noise_sample = way3_features_100d[sample_indices]
        
        # PCA dimensionality reduction to 30 dimensions
        from sklearn.decomposition import PCA
        pca_noise = PCA(n_components=30)
        way3_features_noise_30d = pca_noise.fit_transform(way3_features_noise_sample)
        
        # UMAP dimensionality reduction to 2 dimensions
        print(f"  Performing UMAP dimensionality reduction ({n_sample} samples)...")
        reducer_noise = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        noise_umap_coords = reducer_noise.fit_transform(way3_features_noise_30d)
        
        # Get GT and predicted labels
        if noise_gt_labels is not None and len(noise_gt_labels) == n_total:
            noise_gt_labels_plot = noise_gt_labels[sample_indices]
        else:
            noise_gt_labels_plot = None
        
        if noise_pred_labels is not None and len(noise_pred_labels) == n_total:
            noise_pred_labels_plot = noise_pred_labels[sample_indices]
        else:
            noise_pred_labels_plot = None
    
    # 2. UMAP visualization for Label Classification
    print("\n" + "=" * 50)
    print("2. Label Classification UMAP Visualization")
    print("=" * 50)
    
    if len(way3_features_30d) == 0:
        print("Warning: no way3 feature data for label classification visualization")
        label_umap_coords = None
        label_gt_labels = None
        label_pred_labels = None
    else:
        # First filter: only keep points where gt label is not unmatch/noise and predicted label is not unmatch
        valid_indices = []
        valid_gt_labels = []
        valid_pred_labels = []
        
        for idx in range(len(way3_features_30d)):
            if idx < len(results_df):
                gt_label = results_df.iloc[idx]['gt_label']
                pred_label = results_df.iloc[idx]['predicted_label']
                
                # Only keep points where gt label is not unmatch/noise and predicted label is not unmatch
                if (gt_label not in ['unmatch', 'noise', 'unknown', None]) and (pred_label != 'unmatch'):
                    valid_indices.append(idx)
                    valid_gt_labels.append(gt_label)
                    valid_pred_labels.append(pred_label)
        
        if len(valid_indices) == 0:
            print("Warning: no valid points for label classification visualization (all points filtered)")
            label_umap_coords = None
            label_gt_labels = None
            label_pred_labels = None
        else:
            print(f"  Valid points after filtering: {len(valid_indices)} / {len(way3_features_30d)}")
            
            # Use filtered indices
            valid_indices = np.array(valid_indices)
            way3_features_label_filtered = way3_features_30d[valid_indices]
            
            # Random sampling (sample from filtered points)
            n_total_filtered = len(way3_features_label_filtered)
            n_sample = min(n_samples, n_total_filtered)
            sample_indices = np.random.choice(n_total_filtered, n_sample, replace=False)
            way3_features_label_sample = way3_features_label_filtered[sample_indices]
            
            # Get corresponding labels
            label_gt_labels = [valid_gt_labels[i] for i in sample_indices]
            label_pred_labels = [valid_pred_labels[i] for i in sample_indices]
            
            # UMAP dimensionality reduction to 2 dimensions
            print(f"  Performing UMAP dimensionality reduction ({n_sample} samples)...")
            reducer_label = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
            label_umap_coords = reducer_label.fit_transform(way3_features_label_sample)
    
    # 3. Plot figures (four independent figures, each is square)
    # Define colors
    spike_color = 'orange'  # Orange
    noise_color = 'lightgray'  # Light gray
    
    # If neuron_inf_color is provided, use it; otherwise generate default colors
    if neuron_inf_color is None:
        # Generate default colors (using tab20 color map)
        cmap = plt.cm.get_cmap('tab20')
        neuron_inf_color = {}
        for i, neuron in enumerate(train_neuron_list):
            neuron_inf_color[neuron] = cmap(i % 20)
    
    figs = []
    
    # Figure 1: Noise Detection - GT Label
    if noise_umap_coords is not None and noise_gt_labels_plot is not None:
        fig1 = plt.figure(figsize=(8, 8))  # Square
        ax1 = fig1.add_subplot(1, 1, 1)
        # Plot noise
        noise_mask = noise_gt_labels_plot == 0
        if noise_mask.sum() > 0:
            ax1.scatter(noise_umap_coords[noise_mask, 0], noise_umap_coords[noise_mask, 1], 
                       c=noise_color, s=0.1, alpha=0.5, label='Noise')
        # Plot spike
        spike_mask = noise_gt_labels_plot == 1
        if spike_mask.sum() > 0:
            ax1.scatter(noise_umap_coords[spike_mask, 0], noise_umap_coords[spike_mask, 1], 
                       c=spike_color, s=0.1, alpha=0.5, label='Spike')
        #ax1.set_title('Noise Detection: GT Label', fontsize=14, fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.grid(False)
        ax1.set_aspect('equal', adjustable='box')
        # Ensure subplot occupies entire figure and remains square
        ax1.set_position([0, 0, 1, 1])
        figs.append(fig1)
    else:
        figs.append(None)
    
    # Figure 2: Noise Detection - Predicted Label
    if noise_umap_coords is not None and noise_pred_labels_plot is not None:
        fig2 = plt.figure(figsize=(8, 8))  # Square
        ax2 = fig2.add_subplot(1, 1, 1)
        # Plot noise
        noise_mask = noise_pred_labels_plot == 0
        if noise_mask.sum() > 0:
            ax2.scatter(noise_umap_coords[noise_mask, 0], noise_umap_coords[noise_mask, 1], 
                       c=noise_color, s=0.1, alpha=0.5, label='Noise')
        # Plot spike
        spike_mask = noise_pred_labels_plot == 1
        if spike_mask.sum() > 0:
            ax2.scatter(noise_umap_coords[spike_mask, 0], noise_umap_coords[spike_mask, 1], 
                       c=spike_color, s=0.1, alpha=0.5, label='Spike')
        #ax2.set_title('Noise Detection: Predicted Label', fontsize=14, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.grid(False)
        ax2.set_aspect('equal', adjustable='box')
        # Ensure subplot occupies entire figure and remains square
        ax2.set_position([0, 0, 1, 1])
        figs.append(fig2)
    else:
        figs.append(None)
    
    # Figure 3: Label Classification - GT Label
    if label_umap_coords is not None and label_gt_labels is not None:
        fig3 = plt.figure(figsize=(8, 8))  # Square
        ax3 = fig3.add_subplot(1, 1, 1)
        # Plot each neuron
        for neuron in train_neuron_list:
            mask = np.array([l == neuron for l in label_gt_labels])
            if mask.sum() > 0:
                ax3.scatter(label_umap_coords[mask, 0], label_umap_coords[mask, 1], 
                           c=[neuron_inf_color[neuron]], s=1, alpha=0.5, label=neuron)
        # Plot unmatch and noise
        unmatch_mask = np.array([l == 'unmatch' or l == 'noise' for l in label_gt_labels])
        if unmatch_mask.sum() > 0:
            ax3.scatter(label_umap_coords[unmatch_mask, 0], label_umap_coords[unmatch_mask, 1], 
                       c=noise_color, s=0.1, alpha=0.5, label='Unmatch/Noise')
        #ax3.set_title('Label Classification: GT Label', fontsize=14, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.grid(False)
        ax3.set_aspect('equal', adjustable='box')
        # Ensure subplot occupies entire figure and remains square
        ax3.set_position([0, 0, 1, 1])
        figs.append(fig3)
    else:
        figs.append(None)
    
    # Figure 4: Label Classification - Predicted Label
    if label_umap_coords is not None and label_pred_labels is not None:
        fig4 = plt.figure(figsize=(8, 8))  # Square
        ax4 = fig4.add_subplot(1, 1, 1)
        # Plot each neuron
        for neuron in train_neuron_list:
            mask = np.array([l == neuron for l in label_pred_labels])
            if mask.sum() > 0:
                ax4.scatter(label_umap_coords[mask, 0], label_umap_coords[mask, 1], 
                           c=[neuron_inf_color[neuron]], s=1, alpha=0.5, label=neuron)
        # Plot unmatch and noise
        unmatch_mask = np.array([l == 'unmatch' or l == 'noise' for l in label_pred_labels])
        if unmatch_mask.sum() > 0:
            ax4.scatter(label_umap_coords[unmatch_mask, 0], label_umap_coords[unmatch_mask, 1], 
                       c=noise_color, s=0.1, alpha=0.5, label='Unmatch/Noise')
        #ax4.set_title('Label Classification: Predicted Label', fontsize=14, fontweight='bold')
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.grid(False)
        ax4.set_aspect('equal', adjustable='box')
        # Ensure subplot occupies entire figure and remains square
        ax4.set_position([0, 0, 1, 1])
        figs.append(fig4)
    else:
        figs.append(None)
    
    return figs


def save_calibration_evaluation_results(
    calibration_results: dict,
    save_dir: str,
    run_name: str = "run_1",
    date_str: str = None,
):
    """
    Save calibration evaluation results to files (same format as reference eval_results directory)
    
    Parameters:
        calibration_results: dict returned from calibration_model, containing evaluation metrics
        save_dir: directory to save results (e.g., "output/clique_XX/model_save/run_X/eval")
        run_name: run name (e.g., "run_1")
        date_str: date string for file naming (e.g., "022522"), if None will use current date
    """
    from pathlib import Path
    from datetime import datetime
    import matplotlib.pyplot as plt
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate date string if not provided
    if date_str is None:
        date_str = datetime.now().strftime("%m%d%y")
    
    # File names
    confusion_matrix_csv = save_dir / f"confusion_matrix_{date_str}.csv"
    classification_accuracy_csv = save_dir / f"classification_accuracy_{date_str}.csv"
    noise_detection_accuracy_csv = save_dir / f"noise_detection_accuracy_{date_str}.csv"
    detection_stats_csv = save_dir / f"detection_stats_{date_str}.csv"  # Detection vs GT statistics
    all_runs_results_csv = save_dir / f"all_runs_results_{date_str}.csv"
    umap_label_csv = save_dir / f"umap_label_classification_{date_str}.csv"
    umap_noise_csv = save_dir / f"umap_noise_detection_{date_str}.csv"
    confusion_matrix_pdf = save_dir / f"confusion_matrix_{date_str}.pdf"
    umap_visualization_pdf = save_dir / f"umap_visualization_{date_str}.pdf"
    
    # 1. Save confusion matrix
    if 'confusion_matrix_df' in calibration_results:
        calibration_results['confusion_matrix_df'].to_csv(confusion_matrix_csv)
        print(f"  Saved confusion matrix to: {confusion_matrix_csv}")
    
    # 2. Save classification accuracy (all classes and unit only)
    if 'classification_accuracy' in calibration_results:
        classification_accuracy_data = {
            'classification_accuracy': [calibration_results['classification_accuracy']]  # All classes (including unmatch and noise)
        }
        if 'unit_classification_accuracy' in calibration_results:
            classification_accuracy_data['unit_classification_accuracy'] = [calibration_results['unit_classification_accuracy']]  # Train neurons only (excluding unmatch and noise)
        classification_accuracy_df = pd.DataFrame(classification_accuracy_data)
        classification_accuracy_df.to_csv(classification_accuracy_csv, index=False)
        print(f"  Saved classification accuracy to: {classification_accuracy_csv}")
    
    # 3. Save noise detection accuracy
    if 'noise_detection_accuracy' in calibration_results and 'noise_detection_accuracy_adjusted' in calibration_results:
        noise_detection_accuracy_df = pd.DataFrame({
            'noise_detection_accuracy': [calibration_results['noise_detection_accuracy']],
            'noise_detection_accuracy_adjusted': [calibration_results['noise_detection_accuracy_adjusted']]
        })
        noise_detection_accuracy_df.to_csv(noise_detection_accuracy_csv, index=False)
        print(f"  Saved noise detection accuracy to: {noise_detection_accuracy_csv}")
    
    # 3.5. Save detection statistics (detection vs GT)
    if 'detection_stats' in calibration_results and calibration_results['detection_stats'] is not None:
        detection_stats = calibration_results['detection_stats']
        detection_stats_df = pd.DataFrame({
            'spike_detection_rate': [detection_stats['spike_detection_rate']],
            'n_matched': [detection_stats['n_matched']],
            'n_unmatched': [detection_stats['n_unmatched']],
        })
        detection_stats_df.to_csv(detection_stats_csv, index=False)
        print(f"  Saved detection statistics to: {detection_stats_csv}")
    
    # 4. Save all runs results (single row for current run)
    if 'classification_accuracy' in calibration_results and 'noise_detection_accuracy' in calibration_results:
        all_runs_results_data = {
            'run': [run_name],
            'classification_accuracy': [calibration_results['classification_accuracy']],  # All classes (including unmatch and noise)
            'noise_detection_accuracy': [calibration_results['noise_detection_accuracy']],
            'noise_detection_accuracy_adjusted': [calibration_results.get('noise_detection_accuracy_adjusted', 0.0)]
        }
        if 'unit_classification_accuracy' in calibration_results:
            all_runs_results_data['unit_classification_accuracy'] = [calibration_results['unit_classification_accuracy']]  # Train neurons only (excluding unmatch and noise)
        all_runs_results_df = pd.DataFrame(all_runs_results_data)
        all_runs_results_df.to_csv(all_runs_results_csv, index=False)
        print(f"  Saved all runs results to: {all_runs_results_csv}")
    
    # 5. Save UMAP data
    if 'umap_label_df' in calibration_results and calibration_results['umap_label_df'] is not None:
        calibration_results['umap_label_df'].to_csv(umap_label_csv, index=False)
        print(f"  Saved UMAP label classification data to: {umap_label_csv}")
    
    if 'umap_noise_df' in calibration_results and calibration_results['umap_noise_df'] is not None:
        calibration_results['umap_noise_df'].to_csv(umap_noise_csv, index=False)
        print(f"  Saved UMAP noise detection data to: {umap_noise_csv}")
    
    # 6. Save cluster_inf.pkl for diagnosis
    if 'cluster_inf' in calibration_results:
        import pickle
        cluster_inf_pkl = save_dir / f"cluster_inf_{date_str}.pkl"
        with open(cluster_inf_pkl, 'wb') as f:
            pickle.dump(calibration_results['cluster_inf'], f)
        print(f"  Saved cluster information to: {cluster_inf_pkl}")
        print(f"    Total clusters: {len(calibration_results['cluster_inf'])}")
        matched_count = sum(1 for c in calibration_results['cluster_inf'].values() if c.get('matched_neuron') != 'unmatch')
        print(f"    Matched clusters: {matched_count}")
        print(f"    Unmatched clusters: {len(calibration_results['cluster_inf']) - matched_count}")
    
    # 7. Save PDF figures
    if 'confusion_matrix_figure' in calibration_results and calibration_results['confusion_matrix_figure'] is not None:
        calibration_results['confusion_matrix_figure'].savefig(confusion_matrix_pdf, bbox_inches='tight', dpi=150)
        print(f"  Saved confusion matrix figure to: {confusion_matrix_pdf}")
    
    if 'visualization_figures' in calibration_results and calibration_results['visualization_figures'] is not None:
        # Combine all UMAP figures into one PDF
        figs = calibration_results['visualization_figures']
        valid_figs = [f for f in figs if f is not None]
        if len(valid_figs) > 0:
            # Save first figure as PDF (or combine multiple figures)
            # For simplicity, save the first valid figure
            valid_figs[0].savefig(umap_visualization_pdf, bbox_inches='tight', dpi=150)
            print(f"  Saved UMAP visualization figure to: {umap_visualization_pdf}")
            # Close figures to free memory
            for fig in valid_figs:
                plt.close(fig)
    
    if 'confusion_matrix_figure' in calibration_results and calibration_results['confusion_matrix_figure'] is not None:
        plt.close(calibration_results['confusion_matrix_figure'])
    
    print(f"\nAll evaluation results saved to: {save_dir}")


