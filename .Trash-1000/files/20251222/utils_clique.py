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
from sklearn.model_selection import train_test_split

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


def detect_spike_no_whiten(
    trace0_car,
    thr_min=3,
    thr_max=30,
    distance=3,
    ch_max_simul_firing=5,
    wlen=5,
    prominence=10,
    valid_channels=None,
):
    """
    AutoSort threshold detection function (no whitening required, identical to utils_clean.py detect_spike)
    
    Parameters:
        trace0_car: numpy array, shape (n_timepoints, n_channels)
        thr_min: minimum threshold multiplier (relative to noise std), default 3
        thr_max: maximum threshold multiplier (for filtering outliers), default 30
        distance: minimum distance between peaks (samples), default 3
        ch_max_simul_firing: maximum number of simultaneous firing channels, default 5
        wlen: window length for peak detection, default 5
        prominence: minimum peak prominence, default 10
        valid_channels: list of valid channel indices to detect on, if None detect on all channels
    
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

        # no simultanous firing!!!!
        thres_cross = ch_max_simul_firing
        spikes[np.sum(spikes, axis=1) > thres_cross, :] = 0
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

def extract_waveforms(trace0_car, X_spiketrain_time, left_sample=10, right_sample=20):
    """
    Extract waveform window
    
    Parameters:
        trace0_car: numpy array, shape (n_timepoints, n_channels) - whitened data
        X_spiketrain_time: numpy array, shape (n_spikes,), spike time points
        left_sample: number of samples before spike to extract, default 10
        right_sample: number of samples after spike to extract, default 20
    
    Returns:
        waveform: numpy array, shape (n_spikes, n_channels, left_sample + right_sample)
        valid_mask: boolean array indicating valid spikes
    """
    # Filter spikes near boundaries (ensure complete window can be extracted)
    window_size = left_sample + right_sample
    valid_mask = (X_spiketrain_time >= left_sample) & (X_spiketrain_time < trace0_car.shape[0] - right_sample)
    X_spiketrain_time = X_spiketrain_time[valid_mask]
    
    # Extract full window
    waveform = np.zeros((len(X_spiketrain_time), trace0_car.shape[1], window_size), dtype=np.float32)
    
    for i, time_range in enumerate(tqdm(np.arange(-left_sample, right_sample), desc="Extracting waveforms")):
        waveform[:, :, i] = trace0_car[X_spiketrain_time + time_range, :]
    
    # waveform shape: (n_spikes, n_channels, left_sample + right_sample)
    return waveform, valid_mask

def prepare_training_data_no_whiten(
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
    valid_channels=None,
):
    """
    Prepare training data using AutoSort threshold detection (no whitening required, identical to utils_clean.py prepare_training_data)
    
    This function performs clique-level detection. It receives a recording_clique (subset of channels)
    and performs detection, matching, and waveform extraction on this clique without whitening preprocessing.
    
    Parameters:
        recording_f: preprocessed recording object (should be recording_clique for clique-level processing)
        spike_inf: DataFrame containing GT spike information (filtered to neurons in clique)
        neuron_inf: DataFrame containing neuron information (filtered to neurons in clique)
        save_dir: save directory path
        duration_seconds: processing duration (seconds), default 200
        thr_min, thr_max, distance, ch_max_simul_firing, wlen, prominence: detection parameters
        left_sample, right_sample: waveform window parameters
        valid_channels: list of valid channels, if None automatically computed from neuron_inf's tract_channel column
    
    Returns:
        train_data_dir: training data save directory
    """
    print("### 1. Threshold Detection (No Whitening Method)")
    
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
    
    # Calculate and save valid_channel list
    # IMPORTANT: valid_channels should be clique channel indices, not probe device indices
    # Get recording channel IDs first to build mapping
    recording_channel_ids = recording_f.get_channel_ids()
    probe_to_clique_index = {}
    for clique_idx, probe_ch in enumerate(recording_channel_ids):
        try:
            probe_ch_int = int(probe_ch)
            probe_to_clique_index[probe_ch_int] = clique_idx
        except (ValueError, TypeError):
            probe_to_clique_index[probe_ch] = clique_idx
    
    if valid_channels is None:
        # If not provided, automatically compute from neuron_inf
        if 'tract_channel' in neuron_inf.columns:
            # Get probe device channel indices
            tract_channels_probe = sorted(neuron_inf['tract_channel'].unique().tolist())
            # Map to clique channel indices
            valid_channels = []
            for probe_ch in tract_channels_probe:
                if probe_ch in probe_to_clique_index:
                    valid_channels.append(probe_to_clique_index[probe_ch])
            valid_channels = sorted(valid_channels)
        else:
            valid_channels = None
            print("Warning: neuron_inf doesn't have tract_channel column, will detect on all channels")
    else:
        # If valid_channels is provided, assume they are probe device indices and map to clique indices
        valid_channels_probe = valid_channels
        valid_channels = []
        for probe_ch in valid_channels_probe:
            if probe_ch in probe_to_clique_index:
                valid_channels.append(probe_to_clique_index[probe_ch])
        valid_channels = sorted(valid_channels)
    
    if valid_channels is not None:
        print(f"Number of valid channels: {len(valid_channels)}")
        print(f"Valid channels list (clique channel indices): {valid_channels}")
    
    # Read data (no whitening)
    trace0_car = recording_f.get_traces(start_frame=0, end_frame=actual_frames).astype(np.float32)
    print(f"Data shape: {trace0_car.shape}")
    
    # Use AutoSort's detect_spike_no_whiten function (detect only on valid channels)
    spikes = detect_spike_no_whiten(
        trace0_car,
        thr_min=thr_min,
        thr_max=thr_max,
        distance=distance,
        ch_max_simul_firing=ch_max_simul_firing,
        wlen=wlen,
        prominence=prominence,
        valid_channels=valid_channels,
    )
    
    # Build detect_array
    print("Building detect_array...")
    all_spike_train = []
    spike_loc = []
    for channel_num in range(trace0_car.shape[1]):
        spiketrain_loc = np.where(spikes[:, channel_num])[0]
        all_spike_train += list(spiketrain_loc)
        spike_loc += [channel_num] * len(spiketrain_loc)
    
    X_spiketrain_time = np.array(all_spike_train)
    Y_spiketrain_id_final = np.array(spike_loc)
    detect_array = np.array([X_spiketrain_time, Y_spiketrain_id_final]).T
    
    print(f"Number of detected spikes: {len(detect_array)}")
    
    print("\n### 2. Load Ground Truth and Match")
    
    # Filter spike_inf, keep only data within specified duration
    spike_inf_filtered = spike_inf[spike_inf['time'] < max_frames].copy()
    
    # Build gt_array
    # IMPORTANT: Need to map tract_channel (probe device channel index) to clique channel index
    # because detect_array uses clique channel indices (0-based indices in the clique)
    print("Building gt_array...")
    
    # Get recording channel IDs (these are the probe device channel indices after rename_channels)
    recording_channel_ids = recording_f.get_channel_ids()
    # Build mapping: probe device channel index -> clique channel index
    # recording_clique's channel IDs were renamed to sorted_device_indices (probe device indices)
    # So: clique channel index i corresponds to recording_channel_ids[i] (probe device index)
    # We need the reverse mapping: probe device index -> clique channel index
    probe_to_clique_index = {}
    for clique_idx, probe_ch in enumerate(recording_channel_ids):
        try:
            probe_ch_int = int(probe_ch)
            probe_to_clique_index[probe_ch_int] = clique_idx
        except (ValueError, TypeError):
            # If channel ID is not a simple integer, try to convert
            probe_to_clique_index[probe_ch] = clique_idx
    
    spike_train_all = []
    y_unit_id = []
    gt_ch = []
    
    for neuron_idx in range(len(neuron_inf)):
        neuron_name = neuron_inf['Neuron'].iloc[neuron_idx]
        neuron_channel_id_probe = neuron_inf['tract_channel'].iloc[neuron_idx]  # Probe device channel index
        
        # Map probe device channel index to clique channel index
        if neuron_channel_id_probe in probe_to_clique_index:
            neuron_channel_id_clique = probe_to_clique_index[neuron_channel_id_probe]
        else:
            # If tract_channel is not in this clique, skip this neuron's spikes
            continue
        
        neuron_spikes = spike_inf_filtered[spike_inf_filtered['neuron'] == neuron_name]
        if len(neuron_spikes) > 0:
            spike_times = neuron_spikes['time'].values
            spike_train_all += list(spike_times)
            y_unit_id += [neuron_name] * len(spike_times)
            gt_ch += [neuron_channel_id_clique] * len(spike_times)  # Use clique channel index
    
    gt_array = np.array([spike_train_all, gt_ch]).T
    print(f"GT spike count: {len(gt_array)}")
    
    # Use AutoSort's map_gt_annotation function (no neuron_to_channel_id needed for old format)
    gt_label_array1 = map_gt_annotation(detect_array, gt_array, time_tolerance=2)
    
    # Calculate detection rate
    detection_rate = np.where(gt_label_array1 > -1)[0].shape[0] / gt_array.shape[0] if gt_array.shape[0] > 0 else 0.0
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
    
    # Calculate per-neuron matching statistics
    print("\n---Per-neuron matching statistics:")
    # y_unit_id already contains neuron names for each GT spike in the same order as gt_array
    # So we can directly use it to map GT index to neuron name
    y_unit_id_array = np.array(y_unit_id, dtype=object)
    
    # Count GT spikes and matched spikes for each neuron
    neuron_stats = {}
    for neuron_name in neuron_inf['Neuron'].unique():
        # Count GT spikes for this neuron
        neuron_gt_mask = y_unit_id_array == neuron_name
        neuron_gt_count = neuron_gt_mask.sum()
        
        if neuron_gt_count > 0:
            # Get GT indices for this neuron
            neuron_gt_indices = np.where(neuron_gt_mask)[0]
            
            # Count how many of these GT spikes were matched
            # gt_label_array1[matched_indices] contains the GT indices that were matched
            matched_gt_indices = gt_label_array1[matched_indices]
            neuron_matched_count = sum(1 for gt_idx in neuron_gt_indices if gt_idx in matched_gt_indices)
            
            neuron_match_rate = neuron_matched_count / neuron_gt_count
            neuron_stats[neuron_name] = {
                'gt_count': neuron_gt_count,
                'matched_count': neuron_matched_count,
                'match_rate': neuron_match_rate
            }
    
    # Print per-neuron statistics
    if len(neuron_stats) > 0:
        print(f"  {'Neuron':<20} {'GT Spikes':<12} {'Matched':<12} {'Match Rate':<12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        for neuron_name in sorted(neuron_stats.keys()):
            stats = neuron_stats[neuron_name]
            print(f"  {neuron_name:<20} {stats['gt_count']:<12} {stats['matched_count']:<12} {stats['match_rate']:<12.4f}")
    else:
        print("  No neuron statistics available")
    
    print("\n### 3. Extract Waveforms")
    
    # Extract waveforms (using original traces, no whitening)
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
    # Important: noise should be 0, neuron ids should be 1, 2, 3, ..., n
    # This ensures labels are continuous from 0 to num_classes-1 for PyTorch CrossEntropyLoss
    unique_neurons = np.unique([x for x in Y_spiketrain_id if x is not None])
    # Map neurons to 1, 2, 3, ..., n (starting from 1, not 0)
    neuron_to_id = {neuron: idx + 1 for idx, neuron in enumerate(unique_neurons)}
    neuron_to_id[None] = 0  # noise is 0
    
    Y_spike_id = np.array([neuron_to_id.get(x, 0) for x in Y_spiketrain_id])  # Default to 0 (noise) if not found
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
    print(f"  - Noise spike count: {np.sum(Y_spike_id == 0)}")
    print(f"  - Valid spike count: {np.sum(Y_spike_id != 0)}")
    
    # Prepare data for SpikeCNN training (adaptive shape)
    print("\n### 5. Prepare data for SpikeCNN training")
    
    # Keep original waveform shape (n_samples, n_channels, n_time)
    n_ch = X_waveform.shape[1]
    n_time = X_waveform.shape[2]
    print(f"  Waveform shape: ({X_waveform.shape[0]}, {n_ch}, {n_time})")
    print(f"  - Number of channels: {n_ch}")
    print(f"  - Time points: {n_time}")
    
    # Convert to format expected by SpikeCNN: (n_samples, n_channels, n_time)
    X_waveform_data = X_waveform.astype(np.float32)
    Y_spike_id_data = Y_spike_id.astype(np.int64)
    
    # Verify labels are valid (0 to max_label, continuous)
    min_label = Y_spike_id_data.min()
    if min_label < 0:
        raise ValueError(f"Invalid label found: {min_label}. Labels must be >= 0 (0=noise, 1+=neurons)")
    
    # ========== 1. Detection Model Data (spike vs noise) ==========
    print("\n### 5.1. Prepare Detection Model Data (spike vs noise)")
    
    # All detected spikes: 0=noise, 1=spike
    Y_detection = (Y_spike_id_data > 0).astype(np.int64)  # 0=noise, 1=spike
    
    # Split into train and validation sets (80/20)
    X_detection_train, X_detection_val, Y_detection_train, Y_detection_val = train_test_split(
        X_waveform_data, Y_detection,
        test_size=0.2,
        random_state=42,
        stratify=Y_detection if len(np.unique(Y_detection)) > 1 else None
    )
    
    # Save detection model data
    np.save(train_data_dir / "x_detection_train.npy", X_detection_train)
    np.save(train_data_dir / "y_detection_train.npy", Y_detection_train)
    np.save(train_data_dir / "x_detection_val.npy", X_detection_val)
    np.save(train_data_dir / "y_detection_val.npy", Y_detection_val)
    
    print(f"  ✓ Detection model data saved:")
    print(f"    - x_detection_train.npy: shape {X_detection_train.shape}")
    print(f"    - y_detection_train.npy: shape {Y_detection_train.shape}")
    print(f"    - x_detection_val.npy: shape {X_detection_val.shape}")
    print(f"    - y_detection_val.npy: shape {Y_detection_val.shape}")
    print(f"    - Training samples: {len(X_detection_train)} (noise: {np.sum(Y_detection_train == 0)}, spike: {np.sum(Y_detection_train == 1)})")
    print(f"    - Validation samples: {len(X_detection_val)} (noise: {np.sum(Y_detection_val == 0)}, spike: {np.sum(Y_detection_val == 1)})")
    
    # ========== 2. Classification Model Data (neuron classification) ==========
    print("\n### 5.2. Prepare Classification Model Data (neuron classification)")
    
    # Get GT spikes (Y_spike_id > 0) and noise spikes (Y_spike_id == 0)
    gt_spike_mask = Y_spike_id_data > 0
    noise_spike_mask = Y_spike_id_data == 0
    
    X_gt_spikes = X_waveform_data[gt_spike_mask]
    Y_gt_spikes = Y_spike_id_data[gt_spike_mask]  # Labels: 1, 2, 3, ..., n
    X_noise_spikes = X_waveform_data[noise_spike_mask]
    
    print(f"  - GT spikes: {len(X_gt_spikes)}")
    print(f"  - Noise spikes (available): {len(X_noise_spikes)}")
    
    # Find the neuron class with the most spikes
    unique_labels, counts = np.unique(Y_gt_spikes, return_counts=True)
    max_spike_count = counts.max()
    max_spike_neuron = unique_labels[np.argmax(counts)]
    print(f"  - Max spike count per neuron: {max_spike_count} (neuron {max_spike_neuron})")
    
    # Sample noise spikes to match max_spike_count
    n_noise_needed = max_spike_count
    if len(X_noise_spikes) >= n_noise_needed:
        # Randomly sample noise spikes
        np.random.seed(42)
        noise_indices = np.random.choice(len(X_noise_spikes), n_noise_needed, replace=False)
        X_noise_sampled = X_noise_spikes[noise_indices]
    else:
        # If not enough noise, use all available and sample with replacement
        print(f"    Warning: Only {len(X_noise_spikes)} noise spikes available, sampling with replacement to get {n_noise_needed}")
        np.random.seed(42)
        noise_indices = np.random.choice(len(X_noise_spikes), n_noise_needed, replace=True)
        X_noise_sampled = X_noise_spikes[noise_indices]
    
    # Combine GT spikes and sampled noise
    # Labels: 0=noise, 1, 2, 3, ..., n=neurons
    X_classification = np.concatenate([X_gt_spikes, X_noise_sampled], axis=0)
    Y_classification = np.concatenate([Y_gt_spikes, np.zeros(len(X_noise_sampled), dtype=np.int64)], axis=0)
    
    print(f"  - Classification data: {len(X_classification)} samples")
    print(f"    - Noise: {np.sum(Y_classification == 0)}")
    for label in sorted(unique_labels):
        count = np.sum(Y_classification == label)
        print(f"    - Neuron {label}: {count}")
    
    # Split into train and validation sets (80/20)
    X_classification_train, X_classification_val, Y_classification_train, Y_classification_val = train_test_split(
        X_classification, Y_classification,
        test_size=0.2,
        random_state=42,
        stratify=Y_classification if len(np.unique(Y_classification)) > 1 else None
    )
    
    # Save classification model data
    np.save(train_data_dir / "x_classification_train.npy", X_classification_train)
    np.save(train_data_dir / "y_classification_train.npy", Y_classification_train)
    np.save(train_data_dir / "x_classification_val.npy", X_classification_val)
    np.save(train_data_dir / "y_classification_val.npy", Y_classification_val)
    
    print(f"  ✓ Classification model data saved:")
    print(f"    - x_classification_train.npy: shape {X_classification_train.shape}")
    print(f"    - y_classification_train.npy: shape {Y_classification_train.shape}")
    print(f"    - x_classification_val.npy: shape {X_classification_val.shape}")
    print(f"    - y_classification_val.npy: shape {Y_classification_val.shape}")
    print(f"    - Training samples: {len(X_classification_train)}")
    print(f"    - Validation samples: {len(X_classification_val)}")
    
    # Save number of classes for classification model
    num_classes_classification = len(unique_labels) + 1  # neurons + noise
    with open(train_data_dir / "num_classes_classification.pkl", "wb") as f:
        pickle.dump(num_classes_classification, f)
    print(f"  ✓ num_classes_classification.pkl saved: {num_classes_classification} classes")
    
    return train_data_dir


# ==================== 4. Training Function ====================

def train_model(
    train_data_dir,
    model_save_dir,
    n_channels,
    left_sample=10,  # Number of samples before spike to extract
    right_sample=20,  # Number of samples after spike to extract
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
            
            # Get tract_channel information
            eval_tract_ch = eval_row.get('tract_channel', None)
            train_tract_ch = train_row_matched.get('tract_channel', None)
            
            # Update eval neuron's tract_channel to match train neuron's tract_channel
            if train_tract_ch is not None and pd.notna(train_tract_ch):
                if eval_tract_ch != train_tract_ch:
                    eval_neuron_inf_matched.loc[eval_idx, 'tract_channel'] = train_tract_ch
                    print(f"  {eval_neuron} (tract_ch: {eval_tract_ch} -> {train_tract_ch}) -> {best_match} (tract_ch: {train_tract_ch}, Similarity: {best_similarity:.4f}, Position distance: {position_distance_final:.2f})")
                else:
                    print(f"  {eval_neuron} (tract_ch: {eval_tract_ch}) -> {best_match} (tract_ch: {train_tract_ch}, Similarity: {best_similarity:.4f}, Position distance: {position_distance_final:.2f})")
            else:
                print(f"  {eval_neuron} (tract_ch: {eval_tract_ch}) -> {best_match} (tract_ch: N/A, Similarity: {best_similarity:.4f}, Position distance: {position_distance_final:.2f})")
    
    print(f"\nMatching completed:")
    print(f"  - Total evaluation neurons: {len(eval_neuron_inf_matched)}")
    print(f"  - Matched: {matched_count}")
    print(f"  - Unmatched: {len(eval_neuron_inf_matched) - matched_count}")
    
    return eval_neuron_inf_matched


# ==================== 6. Evaluation Function ====================

def evaluate_model(
    train_data_dir,
    model_save_dir,
    n_channels,
    left_sample=10,  # Number of samples before spike to extract
    right_sample=20,  # Number of samples after spike to extract
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


def calibration_model_no_whiten(
    recording_f,
    autosort_model: "SimpleAutoSort",
    train_neuron_inf: pd.DataFrame,
    probe,  # Probe object (required, obtained from read_probeinterface)
    calibration_duration_seconds: int = 60,
    n_additional_clusters: int = 5,
    thr_min=3.5,  # No-whiten detection: threshold multiplier
    thr_max=30,  # No-whiten detection: maximum threshold multiplier
    distance=3,  # No-whiten detection: minimum distance between peaks
    ch_max_simul_firing=5,  # No-whiten detection: maximum simultaneous firing channels
    wlen=5,  # No-whiten detection: window length for peak detection
    prominence=10,  # No-whiten detection: minimum peak prominence
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
    train_data_dir: str = None,  # Training data directory path (not used, kept for compatibility)
    skip_noise_classifier: bool = False,  # 若为 True，则跳过噪声分类，直接送入 label classifier
):
    """
    Stage 1: Calibration stage (first calibration_duration_seconds) using no-whiten detection method
    
    Process:
    1. Threshold detection (no-whiten method, on original data, no whitening)
    2. Extract waveforms (from original traces)
    3. Pass through noise classifier, classified as spikes
    4. Extract way4 layer features (30 dimensions, no PCA needed)
    5. K-means clustering directly on way4 features (number of classes = train neurons + n)
    6. Calculate position and waveform for each cluster (using original waveforms)
    7. Match with train neurons, establish mapping relationship
    
    Parameters:
        recording_f: preprocessed recording object
        autosort_model: trained SimpleAutoSort model
        train_neuron_inf: training data neuron_inf DataFrame
        probe: Probe object (required, obtained from read_probeinterface)
        calibration_duration_seconds: calibration duration (seconds)，default 60
        n_additional_clusters: number of additional clusters (n)，default 5
        thr_min, thr_max, distance, ch_max_simul_firing, wlen, prominence: no-whiten detection parameters
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
            - cluster_features: features for each cluster (position, waveform, etc., computed from original waveforms)
            - whitening_matrix_W: None (no whitening used)
            - whitening_matrix_M: None (no whitening used)
    """
    from sklearn.cluster import KMeans
    from scipy.stats import pearsonr
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    
    # Window size (30 samples)
    left_sample_extract = window_params['left_sample']  # 10
    right_sample_extract = window_params['right_sample']  # 20
    window_size = left_sample_extract + right_sample_extract  # 30
    window_size_extract = window_size
    
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
    print("Stage 1: Calibration (No-Whiten Method)")
    print("=" * 50)
    
    # 1. Load first calibration_duration_seconds of data
    max_duration_samples = int(calibration_duration_seconds * sampling_frequency)
    total_samples = recording_f.get_num_samples()
    actual_samples = min(max_duration_samples, total_samples)
    print(f"Loading first {calibration_duration_seconds} seconds of data...")
    # 1. Load calibration data (no whitening)
    recording_calibration = recording_f.frame_slice(start_frame=0, end_frame=actual_samples)
    
    # 2. Get original traces (no whitening needed)
    print(f"\n### 1. Load calibration data (no whitening)")
    traces_original = recording_calibration.get_traces()
    if traces_original.shape[0] > traces_original.shape[1] and traces_original.shape[0] > 100:
        traces_original = traces_original.T
    traces_original = traces_original.astype(np.float32)
    
    # Ensure traces_original is in (n_channels, n_timepoints) format for consistency with detection
    if traces_original.shape[0] > traces_original.shape[1]:
        traces_original = traces_original.T  # Transpose to (n_channels, n_timepoints)
    
    # 2. Threshold detection using no-whiten method (on original data)
    print("\n### 2. Threshold detection (No-Whiten Method)")
    print(f"  thr_min: {thr_min}, thr_max: {thr_max}, distance: {distance}, prominence: {prominence}")
    if valid_channels is not None:
        print(f"共 {len(valid_channels)} 个通道")
    else:
        print(f"  在所有通道上进行检测 (共 {n_channels} 个通道)")
    trace0_car_detect = traces_original.T  # (n_timepoints, n_channels) - original data for detection
    spikes = detect_spike_no_whiten(
        trace0_car_detect,
        thr_min=thr_min,
        thr_max=thr_max,
        distance=distance,
        ch_max_simul_firing=ch_max_simul_firing,
        wlen=wlen,
        prominence=prominence,
        valid_channels=valid_channels,
    )
    spike_coords = np.argwhere(spikes == 1)  # (n_spikes, 2) [time, channel]
    print(f"Number of detected spikes: {len(spike_coords)}")
    
    # The rest is identical to calibration_model, but we need to import necessary functions
    # Note: The implementation below is copied from calibration_model (lines 3002-4328)
    # with the following modifications:
    # 1. All traces_whitened references are replaced with traces_original (already done above)
    # 2. At return value: whitening_matrix_W: None, whitening_matrix_M: None
    
    # Import tqdm if not already imported
    from tqdm import tqdm
    from collections import defaultdict
    
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
            
            # Additional filtering: only keep neurons whose channels are all in valid_channels
            if valid_channels is not None and len(gt_spikes_in_calibration) > 0:
                valid_channels_set = set(valid_channels)
                import ast
                
                neuron_col = 'neuron' if 'neuron' in gt_spikes_in_calibration.columns else 'cluster'
                
                # Build a set of neurons whose tract_channel is in valid_channels
                # For GT spike filtering: only check tract_channel (not best_channels or channel_id)
                valid_neurons_set = set()
                for _, row in eval_neuron_inf.iterrows():
                    neuron_name = row.get('Neuron', None)
                    if neuron_name is None:
                        continue
                    
                    # Check tract_channel directly
                    if 'tract_channel' in row:
                        tract_ch = row.get('tract_channel', None)
                        if pd.notna(tract_ch) and tract_ch is not None:
                            try:
                                tract_ch_int = int(tract_ch)
                                # Convert probe channel index to clique channel index
                                if tract_ch_int in probe_to_clique_index:
                                    tract_ch_clique = probe_to_clique_index[tract_ch_int]
                                    # Check if tract_channel (clique index) is in valid_channels
                                    if tract_ch_clique in valid_channels_set:
                                        valid_neurons_set.add(neuron_name)
                            except (ValueError, TypeError):
                                continue
                
                if len(valid_neurons_set) > 0:
                    gt_spikes_in_calibration = gt_spikes_in_calibration[
                        gt_spikes_in_calibration[neuron_col].isin(valid_neurons_set)
                    ].copy()
                else:
                    gt_spikes_in_calibration = gt_spikes_in_calibration.iloc[0:0].copy()

        if len(gt_spikes_in_calibration) > 0:
            # Record GT spike statistics (after valid_channels filtering)
            gt_neurons_set = set(gt_spikes_in_calibration['neuron'].unique() if 'neuron' in gt_spikes_in_calibration.columns else gt_spikes_in_calibration['cluster'].unique())
            n_gt_neurons = len(gt_neurons_set)
            n_gt_spikes = len(gt_spikes_in_calibration)
            print(f"\n### GT Spike Statistics:")
            print(f"  GT neurons: {n_gt_neurons}, GT spikes: {n_gt_spikes}")
            
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
            spike_detection_rate = n_matched / n_gt_spikes if n_gt_spikes > 0 else 0.0

            print(f"\n### Detection Rate (after threshold detection): {spike_detection_rate:.4f} ({n_matched}/{n_gt_spikes})")

            detection_stats = {
                'n_gt_neurons': n_gt_neurons,
                'n_gt_spikes': n_gt_spikes,
                'spike_detection_rate_after_threshold': spike_detection_rate,
                'n_matched_after_threshold': n_matched,
            }
        else:
            detection_stats = None
    else:
        detection_stats = None
    
    # 3. Extract waveforms and filter boundaries (using original traces)
    print("\n### 3. Extract waveforms (using original traces)")
    valid_spikes = []
    waveforms = []
    spike_times = []
    spike_channels = []
    
    # Use original traces for waveform extraction (keep same shape as training)
    trace0_original_for_extract = traces_original.T  # (n_timepoints, n_channels) original data for waveform
    spike_time_indices = np.array([time_idx for time_idx, _ in spike_coords])
    spike_channel_indices = np.array([channel_idx for _, channel_idx in spike_coords])
    
    # Filter spikes near boundaries (ensure complete window can be extracted)
    valid_mask = (spike_time_indices >= left_sample_extract) & (spike_time_indices < trace0_original_for_extract.shape[0] - right_sample_extract)
    spike_time_indices = spike_time_indices[valid_mask]
    spike_channel_indices = spike_channel_indices[valid_mask]
    
    if len(spike_time_indices) == 0:
        raise ValueError("No valid spikes after boundary filtering")
    
    waveform = np.zeros((len(spike_time_indices), trace0_original_for_extract.shape[1], window_size_extract), dtype=np.float32)
    
    for i, time_range in enumerate(np.arange(-left_sample_extract, right_sample_extract)):
        waveform[:, :, i] = trace0_original_for_extract[spike_time_indices + time_range, :]
    
    # Store waveforms and metadata
    for idx, (time_idx, channel_idx) in enumerate(zip(spike_time_indices, spike_channel_indices)):
        waveforms.append(waveform[idx])  # (n_channels, window_size_extract=30)
        valid_spikes.append((time_idx, channel_idx))
        spike_times.append(time_idx)
        spike_channels.append(channel_idx)
    
    waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size_extract=30)
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
            
            # Additional filtering: only keep neurons whose channels are all in valid_channels
            if valid_channels is not None and len(gt_spikes_in_calibration) > 0:
                valid_channels_set = set(valid_channels)
                import ast
                
                neuron_col = 'neuron' if 'neuron' in gt_spikes_in_calibration.columns else 'cluster'
                
                # Build a set of neurons whose tract_channel is in valid_channels
                # For GT spike filtering: only check tract_channel (not best_channels or channel_id)
                valid_neurons_set = set()
                for _, row in eval_neuron_inf.iterrows():
                    neuron_name = row.get('Neuron', None)
                    if neuron_name is None:
                        continue
                    
                    # Check tract_channel directly
                    if 'tract_channel' in row:
                        tract_ch = row.get('tract_channel', None)
                        if pd.notna(tract_ch) and tract_ch is not None:
                            try:
                                tract_ch_int = int(tract_ch)
                                # Convert probe channel index to clique channel index
                                if tract_ch_int in probe_to_clique_index:
                                    tract_ch_clique = probe_to_clique_index[tract_ch_int]
                                    # Check if tract_channel (clique index) is in valid_channels
                                    if tract_ch_clique in valid_channels_set:
                                        valid_neurons_set.add(neuron_name)
                            except (ValueError, TypeError):
                                continue
                
                if len(valid_neurons_set) > 0:
                    gt_spikes_in_calibration = gt_spikes_in_calibration[
                        gt_spikes_in_calibration[neuron_col].isin(valid_neurons_set)
                    ].copy()
                else:
                    print(f"  Warning: No neurons have all channels in valid_channels, all GT spikes filtered out")
                    gt_spikes_in_calibration = gt_spikes_in_calibration.iloc[0:0].copy()

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
            # Note: GT spike statistics and threshold detection rate were already printed, skip duplicate output here
            pass
        else:
            pass  # GT statistics already printed, no need to repeat
    else:
        pass  # No GT data provided
    
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
                batch_waveforms = waveforms[i:i+batch_size]
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
        # Track noise classifier predictions for matched spikes
        matched_spike_indices_set = set()
        if eval_spike_inf is not None:
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
                
                for det_idx, det_time in enumerate(detected_spike_times_after_extraction):
                    time_diffs = np.abs(gt_spike_times - det_time)
                    min_diff = np.min(time_diffs)
                    if min_diff <= tolerance_samples:
                        matched_spike_indices_set.add(det_idx)
        
        matched_spike_noise_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, n_spikes, batch_size), desc="Noise classification"):
                batch_waveforms = waveforms[i:i+batch_size]
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

                noise_output = autosort_model.clsfier_noise(codes)
                noise_probs = torch.softmax(noise_output, dim=1)
                noise_pred = torch.argmax(noise_output, dim=1)
                
                batch_start_idx = i
                for batch_idx, global_idx in enumerate(range(batch_start_idx, min(batch_start_idx + batch_size, n_spikes))):
                    if global_idx in matched_spike_indices_set:
                        noise_prob = noise_probs[batch_idx, 0].item()
                        spike_prob = noise_probs[batch_idx, 1].item()
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
                    
                    codes_spike = codes[spike_mask]
                    way4_batch = autosort_model.clsfier_label.intermediate_forward(codes_spike)
                    way3_features.append(way4_batch.cpu().numpy())
        
        if len(spike_indices) == 0:
            raise ValueError("No spikes passed noise classifier")
        
        way4_features = np.concatenate(way3_features, axis=0)
        spike_indices = np.array(spike_indices)
        print(f"Number of spikes passing noise classifier: {len(spike_indices)}")
        
        if len(matched_spike_noise_probs) > 0:
            matched_spike_indices_set_filtered = set(spike_indices)
            matched_spikes_passed = sum(1 for item in matched_spike_noise_probs if item['index'] in matched_spike_indices_set_filtered)
            matched_spikes_rejected = len(matched_spike_noise_probs) - matched_spikes_passed
            
            print(f"\n---Noise classifier analysis on matched spikes:")
            print(f"  Total matched spikes (from waveform extraction): {len(matched_spike_noise_probs)}")
            print(f"  Matched spikes passing noise classifier: {matched_spikes_passed}")
            print(f"  Matched spikes rejected by noise classifier: {matched_spikes_rejected}")
            if len(matched_spike_noise_probs) > 0:
                print(f"  Matched spike retention rate: {matched_spikes_passed / len(matched_spike_noise_probs):.4f}")

    # 2.6. Compare spikes passing noise classifier with GT (if available)
    if eval_spike_inf is not None:
        spikes_passing_noise_times = np.array([spike_times[i] for i in spike_indices]).astype(int)
        
        calibration_end_frame = actual_samples
        gt_spikes_in_calibration = eval_spike_inf[
            (eval_spike_inf['time'] >= 0) & (eval_spike_inf['time'] < calibration_end_frame)
        ].copy()
        
        print(f"  GT spikes after time filtering: {len(gt_spikes_in_calibration)}")
        
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
            
            if valid_channels is not None and len(gt_spikes_in_calibration) > 0:
                valid_channels_set = set(valid_channels)
                import ast
                
                neuron_col = 'neuron' if 'neuron' in gt_spikes_in_calibration.columns else 'cluster'
                
                # Build a set of neurons whose tract_channel is in valid_channels
                # For GT spike filtering: only check tract_channel (not best_channels or channel_id)
                valid_neurons_set = set()
                for _, row in eval_neuron_inf.iterrows():
                    neuron_name = row.get('Neuron', None)
                    if neuron_name is None:
                        continue
                    
                    # Check tract_channel directly
                    if 'tract_channel' in row:
                        tract_ch = row.get('tract_channel', None)
                        if pd.notna(tract_ch) and tract_ch is not None:
                            try:
                                tract_ch_int = int(tract_ch)
                                # Convert probe channel index to clique channel index
                                if tract_ch_int in probe_to_clique_index:
                                    tract_ch_clique = probe_to_clique_index[tract_ch_int]
                                    # Check if tract_channel (clique index) is in valid_channels
                                    if tract_ch_clique in valid_channels_set:
                                        valid_neurons_set.add(neuron_name)
                            except (ValueError, TypeError):
                                continue
                
                if len(valid_neurons_set) > 0:
                    gt_spikes_in_calibration = gt_spikes_in_calibration[
                        gt_spikes_in_calibration[neuron_col].isin(valid_neurons_set)
                    ].copy()
                else:
                    print(f"  Warning: No neurons have all channels in valid_channels, all GT spikes filtered out")
                    gt_spikes_in_calibration = gt_spikes_in_calibration.iloc[0:0].copy()
        
        if len(gt_spikes_in_calibration) > 0:
            gt_spike_times = gt_spikes_in_calibration['time'].values.astype(int)
            gt_spike_neurons = gt_spikes_in_calibration['neuron'].values if 'neuron' in gt_spikes_in_calibration.columns else gt_spikes_in_calibration['cluster'].values
            gt_array_noise = np.stack([gt_spike_times, gt_spike_neurons], axis=1)

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
            # Get GT spike count (should be same as after threshold detection)
            n_gt_spikes_total = len(gt_spikes_in_calibration)
            spike_detection_rate = n_matched / n_gt_spikes_total if n_gt_spikes_total > 0 else 0.0

            print(f"\n### Detection Rate (after noise classifier): {spike_detection_rate:.4f} ({n_matched}/{n_gt_spikes_total})")
        else:
            pass  # GT statistics already printed, no need to repeat
    else:
        pass  # No GT data provided
    
    # Filter spike_times and spike_channels to only include spikes that passed noise classifier
    spike_times_filtered = [spike_times[i] for i in spike_indices]
    spike_channels_filtered = [spike_channels[i] for i in spike_indices]
    
    spike_indices_to_filtered_idx = {val: idx for idx, val in enumerate(spike_indices)}
    
    # 5. K-means clustering (directly on way4 features, no PCA needed)
    print("\n### 5. K-means clustering (using way4 features, 30 dimensions, no PCA)")
    n_train_neurons = len(train_neuron_inf)
    n_clusters = n_train_neurons + n_additional_clusters
    print(f"Number of clusters: {n_clusters} (Training neurons: {n_train_neurons}, additional: {n_additional_clusters})")
    print(f"Feature shape: {way4_features.shape} (way4 features, 30 dimensions)")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(way4_features)

    # 7. Calculate position and waveform for each train neuron and each cluster, then match
    print("\n### 7. Calculate cluster position and waveform (based on train neuron channel_id) and match")
    cluster_to_neuron_mapping = {}
    neuron_to_clusters = defaultdict(list)
    cluster_features = {}
    neuron_cluster_comparison = {}
    
    # Prepare traces_original_for_extraction in (n_channels, n_timepoints) format
    if traces_original.shape[0] > traces_original.shape[1] and traces_original.shape[0] > 100:
        traces_original_for_extraction = traces_original.T
    else:
        traces_original_for_extraction = traces_original
    
    # Outer loop: iterate through each train neuron
    for train_idx, train_row in train_neuron_inf.iterrows():
        train_neuron = train_row['Neuron']
        train_pos = np.array([train_row['position_1'], train_row['position_2']])
        train_waveform = np.asarray(train_row['position_waveform'], dtype=np.float32)
        
        train_channel_id_probe = train_row['channel_id']
        if not isinstance(train_channel_id_probe, list):
            if isinstance(train_channel_id_probe, (np.ndarray, tuple)):
                train_channel_id_probe = list(train_channel_id_probe)
            else:
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
            if cluster_id in cluster_to_neuron_mapping:
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            
            if len(cluster_spike_indices) == 0:
                continue
            
            cluster_spike_times = [spike_times_filtered[spike_indices_to_filtered_idx[i]] for i in cluster_spike_indices]
            cluster_spike_chs = [spike_channels_filtered[spike_indices_to_filtered_idx[i]] for i in cluster_spike_indices]
            
            cluster_waveforms_full = []
            
            for time_idx, channel_idx in zip(cluster_spike_times, cluster_spike_chs):
                start = time_idx - left_sample_extract
                end = time_idx + right_sample_extract
                
                if start < 0 or end > traces_original_for_extraction.shape[1]:
                    continue
                if end - start != window_size_extract:
                    continue
                
                waveform = traces_original_for_extraction[:, start:end]
                cluster_waveforms_full.append(waveform)
            
            if len(cluster_waveforms_full) == 0:
                continue
            
            cluster_waveforms_full = np.array(cluster_waveforms_full)
            actual_window_size = cluster_waveforms_full.shape[2]
            
            valid_channel_id = [ch for ch in train_channel_id if 0 <= ch < n_channels]
            if len(valid_channel_id) == 0:
                continue
            
            cluster_waveforms = cluster_waveforms_full[:, valid_channel_id, :]
            actual_window_size_for_compute = cluster_waveforms.shape[2]
            valid_probe_channel_id = [clique_to_probe_index[ch] for ch in valid_channel_id if ch in clique_to_probe_index]
            
            position_1, position_2, position_waveform = compute_cluster_position_waveform(
                cluster_waveforms, valid_probe_channel_id, channel_positions, actual_window_size_for_compute
            )
            
            cluster_waveforms_all = cluster_waveforms
            
            cluster_pos = np.array([position_1, position_2])
            pos_distance = np.linalg.norm(cluster_pos - train_pos)
            
            min_len = min(len(position_waveform), len(train_waveform))
            if min_len == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(position_waveform[:min_len], train_waveform[:min_len])
            
            if train_neuron not in neuron_cluster_comparison:
                neuron_cluster_comparison[train_neuron] = {}
            
            neuron_cluster_comparison[train_neuron][cluster_id] = {
                'position': [position_1, position_2],
                'waveform': position_waveform,
                'position_distance': pos_distance,
                'waveform_corr': corr,
                'n_spikes': len(cluster_spike_indices),
            }
            
            if pos_distance >= position_threshold:
                continue
            
            if corr < waveform_similarity_threshold:
                continue
            
            score = corr / (1 + pos_distance / position_threshold)
            
            if cluster_id not in cluster_to_neuron_mapping:
                cluster_to_neuron_mapping[cluster_id] = train_neuron
                neuron_to_clusters[train_neuron].append(cluster_id)
                cluster_features[cluster_id] = {
                    'position_1': position_1,
                    'position_2': position_2,
                    'position_waveform': position_waveform,
                    'waveforms': cluster_waveforms_all,
                    'n_spikes': len(cluster_spike_indices),
                    'matched_neuron': train_neuron,
                    'score': score,
                    'pos_distance': pos_distance,
                    'waveform_corr': corr,
                }
            else:
                existing_neuron = cluster_to_neuron_mapping[cluster_id]
                existing_distance = cluster_features[cluster_id]['pos_distance']
                if pos_distance < existing_distance:
                    neuron_to_clusters[existing_neuron].remove(cluster_id)
                    cluster_to_neuron_mapping[cluster_id] = train_neuron
                    neuron_to_clusters[train_neuron].append(cluster_id)
                    cluster_features[cluster_id] = {
                        'position_1': position_1,
                        'position_2': position_2,
                        'position_waveform': position_waveform,
                        'waveforms': cluster_waveforms_all,
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
        
        for probe_ch in train_channel_id_probe:
            if int(probe_ch) in probe_to_clique_index:
                clique_ch = probe_to_clique_index[int(probe_ch)]
                all_train_channel_ids_clique.add(clique_ch)
    
    default_channel_id = sorted(list(all_train_channel_ids_clique)) if len(all_train_channel_ids_clique) > 0 else list(range(n_channels))
    
    for cluster_id in range(n_clusters):
        if cluster_id not in cluster_to_neuron_mapping:
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            
            if len(cluster_spike_indices) > 0:
                cluster_spike_times = [spike_times_filtered[spike_indices_to_filtered_idx[i]] for i in cluster_spike_indices]
                cluster_spike_chs = [spike_channels_filtered[spike_indices_to_filtered_idx[i]] for i in cluster_spike_indices]
                
                cluster_waveforms_full = []
                
                for time_idx, channel_idx in zip(cluster_spike_times, cluster_spike_chs):
                    start = time_idx - left_sample_extract
                    end = time_idx + right_sample_extract
                    
                    if start < 0 or end > traces_original_for_extraction.shape[1]:
                        continue
                    if end - start != window_size_extract:
                        continue
                    
                    waveform = traces_original_for_extraction[:, start:end]
                    cluster_waveforms_full.append(waveform)
                
                if len(cluster_waveforms_full) > 0:
                    cluster_waveforms_full = np.array(cluster_waveforms_full)
                    actual_window_size = cluster_waveforms_full.shape[2]
                
                    valid_channel_id = [ch for ch in default_channel_id if 0 <= ch < n_channels]
                    if len(valid_channel_id) > 0:
                        cluster_waveforms = cluster_waveforms_full[:, valid_channel_id, :]
                        actual_window_size_for_compute = cluster_waveforms.shape[2]
                        valid_probe_channel_id = [clique_to_probe_index[ch] for ch in valid_channel_id if ch in clique_to_probe_index]
                        position_1, position_2, position_waveform = compute_cluster_position_waveform(
                            cluster_waveforms, valid_probe_channel_id, channel_positions, actual_window_size_for_compute
                        )
                        cluster_waveforms_all = cluster_waveforms
                        cluster_features[cluster_id] = {
                            'position_1': position_1,
                            'position_2': position_2,
                            'position_waveform': position_waveform,
                            'waveforms': cluster_waveforms_all,
                            'n_spikes': len(cluster_spike_indices),
                            'matched_neuron': 'unmatch',
                            'score': None,
                            'pos_distance': None,
                            'waveform_corr': None,
                        }
    
    print(f"\nMatching results:")
    print(f"  - Total clusters: {n_clusters}")
    print(f"  - Matched clusters: {len(cluster_to_neuron_mapping)}")
    print(f"  - Unmatched clusters: {n_clusters - len(cluster_to_neuron_mapping)}")
    print(f"  - Matched neurons: {len(neuron_to_clusters)}")
    
    # Build results DataFrame
    results_df = pd.DataFrame({
        'spike_time': [spike_times[i] for i in spike_indices],
        'spike_channel': [spike_channels[i] for i in spike_indices],
        'predicted_label': [cluster_to_neuron_mapping.get(cluster_labels[i], 'unmatch') for i in range(len(spike_indices))],
    })
    
    # If eval data exists, add GT labels
    if eval_neuron_inf is not None and eval_spike_inf is not None:
        if 'neuron_match' in eval_neuron_inf.columns:
            eval_to_train_mapping = {}
            for _, row in eval_neuron_inf.iterrows():
                eval_neuron = row['Neuron']
                match = row['neuron_match']
                if match != 'unmatch':
                    eval_to_train_mapping[eval_neuron] = match
                else:
                    eval_to_train_mapping[eval_neuron] = 'unmatch'
            
            spike_times_array = np.array([spike_times[i] for i in spike_indices])
            spike_inf_sorted = eval_spike_inf.sort_values('time').reset_index(drop=True)
            
            gt_labels = []
            for spike_time in spike_times_array:
                time_diff = (spike_inf_sorted['time'] - spike_time).abs()
                min_diff_idx = time_diff.idxmin()
                min_diff = time_diff.loc[min_diff_idx]
                
                if min_diff <= 1:
                    eval_neuron = spike_inf_sorted.loc[min_diff_idx, 'neuron']
                    
                    if eval_neuron in eval_to_train_mapping:
                        gt_label = eval_to_train_mapping[eval_neuron]
                    else:
                        gt_label = 'unmatch'
                else:
                    gt_label = 'noise'
                
                gt_labels.append(gt_label)
            
            results_df['gt_label'] = gt_labels
        else:
            print("Warning: eval_neuron_inf has no neuron_match column, cannot establish GT label mapping")
            results_df['gt_label'] = 'unknown'
    else:
        results_df['gt_label'] = None
    
    # Evaluation Metrics Calculation
    print("\n" + "=" * 80)
    print("Computing evaluation metrics...")
    print("=" * 80)
    
    train_neuron_list = train_neuron_inf['Neuron'].tolist() if train_neuron_inf is not None else []
    
    print("\n1. Generating confusion matrix...")
    confusion_matrix_df, summary_df = generate_confusion_matrix_df(
        results_df=results_df,
        train_neuron_list=train_neuron_list
    )
    
    print("2. Calculating classification accuracy (all classes, merging unmatch & noise)...")
    if 'All' in confusion_matrix_df.index and 'All' in confusion_matrix_df.columns:
        total_samples = confusion_matrix_df.loc['All', 'All']
        correct_predictions = 0
        for neuron in train_neuron_list:
            if neuron in confusion_matrix_df.index and neuron in confusion_matrix_df.columns:
                correct_predictions += confusion_matrix_df.loc[neuron, neuron]
        bg_rows = [lbl for lbl in ['unmatch', 'noise'] if lbl in confusion_matrix_df.index]
        bg_cols = [lbl for lbl in ['unmatch', 'noise'] if lbl in confusion_matrix_df.columns]
        if bg_rows and bg_cols:
            correct_predictions += confusion_matrix_df.loc[bg_rows, bg_cols].to_numpy().sum()
        classification_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    else:
        classification_accuracy = 0.0
    print(f"   Classification accuracy (all classes, unmatch+noise merged): {classification_accuracy:.6f}")
    
    print("2.1. Calculating unit classification accuracy (train neurons only)...")
    if 'All' in confusion_matrix_df.index and 'All' in confusion_matrix_df.columns:
        total_unit_samples = 0
        correct_unit_predictions = 0
        for neuron in train_neuron_list:
            if neuron in confusion_matrix_df.index:
                total_unit_samples += confusion_matrix_df.loc[neuron, 'All']
                if neuron in confusion_matrix_df.columns:
                    correct_unit_predictions += confusion_matrix_df.loc[neuron, neuron]
        unit_classification_accuracy = correct_unit_predictions / total_unit_samples if total_unit_samples > 0 else 0.0
    else:
        unit_classification_accuracy = 0.0
    print(f"   Unit classification accuracy (train neurons only): {unit_classification_accuracy:.6f}")
    
    print("3. Calculating noise detection metrics...")
    noise_detection_metrics = compute_noise_detection_metrics(
        results_df=results_df,
        train_neuron_list=train_neuron_list
    )
    noise_detection_accuracy = noise_detection_metrics['accuracy']
    noise_detection_accuracy_adjusted = noise_detection_metrics.get('accuracy_adjusted', noise_detection_metrics.get('accuracy', 0.0))
    print(f"   Noise detection accuracy: {noise_detection_accuracy:.6f}")
    print(f"   Noise detection accuracy (adjusted): {noise_detection_accuracy_adjusted:.6f}")
    
    # Build cluster_inf dictionary
    cluster_inf = {}
    for cluster_id in range(n_clusters):
        if cluster_id in cluster_features:
            cluster_inf[cluster_id] = {
                'position_1': cluster_features[cluster_id]['position_1'],
                'position_2': cluster_features[cluster_id]['position_2'],
                'position_waveform': cluster_features[cluster_id]['position_waveform'],
                'waveforms': cluster_features[cluster_id].get('waveforms', None),
                'n_spikes': cluster_features[cluster_id]['n_spikes'],
                'matched_neuron': cluster_features[cluster_id].get('matched_neuron', 'unmatch'),
                'score': cluster_features[cluster_id].get('score', None),
                'pos_distance': cluster_features[cluster_id].get('pos_distance', None),
                'waveform_corr': cluster_features[cluster_id].get('waveform_corr', None),
            }
        else:
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            cluster_inf[cluster_id] = {
                'position_1': None,
                'position_2': None,
                'position_waveform': None,
                'waveforms': None,
                'n_spikes': len(cluster_spike_indices),
                'matched_neuron': 'unmatch',
                'score': None,
                'pos_distance': None,
                'waveform_corr': None,
            }
    
    calibration_results = {
        'kmeans_model': kmeans,
        'pca_model': None,
        'cluster_to_neuron_mapping': cluster_to_neuron_mapping,
        'neuron_to_clusters': dict(neuron_to_clusters),
        'neuron_cluster_comparison': neuron_cluster_comparison,
        'cluster_features': cluster_features,
        'cluster_inf': cluster_inf,
        'spike_indices': spike_indices,
        'cluster_labels': cluster_labels,
        'results_df': results_df,
        'way4_features_30d': way4_features,
        'whitening_matrix_W': None,  # No whitening used
        'whitening_matrix_M': None,  # No whitening used
        'detection_stats': detection_stats,
        'confusion_matrix_df': confusion_matrix_df,
        'classification_accuracy': classification_accuracy,
        'unit_classification_accuracy': unit_classification_accuracy,
        'noise_detection_accuracy': noise_detection_accuracy,
        'noise_detection_accuracy_adjusted': noise_detection_accuracy_adjusted,
        'noise_detection_metrics': noise_detection_metrics,
    }
    
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
    
    def _to_noise_label(lbl, train_list):
        if lbl in ['noise', 'unmatch', None, 'unknown']:
            return 'noise'
        if train_list is not None and lbl in train_list:
            return 'spike'
        return 'noise'

    simple_df = results_df.copy()
    train_neuron_list_simple = train_neuron_inf['Neuron'].tolist() if train_neuron_inf is not None else []
    simple_df['gt_noise_label'] = simple_df['gt_label'].apply(lambda x: _to_noise_label(x, train_neuron_list_simple))
    simple_df['predicted_noise_label'] = simple_df['predicted_label'].apply(lambda x: _to_noise_label(x, train_neuron_list_simple))
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
    autosort_model: "SimpleAutoSort",
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
            'left_sample': 10,  # Number of samples before spike to extract
            'right_sample': 20,  # Number of samples after spike to extract
        }
    
    left_sample = window_params['left_sample']  # 10
    right_sample = window_params['right_sample']  # 20
    window_size = left_sample + right_sample  # 30
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
        
        waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size=30)
        
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
        # Handle both old and new key formats
        detection_stats_data = {}
        if 'n_gt_neurons' in detection_stats:
            detection_stats_data['n_gt_neurons'] = [detection_stats['n_gt_neurons']]
        if 'n_gt_spikes' in detection_stats:
            detection_stats_data['n_gt_spikes'] = [detection_stats['n_gt_spikes']]
        if 'spike_detection_rate_after_threshold' in detection_stats:
            detection_stats_data['spike_detection_rate_after_threshold'] = [detection_stats['spike_detection_rate_after_threshold']]
        elif 'spike_detection_rate' in detection_stats:
            # Fallback for old format
            detection_stats_data['spike_detection_rate'] = [detection_stats['spike_detection_rate']]
        if 'n_matched_after_threshold' in detection_stats:
            detection_stats_data['n_matched_after_threshold'] = [detection_stats['n_matched_after_threshold']]
        elif 'n_matched' in detection_stats:
            # Fallback for old format
            detection_stats_data['n_matched'] = [detection_stats['n_matched']]
        if 'n_unmatched' in detection_stats:
            detection_stats_data['n_unmatched'] = [detection_stats['n_unmatched']]
        
        if len(detection_stats_data) > 0:
            detection_stats_df = pd.DataFrame(detection_stats_data)
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


# ==================== 7. SpikeCNN Model and Training (from train_spike.py) ====================

class SpatialAttn(nn.Module):
    """通道-时间外积注意力（自适应输入尺寸）"""
    def __init__(self, ch):
        super().__init__()
        # 使用自适应池化，适应任意输入尺寸
        self.t_pool = nn.AdaptiveAvgPool2d((None, 1))  # 时间轴池化到1
        self.c_pool = nn.AdaptiveAvgPool2d((1, None))  # 通道轴池化到1
        self.compress_t = nn.Conv1d(ch, 4, 1, bias=False)
        self.compress_c = nn.Conv1d(ch, 4, 1, bias=False)
        self.expand   = nn.Conv2d(4, ch, 1, bias=False)

    def forward(self, x):          # x: (B,ch,H,W) - 任意尺寸
        B, C, H, W = x.shape
        # 时间平均分支
        t_feat = self.t_pool(x)               # (B,C,H,1)
        t_feat = self.compress_t(t_feat.squeeze(-1)).unsqueeze(-1)  # (B,4,H,1)
        # 通道平均分支
        c_feat = self.c_pool(x)               # (B,C,1,W)
        c_feat = self.compress_c(c_feat.squeeze(2)).unsqueeze(2)   # (B,4,1,W)
        # 外积得 mask (广播机制会自动处理尺寸匹配)
        mask = torch.sigmoid(self.expand(t_feat * c_feat))  # (B,4,H,W) -> (B,C,H,W)
        return x * mask


class SpikeCNN(nn.Module):
    """SpikeCNN model with adaptive input shape (n_channels × n_time)"""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 1->16
            nn.BatchNorm2d(16),
            SpatialAttn(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            SpatialAttn(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            SpatialAttn(64),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # 自适应池化到 (64,1,1)，适应任意输入尺寸
        self.fc  = nn.Sequential(
            nn.Flatten(),                   # 64
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, H, W) where H=n_channels, W=n_time
        feat = self.gap(self.cnn(x)).flatten(1)  # (B, 64)
        return self.fc(feat)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, target):
        ce_loss = nn.functional.cross_entropy(logits, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        return (self.alpha * (1 - p_t) ** self.gamma * ce_loss).mean()


class SpikeSet(data.Dataset):
    """Dataset for loading spike waveforms (adaptive shape: n_channels × n_time)"""
    def __init__(self, x_npy, y_npy):
        self.x = torch.from_numpy(np.load(x_npy).astype(np.float32))
        self.y = torch.from_numpy(np.load(y_npy).astype(np.int64))
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        # Add channel dimension: (n_channels, n_time) -> (1, n_channels, n_time)
        return self.x[idx].unsqueeze(0), self.y[idx]


def train_detection_model(
    train_data_dir,
    model_save_dir,
    batch_size=256,
    epochs=60,
    lr=1e-3,
    device=None,
):
    """
    Train Detection Model (spike vs noise) using data prepared by prepare_training_data_no_whiten
    
    Parameters:
        train_data_dir: training data directory (should contain x_detection_train.npy, y_detection_train.npy, etc.)
        model_save_dir: model save directory
        batch_size: batch size, default 256
        epochs: number of training epochs, default 60
        lr: learning rate, default 1e-3
        device: device (if None, auto-select)
    
    Returns:
        model: trained model
        training_log: training log dictionary
    """
    from torch.cuda.amp import autocast, GradScaler
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model save directory
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_data_dir = Path(train_data_dir)
    train_loader = data.DataLoader(
        SpikeSet(str(train_data_dir / 'x_detection_train.npy'),
                str(train_data_dir / 'y_detection_train.npy')),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(
        SpikeSet(str(train_data_dir / 'x_detection_val.npy'),
                str(train_data_dir / 'y_detection_val.npy')),
        batch_size=batch_size, shuffle=False
    )
    
    # Get input shape from data
    sample_x, _ = train_loader.dataset[0]
    input_shape = sample_x.shape[1:]  # (n_channels, n_time)
    n_channels, n_time = input_shape
    print(f"Input shape: {input_shape} (n_channels={n_channels}, n_time={n_time})")
    
    # Detection model: 2 classes (0=noise, 1=spike)
    NUM_CLASSES = 2
    print(f"Detection model: {NUM_CLASSES} classes (0=noise, 1=spike)")
    
    # Create model
    model = SpikeCNN(NUM_CLASSES).to(device)
    print(f"Model created (adaptive to input shape {input_shape})")
    
    # Loss and optimizer
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    # Training log
    training_log = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    def run_one_epoch(loader, training=False):
        if training:
            model.train()
        else:
            model.eval()
        total_loss, total_correct, total_samples = 0., 0, 0
        with torch.set_grad_enabled(training):
            for x, y in tqdm(loader, leave=False):
                x, y = x.to(device), y.to(device)
                with autocast():
                    out = model(x)
                    loss = criterion(out, y)
                if training:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                total_loss += loss.item() * x.size(0)
                total_correct += (out.argmax(1) == y).sum().item()
                total_samples += x.size(0)
        return total_loss / total_samples, total_correct / total_samples
    
    best_acc = 0.
    print(f"\nStarting detection model training (total {epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(train_loader, training=True)
        val_loss, val_acc = run_one_epoch(val_loader, training=False)
        scheduler.step()
        
        # Log results
        training_log['epoch'].append(epoch)
        training_log['train_loss'].append(train_loss)
        training_log['train_acc'].append(train_acc)
        training_log['val_loss'].append(val_loss)
        training_log['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} '
              f'| val loss {val_loss:.4f} acc {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = Path(model_save_dir) / 'best_detection_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f'  → Best model saved (val_acc: {best_acc:.4f})')
    
    # Save final model
    final_model_path = Path(model_save_dir) / 'final_detection_model.pth'
    torch.save(model.state_dict(), final_model_path)
    
    # Save training log
    log_path = Path(model_save_dir) / 'detection_training_log.csv'
    pd.DataFrame(training_log).to_csv(log_path, index=False)
    print(f'\nDetection model training completed!')
    print(f'  - Best validation accuracy: {best_acc:.4f}')
    print(f'  - Model saved to: {model_save_dir}')
    print(f'  - Training log saved to: {log_path}')
    
    return model, training_log


def train_classification_model(
    train_data_dir,
    model_save_dir,
    batch_size=256,
    epochs=60,
    lr=1e-3,
    device=None,
):
    """
    Train Classification Model (neuron classification) using data prepared by prepare_training_data_no_whiten
    
    Parameters:
        train_data_dir: training data directory (should contain x_classification_train.npy, y_classification_train.npy, etc.)
        model_save_dir: model save directory
        batch_size: batch size, default 256
        epochs: number of training epochs, default 60
        lr: learning rate, default 1e-3
        device: device (if None, auto-select)
    
    Returns:
        model: trained model
        training_log: training log dictionary
    """
    from torch.cuda.amp import autocast, GradScaler
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model save directory
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_data_dir = Path(train_data_dir)
    train_loader = data.DataLoader(
        SpikeSet(str(train_data_dir / 'x_classification_train.npy'),
                str(train_data_dir / 'y_classification_train.npy')),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(
        SpikeSet(str(train_data_dir / 'x_classification_val.npy'),
                str(train_data_dir / 'y_classification_val.npy')),
        batch_size=batch_size, shuffle=False
    )
    
    # Get input shape from data
    sample_x, _ = train_loader.dataset[0]
    input_shape = sample_x.shape[1:]  # (n_channels, n_time)
    n_channels, n_time = input_shape
    print(f"Input shape: {input_shape} (n_channels={n_channels}, n_time={n_time})")
    
    # Get number of classes from saved file or infer from data
    num_classes_path = train_data_dir / "num_classes_classification.pkl"
    if num_classes_path.exists():
        with open(num_classes_path, 'rb') as f:
            NUM_CLASSES = pickle.load(f)
        print(f"Classification model: {NUM_CLASSES} classes (loaded from file)")
    else:
        # Fallback: infer from data
        max_label = max(train_loader.dataset.y.max().item(),
                        val_loader.dataset.y.max().item())
        min_label = min(train_loader.dataset.y.min().item(),
                        val_loader.dataset.y.min().item())
        
        if min_label < 0:
            raise ValueError(f"Invalid label: {min_label}. Labels must be >= 0")
        
        NUM_CLASSES = max_label + 1
        print(f"Classification model: {NUM_CLASSES} classes (inferred from data)")
    
    # Create model
    model = SpikeCNN(NUM_CLASSES).to(device)
    print(f"Model created (adaptive to input shape {input_shape})")
    
    # Loss and optimizer
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    # Training log
    training_log = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    def run_one_epoch(loader, training=False):
        if training:
            model.train()
        else:
            model.eval()
        total_loss, total_correct, total_samples = 0., 0, 0
        with torch.set_grad_enabled(training):
            for x, y in tqdm(loader, leave=False):
                x, y = x.to(device), y.to(device)
                with autocast():
                    out = model(x)
                    loss = criterion(out, y)
                if training:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                total_loss += loss.item() * x.size(0)
                total_correct += (out.argmax(1) == y).sum().item()
                total_samples += x.size(0)
        return total_loss / total_samples, total_correct / total_samples
    
    best_acc = 0.
    print(f"\nStarting classification model training (total {epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(train_loader, training=True)
        val_loss, val_acc = run_one_epoch(val_loader, training=False)
        scheduler.step()
        
        # Log results
        training_log['epoch'].append(epoch)
        training_log['train_loss'].append(train_loss)
        training_log['train_acc'].append(train_acc)
        training_log['val_loss'].append(val_loss)
        training_log['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} '
              f'| val loss {val_loss:.4f} acc {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = Path(model_save_dir) / 'best_classification_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f'  → Best model saved (val_acc: {best_acc:.4f})')
    
    # Save final model
    final_model_path = Path(model_save_dir) / 'final_classification_model.pth'
    torch.save(model.state_dict(), final_model_path)
    
    # Save training log
    log_path = Path(model_save_dir) / 'classification_training_log.csv'
    pd.DataFrame(training_log).to_csv(log_path, index=False)
    print(f'\nClassification model training completed!')
    print(f'  - Best validation accuracy: {best_acc:.4f}')
    print(f'  - Model saved to: {model_save_dir}')
    print(f'  - Training log saved to: {log_path}')
    
    return model, training_log


def prepare_evaluation_data_no_whiten(
    recording_f,
    spike_inf,
    neuron_inf,
    train_data_dir,
    save_dir,
    train_neuron_inf=None,
    duration_seconds=None,
    thr_min=3.5,
    thr_max=30,
    distance=3,
    ch_max_simul_firing=5,
    wlen=5,
    prominence=10,
    left_sample=10,
    right_sample=20,
    valid_channels=None,
    position_threshold=10.0,
    waveform_similarity_threshold=0.95,
):
    """
    Prepare evaluation data using AutoSort threshold detection (no whitening required)
    
    This function is similar to prepare_training_data_no_whiten but for evaluation data.
    It uses the neuron mapping from training data to ensure label consistency.
    It also uses match_neurons to establish mapping between evaluation and training neurons.
    
    Parameters:
        recording_f: preprocessed recording object (should be recording_clique for clique-level processing)
        spike_inf: DataFrame containing GT spike information (filtered to neurons in clique)
        neuron_inf: DataFrame containing neuron information (filtered to neurons in clique)
        train_data_dir: training data directory (to load neuron_mapping.pkl for label consistency)
        save_dir: save directory path
        train_neuron_inf: Training data neuron_inf DataFrame (for neuron matching), if None will try to load from train_data_dir
        duration_seconds: processing duration (seconds), if None process entire recording
        thr_min, thr_max, distance, ch_max_simul_firing, wlen, prominence: detection parameters
        left_sample, right_sample: waveform window parameters
        valid_channels: list of valid channels, if None automatically computed from neuron_inf's tract_channel column
        position_threshold: Position distance threshold for neuron matching (microns), default 10.0
        waveform_similarity_threshold: Waveform similarity threshold for neuron matching, default 0.95
    
    Returns:
        eval_data_dir: evaluation data save directory
    """
    print("### 1. Threshold Detection (No Whitening Method)")
    
    # Load neuron mapping from training data
    train_data_dir = Path(train_data_dir)
    neuron_mapping_path = train_data_dir / "neuron_mapping.pkl"
    if not neuron_mapping_path.exists():
        raise FileNotFoundError(f"neuron_mapping.pkl not found at {neuron_mapping_path}. Please ensure training data has been prepared first.")
    
    with open(neuron_mapping_path, 'rb') as f:
        train_neuron_mapping = pickle.load(f)
    
    train_neuron_to_id = train_neuron_mapping['neuron_to_id']
    train_id_to_neuron = train_neuron_mapping['id_to_neuron']
    train_unique_neurons = set(train_neuron_mapping['unique_neurons'])
    
    print(f"Loaded neuron mapping from training data: {len(train_unique_neurons)} neurons")
    
    # Match neurons between training and evaluation data
    if train_neuron_inf is None:
        # Try to load train_neuron_inf from train_data_dir
        train_neuron_inf_path = train_data_dir / "neuron_inf.pkl"
        if train_neuron_inf_path.exists():
            with open(train_neuron_inf_path, 'rb') as f:
                train_neuron_inf = pickle.load(f)
        else:
            print("Warning: train_neuron_inf not provided and not found in train_data_dir. Neuron matching will be skipped.")
            train_neuron_inf = None
    
    # Perform neuron matching if train_neuron_inf is available
    eval_neuron_inf_matched = None
    eval_neuron_to_train_neuron = {}  # Mapping: eval_neuron_name -> train_neuron_name
    
    if train_neuron_inf is not None:
        # Check if neuron_inf has required columns for matching
        required_cols = ['position_1', 'position_2', 'position_waveform']
        train_has_cols = all(col in train_neuron_inf.columns for col in required_cols)
        eval_has_cols = all(col in neuron_inf.columns for col in required_cols)
        
        if train_has_cols and eval_has_cols:
            print("\n### 1.5. Neuron Matching (Training Set <-> Evaluation Set)")
            eval_neuron_inf_matched = match_neurons(
                train_neuron_inf=train_neuron_inf,
                eval_neuron_inf=neuron_inf,
                position_threshold=position_threshold,
                waveform_similarity_threshold=waveform_similarity_threshold
            )
            
            # Build mapping dictionary: eval_neuron_name -> train_neuron_name
            for idx, row in eval_neuron_inf_matched.iterrows():
                eval_neuron = row['Neuron']
                train_neuron = row['neuron_match']
                if train_neuron != 'unmatch':
                    eval_neuron_to_train_neuron[eval_neuron] = train_neuron
            
            # Use updated neuron_inf (with updated tract_channel for matched neurons)
            neuron_inf = eval_neuron_inf_matched.copy()
            print(f"Neuron matching completed: {len(eval_neuron_to_train_neuron)} matched neurons")
            print(f"  Updated neuron_inf with matched tract_channel values")
        else:
            print("Warning: Missing required columns for neuron matching. Will use direct neuron name matching.")
            if not train_has_cols:
                print(f"  - Training neuron_inf missing columns: {[col for col in required_cols if col not in train_neuron_inf.columns]}")
            if not eval_has_cols:
                print(f"  - Evaluation neuron_inf missing columns: {[col for col in required_cols if col not in neuron_inf.columns]}")
    
    # Get recording sampling rate and number of channels
    sampling_rate = recording_f.get_sampling_frequency()
    n_channels = recording_f.get_num_channels()
    print(f"Sampling rate: {sampling_rate} Hz, Number of channels: {n_channels}")
    
    # Calculate corresponding number of samples
    total_frames = recording_f.get_num_frames()
    if duration_seconds is None:
        actual_frames = total_frames
        print(f"Processing entire recording: {actual_frames} samples ({actual_frames/sampling_rate:.2f} seconds)")
    else:
        max_frames = int(duration_seconds * sampling_rate)
        actual_frames = min(max_frames, total_frames)
        print(f"Recording total length: {total_frames} samples ({total_frames/sampling_rate:.2f} seconds)")
        print(f"Will process first {actual_frames} samples ({actual_frames/sampling_rate:.2f} seconds)")
    
    # Calculate and save valid_channel list
    # IMPORTANT: valid_channels should be based on training set neurons' tract_channel, not evaluation set
    # This ensures detection is only performed on channels where the model was trained to detect spikes
    recording_channel_ids = recording_f.get_channel_ids()
    probe_to_clique_index = {}
    for clique_idx, probe_ch in enumerate(recording_channel_ids):
        try:
            probe_ch_int = int(probe_ch)
            probe_to_clique_index[probe_ch_int] = clique_idx
        except (ValueError, TypeError):
            probe_to_clique_index[probe_ch] = clique_idx
    
    if valid_channels is None:
        # Try to get valid_channels from training set neurons
        train_neuron_inf_for_channels = train_neuron_inf
        if train_neuron_inf_for_channels is None:
            # Try to load from train_data_dir
            train_neuron_inf_path = train_data_dir / "neuron_inf.pkl"
            if train_neuron_inf_path.exists():
                with open(train_neuron_inf_path, 'rb') as f:
                    train_neuron_inf_for_channels = pickle.load(f)
        
        if train_neuron_inf_for_channels is not None and 'tract_channel' in train_neuron_inf_for_channels.columns:
            # Use training set neurons' tract_channel
            tract_channels_probe = sorted(train_neuron_inf_for_channels['tract_channel'].dropna().unique().tolist())
            valid_channels = []
            for probe_ch in tract_channels_probe:
                try:
                    probe_ch_int = int(probe_ch)
                    if probe_ch_int in probe_to_clique_index:
                        valid_channels.append(probe_to_clique_index[probe_ch_int])
                except (ValueError, TypeError):
                    if probe_ch in probe_to_clique_index:
                        valid_channels.append(probe_to_clique_index[probe_ch])
            valid_channels = sorted(valid_channels)
            print(f"Using training set neurons' tract_channel for valid_channels: {len(valid_channels)} channels")
        elif 'tract_channel' in neuron_inf.columns:
            # Fallback: use evaluation set neurons' tract_channel (not ideal)
            print("Warning: train_neuron_inf not available, using evaluation set neurons' tract_channel (may not be optimal)")
            tract_channels_probe = sorted(neuron_inf['tract_channel'].dropna().unique().tolist())
            valid_channels = []
            for probe_ch in tract_channels_probe:
                try:
                    probe_ch_int = int(probe_ch)
                    if probe_ch_int in probe_to_clique_index:
                        valid_channels.append(probe_to_clique_index[probe_ch_int])
                except (ValueError, TypeError):
                    if probe_ch in probe_to_clique_index:
                        valid_channels.append(probe_to_clique_index[probe_ch])
            valid_channels = sorted(valid_channels)
        else:
            valid_channels = None
            print("Warning: Neither train_neuron_inf nor neuron_inf has tract_channel column, will detect on all channels")
    else:
        # valid_channels provided as parameter, convert probe indices to clique indices
        valid_channels_probe = valid_channels
        valid_channels = []
        for probe_ch in valid_channels_probe:
            try:
                probe_ch_int = int(probe_ch)
                if probe_ch_int in probe_to_clique_index:
                    valid_channels.append(probe_to_clique_index[probe_ch_int])
            except (ValueError, TypeError):
                if probe_ch in probe_to_clique_index:
                    valid_channels.append(probe_to_clique_index[probe_ch])
        valid_channels = sorted(valid_channels)
    
    if valid_channels is not None:
        print(f"Number of valid channels: {len(valid_channels)}")
    
    # Read data (no whitening)
    trace0_car = recording_f.get_traces(start_frame=0, end_frame=actual_frames).astype(np.float32)
    print(f"Data shape: {trace0_car.shape}")
    
    # Use AutoSort's detect_spike_no_whiten function
    spikes = detect_spike_no_whiten(
        trace0_car,
        thr_min=thr_min,
        thr_max=thr_max,
        distance=distance,
        ch_max_simul_firing=ch_max_simul_firing,
        wlen=wlen,
        prominence=prominence,
        valid_channels=valid_channels,
    )
    
    # Build detect_array
    print("Building detect_array...")
    all_spike_train = []
    spike_loc = []
    for channel_num in range(trace0_car.shape[1]):
        spiketrain_loc = np.where(spikes[:, channel_num])[0]
        all_spike_train += list(spiketrain_loc)
        spike_loc += [channel_num] * len(spiketrain_loc)
    
    X_spiketrain_time = np.array(all_spike_train)
    Y_spiketrain_id_final = np.array(spike_loc)
    detect_array = np.array([X_spiketrain_time, Y_spiketrain_id_final]).T
    
    print(f"Number of detected spikes: {len(detect_array)}")
    
    print("\n### 2. Load Ground Truth and Match")
    
    # Filter spike_inf by duration
    if duration_seconds is None:
        spike_inf_filtered = spike_inf.copy()
        print(f"Using all spikes in spike_inf (no duration filter)")
    else:
        max_frames = int(duration_seconds * sampling_rate)
        spike_inf_filtered = spike_inf[spike_inf['time'] < max_frames].copy()
        print(f"Filtered spike_inf to duration: {duration_seconds}s (max_frames={max_frames})")
        print(f"  Original spike count: {len(spike_inf)}, Filtered spike count: {len(spike_inf_filtered)}")
    
    # Build gt_array
    print("Building gt_array...")
    recording_channel_ids = recording_f.get_channel_ids()
    probe_to_clique_index = {}
    for clique_idx, probe_ch in enumerate(recording_channel_ids):
        try:
            probe_ch_int = int(probe_ch)
            probe_to_clique_index[probe_ch_int] = clique_idx
        except (ValueError, TypeError):
            probe_to_clique_index[probe_ch] = clique_idx
    
    spike_train_all = []
    y_unit_id = []
    gt_ch = []
    
    # IMPORTANT: Only use matched training set neurons for GT array
    # Unmatched neurons should not be included in GT, as they will be treated as noise
    matched_eval_neurons = set(eval_neuron_to_train_neuron.keys()) if len(eval_neuron_to_train_neuron) > 0 else set()
    
    # If no matching was performed, fallback to using all eval neurons that are in training set
    # (direct name match)
    if len(matched_eval_neurons) == 0:
        print("Warning: No neuron matching performed, using direct name matching with training set")
        # Check which eval neurons are directly in training set
        for eval_neuron_name in neuron_inf['Neuron'].unique():
            if eval_neuron_name in train_neuron_to_id:
                matched_eval_neurons.add(eval_neuron_name)
    
    print(f"Building GT array using {len(matched_eval_neurons)} matched neurons (out of {len(neuron_inf)} total eval neurons)")
    
    neuron_spike_counts = {}  # Track spike counts per neuron
    
    for neuron_idx in range(len(neuron_inf)):
        eval_neuron_name = neuron_inf['Neuron'].iloc[neuron_idx]
        neuron_channel_id_probe = neuron_inf['tract_channel'].iloc[neuron_idx]
        
        # Only include matched neurons in GT array
        if eval_neuron_name not in matched_eval_neurons:
            continue
        
        if neuron_channel_id_probe in probe_to_clique_index:
            neuron_channel_id_clique = probe_to_clique_index[neuron_channel_id_probe]
        else:
            continue
        
        # Get training neuron name (from matching or direct match)
        if eval_neuron_name in eval_neuron_to_train_neuron:
            train_neuron_name = eval_neuron_to_train_neuron[eval_neuron_name]
        else:
            # Direct match (fallback case)
            train_neuron_name = eval_neuron_name
        
        # Match spikes using eval neuron name (because spike_inf uses eval neuron names)
        # Note: spike_inf_filtered is already filtered by duration
        neuron_spikes = spike_inf_filtered[spike_inf_filtered['neuron'] == eval_neuron_name]
        if len(neuron_spikes) > 0:
            spike_times = neuron_spikes['time'].values
            spike_train_all += list(spike_times)
            # Use matched training neuron name for y_unit_id
            y_unit_id += [train_neuron_name] * len(spike_times)
            gt_ch += [neuron_channel_id_clique] * len(spike_times)
            
            # Track spike count
            neuron_key = f"{eval_neuron_name} -> {train_neuron_name}"
            neuron_spike_counts[neuron_key] = len(spike_times)
    
    gt_array = np.array([spike_train_all, gt_ch]).T
    print(f"GT spike count: {len(gt_array)} (within duration: {duration_seconds if duration_seconds is not None else 'all'}s)")
    
    # Print spike counts per matched neuron
    if len(neuron_spike_counts) > 0:
        print(f"Spike counts per matched neuron:")
        for neuron_key, count in sorted(neuron_spike_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {neuron_key}: {count} spikes")
    
    # Use AutoSort's map_gt_annotation function
    # Use same time_tolerance as training (time_tolerance=2)
    gt_label_array1 = map_gt_annotation(detect_array, gt_array, time_tolerance=2)
    
    # Calculate detection rate
    detection_rate = np.where(gt_label_array1 > -1)[0].shape[0] / gt_array.shape[0] if gt_array.shape[0] > 0 else 0.0
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
    
    print("\n### 3. Extract Waveforms")
    
    # Extract waveforms
    waveform, valid_mask = extract_waveforms(
        trace0_car, X_spiketrain_time, left_sample, right_sample
    )
    
    # Apply valid_mask filter
    X_spiketrain_time = X_spiketrain_time[valid_mask]
    Y_spiketrain_id = Y_spiketrain_id[valid_mask]
    Y_spiketrain_id_final = Y_spiketrain_id_final[valid_mask]
    
    print(f"Waveform extraction completed!")
    print(f"waveform shape: {waveform.shape}")
    
    print("\n### 4. Save Evaluation Data")
    
    # Create save directory
    eval_data_dir = Path(save_dir) / "eval_data"
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {eval_data_dir}")
    
    # Prepare data
    X_waveform = waveform
    
    # Convert Y_spike_id using training neuron mapping
    # Note: Y_spiketrain_id already contains matched training neuron names (if matching was performed)
    # If neuron not in training set, mark as noise (0)
    Y_spike_id = []
    for neuron_name in Y_spiketrain_id:
        if neuron_name is None:
            Y_spike_id.append(0)  # noise (unmatched spike)
        elif neuron_name in train_neuron_to_id:
            # Neuron name is already a training neuron name (from matching or direct match)
            # Use training neuron ID directly
            Y_spike_id.append(train_neuron_to_id[neuron_name])
        else:
            # Neuron not in training set (unmatched eval neuron or other), mark as noise
            Y_spike_id.append(0)
    
    Y_spike_id = np.array(Y_spike_id, dtype=np.int64)
    
    # Verify labels are valid
    min_label = Y_spike_id.min()
    if min_label < 0:
        raise ValueError(f"Invalid label found: {min_label}. Labels must be >= 0")
    
    # Convert to format expected by SpikeCNN: (n_samples, n_channels, n_time)
    X_eval_data = X_waveform.astype(np.float32)
    Y_eval_data = Y_spike_id.astype(np.int64)
    
    # Save as .npy files
    np.save(eval_data_dir / "x_test.npy", X_eval_data)
    np.save(eval_data_dir / "y_test.npy", Y_eval_data)
    
    print(f"  ✓ x_test.npy saved: shape {X_eval_data.shape}")
    print(f"  ✓ y_test.npy saved: shape {Y_eval_data.shape}")
    print(f"  - Evaluation samples: {len(X_eval_data)}")
    print(f"  - Noise spike count: {np.sum(Y_spike_id == 0)}")
    print(f"  - Valid spike count: {np.sum(Y_spike_id != 0)}")
    
    return eval_data_dir


def evaluate_spike_cnn(
    eval_data_dir,
    model_save_dir,
    batch_size=256,
    device=None,
    save_results=True,
    results_save_dir=None,
):
    """
    Evaluate SpikeCNN model on evaluation data
    
    Parameters:
        eval_data_dir: evaluation data directory (should contain x_test.npy, y_test.npy)
        model_save_dir: model save directory (should contain best_spike_cls.pth or final_spike_cls.pth)
        batch_size: batch size, default 256
        device: device (if None, auto-select)
        save_results: whether to save evaluation results, default True
        results_save_dir: results save directory (if None, use model_save_dir/eval)
    
    Returns:
        results: dictionary containing:
            - accuracy: overall accuracy
            - confusion_matrix: confusion matrix
            - per_class_accuracy: per-class accuracy
            - predictions: predicted labels
            - ground_truth: ground truth labels
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    eval_data_dir = Path(eval_data_dir)
    model_save_dir = Path(model_save_dir)
    
    test_loader = data.DataLoader(
        SpikeSet(str(eval_data_dir / 'x_test.npy'),
                str(eval_data_dir / 'y_test.npy')),
        batch_size=batch_size, shuffle=False
    )
    
    # Get input shape from data
    sample_x, _ = test_loader.dataset[0]
    input_shape = sample_x.shape[1:]  # (n_channels, n_time)
    n_channels, n_time = input_shape
    print(f"Input shape: {input_shape} (n_channels={n_channels}, n_time={n_time})")
    
    # Get number of classes from training data (not from evaluation data!)
    # Evaluation data may only contain a subset of classes
    train_data_dir = model_save_dir.parent.parent / "train_data"
    neuron_mapping_path = train_data_dir / "neuron_mapping.pkl"
    
    if neuron_mapping_path.exists():
        with open(neuron_mapping_path, 'rb') as f:
            train_neuron_mapping = pickle.load(f)
        # Number of classes = number of unique neurons + 1 (for noise class 0)
        NUM_CLASSES = len(train_neuron_mapping['unique_neurons']) + 1
        print(f"Number of classes from training data: {NUM_CLASSES} (neurons: {len(train_neuron_mapping['unique_neurons'])}, noise: 1)")
    else:
        # Fallback: try to infer from evaluation data (may be incorrect if eval data has fewer classes)
        print("Warning: neuron_mapping.pkl not found, inferring NUM_CLASSES from evaluation data (may be incorrect)")
        max_label = test_loader.dataset.y.max().item()
        min_label = test_loader.dataset.y.min().item()
        
        if min_label < 0:
            raise ValueError(f"Invalid label: {min_label}. Labels must be >= 0")
        
        NUM_CLASSES = max_label + 1
        print(f"Number of classes (inferred from eval data): {NUM_CLASSES}")
    
    # Create model with correct number of classes
    model = SpikeCNN(NUM_CLASSES).to(device)
    
    # Load model weights
    best_model_path = model_save_dir / 'best_spike_cls.pth'
    final_model_path = model_save_dir / 'final_spike_cls.pth'
    
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from: {best_model_path}")
    elif final_model_path.exists():
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        print(f"Loaded final model from: {final_model_path}")
    else:
        raise FileNotFoundError(f"Model file not found. Expected {best_model_path} or {final_model_path}")
    
    model.eval()
    
    # Evaluation
    all_predictions = []
    all_ground_truth = []
    
    print(f"\nEvaluating on {len(test_loader.dataset)} samples...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            
            all_predictions.append(pred.cpu().numpy())
            all_ground_truth.append(y.cpu().numpy())
    
    # Combine results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    # Calculate metrics
    accuracy = accuracy_score(all_ground_truth, all_predictions)
    cm = confusion_matrix(all_ground_truth, all_predictions)
    
    # Per-class accuracy
    per_class_accuracy = []
    for i in range(NUM_CLASSES):
        mask = all_ground_truth == i
        if mask.sum() > 0:
            class_acc = (all_predictions[mask] == i).sum() / mask.sum()
            per_class_accuracy.append(class_acc)
        else:
            per_class_accuracy.append(0.0)
    
    print(f"\nEvaluation Results:")
    print(f"  - Overall accuracy: {accuracy:.4f}")
    print(f"  - Total samples: {len(all_ground_truth)}")
    print(f"  - Per-class accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        class_name = "noise" if i == 0 else f"neuron_{i}"
        print(f"    Class {i} ({class_name}): {acc:.4f}")
    
    # Build results dictionary
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_accuracy,
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'classification_report': classification_report(all_ground_truth, all_predictions, output_dict=True)
    }
    
    # Save results if requested
    if save_results:
        if results_save_dir is None:
            results_save_dir = model_save_dir / 'eval'
        results_save_dir = Path(results_save_dir)
        results_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(results_save_dir / 'confusion_matrix.csv')
        print(f"\nResults saved to: {results_save_dir}")
        print(f"  - confusion_matrix.csv")
        
        # Save evaluation summary
        summary_df = pd.DataFrame({
            'metric': ['overall_accuracy'] + [f'class_{i}_accuracy' for i in range(NUM_CLASSES)],
            'value': [accuracy] + per_class_accuracy
        })
        summary_df.to_csv(results_save_dir / 'evaluation_summary.csv', index=False)
        print(f"  - evaluation_summary.csv")
    
    return results
