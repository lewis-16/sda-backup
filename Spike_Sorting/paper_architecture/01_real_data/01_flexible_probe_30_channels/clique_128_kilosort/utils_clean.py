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

@dataclass
class CliqueInfo:
    """Information about a clique (subset of channels)"""
    clique_id: int
    device_channel_indices: List[int]
    contact_ids: List[str]
    center: Tuple[float, float]
# ==================== 1. Threshold Detection ====================

def detect_on_channel_ms4(data: np.ndarray, *, detect_threshold: float, detect_interval: float, detect_sign: int, margin: int = 0):
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
            detected_times = detect_on_channel_ms4(
                signal,
                detect_threshold=detect_threshold_per_channel[i],
                detect_interval=detect_interval,
                detect_sign=detect_sign,
                margin=margin
            )
            
            # Mark detected spikes
            spikes[detected_times, i] = 1
    
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


# ==================== 2. Training Data Preparation ====================

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


def prepare_training_data(
    recording_f,
    spike_inf,
    neuron_inf,
    save_dir,
    duration_seconds=200,
    detect_threshold=None,
    detect_interval=10,
    detect_sign=-1,
    margin=0,
    left_sample=10,
    right_sample=20,
    valid_channels=None,
    thr_min=3.5,
):
    """
    Prepare training data (complete pipeline: detection -> matching -> waveform extraction -> saving)
    
    Parameters:
        recording_f: preprocessed recording object
        spike_inf: DataFrame containing GT spike information
        neuron_inf: DataFrame containing neuron information
        save_dir: save directory path
        duration_seconds: processing duration (seconds), default 200
        detect_threshold: detection threshold value (if None, calculated from thr_min * noise_std)
        detect_interval: minimum interval between detections (samples), default 10
        detect_sign: detection sign (-1 for negative, 0 for absolute, 1 for positive), default -1
        margin: margin to exclude from boundaries (samples), default 0
        left_sample, right_sample: waveform window parameters
        valid_channels: list of valid channels, if None automatically computed from neuron_inf's tract_channel column
        thr_min: minimum threshold multiplier (relative to noise std), default 3.5 (used if detect_threshold is None)
    
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
    
    # Calculate and save valid_channel list
    if valid_channels is None:
        # If not provided, automatically compute from neuron_inf
        if 'tract_channel' in neuron_inf.columns:
            valid_channels = sorted(neuron_inf['tract_channel'].unique().tolist())
        else:
            valid_channels = None
            print("Warning: neuron_inf doesn't have tract_channel column, will detect on all channels")
    
    if valid_channels is not None:
        print(f"Number of valid channels: {len(valid_channels)}")
        print(f"Valid channels list: {valid_channels}")
    
    # Read data
    trace0_car = recording_f.get_traces(start_frame=0, end_frame=actual_frames).astype(np.float32)
    print(f"Data shape: {trace0_car.shape}")
    
    # Use MountainSort4 detect_spike function (detect only on valid channels)
    spikes = detect_spike(
        trace0_car,
        detect_threshold=detect_threshold,
        detect_interval=detect_interval,
        detect_sign=detect_sign,
        margin=margin,
        valid_channels=valid_channels,
        thr_min=thr_min,
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
    print("Building gt_array...")
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
    print(f"GT spike count: {len(gt_array)}")
    
    # Calculate GT spike metrics (is_peak, peak_heights, prominences)
    # Directly calculate metrics at GT spike locations (not using detection algorithm)
    # This is for understanding GT spike characteristics to guide detection parameters
    print("\n### 2.1. Calculate GT Spike Metrics")
    print("Calculating peak metrics directly at GT spike locations...")
    
    # Initialize arrays to store results (same order as gt_array)
    gt_is_peak = np.zeros(len(gt_array), dtype=bool)
    gt_peak_heights = np.zeros(len(gt_array), dtype=np.float32)
    gt_prominences = np.zeros(len(gt_array), dtype=np.float32)
    
    # Process each GT spike directly
    for gt_idx in range(len(gt_array)):
        gt_time = int(gt_array[gt_idx, 0])
        gt_channel = int(gt_array[gt_idx, 1])
        
        # Check if time and channel are valid
        if gt_time < 0 or gt_time >= trace0_car.shape[0] or gt_channel < 0 or gt_channel >= trace0_car.shape[1]:
            continue
        
        # Get signal for this channel (negative signal, same as detect_spike)
        signal = -trace0_car[:, gt_channel]
        
        # Check if GT time point is a local peak (using find_peaks with relaxed conditions)
        # Use a small window around GT time to check if it's a peak
        window_size = max(10, distance * 2)
        start_idx = max(0, gt_time - window_size)
        end_idx = min(len(signal), gt_time + window_size + 1)
        
        if end_idx - start_idx < 3:
            continue
        
        # Check if GT time is a peak in the local window (use relaxed conditions)
        # We want to know if this location is naturally a peak, not if it passes detection filters
        local_peaks, local_props = scipy.signal.find_peaks(
            signal[start_idx:end_idx],
            distance=1,  # Minimal distance to detect any peak
            prominence=0,  # No prominence requirement
        )
        
        # Adjust peak indices to global coordinates
        local_peaks = local_peaks + start_idx
        
        # Check if GT time is a peak (within 1 sample tolerance)
        is_peak = False
        if len(local_peaks) > 0:
            closest_peak_idx = np.argmin(np.abs(local_peaks - gt_time))
            closest_peak = local_peaks[closest_peak_idx]
            if abs(closest_peak - gt_time) <= 1:
                is_peak = True
                peak_time = closest_peak
            else:
                peak_time = gt_time  # Use GT time even if not exactly a peak
        else:
            peak_time = gt_time  # Use GT time even if not detected as peak
        
        gt_is_peak[gt_idx] = is_peak
        
        # Calculate peak height at GT spike location (using same method as detect_spike)
        peak_height = signal[peak_time]
        gt_peak_heights[gt_idx] = abs(peak_height)
        
        # Calculate prominence at GT spike location (using same wlen as detect_spike: 15)
        try:
            prominences_result = scipy.signal.peak_prominences(
                signal, [peak_time], wlen=15
            )
            prominence_val = prominences_result[0][0]
            gt_prominences[gt_idx] = prominence_val
        except:
            # If prominence calculation fails, set to 0
            gt_prominences[gt_idx] = 0.0
    
    # Print statistics
    n_peaks = np.sum(gt_is_peak)
    print(f"GT spike metrics calculated:")
    print(f"  - Total GT spikes: {len(gt_array)}")
    print(f"  - GT spikes that are peaks: {n_peaks} ({100*n_peaks/len(gt_array):.2f}%)")
    print(f"  - GT spikes that are NOT peaks: {len(gt_array)-n_peaks} ({100*(len(gt_array)-n_peaks)/len(gt_array):.2f}%)")
    if n_peaks > 0:
        print(f"  - Peak heights (mean ± std): {np.mean(gt_peak_heights[gt_is_peak]):.2f} ± {np.std(gt_peak_heights[gt_is_peak]):.2f}")
        print(f"  - Prominences (mean ± std): {np.mean(gt_prominences[gt_is_peak]):.2f} ± {np.std(gt_prominences[gt_is_peak]):.2f}")
    
    # Save GT metrics to file
    gt_metrics = {
        'is_peak': gt_is_peak,
        'peak_heights': gt_peak_heights,
        'prominences': gt_prominences,
        'gt_array': gt_array,
    }
    train_data_dir = Path(save_dir) / "train_data"
    train_data_dir.mkdir(parents=True, exist_ok=True)
    with open(train_data_dir / "gt_metrics.pkl", "wb") as f:
        pickle.dump(gt_metrics, f)
    print(f"  ✓ GT metrics saved to: {train_data_dir / 'gt_metrics.pkl'}")
    
    # Use AutoSort's map_gt_annotation function
    gt_label_array1 = map_gt_annotation(detect_array, gt_array)
    
    # Calculate detection rate
    detection_rate = np.where(gt_label_array1 > -1)[0].shape[0] / gt_array.shape[0]
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
        # Returns: multi-channel waveform, unit classification labels, noise/non-noise labels, single-channel waveform
        return (
            self.Img[index, ...],      # (n_channels, window_length)
            self.GT[index, ...],       # (n_units,) one-hot
            self.GT_binary[index, ...], # (2,) [noise, spike]
            self.Img_single[index, ...] # (window_length,)
        )


class SimpleAutoSort:
    """
    Simplified AutoSort model (without position information)
    Input: multi-waveform + single-waveform
    Identical to original AutoSort except position information is removed
    """
    def __init__(self, ch_num, samplepoints, device, set_shank_id, save_dir, 
                 pos_weight_noise=None, pos_weight_label=None):
        # Input dimension: (ch_num + 1) * samplepoints (without position information)
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
            train_classification_loss = torch.tensor(0)
            gt_label_class = torch.tensor([])
            pred_class = torch.tensor([])
        
        train_detection_loss = 1000 * self.bceloss(cls_output, labels)
        train_loss = train_detection_loss + train_classification_loss
        
        return train_detection_loss.item(), train_classification_loss.item(), gt, pred, gt_label_class, pred_class


# ==================== 4. Training Function ====================

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
    keep_id_path = model_save_dir + 'keep_id.pkl'
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
    
    # Split training and validation sets
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset split:")
    print(f"  - Training set: {train_size} samples")
    print(f"  - Validation set: {val_size} samples")
    
    # Training parameters
    min_valid_loss = np.inf
    min_val_acc = np.inf  # Track minimum validation accuracy (for early stopping)
    best_acc_epoch = 0  # Epoch with minimum accuracy
    patience_counter = 0  # Early stopping counter
    
    # Check if model already exists
    import os
    if os.path.exists(autosort_model.save_model_path_2):
        autosort_model.load_model()
        print("Loaded existing model")
        return autosort_model, None
    
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
    
    # Save training log
    pd.DataFrame(training_log).to_csv(model_save_dir + 'training_log.csv')
    print(f"\nTraining completed! Training log saved to: {model_save_dir}training_log.csv")
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
    keep_id_path = model_save_dir + 'keep_id.pkl'
    import os
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
    dataset = SimpleWaveformLoader(
        root=str(train_data_dir) + '/',
        shank_channel=np.arange(n_channels),
        Keep_id=train_keep_id  # Use training unit ID list
    )
    
    set_shank_id = dataset.keep_id
    print(f"Model parameters:")
    print(f"  - Number of channels: {n_channels}")
    print(f"  - Window length: {samplepoints}")
    print(f"  - Number of units: {len(set_shank_id)}")
    print(f"  - Using training unit ID list: {set_shank_id == train_keep_id}")
    
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
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nStarting evaluation (total {len(dataset)} samples)...")
    
    # Evaluation
    all_gt_noise = []
    all_pred_noise = []
    all_gt_units = []
    all_pred_units = []
    all_noise_probs = []
    all_unit_probs = []
    
    with torch.no_grad():
        for batch_features, classify_labels, labels, single_waveform in tqdm(test_loader, desc="Evaluating"):
            classify_labels = classify_labels.to(device)
            batch_features = batch_features.view(-1, samplepoints * n_channels).to(device)
            labels = labels.to(device)
            single_waveform = single_waveform.to(device)
            
            # Forward propagation
            codes = torch.cat((batch_features, single_waveform), axis=1)
            
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
                cls_label_output = autosort_model.clsfier_label(codes.float()[test, :])
                unit_probs = torch.softmax(cls_label_output, dim=1)
                pred_units = torch.argmax(cls_label_output, axis=1)
                gt_units = torch.argmax(classify_labels[test, :len(set_shank_id)], axis=1)
                
                all_gt_units.append(gt_units.detach().cpu().numpy())
                all_pred_units.append(pred_units.detach().cpu().numpy())
                all_unit_probs.append(unit_probs.detach().cpu().numpy())
    
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
    
    # Calculate metrics
    noise_accuracy = accuracy_score(all_gt_noise, all_pred_noise)
    
    if len(all_gt_units) > 0:
        unit_f1_score = f1_score(all_gt_units, all_pred_units, average='micro')
        unit_accuracy = accuracy_score(all_gt_units, all_pred_units)
    else:
        unit_f1_score = 0.0
        unit_accuracy = 0.0
    
    print(f"\nEvaluation results (original):")
    print(f"  - Noise classification accuracy: {noise_accuracy:.4f}")
    if len(all_gt_units) > 0:
        print(f"  - Unit classification accuracy: {unit_accuracy:.4f}")
        print(f"  - Unit classification F1 score: {unit_f1_score:.4f}")
        print(f"  - Number of unit samples evaluated: {len(all_gt_units)}")
    print(f"  - Total samples: {len(all_gt_noise)}")
    
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
        
        results_df.to_csv(results_save_dir + 'evaluation_results.csv')
        print(f"\nEvaluation results saved to: {results_save_dir}evaluation_results.csv")
        
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
        summary.to_csv(results_save_dir + 'evaluation_summary.csv', index=False)
        print(f"Evaluation summary saved to: {results_save_dir}evaluation_summary.csv")
    
    return results


# ==================== 6. Optimized Classification Pipeline (Two-Stage) ====================

# Channel position mapping (consistent with generate_neuron_inf_phy_template.py)
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
    Compute cluster position and position_waveform from snippets (reference: generate_neuron_inf_phy_template.py)
    
    Parameters:
        snippets: numpy array, shape (n_spikes, n_channels, window_size)
        channel_id: list of channel IDs
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
        
        # Calculate position_waveform (based on spike position and channel_id channels)
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
    eval_spike_inf: pd.DataFrame = None,
    device=None,
):
    """
    Stage 1: Calibration stage (first 60s)
    
    Process:
    1. Threshold detection
    2. Pass through noise classifier, classified as spikes
    3. Extract way3 layer (100 dimensions)
    4. PCA dimensionality reduction to 30 dimensions
    5. K-means clustering (number of classes = train neurons + n)
    6. Calculate position and waveform for each cluster
    7. Match with train neurons, establish mapping relationship
    
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
        device: device
    
    Returns:
        calibration_results: dictionary, containing:
            - kmeans_model: trained K-means model
            - pca_model: trained PCA model
            - cluster_to_neuron_mapping: mapping from cluster to train neuron
            - cluster_features: features for each cluster (position, waveform, etc.)
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
    spikes = detect_spike(trace0_car, **detection_params)
    spike_coords = np.argwhere(spikes == 1)  # (n_spikes, 2) [time, channel]
    print(f"Number of detected spikes: {len(spike_coords)}")
    
    # 3. Extract waveforms and filter boundaries
    print("\n### 3. Extract waveforms")
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
        
        # Extract waveform (n_channels, window_size)
        waveform = traces[:, start:end]  # (n_channels, window_size)
        waveforms.append(waveform)
        valid_spikes.append((time_idx, channel_idx))
        spike_times.append(time_idx)
        spike_channels.append(channel_idx)
    
    waveforms = np.array(waveforms)  # (n_spikes, n_channels, window_size)
    print(f"Number of valid spikes: {len(waveforms)}")
    
    if len(waveforms) == 0:
        raise ValueError("No valid spikes for calibration")
    
    # 4. Pass through noise classifier, classified as spikes
    print("\n### 4. Noise classifier filtering")
    autosort_model.eval()
    
    # Prepare data
    batch_size = 512
    n_spikes = len(waveforms)
    spike_indices = []
    way3_features = []
    
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
            
            # Concatenate
            codes = torch.cat((batch_multi, batch_single), dim=1)  # (batch, (n_channels+1)*window_size)
            
            # Noise classification
            noise_output = autosort_model.clsfier_noise(codes)
            noise_pred = torch.argmax(noise_output, dim=1)  # 0=noise, 1=spike
            
            # Keep only samples classified as spikes
            spike_mask = noise_pred == 1
            if spike_mask.sum() > 0:
                batch_indices = np.arange(i, min(i+batch_size, n_spikes))[spike_mask.cpu().numpy()]
                spike_indices.extend(batch_indices)
                
                # Extract way3 layer features (only for spike samples)
                codes_spike = codes[spike_mask]
                way3_batch = autosort_model.clsfier_label.intermediate_forward(codes_spike)
                way3_features.append(way3_batch.cpu().numpy())
    
    if len(spike_indices) == 0:
        raise ValueError("No spikes passed noise classifier")
    
    way3_features = np.concatenate(way3_features, axis=0)  # (n_spikes, 100)
    spike_indices = np.array(spike_indices)
    print(f"Number of spikes passing noise classifier: {len(spike_indices)}")
    
    # 5. PCA dimensionality reduction to 30 dimensions
    print("\n### 5. PCA dimensionality reduction")
    pca = PCA(n_components=30)
    way3_pca = pca.fit_transform(way3_features)  # (n_spikes, 30)
    print(f"Feature shape after PCA dimensionality reduction: {way3_pca.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # 6. K-means clustering
    print("\n### 6. K-means clustering")
    n_train_neurons = len(train_neuron_inf)
    n_clusters = n_train_neurons + n_additional_clusters
    print(f"Number of clusters: {n_clusters} (Training neurons: {n_train_neurons}, additional: {n_additional_clusters})")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(way3_pca)  # (n_spikes,)
    print(f"Clustering completed, number of samples per cluster:")
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} samples")
    
    # 7. Calculate position and waveform for each train neuron and each cluster, then match
    print("\n### 7. Calculate cluster position and waveform (based on train neuron channel_id) and match")
    cluster_to_neuron_mapping = {}  # {cluster_id: train_neuron_name}
    neuron_to_clusters = defaultdict(list)  # {train_neuron_name: [cluster_ids]}
    cluster_features = {}  # Save matched cluster features
    
    # Outer loop: iterate through each train neuron
    for train_idx, train_row in train_neuron_inf.iterrows():
        train_neuron = train_row['Neuron']
        train_pos = np.array([train_row['position_1'], train_row['position_2']])
        train_waveform = np.asarray(train_row['position_waveform'], dtype=np.float32)
        
        # Get train neuron channel_id
        train_channel_id = train_row['channel_id']
        if not isinstance(train_channel_id, list):
            if isinstance(train_channel_id, (np.ndarray, tuple)):
                train_channel_id = list(train_channel_id)
            else:
                # Try to parse string
                import ast
                try:
                    train_channel_id = ast.literal_eval(str(train_channel_id))
                    if not isinstance(train_channel_id, list):
                        train_channel_id = [train_channel_id]
                except:
                    print(f"  Warning: Neuron {train_neuron} channel_id cannot be parsed, skipping")
                    continue
        
        if len(train_channel_id) == 0:
            print(f"  Warning: Neuron {train_neuron} has no valid channel_id, skipping")
            continue
        
        print(f"\n  Processing Neuron {train_neuron} (channel_id: {train_channel_id})")
        
        # Inner loop: iterate through each kmeans cluster
        for cluster_id in range(n_clusters):
            # If this cluster is already matched to another neuron, skip (one cluster can only match one neuron)
            if cluster_id in cluster_to_neuron_mapping:
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            
            if len(cluster_spike_indices) == 0:
                continue
            
            # Get all waveforms and channels for this cluster
            cluster_waveforms_full = waveforms[cluster_spike_indices]  # (n_spikes, n_channels, window_size)
            cluster_spike_channels = np.array([spike_channels[i] for i in cluster_spike_indices])  # (n_spikes,)
            
            # Discriminative mechanism: find the dominant channel for this cluster
            # Count occurrences of each channel
            unique_channels, channel_counts = np.unique(cluster_spike_channels, return_counts=True)
            dominant_channel = unique_channels[np.argmax(channel_counts)]
            dominant_channel_ratio = channel_counts.max() / len(cluster_spike_channels)
            
            # Filter: keep only spikes from the dominant channel
            dominant_channel_mask = cluster_spike_channels == dominant_channel
            cluster_spike_indices_filtered = cluster_spike_indices[dominant_channel_mask]
            cluster_spike_indices_excluded = cluster_spike_indices[~dominant_channel_mask]  # Spikes to be marked as unmatch
            cluster_waveforms_filtered = cluster_waveforms_full[dominant_channel_mask]  # (n_filtered_spikes, n_channels, window_size)
            
            if len(cluster_spike_indices_filtered) == 0:
                continue
            
            # Use train neuron channel_id to extract corresponding channels
            # Ensure channel_id is within valid range
            valid_channel_id = [ch for ch in train_channel_id if 0 <= ch < n_channels]
            if len(valid_channel_id) == 0:
                continue
            
            # Extract channels corresponding to train neuron channel_id from filtered cluster_waveforms
            cluster_waveforms = cluster_waveforms_filtered[:, valid_channel_id, :]  # (n_filtered_spikes, n_valid_channels, window_size)
            
            # Calculate position and waveform (using train neuron channel_id, only from dominant channel spikes)
            position_1, position_2, position_waveform = compute_cluster_position_waveform(
                cluster_waveforms, valid_channel_id, window_size
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
            
            # Calculate comprehensive score (smaller distance, higher correlation, higher score)
            score = corr / (1 + pos_distance / position_threshold)
            
            # Establish mapping relationship (one cluster can only match one neuron, choose optimal)
            # If this cluster is not yet matched, or current match has higher score, update
            if cluster_id not in cluster_to_neuron_mapping:
                cluster_to_neuron_mapping[cluster_id] = train_neuron
                neuron_to_clusters[train_neuron].append(cluster_id)
                cluster_features[cluster_id] = {
                    'position_1': position_1,
                    'position_2': position_2,
                    'position_waveform': position_waveform,
                    'n_spikes': len(cluster_spike_indices),
                    'n_spikes_filtered': len(cluster_spike_indices_filtered),
                    'n_spikes_excluded': len(cluster_spike_indices_excluded),
                    'dominant_channel': int(dominant_channel),
                    'dominant_channel_ratio': dominant_channel_ratio,
                    'matched_neuron': train_neuron,
                    'score': score,
                    'pos_distance': pos_distance,
                    'waveform_corr': corr,
                    'cluster_spike_indices_filtered': cluster_spike_indices_filtered,  # Spikes used for matching
                    'cluster_spike_indices_excluded': cluster_spike_indices_excluded,  # Spikes to be marked as unmatch
                }
                print(f"    Cluster {cluster_id} -> {train_neuron} (score: {score:.4f}, distance: {pos_distance:.2f}, correlation: {corr:.4f})")
                print(f"      Dominant channel: {dominant_channel} ({dominant_channel_ratio*100:.1f}%), filtered spikes: {len(cluster_spike_indices_filtered)}/{len(cluster_spike_indices)}, excluded: {len(cluster_spike_indices_excluded)}")
            else:
                # If already matched, compare scores, choose optimal
                existing_neuron = cluster_to_neuron_mapping[cluster_id]
                existing_score = cluster_features[cluster_id]['score']
                if score > existing_score:
                    # Remove old mapping
                    neuron_to_clusters[existing_neuron].remove(cluster_id)
                    # Establish new mapping
                    cluster_to_neuron_mapping[cluster_id] = train_neuron
                    neuron_to_clusters[train_neuron].append(cluster_id)
                    cluster_features[cluster_id] = {
                        'position_1': position_1,
                        'position_2': position_2,
                        'position_waveform': position_waveform,
                        'n_spikes': len(cluster_spike_indices),
                        'n_spikes_filtered': len(cluster_spike_indices_filtered),
                        'n_spikes_excluded': len(cluster_spike_indices_excluded),
                        'dominant_channel': int(dominant_channel),
                        'dominant_channel_ratio': dominant_channel_ratio,
                        'matched_neuron': train_neuron,
                        'score': score,
                        'pos_distance': pos_distance,
                        'waveform_corr': corr,
                        'cluster_spike_indices_filtered': cluster_spike_indices_filtered,  # Spikes used for matching
                        'cluster_spike_indices_excluded': cluster_spike_indices_excluded,  # Spikes to be marked as unmatch
                    }
                    print(f"    Cluster {cluster_id} -> {train_neuron} (updated match, score: {score:.4f} > {existing_score:.4f})")
                    print(f"      Dominant channel: {dominant_channel} ({dominant_channel_ratio*100:.1f}%), filtered spikes: {len(cluster_spike_indices_filtered)}/{len(cluster_spike_indices)}, excluded: {len(cluster_spike_indices_excluded)}")
    
    # Mark unmatched clusters
    print("\n  Unmatched clusters:")
    for cluster_id in range(n_clusters):
        if cluster_id not in cluster_to_neuron_mapping:
            cluster_mask = cluster_labels == cluster_id
            cluster_spike_indices = spike_indices[cluster_mask]
            print(f"    Cluster {cluster_id}: {len(cluster_spike_indices)} spikes")
    
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
    # Initialize predicted_label for all spikes as 'unmatch'
    predicted_labels = ['unmatch'] * len(spike_indices)
    
    # For each matched cluster, assign neuron label only to filtered spikes (dominant channel)
    # Excluded spikes (non-dominant channel) remain as 'unmatch'
    for cluster_id, neuron_name in cluster_to_neuron_mapping.items():
        if cluster_id in cluster_features:
            # Get filtered spike indices (dominant channel spikes)
            # These are indices in the original waveforms array
            filtered_indices = cluster_features[cluster_id].get('cluster_spike_indices_filtered', [])
            if len(filtered_indices) > 0:
                # Find positions of these indices in spike_indices array
                # spike_indices contains all spike indices that passed noise classifier
                # filtered_indices are a subset of spike_indices
                filtered_indices_set = set(filtered_indices)
                for pos, spike_idx in enumerate(spike_indices):
                    if spike_idx in filtered_indices_set:
                        predicted_labels[pos] = neuron_name
    
    results_df = pd.DataFrame({
        'spike_time': [spike_times[i] for i in spike_indices],
        'spike_channel': [spike_channels[i] for i in spike_indices],
        'predicted_label': predicted_labels,
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
    all_way3_features_noise = []  # Way3 features for all detected spikes (100 dimensions)
    all_noise_gt_labels = []  # GT noise/spike labels
    all_noise_pred_labels = []  # Predicted noise/spike labels
    
    # Reprocess all detected spikes to get way3 features
    autosort_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(waveforms), batch_size), desc="Extracting way3 features for all spikes"):
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
            
            # Noise classification
            noise_output = autosort_model.clsfier_noise(codes)
            noise_pred = torch.argmax(noise_output, dim=1)
            
            # Extract way3 features (for all spikes, including those classified as noise)
            way3_batch = autosort_model.clsfier_label.intermediate_forward(codes)
            all_way3_features_noise.append(way3_batch.cpu().numpy())
            all_noise_pred_labels.extend(noise_pred.cpu().numpy().tolist())
            
            # Get GT noise labels (if eval data exists)
            if eval_neuron_inf is not None and eval_spike_inf is not None:
                batch_spike_times = spike_times[i:i+batch_size]
                batch_gt_noise = []
                for st in batch_spike_times:
                    time_diff = (eval_spike_inf['time'] - st).abs()
                    if time_diff.min() <= 1:
                        batch_gt_noise.append(1)  # spike
                    else:
                        batch_gt_noise.append(0)  # noise
                all_noise_gt_labels.extend(batch_gt_noise)
            else:
                all_noise_gt_labels.extend([-1] * len(batch_waveforms))  # Unknown
    
    all_way3_features_noise = np.concatenate(all_way3_features_noise, axis=0) if len(all_way3_features_noise) > 0 else np.array([])
    
    # Calculate 6-channel waveforms and positions for each matched neuron
    print("\n### 8. Calculate 6-channel waveforms for matched neurons")
    neuron_waveforms_6ch = {}  # {neuron_name: {'waveforms': (6, window_size), 'channel_ids': [6], 'channel_positions': [(x, y), ...]}}
    
    for train_neuron in neuron_to_clusters.keys():
        # Get the best cluster for this neuron (highest score)
        best_cluster_id = None
        best_score = -1
        
        for cluster_id in neuron_to_clusters[train_neuron]:
            if cluster_id in cluster_features:
                score = cluster_features[cluster_id].get('score', 0)
                if score > best_score:
                    best_score = score
                    best_cluster_id = cluster_id
        
        if best_cluster_id is None:
            continue
        
        # Get train neuron channel_id
        train_row = train_neuron_inf[train_neuron_inf['Neuron'] == train_neuron]
        if len(train_row) == 0:
            continue
        
        train_channel_id = train_row.iloc[0]['channel_id']
        if not isinstance(train_channel_id, list):
            if isinstance(train_channel_id, (np.ndarray, tuple)):
                train_channel_id = list(train_channel_id)
            else:
                import ast
                try:
                    train_channel_id = ast.literal_eval(str(train_channel_id))
                    if not isinstance(train_channel_id, list):
                        train_channel_id = [train_channel_id]
                except:
                    continue
        
        if len(train_channel_id) == 0:
            continue
        
        # Get cluster waveforms (from filtered spikes, dominant channel only)
        cluster_feat = cluster_features[best_cluster_id]
        cluster_spike_indices_filtered = cluster_feat.get('cluster_spike_indices_filtered', [])
        
        if len(cluster_spike_indices_filtered) == 0:
            continue
        
        # Get waveforms for filtered spikes
        cluster_waveforms_filtered = waveforms[cluster_spike_indices_filtered]  # (n_filtered_spikes, n_channels, window_size)
        
        # Extract 6 channels (or as many as available)
        valid_channel_id = [ch for ch in train_channel_id[:6] if 0 <= ch < n_channels]
        if len(valid_channel_id) == 0:
            continue
        
        # Calculate average waveform for each channel
        channel_waveforms = []
        channel_ids = []
        channel_positions = []
        
        for ch_id in valid_channel_id[:6]:
            # Get channel index in waveforms (assuming waveforms channels are in same order as recording channels)
            # waveforms shape: (n_spikes, n_channels, window_size)
            # n_channels should match recording_f.get_num_channels()
            # We need to map probe channel ID to waveform column index
            # For now, assume channel IDs are 0-based and match column indices
            if ch_id < cluster_waveforms_filtered.shape[1]:
                channel_waveform = cluster_waveforms_filtered[:, ch_id, :].mean(axis=0)  # Average across spikes
                channel_waveforms.append(channel_waveform)
                channel_ids.append(ch_id)
                # Get channel position
                pos = CHANNEL_POSITION.get(ch_id, (0, 0))
                channel_positions.append(pos)
        
        # Pad to 6 channels if needed
        while len(channel_waveforms) < 6:
            channel_waveforms.append(np.zeros(window_size, dtype=np.float32))
            if len(channel_ids) < len(train_channel_id):
                ch_id = train_channel_id[len(channel_ids)]
                channel_ids.append(ch_id)
                pos = CHANNEL_POSITION.get(ch_id, (0, 0))
                channel_positions.append(pos)
            else:
                channel_ids.append(0)
                channel_positions.append((0, 0))
        
        # Convert to numpy array: (6, window_size)
        channel_waveforms_array = np.array(channel_waveforms[:6])
        
        neuron_waveforms_6ch[train_neuron] = {
            'waveforms': channel_waveforms_array,  # (6, window_size)
            'channel_ids': channel_ids[:6],  # List of 6 channel IDs
            'channel_positions': channel_positions[:6],  # List of (x, y) tuples
            'cluster_id': best_cluster_id,
            'n_spikes': len(cluster_spike_indices_filtered),
        }
        
        print(f"  Neuron {train_neuron}: {len(channel_waveforms_array)} channels, {len(cluster_spike_indices_filtered)} spikes")
    
    print(f"Calculated 6-channel waveforms for {len(neuron_waveforms_6ch)} neurons")
    
    calibration_results = {
        'kmeans_model': kmeans,
        'pca_model': pca,
        'cluster_to_neuron_mapping': cluster_to_neuron_mapping,
        'neuron_to_clusters': dict(neuron_to_clusters),
        'cluster_features': cluster_features,
        'spike_indices': spike_indices,
        'cluster_labels': cluster_labels,
        'results_df': results_df,  # Add results_df for confusion matrix
        'way3_features_100d': way3_features,  # Way3 features for spikes passing noise classifier (100 dimensions)
        'way3_features_30d': way3_pca,  # Features after PCA dimensionality reduction (30 dimensions)
        'way3_features_noise_100d': all_way3_features_noise,  # Way3 features for all detected spikes (100 dimensions, for noise detection visualization)
        'noise_gt_labels': np.array(all_noise_gt_labels),  # GT noise/spike labels
        'noise_pred_labels': np.array(all_noise_pred_labels),  # Predicted noise/spike labels
        'neuron_waveforms_6ch': neuron_waveforms_6ch,  # 6-channel waveforms for each matched neuron
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
    Stage 2: Real-time processing (process by time_window)
    
    Process:
    1. Load data by time_window
    2. Threshold detection
    3. Pass through noise classifier, classified as spikes
    4. Extract way3 layer → PCA dimensionality reduction → K-means prediction → Map to train neuron ID
    
    Parameters:
        recording_f: preprocessed recording object
        autosort_model: trained SimpleAutoSort model
        calibration_results: results from calibration stage (contains kmeans_model, pca_model, cluster_to_neuron_mapping)
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
    
    # Get models and mapping from calibration stage
    kmeans_model = calibration_results['kmeans_model']
    pca_model = calibration_results['pca_model']
    cluster_to_neuron_mapping = calibration_results['cluster_to_neuron_mapping']
    
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
        
        # 2. Threshold detection
        trace0_car = traces.T  # (n_timepoints, n_channels)
        spikes = detect_spike(trace0_car, **detection_params)
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
            
            # Extract waveform (n_channels, window_size)
            waveform = traces[:, local_start:local_end]  # (n_channels, window_size)
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
        batch_size = 512
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
        way3_features = np.concatenate(way3_features_list, axis=0)  # (n_spikes_passed, 100)
        way3_spike_indices = np.array(way3_spike_indices)  # Corresponding original spike indices
        
        # Save way3 features
        all_way3_features_100d.append(way3_features)
        
        # Save way3 features for all detected spikes (for noise detection visualization)
        if len(window_noise_way3_features) > 0:
            window_noise_way3_all = np.concatenate(window_noise_way3_features, axis=0)
            all_noise_way3_features_100d.append(window_noise_way3_all)
            all_noise_gt_labels_list.extend(window_noise_gt_labels)
            all_noise_pred_labels_list.extend(window_noise_pred_labels)
        
        # 5. PCA dimensionality reduction
        way3_pca = pca_model.transform(way3_features)  # (n_spikes_passed, 30)
        all_way3_features_30d.append(way3_pca)
        
        # 6. K-means prediction
        cluster_labels = kmeans_model.predict(way3_pca)  # (n_spikes_passed,)
        
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

