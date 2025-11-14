#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuropixels 384-channel spike-detection training script with clique grouping.

This script converts the notebook-based workflow in
`spike_detection.ipynb` into a reusable Python program.  Channels are
partitioned into partially-overlapping cliques (size range 25-30, minimum
overlap 6, target 15 cliques).  A separate detection model is trained for
each clique and every recording session.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import distance_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from itertools import chain
from math import ceil
from tqdm import tqdm
from umap import UMAP


###############################################################################
# Configuration
###############################################################################

DATA_ROOT = Path(
    "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/"
    "03_Neuropixel_384_channels_visual_stimuli"
)

RAW_DATA_DIR = DATA_ROOT / "raw_data"
SORTING_DIR = DATA_ROOT / "spike_sorting"
RESULTS_DIR = DATA_ROOT / "spike_detection" / "train_results_clique"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_TEMPLATE_PATH = Path(
    "/media/ubuntu/sda/Spike_Sorting/paper_architecture/02_simulation_data/"
    "02_Neuropixel_384_channels/data_generation/recording_neuropixels_type1.h5"
)

SAMPLE_IDS = ["810755797", "810755799", "810755801", "810755803", "810755805", "810755807"]

CLIQUE_SIZE = 50
MIN_CLIQUE_SIZE = 25
MIN_OVERLAP = 16
TARGET_CLIQUE_COUNT = 11
DISTANCE_THRESHOLD_UM = 120.0

WINDOW_SIZE = 31
WINDOW_HALF = WINDOW_SIZE // 2
CHUNK_SIZE = 100_000
STD_MULTIPLIER = 1

SAMPLING_FREQUENCY = 30_000
TARGET_SAMPLING_FREQUENCY = 10_000

TRAIN_BATCH_SIZE = 1024
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 64
NUM_TRAILS = 5
NUM_EPOCHS = 50
PATIENCE = 3
LEARNING_RATE = 1e-4

LABEL_MATCH_THRESHOLD = 1
UMAP_SAMPLE_SIZE = 100_000
RNG = np.random.default_rng(42)

TRAIN_PORTION = 0.2
EVAL_PORTION = 0.2


###############################################################################
# Helper dataclasses and utilities
###############################################################################


@dataclass
class CliqueInfo:
    clique_id: int
    device_channel_indices: List[int]
    contact_ids: List[str]
    center: Tuple[float, float]


def detect_local_minimum_in_window(
    data: np.ndarray,
    window_size: int = 20,
    std_multiplier: float = 2.0,
) -> List[int]:
    minima_indices: List[int] = []
    for row in data:
        row_f = row.astype(np.float32)
        row_mean = float(np.mean(row_f))
        row_std = float(np.std(row_f))
        threshold = row_mean - std_multiplier * row_std
        for start in range(0, len(row), window_size):
            window = row_f[start : start + window_size]
            if window.size == 0:
                continue
            local_min_index = int(np.argmin(window))
            local_min_value = float(window[local_min_index])
            if local_min_value < threshold:
                minima_indices.append(start + local_min_index)
    return list(sorted(set(minima_indices)))


def label_array1_based_on_array2(
    array1: np.ndarray, array2: Sequence[int], threshold: int = 5
) -> np.ndarray:
    sorted_array2 = np.sort(np.asarray(array2, dtype=np.int64))
    labels = np.zeros(len(array1), dtype=np.int8)
    for i, value in enumerate(array1):
        left = value - threshold
        right = value + threshold
        left_index = np.searchsorted(sorted_array2, left, side="left")
        right_index = np.searchsorted(sorted_array2, right, side="right")
        if right_index > left_index:
            labels[i] = 1
    return labels


class SpikeDetectionDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        return self.windows[idx], self.labels[idx]


class SpikeDetectionMLP(nn.Module):
    def __init__(
        self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_size2, 16)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, return_fc3: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        fc3_out = self.fc3(x)
        x = self.relu3(fc3_out)
        logits = self.fc4(x)
        output = self.sigmoid(logits)
        if return_fc3:
            return output, fc3_out
        return output


def evaluate_model(
    model: SpikeDetectionMLP,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for batch_windows, batch_labels in loader:
            batch_windows = batch_windows.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_windows)
            pred_labels = (outputs > 0.5).float()

            total += batch_labels.size(0)
            correct += (pred_labels == batch_labels).sum().item()
            tp += ((pred_labels == 1) & (batch_labels == 1)).sum().item()
            tn += ((pred_labels == 0) & (batch_labels == 0)).sum().item()
            fp += ((pred_labels == 1) & (batch_labels == 0)).sum().item()
            fn += ((pred_labels == 0) & (batch_labels == 1)).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"accuracy": accuracy, "tpr": tpr, "tnr": tnr}


def collect_fc3_features(
    model: SpikeDetectionMLP,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    features: List[np.ndarray] = []
    gt_labels: List[np.ndarray] = []
    pred_labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch_windows, batch_labels in loader:
            batch_windows = batch_windows.to(device)
            batch_labels = batch_labels.to(device)
            outputs, fc3 = model(batch_windows, return_fc3=True)
            preds = (outputs > 0.5).float()
            features.append(fc3.cpu().numpy())
            gt_labels.append(batch_labels.cpu().numpy())
            pred_labels.append(preds.cpu().numpy())
    return (
        np.concatenate(features, axis=0),
        np.concatenate(gt_labels, axis=0).ravel(),
        np.concatenate(pred_labels, axis=0).ravel(),
    )


def plot_umap_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="coolwarm",
        s=5,
        alpha=0.6,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(scatter, label="Label")
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


###############################################################################
# Clique construction
###############################################################################


def build_probe_group() -> se.ProbeGroup:
    print("[INFO] Loading probe template")
    template_recording = se.MEArecRecordingExtractor(file_path=str(PROBE_TEMPLATE_PATH))
    probegroup = template_recording.get_probegroup()
    offset = 0
    for probe in probegroup.probes:
        n_contacts = probe.get_contact_count()
        device_indices = np.arange(offset, offset + n_contacts, dtype=int)
        probe.set_device_channel_indices(device_indices)
        offset += n_contacts
    return probegroup


def build_sliding_cliques(
    probe_group: se.ProbeGroup,
    clique_size: int = CLIQUE_SIZE,
    min_size: int = MIN_CLIQUE_SIZE,
    min_overlap: int = MIN_OVERLAP,
    target_groups: int = TARGET_CLIQUE_COUNT,
) -> List[CliqueInfo]:
    df = probe_group.to_dataframe()
    if "device_channel_indices" in df.columns:
        device_indices = df["device_channel_indices"].astype(int).to_numpy()
    else:
        device_indices = np.arange(len(df), dtype=int)
    positions = df.loc[:, ["x", "y"]].to_numpy()
    contact_ids = df["contact_ids"].astype(str).to_numpy()

    order = np.argsort(positions[:, 1])
    ordered_device = device_indices[order]
    ordered_contacts = contact_ids[order]
    ordered_positions = positions[order]

    step = clique_size - min_overlap
    cliques: List[CliqueInfo] = []

    start_indices = list(range(0, len(ordered_device) - clique_size + 1, step))
    if start_indices[-1] + clique_size < len(ordered_device):
        start_indices.append(len(ordered_device) - clique_size)

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


def plot_clique_groups(
    probe_group: se.ProbeGroup,
    cliques: List[CliqueInfo],
    output_path: Path,
) -> None:
    df = probe_group.to_dataframe()
    positions = df.loc[:, ["x", "y"]].to_numpy()
    if "device_channel_indices" in df.columns:
        device_indices = df["device_channel_indices"].astype(int).to_numpy()
    else:
        device_indices = np.arange(len(df), dtype=int)
    index_map = {dev: idx for idx, dev in enumerate(device_indices)}

    x = positions[:, 0]
    y = positions[:, 1]

    if len(np.unique(x)) > 1:
        x_spacing = np.min(np.diff(np.sort(np.unique(x))))
    else:
        x_spacing = 20.0
    if len(np.unique(y)) > 1:
        y_spacing = np.min(np.diff(np.sort(np.unique(y))))
    else:
        y_spacing = 20.0
    rect_width = x_spacing * 0.8
    rect_height = y_spacing * 0.6

    n_groups = len(cliques)
    cols = 5
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 6))
    axes = np.atleast_1d(axes).flatten()

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_center = (x_min + x_max) / 2
    x_range = x_max - x_min
    x_min_plot = x_center - x_range
    x_max_plot = x_center + x_range
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 50.0

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n_groups:
            ax.axis("off")
            continue
        clique = cliques[ax_idx]
        ax.set_title(
            f"Clique {clique.clique_id:02d}\n({len(clique.device_channel_indices)} channels)",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_aspect("equal", adjustable="box")

        clique_indices = {
            index_map[dev] for dev in clique.device_channel_indices if dev in index_map
        }
        for idx, (xx, yy) in enumerate(zip(x, y)):
            if idx in clique_indices:
                facecolor = "#ED7B85"
                alpha = 0.9
            else:
                facecolor = "lightgrey"
                alpha = 0.4
            rect = Rectangle(
                (xx - rect_width / 2, yy - rect_height / 2),
                rect_width,
                rect_height,
                facecolor=facecolor,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05, wspace=0.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)
    print(f"[INFO] Clique visualization saved to {output_path}")
    plt.close(fig)


###############################################################################
# Training pipeline
###############################################################################


def load_recording(sample_id: str, probe_group: se.ProbeGroup):
    raw_file = RAW_DATA_DIR / sample_id / "spike_band.dat"
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw data for {sample_id}: {raw_file}")

    recording = se.read_binary(
        file_paths=str(raw_file),
        sampling_frequency=SAMPLING_FREQUENCY,
        num_channels=384,
        dtype=np.int16,
    )
    recording = recording.set_probegroup(probe_group)
    return recording


def preprocess_recording(recording):
    print("[INFO]   Resampling recording to 10 kHz")
    recording = spre.resample(recording, TARGET_SAMPLING_FREQUENCY)
    print("[INFO]   Applying bandpass filter and common reference")
    recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=3000)
    recording_f = spre.common_reference(recording_f, reference="global", operator="median")
    return recording_f


def load_sorting_info(sample_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spike_inf_path = SORTING_DIR / sample_id / "spike_inf.tsv"
    cluster_inf_path = SORTING_DIR / sample_id / "cluster_inf.csv"
    if not spike_inf_path.exists() or not cluster_inf_path.exists():
        raise FileNotFoundError(
            f"Missing spike/cluster info for {sample_id}:\n"
            f"  spike_inf: {spike_inf_path}\n  cluster_inf: {cluster_inf_path}"
        )
    spike_inf = pd.read_csv(spike_inf_path, index_col=0, sep="\t")
    cluster_inf = pd.read_csv(cluster_inf_path, index_col=0)
    return spike_inf, cluster_inf


def select_clusters_for_clique(
    cluster_inf: pd.DataFrame, clique_center: Tuple[float, float], distance_threshold: float
) -> List[int]:
    if {"position_1", "position_2"}.issubset(cluster_inf.columns):
        positions = cluster_inf.loc[:, ["position_1", "position_2"]].to_numpy()
        distances = np.linalg.norm(positions - clique_center, axis=1)
        mask = distances <= distance_threshold
        return cluster_inf.loc[mask, "cluster_id"].astype(int).tolist()
    if "cluster_id" in cluster_inf.columns:
        return cluster_inf["cluster_id"].astype(int).tolist()
    fallback_col = cluster_inf.columns[0]
    print(
        f"[WARN] Cluster positions unavailable; using column '{fallback_col}' "
        "to select clusters."
    )
    return cluster_inf[fallback_col].astype(int).tolist()


def extract_windows(
    recording_f,
    channel_ids: Sequence[int],
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    data_chunk = recording_f.get_traces(
        start_frame=start_frame,
        end_frame=end_frame,
        channel_ids=list(channel_ids),
    )
    return data_chunk


def process_clique(
    sample_id: str,
    clique: CliqueInfo,
    recording_f,
    spike_inf: pd.DataFrame,
    cluster_inf: pd.DataFrame,
    device: str,
) -> None:
    print(f"\n[INFO] ----- Sample {sample_id} | Clique {clique.clique_id:02d} -----")
    print(f"[INFO]   Channels: {len(clique.device_channel_indices)}")
    output_dir = RESULTS_DIR / sample_id / f"clique_{clique.clique_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = recording_f.get_num_samples()
    all_valid_indices: List[int] = []
    all_windows: List[np.ndarray] = []

    cluster_ids = select_clusters_for_clique(cluster_inf, clique.center, DISTANCE_THRESHOLD_UM)
    if not cluster_ids:
        print("[WARN]   No clusters associated with this clique; skipping.")
        return

    spike_subset = spike_inf[spike_inf["cluster"].isin(cluster_ids)].copy()
    if spike_subset.empty:
        print("[WARN]   Spike info subset is empty; skipping clique.")
        return

    gt_times = np.sort(spike_subset["time"].to_numpy(dtype=np.int64))
    gt_start_sample = int(gt_times[0])
    gt_end_sample = int(gt_times[-1])
    train_count = max(1, int(len(gt_times) * TRAIN_PORTION))
    eval_count = max(1, int(len(gt_times) * EVAL_PORTION))

    train_region_end = int(gt_times[train_count - 1] + WINDOW_HALF + 1)
    eval_region_start = int(gt_times[-eval_count] - WINDOW_HALF - 1)
    eval_region_end = int(gt_end_sample + WINDOW_HALF + 1)

    train_region_start = max(gt_start_sample - WINDOW_HALF - 1, 0)
    eval_region_start = max(eval_region_start, train_region_end + 1)
    eval_region_end = min(eval_region_end, total_samples)
    train_region_end = min(train_region_end, eval_region_start)

    if eval_region_start >= eval_region_end:
        eval_region_start = max(train_region_end + CHUNK_SIZE, train_region_end + WINDOW_SIZE)
        eval_region_start = min(eval_region_start, total_samples - WINDOW_SIZE)
        eval_region_end = min(total_samples, eval_region_start + (gt_end_sample - gt_start_sample))

    eval_region_start = max(eval_region_start, train_region_end + 1)
    if eval_region_start >= eval_region_end:
        print("[WARN]   Insufficient labeled coverage for evaluation; skipping clique.")
        return

    total_iters = (
        ceil(max(train_region_end - train_region_start, 0) / CHUNK_SIZE)
        + ceil(max(eval_region_end - eval_region_start, 0) / CHUNK_SIZE)
    )
    iter_ranges = chain(
        range(train_region_start, train_region_end, CHUNK_SIZE),
        range(eval_region_start, eval_region_end, CHUNK_SIZE),
    )

    for start_frame in tqdm(
        iter_ranges,
        total=total_iters,
        desc=f"Sample {sample_id} Clique {clique.clique_id:02d}",
    ):
        end_frame = min(start_frame + CHUNK_SIZE, total_samples)
        data_chunk = extract_windows(
            recording_f, clique.device_channel_indices, start_frame, end_frame
        )

        peaks = detect_local_minimum_in_window(
            data_chunk.T, window_size=80, std_multiplier=STD_MULTIPLIER
        )
        peaks = np.asarray(peaks, dtype=np.int64) + start_frame
        valid_indices = peaks[
            (peaks >= start_frame + WINDOW_HALF + 1) & (peaks < end_frame - WINDOW_HALF)
        ]
        for idx in valid_indices:
            rel_idx = idx - start_frame
            window = data_chunk.T[:, rel_idx - WINDOW_HALF : rel_idx + WINDOW_HALF + 1]
            all_windows.append(window)
        all_valid_indices.extend(valid_indices.tolist())

    total_duration = total_samples / TARGET_SAMPLING_FREQUENCY

    if len(all_valid_indices) == 0:
        print("[WARN]   No spikes detected for this clique; skipping.")
        return

    all_valid_indices = np.asarray(all_valid_indices, dtype=np.int64)
    all_windows = np.stack(all_windows)
    print(f"[INFO]   Detected {len(all_valid_indices):,} spike candidates")

    labels = label_array1_based_on_array2(
        all_valid_indices, spike_subset["time"].values, threshold=LABEL_MATCH_THRESHOLD
    )
    detected_spike_count = int(labels.sum())
    coverage = detected_spike_count / len(spike_subset) * 100 if len(spike_subset) > 0 else 0.0
    print(
        f"[INFO]   Ground-truth spikes: {len(spike_subset):,} | "
        f"Matches: {detected_spike_count:,} | Coverage: {coverage:.2f}%"
    )

    train_mask = (all_valid_indices >= train_region_start) & (all_valid_indices < train_region_end)
    eval_mask = (all_valid_indices >= eval_region_start) & (all_valid_indices < eval_region_end)

    train_windows = all_windows[train_mask]
    eval_windows = all_windows[eval_mask]
    train_labels = labels[train_mask]
    eval_labels = labels[eval_mask]

    train_start_time = train_region_start / TARGET_SAMPLING_FREQUENCY
    train_end_time = train_region_end / TARGET_SAMPLING_FREQUENCY
    train_duration = train_end_time - train_start_time
    eval_start_time = eval_region_start / TARGET_SAMPLING_FREQUENCY
    eval_end_time = eval_region_end / TARGET_SAMPLING_FREQUENCY
    eval_duration = eval_end_time - eval_start_time
    total_duration = total_samples / TARGET_SAMPLING_FREQUENCY
    print(
        f"[INFO]   Training window: {train_start_time:.2f}s - "
        f"{train_end_time:.2f}s | "
        f"Evaluation window: {eval_start_time:.2f}s - "
        f"{eval_end_time:.2f}s"
    )

    train_pos_idx = np.where(train_labels == 1)[0]
    train_neg_idx = np.where(train_labels == 0)[0]
    eval_pos_count = int(eval_labels.sum())
    eval_neg_count = len(eval_labels) - eval_pos_count

    print(
        f"[INFO]   Training candidates: {len(train_labels):,} "
        f"(pos {len(train_pos_idx):,} / neg {len(train_neg_idx):,})"
    )
    print(
        f"[INFO]   Evaluation candidates: {len(eval_labels):,} "
        f"(pos {eval_pos_count:,} / neg {eval_neg_count:,})"
    )

    if len(train_pos_idx) == 0:
        print("[WARN]   No positive samples in training window; skipping clique.")
        return

    if len(train_neg_idx) == 0:
        print("[WARN]   No negative samples in training window; skipping clique.")
        return

    balanced_train_count = min(len(train_pos_idx), len(train_neg_idx))
    if balanced_train_count == 0:
        print("[WARN]   Unable to balance training samples; skipping clique.")
        return

    if len(train_pos_idx) > balanced_train_count:
        train_pos_idx = RNG.choice(train_pos_idx, balanced_train_count, replace=False)
    if len(train_neg_idx) > balanced_train_count:
        train_neg_idx = RNG.choice(train_neg_idx, balanced_train_count, replace=False)

    selected_train_indices = np.concatenate([train_pos_idx, train_neg_idx])
    RNG.shuffle(selected_train_indices)

    train_windows_balanced = train_windows[selected_train_indices]
    train_labels_balanced = train_labels[selected_train_indices]

    print(
        f"[INFO]   Balanced training set -> total {len(train_labels_balanced):,} "
        f"(pos {balanced_train_count:,} / neg {balanced_train_count:,})"
    )

    if eval_pos_count > 0 and eval_neg_count > 0:
        eval_pos_idx = np.where(eval_labels == 1)[0]
        eval_neg_idx = np.where(eval_labels == 0)[0]
        balanced_eval_count = min(len(eval_pos_idx), len(eval_neg_idx))
        if balanced_eval_count > 0:
            eval_pos_idx = (
                RNG.choice(eval_pos_idx, balanced_eval_count, replace=False)
                if len(eval_pos_idx) > balanced_eval_count
                else eval_pos_idx
            )
            eval_neg_idx = (
                RNG.choice(eval_neg_idx, balanced_eval_count, replace=False)
                if len(eval_neg_idx) > balanced_eval_count
                else eval_neg_idx
            )
            balanced_eval_indices = np.concatenate([eval_pos_idx, eval_neg_idx])
            RNG.shuffle(balanced_eval_indices)
            eval_windows = eval_windows[balanced_eval_indices]
            eval_labels = eval_labels[balanced_eval_indices]
            print(
                f"[INFO]   Balanced evaluation set -> total {len(eval_labels):,} "
                f"(pos {balanced_eval_count:,} / neg {balanced_eval_count:,})"
            )
    elif eval_pos_count == 0:
        print("[WARN]   No positive samples in evaluation window; metrics may be degenerate.")
    elif eval_neg_count == 0:
        print("[WARN]   No negative samples in evaluation window; metrics may be degenerate.")

    # 保存clique元信息
    info_payload = {
        "sample_id": str(sample_id),
        "clique_id": int(clique.clique_id),
        "device_channel_indices": [int(x) for x in clique.device_channel_indices],
        "contact_ids": [str(x) for x in clique.contact_ids],
        "center": {"x": float(clique.center[0]), "y": float(clique.center[1])},
        "cluster_ids": [int(x) for x in cluster_ids],
        "total_duration_sec": float(total_duration),
        "train_duration_sec": float(train_duration),
        "eval_duration_sec": float(eval_duration),
        "train_start_sec": float(train_start_time),
        "train_end_sec": float(train_end_time),
        "eval_start_sec": float(eval_start_time),
        "eval_end_sec": float(eval_end_time),
        "chunk_size_samples": int(CHUNK_SIZE),
        "chunk_size_sec": float(CHUNK_SIZE / TARGET_SAMPLING_FREQUENCY),
        "candidate_spikes": int(len(all_valid_indices)),
        "matched_spikes": int(detected_spike_count),
        "ground_truth_spikes": int(len(spike_subset)),
        "coverage_percent": float(coverage),
        "train_candidate_spikes": int(train_mask.sum()),
        "eval_candidate_spikes": int(eval_mask.sum()),
        "train_positive_spikes": int(train_labels.sum()),
        "eval_positive_spikes": int(eval_pos_count),
        "window_size": int(WINDOW_SIZE),
        "label_match_threshold": int(LABEL_MATCH_THRESHOLD),
    }
    with open(output_dir / "clique_info.json", "w", encoding="utf-8") as f:
        json.dump(info_payload, f, indent=2)

    train_dataset_balanced = SpikeDetectionDataset(train_windows_balanced, train_labels_balanced)
    train_size = int(0.8 * len(train_dataset_balanced))
    test_size = len(train_dataset_balanced) - train_size
    train_dataset, test_dataset = random_split(train_dataset_balanced, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    input_size = train_windows_balanced.shape[1] * train_windows_balanced.shape[2]
    criterion = nn.BCELoss()

    accuracy_history: List[float] = []
    tpr_history: List[float] = []
    tnr_history: List[float] = []
    best_overall = {"tpr": -1.0, "path": None, "trail": None, "metrics": None}

    for trail in range(1, NUM_TRAILS + 1):
        print(f"[INFO]   Trail {trail}/{NUM_TRAILS}")
        model = SpikeDetectionMLP(input_size, HIDDEN_SIZE1, HIDDEN_SIZE2, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_tpr = 0.0
        best_metrics = {"acc": 0.0, "tpr": 0.0, "tnr": 0.0, "epoch": 0}
        patience_counter = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for batch_windows, batch_labels in train_loader:
                batch_windows = batch_windows.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                preds = model(batch_windows)
                loss = criterion(preds, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(train_loader), 1)
            metrics = evaluate_model(model, test_loader, device)
            accuracy = metrics["accuracy"]
            tpr = metrics["tpr"]
            tnr = metrics["tnr"]

            print(
                f"[INFO]     Epoch {epoch:02d} | Loss {avg_loss:.4f} | "
                f"Acc {accuracy * 100:.2f}% | TPR {tpr * 100:.2f}% | TNR {tnr * 100:.2f}%"
            )

            if tpr > best_tpr:
                best_tpr = tpr
                best_metrics = {"acc": accuracy, "tpr": tpr, "tnr": tnr, "epoch": epoch}
                model_path = output_dir / f"trail_{trail}.pt"
                torch.save(model.state_dict(), model_path)
                print(f"[INFO]       Saved new best model (epoch {epoch})")
                patience_counter = 0
                if tpr > best_overall["tpr"]:
                    best_overall = {
                        "tpr": tpr,
                        "path": str(model_path),
                        "trail": trail,
                        "metrics": best_metrics.copy(),
                    }
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(
                        f"[INFO]       Early stopping at epoch {epoch} "
                        f"(best TPR {best_tpr * 100:.2f}%)"
                    )
                    break

        accuracy_history.append(best_metrics["acc"])
        tpr_history.append(best_metrics["tpr"])
        tnr_history.append(best_metrics["tnr"])
        with open(output_dir / f"trail_{trail}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": float(best_metrics["acc"]),
                    "tpr": float(best_metrics["tpr"]),
                    "tnr": float(best_metrics["tnr"]),
                    "epoch": int(best_metrics["epoch"]),
                },
                f,
                indent=2,
            )

    eval_metrics: Dict[str, float] | None = None
    if best_overall["path"] is None:
        print("[WARN]   No trained model was saved; skipping evaluation.")
    elif len(eval_labels) == 0:
        print("[WARN]   Evaluation window contains no candidates; skipping evaluation.")
    else:
        eval_dataset = SpikeDetectionDataset(eval_windows, eval_labels)
        eval_loader = DataLoader(eval_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

        best_model_path = Path(best_overall["path"])
        best_model = SpikeDetectionMLP(input_size, HIDDEN_SIZE1, HIDDEN_SIZE2, 1).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.eval()

        eval_metrics = evaluate_model(best_model, eval_loader, device)
        print(
            f"[INFO]   Evaluation metrics | Acc {eval_metrics['accuracy'] * 100:.2f}% | "
            f"TPR {eval_metrics['tpr'] * 100:.2f}% | TNR {eval_metrics['tnr'] * 100:.2f}%"
        )

        evaluation_payload = {
            "accuracy": float(eval_metrics["accuracy"]),
            "tpr": float(eval_metrics["tpr"]),
            "tnr": float(eval_metrics["tnr"]),
            "samples": int(len(eval_dataset)),
            "umap_sample_count": 0,
        }

        sample_size = min(UMAP_SAMPLE_SIZE, len(eval_dataset))
        if sample_size >= 2:
            sample_indices = RNG.choice(len(eval_dataset), size=sample_size, replace=False)
            sample_windows = eval_windows[sample_indices]
            sample_labels = eval_labels[sample_indices]
            sample_dataset = SpikeDetectionDataset(sample_windows, sample_labels)
            sample_loader = DataLoader(sample_dataset, batch_size=2048, shuffle=False)

            fc3_features, gt_labels, pred_labels = collect_fc3_features(
                best_model, sample_loader, device
            )
            reducer = UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(fc3_features)

            plot_umap_scatter(
                embedding,
                gt_labels,
                f"{sample_id} Clique {clique.clique_id:02d} UMAP (Ground Truth)",
                output_dir / "umap_ground_truth.png",
            )
            plot_umap_scatter(
                embedding,
                pred_labels,
                f"{sample_id} Clique {clique.clique_id:02d} UMAP (Predicted)",
                output_dir / "umap_predicted.png",
            )
            evaluation_payload["umap_sample_count"] = int(sample_size)
        else:
            print("[WARN]   Not enough samples for UMAP visualization; skipping.")

        with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_payload, f, indent=2)

    summary = {
        "accuracy_mean": float(np.mean(accuracy_history)),
        "accuracy_std": float(np.std(accuracy_history)),
        "tpr_mean": float(np.mean(tpr_history)),
        "tpr_std": float(np.std(tpr_history)),
        "tnr_mean": float(np.mean(tnr_history)),
        "tnr_std": float(np.std(tnr_history)),
        "best_trail": int(best_overall["trail"]) if best_overall["trail"] is not None else None,
    }
    if eval_metrics is not None:
        summary.update(
            {
                "eval_accuracy": eval_metrics["accuracy"],
                "eval_tpr": eval_metrics["tpr"],
                "eval_tnr": eval_metrics["tnr"],
            }
        )
    print(
        f"[INFO]   Clique {clique.clique_id:02d} summary | "
        f"Acc {summary['accuracy_mean'] * 100:.2f}±{summary['accuracy_std'] * 100:.2f}% | "
        f"TPR {summary['tpr_mean'] * 100:.2f}±{summary['tpr_std'] * 100:.2f}% | "
        f"TNR {summary['tnr_mean'] * 100:.2f}±{summary['tnr_std'] * 100:.2f}%"
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def process_sample(sample_id: str, cliques: List[CliqueInfo], device: str) -> None:
    print(f"\n[INFO] Processing sample {sample_id}")
    recording = load_recording(sample_id, probe_group)
    recording_f = preprocess_recording(recording)
    spike_inf, cluster_inf = load_sorting_info(sample_id)

    for clique in cliques:
        process_clique(sample_id, clique, recording_f, spike_inf, cluster_inf, device)


###############################################################################
# Entry point
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neuropixels spike detection training.")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=SAMPLE_IDS,
        help="Sample IDs to process (default: all available).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (default: detect automatically).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    probe_group = build_probe_group()
    cliques = build_sliding_cliques(
        probe_group,
        clique_size=CLIQUE_SIZE,
        min_size=MIN_CLIQUE_SIZE,
        min_overlap=MIN_OVERLAP,
        target_groups=TARGET_CLIQUE_COUNT,
    )
    plot_clique_groups(
        probe_group,
        cliques,
        RESULTS_DIR / "clique_groups.pdf",
    )
    for sample in args.samples:
        try:
            process_sample(sample, cliques, args.device)
        except Exception as exc:
            print(f"[ERROR] Failed to process sample {sample}: {exc}")

