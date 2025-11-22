#!/usr/bin/env python3
"""
Generate the consolidated neuron information table (`neuron_inf.pkl`) from
existing Kilosort outputs without re-running sorting.

The pipeline reproduces the post-processing steps from
`spike_sorting_kilosort.ipynb` and adds an extra de-duplication stage:
if two neurons are within `--position-threshold` micrometers and their
position waveforms correlate above `--waveform-threshold`, the neuron
with the larger numeric id is removed.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from probeinterface import read_probeinterface
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CHANNEL_INDICES: Dict[str, List[int]] = {
    "1": [1, 3, 5, 7, 9, 11],
    "2": [13, 15, 17, 19, 21, 23],
    "3": [24, 25, 26, 27, 28, 29],
    "4": [12, 14, 16, 18, 20, 22],
    "5": [0, 2, 4, 6, 8, 10],
}

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


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def setup_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def ensure_neuron_column(df: pd.DataFrame, stage: str) -> None:
    if "Neuron" not in df.columns:
        raise KeyError(f"Column 'Neuron' missing after stage: {stage}")


def list_replicate_dirs(session_dir: Path) -> List[Path]:
    reps = sorted(p for p in session_dir.glob("whole_segment_rep*") if p.is_dir())
    if not reps:
        raise FileNotFoundError(
            f"No replicate directories found under {session_dir}. Expected folders like "
            "'whole_segment_rep1'."
        )
    return reps


def load_recording(raw_file: Path, probe_file: Path) -> any:
    logging.info("Loading raw recording from %s", raw_file)
    recording_raw = se.read_blackrock(file_path=str(raw_file))
    recording_recorded = recording_raw.remove_channels(["98", "31", "32"])
    recording_recorded = recording_recorded.set_probegroup(read_probeinterface(str(probe_file)))
    recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
    recording_cmr = spre.common_reference(recording_f, reference="global", operator="median")
    return recording_cmr


def load_cluster_and_spike(rep_dir: Path, date_label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = rep_dir / "analyzer_kilosort4_binary" / "extensions" / "quality_metrics" / "metrics.csv"
    sorter_dir = rep_dir / "kilosort" / "sorter_output"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing quality metrics CSV: {metrics_path}")
    if not sorter_dir.exists():
        raise FileNotFoundError(f"Missing Kilosort outputs: {sorter_dir}")

    cluster_inf = pd.read_csv(metrics_path)
    cluster_inf.columns = [
        "cluster",
        "num_spikes",
        "firing_rate",
        "presence_ratio",
        "snr",
        "isi_violations_ratio",
        "isi_violations_count",
        "rp_contamination",
        "rp_violations",
        "sliding_rp_violation",
        "amplitude_cutoff",
        "amplitude_median",
        "amplitude_cv_median",
        "amplitude_cv_range",
        "sync_spike_2",
        "sync_spike_4",
        "sync_spike_8",
        "firing_range",
        "drift_ptp",
        "drift_std",
        "drift_mad",
        "sd_ratio",
    ]

    spike_clusters = np.load(sorter_dir / "spike_clusters.npy").astype(str)
    spike_positions = np.load(sorter_dir / "spike_positions.npy").astype(float)
    spike_templates = np.load(sorter_dir / "spike_templates.npy")
    spike_times = np.load(sorter_dir / "spike_times.npy").astype(int)
    tf = np.load(sorter_dir / "tF.npy")[:, 0, :]

    spike_inf = pd.DataFrame(
        np.column_stack([spike_clusters, spike_positions, spike_templates, spike_times, tf]),
        columns=[
            "cluster",
            "position_1",
            "position_2",
            "templates",
            "time",
            "PC_1",
            "PC_2",
            "PC_3",
            "PC_4",
            "PC_5",
            "PC_6",
        ],
    )

    spike_inf["cluster"] = spike_inf["cluster"].astype(str)
    spike_inf["time"] = spike_inf["time"].astype(int)
    spike_inf[["position_1", "position_2"]] = spike_inf[["position_1", "position_2"]].astype(float)

    cluster_inf["cluster"] = cluster_inf["cluster"].astype(str)
    cluster_inf["position_1"] = np.nan
    cluster_inf["position_2"] = np.nan

    for cluster_id, temp in spike_inf.groupby("cluster"):
        cluster_inf.loc[cluster_inf["cluster"] == cluster_id, "position_1"] = temp["position_1"].mean()
        cluster_inf.loc[cluster_inf["cluster"] == cluster_id, "position_2"] = temp["position_2"].mean()

    cluster_inf["probe_group"] = "1"
    for cluster_id, rows in cluster_inf.groupby("cluster"):
        pos1 = rows["position_1"]
        if ((pos1 > 100) & (pos1 < 250)).any():
            cluster_inf.loc[rows.index, "probe_group"] = "2"
        elif ((pos1 > 250) & (pos1 < 400)).any():
            cluster_inf.loc[rows.index, "probe_group"] = "3"
        elif ((pos1 > 400) & (pos1 < 550)).any():
            cluster_inf.loc[rows.index, "probe_group"] = "4"
        elif (pos1 > 550).any():
            cluster_inf.loc[rows.index, "probe_group"] = "5"

    templates = np.load(sorter_dir / "templates.npy")
    cluster_inf["waveform"] = [templates[i] for i in range(templates.shape[0])]

    # Filter with the same rule as the notebook
    cluster_inf = cluster_inf[
        ((cluster_inf["snr"] > 3) & (cluster_inf["num_spikes"] > 100))
        | ((cluster_inf["snr"] <= 3) & (cluster_inf["num_spikes"] > 1000))
    ].copy()
    spike_inf = spike_inf[spike_inf["cluster"].isin(cluster_inf["cluster"])]
    spike_inf = spike_inf[spike_inf["time"] > 200]

    cluster_inf["date"] = date_label
    spike_inf["date"] = date_label

    for idx, row in cluster_inf.iterrows():
        probe_group = str(row["probe_group"])
        selected_channels = CHANNEL_INDICES.get(probe_group)
        if selected_channels is None:
            continue
        cluster_inf.at[idx, "waveform"] = row["waveform"][:, selected_channels]

    logging.info(
        "Loaded replicate %s | clusters: %d | spikes: %d",
        date_label,
        len(cluster_inf),
        len(spike_inf),
    )
    return cluster_inf, spike_inf


def calculate_position(row: pd.Series) -> pd.Series:
    channels = CHANNEL_INDICES[str(row["probe_group"])]
    waveform = row["waveform"]
    a_squared = [float(np.sum(waveform[:, j] ** 2)) for j in range(len(channels))]

    sum_x_a = 0.0
    sum_y_a = 0.0
    sum_a = 0.0

    for channel, amplitude_sq in zip(channels, a_squared):
        x_i, y_i = CHANNEL_POSITION.get(channel, (0.0, 0.0))
        sum_x_a += x_i * amplitude_sq
        sum_y_a += y_i * amplitude_sq
        sum_a += amplitude_sq

    if sum_a == 0:
        return pd.Series({"position_1": 0.0, "position_2": 0.0})
    return pd.Series({"position_1": sum_x_a / sum_a, "position_2": sum_y_a / sum_a})


def calculate_position_waveform(
    row: pd.Series,
    power: int = 2,
) -> np.ndarray:
    x_target = float(row["position_1"])
    y_target = float(row["position_2"])
    channels = CHANNEL_INDICES[str(row["probe_group"])]
    waveforms = row["waveform"]

    distances = []
    for channel in channels:
        x_channel, y_channel = CHANNEL_POSITION.get(channel, (np.nan, np.nan))
        if np.isnan(x_channel) or np.isnan(y_channel):
            continue
        distances.append(np.sqrt((x_target - x_channel) ** 2 + (y_target - y_channel) ** 2))

    if not distances:
        return np.zeros(61, dtype=np.float32)

    distances = np.asarray(distances, dtype=np.float32)
    weights = 1.0 / np.power(distances, power, dtype=np.float32)
    if np.any(distances == 0):
        zero_idx = np.where(distances == 0)[0][0]
        return waveforms[:, zero_idx].astype(np.float32)

    weights /= weights.sum()
    synthesized = np.zeros(61, dtype=np.float32)
    for t in range(61):
        synthesized[t] = float(np.dot(waveforms[t, :], weights))
    return synthesized


def assign_initial_neuron_ids(all_cluster_inf: pd.DataFrame) -> pd.DataFrame:
    all_cluster_inf = all_cluster_inf.copy()
    all_cluster_inf[["position_1", "position_2"]] = all_cluster_inf.apply(
        calculate_position, axis=1
    )
    all_cluster_inf.sort_values(["position_1", "position_2"], inplace=True)
    all_cluster_inf.reset_index(drop=True, inplace=True)

    current_max = 0
    all_cluster_inf["Neuron"] = None

    for idx, row in all_cluster_inf.iterrows():
        previous = all_cluster_inf.iloc[:idx]
        if previous.empty:
            matched = pd.DataFrame()
        else:
            mask = (
                (previous["position_1"] - row["position_1"]).abs() < 3
            ) & ((previous["position_2"] - row["position_2"]).abs() < 5)
            matched = previous[mask]

        if not matched.empty and matched["Neuron"].notna().any():
            all_cluster_inf.at[idx, "Neuron"] = matched["Neuron"].dropna().iloc[-1]
            continue

        current_max += 1
        all_cluster_inf.at[idx, "Neuron"] = f"Neuron_{current_max}"

    logging.info("Assigned initial neuron ids: %d clusters", len(all_cluster_inf))
    ensure_neuron_column(all_cluster_inf, "assign_initial_neuron_ids")
    return all_cluster_inf


def filter_neurons_present_all_days(all_cluster_inf: pd.DataFrame) -> pd.DataFrame:
    if "Neuron" not in all_cluster_inf.columns:
        raise ValueError("Column 'Neuron' missing before cross-day filtering.")
    cross_tab = pd.crosstab(all_cluster_inf["Neuron"], all_cluster_inf["date"])
    presence = (cross_tab > 0).all(axis=1)
    keep_neurons = cross_tab.index[presence]
    filtered = all_cluster_inf[all_cluster_inf["Neuron"].isin(keep_neurons)].copy()
    logging.info("Neurons present across all days: %d", len(keep_neurons))
    ensure_neuron_column(filtered, "filter_neurons_present_all_days")
    return filtered


def compute_position_waveforms(all_cluster_inf: pd.DataFrame) -> pd.DataFrame:
    all_cluster_inf = all_cluster_inf.copy()
    all_cluster_inf["position_waveform"] = all_cluster_inf.apply(
        calculate_position_waveform, axis=1
    )
    return all_cluster_inf


def refine_with_dbscan(all_cluster_inf: pd.DataFrame) -> pd.DataFrame:
    ensure_neuron_column(all_cluster_inf, "before_dbscan")
    original_labels = all_cluster_inf.get("Neuron", pd.Series(dtype=object))
    waveform_dict: Dict[str, pd.DataFrame] = {}
    for neuron, temp in all_cluster_inf.groupby("Neuron"):
        temp = temp.copy()
        temp.index = temp["date"] + "_" + temp["cluster"]
        waveform_dict[neuron] = temp["position_waveform"].apply(pd.Series)

    results: Dict[int, np.ndarray] = {}
    cluster_id = 0

    for df in waveform_dict.values():
        if len(df) == 0:
            continue
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df)
        dbscan = DBSCAN(eps=3, min_samples=1)
        dbscan.fit(principal_components)

        labels = pd.DataFrame({"labels": dbscan.labels_, "cluster_date": df.index})
        labels["date"] = labels["cluster_date"].apply(lambda x: "_".join(x.split("_")[:2]))

        label_counts = labels["labels"].value_counts()
        label_counts = label_counts[label_counts >= len(labels["date"].unique())]

        valid_labels = []
        for label_id in label_counts.index:
            subset = labels[labels["labels"] == label_id]
            if subset["date"].nunique() == len(labels["date"].unique()):
                valid_labels.append(label_id)

        labels = labels[labels["labels"].isin(valid_labels)]
        for label_id in labels["labels"].unique():
            results[cluster_id] = labels.loc[labels["labels"] == label_id, "cluster_date"].values
            cluster_id += 1

    all_cluster_inf = all_cluster_inf.copy()
    if not results:
        logging.warning(
            "DBSCAN refinement produced no stable clusters; retaining initial neuron labels."
        )
    all_cluster_inf["Neuron"] = None
    all_cluster_inf["cluster_date"] = all_cluster_inf["date"] + "_" + all_cluster_inf["cluster"]
    for new_idx, cluster_dates in results.items():
        mask = all_cluster_inf["cluster_date"].isin(cluster_dates)
        all_cluster_inf.loc[mask, "Neuron"] = f"Neuron_{new_idx + 1}"

    if not results:
        all_cluster_inf["Neuron"] = original_labels.values

    all_cluster_inf.dropna(subset=["Neuron"], inplace=True)
    all_cluster_inf["neuron_date"] = all_cluster_inf["date"] + "_" + all_cluster_inf["Neuron"]
    ensure_neuron_column(all_cluster_inf, "refine_with_dbscan")
    return all_cluster_inf


def compute_reference_waveforms(
    recording: any,
    all_cluster_inf: pd.DataFrame,
    all_spike_inf: pd.DataFrame,
    reference_date: str,
) -> pd.DataFrame:
    cluster_ref = all_cluster_inf[all_cluster_inf["date"] == reference_date].copy()
    spike_ref = all_spike_inf[all_spike_inf["date"] == reference_date].copy()

    spike_counts = spike_ref["Neuron"].value_counts()
    keep_neurons = spike_counts[spike_counts >= 8000].index
    cluster_ref = cluster_ref[cluster_ref["Neuron"].isin(keep_neurons)].copy()

    cluster_ref["channel_id"] = cluster_ref["probe_group"].astype(str).map(CHANNEL_INDICES)

    traces = recording.get_traces().astype(np.float32)
    max_frame = traces.shape[1]

    cluster_ref["waveform"] = None
    window_radius = 30
    window_size = 2 * window_radius + 1

    for idx, row in cluster_ref.iterrows():
        neuron = row["Neuron"]
        channel_ids = row["channel_id"]
        if channel_ids is None:
            continue
        neuron_spikes = spike_ref[spike_ref["Neuron"] == neuron]
        if neuron_spikes.empty:
            continue

        waveform_stack = np.zeros((len(neuron_spikes), len(channel_ids), window_size), dtype=np.float32)

        valid_count = 0
        for j, spike_time in enumerate(neuron_spikes["time"].astype(int).values):
            start = max(spike_time - window_radius, 0)
            end = min(spike_time + window_radius + 1, max_frame)
            if end - start != window_size:
                continue
            snippet = traces[channel_ids, start:end]
            waveform_stack[j] = snippet
            valid_count += 1

        if valid_count == 0 or np.count_nonzero(waveform_stack) == 0:
            continue

        waveform_mean = waveform_stack[:valid_count].mean(axis=0)
        cluster_ref.at[idx, "waveform"] = waveform_mean
        cluster_ref.at[idx, "position_waveform"] = calculate_position_waveform(row.assign(waveform=waveform_mean))

    return cluster_ref


def build_neuron_inf(cluster_ref: pd.DataFrame) -> pd.DataFrame:
    neuron_inf_rows = []
    for neuron, temp in cluster_ref.groupby("Neuron"):
        temp = temp.dropna(subset=["waveform", "position_waveform"])
        if temp.empty:
            continue

        position_waveform = np.stack(temp["position_waveform"].values).mean(axis=0)
        channel_waveform = np.stack(temp["waveform"].values).mean(axis=0)
        neuron_inf_rows.append(
            {
                "Neuron": neuron,
                "position_1": float(temp["position_1"].mean()),
                "position_2": float(temp["position_2"].mean()),
                "position_waveform": position_waveform.astype(np.float32),
                "channel_id": temp["channel_id"].iloc[0],
                "channel_waveform": channel_waveform.astype(np.float32),
                "cluster": temp["cluster"].iloc[0],
                "probe_group": temp["probe_group"].iloc[0],
            }
        )
    neuron_inf = pd.DataFrame(neuron_inf_rows)
    logging.info("Constructed neuron_inf with %d neurons", len(neuron_inf))
    return neuron_inf


def extract_neuron_index(neuron_label: str) -> int:
    match = re.search(r"(\d+)", str(neuron_label))
    return int(match.group(1)) if match else 0


def deduplicate_neurons(
    neuron_inf: pd.DataFrame,
    position_threshold: float,
    waveform_threshold: float,
) -> pd.DataFrame:
    if neuron_inf.empty:
        return neuron_inf

    neuron_inf = neuron_inf.copy()
    neuron_inf["neuron_index"] = neuron_inf["Neuron"].apply(extract_neuron_index)
    neuron_inf.sort_values("neuron_index", inplace=True)
    neuron_inf.reset_index(drop=True, inplace=True)

    keep_mask = np.ones(len(neuron_inf), dtype=bool)

    for i in range(len(neuron_inf)):
        if not keep_mask[i]:
            continue
        pos_i = neuron_inf.loc[i, ["position_1", "position_2"]].to_numpy(dtype=float)
        waveform_i = neuron_inf.loc[i, "position_waveform"]
        waveform_i = np.asarray(waveform_i, dtype=np.float32)

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
                keep_mask[j] = False

    removed = np.count_nonzero(~keep_mask)
    if removed:
        logging.info("Deduplicated %d neurons based on position/waveform similarity", removed)
    neuron_inf = neuron_inf[keep_mask].drop(columns=["neuron_index"]).reset_index(drop=True)
    return neuron_inf


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    setup_logger(verbose=False)

    parser = argparse.ArgumentParser(
        description="Generate neuron_inf using either registration or deduplication method"
    )
    parser.add_argument(
        "--method",
        choices=["registration", "dedup"],
        default="registration",
        help="Which method to use to generate neuron_inf. Default: registration (use spike registration notebook flow)",
    )
    args = parser.parse_args()

    logging.info("Selected neuron_inf generation method: %s", args.method)

    base_dir = Path(
        "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels"
    )
    sorting_root = base_dir / "kilosort_spike_sorting"
    data_root = Path("/media/ubuntu/sda/data/mouse6/ns4/natural_image")
    probe_file = Path("/media/ubuntu/sda/data/probe.json")

    session_dirs = sorted(
        p for p in (sorting_root / "sorting_new").iterdir() if p.is_dir()
    )
    if not session_dirs:
        raise FileNotFoundError(f"No session directories found under {sorting_root}/sorting_results")

    for session_dir in session_dirs:
        logging.info("=" * 80)
        logging.info("Processing session %s", session_dir.name)

        raw_candidates = list(data_root.glob(f"mouse6_{session_dir.name}*.ns4"))
        if not raw_candidates:
            logging.warning("No raw file found for session %s, skipping.", session_dir.name)
            continue
        raw_file = raw_candidates[0]

        try:
            recording = load_recording(raw_file, probe_file)
        except FileNotFoundError as exc:
            logging.error("Skipping %s due to missing data: %s", session_dir.name, exc)
            continue

        try:
            all_cluster_list = []
            all_spike_list = []
            replicate_dirs = list_replicate_dirs(session_dir)
            for rep_dir in replicate_dirs:
                date_label = rep_dir.name.split("whole_segment_rep")[-1]
                cluster_inf, spike_inf = load_cluster_and_spike(rep_dir, date_label)
                all_cluster_list.append(cluster_inf)
                all_spike_list.append(spike_inf)

            all_cluster_inf = pd.concat(all_cluster_list, ignore_index=True)
            all_spike_inf = pd.concat(all_spike_list, ignore_index=True)

            if args.method == "registration":
                # ------------------------------------------------------------------
                # Replicate notebook logic for neuron alignment (registration)
                # ------------------------------------------------------------------
                all_cluster_inf[["position_1", "position_2"]] = all_cluster_inf.apply(
                    calculate_position, axis=1
                )
                all_cluster_inf["Neuron"] = None
                current_max_neuron = 1
                if len(all_cluster_inf) > 0:
                    all_cluster_inf.at[0, "Neuron"] = f"Neuron_{current_max_neuron}"
                for i in range(1, len(all_cluster_inf)):
                    current_pos1 = all_cluster_inf.at[i, "position_1"]
                    current_pos2 = all_cluster_inf.at[i, "position_2"]
                    mask = (
                        (all_cluster_inf.loc[: i - 1, "position_1"] - current_pos1).abs().lt(3)
                        & (all_cluster_inf.loc[: i - 1, "position_2"] - current_pos2).abs().lt(5)
                    )
                    matched = all_cluster_inf.loc[: i - 1][mask]
                    if not matched.empty and matched["Neuron"].notna().any():
                        all_cluster_inf.at[i, "Neuron"] = matched["Neuron"].dropna().iloc[-1]
                    else:
                        current_max_neuron += 1
                        all_cluster_inf.at[i, "Neuron"] = f"Neuron_{current_max_neuron}"

                neuron_date = pd.crosstab(all_cluster_inf["Neuron"], all_cluster_inf["date"])
                neuron_date[neuron_date > 1] = 1
                neuron_date = neuron_date.sum(axis=1)
                neuron_date = neuron_date[neuron_date == len(replicate_dirs)]
                neuron_keep = neuron_date.index
                all_cluster_inf = all_cluster_inf[all_cluster_inf["Neuron"].isin(neuron_keep)].copy()
                all_cluster_inf["cluster_date"] = (
                    all_cluster_inf["date"] + "_" + all_cluster_inf["cluster"]
                )

                all_cluster_inf["position_waveform"] = None
                for idx, row in all_cluster_inf.iterrows():
                    all_cluster_inf.at[idx, "position_waveform"] = calculate_position_waveform(
                        row, power=2
                    )

                waveform_dict: Dict[str, pd.DataFrame] = {}
                for neuron_label in all_cluster_inf["Neuron"].unique():
                    temp = all_cluster_inf[all_cluster_inf["Neuron"] == neuron_label]
                    temp = temp.copy()
                    temp.index = temp["cluster_date"]
                    waveform_dict[neuron_label] = temp["position_waveform"].apply(pd.Series)

                num = 0
                results: Dict[int, np.ndarray] = {}
                for df in waveform_dict.values():
                    if df.empty:
                        continue
                    pca = PCA(n_components=2)
                    principal_components = pca.fit_transform(df)
                    dbscan = DBSCAN(eps=3, min_samples=1)
                    dbscan.fit(principal_components)

                    label = pd.DataFrame({"labels": dbscan.labels_, "cluster_date": df.index})
                    label["date"] = label["cluster_date"].apply(lambda x: "_".join(x.split("_")[:2]))

                    remain_label = label["labels"].value_counts()
                    remain_label = remain_label[remain_label >= len(replicate_dirs)]
                    for lbl in remain_label.index.tolist():
                        temp_label = label[label["labels"] == lbl]
                        if temp_label["date"].nunique() != len(replicate_dirs):
                            remain_label = remain_label.drop(lbl)
                    label = label[label["labels"].isin(remain_label.index)]
                    for lbl in label["labels"].unique():
                        results[num] = label.loc[label["labels"] == lbl, "cluster_date"].values
                        num += 1

                all_cluster_inf["Neuron"] = None
                for key, item in results.items():
                    all_cluster_inf.loc[
                        all_cluster_inf["cluster_date"].isin(item), "Neuron"
                    ] = f"Neuron_{key + 1}"

                all_cluster_inf = all_cluster_inf.dropna(subset=["Neuron"]).copy()
                all_cluster_inf["neuron_date"] = all_cluster_inf["date"] + "_" + all_cluster_inf["Neuron"]

                waveform_mean = pd.DataFrame()
                for df in waveform_dict.values():
                    waveform_mean = pd.concat((waveform_mean, df), axis=0)
                waveform_mean = waveform_mean.loc[list(all_cluster_inf["cluster_date"])]

                all_cluster_inf = all_cluster_inf.set_index("cluster_date")
                all_cluster_inf = all_cluster_inf.join(waveform_mean, how="right")
                all_cluster_inf["cluster_date"] = all_cluster_inf.index

                all_spike_inf["cluster_date"] = all_spike_inf["date"] + "_" + all_spike_inf["cluster"]
                all_spike_inf = all_spike_inf[
                    all_spike_inf["cluster_date"].isin(all_cluster_inf["cluster_date"])
                ].copy()
                all_spike_inf["Neuron"] = None
                for i in range(len(all_cluster_inf)):
                    cluster_date = all_cluster_inf.iloc[i]["cluster_date"]
                    neuron_label = all_cluster_inf.iloc[i]["Neuron"]
                    all_spike_inf.loc[
                        all_spike_inf["cluster_date"] == cluster_date, "Neuron"
                    ] = neuron_label

                reference_date = "1"
                all_cluster_inf_rep1 = all_cluster_inf[all_cluster_inf["date"] == reference_date].copy()
                all_spike_inf_rep1 = all_spike_inf[all_spike_inf["date"] == reference_date].copy()

                del_neuron = all_spike_inf_rep1["Neuron"].value_counts()
                del_neuron = del_neuron[del_neuron < 8000].index
                all_cluster_inf_rep1 = all_cluster_inf_rep1[
                    ~all_cluster_inf_rep1["Neuron"].isin(del_neuron)
                ].copy()

                all_cluster_inf_rep1["channel_id"] = None
                for index, row in all_cluster_inf_rep1.iterrows():
                    probe_group = row["probe_group"]
                    if probe_group in CHANNEL_INDICES:
                        all_cluster_inf_rep1.at[index, "channel_id"] = CHANNEL_INDICES[probe_group]

                waveform_matrix = recording.get_traces().astype("float32")
                all_spike_inf_rep1 = all_spike_inf_rep1[
                    all_spike_inf_rep1["time"] < waveform_matrix.shape[0] - 35
                ].copy()

                if "waveform" not in all_cluster_inf_rep1.columns:
                    all_cluster_inf_rep1["waveform"] = [None] * len(all_cluster_inf_rep1)

                for i in range(len(all_cluster_inf_rep1)):
                    neuron_label = all_cluster_inf_rep1["Neuron"].values[i]
                    channel_id = all_cluster_inf_rep1["channel_id"].values[i]
                    if channel_id is None:
                        continue

                    spike_temp = all_spike_inf_rep1[all_spike_inf_rep1["Neuron"] == neuron_label]
                    if spike_temp.empty:
                        continue

                    waveform_temp = waveform_matrix[:, channel_id].T
                    n = len(spike_temp)
                    n_channels = len(channel_id)
                    waveform_length = 61

                    waveform_stack = np.zeros((n, n_channels, waveform_length)).astype(np.float32)

                    for j in range(n):
                        start = spike_temp["time"].values[j] - 30
                        end = spike_temp["time"].values[j] + 31
                        waveform_stack[i, :, :] += waveform_temp[:, start:end]

                    waveform_mean_temp = np.mean(waveform_stack, axis=0)
                    all_cluster_inf_rep1["waveform"].values[i] = waveform_mean_temp.T

                all_cluster_inf_rep1["position_waveform"] = None
                for idx, row in all_cluster_inf_rep1.iterrows():
                    all_cluster_inf_rep1.at[idx, "position_waveform"] = calculate_position_waveform(
                        row, power=2
                    )

                neuron_inf_records = []
                for neuron_label in all_cluster_inf_rep1["Neuron"].unique():
                    temp = all_cluster_inf_rep1[all_cluster_inf_rep1["Neuron"] == neuron_label]
                    if len(temp) > 1:
                        neuron_inf_records.append(
                            [
                                neuron_label,
                                np.mean(temp["position_1"]),
                                np.mean(temp["position_2"]),
                                np.mean(list(temp["position_waveform"]), axis=0),
                                temp["channel_id"].iloc[0],
                                np.stack(temp["waveform"].values).mean(axis=0),
                                temp["cluster"].values[0],
                                temp["probe_group"].values[0],
                            ]
                        )
                    else:
                        neuron_inf_records.append(
                            [
                                neuron_label,
                                temp["position_1"].iloc[0],
                                temp["position_2"].iloc[0],
                                temp["position_waveform"].iloc[0],
                                temp["channel_id"].iloc[0],
                                temp["waveform"].values[0],
                                temp["cluster"].values[0],
                                temp["probe_group"].values[0],
                            ]
                        )

                if neuron_inf_records:
                    neuron_inf = pd.DataFrame(
                        neuron_inf_records,
                        columns=[
                            "Neuron",
                            "position_1",
                            "position_2",
                            "position_waveform",
                            "channel_id",
                            "channel_waveform",
                            "cluster",
                            "probe_group",
                        ],
                    )
                else:
                    neuron_inf = pd.DataFrame(
                        columns=[
                            "Neuron",
                            "position_1",
                            "position_2",
                            "position_waveform",
                            "channel_id",
                            "channel_waveform",
                            "cluster",
                            "probe_group",
                        ]
                    )

                neuron_inf = deduplicate_neurons(
                    neuron_inf,
                    position_threshold=10.0,
                    waveform_threshold=0.95,
                )

                output_path = session_dir / "neuron_inf.pkl"
                with open(output_path, "wb") as f:
                    pickle.dump(neuron_inf, f)

                logging.info(
                    "Session %s complete. Saved %d neurons to %s",
                    session_dir.name,
                    len(neuron_inf),
                    output_path,
                )
            else:
                # ------------------------------------------------------------------
                # Alternative dedup flow (modular): assign initial neuron ids, filter
                # by presence across days, compute position waveforms, refine,
                # compute reference waveforms and build neuron inf, then deduplicate
                # ------------------------------------------------------------------
                all_cluster_inf = assign_initial_neuron_ids(all_cluster_inf)
                all_cluster_inf = filter_neurons_present_all_days(all_cluster_inf)
                all_cluster_inf = compute_position_waveforms(all_cluster_inf)
                all_cluster_inf = refine_with_dbscan(all_cluster_inf)

                # map spikes to Neuron labels
                all_spike_inf["cluster_date"] = all_spike_inf["date"] + "_" + all_spike_inf["cluster"]
                all_cluster_inf["cluster_date"] = all_cluster_inf["date"] + "_" + all_cluster_inf["cluster"]
                all_spike_inf = all_spike_inf[all_spike_inf["cluster_date"].isin(all_cluster_inf["cluster_date"])].copy()
                all_spike_inf["Neuron"] = None
                for idx, row in all_cluster_inf.iterrows():
                    all_spike_inf.loc[all_spike_inf["cluster_date"] == row["cluster_date"], "Neuron"] = row["Neuron"]

                reference_date = "1"
                cluster_ref = compute_reference_waveforms(recording, all_cluster_inf, all_spike_inf, reference_date)
                neuron_inf = build_neuron_inf(cluster_ref)
                neuron_inf = deduplicate_neurons(neuron_inf, position_threshold=10.0, waveform_threshold=0.95)

                output_path = session_dir / "neuron_inf.pkl"
                with open(output_path, "wb") as f:
                    pickle.dump(neuron_inf, f)

                logging.info(
                    "Session %s complete (dedup). Saved %d neurons to %s",
                    session_dir.name,
                    len(neuron_inf),
                    output_path,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to process session %s: %s", session_dir.name, exc)


if __name__ == "__main__":
    main()

