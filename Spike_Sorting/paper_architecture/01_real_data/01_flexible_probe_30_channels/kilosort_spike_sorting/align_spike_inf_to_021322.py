#!/usr/bin/env python3
"""
Align spike_inf.csv of every month to the reference month (021322) neuron IDs.

For each month:
  1. Load the month's neuron_inf.pkl and spike_inf.csv.
  2. Compare each neuron in the month to reference neurons from 021322 using
     position distance (< 10 µm) and waveform Pearson correlation (> 0.95).
  3. Build a cluster -> reference neuron mapping based on waveform similarity.
  4. Apply the mapping to spike_inf, writing the aligned reference neuron ID
     to a new column "Neuron". Clusters with no match are set to NA.
  5. Save the result as spike_inf_with_alignment.csv beside the original file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import json


# Reference configuration
BASE_DIR = Path(
    "/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels"
)
SORTING_ROOT = BASE_DIR / "kilosort_spike_sorting"
RESULTS_DIR = SORTING_ROOT / "sorting_results"
REFERENCE_SESSION = "021322"
POSITION_THRESHOLD = 10.0
WAVEFORM_THRESHOLD = 0.95


def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_neuron_inf(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing neuron_inf.pkl: {path}")
    return pd.read_pickle(path)


def compare_neurons(
    ref_neuron: pd.Series, target_neuron: pd.Series
) -> Tuple[float, float]:
    ref_pos = np.array([ref_neuron["position_1"], ref_neuron["position_2"]], dtype=float)
    tgt_pos = np.array([target_neuron["position_1"], target_neuron["position_2"]], dtype=float)
    position_distance = float(np.linalg.norm(ref_pos - tgt_pos))

    ref_waveform = np.asarray(ref_neuron["position_waveform"], dtype=np.float32)
    tgt_waveform = np.asarray(target_neuron["position_waveform"], dtype=np.float32)
    min_len = min(len(ref_waveform), len(tgt_waveform))
    if min_len == 0:
        waveform_corr = -1.0
    else:
        waveform_corr, _ = pearsonr(ref_waveform[:min_len], tgt_waveform[:min_len])

    return position_distance, float(waveform_corr)


def build_neuron_mapping(
    ref_neuron_inf: pd.DataFrame,
    month_neuron_inf: pd.DataFrame,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    for _, month_row in month_neuron_inf.iterrows():
        best_ref_neuron: Optional[str] = None
        best_corr = -1.0

        for _, ref_row in ref_neuron_inf.iterrows():
            pos_dist, corr = compare_neurons(ref_row, month_row)
            if pos_dist <= POSITION_THRESHOLD and corr >= WAVEFORM_THRESHOLD and corr > best_corr:
                best_corr = corr
                best_ref_neuron = ref_row["Neuron"]

        if best_ref_neuron is not None:
            mapping[month_row["Neuron"]] = best_ref_neuron

    return mapping


def ensure_cluster_and_spike_files(
    month_dir: Path, neuron_inf: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cluster_path = month_dir / "cluster_inf.csv"
    spike_original_path = month_dir / "spike_inf.csv"
    spike_with_neuron_path = month_dir / "spike_inf_with_neuron.csv"

    if not cluster_path.exists():
        logging.info("Creating cluster_inf.csv for %s", month_dir.name)
        records = []
        for _, row in neuron_inf.iterrows():
            records.append(
                {
                    "cluster_date": f"{month_dir.name}_rep1_{row['cluster']}",
                    "cluster": row["cluster"],
                    "position_1": row["position_1"],
                    "position_2": row["position_2"],
                    "probe_group": row["probe_group"],
                    "waveform": json.dumps(np.asarray(row["channel_waveform"]).tolist()),
                    "date": "1",
                    "Neuron": row["Neuron"],
                    "position_waveform": json.dumps(
                        np.asarray(row["position_waveform"]).tolist()
                    ),
                    "channel_id": json.dumps(list(row["channel_id"])),
                }
            )
        cluster_inf = pd.DataFrame(records)
        cluster_inf.to_csv(cluster_path, index=False)
    else:
        cluster_inf = pd.read_csv(cluster_path)

    if spike_with_neuron_path.exists():
        spike_inf = pd.read_csv(spike_with_neuron_path, index_col=0)
    elif spike_original_path.exists():
        logging.info("Augmenting spike_inf.csv with neuron labels for %s", month_dir.name)
        spike_inf = pd.read_csv(spike_original_path, index_col=0)
        cluster_to_neuron = cluster_inf.set_index("cluster")["Neuron"].to_dict()
        spike_inf["Neuron"] = spike_inf["cluster"].map(cluster_to_neuron)
        spike_inf.to_csv(spike_with_neuron_path)
    else:
        raise FileNotFoundError(f"Missing spike_inf.csv in {month_dir}")

    return cluster_inf, spike_inf


def align_spike_inf(
    month_dir: Path,
    ref_mapping: Dict[str, str],
    cluster_inf: pd.DataFrame,
    spike_inf: pd.DataFrame,
) -> None:
    cluster_mapping: Dict[str, str] = {}
    if "cluster" in cluster_inf.columns and "Neuron" in cluster_inf.columns:
        cluster_mapping = cluster_inf.set_index("cluster")["Neuron"].to_dict()
    else:
        logging.warning(
            "cluster_inf.csv in %s missing required columns; alignment accuracy may suffer.",
            month_dir.name,
        )

    aligned_neurons = []
    for _, row in spike_inf.iterrows():
        cluster_id = str(row["cluster"])
        month_neuron = cluster_mapping.get(cluster_id)
        if month_neuron is None and "Neuron" in row:
            month_neuron = row["Neuron"]

        if month_neuron is None:
            aligned_neurons.append(np.nan)
        else:
            aligned_neurons.append(ref_mapping.get(month_neuron, np.nan))

    spike_inf["Neuron"] = aligned_neurons
    output_path = month_dir / "spike_inf_with_alignment.csv"
    spike_inf.to_csv(output_path)
    logging.info("Wrote aligned spike info to %s", output_path)


def main() -> None:
    setup_logger()

    ref_dir = RESULTS_DIR / REFERENCE_SESSION
    ref_neuron_inf = load_neuron_inf(ref_dir / "neuron_inf.pkl")

    for month_dir in sorted(RESULTS_DIR.iterdir()):
        if not month_dir.is_dir() or month_dir.name == REFERENCE_SESSION:
            continue
        logging.info("=" * 80)
        logging.info("Processing month %s", month_dir.name)

        try:
            month_neuron_inf = load_neuron_inf(month_dir / "neuron_inf.pkl")
            mapping = build_neuron_mapping(ref_neuron_inf, month_neuron_inf)
            logging.info(
                "Mapped %d neurons from %s to reference %s",
                len(mapping),
                month_dir.name,
                REFERENCE_SESSION,
            )
            cluster_inf, spike_inf = ensure_cluster_and_spike_files(month_dir, month_neuron_inf)
            align_spike_inf(month_dir, mapping, cluster_inf, spike_inf)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to process month %s: %s", month_dir.name, exc)


if __name__ == "__main__":
    main()

