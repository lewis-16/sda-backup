import json
import math
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from tqdm import tqdm

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import torch
from torch.utils.data import DataLoader

import utils
from utils import (
    SpikeDataset,
    Spike_Detection_MLP,
    cluster_label_array1_based_on_array2,
    create_channel_groups_using_cliques,
    extract_windows,
    label_array1_based_on_array2,
)

warnings.filterwarnings("ignore")

PROBE_IDS = list(range(1, 8))
CALIBRATION_SECONDS = 60.0
INFERENCE_WINDOWS = [1.0, 0.5, 0.25, 0.125]
WINDOW_SIZE = 71
HALF_WINDOW = WINDOW_SIZE // 2
DETECTION_WINDOW = 60
CHUNK_SIZE = 120_000
BATCH_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KMEANS_EXTRA_CLUSTERS = 10
KMEANS_CORR_THRESHOLD = 0.9
KMEANS_POSITION_THRESHOLD = 10.0
RANDOM_STATE = 42

DEFAULT_PROBE_WORKERS = int(os.environ.get("PROBE_WORKERS", min(len(PROBE_IDS), os.cpu_count() or 1)))
DEFAULT_MODEL_WORKERS = int(os.environ.get("MODEL_WORKERS", 3))

DAY2_RECORDING_PATH = Path("/home/ubuntu/Downloads/paper/20250613_1.group0.bin")
SORTING_DAY1_DIR = Path("/media/ubuntu/sda/duan/rat/sorting_results/day1")
SORTING_DAY2_DIR = Path("/media/ubuntu/sda/duan/rat/sorting_results/day2")
DETECTION_RESULTS_ROOT = Path("/media/ubuntu/sda/duan/rat/spike_detection_results")
TIMING_OUTPUT_PATH = Path("/media/ubuntu/sda/duan/rat/spike_detection_timing.json")


def detect_local_maxima_in_window(data, window_size=20, std_multiplier=2):
    per_channel_indices = []
    global_indices_set = set()
    for row in data:
        maxima_indices = []
        row_std = np.std(row.astype(np.float32))
        threshold = std_multiplier * row_std
        for start in range(0, len(row), window_size):
            end = min(start + window_size, len(row))
            window = np.abs(row[start:end])
            if window.size == 0:
                continue
            local_max_index = int(np.argmax(window))
            local_max_value = window[local_max_index]
            if local_max_value > threshold:
                global_idx = start + local_max_index
                maxima_indices.append(global_idx)
                global_indices_set.add(global_idx)
        per_channel_indices.append(np.array(sorted(set(maxima_indices)), dtype=int))
    global_indices = np.array(sorted(global_indices_set), dtype=int)
    return per_channel_indices, global_indices


def build_probe_instance():
    from probeinterface import Probe

    probe = Probe()
    probe.set_contacts(
        positions=probe_position,
        contact_ids=probe_data["chanMap"][:, 0],
    )
    probe.set_device_channel_indices(range(128))
    return probe


def load_model(model_path, window_shape):
    n_channels, time_window = window_shape
    input_size = n_channels * time_window
    hidden_size1 = 256
    hidden_size2 = 64
    output_size = 1

    model = Spike_Detection_MLP(
        input_size,
        hidden_size1,
        hidden_size2,
        output_size,
        n_channels=n_channels,
        time_window=time_window,
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def run_model_inference(model, windows):
    dataset = SpikeDataset(windows, np.zeros(len(windows), dtype=int))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions = []
    features = []
    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(DEVICE)
            outputs = model(batch_data)
            predicted = (outputs > 0.5).float().squeeze()
            features.append(model.extract_features(batch_data).cpu().numpy())
            predictions.append(predicted.cpu().numpy())
    predictions = np.concatenate(predictions).astype(int) if predictions else np.array([], dtype=int)
    features = np.concatenate(features).astype(np.float32) if features else np.empty((0, 0), dtype=np.float32)
    return predictions, features


def prepare_reference_waveform(array_like):
    if array_like is None:
        return None
    arr = np.asarray(array_like, dtype=float)
    if arr.size == 0:
        return None
    arr = arr - np.mean(arr)
    std = np.std(arr)
    if std > 0:
        arr = arr / std
    return arr


def build_kmeans_assets(
    calibration_features,
    calibration_windows,
    model_templates,
    device_positions_array,
    device_index_to_valid,
    cluster_to_neuron_global,
):
    if calibration_features.size == 0:
        return None

    n_clusters = min(len(model_templates), len(calibration_features))
    n_clusters = max(1, n_clusters)
    kmeans = KMeans(n_clusters=n_clusters + KMEANS_EXTRA_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(calibration_features)

    kmeans_cluster_windows = {}
    kmeans_template_info = {}
    for lbl in np.unique(kmeans_labels):
        mask_lbl = kmeans_labels == lbl
        windows_lbl = calibration_windows[mask_lbl]
        if windows_lbl.size == 0:
            continue
        kmeans_cluster_windows[int(lbl)] = windows_lbl
        rms_per_channel = np.sqrt(np.mean(windows_lbl**2, axis=(0, 2)))
        mean_waveform_channels = np.mean(windows_lbl, axis=0)
        total_amp = rms_per_channel.sum()
        if total_amp > 0:
            weights = rms_per_channel / total_amp
        else:
            weights = np.ones(mean_waveform_channels.shape[0], dtype=float) / mean_waveform_channels.shape[0]
        cluster_pos = np.sum(device_positions_array * weights[:, None], axis=0)
        synth_waveform = np.sum(mean_waveform_channels * weights[:, None], axis=0)
        synth_waveform = synth_waveform - np.mean(synth_waveform)
        std_waveform = np.std(synth_waveform)
        if std_waveform > 0:
            synth_waveform = synth_waveform / std_waveform
        kmeans_template_info[int(lbl)] = {
            "waveform": synth_waveform.astype(np.float32),
            "position": cluster_pos.astype(np.float32),
            "n_samples": int(windows_lbl.shape[0]),
        }

    kmeans_mapping = {}
    mapping_records = []
    for lbl, template_info in kmeans_template_info.items():
        windows_lbl = kmeans_cluster_windows.get(int(lbl))
        if windows_lbl is None or windows_lbl.size == 0:
            continue

        best_match = None
        best_corr = -1.0
        best_delta = np.inf

        for cid, tmpl in model_templates.items():
            cluster_device_indices = tmpl.get("channel_indices", []) or []
            subset_indices = [device_index_to_valid.get(int(dev_idx)) for dev_idx in cluster_device_indices]
            subset_indices = [idx_tmp for idx_tmp in subset_indices if idx_tmp is not None]
            if not subset_indices:
                continue
            subset_indices = np.asarray(subset_indices, dtype=int)

            windows_subset = windows_lbl[:, subset_indices, :]
            if windows_subset.size == 0:
                continue

            rms_subset = np.sqrt(np.mean(windows_subset**2, axis=(0, 2)))
            if np.allclose(rms_subset.sum(), 0):
                continue
            weights_subset = rms_subset / (rms_subset.sum() + 1e-12)
            mean_waveform_subset = np.mean(windows_subset, axis=0)
            synth_waveform_subset = np.sum(mean_waveform_subset * weights_subset[:, None], axis=0)
            synth_waveform_subset = synth_waveform_subset - np.mean(synth_waveform_subset)
            std_subset = np.std(synth_waveform_subset)
            if std_subset > 0:
                synth_waveform_subset = synth_waveform_subset / std_subset

            positions_subset = device_positions_array[subset_indices]
            cluster_pos_subset = np.sum(positions_subset * weights_subset[:, None], axis=0)

            ref_waveform = prepare_reference_waveform(tmpl.get("waveform"))
            if ref_waveform is None or ref_waveform.size != synth_waveform_subset.size:
                continue
            ref_pos = np.asarray(tmpl.get("position"), dtype=float)
            corr, _ = pearsonr(synth_waveform_subset, ref_waveform)
            delta_pos = float(np.linalg.norm(cluster_pos_subset - ref_pos)) if ref_pos.size == 2 else np.nan

            if corr >= KMEANS_CORR_THRESHOLD and (np.isnan(delta_pos) or delta_pos <= KMEANS_POSITION_THRESHOLD):
                if corr > best_corr:
                    best_corr = corr
                    best_delta = delta_pos
                    best_match = int(cid)

        record = {
            "kmeans_cluster": int(lbl),
            "n_samples": template_info["n_samples"],
            "mapped_cluster_id": best_match if best_match is not None else -1,
            "mapped_neuron": cluster_to_neuron_global.get(best_match) if best_match is not None else None,
            "waveform_corr": best_corr if best_corr >= 0 else np.nan,
            "delta_position": best_delta if np.isfinite(best_delta) else np.nan,
            "day1_n_spikes": model_templates.get(best_match, {}).get("n_spikes") if best_match is not None else None,
        }
        mapping_records.append(record)
        if best_match is not None:
            kmeans_mapping[int(lbl)] = best_match

    return {
        "kmeans": kmeans,
        "mapping": kmeans_mapping,
        "template_info": kmeans_template_info,
        "records": mapping_records,
    }


def apply_mapping(detection_features, kmeans_assets):
    if detection_features.size == 0:
        return np.array([], dtype=int)
    kmeans = kmeans_assets["kmeans"]
    preds = kmeans.predict(detection_features)
    mapped = np.array([kmeans_assets["mapping"].get(int(lbl), -1) for lbl in preds], dtype=int)
    return mapped


def collect_windows_for_range(recording_f, all_channel_ids, contexts, start_frame, end_frame):
    results = {model_id: {"indices": [], "windows": []} for model_id in contexts}
    for chunk_start in range(start_frame, end_frame, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, end_frame)
        if chunk_end - chunk_start <= WINDOW_SIZE:
            continue
        try:
            data_chunk = recording_f.get_traces(
                start_frame=chunk_start,
                end_frame=chunk_end,
                channel_ids=all_channel_ids,
            ).T
        except Exception as exc:
            print(f"读取数据块 [{chunk_start}, {chunk_end}) 失败: {exc}")
            continue

        per_channel_indices, _ = detect_local_maxima_in_window(
            data_chunk,
            std_multiplier=3,
            window_size=DETECTION_WINDOW,
        )

        for model_id, context in contexts.items():
            positions = context["positions"]
            candidate_set = set()
            for pos in positions:
                spikes = per_channel_indices[pos]
                if spikes.size > 0:
                    candidate_set.update(int(x) for x in spikes)
            if not candidate_set:
                continue

            candidate_indices = np.array(sorted(candidate_set), dtype=int)
            valid_mask = (
                (candidate_indices >= HALF_WINDOW + 1)
                & (candidate_indices < data_chunk.shape[1] - HALF_WINDOW)
            )
            if not np.any(valid_mask):
                continue

            rel_indices = candidate_indices[valid_mask]
            global_indices = rel_indices + chunk_start
            try:
                clique_chunk = data_chunk[positions, :]
                windows = extract_windows(clique_chunk, rel_indices, window_size=WINDOW_SIZE)
            except Exception:
                continue

            results[model_id]["indices"].append(global_indices.astype(int))
            results[model_id]["windows"].append(windows.astype(np.float32))

    for model_id in contexts:
        if results[model_id]["indices"]:
            indices = np.concatenate(results[model_id]["indices"]).astype(int)
            windows = np.concatenate(results[model_id]["windows"]).astype(np.float32)
            sort_order = np.argsort(indices)
            results[model_id]["indices"] = indices[sort_order]
            results[model_id]["windows"] = windows[sort_order]
        else:
            n_channels = len(contexts[model_id]["positions"])
            results[model_id]["indices"] = np.empty((0,), dtype=int)
            results[model_id]["windows"] = np.empty((0, n_channels, WINDOW_SIZE), dtype=np.float32)
    return results


def load_day1_templates(probe_label: str):
    per_probe_path = SORTING_DAY1_DIR / f"{probe_label}_day1_model_templates.pkl"
    if per_probe_path.exists():
        with open(per_probe_path, "rb") as f:
            return pickle.load(f)

    aggregated_path = SORTING_DAY1_DIR / "all_probes_day1_model_templates.pkl"
    if aggregated_path.exists():
        with open(aggregated_path, "rb") as f:
            aggregated = pickle.load(f)
        return aggregated.get(probe_label, {})

    legacy_path = SORTING_DAY1_DIR / "day1_model_templates.pkl"
    if legacy_path.exists():
        with open(legacy_path, "rb") as f:
            legacy_templates = pickle.load(f)
        return legacy_templates

    return {}


def process_model(
    probe_label,
    model_id,
    context,
    main_result_dir,
    calibration_detection_time,
    calib_results,
    window_results,
    window_detection_times,
    cluster_to_neuron_global,
):
    calib_windows = calib_results["windows"]
    if calib_windows.size == 0:
        return None

    model_path = main_result_dir / model_id / "best_model.pth"
    if not model_path.exists():
        print(f"  [WARN] {model_id} 模型文件缺失，跳过")
        return None

    model = load_model(model_path, (calib_windows.shape[1], calib_windows.shape[2]))

    processing_start = time.perf_counter()
    _, calibration_features = run_model_inference(model, calib_windows)
    kmeans_assets = build_kmeans_assets(
        calibration_features,
        calib_windows,
        context["templates"],
        context["device_positions"],
        context["device_index_to_valid"],
        cluster_to_neuron_global,
    )
    if kmeans_assets is None:
        print(f"  [WARN] {model_id} 构建 KMeans 资产失败，跳过")
        return None
    calibration_processing_time = time.perf_counter() - processing_start

    window_entries = {}
    window_counts = {}
    for window_key, detection_time in window_detection_times.items():
        window_data = window_results.get(window_key, {}).get(model_id)
        if not window_data:
            continue
        window_windows = window_data["windows"]
        if window_windows.size == 0:
            continue

        window_processing_start = time.perf_counter()
        _, window_features = run_model_inference(model, window_windows)
        _ = apply_mapping(window_features, kmeans_assets)
        window_processing_time = time.perf_counter() - window_processing_start

        window_entries[window_key] = {
            "detection": detection_time,
            "processing": window_processing_time,
            "n_windows": int(window_windows.shape[0]),
        }
        window_counts[window_key] = int(window_windows.shape[0])

    return {
        "type": "model",
        "probe": probe_label,
        "model": model_id,
        "calibration_detection_sec": calibration_detection_time,
        "calibration_processing_sec": calibration_processing_time,
        "window_times_sec": window_entries,
    }


def process_probe(probe_idx):
    probe_label = f"probe_{probe_idx}"
    entries = []

    print(f"\n{'=' * 120}")
    print(f"开始计算 {probe_label} 流程时间")
    print(f"{'=' * 120}")

    main_result_dir = DETECTION_RESULTS_ROOT / f"{probe_label}_models"
    if not main_result_dir.exists():
        print(f"✗ 缺少 Day1 检测结果目录: {main_result_dir}，跳过")
        return entries

    spike_inf_day2_path = SORTING_DAY2_DIR / f"spike_inf_{probe_label}.tsv"
    cluster_inf_day2_path = SORTING_DAY2_DIR / f"cluster_inf_{probe_label}.csv"
    cluster_inf_day1_path = SORTING_DAY1_DIR / f"cluster_inf_{probe_label}.csv"

    if not spike_inf_day2_path.exists() or not cluster_inf_day2_path.exists():
        print(f"✗ 缺少 Day2 排序文件，跳过 {probe_label}")
        return entries
    if not cluster_inf_day1_path.exists():
        print(f"✗ 缺少 Day1 cluster 信息，跳过 {probe_label}")
        return entries

    spike_inf_day2 = pd.read_csv(spike_inf_day2_path, sep="\t")
    cluster_inf_day2 = pd.read_csv(cluster_inf_day2_path)
    cluster_inf_day1 = pd.read_csv(cluster_inf_day1_path, index_col=0)
    if "Neuron" not in cluster_inf_day1.columns:
        if "cluster_id" in cluster_inf_day1.columns:
            cluster_inf_day1["Neuron"] = cluster_inf_day1["cluster_id"]
        else:
            cluster_inf_day1["Neuron"] = np.nan
    cluster_to_neuron_global = cluster_inf_day1.set_index("cluster_id")["Neuron"].to_dict()

    day1_templates = load_day1_templates(probe_label)
    if not day1_templates:
        print(f"✗ 未找到 {probe_label} 的 Day1 模板文件，跳过")
        return entries

    channel_offset = 128 * (probe_idx - 1)
    channel_ids_for_probe = [channel_offset + c for c in range(128)]

    try:
        recording_raw_day2 = se.read_binary(
            str(DAY2_RECORDING_PATH),
            sampling_frequency=30000,
            dtype=np.int16,
            num_channels=128 * 7,
        )
        recording_day2_raw = recording_raw_day2.select_channels(channel_ids_for_probe)
    except Exception as exc:
        print(f"✗ 选择 {probe_label} 通道时出错: {exc}")
        return entries

    recording_day2 = spre.bandpass_filter(recording_day2_raw, freq_min=300, freq_max=3000)
    recording_day2 = spre.notch_filter(recording_day2, freq=50)
    recording_day2_f = spre.common_reference(recording_day2, reference="global", operator="median")
    recording_day2_f = recording_day2_f.set_probegroup(probe_template)
    sampling_rate = recording_day2_f.get_sampling_frequency()
    total_frames_day2 = recording_day2_f.get_num_samples()

    all_channel_ids = list(recording_day2_f.channel_ids)
    model_contexts = {}
    for channels_tuple, model_ids in model_channel_dict.items():
        model_id = f"model_{model_ids[0]}"
        group_info = channel_groups[model_id]
        channel_indices = group_info["channel_indices"]
        model_templates = day1_templates.get(model_id, {})
        if not model_templates:
            continue

        positions = []
        valid = True
        for contact_idx in channel_indices:
            contact_idx = int(contact_idx)
            if contact_idx >= len(all_channel_ids):
                valid = False
                break
            positions.append(contact_idx)
        if not valid:
            continue

        device_positions_array = np.asarray(probe_template.contact_positions[channel_indices], dtype=float)
        device_index_to_valid = {int(channel_indices[i]): i for i in range(len(channel_indices))}

        model_contexts[model_id] = {
            "positions": positions,
            "channel_indices": channel_indices,
            "device_positions": device_positions_array,
            "device_index_to_valid": device_index_to_valid,
            "templates": model_templates,
        }

    if not model_contexts:
        print(f"✗ {probe_label} 无可用模型，跳过")
        return entries

    calibration_frames = min(int(CALIBRATION_SECONDS * sampling_rate), total_frames_day2)

    calib_detection_start = time.perf_counter()
    calib_results_all = collect_windows_for_range(
        recording_day2_f,
        all_channel_ids,
        model_contexts,
        start_frame=0,
        end_frame=calibration_frames,
    )
    calibration_detection_time = time.perf_counter() - calib_detection_start

    window_results = {}
    window_detection_times = {}
    current_frame = calibration_frames
    for window_sec in INFERENCE_WINDOWS:
        window_frames = int(window_sec * sampling_rate)
        if current_frame >= total_frames_day2 or window_frames <= WINDOW_SIZE:
            continue
        start_frame = current_frame
        end_frame = min(start_frame + window_frames, total_frames_day2)
        if end_frame - start_frame <= WINDOW_SIZE:
            continue

        detection_start = time.perf_counter()
        window_results[str(window_sec)] = collect_windows_for_range(
            recording_day2_f,
            all_channel_ids,
            model_contexts,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        window_detection_times[str(window_sec)] = time.perf_counter() - detection_start
        current_frame = end_frame

    model_entries = []
    model_workers = max(1, min(len(model_contexts), DEFAULT_MODEL_WORKERS))
    models_start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=model_workers) as executor:
        future_to_model = {
            executor.submit(
                process_model,
                probe_label,
                model_id,
                context,
                main_result_dir,
                calibration_detection_time,
                calib_results_all[model_id],
                window_results,
                window_detection_times,
                cluster_to_neuron_global,
            ): model_id
            for model_id, context in model_contexts.items()
        }
        for future in as_completed(future_to_model):
            result = future.result()
            if result is not None:
                model_entries.append(result)
    total_model_time = time.perf_counter() - models_start_time

    entries.extend(model_entries)

    window_processing_summary = {}
    for window_key, detection_time in window_detection_times.items():
        processing_times = [entry["window_times_sec"].get(window_key, {}).get("processing", 0.0) for entry in model_entries]
        window_counts = [entry["window_times_sec"].get(window_key, {}).get("n_windows", 0) for entry in model_entries]
        window_processing_summary[window_key] = {
            "detection": detection_time,
            "processing_max": max(processing_times) if processing_times else 0.0,
            "processing_sum": float(np.sum(processing_times)) if processing_times else 0.0,
            "n_windows_total": int(np.sum(window_counts)) if window_counts else 0,
        }

    probe_entry = {
        "type": "probe_summary",
        "probe": probe_label,
        "models_processed": len(model_entries),
        "calibration_detection_sec": calibration_detection_time,
        "calibration_processing_sec_total": total_model_time,
        "window_summary": window_processing_summary,
    }
    entries.append(probe_entry)

    return entries


probe_data = loadmat("/media/ubuntu/sda/duan/rat/probe/chanMapQPX_mice1.mat")
probe_x = probe_data["xcoords"]
probe_y = probe_data["ycoords"]
probe_position = pd.DataFrame(probe_x)
probe_position[1] = probe_y

probe_template = build_probe_instance()
channel_groups = create_channel_groups_using_cliques(
    probe=probe_template,
    distance_threshold=None,
    min_channels=18,
    max_channels=27,
    target_n_groups=6,
)

model_channel_dict = {}
for model_id, group_info in channel_groups.items():
    channel_tuple = tuple(sorted(group_info["device_channel_indices"]))
    model_channel_dict[channel_tuple] = [int(model_id.split("_")[1])]

print(f"\n创建了{len(model_channel_dict)}个模型组用于时间评估")


def main():
    summary = []

    probe_workers = max(1, DEFAULT_PROBE_WORKERS)
    if probe_workers > 1:
        with ProcessPoolExecutor(max_workers=probe_workers) as executor:
            futures = executor.map(process_probe, PROBE_IDS)
            for result in futures:
                summary.extend(result)
    else:
        for probe_idx in PROBE_IDS:
            summary.extend(process_probe(probe_idx))

    with open(TIMING_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n时间统计结果已保存至: {TIMING_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
