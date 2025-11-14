import json
import math
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from scipy.io import loadmat
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from tqdm import tqdm
from umap import UMAP

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
    detect_local_maxima_in_window,
    label_array1_based_on_array2,
    _normalize_best_channels,
)

warnings.filterwarnings("ignore")

# === Probe 准备，与训练阶段保持一致 ===
probe_data = loadmat("/media/ubuntu/sda/duan/rat/probe/chanMapQPX_mice1.mat")
probe_x = probe_data["xcoords"]
probe_y = probe_data["ycoords"]

probe_position = pd.DataFrame(probe_x)
probe_position[1] = probe_y


def build_probe_instance():
    from probeinterface import Probe

    probe = Probe()
    probe.set_contacts(
        positions=probe_position,
        contact_ids=probe_data["chanMap"][:, 0],
    )
    probe.set_device_channel_indices(range(128))
    return probe


probe_template = build_probe_instance()

# === Clique 计算（全局一次），与训练阶段保持一致 ===
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

print(f"\n创建了{len(model_channel_dict)}个模型组用于Day2评估")

# === 路径与全局参数 ===
day2_recording_raw_path = Path("/home/ubuntu/Downloads/paper/20250613_1.group0.bin")
sorting_day1_dir = Path("/media/ubuntu/sda/duan/rat/sorting_results/day1")
sorting_day2_aligned_dir = Path("/media/ubuntu/sda/duan/rat/sorting_results/day2")
detection_results_root = Path("/media/ubuntu/sda/duan/rat/spike_detection_results")
classification_results_root = Path("/media/ubuntu/sda/duan/rat/spike_classification_results")

recording_raw_day2 = se.read_binary(
    str(day2_recording_raw_path),
    sampling_frequency=30000,
    dtype=np.int16,
    num_channels=128 * 7,
)

device = "cuda"
window_size = 71
half_window = window_size // 2
chunk_size = 120000
batch_size = 1024
calibration_duration_sec = 60
kmeans_corr_threshold = 0.9
kmeans_position_threshold = 10.0

probe_ids = [4, 5]

overall_probe_results = {}


def _load_day1_templates(probe_label: str):
    per_probe_path = sorting_day1_dir / f"{probe_label}_day1_model_templates.pkl"
    if per_probe_path.exists():
        with open(per_probe_path, "rb") as f:
            return pickle.load(f)

    aggregated_path = sorting_day1_dir / "all_probes_day1_model_templates.pkl"
    if aggregated_path.exists():
        with open(aggregated_path, "rb") as f:
            aggregated = pickle.load(f)
        return aggregated.get(probe_label, {})

    legacy_path = sorting_day1_dir / "day1_model_templates.pkl"
    if legacy_path.exists():
        with open(legacy_path, "rb") as f:
            legacy_templates = pickle.load(f)
        return legacy_templates

    return {}


for probe_idx in probe_ids:
    probe_label = f"probe_{probe_idx}"
    print(f"\n{'=' * 120}")
    print(f"开始评估 {probe_label} Day2 数据")
    print(f"{'=' * 120}")

    main_result_dir = detection_results_root / f"{probe_label}_models"
    classification_main_dir = classification_results_root / f"{probe_label}_models"
    classification_main_dir.mkdir(parents=True, exist_ok=True)

    if not main_result_dir.exists():
        print(f"✗ Day1 训练结果目录不存在: {main_result_dir}，跳过 {probe_label}")
        continue

    spike_inf_day2_path = sorting_day2_aligned_dir / f"spike_inf_{probe_label}.tsv"
    cluster_inf_day2_path = sorting_day2_aligned_dir / f"cluster_inf_{probe_label}.csv"

    if not spike_inf_day2_path.exists() or not cluster_inf_day2_path.exists():
        print(f"✗ 缺少 Day2 排序文件 ({spike_inf_day2_path} / {cluster_inf_day2_path})，跳过 {probe_label}")
        continue

    spike_inf_day2 = pd.read_csv(spike_inf_day2_path, sep="\t")
    cluster_inf_day2 = pd.read_csv(cluster_inf_day2_path)

    cluster_inf_day1_path = sorting_day1_dir / f"cluster_inf_{probe_label}.csv"
    if not cluster_inf_day1_path.exists():
        print(f"✗ 缺少 Day1 cluster 信息: {cluster_inf_day1_path}，跳过 {probe_label}")
        continue

    cluster_inf_day1 = pd.read_csv(cluster_inf_day1_path, index_col=0)
    if "Neuron" not in cluster_inf_day1.columns:
        if "cluster_id" in cluster_inf_day1.columns:
            cluster_inf_day1["Neuron"] = cluster_inf_day1["cluster_id"]
        else:
            cluster_inf_day1["Neuron"] = np.nan
    cluster_to_neuron_global = cluster_inf_day1.set_index("cluster_id")["Neuron"].to_dict()

    day1_templates = _load_day1_templates(probe_label)
    if not day1_templates:
        print(f"✗ 未找到 {probe_label} 的 Day1 模板文件，跳过")
        continue

    channel_offset = 128 * (probe_idx - 1)
    channel_ids_for_probe = [channel_offset + c for c in range(128)]

    try:
        recording_day2_raw = recording_raw_day2.select_channels(channel_ids_for_probe)
    except Exception as exc:
        print(f"✗ 选择 {probe_label} 通道时出错: {exc}")
        continue

    print("预处理 Day2 recording...")
    recording_day2 = spre.bandpass_filter(recording_day2_raw, freq_min=300, freq_max=3000)
    recording_day2 = spre.notch_filter(recording_day2, freq=50)
    recording_day2_f = spre.common_reference(recording_day2, reference="global", operator="median")
    recording_day2_f = recording_day2_f.set_probegroup(probe_template)

    total_frames_day2 = recording_day2_f.get_num_samples()
    print(f"{probe_label} Day2 recording 总帧数: {total_frames_day2}")

    potential_spike_tables = []

    for idx, (channels_tuple, model_ids) in enumerate(model_channel_dict.items()):
        model_id = f"model_{model_ids[0]}"
        model_dir = main_result_dir / model_id
        if not model_dir.exists():
            print(f"\n✗ {probe_label}-{model_id} : Day1 训练目录不存在，跳过")
            continue

        model_templates = day1_templates.get(model_id, {})
        if not model_templates:
            print(f"\n✗ {probe_label}-{model_id} : Day1 模板为空，跳过")
            continue

        print(f"\n{'=' * 80}")
        print(f"{probe_label} - 验证第 {idx + 1}/{len(model_channel_dict)} 个模型: {model_id}")
        print(f"{'=' * 80}")

        try:
            group_info = channel_groups[model_id]
            channel_indices = group_info["channel_indices"]

            if {"position_1", "position_2"}.issubset(cluster_inf_day2.columns):
                clique_positions = probe_template.contact_positions[channel_indices]
                if len(clique_positions) >= 3:
                    hull = ConvexHull(clique_positions)
                    hull_path = MplPath(clique_positions[hull.vertices])
                    cluster_positions = cluster_inf_day2[["position_1", "position_2"]].to_numpy()
                    inside_mask = hull_path.contains_points(cluster_positions)
                else:
                    x_min, x_max = clique_positions[:, 0].min(), clique_positions[:, 0].max()
                    y_min, y_max = clique_positions[:, 1].min(), clique_positions[:, 1].max()
                    cluster_positions = cluster_inf_day2[["position_1", "position_2"]].to_numpy()
                    inside_mask = (
                        (cluster_positions[:, 0] >= x_min)
                        & (cluster_positions[:, 0] <= x_max)
                        & (cluster_positions[:, 1] >= y_min)
                        & (cluster_positions[:, 1] <= y_max)
                    )
                clusters_in_clique_day2 = cluster_inf_day2.loc[inside_mask]
            else:
                clusters_in_clique_day2 = cluster_inf_day2.copy()

            if "cluster_id" in clusters_in_clique_day2.columns:
                cluster_ids_in_clique = clusters_in_clique_day2["cluster_id"].astype(int).tolist()
            else:
                cluster_ids_in_clique = []

            if not cluster_ids_in_clique:
                print("警告: Day2 中未找到位于该 clique 的 cluster，跳过")
                continue

            valid_channels = []
            device_indices_in_group = []
            device_positions = []
            device_index_to_valid = {}

            for device_idx in channel_indices:
                device_idx = int(device_idx)
                if device_idx < len(recording_day2_f.channel_ids):
                    device_index_to_valid[device_idx] = len(valid_channels)
                    device_indices_in_group.append(device_idx)
                    valid_channels.append(recording_day2_f.channel_ids[device_idx])
                    device_positions.append(probe_template.contact_positions[device_idx])

            if not valid_channels:
                print(f"错误: {probe_label}-{model_id} 无可用通道，跳过")
                continue

            device_indices_in_group = np.asarray(device_indices_in_group, dtype=int)
            device_positions_array = np.asarray(device_positions, dtype=float)

            print(f"\n开始遍历 Day2 数据块，总帧数 {total_frames_day2}")
            all_valid_indices = []
            all_windows = []

            for start_frame in tqdm(range(0, total_frames_day2, chunk_size), desc=f"{probe_label}-{model_id} chunks"):
                end_frame = min(start_frame + chunk_size, total_frames_day2)
                try:
                    data_chunk = recording_day2_f.get_traces(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        channel_ids=valid_channels,
                    )
                    threshold_result = detect_local_maxima_in_window(
                        data_chunk.T,
                        std_multiplier=3,
                        window_size=60,
                    )
                    threshold_result = np.array(threshold_result) + start_frame
                    valid_indices = threshold_result[
                        (threshold_result >= start_frame + half_window + 1)
                        & (threshold_result < end_frame - half_window)
                    ]
                    for idx_val in valid_indices:
                        rel_idx = idx_val - start_frame
                        window = data_chunk.T[:, rel_idx - half_window : rel_idx + half_window + 1]
                        all_windows.append(window)
                    all_valid_indices.extend(valid_indices)
                except Exception as exc:
                    print(f"处理 chunk 时出错: {exc}")
                    continue

            all_valid_indices = np.array(all_valid_indices)
            all_windows = np.stack(all_windows) if len(all_windows) > 0 else np.array([])

            print(f"\n--- Spike 检测统计 ---")
            print(f"检测到的候选数量: {len(all_valid_indices):,}")
            print(f"提取的时间窗数量: {len(all_windows):,}")

            cluster_col_name = None
            for col in ["cluster", "cluster_id"]:
                if col in spike_inf_day2.columns:
                    cluster_col_name = col
                    break

            if cluster_col_name is None:
                print("警告: spike_inf_day2 中缺少 cluster 列，使用全部数据")
                spike_inf_temp = spike_inf_day2.copy()
            else:
                spike_inf_temp = spike_inf_day2[spike_inf_day2[cluster_col_name].isin(cluster_ids_in_clique)].copy()
                if spike_inf_temp.empty:
                    print("错误: Day2 spike 数据没有匹配的 cluster，跳过")
                    continue

            labels = label_array1_based_on_array2(all_valid_indices, spike_inf_temp["time"], threshold=1)
            cluster_labels_all = cluster_label_array1_based_on_array2(all_valid_indices, spike_inf_temp, threshold=2)
            cluster_labels_all = np.asarray(cluster_labels_all, dtype=int)

            cluster_labels_adjusted = cluster_labels_all.copy()
            cluster_labels_adjusted[labels == 0] = -1

            detected_spike_count = int(np.sum(labels == 1))
            total_detected = len(all_valid_indices)
            total_real_spikes = len(spike_inf_temp)

            print(f"真实 spike 总数: {total_real_spikes:,}")
            print(f"匹配真实 spike 数量: {detected_spike_count:,}")

            model_path = model_dir / "best_model.pth"
            if not model_path.exists():
                print(f"错误: 模型文件缺失 {model_path}，跳过")
                continue

            input_size = all_windows.shape[1] * all_windows.shape[2]
            hidden_size1 = 256
            hidden_size2 = 64
            output_size = 1

            model = Spike_Detection_MLP(
                input_size,
                hidden_size1,
                hidden_size2,
                output_size,
                n_channels=all_windows.shape[1],
                time_window=all_windows.shape[2],
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()

            dataset = SpikeDataset(all_windows, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            all_predictions = []
            all_labels_gt = []
            detection_feature_batches = []
            with torch.no_grad():
                for batch_data, batch_labels in dataloader:
                    batch_data = batch_data.to(device)
                    outputs = model(batch_data)
                    predicted = (outputs > 0.6).float().squeeze()
                    all_predictions.append(predicted.cpu().numpy())
                    all_labels_gt.append(batch_labels.numpy())
                    detection_feature_batches.append(model.extract_features(batch_data).cpu().numpy())

            all_predictions = np.concatenate(all_predictions).astype(int) if all_predictions else np.array([], dtype=int)
            all_labels_gt = np.concatenate(all_labels_gt).astype(int) if all_labels_gt else np.array([], dtype=int)
            detection_features = np.concatenate(detection_feature_batches).astype(np.float32) if detection_feature_batches else np.empty((0, 0), dtype=np.float32)

            if all_predictions.size > 0 and all_labels_gt.size == all_predictions.size:
                tp = np.sum((all_labels_gt == 1) & (all_predictions == 1))
                tn = np.sum((all_labels_gt == 0) & (all_predictions == 0))
                fp = np.sum((all_labels_gt == 0) & (all_predictions == 1))
                fn = np.sum((all_labels_gt == 1) & (all_predictions == 0))
                accuracy = (tp + tn) / all_predictions.size if all_predictions.size else 0.0
                tpr_model = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                print(f"Detection accuracy: {accuracy:.4f}, TPR: {tpr_model:.4f}")
            else:
                print("Detection accuracy: N/A, TPR: N/A (无有效预测)")

            umap_coords = np.full((all_valid_indices.shape[0], 2), np.nan, dtype=float)
            umap_sample_mask = np.zeros(all_valid_indices.shape[0], dtype=bool)
            if detection_features.size > 0:
                n_samples_umap = min(100000, detection_features.shape[0])
                if n_samples_umap > 0:
                    rng_umap = np.random.default_rng(42)
                    sampled_idx_umap = rng_umap.choice(detection_features.shape[0], size=n_samples_umap, replace=False)
                    try:
                        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                        embedding = reducer.fit_transform(detection_features[sampled_idx_umap])
                        umap_coords[sampled_idx_umap] = embedding
                        umap_sample_mask[sampled_idx_umap] = True
                    except Exception as umap_exc:
                        print(f"UMAP 计算失败: {umap_exc}")

            detection_positive_mask = all_predictions == 1
            classification_pred_classes = np.full(all_predictions.shape, -1, dtype=int)
            classification_pred_clusters = np.full(all_predictions.shape, -1, dtype=int)
            classification_pred_neurons = np.full(all_predictions.shape, None, dtype=object)
            predicted_alignment_array = np.zeros(all_predictions.shape, dtype=int)

            classification_dir = classification_main_dir / model_id
            classification_dir.mkdir(parents=True, exist_ok=True)
            kmeans_mapping = {}
            kmeans_template_info = {}
            kmeans_cluster_windows = {}
            kmeans_stats_records = []

            try:
                if detection_features.size > 0:
                    sampling_rate = recording_day2_f.get_sampling_frequency()
                    calibration_mask = all_valid_indices < int(calibration_duration_sec * sampling_rate)
                    calibration_features_all = detection_features[calibration_mask]
                    calibration_windows_all = all_windows[calibration_mask]

                    if calibration_features_all.size > 0:
                        calibration_gt_full = (
                            cluster_labels_adjusted[calibration_mask] if cluster_labels_adjusted.size > 0 else np.array([], dtype=int)
                        )

                        n_clusters = min(len(model_templates), len(calibration_features_all))
                        n_clusters = max(1, n_clusters)
                        kmeans = KMeans(n_clusters=n_clusters + 10, random_state=42, n_init=10)
                        kmeans_labels_all = kmeans.fit_predict(calibration_features_all)

                        for lbl in np.unique(kmeans_labels_all):
                            mask_lbl = kmeans_labels_all == lbl
                            windows_lbl = calibration_windows_all[mask_lbl]
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

                        for lbl, template_info in kmeans_template_info.items():
                            windows_lbl = kmeans_cluster_windows.get(int(lbl))
                            if windows_lbl is None or windows_lbl.size == 0:
                                continue

                            best_match = None
                            best_corr = -1.0
                            best_delta = np.inf

                            for cid, tmpl in model_templates.items():
                                cluster_device_indices = tmpl.get("channel_indices", []) or []
                                subset_indices = [
                                    device_index_to_valid.get(int(dev_idx)) for dev_idx in cluster_device_indices
                                ]
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

                                if corr >= kmeans_corr_threshold and (
                                    np.isnan(delta_pos) or delta_pos <= kmeans_position_threshold
                                ):
                                    if corr > best_corr:
                                        best_corr = corr
                                        best_delta = delta_pos
                                        best_match = int(cid)

                            record = {
                                "kmeans_cluster": int(lbl),
                                "n_samples": template_info["n_samples"],
                                "mapped_cluster_id": best_match if best_match is not None else -1,
                                "mapped_neuron": cluster_to_neuron_global.get(best_match)
                                if best_match is not None
                                else None,
                                "waveform_corr": best_corr if best_corr >= 0 else np.nan,
                                "delta_position": best_delta if np.isfinite(best_delta) else np.nan,
                                "day1_n_spikes": model_templates.get(best_match, {}).get("n_spikes")
                                if best_match is not None
                                else None,
                            }
                            kmeans_stats_records.append(record)
                            if best_match is not None:
                                kmeans_mapping[int(lbl)] = best_match

                        if kmeans_stats_records:
                            mapping_df_day2 = pd.DataFrame(kmeans_stats_records)
                            mapping_df_day2.to_csv(classification_dir / "day2_kmeans_mapping.csv", index=False)

                        unique_gt_clusters = np.unique(calibration_gt_full)
                        unique_gt_clusters = unique_gt_clusters[unique_gt_clusters >= 0]
                        gt_mapping = {}
                        gt_template_info = {}

                        for gt_cluster_id in unique_gt_clusters:
                            mask_gt_cluster = calibration_gt_full == gt_cluster_id
                            windows_gt_all = calibration_windows_all[mask_gt_cluster]
                            if windows_gt_all.size == 0:
                                continue

                            best_match = None
                            best_corr = -1.0
                            best_delta = np.inf
                            best_waveform = None
                            best_position = None

                            for cid, tmpl in model_templates.items():
                                cluster_device_indices = tmpl.get("channel_indices", []) or []
                                subset_indices = [
                                    device_index_to_valid.get(int(dev_idx)) for dev_idx in cluster_device_indices
                                ]
                                subset_indices = [idx_tmp for idx_tmp in subset_indices if idx_tmp is not None]
                                if not subset_indices:
                                    continue
                                subset_indices = np.asarray(subset_indices, dtype=int)

                                windows_gt_subset = windows_gt_all[:, subset_indices, :]
                                if windows_gt_subset.size == 0:
                                    continue

                                rms_gt_subset = np.sqrt(np.mean(windows_gt_subset**2, axis=(0, 2)))
                                if np.allclose(rms_gt_subset.sum(), 0):
                                    continue
                                weights_gt_subset = rms_gt_subset / (rms_gt_subset.sum() + 1e-12)
                                mean_waveform_gt_subset = np.mean(windows_gt_subset, axis=0)
                                synth_waveform_gt_subset = np.sum(mean_waveform_gt_subset * weights_gt_subset[:, None], axis=0)
                                synth_waveform_gt_subset = synth_waveform_gt_subset - np.mean(synth_waveform_gt_subset)
                                std_gt_subset = np.std(synth_waveform_gt_subset)
                                if std_gt_subset > 0:
                                    synth_waveform_gt_subset = synth_waveform_gt_subset / std_gt_subset

                                positions_subset = device_positions_array[subset_indices]
                                cluster_pos_gt_subset = np.sum(positions_subset * weights_gt_subset[:, None], axis=0)

                                ref_waveform = prepare_reference_waveform(tmpl.get("waveform"))
                                if ref_waveform is None or ref_waveform.size != synth_waveform_gt_subset.size:
                                    continue
                                ref_pos = np.asarray(tmpl.get("position"), dtype=float)
                                corr, _ = pearsonr(synth_waveform_gt_subset, ref_waveform)
                                delta_pos = float(np.linalg.norm(cluster_pos_gt_subset - ref_pos)) if ref_pos.size == 2 else np.nan

                                if corr >= kmeans_corr_threshold and (
                                    np.isnan(delta_pos) or delta_pos <= kmeans_position_threshold
                                ):
                                    if corr > best_corr:
                                        best_corr = corr
                                        best_delta = delta_pos
                                        best_match = int(cid)
                                        best_waveform = synth_waveform_gt_subset.astype(np.float32)
                                        best_position = cluster_pos_gt_subset.astype(np.float32)

                            if best_match is not None:
                                gt_mapping[int(gt_cluster_id)] = best_match
                                gt_template_info[int(gt_cluster_id)] = {
                                    "waveform": best_waveform,
                                    "position": best_position,
                                    "n_samples": int(windows_gt_all.shape[0]),
                                }

                        preds_all = kmeans.predict(detection_features)
                        classification_pred_classes = preds_all.astype(int)
                        classification_pred_clusters = np.array(
                            [kmeans_mapping.get(int(lbl), -1) if lbl >= 0 else -1 for lbl in classification_pred_classes],
                            dtype=int,
                        )

                        if kmeans_template_info:
                            n_clusters_plot = len(kmeans_template_info)
                            cols = min(4, n_clusters_plot)
                            rows = math.ceil(n_clusters_plot / cols)
                            fig, axes_grid = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False, sharex=True)
                            axes_flat = np.array(axes_grid).reshape(-1)
                            for ax in axes_flat[n_clusters_plot:]:
                                ax.remove()
                            remaining_axes = fig.axes
                            for ax, cluster_label in zip(remaining_axes, sorted(kmeans_template_info.keys())):
                                waveform = kmeans_template_info[cluster_label]["waveform"]
                                ax.plot(np.arange(len(waveform)), waveform, color="#1f77b4", linewidth=1.0)
                                ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
                                mapped_cluster = kmeans_mapping.get(cluster_label, -1)
                                neuron_name = cluster_to_neuron_global.get(mapped_cluster)
                                title = f"KMeans {cluster_label}"
                                if mapped_cluster >= 0:
                                    title += f"\nDay1 {mapped_cluster}"
                                    if neuron_name:
                                        title += f" ({neuron_name})"
                                else:
                                    title += "\n未映射"
                                ax.set_title(title, fontsize=10)
                                ax.set_xlim(0, len(waveform) - 1)
                                ax.set_xlabel("Sample Index")
                                ax.set_ylabel("Amplitude")
                            plt.tight_layout()
                            fig.savefig(classification_dir / "day2_kmeans_waveforms.pdf")
                            plt.close(fig)

                        for idx_cp, cluster_id in enumerate(classification_pred_clusters):
                            if not detection_positive_mask[idx_cp]:
                                continue
                            if cluster_id >= 0:
                                neuron_name = cluster_to_neuron_global.get(int(cluster_id))
                                if neuron_name is not None:
                                    classification_pred_neurons[idx_cp] = neuron_name

                        predicted_alignment_array = np.zeros_like(predicted_alignment_array)
                        mapped_mask = (classification_pred_clusters >= 0) & detection_positive_mask
                        predicted_alignment_array[mapped_mask] = 1

                        if kmeans_mapping:
                            mapped_cluster_ids = [
                                cid for cid in kmeans_mapping.values() if cid is not None and cid >= 0
                            ]
                            mapped_day1_clusters = len(set(mapped_cluster_ids))
                            mapped_day1_neurons = len(
                                {
                                    cluster_to_neuron_global.get(cid)
                                    for cid in mapped_cluster_ids
                                    if cluster_to_neuron_global.get(cid) is not None
                                }
                            )
                            print(
                                f"与 Day1 匹配的 Day1 cluster 数量: {mapped_day1_clusters}"
                                + (f"，neuron 数量: {mapped_day1_neurons}" if mapped_day1_neurons else "")
                            )
                        else:
                            print("未与 Day1 cluster 建立映射。")

                        print(f"Ground truth 匹配 Day1 cluster 数量: {len(gt_mapping)}")
                    else:
                        print("校准阶段未采集到有效检测特征")
                else:
                    print("缺少检测特征或无样本")
            except Exception as exc:
                print(f"KMeans 映射失败: {exc}")
                import traceback

                traceback.print_exc()
                continue

            gt_cluster_aligned = np.full(cluster_labels_adjusted.shape, -1, dtype=int)
            mask_cluster_ids = cluster_labels_adjusted >= 0
            if mask_cluster_ids.any():
                gt_cluster_aligned[mask_cluster_ids] = np.array(
                    [gt_mapping.get(int(cid), -1) for cid in cluster_labels_adjusted[mask_cluster_ids]],
                    dtype=int,
                )

            classification_gt_int = np.where(all_labels_gt == 1, gt_cluster_aligned, -1).astype(int)
            ground_truth_alignment_col = np.where(
                (all_labels_gt == 1) & (classification_gt_int >= 0), 1, -1
            ).astype(int)
            predicted_spike_classification_col = np.full(all_predictions.shape, -1, dtype=int)
            predicted_spike_classification_col[detection_positive_mask] = classification_pred_clusters[detection_positive_mask]
            predicted_alignment_col = predicted_alignment_array.astype(int)

            potential_spike_df = pd.DataFrame(
                {
                    "time": all_valid_indices.astype(int),
                    "ground_truth_spike_detection": all_labels_gt.astype(int),
                    "predicted_spike_detection": all_predictions.astype(int),
                    "ground_truth_spike_classification": classification_gt_int.astype(int),
                    "predicted_spike_classification": predicted_spike_classification_col.astype(int),
                    "ground_truth_alignment": ground_truth_alignment_col.astype(int),
                    "predicted_alignment": predicted_alignment_col.astype(int),
                    "predicted_neuron": classification_pred_neurons,
                }
            )
            potential_spike_df["UMAP_1"] = umap_coords[:, 0]
            potential_spike_df["UMAP_2"] = umap_coords[:, 1]
            potential_spike_df.loc[~umap_sample_mask, ["UMAP_1", "UMAP_2"]] = None

            potential_spike_df.to_csv(model_dir / "day2_potential_spikes.csv", index=False)
            potential_spike_tables.append((model_id, potential_spike_df))

            print(f"✓ {probe_label}-{model_id} 验证完成")

        except Exception as exc:
            print(f"\n✗ 验证 {probe_label}-{model_id} 时出错: {exc}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{probe_label} 模型验证完成：成功 {len(potential_spike_tables)}/{len(model_channel_dict)} 个模型")
    overall_probe_results[probe_label] = {
        "validated_models": len(potential_spike_tables),
        "total_models": len(model_channel_dict),
    }

print(f"\n{'=' * 120}")
print("所有探针 Day2 评估完成")
for probe_label, stats in overall_probe_results.items():
    print(f"{probe_label}: {stats['validated_models']}/{stats['total_models']} 个模型完成验证")
print(f"{'=' * 120}")
