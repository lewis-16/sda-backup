import ast
import json
import os
import pickle
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from probeinterface import Probe, read_probeinterface, write_probeinterface
from scipy.io import loadmat
from scipy.spatial import ConvexHull
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from collections import defaultdict
import json
import pickle
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import spikeinterface as si
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from umap import UMAP
import matplotlib.patches as mpatches
from tqdm import tqdm
from scipy.io import loadmat
from collections import Counter
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import traceback
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from collections import defaultdict
import json
import pickle
import ast
import sys
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre


import torch.nn.functional as F
from pathlib import Path
from matplotlib.patches import Rectangle


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import time
import pickle
import networkx as nx
from probeinterface import write_probeinterface, read_probeinterface, Probe
import torch.nn as nn

import utils
from utils import (
    SpikeDataset,
    Spike_Detection_MLP,
    cluster_label_array1_based_on_array2,
    create_channel_groups_using_cliques,
    detect_local_maxima_in_window,
    extract_windows,
    label_array1_based_on_array2,
    visualize_model_umap,
    _normalize_best_channels,
    _resolve_best_channel_indices,
)

warnings.filterwarnings("ignore")

# === Probe 准备 ===
probe_data = loadmat("/media/ubuntu/sda/duan/rat/probe/chanMapQPX_mice1.mat")
probe_x = probe_data["xcoords"]
probe_y = probe_data["ycoords"]

probe_position = pd.DataFrame(probe_x)
probe_position[1] = probe_y


def build_probe_instance() -> Probe:
    probe = Probe()
    probe.set_contacts(
        positions=probe_position,
        contact_ids=probe_data["chanMap"][:, 0],
    )
    probe.set_device_channel_indices(range(128))
    return probe


probe_template = build_probe_instance()

# === Clique 计算（仅一次）===
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

print(f"\n创建了{len(model_channel_dict)}个模型组")

# === 全局路径与参数 ===
recording_raw_path = Path("/home/ubuntu/Downloads/paper/20250612_1.group0.bin")
recording_raw = se.read_binary(
    str(recording_raw_path),
    sampling_frequency=30000,
    dtype=np.int16,
    num_channels=128 * 7,
)

sorting_results_dir = Path("/media/ubuntu/sda/duan/rat/sorting_results/day1")
detection_results_root = Path("/media/ubuntu/sda/duan/rat/spike_detection_results")
classification_results_root = Path("/media/ubuntu/sda/duan/rat/spike_classification_results")
figure_dir = Path("/media/ubuntu/sda/duan/rat/figure")
figure_dir.mkdir(parents=True, exist_ok=True)

hidden_size1 = 256
hidden_size2 = 64
output_size = 1
device = "cuda"
num_epochs = 50
batch_size = 1024
window_size = 71
half_window = window_size // 2
chunk_size = 120000
enable_visualization = True
min_spikes_per_cluster = 300
num_segments = 100
segment_frames = 180000

probe_ids = list(range(1, 8))

overall_results_by_probe = {}
overall_templates_by_probe = {}
overall_failed_clusters = {}

# === 遍历所有 probe ===
for probe_idx in probe_ids:
    probe_label = f"probe_{probe_idx}"
    print(f"\n{'=' * 120}")
    print(f"开始处理 {probe_label}")
    print(f"{'=' * 120}")

    main_result_dir = detection_results_root / f"{probe_label}_models"
    classification_main_dir = classification_results_root / f"{probe_label}_models"
    main_result_dir.mkdir(parents=True, exist_ok=True)
    classification_main_dir.mkdir(parents=True, exist_ok=True)

    spike_inf_path = sorting_results_dir / f"spike_inf_{probe_label}.tsv"
    cluster_inf_path = sorting_results_dir / f"cluster_inf_{probe_label}.csv"

    if not spike_inf_path.exists():
        print(f"✗ 缺少 {spike_inf_path}，跳过 {probe_label}")
        continue
    if not cluster_inf_path.exists():
        print(f"✗ 缺少 {cluster_inf_path}，跳过 {probe_label}")
        continue

    spike_inf_df = pd.read_csv(spike_inf_path, index_col=0, sep="\t")
    cluster_inf_df = pd.read_csv(cluster_inf_path, index_col=0)

    cluster_id_col = None
    for candidate in ["cluster_id", "cluster"]:
        if candidate in cluster_inf_df.columns:
            cluster_id_col = candidate
            break
    if cluster_id_col is None:
        cluster_id_col = cluster_inf_df.columns[0]

    probe_instance = build_probe_instance()

    channel_offset = 128 * (probe_idx - 1)
    channel_ids_for_probe = [channel_offset + c for c in range(128)]

    try:
        recording_probe_raw = recording_raw.select_channels(channel_ids_for_probe)
    except Exception as exc:
        print(f"✗ 选择 {probe_label} 通道时出错: {exc}")
        continue

    recording_probe = spre.bandpass_filter(recording_probe_raw, freq_min=300, freq_max=3000)
    recording_probe = spre.notch_filter(recording_probe, freq=50)
    recording_f = spre.common_reference(recording_probe, reference="global", operator="median")
    recording_f = recording_f.set_probegroup(probe_instance)

    total_frames = recording_f.get_num_samples()
    print(f"{probe_label} recording总帧数: {total_frames}")

    all_results = {}

    # === 遍历模型组 ===
    for idx, (channels_tuple, model_ids) in enumerate(model_channel_dict.items()):
        channel_group_id = str(list(channels_tuple))
        model_id = f"model_{model_ids[0]}"
        print(f"\n{'=' * 80}")
        print(f"{probe_label} - 处理第 {idx + 1}/{len(model_channel_dict)} 个模型组: {model_id}")
        print(f"通道组合: {channel_group_id}")
        print(f"{'=' * 80}")

        try:
            result_dir = main_result_dir / model_id
            result_dir.mkdir(parents=True, exist_ok=True)

            group_info = channel_groups[model_id]
            channel_indices = group_info["channel_indices"]
            device_channel_indices = group_info["device_channel_indices"]
            clique_center = group_info["center"]

            if {"position_1", "position_2"}.issubset(cluster_inf_df.columns):
                clique_channel_positions = probe_template.contact_positions[channel_indices]
                if len(clique_channel_positions) >= 3:
                    hull = ConvexHull(clique_channel_positions)
                    hull_points = clique_channel_positions[hull.vertices]
                    hull_path = MplPath(hull_points)
                    cluster_positions = cluster_inf_df[["position_1", "position_2"]].values
                    is_inside = hull_path.contains_points(cluster_positions)
                else:
                    x_min, x_max = clique_channel_positions[:, 0].min(), clique_channel_positions[:, 0].max()
                    y_min, y_max = clique_channel_positions[:, 1].min(), clique_channel_positions[:, 1].max()
                    cluster_positions = cluster_inf_df[["position_1", "position_2"]].values
                    is_inside = (
                        (cluster_positions[:, 0] >= x_min)
                        & (cluster_positions[:, 0] <= x_max)
                        & (cluster_positions[:, 1] >= y_min)
                        & (cluster_positions[:, 1] <= y_max)
                    )
                    print("警告: clique通道数少于3个，使用边界框判断")
                clusters_in_clique = cluster_inf_df.loc[is_inside, cluster_id_col].astype(int).tolist()
            else:
                clusters_in_clique = cluster_inf_df[cluster_id_col].astype(int).tolist()

            n_clusters_in_clique = len(clusters_in_clique)

            if n_clusters_in_clique == 0:
                print(f"✗ {probe_label}-{model_id}: 当前clique内Day1没有cluster，跳过该模型")
                continue

            valid_channels = []
            for ch_idx in channel_indices:
                if ch_idx < len(recording_f.channel_ids):
                    valid_channels.append(recording_f.channel_ids[ch_idx])
                else:
                    print(f"警告: 通道索引 {ch_idx} 超出范围，跳过")

            if not valid_channels:
                print(f"错误: 模型组 {model_id} 中没有可用通道，跳过")
                continue

            print(f"\n--- Clique信息 ---")
            print(f"Clique中心位置: ({clique_center[0]:.1f}, {clique_center[1]:.1f}) μm")
            print(f"Clique覆盖的通道数: {len(valid_channels)}")
            print(f"Clique覆盖的cluster数: {n_clusters_in_clique}")
            if clusters_in_clique:
                preview = clusters_in_clique[:10]
                suffix = "..." if len(clusters_in_clique) > 10 else ""
                print(f"Cluster IDs (前10个): {preview}{suffix}")
            print(f"\n--- 通道信息 ---")
            print(f"通道索引范围: {min(channel_indices)}-{max(channel_indices)}")
            preview_device = device_channel_indices[:10]
            suffix_device = "..." if len(device_channel_indices) > 10 else ""
            print(f"Device channel indices: {preview_device}{suffix_device}")

            print(f"开始处理所有chunks，总共 {total_frames} 帧...")

            all_valid_indices = []
            all_windows = []

            for start_frame in tqdm(range(0, total_frames, chunk_size), desc=f"{probe_label} chunks"):
                end_frame = min(start_frame + chunk_size, total_frames)

                try:
                    data_chunk = recording_f.get_traces(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        channel_ids=valid_channels,
                    )

                    threshold_result = detect_local_maxima_in_window(
                        data_chunk.T,
                        std_multiplier=3,
                        window_size=30,
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

                except Exception as e:
                    print(f"处理chunk时出错: {e}")
                    continue

            all_valid_indices = np.array(all_valid_indices)
            all_windows = np.stack(all_windows) if len(all_windows) > 0 else np.array([])

            print(f"\n--- Spike检测结果 ---")
            print(f"检测到的spike候选数量: {len(all_valid_indices):,}")
            print(f"提取的时间窗数量: {len(all_windows):,}")

            spike_cluster_col = None
            for col in ["cluster", "cluster_id"]:
                if col in spike_inf_df.columns:
                    spike_cluster_col = col
                    break

            if spike_cluster_col is None:
                print("警告: spike_inf中没有找到'cluster'或'cluster_id'列")
                print(f"可用列: {list(spike_inf_df.columns)}")
                spike_inf_temp = spike_inf_df.copy()
                print(f"使用所有spike数据: {len(spike_inf_temp):,}")
            elif clusters_in_clique:
                spike_inf_temp = spike_inf_df[spike_inf_df[spike_cluster_col].isin(clusters_in_clique)].copy()
                print(f"使用的spike数据数量（来自{n_clusters_in_clique}个cluster）: {len(spike_inf_temp):,}")
            else:
                spike_inf_temp = spike_inf_df.copy()
                print(f"警告: 未找到clique范围内的cluster，使用所有spike数据: {len(spike_inf_temp):,}")

            if len(spike_inf_temp) == 0:
                print("错误: 没有找到对应的spike数据，跳过此模型组")
                continue

            if "time" not in spike_inf_temp.columns:
                print("错误: spike_inf缺少'time'列，无法继续")
                continue

            labels = label_array1_based_on_array2(all_valid_indices, spike_inf_temp["time"], threshold=2)
            cluster_labels_full = cluster_label_array1_based_on_array2(all_valid_indices, spike_inf_temp, threshold=2)
            cluster_labels_full = np.asarray(cluster_labels_full, dtype=int)

            detected_spike_count = np.sum(labels == 1)
            total_detected = len(all_valid_indices)
            total_real_spikes = len(spike_inf_temp)

            detection_recall = detected_spike_count / total_real_spikes * 100 if total_real_spikes > 0 else 0
            detection_precision = detected_spike_count / total_detected * 100 if total_detected > 0 else 0

            print(f"\n--- Spike检测统计 ---")
            print(f"检测到的spike候选总数: {total_detected:,}")
            print(f"真实spike总数（来自{n_clusters_in_clique}个cluster）: {total_real_spikes:,}")
            print(f"匹配的真实spike数量: {detected_spike_count:,}")
            print(f"检测召回率 (Recall): {detection_recall:.2f}%")
            print(f"检测精确率 (Precision): {detection_precision:.2f}%")

            indices_0 = np.where(labels == 0)[0]
            indices_1 = np.where(labels == 1)[0]

            target_0_count = len(indices_1) * 3
            if len(indices_0) > target_0_count:
                sampled_indices_0 = np.random.choice(indices_0, target_0_count, replace=False)
            else:
                sampled_indices_0 = indices_0

            final_indices = np.concatenate([sampled_indices_0, indices_1])
            np.random.shuffle(final_indices)

            if len(all_windows) == 0:
                print("警告: 未能提取到有效的时间窗，跳过此模型组")
                continue

            sampled_windows = all_windows[final_indices]
            sampled_labels = labels[final_indices]
            sampled_cluster_labels = cluster_labels_full[final_indices]

            print(f"\n--- 训练数据准备 ---")
            print(f"正样本数量 (spike): {len(indices_1):,}")
            print(f"负样本数量 (non-spike): {len(indices_0):,}")
            print(f"平衡后总样本数: {len(sampled_windows):,}")
            print(f"时间窗形状: {sampled_windows.shape}")

            dataset = SpikeDataset(sampled_windows, sampled_labels)

            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            if train_size == 0 or test_size == 0:
                print("警告: 数据集过小，无法划分训练/测试集，跳过")
                continue

            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print(f"训练集大小: {train_size:,} ({train_size / len(dataset) * 100:.1f}%)")
            print(f"测试集大小: {test_size:,} ({test_size / len(dataset) * 100:.1f}%)")

            input_size = sampled_windows.shape[1] * sampled_windows.shape[2]
            model = Spike_Detection_MLP(
                input_size,
                hidden_size1,
                hidden_size2,
                output_size,
                n_channels=sampled_windows.shape[1],
                time_window=sampled_windows.shape[2],
            )
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.BCELoss()

            print(f"\n--- 开始训练模型 ---")
            print(f"模型参数: hidden_size1={hidden_size1}, hidden_size2={hidden_size2}")
            print(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, patience=5")
            training_history = []
            best_tpr = 0
            best_model_state = None
            patience = 5
            patience_counter = 0

            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_data, batch_labels in train_loader:
                    batch_labels = batch_labels.float().unsqueeze(1)
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)

                    predicted = (outputs > 0.5).float()
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                model.eval()
                test_correct = 0
                test_total = 0
                true_positive = 0
                true_negative = 0
                false_positive = 0
                false_negative = 0

                with torch.no_grad():
                    for batch_data, batch_labels in test_loader:
                        batch_labels = batch_labels.float().unsqueeze(1)
                        batch_data = batch_data.to(device)
                        batch_labels = batch_labels.to(device)

                        outputs = model(batch_data)
                        predicted = (outputs > 0.5).float()
                        test_total += batch_labels.size(0)
                        test_correct += (predicted == batch_labels).sum().item()

                        true_positive += ((predicted == 1) & (batch_labels == 1)).sum().item()
                        true_negative += ((predicted == 0) & (batch_labels == 0)).sum().item()
                        false_positive += ((predicted == 1) & (batch_labels == 0)).sum().item()
                        false_negative += ((predicted == 0) & (batch_labels == 1)).sum().item()

                tpr = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                tnr = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                accuracy = test_correct / test_total if test_total > 0 else 0

                if tpr > best_tpr:
                    best_tpr = tpr
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    if epoch % 5 == 0 or epoch < 5:
                        print(
                            f"Epoch {epoch + 1:3d}/{num_epochs}: ✓ 新最佳! "
                            f"TPR={tpr:.4f}, TNR={tnr:.4f}, Acc={accuracy:.4f}, Loss={train_loss / len(train_loader):.4f}"
                        )
                else:
                    patience_counter += 1
                    if epoch % 5 == 0 or epoch < 5:
                        print(
                            f"Epoch {epoch + 1:3d}/{num_epochs}: TPR={tpr:.4f} (最佳: {best_tpr:.4f}), "
                            f"TNR={tnr:.4f}, Acc={accuracy:.4f}, 早停: {patience_counter}/{patience}"
                        )

                training_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss / len(train_loader),
                        "train_accuracy": train_correct / train_total if train_total > 0 else 0,
                        "test_accuracy": accuracy,
                        "tpr": tpr,
                        "tnr": tnr,
                        "patience_counter": patience_counter,
                    }
                )

                if patience_counter >= patience:
                    print(f"\n--- 早停触发 ---")
                    print(f"连续 {patience} 个epoch没有提升，在第 {epoch + 1} 个epoch停止训练")
                    print(f"最佳TPR: {best_tpr:.4f}")
                    break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                torch.save(model.state_dict(), result_dir / "best_model.pth")

            actual_epochs = len(training_history)
            early_stopped = actual_epochs < num_epochs

            result_summary = {
                "probe": probe_label,
                "model_id": model_id,
                "channel_group_id": channel_group_id,
                "channels": valid_channels,
                "n_channels": len(valid_channels),
                "detection_stats": {
                    "total_detected": int(total_detected),
                    "total_real_spikes": int(total_real_spikes),
                    "detected_real_spikes": int(detected_spike_count),
                    "detection_recall": float(detection_recall),
                    "detection_precision": float(detection_precision),
                },
                "training_stats": {
                    "best_tpr": float(best_tpr),
                    "final_tnr": float(tnr),
                    "final_accuracy": float(accuracy),
                    "total_epochs": num_epochs,
                    "actual_epochs": actual_epochs,
                    "early_stopped": early_stopped,
                    "patience": patience,
                },
                "data_stats": {
                    "window_shape": sampled_windows.shape,
                    "train_samples": len(train_dataset),
                    "test_samples": len(test_dataset),
                },
            }

            with open(result_dir / "result_summary.pkl", "wb") as f:
                pickle.dump(result_summary, f)

            with open(result_dir / "training_history.pkl", "wb") as f:
                pickle.dump(training_history, f)

            all_results[model_id] = result_summary

            test_windows = []
            test_labels = []
            test_clusters = []
            subset_indices = getattr(test_dataset, "indices", [])
            for idx_sample in subset_indices:
                test_windows.append(sampled_windows[idx_sample])
                test_labels.append(sampled_labels[idx_sample])
                test_clusters.append(sampled_cluster_labels[idx_sample])

            if test_windows:
                test_windows = np.stack(test_windows)
                test_labels = np.array(test_labels)
                test_clusters = np.array(test_clusters, dtype=int)

                with open(result_dir / "test_dataset.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "windows": test_windows,
                            "labels": test_labels,
                            "cluster_labels": test_clusters,
                        },
                        f,
                    )
            else:
                test_windows = None
                test_labels = None
                test_clusters = None

            if enable_visualization and test_windows is not None:
                try:
                    print(f"\n--- 开始 {probe_label}-{model_id} UMAP可视化 ---")
                    output_pdf_path = figure_dir / f"{probe_label}_{model_id}_umap_visualization.pdf"
                    test_dataset_for_viz = SpikeDataset(test_windows, test_labels)
                    visualize_model_umap(
                        f"{probe_label}_{model_id}",
                        str(result_dir),
                        test_dataset_for_viz,
                        model,
                        device=device,
                        n_samples=100000,
                        output_pdf_path=str(output_pdf_path),
                        cluster_labels=test_clusters,
                    )
                    print(f"✓ {probe_label}-{model_id} UMAP可视化完成")
                except Exception as e:
                    print(f"✗ {probe_label}-{model_id} UMAP可视化失败: {e}")
                    import traceback

                    traceback.print_exc()

            print("跳过独立的spike classification训练，改为Day2阶段对检测特征执行KMeans。")
            print(f"\n--- {probe_label}-{model_id} 处理完成 ---")
            print(f"最佳TPR: {best_tpr:.4f}")
            print(f"最终TNR: {tnr:.4f}")
            print(f"最终Accuracy: {accuracy:.4f}")
            print(f"训练epoch数: {actual_epochs}/{num_epochs} {'(早停)' if early_stopped else '(完成)'}")
            print(f"结果已保存至: {result_dir}")

        except Exception as e:
            print(f"\n✗ 处理 {probe_label} 模型组 {model_id} 时出错: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{probe_label} 模型处理完成: {len(all_results)}/{len(model_channel_dict)} 个模型组")
    with open(main_result_dir / "all_results_summary.pkl", "wb") as f:
        pickle.dump(all_results, f)
    overall_results_by_probe[probe_label] = all_results

    # === Day1 模板生成 ===
    if "Neuron" not in cluster_inf_df.columns and cluster_id_col in cluster_inf_df.columns:
        cluster_inf_df["Neuron"] = cluster_inf_df[cluster_id_col]

    cluster_best_channels_map = {}
    if "best_channels" in cluster_inf_df.columns and cluster_id_col in cluster_inf_df.columns:
        for row in cluster_inf_df[[cluster_id_col, "best_channels"]].itertuples(index=False):
            parsed = _normalize_best_channels(row.best_channels)
            if parsed:
                cluster_best_channels_map[int(getattr(row, cluster_id_col))] = parsed

    utils.cluster_best_channels_map = cluster_best_channels_map

    if "time" not in spike_inf_df.columns:
        print(f"{probe_label}: spike_inf缺少'time'列，跳过模板生成")
        overall_templates_by_probe[probe_label] = {}
        overall_failed_clusters[probe_label] = []
        continue

    total_frames_day1 = recording_f.get_num_samples()

    def _filter_spike_times(times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=int)
        mask = (times >= half_window) & (times < total_frames_day1 - half_window)
        return np.sort(times[mask])

    day1_model_templates = {}
    failed_clusters = []
    updated_clusters = 0
    model_total = len(channel_groups)

    for model_idx, (model_id, group_info) in enumerate(channel_groups.items(), start=1):
        print(f"\n{'=' * 80}")
        print(f"{probe_label} - 开始处理模型 {model_id} ({model_idx}/{model_total})")
        print(f"{'=' * 80}")

        channel_indices = np.asarray(group_info["channel_indices"], dtype=int)
        clique_channel_positions = probe_template.contact_positions[channel_indices]
        channel_pos_array = np.asarray(clique_channel_positions, dtype=float)

        if {"position_1", "position_2"}.issubset(cluster_inf_df.columns):
            cluster_positions = cluster_inf_df[["position_1", "position_2"]].to_numpy()
            if len(channel_pos_array) >= 3:
                hull = ConvexHull(channel_pos_array)
                hull_points = channel_pos_array[hull.vertices]
                hull_path = MplPath(hull_points)
                inside_mask = hull_path.contains_points(cluster_positions)
            else:
                x_min, x_max = channel_pos_array[:, 0].min(), channel_pos_array[:, 0].max()
                y_min, y_max = channel_pos_array[:, 1].min(), channel_pos_array[:, 1].max()
                inside_mask = (
                    (cluster_positions[:, 0] >= x_min)
                    & (cluster_positions[:, 0] <= x_max)
                    & (cluster_positions[:, 1] >= y_min)
                    & (cluster_positions[:, 1] <= y_max)
                )
            clusters_in_clique = cluster_inf_df.loc[inside_mask, cluster_id_col].astype(int).tolist()
        else:
            clusters_in_clique = cluster_inf_df[cluster_id_col].astype(int).tolist()

        if not clusters_in_clique:
            print("  [WARN] 未找到位于该clique范围内的Day1 clusters，跳过")
            continue

        device_indices_in_group = []
        recording_channel_ids = []
        device_channel_positions = []
        for idx_in_group, device_idx in enumerate(channel_indices):
            device_idx = int(device_idx)
            if device_idx < len(recording_f.channel_ids):
                device_indices_in_group.append(device_idx)
                recording_channel_ids.append(int(recording_f.channel_ids[device_idx]))
                device_channel_positions.append(channel_pos_array[idx_in_group])
        if not recording_channel_ids:
            print("  [WARN] 无有效通道，跳过")
            continue

        device_channel_positions_array = np.asarray(device_channel_positions, dtype=float)
        device_index_lookup = {ch: idx for idx, ch in enumerate(device_indices_in_group)}

        cluster_channel_selection = {}
        for cluster_id in clusters_in_clique:
            indices, resolved_channels, has_best = _resolve_best_channel_indices(
                int(cluster_id), device_indices_in_group, device_index_lookup
            )
            if has_best and not indices:
                print(f"  [WARN] Cluster {cluster_id} 的best_channels不在当前clique中，跳过")
                failed_clusters.append(int(cluster_id))
                continue
            if not indices:
                continue
            cluster_channel_selection[int(cluster_id)] = {
                "indices": indices,
                "channel_ids": [recording_channel_ids[i] for i in indices],
                "channel_indices": [device_indices_in_group[i] for i in indices],
                "positions": device_channel_positions_array[indices],
            }

        if not cluster_channel_selection:
            print("  [WARN] 该clique内无满足条件的cluster，跳过")
            continue

        model_templates = {}
        day1_model_templates[model_id] = model_templates

        cluster_spike_times = {}
        for cluster_id in sorted(cluster_channel_selection.keys()):
            times = (
                spike_inf_df.loc[spike_inf_df[spike_cluster_col] == cluster_id, "time"]
                .to_numpy(dtype=int)
            )
            times = _filter_spike_times(times)
            if times.size == 0:
                failed_clusters.append(int(cluster_id))
                continue
            cluster_spike_times[int(cluster_id)] = times

        if not cluster_spike_times:
            print("  [WARN] 该clique内无可用spike，跳过")
            continue

        cluster_windows_map = defaultdict(list)
        max_start = max(total_frames_day1 - segment_frames - 1, 0)
        if num_segments > 1 and max_start > 0:
            segment_starts = np.linspace(0, max_start, num=num_segments)
        else:
            segment_starts = np.array([0], dtype=float)
        segment_starts = np.unique(segment_starts.astype(int))

        print(f"  计划读取 {len(segment_starts)} 段数据，每段 {segment_frames} 帧")

        for start_frame in tqdm(segment_starts, desc=f"{probe_label}-{model_id} segments", leave=False):
            end_frame = min(start_frame + segment_frames, total_frames_day1)
            if end_frame - start_frame <= window_size:
                continue

            try:
                data_chunk = recording_f.get_traces(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    channel_ids=recording_channel_ids,
                )
            except Exception:
                continue

            data_chunk = data_chunk.T

            for cluster_id, times in cluster_spike_times.items():
                selection = cluster_channel_selection.get(cluster_id)
                if selection is None:
                    continue
                indices = selection["indices"]
                if not indices:
                    continue

                idx_start = np.searchsorted(times, start_frame, side="left")
                idx_end = np.searchsorted(times, end_frame, side="right")
                if idx_end <= idx_start:
                    continue
                selected_times = times[idx_start:idx_end]
                mask_valid = (
                    (selected_times >= start_frame + half_window)
                    & (selected_times < end_frame - half_window)
                )
                if not np.any(mask_valid):
                    continue
                selected_times = selected_times[mask_valid]
                rel_indices = (selected_times - start_frame).astype(int)
                if rel_indices.size == 0:
                    continue
                try:
                    data_chunk_cluster = data_chunk[indices, :]
                    chunk_windows = extract_windows(
                        data_chunk_cluster,
                        rel_indices,
                        window_size=window_size,
                    )
                except ValueError:
                    continue
                cluster_windows_map[cluster_id].append(chunk_windows)

        for cluster_id, selection in cluster_channel_selection.items():
            windows_list = cluster_windows_map.get(cluster_id)
            if not windows_list:
                failed_clusters.append(int(cluster_id))
                continue

            windows = np.concatenate(windows_list, axis=0)
            if windows.shape[0] < min_spikes_per_cluster:
                failed_clusters.append(int(cluster_id))
                continue

            rms_per_channel = np.sqrt(np.mean(windows**2, axis=(0, 2)))
            if np.allclose(rms_per_channel.sum(), 0):
                failed_clusters.append(int(cluster_id))
                continue

            weights = rms_per_channel / (rms_per_channel.sum() + 1e-12)
            mean_waveform_channels = np.mean(windows, axis=0)
            synth_waveform = np.sum(mean_waveform_channels * weights[:, None], axis=0)
            synth_waveform = synth_waveform - np.mean(synth_waveform)
            std_waveform = np.std(synth_waveform)
            if std_waveform > 0:
                synth_waveform = synth_waveform / std_waveform

            best_channel_positions = selection["positions"]
            cluster_pos = np.sum(best_channel_positions * weights[:, None], axis=0)

            model_templates[int(cluster_id)] = {
                "position": cluster_pos.tolist(),
                "waveform": synth_waveform.tolist(),
                "channel_waveforms": mean_waveform_channels.tolist(),
                "n_spikes": int(windows.shape[0]),
                "channel_ids": [int(ch) for ch in selection["channel_ids"]],
                "channel_indices": [int(idx) for idx in selection["channel_indices"]],
                "position_top_channels": len(selection["channel_ids"]),
            }
            updated_clusters += 1

        print(f"  [INFO] {probe_label} - 模型 {model_id} 已生成 {len(model_templates)} 个模板")

    print(f"\n{probe_label} 共生成 {updated_clusters} 个Day1 cluster模板。")
    if failed_clusters:
        preview_failed = sorted(set(failed_clusters))[:20]
        suffix_failed = "..." if len(set(failed_clusters)) > 20 else ""
        print(f"以下cluster因数据不足或窗口提取失败未生成模板: {preview_failed}{suffix_failed}")

    templates_output_path = sorting_results_dir / f"{probe_label}_day1_model_templates.pkl"
    with open(templates_output_path, "wb") as f:
        pickle.dump(day1_model_templates, f)
    print(f"{probe_label} Day1模板已保存到: {templates_output_path}")

    overall_templates_by_probe[probe_label] = day1_model_templates
    overall_failed_clusters[probe_label] = sorted(set(failed_clusters))


# === 汇总输出 ===
if overall_results_by_probe:
    summary_path = detection_results_root / "all_probes_results_summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(overall_results_by_probe, f)
    print(f"\n所有probe检测结果汇总保存至: {summary_path}")

if overall_templates_by_probe:
    templates_summary_path = sorting_results_dir / "all_probes_day1_model_templates.pkl"
    with open(templates_summary_path, "wb") as f:
        pickle.dump(overall_templates_by_probe, f)
    print(f"所有probe模板汇总保存至: {templates_summary_path}")

print("\n全部处理完成！")
import ast
import json
import os
import pickle
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from probeinterface import Probe, read_probeinterface, write_probeinterface
from scipy.io import loadmat
from scipy.spatial import ConvexHull
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from utils import (
    SpikeDataset,
    Spike_Detection_MLP,
    cluster_label_array1_based_on_array2,
    create_channel_groups_using_cliques,
    detect_local_maxima_in_window,
    extract_windows,
    label_array1_based_on_array2,
    visualize_model_umap,
    _normalize_best_channels,
    _resolve_best_channel_indices,
)

warnings.filterwarnings("ignore")

# === Probe 准备 ===
probe_data = loadmat("/media/ubuntu/sda/duan/rat/probe/chanMapQPX_mice1.mat")
probe_x = probe_data["xcoords"]
probe_y = probe_data["ycoords"]

probe_position = pd.DataFrame(probe_x)
probe_position[1] = probe_y


def build_probe_instance() -> Probe:
    probe = Probe()
    probe.set_contacts(
        positions=probe_position,
        contact_ids=probe_data["chanMap"][:, 0],
    )
    probe.set_device_channel_indices(range(128))
    return probe


probe_template = build_probe_instance()

# === Clique 计算（仅一次）===
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

print(f"\n创建了{len(model_channel_dict)}个模型组")

# === 全局路径与参数 ===
recording_raw_path = Path("/home/ubuntu/Downloads/paper/20250612_1.group0.bin")
recording_raw = se.read_binary(
    str(recording_raw_path),
    sampling_frequency=30000,
    dtype=np.int16,
    num_channels=128 * 7,
)

sorting_results_dir = Path("/media/ubuntu/sda/duan/rat/sorting_results/day1")
detection_results_root = Path("/media/ubuntu/sda/duan/rat/spike_detection_results")
classification_results_root = Path("/media/ubuntu/sda/duan/rat/spike_classification_results")
figure_dir = Path("/media/ubuntu/sda/duan/rat/figure")
figure_dir.mkdir(parents=True, exist_ok=True)

hidden_size1 = 256
hidden_size2 = 64
output_size = 1
device = "cuda"
num_epochs = 50
batch_size = 1024
window_size = 71
half_window = window_size // 2
chunk_size = 120000
enable_visualization = True
min_spikes_per_cluster = 300
num_segments = 100
segment_frames = 180000

probe_ids = list(range(1, 8))

overall_results_by_probe = {}
overall_templates_by_probe = {}
overall_failed_clusters = {}

# === 遍历所有 probe ===
for probe_idx in probe_ids:
    probe_label = f"probe_{probe_idx}"
    print(f"\n{'=' * 120}")
    print(f"开始处理 {probe_label}")
    print(f"{'=' * 120}")

    main_result_dir = detection_results_root / f"{probe_label}_models"
    classification_main_dir = classification_results_root / f"{probe_label}_models"
    main_result_dir.mkdir(parents=True, exist_ok=True)
    classification_main_dir.mkdir(parents=True, exist_ok=True)

    spike_inf_path = sorting_results_dir / f"spike_inf_{probe_label}.tsv"
    cluster_inf_path = sorting_results_dir / f"cluster_inf_{probe_label}.csv"

    if not spike_inf_path.exists():
        print(f"✗ 缺少 {spike_inf_path}，跳过 {probe_label}")
        continue
    if not cluster_inf_path.exists():
        print(f"✗ 缺少 {cluster_inf_path}，跳过 {probe_label}")
        continue

    spike_inf_df = pd.read_csv(spike_inf_path, index_col=0, sep="\t")
    cluster_inf_df = pd.read_csv(cluster_inf_path, index_col=0)

    cluster_id_col = None
    for candidate in ["cluster_id", "cluster"]:
        if candidate in cluster_inf_df.columns:
            cluster_id_col = candidate
            break
    if cluster_id_col is None:
        cluster_id_col = cluster_inf_df.columns[0]

    probe_instance = build_probe_instance()

    channel_offset = 128 * (probe_idx - 1)
    channel_ids_for_probe = [channel_offset + c for c in range(128)]

    try:
        recording_probe_raw = recording_raw.select_channels(channel_ids_for_probe)
    except Exception as exc:
        print(f"✗ 选择 {probe_label} 通道时出错: {exc}")
        continue

    recording_probe = spre.bandpass_filter(recording_probe_raw, freq_min=300, freq_max=3000)
    recording_probe = spre.notch_filter(recording_probe, freq=50)
    recording_f = spre.common_reference(recording_probe, reference="global", operator="median")
    recording_f = recording_f.set_probegroup(probe_instance)

    total_frames = recording_f.get_num_samples()
    print(f"{probe_label} recording总帧数: {total_frames}")

    all_results = {}

    # === 遍历模型组 ===
    for idx, (channels_tuple, model_ids) in enumerate(model_channel_dict.items()):
        channel_group_id = str(list(channels_tuple))
        model_id = f"model_{model_ids[0]}"
        print(f"\n{'=' * 80}")
        print(f"{probe_label} - 处理第 {idx + 1}/{len(model_channel_dict)} 个模型组: {model_id}")
        print(f"通道组合: {channel_group_id}")
        print(f"{'=' * 80}")

        try:
            result_dir = main_result_dir / model_id
            result_dir.mkdir(parents=True, exist_ok=True)

            group_info = channel_groups[model_id]
            channel_indices = group_info["channel_indices"]
            device_channel_indices = group_info["device_channel_indices"]
            clique_center = group_info["center"]

            if {"position_1", "position_2"}.issubset(cluster_inf_df.columns):
                clique_channel_positions = probe_template.contact_positions[channel_indices]
                if len(clique_channel_positions) >= 3:
                    hull = ConvexHull(clique_channel_positions)
                    hull_points = clique_channel_positions[hull.vertices]
                    hull_path = MplPath(hull_points)
                    cluster_positions = cluster_inf_df[["position_1", "position_2"]].values
                    is_inside = hull_path.contains_points(cluster_positions)
                else:
                    x_min, x_max = clique_channel_positions[:, 0].min(), clique_channel_positions[:, 0].max()
                    y_min, y_max = clique_channel_positions[:, 1].min(), clique_channel_positions[:, 1].max()
                    cluster_positions = cluster_inf_df[["position_1", "position_2"]].values
                    is_inside = (
                        (cluster_positions[:, 0] >= x_min)
                        & (cluster_positions[:, 0] <= x_max)
                        & (cluster_positions[:, 1] >= y_min)
                        & (cluster_positions[:, 1] <= y_max)
                    )
                    print("警告: clique通道数少于3个，使用边界框判断")
                clusters_in_clique = cluster_inf_df.loc[is_inside, cluster_id_col].astype(int).tolist()
            else:
                clusters_in_clique = cluster_inf_df[cluster_id_col].astype(int).tolist()

            n_clusters_in_clique = len(clusters_in_clique)

            valid_channels = []
            for ch_idx in channel_indices:
                if ch_idx < len(recording_f.channel_ids):
                    valid_channels.append(recording_f.channel_ids[ch_idx])
                else:
                    print(f"警告: 通道索引 {ch_idx} 超出范围，跳过")

            if not valid_channels:
                print(f"错误: 模型组 {model_id} 中没有可用通道，跳过")
                continue

            print(f"\n--- Clique信息 ---")
            print(f"Clique中心位置: ({clique_center[0]:.1f}, {clique_center[1]:.1f}) μm")
            print(f"Clique覆盖的通道数: {len(valid_channels)}")
            print(f"Clique覆盖的cluster数: {n_clusters_in_clique}")
            if clusters_in_clique:
                preview = clusters_in_clique[:10]
                suffix = "..." if len(clusters_in_clique) > 10 else ""
                print(f"Cluster IDs (前10个): {preview}{suffix}")
            print(f"\n--- 通道信息 ---")
            print(f"通道索引范围: {min(channel_indices)}-{max(channel_indices)}")
            preview_device = device_channel_indices[:10]
            suffix_device = "..." if len(device_channel_indices) > 10 else ""
            print(f"Device channel indices: {preview_device}{suffix_device}")

            print(f"开始处理所有chunks，总共 {total_frames} 帧...")

            all_valid_indices = []
            all_windows = []

            for start_frame in tqdm(range(0, total_frames, chunk_size), desc=f"{probe_label} chunks"):
                end_frame = min(start_frame + chunk_size, total_frames)

                try:
                    data_chunk = recording_f.get_traces(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        channel_ids=valid_channels,
                    )

                    threshold_result = detect_local_maxima_in_window(
                        data_chunk.T,
                        std_multiplier=3,
                        window_size=30,
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

                except Exception as e:
                    print(f"处理chunk时出错: {e}")
                    continue

            all_valid_indices = np.array(all_valid_indices)
            all_windows = np.stack(all_windows) if len(all_windows) > 0 else np.array([])

            print(f"\n--- Spike检测结果 ---")
            print(f"检测到的spike候选数量: {len(all_valid_indices):,}")
            print(f"提取的时间窗数量: {len(all_windows):,}")

            spike_cluster_col = None
            for col in ["cluster", "cluster_id"]:
                if col in spike_inf_df.columns:
                    spike_cluster_col = col
                    break

            if spike_cluster_col is None:
                print("警告: spike_inf中没有找到'cluster'或'cluster_id'列")
                print(f"可用列: {list(spike_inf_df.columns)}")
                spike_inf_temp = spike_inf_df.copy()
                print(f"使用所有spike数据: {len(spike_inf_temp):,}")
            elif clusters_in_clique:
                spike_inf_temp = spike_inf_df[spike_inf_df[spike_cluster_col].isin(clusters_in_clique)].copy()
                print(f"使用的spike数据数量（来自{n_clusters_in_clique}个cluster）: {len(spike_inf_temp):,}")
            else:
                spike_inf_temp = spike_inf_df.copy()
                print(f"警告: 未找到clique范围内的cluster，使用所有spike数据: {len(spike_inf_temp):,}")

            if len(spike_inf_temp) == 0:
                print("错误: 没有找到对应的spike数据，跳过此模型组")
                continue

            if "time" not in spike_inf_temp.columns:
                print("错误: spike_inf缺少'time'列，无法继续")
                continue

            labels = label_array1_based_on_array2(all_valid_indices, spike_inf_temp["time"], threshold=2)
            cluster_labels_full = cluster_label_array1_based_on_array2(all_valid_indices, spike_inf_temp, threshold=2)
            cluster_labels_full = np.asarray(cluster_labels_full, dtype=int)

            detected_spike_count = np.sum(labels == 1)
            total_detected = len(all_valid_indices)
            total_real_spikes = len(spike_inf_temp)

            detection_recall = detected_spike_count / total_real_spikes * 100 if total_real_spikes > 0 else 0
            detection_precision = detected_spike_count / total_detected * 100 if total_detected > 0 else 0

            print(f"\n--- Spike检测统计 ---")
            print(f"检测到的spike候选总数: {total_detected:,}")
            print(f"真实spike总数（来自{n_clusters_in_clique}个cluster）: {total_real_spikes:,}")
            print(f"匹配的真实spike数量: {detected_spike_count:,}")
            print(f"检测召回率 (Recall): {detection_recall:.2f}%")
            print(f"检测精确率 (Precision): {detection_precision:.2f}%")

            indices_0 = np.where(labels == 0)[0]
            indices_1 = np.where(labels == 1)[0]

            target_0_count = len(indices_1) * 3
            if len(indices_0) > target_0_count:
                sampled_indices_0 = np.random.choice(indices_0, target_0_count, replace=False)
            else:
                sampled_indices_0 = indices_0

            final_indices = np.concatenate([sampled_indices_0, indices_1])
            np.random.shuffle(final_indices)

            if len(all_windows) == 0:
                print("警告: 未能提取到有效的时间窗，跳过此模型组")
                continue

            sampled_windows = all_windows[final_indices]
            sampled_labels = labels[final_indices]
            sampled_cluster_labels = cluster_labels_full[final_indices]

            print(f"\n--- 训练数据准备 ---")
            print(f"正样本数量 (spike): {len(indices_1):,}")
            print(f"负样本数量 (non-spike): {len(indices_0):,}")
            print(f"平衡后总样本数: {len(sampled_windows):,}")
            print(f"时间窗形状: {sampled_windows.shape}")

            dataset = SpikeDataset(sampled_windows, sampled_labels)

            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            if train_size == 0 or test_size == 0:
                print("警告: 数据集过小，无法划分训练/测试集，跳过")
                continue

            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print(f"训练集大小: {train_size:,} ({train_size / len(dataset) * 100:.1f}%)")
            print(f"测试集大小: {test_size:,} ({test_size / len(dataset) * 100:.1f}%)")

            input_size = sampled_windows.shape[1] * sampled_windows.shape[2]
            model = Spike_Detection_MLP(
                input_size,
                hidden_size1,
                hidden_size2,
                output_size,
                n_channels=sampled_windows.shape[1],
                time_window=sampled_windows.shape[2],
            )
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.BCELoss()

            print(f"\n--- 开始训练模型 ---")
            print(f"模型参数: hidden_size1={hidden_size1}, hidden_size2={hidden_size2}")
            print(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, patience=5")
            training_history = []
            best_tpr = 0
            best_model_state = None
            patience = 5
            patience_counter = 0

            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_data, batch_labels in train_loader:
                    batch_labels = batch_labels.float().unsqueeze(1)
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)

                    predicted = (outputs > 0.5).float()
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                model.eval()
                test_correct = 0
                test_total = 0
                true_positive = 0
                true_negative = 0
                false_positive = 0
                false_negative = 0

                with torch.no_grad():
                    for batch_data, batch_labels in test_loader:
                        batch_labels = batch_labels.float().unsqueeze(1)
                        batch_data = batch_data.to(device)
                        batch_labels = batch_labels.to(device)

                        outputs = model(batch_data)
                        predicted = (outputs > 0.5).float()
                        test_total += batch_labels.size(0)
                        test_correct += (predicted == batch_labels).sum().item()

                        true_positive += ((predicted == 1) & (batch_labels == 1)).sum().item()
                        true_negative += ((predicted == 0) & (batch_labels == 0)).sum().item()
                        false_positive += ((predicted == 1) & (batch_labels == 0)).sum().item()
                        false_negative += ((predicted == 0) & (batch_labels == 1)).sum().item()

                tpr = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                tnr = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                accuracy = test_correct / test_total if test_total > 0 else 0

                if tpr > best_tpr:
                    best_tpr = tpr
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    if epoch % 5 == 0 or epoch < 5:
                        print(
                            f"Epoch {epoch + 1:3d}/{num_epochs}: ✓ 新最佳! "
                            f"TPR={tpr:.4f}, TNR={tnr:.4f}, Acc={accuracy:.4f}, Loss={train_loss / len(train_loader):.4f}"
                        )
                else:
                    patience_counter += 1
                    if epoch % 5 == 0 or epoch < 5:
                        print(
                            f"Epoch {epoch + 1:3d}/{num_epochs}: TPR={tpr:.4f} (最佳: {best_tpr:.4f}), "
                            f"TNR={tnr:.4f}, Acc={accuracy:.4f}, 早停: {patience_counter}/{patience}"
                        )

                training_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss / len(train_loader),
                        "train_accuracy": train_correct / train_total if train_total > 0 else 0,
                        "test_accuracy": accuracy,
                        "tpr": tpr,
                        "tnr": tnr,
                        "patience_counter": patience_counter,
                    }
                )

                if patience_counter >= patience:
                    print(f"\n--- 早停触发 ---")
                    print(f"连续 {patience} 个epoch没有提升，在第 {epoch + 1} 个epoch停止训练")
                    print(f"最佳TPR: {best_tpr:.4f}")
                    break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                torch.save(model.state_dict(), result_dir / "best_model.pth")

            actual_epochs = len(training_history)
            early_stopped = actual_epochs < num_epochs

            result_summary = {
                "probe": probe_label,
                "model_id": model_id,
                "channel_group_id": channel_group_id,
                "channels": valid_channels,
                "n_channels": len(valid_channels),
                "detection_stats": {
                    "total_detected": int(total_detected),
                    "total_real_spikes": int(total_real_spikes),
                    "detected_real_spikes": int(detected_spike_count),
                    "detection_recall": float(detection_recall),
                    "detection_precision": float(detection_precision),
                },
                "training_stats": {
                    "best_tpr": float(best_tpr),
                    "final_tnr": float(tnr),
                    "final_accuracy": float(accuracy),
                    "total_epochs": num_epochs,
                    "actual_epochs": actual_epochs,
                    "early_stopped": early_stopped,
                    "patience": patience,
                },
                "data_stats": {
                    "window_shape": sampled_windows.shape,
                    "train_samples": len(train_dataset),
                    "test_samples": len(test_dataset),
                },
            }

            with open(result_dir / "result_summary.pkl", "wb") as f:
                pickle.dump(result_summary, f)

            with open(result_dir / "training_history.pkl", "wb") as f:
                pickle.dump(training_history, f)

            all_results[model_id] = result_summary

            test_windows = []
            test_labels = []
            test_clusters = []
            subset_indices = getattr(test_dataset, "indices", [])
            for idx_sample in subset_indices:
                test_windows.append(sampled_windows[idx_sample])
                test_labels.append(sampled_labels[idx_sample])
                test_clusters.append(sampled_cluster_labels[idx_sample])

            if test_windows:
                test_windows = np.stack(test_windows)
                test_labels = np.array(test_labels)
                test_clusters = np.array(test_clusters, dtype=int)

                with open(result_dir / "test_dataset.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "windows": test_windows,
                            "labels": test_labels,
                            "cluster_labels": test_clusters,
                        },
                        f,
                    )
            else:
                test_windows = None
                test_labels = None
                test_clusters = None

            if enable_visualization and test_windows is not None:
                try:
                    print(f"\n--- 开始 {probe_label}-{model_id} UMAP可视化 ---")
                    output_pdf_path = figure_dir / f"{probe_label}_{model_id}_umap_visualization.pdf"
                    test_dataset_for_viz = SpikeDataset(test_windows, test_labels)
                    visualize_model_umap(
                        f"{probe_label}_{model_id}",
                        str(result_dir),
                        test_dataset_for_viz,
                        model,
                        device=device,
                        n_samples=100000,
                        output_pdf_path=str(output_pdf_path),
                        cluster_labels=test_clusters,
                    )
                    print(f"✓ {probe_label}-{model_id} UMAP可视化完成")
                except Exception as e:
                    print(f"✗ {probe_label}-{model_id} UMAP可视化失败: {e}")
                    import traceback

                    traceback.print_exc()

            print("跳过独立的spike classification训练，改为Day2阶段对检测特征执行KMeans。")
            print(f"\n--- {probe_label}-{model_id} 处理完成 ---")
            print(f"最佳TPR: {best_tpr:.4f}")
            print(f"最终TNR: {tnr:.4f}")
            print(f"最终Accuracy: {accuracy:.4f}")
            print(f"训练epoch数: {actual_epochs}/{num_epochs} {'(早停)' if early_stopped else '(完成)'}")
            print(f"结果已保存至: {result_dir}")

        except Exception as e:
            print(f"\n✗ 处理 {probe_label} 模型组 {model_id} 时出错: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{probe_label} 模型处理完成: {len(all_results)}/{len(model_channel_dict)} 个模型组")
    with open(main_result_dir / "all_results_summary.pkl", "wb") as f:
        pickle.dump(all_results, f)
    overall_results_by_probe[probe_label] = all_results

    # === Day1 模板生成 ===
    if "Neuron" not in cluster_inf_df.columns and cluster_id_col in cluster_inf_df.columns:
        cluster_inf_df["Neuron"] = cluster_inf_df[cluster_id_col]

    cluster_best_channels_map = {}
    if "best_channels" in cluster_inf_df.columns and cluster_id_col in cluster_inf_df.columns:
        for row in cluster_inf_df[[cluster_id_col, "best_channels"]].itertuples(index=False):
            parsed = _normalize_best_channels(row.best_channels)
            if parsed:
                cluster_best_channels_map[int(getattr(row, cluster_id_col))] = parsed

    total_frames_day1 = recording_f.get_num_samples()

    def _filter_spike_times(times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=int)
        mask = (times >= half_window) & (times < total_frames_day1 - half_window)
        return np.sort(times[mask])

    day1_model_templates = {}
    failed_clusters = []
    updated_clusters = 0
    model_total = len(channel_groups)

    for model_idx, (model_id, group_info) in enumerate(channel_groups.items(), start=1):
        print(f"\n{'=' * 80}")
        print(f"{probe_label} - 开始处理模型 {model_id} ({model_idx}/{model_total})")
        print(f"{'=' * 80}")

        channel_indices = np.asarray(group_info["channel_indices"], dtype=int)
        clique_channel_positions = probe_template.contact_positions[channel_indices]
        channel_pos_array = np.asarray(clique_channel_positions, dtype=float)

        if {"position_1", "position_2"}.issubset(cluster_inf_df.columns):
            cluster_positions = cluster_inf_df[["position_1", "position_2"]].to_numpy()
            if len(channel_pos_array) >= 3:
                hull = ConvexHull(channel_pos_array)
                hull_points = channel_pos_array[hull.vertices]
                hull_path = MplPath(hull_points)
                inside_mask = hull_path.contains_points(cluster_positions)
            else:
                x_min, x_max = channel_pos_array[:, 0].min(), channel_pos_array[:, 0].max()
                y_min, y_max = channel_pos_array[:, 1].min(), channel_pos_array[:, 1].max()
                inside_mask = (
                    (cluster_positions[:, 0] >= x_min)
                    & (cluster_positions[:, 0] <= x_max)
                    & (cluster_positions[:, 1] >= y_min)
                    & (cluster_positions[:, 1] <= y_max)
                )
            clusters_in_clique = cluster_inf_df.loc[inside_mask, cluster_id_col].astype(int).tolist()
        else:
            clusters_in_clique = cluster_inf_df[cluster_id_col].astype(int).tolist()

        if not clusters_in_clique:
            print("  [WARN] 未找到位于该clique范围内的Day1 clusters，跳过")
            continue

        device_indices_in_group = []
        recording_channel_ids = []
        device_channel_positions = []
        for idx_in_group, device_idx in enumerate(channel_indices):
            device_idx = int(device_idx)
            if device_idx < len(recording_f.channel_ids):
                device_indices_in_group.append(device_idx)
                recording_channel_ids.append(int(recording_f.channel_ids[device_idx]))
                device_channel_positions.append(channel_pos_array[idx_in_group])
        if not recording_channel_ids:
            print("  [WARN] 无有效通道，跳过")
            continue

        device_channel_positions_array = np.asarray(device_channel_positions, dtype=float)
        device_index_lookup = {ch: idx for idx, ch in enumerate(device_indices_in_group)}

        cluster_channel_selection = {}
        for cluster_id in clusters_in_clique:
            indices, resolved_channels, has_best = _resolve_best_channel_indices(
                int(cluster_id), device_indices_in_group, device_index_lookup
            )
            if has_best and not indices:
                print(f"  [WARN] Cluster {cluster_id} 的best_channels不在当前clique中，跳过")
                failed_clusters.append(int(cluster_id))
                continue
            if not indices:
                continue
            cluster_channel_selection[int(cluster_id)] = {
                "indices": indices,
                "channel_ids": [recording_channel_ids[i] for i in indices],
                "channel_indices": [device_indices_in_group[i] for i in indices],
                "positions": device_channel_positions_array[indices],
            }

        if not cluster_channel_selection:
            print("  [WARN] 该clique内无满足条件的cluster，跳过")
            continue

        model_templates = {}
        day1_model_templates[model_id] = model_templates

        if "time" not in spike_inf_df.columns:
            print("  [WARN] spike_inf缺少'time'列，无法生成模板")
            continue

        cluster_spike_times = {}
        for cluster_id in sorted(cluster_channel_selection.keys()):
            times = (
                spike_inf_df.loc[spike_inf_df[spike_cluster_col] == cluster_id, "time"]
                .to_numpy(dtype=int)
            )
            times = _filter_spike_times(times)
            if times.size == 0:
                failed_clusters.append(int(cluster_id))
                continue
            cluster_spike_times[int(cluster_id)] = times

        if not cluster_spike_times:
            print("  [WARN] 该clique内无可用spike，跳过")
            continue

        cluster_windows_map = defaultdict(list)
        max_start = max(total_frames_day1 - segment_frames - 1, 0)
        if num_segments > 1 and max_start > 0:
            segment_starts = np.linspace(0, max_start, num=num_segments)
        else:
            segment_starts = np.array([0], dtype=float)
        segment_starts = np.unique(segment_starts.astype(int))

        print(f"  计划读取 {len(segment_starts)} 段数据，每段 {segment_frames} 帧")

        for start_frame in tqdm(segment_starts, desc=f"{probe_label}-{model_id} segments", leave=False):
            end_frame = min(start_frame + segment_frames, total_frames_day1)
            if end_frame - start_frame <= window_size:
                continue

            try:
                data_chunk = recording_f.get_traces(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    channel_ids=recording_channel_ids,
                )
            except Exception:
                continue

            data_chunk = data_chunk.T

            for cluster_id, times in cluster_spike_times.items():
                selection = cluster_channel_selection.get(cluster_id)
                if selection is None:
                    continue
                indices = selection["indices"]
                if not indices:
                    continue

                idx_start = np.searchsorted(times, start_frame, side="left")
                idx_end = np.searchsorted(times, end_frame, side="right")
                if idx_end <= idx_start:
                    continue
                selected_times = times[idx_start:idx_end]
                mask_valid = (
                    (selected_times >= start_frame + half_window)
                    & (selected_times < end_frame - half_window)
                )
                if not np.any(mask_valid):
                    continue
                selected_times = selected_times[mask_valid]
                rel_indices = (selected_times - start_frame).astype(int)
                if rel_indices.size == 0:
                    continue
                try:
                    data_chunk_cluster = data_chunk[indices, :]
                    chunk_windows = extract_windows(
                        data_chunk_cluster,
                        rel_indices,
                        window_size=window_size,
                    )
                except ValueError:
                    continue
                cluster_windows_map[cluster_id].append(chunk_windows)

        for cluster_id, selection in cluster_channel_selection.items():
            windows_list = cluster_windows_map.get(cluster_id)
            if not windows_list:
                failed_clusters.append(int(cluster_id))
                continue

            windows = np.concatenate(windows_list, axis=0)
            if windows.shape[0] < min_spikes_per_cluster:
                failed_clusters.append(int(cluster_id))
                continue

            rms_per_channel = np.sqrt(np.mean(windows**2, axis=(0, 2)))
            if np.allclose(rms_per_channel.sum(), 0):
                failed_clusters.append(int(cluster_id))
                continue

            weights = rms_per_channel / (rms_per_channel.sum() + 1e-12)
            mean_waveform_channels = np.mean(windows, axis=0)
            synth_waveform = np.sum(mean_waveform_channels * weights[:, None], axis=0)
            synth_waveform = synth_waveform - np.mean(synth_waveform)
            std_waveform = np.std(synth_waveform)
            if std_waveform > 0:
                synth_waveform = synth_waveform / std_waveform

            best_channel_positions = selection["positions"]
            cluster_pos = np.sum(best_channel_positions * weights[:, None], axis=0)

            model_templates[int(cluster_id)] = {
                "position": cluster_pos.tolist(),
                "waveform": synth_waveform.tolist(),
                "channel_waveforms": mean_waveform_channels.tolist(),
                "n_spikes": int(windows.shape[0]),
                "channel_ids": [int(ch) for ch in selection["channel_ids"]],
                "channel_indices": [int(idx) for idx in selection["channel_indices"]],
                "position_top_channels": len(selection["channel_ids"]),
            }
            updated_clusters += 1

        print(f"  [INFO] {probe_label} - 模型 {model_id} 已生成 {len(model_templates)} 个模板")

    print(f"\n{probe_label} 共生成 {updated_clusters} 个Day1 cluster模板。")
    if failed_clusters:
        preview_failed = sorted(set(failed_clusters))[:20]
        suffix_failed = "..." if len(set(failed_clusters)) > 20 else ""
        print(f"以下cluster因数据不足或窗口提取失败未生成模板: {preview_failed}{suffix_failed}")

    templates_output_path = sorting_results_dir / f"{probe_label}_day1_model_templates.pkl"
    with open(templates_output_path, "wb") as f:
        pickle.dump(day1_model_templates, f)
    print(f"{probe_label} Day1模板已保存到: {templates_output_path}")

    overall_templates_by_probe[probe_label] = day1_model_templates
    overall_failed_clusters[probe_label] = sorted(set(failed_clusters))


# === 汇总输出 ===
if overall_results_by_probe:
    summary_path = detection_results_root / "all_probes_results_summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(overall_results_by_probe, f)
    print(f"\n所有probe检测结果汇总保存至: {summary_path}")

if overall_templates_by_probe:
    templates_summary_path = sorting_results_dir / "all_probes_day1_model_templates.pkl"
    with open(templates_summary_path, "wb") as f:
        pickle.dump(overall_templates_by_probe, f)
    print(f"所有probe模板汇总保存至: {templates_summary_path}")

print("\n全部处理完成！")

