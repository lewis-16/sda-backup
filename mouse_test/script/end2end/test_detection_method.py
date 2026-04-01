import sys
import os
sys.path.append('/media/ubuntu/sda/mouse_test/script/end2end')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import scipy.signal
import scipy

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import probeinterface
from probeinterface import read_probeinterface

# 直接从detection.py复制函数，避免导入问题
def detect_spike(
    trace0_car,
    thr_min=5,
    thr_max=30,
    distance=3,
    ch_max_simul_firing=3,
    wlen=5,
    prominence=10,
):
    noise_std_detect = np.median(abs(trace0_car) / 0.6745, axis=0)
    thr = thr_min * noise_std_detect
    thrmax = thr_max * noise_std_detect

    spikes = np.zeros(trace0_car.shape)
    if trace0_car.ndim > 1:
        for i in range(noise_std_detect.shape[0]):
            peaks, props = scipy.signal.find_peaks(
                -trace0_car[:, i],
                thr[i],
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


def map_gt_annotation(detect_array, gt_array):
    gt_label_array1 = np.zeros((detect_array.shape[0],)) - 1

    for ind, i in enumerate(detect_array):
        f = 1
        indj = np.where(gt_array[:, 0] == i[0])[0]
        for j in indj:
            if gt_array[j, 1] == i[1]:
                f = 0
                break
        if f:
            indj = np.where(gt_array[:, 0] == i[0] - 1)[0]
            for j in indj:
                if gt_array[j, 1] == i[1]:
                    f = 0
                    break
        if f:
            indj = np.where(gt_array[:, 0] == i[0] + 1)[0]
            for j in indj:
                if gt_array[j, 1] == i[1]:
                    f = 0
                    break
        if f == 0:
            gt_label_array1[ind] = j

    return gt_label_array1

print("="*70)
print("使用detection.py的方法测试spike detection准确率")
print("="*70)

# 1. 加载数据（与train_full notebook一致）
print("\n1. 加载数据...")
session_name = 'mouse6_021322_natural_image_001'
recording_raw = se.read_blackrock(file_path=f'/media/ubuntu/sda/data/mouse6/ns4/natural_image/{session_name}')
recording_recorded = recording_raw.remove_channels(["98", '31', '32']).time_slice(start_time=0, end_time=2000)

probe_30channel = read_probeinterface('/media/ubuntu/sda/data/probe.json')
recording_recorded = recording_recorded.set_probegroup(probe_30channel)

recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
recording_cmr = spre.common_reference(recording_f, reference="global", operator="median")
recording_cmr = recording_cmr.rename_channels(['A-000', 'A-001', 'A-002', 'A-003', 'A-004',
                               'A-005', 'A-006', 'A-007', 'A-008', 'A-009',
                               'A-0010', 'A-011', 'A-012', 'A-013', 'A-014',
                               'A-015', 'A-016', 'A-017', 'A-018', 'A-019',
                               'A-020', 'A-021', 'A-022', 'A-023', 'A-024',
                               'A-025', 'A-026', 'A-027', 'A-028', 'A-029'])

print(f"  Recording shape: {recording_cmr.get_num_samples()} samples, {recording_cmr.get_num_channels()} channels")
print(f"  Duration: {recording_cmr.get_num_samples() / recording_cmr.get_sampling_frequency():.2f} seconds")

# 2. 限制为前300秒（用于快速测试）
print("\n2. 限制为前300秒（用于快速测试）...")
sampling_rate = recording_cmr.get_sampling_frequency()
max_samples = int(300 * sampling_rate)  # 先测试300秒
recording_cmr = recording_cmr.frame_slice(start_frame=0, end_frame=max_samples)
print(f"  限制后: {recording_cmr.get_num_samples()} samples ({recording_cmr.get_num_samples() / sampling_rate:.2f} seconds)")

# 3. 获取trace数据
print("\n3. 获取trace数据...")
trace0_car = recording_cmr.get_traces(segment_index=0)
print(f"  Trace shape: {trace0_car.shape}")

# 4. 使用detection.py的方法检测spikes
print("\n4. 使用detection.py的detect_spike方法检测spikes...")
print("  参数: thr_min=5, thr_max=30, distance=3, ch_max_simul_firing=3, wlen=5, prominence=10")
spikes = detect_spike(
    trace0_car,
    thr_min=5,
    thr_max=30,
    distance=3,
    ch_max_simul_firing=3,
    wlen=5,
    prominence=10,
)

# 5. 构建detect_array
print("\n5. 构建detect_array...")
spike_coords = np.argwhere(spikes == 1)
detect_array = spike_coords  # shape: (n_detected, 2), columns: [time, channel]
print(f"  检测到的spikes数量: {len(detect_array):,}")

# 6. 加载GT数据
print("\n6. 加载GT数据...")
output_folder = '/media/ubuntu/sda/mouse_test/script/end2end/spike_sorting_model'
session_data_folder = f'{output_folder}/clique_0/{session_name}'
gt_detect_array_path = f'{session_data_folder}/gt_detect_array.csv'

if not os.path.exists(gt_detect_array_path):
    raise FileNotFoundError(f"GT文件不存在: {gt_detect_array_path}")

gt_detect_array = pd.read_csv(gt_detect_array_path)
print(f"  GT spikes数量: {len(gt_detect_array):,}")

# 7. 筛选GT数据（只保留前1500秒内的spikes）
print("\n7. 筛选GT数据（只保留前1500秒内的spikes）...")
gt_detect_array_filtered = gt_detect_array[gt_detect_array['time'] < max_samples].copy()
print(f"  筛选后GT spikes数量: {len(gt_detect_array_filtered):,}")

# 8. 构建gt_array（格式：[time, channel]）
print("\n8. 构建gt_array...")
print(f"  GT文件列名: {gt_detect_array_filtered.columns.tolist()}")
# gt_detect_array的列可能是'time'和'channel'或'extremum_channel'或'extremum_channels'
if 'extremum_channel' in gt_detect_array_filtered.columns:
    channel_col = 'extremum_channel'
elif 'channel' in gt_detect_array_filtered.columns:
    channel_col = 'channel'
elif 'extremum_channels' in gt_detect_array_filtered.columns:
    channel_col = 'extremum_channels'
else:
    # 尝试找到包含'channel'的列
    channel_cols = [col for col in gt_detect_array_filtered.columns if 'channel' in col.lower()]
    if len(channel_cols) > 0:
        channel_col = channel_cols[0]
        print(f"  使用列: {channel_col}")
    else:
        raise ValueError(f"GT文件中找不到channel列，可用列: {gt_detect_array_filtered.columns.tolist()}")

# 需要将channel名称转换为索引
channel_names = recording_cmr.get_channel_ids()
channel_name_to_idx = {name: idx for idx, name in enumerate(channel_names)}

gt_times = gt_detect_array_filtered['time'].values
gt_channels = gt_detect_array_filtered[channel_col].values

# 转换channel名称到索引
gt_channel_indices = []
for ch_name in gt_channels:
    if ch_name in channel_name_to_idx:
        gt_channel_indices.append(channel_name_to_idx[ch_name])
    else:
        print(f"  警告: channel {ch_name} 不在recording中，跳过")
        continue

# 只保留有效的GT spikes
valid_mask = np.array([ch_name in channel_name_to_idx for ch_name in gt_channels])
gt_times_valid = gt_times[valid_mask]
gt_channel_indices = np.array(gt_channel_indices)

gt_array = np.column_stack([gt_times_valid, gt_channel_indices])
print(f"  有效GT spikes数量: {len(gt_array):,}")

# 9. 使用map_gt_annotation匹配
print("\n9. 使用map_gt_annotation匹配GT spikes...")
gt_label_array = map_gt_annotation(detect_array, gt_array)
matched_indices = np.where(gt_label_array > -1)[0]
n_matched = len(matched_indices)
n_detected = len(detect_array)
n_gt = len(gt_array)

print(f"  匹配结果:")
print(f"    检测到的spikes: {n_detected:,}")
print(f"    GT spikes: {n_gt:,}")
print(f"    匹配的spikes: {n_matched:,}")
print(f"    召回率 (Recall): {n_matched/n_gt:.4f} ({n_matched/n_gt*100:.2f}%)")
print(f"    精确率 (Precision): {n_matched/n_detected:.4f} ({n_matched/n_detected*100:.2f}%)")

# 10. 计算F1 score
f1_score = 2 * (n_matched/n_gt) * (n_matched/n_detected) / ((n_matched/n_gt) + (n_matched/n_detected))
print(f"    F1 Score: {f1_score:.4f}")

# 11. 分析noise比例
print("\n10. 分析noise比例...")
n_noise = n_detected - n_matched
noise_ratio = n_noise / n_detected
print(f"    Noise spikes: {n_noise:,}")
print(f"    Noise比例: {noise_ratio:.4f} ({noise_ratio*100:.2f}%)")

# 12. 对比当前utils_clique.py中的方法
print("\n" + "="*70)
print("对比分析：detection.py vs utils_clique.py")
print("="*70)
print("\n当前train_full notebook使用的参数（utils_clique.py）:")
print("  thr_min=4, thr_max=10, distance=3, prominence=15")
print("\ndetection.py使用的参数:")
print("  thr_min=5, thr_max=30, distance=3, prominence=10, ch_max_simul_firing=3")
print("\n主要区别:")
print("  1. detection.py使用更严格的thr_max (30 vs 10)")
print("  2. detection.py使用更低的prominence (10 vs 15)")
print("  3. detection.py有ch_max_simul_firing限制 (3个channel同时firing)")
print("  4. detection.py有额外的peak height过滤 (>10)")

print("\n" + "="*70)
print("测试完成！")
print("="*70)

