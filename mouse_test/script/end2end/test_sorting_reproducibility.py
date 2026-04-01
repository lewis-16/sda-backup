import sys
import os
sys.path.append('/media/ubuntu/sda/mouse_test/script/end2end')

import spikeinterface as si
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import spikeinterface.qualitymetrics as sqm
import json
import probeinterface
from probeinterface import Probe, ProbeGroup, read_probeinterface

import numpy as np
from spikeinterface.core import concatenate_recordings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pathlib import Path
import pickle

from spikeinterface.core import get_template_extremum_channel
import scipy.spatial.distance
from scipy.sparse.csgraph import connected_components

print("="*70)
print("测试sorting重复性：使用第一个月数据进行sorting，然后与GT匹配")
print("="*70)

# 输出目录
output_base = '/media/ubuntu/sda/mouse_test/processed_results/test'
os.makedirs(output_base, exist_ok=True)

# ===== 步骤1: 加载第一个月的数据并进行sorting（参考sorting.ipynb） =====
print("\n" + "="*70)
print("步骤1: 加载第一个月的数据并进行sorting")
print("="*70)

session_name = 'mouse6_021322_natural_image_001'
print(f"\n加载session: {session_name}")

# 加载数据（限制为前1500秒，与train_full notebook一致）
recording_raw = se.read_blackrock(file_path=f'/media/ubuntu/sda/data/mouse6/ns4/natural_image/{session_name}')
recording_recorded = recording_raw.remove_channels(["98", '31', '32']).time_slice(start_time=0, end_time=1500)

probe_30channel = read_probeinterface('/media/ubuntu/sda/data/probe.json')
recording_recorded = recording_recorded.set_probegroup(probe_30channel)

# 预处理（完全按照sorting.ipynb）
recording_f = spre.bandpass_filter(recording_recorded, freq_min=300, freq_max=3000)
recording_notch = spre.notch_filter(recording_f, freq=60)
recording_cmr = spre.common_reference(recording_notch, reference="global", operator="median")
recording_cmr = recording_cmr.rename_channels(['A-000', 'A-001', 'A-002', 'A-003', 'A-004',
                               'A-005', 'A-006', 'A-007', 'A-008', 'A-009',
                               'A-0010', 'A-011', 'A-012', 'A-013', 'A-014',
                               'A-015', 'A-016', 'A-017', 'A-018', 'A-019',
                               'A-020', 'A-021', 'A-022', 'A-023', 'A-024',
                               'A-025', 'A-026', 'A-027', 'A-028', 'A-029'])

print(f"预处理完成: {recording_cmr.get_num_samples()} samples, {recording_cmr.get_num_channels()} channels")
print(f"Duration: {recording_cmr.get_num_samples() / recording_cmr.get_sampling_frequency():.2f} seconds")

# 保存为binary格式（Mountainsort4需要）
print("\n保存为binary格式...")
recording_preprocessed = recording_cmr.save(format="binary", n_jobs=20)
print("保存完成")

# Mountainsort4参数（完全按照sorting.ipynb）
default_params = {
    'detect_sign': -1,
    'adjacency_radius': 120,
    'freq_min': 300,
    'freq_max': 3000,
    'filter': True,
    'whiten': True,
    'num_workers': 20,
    'clip_size': 50,
    'detect_threshold': 5,
    'detect_interval': 3,
}

# 运行Mountainsort4
print("\n运行Mountainsort4 sorting...")
sorting_output_folder = os.path.join(output_base, 'mountainsort4_output')
firings_path = os.path.join(sorting_output_folder, 'sorter_output', 'firings.npz')

if os.path.exists(firings_path):
    print(f"发现已存在的sorting结果，直接加载: {firings_path}")
    sorting_mountainsort = se.NpzSortingExtractor(firings_path)
else:
    sorting_mountainsort = ss.run_sorter(
        sorter_name='mountainsort4',
        recording=recording_preprocessed,
        remove_existing_folder=True,
        folder=sorting_output_folder,
        **default_params
    )

print(f"Sorting完成，检测到 {len(sorting_mountainsort.unit_ids)} 个units")

# ===== 步骤2: 导出到phy格式 =====
print("\n" + "="*70)
print("步骤2: 导出到phy格式")
print("="*70)

phy_folder = os.path.join(output_base, 'phy_folder')
print(f"\n导出到phy格式: {phy_folder}")

analyzer_mountainsort = si.create_sorting_analyzer(
    sorting=sorting_mountainsort,
    recording=recording_preprocessed,
    format='binary_folder',
    folder=os.path.join(output_base, 'analyzer_binary')
)

# 计算必要的extensions
extensions_to_compute = [
    "random_spikes",
    "waveforms",
    "noise_levels",
    "templates",
    "unit_locations",
]

extension_params = {
    "unit_locations": {"method": "center_of_mass"},
}

print("计算extensions...")
analyzer_mountainsort.compute(extensions_to_compute, extension_params=extension_params, n_jobs=20)
print("Extensions计算完成")

# 导出到phy
import spikeinterface.exporters as sexp
sexp.export_to_phy(analyzer_mountainsort, phy_folder, verbose=True, n_jobs=20)
print("Phy导出完成")

# ===== 步骤3: 读取phy结果并进行post_sort（参考recordings_30channels_12_month.ipynb） =====
print("\n" + "="*70)
print("步骤3: 读取phy结果并进行post_sort")
print("="*70)

# 注意：不排除noise cluster
print("\n读取phy结果（不排除noise cluster）...")
sorting_curated_phy = se.read_phy(phy_folder)  # 不设置exclude_cluster_groups
print(f"读取到 {len(sorting_curated_phy.unit_ids)} 个units\n")

# 创建analyzer
print("创建analyzer并计算extensions...")
analyzer_curated_phy = si.create_sorting_analyzer(
    sorting=sorting_curated_phy,
    recording=recording_cmr,
    format='binary_folder',
    folder=os.path.join(output_base, 'analyzer_curated_temp'),
    n_jobs=20,
    verbose=False
)

# 计算extensions
extensions_to_compute = [
    "random_spikes",
    "waveforms",
    "noise_levels",
    "templates",
    "unit_locations",
]

extension_params = {
    "unit_locations": {"method": "center_of_mass"},
}

analyzer_curated_phy.compute(extensions_to_compute, extension_params=extension_params, n_jobs=20)
print("完成extensions计算")

# ===== 步骤4: 生成detect_array（参考recordings_30channels_12_month.ipynb） =====
print("\n" + "="*70)
print("步骤4: 生成detect_array")
print("="*70)

# 获取unit信息
sorting_final = sorting_curated_phy
unit_ids_list_final = sorting_final.unit_ids.tolist()
print(f"共 {len(unit_ids_list_final)} 个units")

# 获取templates和extremum channels
templates = analyzer_curated_phy.get_extension("templates").get_data()
unit_locations = analyzer_curated_phy.get_extension("unit_locations").get_data()
noise_levels = analyzer_curated_phy.get_extension("noise_levels").get_data()

print("计算extremum channels...")
extremum_channels_final = {}
for unit_id in unit_ids_list_final:
    unit_idx = list(unit_ids_list_final).index(unit_id)
    template = templates[unit_idx]
    extremum_channel = get_template_extremum_channel(template, peak_sign='neg')
    extremum_channels_final[unit_id] = extremum_channel

print("计算unit locations...")
unit_locations_final = unit_locations

# 计算position_waveforms
print("计算position_waveforms...")
position_waveforms_final = unit_locations_final

# 计算channel_ids（每个unit的所有channels）
print("计算每个unit的channel_id...")
channel_ids_list = recording_cmr.get_channel_ids().tolist()
channel_ids_dict = {}

for unit_id in unit_ids_list_final:
    unit_idx = list(unit_ids_list_final).index(unit_id)
    template = templates[unit_idx]  # shape: (n_samples, n_channels)
    
    # 计算每个channel的amplitude（使用模板的峰值）
    channel_amplitudes = np.abs(np.min(template, axis=0))  # 负向峰值
    
    # 使用阈值筛选channels（例如，amplitude > 噪声水平的3倍）
    noise_threshold = noise_levels[unit_idx] * 3
    significant_channels = np.where(channel_amplitudes > noise_threshold)[0]
    
    # 转换为channel IDs
    unit_channel_ids = [str(channel_ids_list[ch_idx]) for ch_idx in significant_channels]
    channel_ids_dict[unit_id] = unit_channel_ids

print(f"完成channel_id计算，共处理{len(channel_ids_dict)}个units\n")

# 计算channel_snr
print("计算channel_snr...")
# 采样一些spikes来计算SNR
all_spike_times = []
all_spike_unit_ids = []
for unit_id in unit_ids_list_final:
    spike_times = sorting_final.get_unit_spike_train(unit_id).tolist()
    all_spike_times.extend(spike_times)
    all_spike_unit_ids.extend([unit_id] * len(spike_times))

n_spikes_total = len(all_spike_times)
n_spikes_sample = min(1000, n_spikes_total)
if n_spikes_sample > 0:
    random_indices = np.random.choice(n_spikes_total, size=n_spikes_sample, replace=False)
    sampled_spike_times = [all_spike_times[i] for i in random_indices]
    sampled_spike_unit_ids = [all_spike_unit_ids[i] for i in random_indices]
else:
    sampled_spike_times = []
    sampled_spike_unit_ids = []

# 计算noise std
noise_std_detect = np.median(np.abs(recording_cmr.get_traces()), axis=0) / 0.6745

channel_snr_dict = {}
left_sample = 10
right_sample = 20

for unit_id in unit_ids_list_final:
    channel_snr_dict[unit_id] = {}
    unit_spike_times = [st for st, uid in zip(sampled_spike_times, sampled_spike_unit_ids) if uid == unit_id]
    
    if len(unit_spike_times) == 0:
        unit_spike_times = sorting_final.get_unit_spike_train(unit_id).tolist()
        if len(unit_spike_times) > 1000:
            unit_spike_times = np.random.choice(unit_spike_times, size=1000, replace=False).tolist()
    
    unit_waveforms = []
    valid_spike_times = []
    
    for spike_time in unit_spike_times:
        start = spike_time - left_sample
        end = spike_time + right_sample
        
        if start < 0:
            start = 0
        if end > recording_cmr.get_num_samples():
            end = recording_cmr.get_num_samples()
        
        waveform = recording_cmr.get_traces(start_frame=start, end_frame=end)
        unit_waveforms.append(waveform)
        valid_spike_times.append(spike_time)
    
    if len(unit_waveforms) == 0:
        continue
    
    unit_waveforms = np.array(unit_waveforms)
    spike_time_values = unit_waveforms[:, left_sample, :]
    channel_amplitudes = np.mean(spike_time_values, axis=0)
    channel_snr = np.abs(channel_amplitudes) / noise_std_detect
    
    unit_channel_ids = channel_ids_dict.get(unit_id, [])
    
    for ch_idx, snr_value in enumerate(channel_snr):
        channel_id = str(channel_ids_list[ch_idx])
        if channel_id in unit_channel_ids:
            channel_snr_dict[unit_id][channel_id] = float(snr_value)

print(f"完成channel_snr计算，共处理{len(channel_snr_dict)}个units\n")

# 生成neuron_inf
neuron_inf_all = {}
for idx, unit_id in enumerate(unit_ids_list_final):
    neuron_inf_all[unit_id] = {
        'location_x': float(unit_locations_final[idx, 0]),
        'location_y': float(unit_locations_final[idx, 1]),
        'position_waveform': position_waveforms_final[idx],
        'extremum_channel': extremum_channels_final[unit_id],
        'channel_id': channel_ids_dict[unit_id],
        'channel_snr': channel_snr_dict.get(unit_id, {})
    }

# 生成detect_array（所有spikes）
print("生成detect_array...")
spike_vector_final = sorting_final.to_spike_vector()
detect_data_all = []

for spike in spike_vector_final:
    unit_index = spike['unit_index']
    unit_id = sorting_final.unit_ids[unit_index]
    sample_index = spike['sample_index']
    
    extremum_channel = extremum_channels_final[unit_id]
    
    detect_data_all.append({
        'time': sample_index,
        'unit_id': unit_id,
        'extremum_channel': str(extremum_channel),
    })

detect_array_df = pd.DataFrame(detect_data_all)
print(f"完成detect_array生成，共{len(detect_array_df)}个spikes\n")

# 保存结果
print("保存结果...")
with open(os.path.join(output_base, 'neuron_inf.pickle'), 'wb') as f:
    pickle.dump(neuron_inf_all, f)
detect_array_df.to_csv(os.path.join(output_base, 'detect_array.csv'), index=False)
print(f"已保存到: {output_base}")

# ===== 步骤5: 与GT进行匹配 =====
print("\n" + "="*70)
print("步骤5: 与GT进行匹配")
print("="*70)

# 加载GT数据
gt_folder = '/media/ubuntu/sda/mouse_test/script/end2end/spike_sorting_model/clique_0'
gt_detect_array_path = os.path.join(gt_folder, session_name, 'gt_detect_array.csv')
gt_neuron_inf_path = os.path.join(gt_folder, session_name, 'neuron_inf.pickle')

if not os.path.exists(gt_detect_array_path) or not os.path.exists(gt_neuron_inf_path):
    raise FileNotFoundError(f"GT文件不存在: {gt_detect_array_path} 或 {gt_neuron_inf_path}")

gt_detect_array = pd.read_csv(gt_detect_array_path)
with open(gt_neuron_inf_path, 'rb') as f:
    gt_neuron_inf = pickle.load(f)

print(f"GT spikes数量: {len(gt_detect_array):,}")
print(f"GT neurons数量: {len(gt_neuron_inf):,}")

# 构建detect_array和gt_array（格式：[time, channel]）
print("\n构建detect_array和gt_array...")
channel_names = recording_cmr.get_channel_ids()
channel_name_to_idx = {name: idx for idx, name in enumerate(channel_names)}

# detect_array
detect_times = detect_array_df['time'].values
detect_channels = detect_array_df['extremum_channel'].values
detect_channel_indices = [channel_name_to_idx.get(str(ch), -1) for ch in detect_channels]
valid_detect_mask = np.array([idx >= 0 for idx in detect_channel_indices])
detect_array = np.column_stack([
    detect_times[valid_detect_mask],
    np.array(detect_channel_indices)[valid_detect_mask]
])

# gt_array
gt_times = gt_detect_array['time'].values
gt_channels = gt_detect_array['extremum_channel'].values
gt_channel_indices = [channel_name_to_idx.get(str(ch), -1) for ch in gt_channels]
valid_gt_mask = np.array([idx >= 0 for idx in gt_channel_indices])
gt_array = np.column_stack([
    gt_times[valid_gt_mask],
    np.array(gt_channel_indices)[valid_gt_mask]
])

print(f"有效detect spikes: {len(detect_array):,}")
print(f"有效GT spikes: {len(gt_array):,}")

# GT匹配函数
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

# 匹配
print("\n进行GT匹配...")
gt_label_array = map_gt_annotation(detect_array, gt_array)
matched_indices = np.where(gt_label_array > -1)[0]
n_matched = len(matched_indices)
n_detected = len(detect_array)
n_gt = len(gt_array)

print(f"\n匹配结果:")
print(f"  检测到的spikes: {n_detected:,}")
print(f"  GT spikes: {n_gt:,}")
print(f"  匹配的spikes: {n_matched:,}")
print(f"  召回率 (Recall): {n_matched/n_gt:.4f} ({n_matched/n_gt*100:.2f}%)")
print(f"  精确率 (Precision): {n_matched/n_detected:.4f} ({n_matched/n_detected*100:.2f}%)")

f1_score = 2 * (n_matched/n_gt) * (n_matched/n_detected) / ((n_matched/n_gt) + (n_matched/n_detected)) if (n_matched/n_gt + n_matched/n_detected) > 0 else 0
print(f"  F1 Score: {f1_score:.4f}")

n_noise = n_detected - n_matched
noise_ratio = n_noise / n_detected if n_detected > 0 else 0
print(f"  Noise比例: {noise_ratio:.4f} ({noise_ratio*100:.2f}%)")

# 分析未匹配的GT spikes
matched_gt_indices = gt_label_array[matched_indices].astype(int)
all_gt_indices = np.arange(len(gt_array))
unmatched_gt_indices = np.setdiff1d(all_gt_indices, matched_gt_indices)
n_unmatched_gt = len(unmatched_gt_indices)

print(f"\n未匹配的GT spikes: {n_unmatched_gt:,} ({n_unmatched_gt/n_gt*100:.2f}%)")

if n_unmatched_gt > 0:
    unmatched_gt_df = gt_detect_array.iloc[valid_gt_mask][unmatched_gt_indices]
    unmatched_neuron_ids = unmatched_gt_df['unit_id'].values
    
    unique_neurons, neuron_counts = np.unique(unmatched_neuron_ids, return_counts=True)
    neuron_counts_sorted_idx = np.argsort(neuron_counts)[::-1]
    
    print(f"\n未匹配GT spikes的neuron分布（前10个）:")
    for i, idx in enumerate(neuron_counts_sorted_idx[:10]):
        neuron_id = unique_neurons[idx]
        count = neuron_counts[idx]
        percentage = count / n_unmatched_gt * 100
        print(f"  Neuron {neuron_id}: {count:,} spikes ({percentage:.2f}%)")

print("\n" + "="*70)
print("测试完成！结果已保存到:")
print(f"  {output_base}")
print("="*70)

