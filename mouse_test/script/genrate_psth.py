# %%
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

import spikeinterface.extractors as se
from spikeinterface.core import concatenate_recordings

# 配置路径
RAW_DATA_DIR = "/media/ubuntu/sda/mouse_test/raw_data/WLF_128ch2mouse1_natima_251215_232216"
MOUSE1_BASE_DIR = "/media/ubuntu/sda/mouse_test/sorted/combined_mountain_sort/mouse1"
OUTPUT_DIR = "/media/ubuntu/sda/mouse_test/sorted/combined_mountain_sort/mouse1/date_1215_results"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# %%
SAMPLING_RATE_ORIGINAL = 20000  # Hz，原始采样率
SAMPLING_RATE = 10000  # Hz，目标采样率（用于后续计算）
EXTEND_TIME = 0.025  # 秒，左右延长0.1s
STIMULUS_DURATION = 0.25  # 秒，刺激持续时间

file_list = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.rhd')]
file_list = sorted(file_list)

recording_raw_list = []
for file in file_list:
    file_path = os.path.join(RAW_DATA_DIR, file)
    recording_raw_list.append(se.read_intan(file_path, stream_id='4'))

recording_raw = concatenate_recordings(recording_list=recording_raw_list)

trigger = recording_raw.get_traces().astype(int)[:, 1]

# 找到所有trigger事件
trigger_indices = np.where(trigger > 0.5)[0]

# 找到连续段的起点
start_indices = trigger_indices[np.concatenate(([True], np.diff(trigger_indices) > 1))]

# 将索引从20000 Hz采样率转换为10000 Hz采样率（除以2）
start_indices = start_indices // 2

trigger_log_csv = os.path.join(RAW_DATA_DIR, "log_234330.csv")
if os.path.exists(trigger_log_csv):
    trigger_log = pd.read_csv(trigger_log_csv)
    
    trigger_log['start_index'] = None
    trigger_log['start_index_2'] = None  
    
    start_idx_counter = 0
    for i in range(len(trigger_log)):
        if trigger_log.loc[i, 'info_type'] == 'rest':
            # rest行对应两个start_indices
            if start_idx_counter < len(start_indices):
                trigger_log.loc[i, 'start_index'] = start_indices[start_idx_counter]
                start_idx_counter += 1
            if start_idx_counter < len(start_indices):
                trigger_log.loc[i, 'start_index_2'] = start_indices[start_idx_counter]
                start_idx_counter += 1
        else:
            # 非rest行对应一个start_indices
            if start_idx_counter < len(start_indices):
                trigger_log.loc[i, 'start_index'] = start_indices[start_idx_counter]
                start_idx_counter += 1
    

    trigger_log = trigger_log[trigger_log['info_type'] != 'rest']
    
    if 'image_name' in trigger_log.columns:
        trigger_log['image'] = None
        trigger_log.index = range(len(trigger_log))
        for i in range(len(trigger_log)):
            img_name = str(trigger_log.loc[i, 'image_name'])
            if img_name and img_name != 'nan':
                parts = img_name.split('.')[0].split('_')
                if len(parts) >= 3:
                    trigger_log.loc[i, 'image'] = parts[2]
    
    trigger_log['start_extended'] = trigger_log['start_index'] - EXTEND_TIME * SAMPLING_RATE
    trigger_log['end_index'] = trigger_log['start_index'] + STIMULUS_DURATION * SAMPLING_RATE
    trigger_log['end_extended'] = trigger_log['start_index'] + (STIMULUS_DURATION + EXTEND_TIME) * SAMPLING_RATE
    
    trigger_log_path = os.path.join(OUTPUT_DIR, "trigger_inf_WLF_128ch2mouse1_natima_251215_232216.csv")
    trigger_log.to_csv(trigger_log_path, index=False)
    print(f"\nTrigger log已保存: {trigger_log_path}")




# %%
print("\n" + "=" * 60)
print("Step 2: 读取各个clique的neuron数据")
print("=" * 60)

# 获取所有clique目录
clique_dirs = []
for item in os.listdir(MOUSE1_BASE_DIR):
    clique_path = os.path.join(MOUSE1_BASE_DIR, item)
    date_1215_path = os.path.join(clique_path, "date_1215")
    if os.path.isdir(clique_path) and item.startswith("clique_") and os.path.exists(date_1215_path):
        clique_dirs.append((item, date_1215_path))

clique_dirs = sorted(clique_dirs)
print(f"找到 {len(clique_dirs)} 个clique目录: {[c[0] for c in clique_dirs]}")

# 读取各个clique的neuron信息
clique_neuron_data = {}
for clique_name, clique_date_dir in clique_dirs:
    neuron_inf_path = os.path.join(clique_date_dir, "neuron_inf.pickle")
    if os.path.exists(neuron_inf_path):
        print(f"\n处理 {clique_name}:")
        print(f"  读取: {neuron_inf_path}")
        with open(neuron_inf_path, 'rb') as f:
            neuron_inf = pickle.load(f)
        
        # 从gt_detect_array.csv读取spike数据
        gt_detect_path = os.path.join(clique_date_dir, "gt_detect_array.csv")
        spike_data = {}
        
        if os.path.exists(gt_detect_path):
            try:
                print(f"  读取: {gt_detect_path}")
                gt_detect_df = pd.read_csv(gt_detect_path)
                
                # 检查必要的列
                if 'time' not in gt_detect_df.columns or 'unit_id' not in gt_detect_df.columns:
                    print(f"  警告: gt_detect_array.csv缺少必要的列 (time, unit_id)")
                    spike_data = None
                else:
                    # 确保time和unit_id是数值类型
                    gt_detect_df['time'] = pd.to_numeric(gt_detect_df['time'], errors='coerce')
                    gt_detect_df['unit_id'] = pd.to_numeric(gt_detect_df['unit_id'], errors='coerce')
                    gt_detect_df = gt_detect_df.dropna(subset=['time', 'unit_id'])
                    
                    # 为每个neuron提取spike times（保持为采样点，不转换为秒）
                    for neuron_id in neuron_inf.keys():
                        neuron_spikes = gt_detect_df[gt_detect_df['unit_id'] == neuron_id]['time'].values
                        if len(neuron_spikes) > 0:
                            spike_data[neuron_id] = neuron_spikes
                    
                    print(f"  成功加载 {len(spike_data)} 个neuron的spike数据")
                    print(f"  总spike数量: {len(gt_detect_df)}")
            except Exception as e:
                print(f"  警告: 无法读取gt_detect_array.csv: {e}")
                import traceback
                traceback.print_exc()
                spike_data = None
        else:
            print(f"  警告: gt_detect_array.csv不存在: {gt_detect_path}")
            spike_data = None
        
        clique_neuron_data[clique_name] = {
            'neuron_inf': neuron_inf,
            'spike_data': spike_data,
            'clique_dir': clique_date_dir
        }
        print(f"  加载了 {len(neuron_inf)} 个neuron信息")
    else:
        print(f"  警告: 未找到neuron_inf.pickle: {neuron_inf_path}")



# %%
# 计算firing rate反应矩阵 (n_trial, n_neuron)
# 参考 generate_psth.ipynb 的方法，计算指定时间窗口内的firing rate

# 时间窗口参数（相对于trial开始时间）
window_start_ms = 50  # 50ms
window_end_ms = 200   # 200ms
window_start_s = window_start_ms / 1000.0  # 0.05秒
window_end_s = window_end_ms / 1000.0      # 0.2秒
window_duration_s = window_end_s - window_start_s  # 0.15秒

print(f"\n{'='*60}")
print("计算firing rate反应矩阵 (50ms-200ms时间窗口)")
print(f"{'='*60}")
print(f"时间窗口: {window_start_ms}ms - {window_end_ms}ms ({window_duration_s*1000:.0f}ms)")

# 收集所有trials和neurons
all_trials = trigger_log.copy().reset_index(drop=True)
n_trials = len(all_trials)

# 收集所有clique的所有neurons（使用(clique_name, neuron_id)作为唯一标识）
all_neuron_keys = []  # 存储(clique_name, neuron_id)的列表
all_neuron_spike_data = {}  # {(clique_name, neuron_id): spike_times_array}

for clique_name, data_dict in clique_neuron_data.items():
    neuron_inf = data_dict['neuron_inf']
    spike_data = data_dict['spike_data']
    
    if spike_data is None:
        continue
    
    for neuron_id, neuron_spikes in spike_data.items():
        neuron_key = (clique_name, neuron_id)
        if neuron_key not in all_neuron_keys:
            all_neuron_keys.append(neuron_key)
            # 将spike times从原始采样率转换为目标采样率（除以2）
            all_neuron_spike_data[neuron_key] = neuron_spikes // 2

# 按照(clique_name, neuron_id)排序
all_neuron_keys = sorted(all_neuron_keys)
n_neurons = len(all_neuron_keys)

print(f"\n找到 {n_trials} 个trials")
print(f"找到 {n_neurons} 个neurons")

# 初始化firing rate矩阵: (n_trial, n_neuron)
firing_rate_matrix = np.zeros((n_trials, n_neurons))
trial_image_id = []

print(f"\n开始计算firing rate矩阵...")
print(f"矩阵形状: ({n_trials}, {n_neurons})")

# 遍历所有trials
for trial_idx, (_, trial) in enumerate(all_trials.iterrows()):
    if trial_idx % 50 == 0:
        print(f"  处理trial {trial_idx}/{n_trials}")
    
    # 获取trial的image信息
    image_id = trial.get('image', 'unknown')
    trial_image_id.append(image_id)
    
    # 计算时间窗口（相对于trial开始时间）
    # trial的开始时间是trigger的start_index（10kHz采样率）
    trial_start = int(trial['start_index'])
    
    # 计算时间窗口的起始和结束采样点
    window_start_sample = trial_start + int(window_start_s * SAMPLING_RATE)
    window_end_sample = trial_start + int(window_end_s * SAMPLING_RATE)
    
    # 遍历所有neurons
    for neuron_idx, neuron_key in enumerate(all_neuron_keys):
        neuron_spikes = all_neuron_spike_data[neuron_key]
        
        # 获取时间窗口内的spikes
        window_spikes = neuron_spikes[
            (neuron_spikes >= window_start_sample) & 
            (neuron_spikes < window_end_sample)
        ]
        
        # 计算firing rate (spikes per second)
        spike_count = len(window_spikes)
        firing_rate = spike_count / window_duration_s
        firing_rate_matrix[trial_idx, neuron_idx] = firing_rate


# 保存结果
firing_rate_output_path = os.path.join(OUTPUT_DIR, "firing_rate_50ms_200ms.npy")
trial_image_output_path = os.path.join(OUTPUT_DIR, "trial_image_id.pkl")

np.save(firing_rate_output_path, firing_rate_matrix)
with open(trial_image_output_path, 'wb') as f:
    pickle.dump(trial_image_id, f)



