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
import neo
from elephant.kernels import GaussianKernel
from elephant.statistics import instantaneous_rate
from quantities import ms

# PSTH参数设置
gk = GaussianKernel(25 * ms)  # 25ms的Gaussian kernel
bin_size_ms = 10  # 10ms的bin size
bin_size_s = bin_size_ms / 1000.0

# 计算时间轴（使用extended时间窗）
total_time_extended = EXTEND_TIME + STIMULUS_DURATION + EXTEND_TIME  # 0.45秒（如果EXTEND_TIME=0.1）
time_bins = np.arange(0, total_time_extended, bin_size_s)
n_time_bins = len(time_bins)

print(f"PSTH参数:")
print(f"  Total time: {total_time_extended} s")
print(f"  Bin size: {bin_size_ms} ms")
print(f"  Number of time bins: {n_time_bins}")

# 收集所有trials和neurons的PSTH
# 首先获取所有trials和所有neurons
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

# 初始化PSTH矩阵: (n_trial, n_time_bins, n_neuron)
psth_matrix = np.zeros((n_trials, n_time_bins, n_neurons))
trial_image_id = []

print(f"\n开始计算PSTH...")
print(f"矩阵形状: ({n_trials}, {n_time_bins}, {n_neurons})")

# 遍历所有trials
for trial_idx, (_, trial) in enumerate(all_trials.iterrows()):
    if trial_idx % 50 == 0:
        print(f"  处理trial {trial_idx}/{n_trials}")
    
    # 获取trial的image信息
    image_id = trial.get('image', 'unknown')
    trial_image_id.append(image_id)
    
    start_ext = int(trial['start_extended'])
    end_ext = int(trial['end_extended'])
    
    # 遍历所有neurons
    for neuron_idx, neuron_key in enumerate(all_neuron_keys):
        neuron_spikes = all_neuron_spike_data[neuron_key]
        
        # 获取该trial内的spikes
        trial_spikes = neuron_spikes[(neuron_spikes >= start_ext) & (neuron_spikes <= end_ext)]
        
        if len(trial_spikes) > 0:
            # 转换为相对时间（秒）
            relative_spikes = (trial_spikes - start_ext) / SAMPLING_RATE
            
            # 创建SpikeTrain对象
            spiketrain = neo.SpikeTrain(
                relative_spikes * 1000 * ms, 
                t_stop=total_time_extended * 1000 * ms, 
                t_start=0 * ms
            )
            
            # 计算instantaneous rate
            inst_rate = instantaneous_rate(spiketrain, kernel=gk, sampling_period=bin_size_ms * ms)
            psth_trial = inst_rate.magnitude.flatten()
        else:
            psth_trial = np.zeros(n_time_bins)
        
        # 确保长度一致
        if len(psth_trial) < n_time_bins:
            psth_trial = np.pad(psth_trial, (0, n_time_bins - len(psth_trial)), 'constant')
        elif len(psth_trial) > n_time_bins:
            psth_trial = psth_trial[:n_time_bins]
        
        # 存储到矩阵中
        psth_matrix[trial_idx, :, neuron_idx] = psth_trial

print(f"\n完成! PSTH矩阵形状: {psth_matrix.shape}")
print(f"Trial image ID列表长度: {len(trial_image_id)}")

# 保存结果
psth_output_path = os.path.join(OUTPUT_DIR, "psth_matrix.npy")
trial_image_output_path = os.path.join(OUTPUT_DIR, "trial_image_id.pkl")

np.save(psth_output_path, psth_matrix)
with open(trial_image_output_path, 'wb') as f:
    pickle.dump(trial_image_id, f)

print(f"\n已保存:")
print(f"  PSTH矩阵: {psth_output_path}")
print(f"  Trial image ID: {trial_image_output_path}")



