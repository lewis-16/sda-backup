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
trigger = recording_raw.get_traces().astype(int)

# %%
trigger.shape

# %%
SAMPLING_RATE_ORIGINAL = 20000  # Hz，原始采样率
SAMPLING_RATE = 10000  # Hz，目标采样率（用于后续计算）
EXTEND_TIME = 0.1  # 秒，左右延长0.1s
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
print("\n" + "=" * 60)
print("Step 3: 绘制raster plot")
print("=" * 60)

# 为每个clique生成raster plot
pdf_path = os.path.join(OUTPUT_DIR, "raster_plots_all_cliques_1215.pdf")
print(f"保存raster plots到: {pdf_path}")

with PdfPages(pdf_path) as pdf:
    for clique_name, data_dict in clique_neuron_data.items():
        neuron_inf = data_dict['neuron_inf']
        spike_data = data_dict['spike_data']
        
        unique_neurons = sorted([str(n) for n in neuron_inf.keys() if n in spike_data])
        
        for neuron in unique_neurons:
            neuron_id = int(neuron)
            
            neuron_spikes = spike_data[neuron_id]  # 采样点为单位的时间
            
            all_trials = trigger_log.copy().reset_index(drop=True)
            n_trials = len(all_trials)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for trial_idx, (_, trial) in enumerate(all_trials.iterrows()):
                start_ext = int(trial['start_extended'])
                end_ext = int(trial['end_extended'])
                trial_spikes = neuron_spikes[(neuron_spikes >= start_ext) & (neuron_spikes <= end_ext)]
                if len(trial_spikes) > 0:
                    relative_spikes = (trial_spikes - start_ext) / SAMPLING_RATE
                    relative_spikes = np.asarray(relative_spikes).flatten()
                    ax.vlines(relative_spikes, trial_idx - 3, trial_idx + 3, colors='black', linewidths=1)
            
            # 标记刺激开始和结束时间
            stim_start = EXTEND_TIME
            stim_end = EXTEND_TIME + STIMULUS_DURATION
            ax.axvline(x=stim_start, color='red', linestyle='--', linewidth=1.5, label='Stimulus Start', alpha=0.7)
            ax.axvline(x=stim_end, color='red', linestyle='--', linewidth=1.5, label='Stimulus End', alpha=0.7)
            
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Trial', fontsize=12)
            ax.set_title(f'Raster Plot: {clique_name} - Neuron {neuron} (Total {n_trials} trials)', fontsize=14)
            ax.set_xlim(0, EXTEND_TIME + STIMULUS_DURATION + EXTEND_TIME)
            ax.set_ylim(-0.5, n_trials - 0.5)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

print(f"\n完成! Raster plots已保存到: {pdf_path}")
print("=" * 60)


