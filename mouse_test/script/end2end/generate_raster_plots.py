import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SAMPLING_RATE = 10000
EXTEND_TIME = 0.25
STIMULUS_DURATION = 1.0

clique_0_dir = '/media/ubuntu/sda/mouse_test/sorted/recordings_30_channel_12_months_mouse6_natim_full/clique_0'
train_session_name = 'mouse6_021322_natural_image_001'
trigger_time_path = '/media/ubuntu/sda/data/mouse6/output/01_get_trigger/trigger_time.tsv'

output_dir = '/media/ubuntu/sda/mouse_test/sorted/recordings_30_channel_12_months_mouse6_natim_full/raster_plots'
os.makedirs(output_dir, exist_ok=True)

trigger_df = pd.read_csv(trigger_time_path, sep='\t')
trigger_df['date'] = trigger_df['date'].astype(str).str.zfill(6)

train_neuron_inf_path = f'{clique_0_dir}/{train_session_name}/neuron_inf.pickle'
with open(train_neuron_inf_path, 'rb') as f:
    train_neuron_inf_dict = pickle.load(f)

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_clique import neuron_inf_dict_to_dataframe

train_neuron_inf = neuron_inf_dict_to_dataframe(train_neuron_inf_dict)
train_neuron_ids_standard = sorted(train_neuron_inf['Neuron'].unique().tolist())

session_dirs = [d for d in os.listdir(clique_0_dir) 
                if os.path.isdir(os.path.join(clique_0_dir, d)) 
                and d.startswith('mouse6_')]

session_dirs.sort()

print(f"找到 {len(session_dirs)} 个session目录")
print(f"训练session: {train_session_name}")
print(f"标准neuron IDs: {train_neuron_ids_standard}")

for sess_idx, sess_name in enumerate(session_dirs):
    date_str = sess_name.split('_')[1]
    print(f"\n{'='*60}")
    print(f"处理session {sess_idx+1}/{len(session_dirs)}: {sess_name} (date={date_str})")
    print(f"{'='*60}")
    
    session_out_dir = os.path.join(output_dir, sess_name)
    os.makedirs(session_out_dir, exist_ok=True)
    
    raster_pdf_path = os.path.join(session_out_dir, f'raster_plot_{sess_name}.pdf')
    
    if os.path.exists(raster_pdf_path):
        print(f"Raster plot已存在，跳过: {raster_pdf_path}")
        continue
    
    gt_detect_array_path = f'{clique_0_dir}/{sess_name}/gt_detect_array.csv'
    if not os.path.exists(gt_detect_array_path):
        print(f"警告: gt_detect_array文件不存在: {gt_detect_array_path}，跳过")
        continue
    
    print(f"加载gt_detect_array: {gt_detect_array_path}")
    gt_detect_array = pd.read_csv(gt_detect_array_path)
    print(f"  加载了 {len(gt_detect_array)} 条spike记录")
    print(f"  列名: {gt_detect_array.columns.tolist()}")
    print(f"  时间范围: {gt_detect_array['time'].min()} 到 {gt_detect_array['time'].max()} (sample points)")
    print(f"  Neuron IDs: {sorted(gt_detect_array['unit_id'].unique().tolist())}")
    
    session_trig = trigger_df[trigger_df['date'] == date_str].copy().reset_index(drop=False)
    print(f'本月trials数: {len(session_trig)}')
    
    if len(session_trig) == 0:
        print(f"警告: 没有找到该session的trigger数据，跳过")
        continue
    
    extend_samples = int(EXTEND_TIME * SAMPLING_RATE)
    stimulus_samples = int(STIMULUS_DURATION * SAMPLING_RATE)
    
    spikes_by_trial = {}
    for trial_i, row in session_trig.iterrows():
        trial_id = int(row['index']) if 'index' in row else int(trial_i)
        trial_start = int(row['start'])
        
        trial_window_start = trial_start - extend_samples
        trial_window_end = trial_start + stimulus_samples + extend_samples
        
        mask = (gt_detect_array['time'] >= trial_window_start) & (gt_detect_array['time'] <= trial_window_end)
        trial_spikes = gt_detect_array[mask].copy()
        
        if len(trial_spikes) > 0:
            trial_spikes['time_relative'] = trial_spikes['time'] - trial_start
            
            trial_key = f'trial_{trial_id}'
            spikes_by_trial[trial_key] = {}
            for _, spike_row in trial_spikes.iterrows():
                neu = int(spike_row['unit_id'])
                spike_time = int(spike_row['time_relative'])
                spikes_by_trial[trial_key].setdefault(neu, []).append(spike_time)
    
    print(f'有效trials: {len(spikes_by_trial)}')
    
    all_neuron_ids = train_neuron_ids_standard
    
    all_detected_neurons = set()
    for trial_key, trial_spikes in spikes_by_trial.items():
        all_detected_neurons.update(trial_spikes.keys())
    
    print(f'训练session标准neuron数: {len(all_neuron_ids)}')
    print(f'该session实际检测到spike的neuron数: {len(all_detected_neurons)}')
    print(f'  实际检测到的neuron IDs: {sorted(all_detected_neurons)}')
    print(f'Raster plot将绘制 {len(all_neuron_ids)} 个neuron（按训练session顺序）')
    
    time_start = -int(EXTEND_TIME * SAMPLING_RATE)
    time_end = int((STIMULUS_DURATION + EXTEND_TIME) * SAMPLING_RATE)
    
    image_to_trials = {}
    for trial_i, row in session_trig.iterrows():
        trial_id = int(row['index']) if 'index' in row else int(trial_i)
        trial_key = f'trial_{trial_id}'
        img_id = int(row['image'])
        if img_id not in image_to_trials:
            image_to_trials[img_id] = []
        image_to_trials[img_id].append((trial_i, trial_key))
    
    print(f'\n开始绘制raster plot: {raster_pdf_path}')
    print(f'时间范围: {time_start} 到 {time_end} (sample points, 相对于stimulus onset)')
    
    with PdfPages(raster_pdf_path) as pdf:
        for nj, neu in enumerate(all_neuron_ids):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            y_pos = 0
            for img_id in sorted(image_to_trials.keys()):
                trials_for_img = image_to_trials[img_id]
                
                for trial_i, trial_key in trials_for_img:
                    if trial_key in spikes_by_trial and neu in spikes_by_trial[trial_key]:
                        spike_times = spikes_by_trial[trial_key][neu]
                        if len(spike_times) > 0:
                            spike_times_array = np.array(spike_times)
                            valid_spikes = spike_times_array[(spike_times_array >= time_start) & (spike_times_array <= time_end)]
                            if len(valid_spikes) > 0:
                                ax.vlines(valid_spikes, y_pos - 0.4, y_pos + 0.4, colors='black', linewidths=0.5)
                    y_pos += 1
            
            stimulus_offset_samples = int(STIMULUS_DURATION * SAMPLING_RATE)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
            ax.axvline(x=stimulus_offset_samples, color='blue', linestyle='--', linewidth=2, label='Stimulus Offset')
            
            ax.set_xlim(time_start, time_end)
            ax.set_ylim(-1, y_pos)
            ax.set_xlabel('Time (samples, relative to stimulus onset)', fontsize=12)
            ax.set_ylabel('Trial', fontsize=12)
            ax.set_title(f'Session: {sess_name}\nNeuron: {neu}\nTotal Trials: {len(session_trig)}', fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f'Raster plot已保存: {raster_pdf_path} ({len(all_neuron_ids)} 页)')

print(f"\n{'='*60}")
print(f"所有session处理完成！Raster plots已保存到: {output_dir}")
print(f"{'='*60}")
