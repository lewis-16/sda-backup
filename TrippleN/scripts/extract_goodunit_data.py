"""
提取 GoodUnit 数据的 Python 脚本
遍历 /media/ubuntu/sda/TrippleN/GoodUnit 下的所有 .mat 文件
提取 waveform, spikepos 并保存为 .npy 格式
"""

import numpy as np
import h5py
import os
from datetime import datetime

goodunit_folder = '/media/ubuntu/sda/TrippleN/GoodUnit'
output_folder = '/media/ubuntu/sda/TrippleN/customize/GoodUnitStr'

os.makedirs(output_folder, exist_ok=True)

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'[{timestamp}] {msg}')

mat_files = sorted([f for f in os.listdir(goodunit_folder) if f.endswith('.mat')])
total_files = len(mat_files)
log(f'找到 {total_files} 个 .mat 文件')

success_count = 0
fail_count = 0

for idx, filename in enumerate(mat_files, 1):
    filepath = os.path.join(goodunit_folder, filename)
    
    try:
        with h5py.File(filepath, 'r') as f:
            goodunit = f['GoodUnitStrc']
            
            n_neurons = goodunit['waveform'].shape[0]
            
            # 获取第一个神经元的数据形状
            first_wf = np.array(f[goodunit['waveform'][0, 0]])
            wf_shape = first_wf.shape  # (61, 383)
            
            first_sp = np.array(f[goodunit['spikepos'][0, 0]])
            
            # 初始化数组
            waveform_data = np.zeros((n_neurons, wf_shape[0], wf_shape[1]), dtype=np.float32)
            spikepos_data = np.zeros((n_neurons, first_sp.shape[0]), dtype=np.float32)
            
            # 转换为 numpy 数组以便索引
            waveform_refs = np.array(goodunit['waveform'])
            spikepos_refs = np.array(goodunit['spikepos'])
            
            for j in range(n_neurons):
                # 提取 waveform (61, 383)
                wf = np.array(f[waveform_refs[j, 0]])
                waveform_data[j] = wf
                
                # 提取 spikepos
                sp = np.array(f[spikepos_refs[j, 0]])
                spikepos_data[j] = sp.flatten()
            
            # 保存为 npy
            name = filename.replace('.mat', '')
            
            np.save(os.path.join(output_folder, f'{name}_waveform.npy'), waveform_data)
            np.save(os.path.join(output_folder, f'{name}_spikepos.npy'), spikepos_data)
            
            log(f'[{idx}/{total_files}] {filename}: {n_neurons} neurons')
            log(f'  - waveform: {waveform_data.shape}')
            log(f'  - spikepos: {spikepos_data.shape}')
            
            success_count += 1
            
    except Exception as e:
        log(f'[{idx}/{total_files}] {filename}: 错误 - {e}')
        fail_count += 1

log(f'\n处理完成！')
log(f'成功: {success_count}/{total_files}')
log(f'失败: {fail_count}/{total_files}')
