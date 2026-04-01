import numpy as np
import scipy.io as sio
import spikeinterface.extractors as se
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def process_multi_block_spike_sorting(monkey='monkeyF', date='20240112', 
                                    base_dir='/media/ubuntu/sda/Monkey/TVSD',
                                    output_base_dir='/media/ubuntu/sda/Monkey/sorted_result_combined',
                                    probe_file='/media/ubuntu/sda/Monkey/scripts/probe_256.json',
                                    mapping_file='/media/ubuntu/sda/Monkey/TVSD/monkeyF/_logs/1024chns_mapping_20220105.mat'):
    """
    处理多Block数据的spike sorting
    
    Args:
        monkey: 猴子名称 (monkeyF 或 monkeyN)
        date: 日期字符串 (如 '20240112')
        base_dir: 原始数据基础目录
        output_base_dir: 输出结果基础目录
        probe_file: probe配置文件路径
        mapping_file: 通道映射文件路径
    """
    
    # 设置路径
    data_dir = Path(base_dir) / monkey / date
    output_dir = Path(output_base_dir) / date
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载映射文件
    mapping_data = sio.loadmat(mapping_file)
    mapping = mapping_data['mapping'].flatten() - 1  # 转换为0-based索引
    
    # 定义脑区映射
    if monkey == 'monkeyN':
        rois = np.ones(1024)  # V1
        rois[512:768] = 2  # V4 (513-768)
        rois[768:1024] = 3  # IT (769-1024)
    else:
        rois = np.ones(1024)  # V1
        rois[512:832] = 3  # IT (513-832)
        rois[832:1024] = 2  # V4 (833-1024)
    
    # 加载probe配置
    with open(probe_file, 'r') as f:
        probe_config = json.load(f)
    
    # 获取所有Block目录
    block_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('Block_')])
    print(f"找到 {len(block_dirs)} 个Block: {[d.name for d in block_dirs]}")
    
    # 定义Hub-instance组合
    hub_instance_combinations = [
        ('Hub1', 'instance1'),
        ('Hub1', 'instance2'), 
        ('Hub2', 'instance1'),
        ('Hub2', 'instance2')
    ]
    
    # 为每个Hub-instance组合处理数据
    for hub_name, instance_name in hub_instance_combinations:
        print(f"\n处理 {hub_name}-{instance_name}")
        
        # 收集所有Block中该Hub-instance的文件
        recording_files = []
        for block_dir in block_dirs:
            block_num = block_dir.name.split('_')[1]
            file_pattern = f"{hub_name}-{instance_name}_B{block_num.zfill(3)}.ns6"
            file_path = block_dir / file_pattern
            
            if file_path.exists():
                recording_files.append(file_path)
                print(f"  找到文件: {file_path}")
            else:
                print(f"  警告: 未找到文件 {file_path}")
        
        if not recording_files:
            print(f"  跳过 {hub_name}-{instance_name}: 未找到任何文件")
            continue
        
        # 读取并合并所有Block的recording
        recordings = []
        for file_path in recording_files:
            print(f"  读取文件: {file_path}")
            recording = se.read_blackrock(file_path)
            
            # 处理多段数据
            if recording.get_num_segments() > 1:
                recording_list = []
                for i in range(recording.get_num_segments()):
                    recording_list.append(recording.select_segments(i))
                recording = si.concatenate_recordings(recording_list)
            
            recordings.append(recording)
        
        # 合并所有recording
        print(f"  合并 {len(recordings)} 个recording...")
        combined_recording = si.concatenate_recordings(recordings)
        
        # 确定当前Hub-instance的脑区
        if hub_name == 'Hub1' and instance_name == 'instance1':
            file_start_idx = 0
        elif hub_name == 'Hub2' and instance_name == 'instance1':
            file_start_idx = 256
        elif hub_name == 'Hub1' and instance_name == 'instance2':
            file_start_idx = 512
        elif hub_name == 'Hub2' and instance_name == 'instance2':
            file_start_idx = 768
        
        # 确定主要脑区
        roi_counts = np.bincount(rois[file_start_idx:file_start_idx+256].astype(int))
        primary_roi = np.argmax(roi_counts)
        
        if primary_roi == 1:
            region_name = 'V1'
        elif primary_roi == 2:
            region_name = 'V4'
        else:
            region_name = 'IT'
        
        print(f"  主要脑区: {region_name}")
        
        # 设置输出目录
        hub_instance_output_dir = output_dir / f"{hub_name}-{instance_name}_{region_name}"
        hub_instance_output_dir.mkdir(exist_ok=True)
        
        # 保存合并后的recording
        recording_save_path = hub_instance_output_dir / "recording"
        print(f"  保存recording到: {recording_save_path}")
        combined_recording.save(folder=recording_save_path)
        
        # 加载probe并设置到recording
        probe = si.read_probeinterface(probe_file)
        combined_recording = combined_recording.set_probe(probe)
        
        # 进行spike sorting
        print(f"  开始spike sorting...")
        sorting_output_dir = hub_instance_output_dir / "sorting"
        
        # 使用MountainSort4进行排序
        sorting = ss.run_mountainsort4(
            recording=combined_recording,
            output_folder=sorting_output_dir,
            verbose=True
        )
        
        # 保存sorting结果
        sorting_save_path = hub_instance_output_dir / "sorting_result"
        print(f"  保存sorting结果到: {sorting_save_path}")
        sorting.save(folder=sorting_save_path)
        
        # 计算质量指标
        print(f"  计算质量指标...")
        waveforms = spost.extract_waveforms(
            recording=combined_recording,
            sorting=sorting,
            folder=hub_instance_output_dir / "waveforms",
            sparse=True,
            max_spikes_per_unit=300
        )
        
        # 计算质量指标
        quality_metrics = sqm.compute_quality_metrics(waveforms)
        
        # 保存质量指标
        quality_metrics.to_csv(hub_instance_output_dir / "quality_metrics.csv")
        
        print(f"  完成 {hub_name}-{instance_name}_{region_name}")
        
        # 清理内存
        del combined_recording, sorting, waveforms, quality_metrics
    
    print(f"\n所有Hub-instance组合处理完成！")
    print(f"结果保存在: {output_dir}")

def main():
    """主函数"""
    # 可以处理多个日期
    dates_to_process = ['20240112', '20240115']
    
    for date in dates_to_process:
        print(f"\n{'='*50}")
        print(f"处理日期: {date}")
        print(f"{'='*50}")
        
        try:
            process_multi_block_spike_sorting(
                monkey='monkeyF',
                date=date,
                base_dir='/media/ubuntu/sda/Monkey/TVSD',
                output_base_dir='/media/ubuntu/sda/Monkey/sorted_result_combined',
                probe_file='/media/ubuntu/sda/Monkey/scripts/probe_256.json',
                mapping_file='/media/ubuntu/sda/Monkey/TVSD/monkeyF/_logs/1024chns_mapping_20220105.mat'
            )
        except Exception as e:
            print(f"处理日期 {date} 时出错: {e}")
            continue

if __name__ == "__main__":
    main()


