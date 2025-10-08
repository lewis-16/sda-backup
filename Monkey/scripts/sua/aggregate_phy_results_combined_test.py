import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import argparse
import json

ROOT_SORT_DIR = "/media/ubuntu/sda/Monkey/sorted_result_combined"
DATE_STR = "20240112"

# 需要的 cluster 指标文件（来自 phy_folder_for_kilosort）
CLUSTER_INFO_FILENAME = "cluster_info.tsv"

# spike 层面
SPIKE_CLUSTERS_FILENAME = "spike_clusters.npy"
SPIKE_TIMES_FILENAME = "spike_times.npy"

# 采样率 (Hz) - 需要根据实际情况调整
SAMPLING_RATE = 30000


def parse_hub_instance_meta(hub_instance_dir_name: str) -> Tuple[str, str, str]:
    """从文件夹名解析 hub, instance 与 region。
    形如: Hub1-instance1_V1 / Hub2-instance2_V4
    返回 (hub, instance, region)
    """
    # 匹配 HubX-instanceY_REGION
    m = re.match(r"(Hub\d+)-instance(\d+)_([A-Za-z0-9]+)$", hub_instance_dir_name)
    if not m:
        raise ValueError(f"无法从目录名解析 hub/instance/region: {hub_instance_dir_name}")
    hub = m.group(1)
    instance = m.group(2)
    region = m.group(3)
    return hub, instance, region


def get_block_durations_from_original_data(date_str: str) -> Dict[str, float]:
    """从原始数据文件获取每个block的时长（秒）
    返回字典: {'B001': duration1, 'B002': duration2, ...}
    """
    block_durations = {}
    
    # 原始数据目录
    original_data_dir = f"/media/ubuntu/sda/Monkey/{date_str}"
    
    if not os.path.exists(original_data_dir):
        print(f"警告: 原始数据目录不存在: {original_data_dir}")
        # 使用默认时长
        block_durations = {
            'B001': 300.0,  # 5分钟
            'B002': 300.0,  # 5分钟
            'B003': 300.0,  # 5分钟
            'B004': 300.0,  # 5分钟
        }
        print("使用默认block时长")
        return block_durations
    
    # 查找所有block目录
    for item in os.listdir(original_data_dir):
        if item.startswith('block'):
            block_dir = os.path.join(original_data_dir, item)
            if os.path.isdir(block_dir):
                # 从目录名提取block编号
                block_match = re.search(r'block(\d+)', item)
                if block_match:
                    block_num = int(block_match.group(1))
                    block_id = f"B{block_num:03d}"
                    
                    # 查找该block中的NS6文件来获取时长
                    ns6_files = glob.glob(os.path.join(block_dir, "*.ns6"))
                    if ns6_files:
                        # 使用第一个NS6文件来估算时长
                        # 这里需要根据实际文件格式来读取时长信息
                        # 暂时使用文件大小来估算（这是一个简化的方法）
                        file_size = os.path.getsize(ns6_files[0])
                        # 假设每个样本2字节，256通道
                        estimated_samples = file_size / (2 * 256)
                        estimated_duration = estimated_samples / SAMPLING_RATE
                        block_durations[block_id] = estimated_duration
                        print(f"估算 {block_id} 时长: {estimated_duration:.2f} 秒")
    
    # 如果没有找到任何block，使用默认值
    if not block_durations:
        block_durations = {
            'B001': 300.0,  # 5分钟
            'B002': 300.0,  # 5分钟
            'B003': 300.0,  # 5分钟
            'B004': 300.0,  # 5分钟
        }
        print("使用默认block时长")
    
    return block_durations


def load_cluster_info(phy_dir: str) -> pd.DataFrame:
    """读取 phy_folder_for_kilosort/cluster_info.tsv 为 DataFrame。
    该表包含所有需要的 cluster 级指标。
    如果cluster_info.tsv不存在，则尝试从其他文件构建基本信息。
    """
    path = os.path.join(phy_dir, CLUSTER_INFO_FILENAME)
    
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t')
        # 标准化主键列名
        if 'cluster_id' not in df.columns:
            raise ValueError(f"{path} 中缺少 cluster_id 列")
        return df
    else:
        print(f"警告: {path} 不存在，尝试从其他文件构建cluster信息")
        
        # 尝试从cluster_group.tsv构建基本信息
        cluster_group_path = os.path.join(phy_dir, "cluster_group.tsv")
        if os.path.exists(cluster_group_path):
            df = pd.read_csv(cluster_group_path, sep='\t')
            if 'cluster_id' not in df.columns:
                raise ValueError(f"{cluster_group_path} 中缺少 cluster_id 列")
            return df
        else:
            # 如果都没有，从spike_clusters.npy中提取唯一的cluster_id
            spike_clusters_path = os.path.join(phy_dir, SPIKE_CLUSTERS_FILENAME)
            if os.path.exists(spike_clusters_path):
                spike_clusters = np.load(spike_clusters_path)
                unique_clusters = np.unique(spike_clusters)
                df = pd.DataFrame({
                    'cluster_id': unique_clusters,
                    'group': 'unsorted'  # 默认分组
                })
                return df
            else:
                raise ValueError(f"无法找到任何cluster信息文件: {phy_dir}")


def load_spike_level(phy_dir: str) -> pd.DataFrame:
    """读取 spike 层面的 numpy 文件并返回 DataFrame: [cluster, time]
    time 使用原始采样点，不做单位转换。
    """
    spike_clusters = np.load(os.path.join(phy_dir, SPIKE_CLUSTERS_FILENAME))
    spike_times = np.load(os.path.join(phy_dir, SPIKE_TIMES_FILENAME))
    # 展平为一维
    spike_clusters = np.asarray(spike_clusters).reshape(-1)
    spike_times = np.asarray(spike_times).reshape(-1)
    if spike_clusters.shape[0] != spike_times.shape[0]:
        raise ValueError(f"spike_clusters 与 spike_times 行数不一致: {phy_dir}")
    df = pd.DataFrame({
        'cluster_id': spike_clusters.astype(int),
        'time': spike_times.astype(int),
    })
    return df


def assign_spikes_to_blocks(spike_df: pd.DataFrame, block_durations: Dict[str, float]) -> pd.DataFrame:
    """为spike分配block信息并重新计算时间
    
    Args:
        spike_df: 包含time列的spike DataFrame
        block_durations: block时长字典，键为'B001'格式，值为秒数
    
    Returns:
        添加了block列并重新计算time的DataFrame
    """
    print(f"  处理 {len(spike_df)} 个spikes...")
    
    # 按时间排序
    spike_df = spike_df.sort_values('time').reset_index(drop=True)
    
    # 计算每个block的采样点数
    block_samples = {}
    cumulative_samples = 0
    for block_id, duration in sorted(block_durations.items()):
        samples = int(duration * SAMPLING_RATE)
        block_samples[block_id] = (cumulative_samples, cumulative_samples + samples)
        cumulative_samples += samples
        print(f"    {block_id}: {duration:.1f}s ({samples} samples)")
    
    # 为每个spike分配block
    spike_df['block'] = None
    spike_df['time'] = spike_df['time'].astype(int)
    
    for block_id, (start_sample, end_sample) in block_samples.items():
        mask = (spike_df['time'] >= start_sample) & (spike_df['time'] < end_sample)
        n_spikes = mask.sum()
        if n_spikes > 0:
            spike_df.loc[mask, 'block'] = block_id
            # 重新计算该block内的时间（相对于block开始）
            spike_df.loc[mask, 'time'] = spike_df.loc[mask, 'time'] - start_sample
            print(f"    {block_id}: {n_spikes} spikes")
    
    # 检查是否有未分配的spike
    unassigned = spike_df['block'].isna().sum()
    if unassigned > 0:
        print(f"  警告: 有 {unassigned} 个spike未分配到任何block")
        # 将未分配的spike分配到最后一个block
        last_block = list(block_samples.keys())[-1]
        spike_df.loc[spike_df['block'].isna(), 'block'] = last_block
        print(f"  将未分配的spike分配到 {last_block}")
    
    return spike_df


def process_single_hub_instance(base_dir: str, date_str: str, hub_instance_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """处理单个Hub-instance的数据"""
    hub_instance_path = os.path.join(base_dir, hub_instance_dir)
    phy_dir = os.path.join(hub_instance_path, 'phy_folder_for_kilosort')
    
    if not os.path.isdir(phy_dir):
        print(f"跳过 {hub_instance_dir}: 无 phy_folder_for_kilosort 目录")
        return None, None
    
    try:
        hub, instance, region = parse_hub_instance_meta(hub_instance_dir)
    except Exception as e:
        print(f"跳过 {hub_instance_dir}: {e}")
        return None, None
    
    print(f"处理 {hub_instance_dir} ({hub}-{instance}-{region})")
    
    # 获取block时长信息
    block_durations = get_block_durations_from_original_data(date_str)
    
    # cluster 信息
    try:
        cluster_df = load_cluster_info(phy_dir)
    except Exception as e:
        print(f"读取 cluster_info 失败 {phy_dir}: {e}")
        return None, None
    
    # 增加元数据列
    cluster_df['hub'] = hub
    cluster_df['instance'] = instance
    cluster_df['region'] = region
    cluster_df['date'] = date_str
    # 删除block_id列（如果存在）
    if 'block_id' in cluster_df.columns:
        cluster_df = cluster_df.drop(columns=['block_id'])
    
    # spike 信息
    try:
        spike_df = load_spike_level(phy_dir)
    except Exception as e:
        print(f"读取 spike 层面失败 {phy_dir}: {e}")
        return None, None
    
    # 为spike分配block信息并重新计算时间
    spike_df = assign_spikes_to_blocks(spike_df, block_durations)
    
    # 增加元数据列
    spike_df['hub'] = hub
    spike_df['instance'] = instance
    spike_df['region'] = region
    spike_df['date'] = date_str
    # 删除block_id列（如果存在）
    if 'block_id' in spike_df.columns:
        spike_df = spike_df.drop(columns=['block_id'])
    
    return cluster_df, spike_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate phy_folder_for_kilosort results from combined sorting into cluster_inf and spike_inf CSVs.")
    parser.add_argument("--date", dest="date_str", type=str, default=DATE_STR, help="Date string, eg 20240112")
    parser.add_argument("--root", dest="root_dir", type=str, default=ROOT_SORT_DIR, help="sorted_result_combined")
    parser.add_argument("--base_dir", dest="base_dir", type=str, default=None, help="root/date")
    parser.add_argument("--sampling_rate", dest="sampling_rate", type=int, default=SAMPLING_RATE, help="Sampling rate in Hz")
    parser.add_argument("--hub_instance", dest="hub_instance", type=str, default=None, help="Specific hub-instance to process (e.g., Hub1-instance1_V1)")
    args = parser.parse_args()
    
    # 更新全局采样率
    global SAMPLING_RATE
    SAMPLING_RATE = args.sampling_rate
    
    if args.base_dir is not None:
        base_sort_dir = args.base_dir
    else:
        base_sort_dir = os.path.join(args.root_dir, args.date_str)
    
    if not os.path.exists(base_sort_dir):
        raise FileNotFoundError(f"基础目录不存在: {base_sort_dir}")
    
    print(f"处理目录: {base_sort_dir}")
    print(f"采样率: {SAMPLING_RATE} Hz")
    
    # 查找所有Hub-instance目录
    hub_instance_dirs = sorted([d for d in os.listdir(base_sort_dir) 
                               if os.path.isdir(os.path.join(base_sort_dir, d)) 
                               and d.startswith('Hub')])
    
    if args.hub_instance:
        if args.hub_instance in hub_instance_dirs:
            hub_instance_dirs = [args.hub_instance]
        else:
            print(f"指定的hub-instance {args.hub_instance} 不存在")
            return
    
    cluster_parts: List[pd.DataFrame] = []
    spike_parts: List[pd.DataFrame] = []
    
    for hub_instance_dir in hub_instance_dirs:
        cluster_df, spike_df = process_single_hub_instance(base_sort_dir, args.date_str, hub_instance_dir)
        if cluster_df is not None and spike_df is not None:
            cluster_parts.append(cluster_df)
            spike_parts.append(spike_df)
    
    if len(cluster_parts) == 0:
        print("未聚合到任何 cluster 信息")
        return
    if len(spike_parts) == 0:
        print("未聚合到任何 spike 信息")
        return
    
    cluster_all = pd.concat(cluster_parts, ignore_index=True)
    spike_all = pd.concat(spike_parts, ignore_index=True)
    
    # 不需要 probe_group 信息，确保若存在该列也不保留
    for col in ['probe_group', 'probe_group_id']:
        if col in cluster_all.columns:
            cluster_all = cluster_all.drop(columns=[col])
        if col in spike_all.columns:
            spike_all = spike_all.drop(columns=[col])
    
    # 保存到 base_dir 下
    cluster_out = os.path.join(base_sort_dir, f"cluster_inf_{args.date_str}_combined.csv")
    spike_out = os.path.join(base_sort_dir, f"spike_inf_{args.date_str}_combined.csv")
    
    cluster_all.to_csv(cluster_out, index=False)
    spike_all.to_csv(spike_out, index=False)
    
    print(f"写出: {cluster_out}")
    print(f"写出: {spike_out}")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"Cluster总数: {len(cluster_all)}")
    print(f"Spike总数: {len(spike_all)}")
    print(f"Hub-instance组合数: {cluster_all[['hub', 'instance']].drop_duplicates().shape[0]}")
    print(f"Block分布:")
    if 'block' in spike_all.columns:
        block_counts = spike_all['block'].value_counts().sort_index()
        for block, count in block_counts.items():
            print(f"  {block}: {count} spikes")


if __name__ == "__main__":
    main()
