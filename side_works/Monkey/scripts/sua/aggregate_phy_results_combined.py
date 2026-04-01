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


def get_block_durations_from_original_data(date_str: str, hub: str, instance: str) -> Dict[str, int]:
    """从原始数据文件获取每个block的时长（采样点数）
    返回字典: {'Block_1': samples1, 'Block_2': samples2, ...}
    """
    import spikeinterface.extractors as se
    
    block_durations = {}
    
    # 原始数据目录
    original_data_dir = f"/media/ubuntu/sda/Monkey/TVSD/monkeyF/{date_str}"
    
    if not os.path.exists(original_data_dir):
        print(f"警告: 原始数据目录不存在: {original_data_dir}")
        return block_durations
    
    # 查找所有Block目录
    for block_dir in os.listdir(original_data_dir):
        if block_dir.startswith('Block_'):
            block_path = os.path.join(original_data_dir, block_dir)
            if os.path.isdir(block_path):
                # 构建NS6文件路径
                block_num = block_dir.split('_')[1]
                ns6_file = os.path.join(block_path, f"{hub}-{instance}_B00{block_num}.ns6")
                
                if os.path.exists(ns6_file):
                    try:
                        # 使用spikeinterface读取文件获取实际时长
                        record_temp = se.read_blackrock(ns6_file)
                        duration_samples = int(record_temp.get_total_duration() * SAMPLING_RATE)
                        block_durations[block_dir] = duration_samples
                        print(f"  {block_dir}: {duration_samples} samples ({duration_samples/SAMPLING_RATE:.2f}s)")
                    except Exception as e:
                        print(f"  读取 {ns6_file} 失败: {e}")
                else:
                    print(f"  文件不存在: {ns6_file}")
    
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


def assign_spikes_to_blocks(spike_df: pd.DataFrame, block_durations: Dict[str, int]) -> pd.DataFrame:
    """为spike分配block信息并重新计算时间
    
    Args:
        spike_df: 包含time列的spike DataFrame
        block_durations: block时长字典，键为'Block_1'格式，值为采样点数
    
    Returns:
        添加了block列并重新计算time的DataFrame
    """
    print(f"  处理 {len(spike_df)} 个spikes...")
    
    # 按时间排序
    spike_df = spike_df.sort_values('time').reset_index(drop=True)
    
    # 初始化block列
    spike_df['block'] = None
    spike_df['time'] = spike_df['time'].astype(int)
    
    # 按照notebook中的逻辑分配spike到block
    sum_time = 0
    for i in [1, 2, 3, 4]:
        block_key = f'Block_{i}'
        if block_key in block_durations:
            block_samples = block_durations[block_key]
            
            if i == 1:
                # 第一个block：time < block_samples
                mask = spike_df['time'] < block_samples
                spike_df.loc[mask, 'block'] = i
                sum_time += block_samples
                print(f"    Block_{i}: {mask.sum()} spikes (0 - {block_samples})")
            else:
                # 后续block：sum_time <= time < sum_time + block_samples
                mask = (spike_df['time'] < sum_time + block_samples) & (spike_df['time'] >= sum_time)
                spike_df.loc[mask, 'block'] = i
                # 重新计算该block内的时间（相对于block开始）
                spike_df.loc[mask, 'time'] = spike_df.loc[mask, 'time'] - sum_time
                sum_time += block_samples
                print(f"    Block_{i}: {mask.sum()} spikes ({sum_time - block_samples} - {sum_time})")
    
    # 检查是否有未分配的spike
    unassigned = spike_df['block'].isna().sum()
    if unassigned > 0:
        print(f"  警告: 有 {unassigned} 个spike未分配到任何block")
        # 将未分配的spike分配到最后一个block
        last_block = max([int(k.split('_')[1]) for k in block_durations.keys()])
        spike_df.loc[spike_df['block'].isna(), 'block'] = last_block
        print(f"  将未分配的spike分配到 Block_{last_block}")
    
    return spike_df


def aggregate_combined(base_dir: str, date_str: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """遍历 base_dir 下所有 Hub*-instance* 目录，聚合 cluster 与 spike 两类信息。
    返回 (cluster_inf_all, spike_inf_all)
    """
    cluster_parts: List[pd.DataFrame] = []
    spike_parts: List[pd.DataFrame] = []
    
    # 获取block时长信息
    block_durations = get_block_durations_from_original_data(date_str)
    if not block_durations:
        print("警告: 无法获取block时长信息，将使用默认值")
        # 使用默认时长（需要根据实际情况调整）
        block_durations = {
            'B001': 300.0,  # 5分钟
            'B002': 300.0,  # 5分钟
            'B003': 300.0,  # 5分钟
            'B004': 300.0,  # 5分钟
        }
    
    # 查找所有Hub-instance目录
    hub_instance_dirs = sorted([d for d in os.listdir(base_dir) 
                               if os.path.isdir(os.path.join(base_dir, d)) 
                               and d.startswith('Hub')])
    
    for hub_instance_dir in hub_instance_dirs:
        hub_instance_path = os.path.join(base_dir, hub_instance_dir)
        phy_dir = os.path.join(hub_instance_path, 'phy_folder_for_kilosort')
        
        if not os.path.isdir(phy_dir):
            print(f"跳过 {hub_instance_dir}: 无 phy_folder_for_kilosort 目录")
            continue
        
        try:
            hub, instance, region = parse_hub_instance_meta(hub_instance_dir)
        except Exception as e:
            print(f"跳过 {hub_instance_dir}: {e}")
            continue
        
        print(f"处理 {hub_instance_dir} ({hub}-{instance}-{region})")
        
        # cluster 信息
        try:
            cluster_df = load_cluster_info(phy_dir)
        except Exception as e:
            print(f"读取 cluster_info 失败 {phy_dir}: {e}")
            continue
        
        # 增加元数据列
        cluster_df['hub'] = hub
        cluster_df['instance'] = instance
        cluster_df['region'] = region
        cluster_df['date'] = date_str
        # 删除block_id列（如果存在）
        if 'block_id' in cluster_df.columns:
            cluster_df = cluster_df.drop(columns=['block_id'])
        cluster_parts.append(cluster_df)
        
        # spike 信息
        try:
            spike_df = load_spike_level(phy_dir)
        except Exception as e:
            print(f"读取 spike 层面失败 {phy_dir}: {e}")
            continue
        
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
        
        spike_parts.append(spike_df)
    
    if len(cluster_parts) == 0:
        raise RuntimeError("未聚合到任何 cluster 信息")
    if len(spike_parts) == 0:
        raise RuntimeError("未聚合到任何 spike 信息")
    
    cluster_all = pd.concat(cluster_parts, ignore_index=True)
    spike_all = pd.concat(spike_parts, ignore_index=True)
    
    # 不需要 probe_group 信息，确保若存在该列也不保留
    for col in ['probe_group', 'probe_group_id']:
        if col in cluster_all.columns:
            cluster_all = cluster_all.drop(columns=[col])
        if col in spike_all.columns:
            spike_all = spike_all.drop(columns=[col])
    
    return cluster_all, spike_all


def build_base_dir(root_dir: str, date_str: str) -> str:
    return os.path.join(root_dir, date_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate phy_folder_for_kilosort results from combined sorting into cluster_inf and spike_inf CSVs.")
    parser.add_argument("--date", dest="date_str", type=str, default=DATE_STR, help="Date string, eg 20240112")
    parser.add_argument("--root", dest="root_dir", type=str, default=ROOT_SORT_DIR, help="sorted_result_combined")
    parser.add_argument("--base_dir", dest="base_dir", type=str, default=None, help="root/date")
    parser.add_argument("--sampling_rate", dest="sampling_rate", type=int, default=SAMPLING_RATE, help="Sampling rate in Hz")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 更新全局采样率
    global SAMPLING_RATE
    SAMPLING_RATE = args.sampling_rate
    
    if args.base_dir is not None:
        base_sort_dir = args.base_dir
    else:
        base_sort_dir = build_base_dir(args.root_dir, args.date_str)
    
    if not os.path.exists(base_sort_dir):
        raise FileNotFoundError(f"基础目录不存在: {base_sort_dir}")
    
    print(f"处理目录: {base_sort_dir}")
    print(f"采样率: {SAMPLING_RATE} Hz")
    
    cluster_all, spike_all = aggregate_combined(base_sort_dir, args.date_str)
    
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
