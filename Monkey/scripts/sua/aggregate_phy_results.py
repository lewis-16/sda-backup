import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple
import argparse

ROOT_SORT_DIR = "/media/ubuntu/sda/Monkey/sorted_result"
DATE_STR = "20240112"
BLOCK_ID = 2

# 需要的 cluster 指标文件（来自 phy_folder_for_kilosort）
# 这些字段基本与 cluster_info.tsv 中一致，但也单独提供，稳健起见使用 cluster_info.tsv 为主。
CLUSTER_INFO_FILENAME = "cluster_info.tsv"

# spike 层面
SPIKE_CLUSTERS_FILENAME = "spike_clusters.npy"
SPIKE_TIMES_FILENAME = "spike_times.npy"


def parse_array_meta(array_dir_name: str) -> Tuple[str, str]:
    """从文件夹名解析 array_id 与 region。
    形如: array_01_V1.npy / array_14_V4.npy / array_11_IT.npy
    返回 (array_id, region)
    """
    # 去掉结尾的 .npy
    base = array_dir_name
    if base.endswith('.npy'):
        base = base[:-4]
    # 匹配 array_XX_REGION
    m = re.match(r"array_(\d{2})_([A-Za-z0-9]+)$", base)
    if not m:
        raise ValueError(f"无法从目录名解析 array_id/region: {array_dir_name}")
    array_id = m.group(1)
    region = m.group(2)
    return array_id, region


def load_cluster_info(phy_dir: str) -> pd.DataFrame:
    """读取 phy_folder_for_kilosort/cluster_info.tsv 为 DataFrame。
    该表包含所有需要的 cluster 级指标。
    """
    path = os.path.join(phy_dir, CLUSTER_INFO_FILENAME)
    df = pd.read_csv(path, sep='\t')
    # 标准化主键列名
    if 'cluster_id' not in df.columns:
        raise ValueError(f"{path} 中缺少 cluster_id 列")
    return df


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


def aggregate(base_dir: str, date_str: str, block_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """遍历 base_dir 下所有 array_* 目录，聚合 cluster 与 spike 两类信息。
    返回 (cluster_inf_all, spike_inf_all)
    """
    cluster_parts: List[pd.DataFrame] = []
    spike_parts: List[pd.DataFrame] = []

    array_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('array_')])

    for array_dir in array_dirs:
        array_path = os.path.join(base_dir, array_dir)
        phy_dir = os.path.join(array_path, 'phy_folder_for_kilosort')
        if not os.path.isdir(phy_dir):
            # 跳过无 phy 目录者
            continue
        try:
            array_id, region = parse_array_meta(array_dir)
        except Exception as e:
            print(f"跳过 {array_dir}: {e}")
            continue

        # cluster 信息
        try:
            cluster_df = load_cluster_info(phy_dir)
        except Exception as e:
            print(f"读取 cluster_info 失败 {phy_dir}: {e}")
            continue
        # 增加元数据列
        cluster_df['array_id'] = array_id
        cluster_df['region'] = region
        cluster_df['date'] = date_str
        cluster_df['block_id'] = block_id
        cluster_parts.append(cluster_df)

        # spike 信息
        try:
            spike_df = load_spike_level(phy_dir)
        except Exception as e:
            print(f"读取 spike 层面失败 {phy_dir}: {e}")
            continue
        # 增加元数据列
        spike_df['array_id'] = array_id
        spike_df['region'] = region
        spike_df['date'] = date_str
        spike_df['block_id'] = block_id
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

    return cluster_all, spike_all


def build_base_dir(root_dir: str, date_str: str, block_id: int) -> str:
    return os.path.join(root_dir, date_str, f"Block_{block_id}", "sort")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate phy_folder_for_kilosort results into cluster_inf and spike_inf CSVs.")
    parser.add_argument("--date", dest="date_str", type=str, default=DATE_STR, help="Date string, eg 20240112")
    parser.add_argument("--block", dest="block_id", type=int, default=BLOCK_ID, help="Block id, eg 1")
    parser.add_argument("--root", dest="root_dir", type=str, default=ROOT_SORT_DIR, help="sorted_result")
    parser.add_argument("--base_dir", dest="base_dir", type=str, default=None, help="root/date/block")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.base_dir is not None:
        base_sort_dir = args.base_dir
    else:
        base_sort_dir = build_base_dir(args.root_dir, args.date_str, args.block_id)

    cluster_all, spike_all = aggregate(base_sort_dir, args.date_str, args.block_id)

    # 保存到 base_dir 下
    cluster_out = os.path.join(base_sort_dir, f"cluster_inf_{args.date_str}_B{args.block_id}.csv")
    spike_out = os.path.join(base_sort_dir, f"spike_inf_{args.date_str}_B{args.block_id}.csv")

    cluster_all.to_csv(cluster_out, index=False)
    spike_all.to_csv(spike_out, index=False)

    print(f"写出: {cluster_out}")
    print(f"写出: {spike_out}")


if __name__ == "__main__":
    main()
