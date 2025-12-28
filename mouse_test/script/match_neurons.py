#!/usr/bin/env python3
"""
匹配两个记录的neuron_inf，根据位置和波形进行配对。

配对条件：
- 位置距离 < 10
- 波形pearson相关系数 > 0.95
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def load_neuron_inf(pkl_path: Path) -> pd.DataFrame:
    """加载neuron_inf.pkl文件"""
    with open(pkl_path, "rb") as f:
        neuron_inf = pickle.load(f)
    return neuron_inf


def compute_position_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """计算两个位置之间的欧氏距离"""
    return float(np.linalg.norm(pos1 - pos2))


def compute_waveform_correlation(waveform1: np.ndarray, waveform2: np.ndarray) -> float:
    """计算两个波形之间的pearson相关系数"""
    # 确保是numpy数组
    waveform1 = np.asarray(waveform1, dtype=np.float32)
    waveform2 = np.asarray(waveform2, dtype=np.float32)
    
    # 取最小长度
    min_len = min(len(waveform1), len(waveform2))
    if min_len == 0:
        return 0.0
    
    # 计算相关系数
    corr, _ = pearsonr(waveform1[:min_len], waveform2[:min_len])
    
    # 如果出现NaN，返回0
    if np.isnan(corr):
        return 0.0
    
    return float(corr)


def match_neurons(
    neuron_inf1: pd.DataFrame,
    neuron_inf2: pd.DataFrame,
    position_threshold: float = 10.0,
    waveform_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    匹配两个neuron_inf中的neuron
    
    Args:
        neuron_inf1: 第一个记录的neuron信息
        neuron_inf2: 第二个记录的neuron信息
        position_threshold: 位置距离阈值（默认10.0）
        waveform_threshold: 波形相关系数阈值（默认0.95）
    
    Returns:
        包含配对结果的DataFrame，列包括：
        - neuron1: 第一个记录中的neuron名称
        - neuron2: 第二个记录中的neuron名称
        - position_distance: 位置距离
        - waveform_correlation: 波形相关系数
        - position_1_1, position_2_1: 第一个neuron的位置
        - position_1_2, position_2_2: 第二个neuron的位置
    """
    matches = []
    
    for idx1, row1 in neuron_inf1.iterrows():
        neuron1 = row1['Neuron']
        pos1 = np.array([row1['position_1'], row1['position_2']], dtype=float)
        waveform1 = row1['position_waveform']
        
        # 检查位置和波形是否有效
        if pd.isna(pos1[0]) or pd.isna(pos1[1]) or waveform1 is None:
            continue
        
        waveform1 = np.asarray(waveform1, dtype=np.float32)
        if len(waveform1) == 0:
            continue
        
        for idx2, row2 in neuron_inf2.iterrows():
            neuron2 = row2['Neuron']
            pos2 = np.array([row2['position_1'], row2['position_2']], dtype=float)
            waveform2 = row2['position_waveform']
            
            # 检查位置和波形是否有效
            if pd.isna(pos2[0]) or pd.isna(pos2[1]) or waveform2 is None:
                continue
            
            waveform2 = np.asarray(waveform2, dtype=np.float32)
            if len(waveform2) == 0:
                continue
            
            # 计算位置距离
            position_distance = compute_position_distance(pos1, pos2)
            
            # 如果位置距离超过阈值，跳过
            if position_distance >= position_threshold:
                continue
            
            # 计算波形相关系数
            waveform_correlation = compute_waveform_correlation(waveform1, waveform2)
            
            # 如果相关系数低于阈值，跳过
            if waveform_correlation <= waveform_threshold:
                continue
            
            # 找到匹配
            match = {
                'neuron1': neuron1,
                'neuron2': neuron2,
                'position_distance': position_distance,
                'waveform_correlation': waveform_correlation,
                'position_1_1': pos1[0],
                'position_2_1': pos1[1],
                'position_1_2': pos2[0],
                'position_2_2': pos2[1],
            }
            
            # 如果neuron_inf中有其他列，也可以添加
            if 'channel_id' in row1:
                match['channel_id_1'] = str(row1['channel_id'])
            if 'channel_id' in row2:
                match['channel_id_2'] = str(row2['channel_id'])
            if 'cluster' in row1:
                match['cluster_1'] = str(row1['cluster'])
            if 'cluster' in row2:
                match['cluster_2'] = str(row2['cluster'])
            if 'tract_channel' in row1:
                match['tract_channel_1'] = row1.get('tract_channel', None)
            if 'tract_channel' in row2:
                match['tract_channel_2'] = row2.get('tract_channel', None)
            
            matches.append(match)
    
    # 转换为DataFrame
    if len(matches) == 0:
        print("未找到任何匹配的neuron")
        return pd.DataFrame()
    
    matches_df = pd.DataFrame(matches)
    
    # 按位置距离排序
    matches_df = matches_df.sort_values('position_distance')
    
    return matches_df


def main():
    parser = argparse.ArgumentParser(
        description="匹配两个记录的neuron_inf，根据位置和波形进行配对"
    )
    parser.add_argument(
        "--neuron-inf1",
        type=str,
        required=True,
        help="第一个记录的neuron_inf.pkl文件路径",
    )
    parser.add_argument(
        "--neuron-inf2",
        type=str,
        required=True,
        help="第二个记录的neuron_inf.pkl文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出CSV文件路径（默认：在第一个文件同目录下生成match_results.csv）",
    )
    parser.add_argument(
        "--position-threshold",
        type=float,
        default=10.0,
        help="位置距离阈值（默认10.0）",
    )
    parser.add_argument(
        "--waveform-threshold",
        type=float,
        default=0.95,
        help="波形相关系数阈值（默认0.95）",
    )
    
    args = parser.parse_args()
    
    # 加载neuron_inf文件
    print(f"加载第一个neuron_inf: {args.neuron_inf1}")
    neuron_inf1 = load_neuron_inf(Path(args.neuron_inf1))
    print(f"  找到 {len(neuron_inf1)} 个neuron")
    
    print(f"加载第二个neuron_inf: {args.neuron_inf2}")
    neuron_inf2 = load_neuron_inf(Path(args.neuron_inf2))
    print(f"  找到 {len(neuron_inf2)} 个neuron")
    
    # 进行匹配
    print(f"\n开始匹配（位置阈值: {args.position_threshold}, 波形阈值: {args.waveform_threshold}）...")
    matches = match_neurons(
        neuron_inf1,
        neuron_inf2,
        position_threshold=args.position_threshold,
        waveform_threshold=args.waveform_threshold,
    )
    
    if len(matches) == 0:
        print("未找到任何匹配")
        return
    
    print(f"\n找到 {len(matches)} 个匹配:")
    print(matches.to_string())
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认保存在第一个文件同目录下
        output_path = Path(args.neuron_inf1).parent / "match_results.csv"
    
    matches.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"  匹配数量: {len(matches)}")
    print(f"  平均位置距离: {matches['position_distance'].mean():.2f}")
    print(f"  平均波形相关系数: {matches['waveform_correlation'].mean():.4f}")
    print(f"  第一个记录中已匹配的neuron数: {matches['neuron1'].nunique()}")
    print(f"  第二个记录中已匹配的neuron数: {matches['neuron2'].nunique()}")


if __name__ == "__main__":
    main()

