import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

try:
    import neo
    import quantities as pq
    from elephant.statistics import instantaneous_rate
    from elephant.kernels import GaussianKernel
except Exception as e:
    neo = None
    pq = None
    instantaneous_rate = None
    GaussianKernel = None


def load_triggers(trigger_csv_paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in trigger_csv_paths:
        df = pd.read_csv(p)
        frames.append(df)
    trig = pd.concat(frames, ignore_index=True)
    # 期望列: start_time, stop_time, id, train_image, test_image, image_rep_num, single_train_rep, valid_image(可选)
    required = [
        'start_time', 'stop_time', 'train_image', 'test_image', 'image_rep_num', 'single_train_rep'
    ]
    for col in required:
        if col not in trig.columns:
            raise ValueError(f"Trigger 文件缺少列: {col}")
    # 仅保留有效图像试次（若列存在）
    if 'valid_image' in trig.columns:
        trig = trig[trig['valid_image'] == 1].copy()
    # 将 train_image/test_image 的 NaN 填 0，避免后续 int 转换报错
    trig['train_image'] = trig['train_image'].fillna(0)
    trig['test_image'] = trig['test_image'].fillna(0)
    trig['image_rep_num'] = trig['image_rep_num'].fillna(0)
    trig['single_train_rep'] = trig['single_train_rep'].fillna(0)
    return trig


def make_trigger_key(row: pd.Series) -> str:
    # 训练阶段：train_image 非 0；否则 test_image
    train_val = int(row['train_image']) if not pd.isna(row['train_image']) else 0
    test_val = int(row['test_image']) if not pd.isna(row['test_image']) else 0
    is_train = train_val != 0
    if is_train:
        image_id = train_val
        phase = 'train'
    else:
        image_id = test_val
        phase = 'test'
    rep_num = int(row['image_rep_num']) if not pd.isna(row['image_rep_num']) else 0
    single_rep = int(row['single_train_rep']) if not pd.isna(row['single_train_rep']) else 0
    key = f"{phase}_{image_id}_{rep_num}_{single_rep}"
    return key


def build_neuron_index(cluster_df: pd.DataFrame, region: str = None) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], int]]:
    # 仅选 group=='good'
    if 'group' in cluster_df.columns:
        cluster_df = cluster_df[cluster_df['group'] == 'good'].copy()
    # 可选区域过滤
    if region is not None:
        if 'region' not in cluster_df.columns:
            raise ValueError('cluster_inf 缺少 region 列，无法按区域过滤')
        cluster_df = cluster_df[cluster_df['region'] == region].copy()
    # 神经元以 (array_id, cluster_id) 定义
    if 'array_id' not in cluster_df.columns or 'cluster_id' not in cluster_df.columns:
        raise ValueError('cluster_inf 缺少 array_id 或 cluster_id 列')
    neurons = cluster_df[['array_id', 'cluster_id']].copy()
    neurons['array_id'] = neurons['array_id'].astype(str)
    neurons['cluster_id'] = neurons['cluster_id'].astype(int)
    neurons = neurons.drop_duplicates().reset_index(drop=True)
    neurons['neuron_index'] = np.arange(len(neurons), dtype=int)
    mapping: Dict[Tuple[str, int], int] = {
        (row['array_id'], row['cluster_id']): int(row['neuron_index'])
        for _, row in neurons.iterrows()
    }
    return neurons, mapping


def group_spike_times(spike_df: pd.DataFrame, sample_rate: float) -> Dict[Tuple[str, int], np.ndarray]:
    # 需要列: array_id, cluster_id, time(样本)
    required = ['array_id', 'cluster_id', 'time']
    for col in required:
        if col not in spike_df.columns:
            raise ValueError(f'spike_inf 缺少列: {col}')
    s = spike_df[['array_id', 'cluster_id', 'time']].copy()
    s['array_id'] = s['array_id'].astype(str)
    s['cluster_id'] = s['cluster_id'].astype(int)
    # 转秒
    s['t_sec'] = s['time'].astype(float) / float(sample_rate)
    grouped: Dict[Tuple[str, int], np.ndarray] = {}
    for (arr, clu), g in s.groupby(['array_id', 'cluster_id']):
        grouped[(arr, int(clu))] = g['t_sec'].to_numpy()
    return grouped


def compute_firing_rate_matrix_for_window_instant(
    start_sec: float,
    stop_sec: float,
    neuron_order: pd.DataFrame,
    spike_times_by_neuron: Dict[Tuple[str, int], np.ndarray],
    sampling_period_sec: float,
    kernel_sigma_sec: float
) -> Tuple[np.ndarray, np.ndarray]:
    if neo is None or instantaneous_rate is None or GaussianKernel is None or pq is None:
        raise RuntimeError('需要安装 neo 和 elephant 库以使用 instantaneous_rate 计算放电率')
    duration = max(0.0, float(stop_sec) - float(start_sec))
    if duration <= 0:
        # 至少一个采样点
        edges = np.array([start_sec, start_sec + sampling_period_sec], dtype=float)
        n_bins = 1
    else:
        n_bins = int(np.ceil(duration / sampling_period_sec))
        edges = start_sec + np.arange(n_bins + 1, dtype=float) * sampling_period_sec
        if edges[-1] < stop_sec:
            edges = np.append(edges, stop_sec)
            n_bins = len(edges) - 1
    n_neurons = len(neuron_order)
    fr = np.zeros((n_neurons, n_bins), dtype=float)
    kernel = GaussianKernel(sigma=kernel_sigma_sec * pq.s)
    sampling_period = sampling_period_sec * pq.s
    # t_stop 至少为一个采样周期，避免 bins=0
    effective_duration = max(duration, sampling_period_sec)
    t_stop = effective_duration * pq.s
    # 对每个神经元计算瞬时发放率
    for _, row in neuron_order.iterrows():
        idx = int(row['neuron_index'])
        key = (row['array_id'], int(row['cluster_id']))
        t = spike_times_by_neuron.get(key)
        if t is None or t.size == 0:
            continue
        # 相对时间（秒）
        rel_t = t[(t >= start_sec) & (t < stop_sec)] - start_sec
        if rel_t.size == 0:
            continue
        st = neo.SpikeTrain(rel_t * pq.s, t_start=0 * pq.s, t_stop=t_stop)
        rates = instantaneous_rate(st, sampling_period=sampling_period, kernel=kernel)
        r = np.asarray(rates.magnitude).reshape(-1)  # Hz
        # 截断或填充到 n_bins
        if r.size >= n_bins:
            fr[idx, :] = r[:n_bins]
        else:
            fr[idx, :r.size] = r
    return fr, edges


def build_firing_rate_dict(
    cluster_csv: str,
    spike_csv: str,
    trigger_csvs: List[str],
    sample_rate: float,
    bin_width: float,
    region: str = None
) -> Dict[str, Dict[str, object]]:
    cluster_df = pd.read_csv(cluster_csv)
    spike_df = pd.read_csv(spike_csv)
    triggers = load_triggers(trigger_csvs)

    neuron_order, neuron_index = build_neuron_index(cluster_df, region=region)
    spikes_by_neuron = group_spike_times(spike_df, sample_rate)

    # 用 instantaneous_rate: kernel 40ms，采样 20ms
    sampling_period_sec = 0.02
    kernel_sigma_sec = 0.04

    fr_dict: Dict[str, Dict[str, object]] = {}
    for _, row in triggers.iterrows():
        start_sec = float(row['start_time'])
        stop_sec = float(row['stop_time'])
        key = make_trigger_key(row)
        fr_mat, edges = compute_firing_rate_matrix_for_window_instant(
            start_sec=start_sec,
            stop_sec=stop_sec,
            neuron_order=neuron_order,
            spike_times_by_neuron=spikes_by_neuron,
            sampling_period_sec=sampling_period_sec,
            kernel_sigma_sec=kernel_sigma_sec,
        )
        fr_dict[key] = {
            'phase': 'train_image' if (int(row['train_image']) if not pd.isna(row['train_image']) else 0) != 0 else 'test_image',
            'image_id': int(row['train_image']) if (not pd.isna(row['train_image']) and int(row['train_image']) != 0) else int(row['test_image']) if not pd.isna(row['test_image']) else 0,
            'image_rep_num': int(row['image_rep_num']) if not pd.isna(row['image_rep_num']) else 0,
            'single_train_rep': int(row['single_train_rep']) if not pd.isna(row['single_train_rep']) else 0,
            'edges_sec': edges,
            'bin_width_sec': sampling_period_sec,
            'neuron_order': neuron_order.copy(),
            'firing_rate': fr_mat,
            'region': region,
        }
    return fr_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build firing-rate matrices per trigger window.')
    p.add_argument('--cluster_csv', required=True, help='Path to cluster_inf CSV')
    p.add_argument('--spike_csv', required=True, help='Path to spike_inf CSV')
    p.add_argument('--triggers', required=True, nargs='+', help='One or more trigger CSVs')
    p.add_argument('--fs', type=float, default=30000.0, help='Sample rate (Hz) for spike_times')
    p.add_argument('--bin_width', type=float, default=0.01, help='[unused] kept for backward-compat; instantaneous rate uses 20ms')
    p.add_argument('--region', type=str, default=None, help='Filter neurons by region, e.g., V1/V4/IT')
    p.add_argument('--out', type=str, default=None, help='Output .npz path to save dict arrays')
    return p.parse_args()


def save_fr_dict_npz(fr_dict: Dict[str, Dict[str, object]], out_path: str) -> None:
    # 将字典序列化为 npz：对每个 key 展平保存；复杂对象（DataFrame）转为 CSV 字节或单独保存
    pack = {}
    neuron_order_any = None
    for k, v in fr_dict.items():
        pack[f'{k}__firing_rate'] = v['firing_rate']
        pack[f'{k}__edges_sec'] = v['edges_sec']
        pack[f'{k}__meta'] = np.array([
            v['phase'],
            str(v['image_id']),
            str(v['image_rep_num']),
            str(v['single_train_rep']),
            str(v['bin_width_sec']),
            str(v.get('region', ''))
        ], dtype=object)
        neuron_order_any = v['neuron_order']
    if neuron_order_any is not None:
        pack['neuron_order_csv'] = neuron_order_any.to_csv(index=False).encode('utf-8')
    np.savez_compressed(out_path, **pack)


def main():
    args = parse_args()
    fr_dict = build_firing_rate_dict(
        cluster_csv=args.cluster_csv,
        spike_csv=args.spike_csv,
        trigger_csvs=args.triggers,
        sample_rate=args.fs,
        bin_width=args.bin_width,
        region=args.region,
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        save_fr_dict_npz(fr_dict, args.out)
        print(f'Saved firing-rate dict to: {args.out}')
    else:
        print(f'Constructed firing-rate dict with {len(fr_dict)} keys')


if __name__ == '__main__':
    main()
