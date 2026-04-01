#!/usr/bin/env python3

import os
import pickle
import numpy as np
from datetime import datetime

DECODING_DIR = '/media/ubuntu/sda/TrippleN/customize/human_fMRI/decoding'
ACCURACY_PATH = os.path.join(DECODING_DIR, 'fmri_decoding_accuracy_curve.pkl')
N_IMAGES = 1000
N_TRIALS = 50000
N_VALUES = [2] + list(range(10, 1001, 10))
RANDOM_SEED = 42

REGIONS = [
    'early', 'midventral', 'midlateral', 'midparietal',
    'ventral', 'lateral', 'parietal',
]


def compute_accuracy_curve(predictions, target_reduced, n_values, n_trials, seed):
    rng = np.random.default_rng(seed)
    n_images = predictions.shape[0]
    accuracies = {}
    chunk_size = 2000

    for n in n_values:
        if n > n_images:
            continue
        correct_count = 0
        n_chunk = min(chunk_size, max(1, 100000 // (n + 1)))
        for start in range(0, n_trials, n_chunk):
            end = min(start + n_chunk, n_trials)
            k = end - start
            candidate_sets = np.array([rng.choice(n_images, size=n, replace=False) for _ in range(k)])
            target_positions = rng.integers(0, n, size=k)
            target_indices = candidate_sets[np.arange(k), target_positions]
            pred_vecs = predictions[target_indices]
            real_mats = target_reduced[candidate_sets]
            dists = np.linalg.norm(real_mats - pred_vecs[:, np.newaxis, :], axis=2)
            correct_count += np.sum(np.argmin(dists, axis=1) == target_positions)
        accuracies[n] = correct_count / n_trials

    return accuracies


def main():
    import argparse
    parser = argparse.ArgumentParser(description='fMRI Decoding accuracy curve -> (n_region, len_n, n_model)')
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--quick', action='store_true', help='n=[2,10,20], trials=1000')
    parser.add_argument('--models', nargs='*', default=None, help='仅计算指定模型')
    args = parser.parse_args()

    n_trials = args.trials
    seed = args.seed
    if args.quick:
        n_values = [2, 10, 20]
        n_trials = 1000
        print('Quick 模式: n=[2,10,20], trials=1000')
    else:
        n_values = N_VALUES

    len_n = len(n_values)
    region_order = []
    model_order = None
    all_curves = {}
    missing = []

    for region_name in REGIONS:
        detail_path = os.path.join(DECODING_DIR, f'fmri_decoding_{region_name}_detail.pkl')
        if not os.path.exists(detail_path):
            missing.append(detail_path)
            continue

        with open(detail_path, 'rb') as f:
            detail = pickle.load(f)

        if args.models is not None:
            detail = {k: v for k, v in detail.items() if k in args.models}
        if not detail:
            continue

        if model_order is None:
            model_order = sorted(detail.keys())
        region_order.append(region_name)

        for model_name, data in detail.items():
            predictions = np.asarray(data['predictions'], dtype=np.float64)
            target_reduced = np.asarray(data['target_reduced'], dtype=np.float64)
            if predictions.shape[0] != N_IMAGES or target_reduced.shape[0] != N_IMAGES:
                print(f'  跳过 {region_name}/{model_name}: 样本数不为 {N_IMAGES}')
                continue

            key = (region_name, model_name)
            print(f'计算 {region_name} / {model_name} ...')
            t0 = datetime.now()
            acc = compute_accuracy_curve(predictions, target_reduced, n_values, n_trials, seed)
            all_curves[key] = acc
            print(f'  耗时: {datetime.now() - t0}')

    if not region_order or model_order is None:
        print('未找到任何 fmri_decoding_*_detail.pkl')
        if missing:
            print('缺少:', missing[0], ('...' if len(missing) > 1 else ''))
        print('请先运行: python scripts/human_fMRI/decoding_fmri.py 生成 _detail.pkl')
        return

    if missing:
        print(f'跳过 {len(missing)} 个无 detail 的脑区')

    n_region = len(region_order)
    n_model = len(model_order)
    acc_array = np.full((n_region, len_n, n_model), np.nan, dtype=np.float64)

    for i_r, region in enumerate(region_order):
        for i_m, model in enumerate(model_order):
            key = (region, model)
            if key not in all_curves:
                continue
            acc = all_curves[key]
            for j_n, n in enumerate(n_values):
                if n in acc:
                    acc_array[i_r, j_n, i_m] = acc[n]

    os.makedirs(DECODING_DIR, exist_ok=True)
    with open(ACCURACY_PATH, 'wb') as f:
        pickle.dump({
            'accuracy': acc_array,
            'region_order': region_order,
            'n_values': n_values,
            'model_order': model_order,
            'n_trials': n_trials,
            'seed': seed,
        }, f)

    print(f'\n已保存: {ACCURACY_PATH}')
    print(f'shape: (n_region={n_region}, len_n={len_n}, n_model={n_model})')
    print('region_order:', region_order)
    print('model_order:', model_order)


if __name__ == '__main__':
    main()
