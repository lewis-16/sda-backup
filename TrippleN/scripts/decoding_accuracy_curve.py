#!/usr/bin/env python3
"""
在 LOOCV 得到 predicted 与 target embedding 后，计算不同候选集大小 n 下的解码准确率。

对每个 n in [2, 10, 20, ..., 1000]:
  重复 50,000 次随机抽样实验:
    - 从 1000 张图像中随机选 n 张作为候选集
    - 在候选集中随机指定一张为目标图像，其余 n-1 张为干扰
    - 用目标图像的预测向量与候选集中 n 张真实向量按 Pearson 相关作为相似度
      (与 09_neural_mapping.ipynb 中 1 - cdist(..., metric='correlation') 一致)
    - 若目标图像的预测与自身真实向量相似度最高则解码正确
  准确率 = 正确次数 / 50000
"""

import os
import pickle
import numpy as np
from datetime import datetime


def correlation_similarity_batch(pred_vecs, real_mats, eps=1e-12):
    """
    pred_vecs: (k, d), real_mats: (k, n, d)
    返回 (k, n) 的 Pearson 相关系数，与 scipy cdist(..., 'correlation') 满足:
    1 - cdist(a, b, 'correlation') == corr(a, b) 对逐对行成立。
    """
    pred = np.asarray(pred_vecs, dtype=np.float64)
    real = np.asarray(real_mats, dtype=np.float64)
    pred_c = pred - pred.mean(axis=1, keepdims=True)
    real_c = real - real.mean(axis=2, keepdims=True)
    num = np.einsum('kd,knd->kn', pred_c, real_c)
    den = np.linalg.norm(pred_c, axis=1, keepdims=True) * np.linalg.norm(real_c, axis=2) + eps
    return num / den

OUTPUT_DIR = '/media/ubuntu/sda/TrippleN/customize/decoding_analysis'
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'decoding_results_loocv.pkl')
ACCURACY_PATH = os.path.join(OUTPUT_DIR, 'decoding_accuracy_curve.pkl')
N_IMAGES = 1000
N_TRIALS = 50000
N_VALUES = [2] + list(range(10, 1001, 10))
RANDOM_SEED = 42


def compute_accuracy_curve(predictions, target_reduced, n_values, n_trials, seed):
    """
    predictions: (1000, dim), target_reduced: (1000, dim)
    返回: accuracies 字典 {n: accuracy}
    分块向量化以控制内存: 每次处理 chunk_size 个 trial。
    """
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
            sims = correlation_similarity_batch(pred_vecs, real_mats)
            correct_count += np.sum(np.argmax(sims, axis=1) == target_positions)
        accuracies[n] = correct_count / n_trials
    
    return accuracies


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Decoding accuracy vs candidate set size n')
    parser.add_argument('--trials', type=int, default=N_TRIALS, help=f'每个 n 的重复次数 (默认 {N_TRIALS})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='随机种子')
    parser.add_argument('--quick', action='store_true', help='仅 n=[2,10,20], trials=1000 用于测试')
    parser.add_argument('--models', nargs='*', default=None,
                        help='仅计算指定模型的准确率曲线，不传则计算 pkl 中全部模型')
    args = parser.parse_args()
    
    n_trials = args.trials
    seed = args.seed
    if args.quick:
        n_values = [2, 10, 20]
        n_trials = 1000
        print("Quick 模式: n=[2,10,20], trials=1000")
    else:
        n_values = N_VALUES
    
    if not os.path.exists(RESULTS_PATH):
        print(f"未找到 LOOCV 结果: {RESULTS_PATH}")
        print("请先运行: conda run -n spike_sorting python decoding_model_loocv.py [--quick]")
        return
    
    with open(RESULTS_PATH, 'rb') as f:
        results = pickle.load(f)
    
    if args.models is not None:
        results = {k: v for k, v in results.items() if k in args.models}
        if not results:
            print("--models 指定的模型在 pkl 中均未找到，退出")
            return
    
    print("=" * 60)
    print("Decoding 准确率曲线 (候选集大小 n, 50k 次/点, 相似度=Pearson 相关 argmax)")
    print("=" * 60)
    
    all_curves = {}
    
    for model_name, data in results.items():
        if "target_reduced" not in data:
            print(f"  跳过 {model_name}: 无 target_reduced，请重新运行 decoding_model_loocv.py 生成")
            continue
        
        predictions = np.asarray(data["predictions"], dtype=np.float64)
        target_reduced = np.asarray(data["target_reduced"], dtype=np.float64)
        if predictions.shape[0] != N_IMAGES or target_reduced.shape[0] != N_IMAGES:
            print(f"  跳过 {model_name}: 样本数不为 {N_IMAGES}")
            continue
        
        print(f"\n计算 {model_name} ...")
        t0 = datetime.now()
        acc = compute_accuracy_curve(predictions, target_reduced, n_values, n_trials, seed)
        all_curves[model_name] = acc
        print(f"  耗时: {datetime.now() - t0}")
        for n in (n_values[:5] + n_values[-3:]):
            if n in acc:
                print(f"    n={n:4d}  accuracy = {acc[n]:.4f}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(ACCURACY_PATH, 'wb') as f:
        pickle.dump({
            "n_values": n_values,
            "n_trials": n_trials,
            "seed": seed,
            "accuracy_curves": all_curves
        }, f)
    
    print(f"\n结果已保存: {ACCURACY_PATH}")
    
    print("\n汇总 (部分 n):")
    models = list(all_curves.keys())
    header = "  n     " + "  ".join(f"{m[:14]:>14}" for m in models)
    print(header)
    for n in n_values[:8] + n_values[-5:]:
        row = f"  {n:4d}  " + "  ".join(f"{all_curves[m].get(n, np.nan):.4f}" for m in models)
        print(row)
    print("=" * 60)


if __name__ == '__main__':
    main()
