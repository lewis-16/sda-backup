#!/usr/bin/env python3
"""
分析 decoding_model.py 生成的结果 (decoding_results.pkl)
使用 spike_sorting 环境: conda run -n spike_sorting python analyze_decoding_results.py
"""

import os
import pickle
import numpy as np

OUTPUT_DIR = '/media/ubuntu/sda/TrippleN/customize/decoding_analysis'
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'decoding_results.pkl')


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"未找到结果文件: {RESULTS_PATH}")
        print("请先运行 (spike_sorting 环境):")
        print("  cd /media/ubuntu/sda/TrippleN/scripts")
        print("  conda run -n spike_sorting python decoding_model.py --quick   # 快速 2 个模型")
        print("  或: conda run -n spike_sorting python decoding_model.py     # 全部 8 个模型")
        print("或执行: ./run_decoding_with_spike_sorting.sh [--quick]")
        return

    with open(RESULTS_PATH, 'rb') as f:
        results = pickle.load(f)

    names = list(results.keys())
    mean_corrs = np.array([results[k]["mean_corr"] for k in names])
    diag_off = np.array([results[k]["diag_offdiag"] for k in names])
    order = np.argsort(mean_corrs)[::-1]

    print("=" * 64)
    print("Decoding 结果分析 (神经元响应 -> 模型特征)")
    print("=" * 64)
    print("数据: neuron_responses_1000.npy, 前900张训练, 后100张测试")
    print("指标: mean_corr = 测试集上 predicted vs target 相似度矩阵对角线均值")
    print()

    print("模型名称                           mean_corr   diag_offdiag   排名")
    print("-" * 64)
    for r, i in enumerate(order, 1):
        print(f"  {names[i]:<30}   {mean_corrs[i]:.4f}      {diag_off[i]:.4f}       {r}")

    print()
    print("汇总:")
    print(f"  最高: {names[order[0]]} = {mean_corrs[order[0]]:.4f}")
    print(f"  最低: {names[order[-1]]} = {mean_corrs[order[-1]]:.4f}")
    print(f"  平均: {np.mean(mean_corrs):.4f} ± {np.std(mean_corrs):.4f}")
    print()
    print("解读: mean_corr 越高表示从神经活动解码出的特征越接近该模型对图像/文本的表征;")
    print("      diag_offdiag 为正说明正确试次相似度高于混淆试次 (解码有效)。")
    print("=" * 64)


if __name__ == '__main__':
    main()
