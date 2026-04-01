#!/usr/bin/env python3
"""
分析 decoding_model_loocv.py 生成的结果 (decoding_results_loocv.pkl)
使用 spike_sorting 环境: conda run -n spike_sorting python analyze_decoding_results_loocv.py
"""

import os
import pickle
import numpy as np

OUTPUT_DIR = '/media/ubuntu/sda/TrippleN/customize/decoding_analysis'
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'decoding_results_loocv.pkl')


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"未找到结果文件: {RESULTS_PATH}")
        print("请先运行 (spike_sorting 环境):")
        print("  cd /media/ubuntu/sda/TrippleN/scripts")
        print("  conda run -n spike_sorting python decoding_model_loocv.py --quick   # 快速 2 个模型")
        print("  或: conda run -n spike_sorting python decoding_model_loocv.py     # 全部 8 个模型")
        return

    with open(RESULTS_PATH, 'rb') as f:
        results = pickle.load(f)

    names = list(results.keys())
    mean_corrs = np.array([results[k]["mean_corr"] for k in names])
    diag_off = np.array([results[k]["diag_offdiag"] for k in names])
    order = np.argsort(mean_corrs)[::-1]

    print("=" * 70)
    print("Decoding 结果分析 (PCA + LOOCV 方法)")
    print("=" * 70)
    print("方法: Response PCA (500维) + Model PCA (100维) + LOOCV (1000折)")
    print("数据: neuron_responses_1000.npy, 1000 张图片")
    print("指标: mean_corr = LOOCV 测试集上 predicted vs target 相似度矩阵对角线均值")
    print()

    print("模型名称                           mean_corr   diag_offdiag   排名")
    print("-" * 70)
    for r, i in enumerate(order, 1):
        print(f"  {names[i]:<30}   {mean_corrs[i]:.4f}      {diag_off[i]:.4f}       {r}")

    print()
    print("汇总:")
    print(f"  最高: {names[order[0]]} = {mean_corrs[order[0]]:.4f}")
    print(f"  最低: {names[order[-1]]} = {mean_corrs[order[-1]]:.4f}")
    print(f"  平均: {np.mean(mean_corrs):.4f} ± {np.std(mean_corrs):.4f}")
    print()
    
    if len(results) > 0:
        first_key = list(results.keys())[0]
        if "pca_neural" in results[first_key]:
            pca_neural = results[first_key]["pca_neural"]
            pca_model = results[first_key]["pca_model"]
            print("PCA 信息 (示例 - 第一个模型):")
            print(f"  Response PCA explained variance: {pca_neural.explained_variance_ratio_.sum():.4f}")
            print(f"  Model PCA explained variance: {pca_model.explained_variance_ratio_.sum():.4f}")
            print()
    
    print("解读:")
    print("  - mean_corr 越高表示从神经活动解码出的特征越接近该模型对图像/文本的表征")
    print("  - diag_offdiag 为正说明正确试次相似度高于混淆试次 (解码有效)")
    print("  - LOOCV 确保每张图片都作为测试样本，避免数据泄露")
    print("=" * 70)


if __name__ == '__main__':
    main()
