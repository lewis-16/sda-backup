#!/usr/bin/env python
# coding: utf-8
"""
计算达到95% TPR所需的阈值
对于021322和022522两个数据集，分别计算达到95% TPR时应该使用的阈值
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import json
from scipy import stats

# ==================== 数据加载 ====================

def load_logits_and_ground_truth(logits_path, spike_inf_path, spike_times):
    """
    加载logits数据和ground truth标签
    
    参数:
    logits_path : str, logits文件路径
    spike_inf_path : str, ground truth spike_inf.tsv文件路径
    spike_times : numpy.ndarray, 检测到的spike时间点（与logits对应）
    
    返回:
    logits : numpy.ndarray, shape (n_samples, 2)
    spike_probs : numpy.ndarray, shape (n_samples,)
    gt_labels : numpy.ndarray, shape (n_samples,), 1=spike, 0=noise
    """
    print(f"\n[INFO] Loading logits from {logits_path}")
    logits = np.load(logits_path)
    
    print(f"[INFO] Loading ground truth from {spike_inf_path}")
    spike_inf = pd.read_csv(spike_inf_path, sep='\t')
    
    # 计算spike概率
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    spike_probs = probs[:, 1]
    
    # 将spike_times与ground truth匹配
    # 使用时间容差匹配（±1个采样点）
    gt_times = spike_inf['time'].values.astype(np.int64)
    
    print(f"[INFO] Matching {len(spike_times):,} detected spikes to {len(gt_times):,} GT spikes...")
    
    # 创建ground truth标签数组
    gt_labels = np.zeros(len(spike_times), dtype=int)
    
    # 对每个检测到的spike，查找是否有匹配的GT spike
    time_tolerance = 1
    matched_count = 0
    
    for i, detect_time in enumerate(spike_times):
        # 查找在容差范围内的GT spike
        matches = np.abs(gt_times - detect_time) <= time_tolerance
        if np.any(matches):
            gt_labels[i] = 1  # 匹配到GT spike，标记为spike
            matched_count += 1
    
    print(f"[INFO] Matched {matched_count:,} spikes ({matched_count/len(spike_times)*100:.2f}%)")
    print(f"[INFO] GT labels: {np.sum(gt_labels)} spikes, {len(gt_labels) - np.sum(gt_labels)} noise")
    
    return logits, spike_probs, gt_labels


# ==================== TPR计算 ====================

def calculate_tpr_at_threshold(spike_probs, gt_labels, threshold):
    """
    计算给定阈值下的TPR
    
    参数:
    spike_probs : numpy.ndarray, spike概率
    gt_labels : numpy.ndarray, ground truth标签 (1=spike, 0=noise)
    threshold : float, 概率阈值
    
    返回:
    tpr : float, True Positive Rate
    fpr : float, False Positive Rate
    precision : float, Precision
    recall : float, Recall (same as TPR)
    """
    # 预测：spike_prob > threshold -> 预测为spike
    predictions = (spike_probs > threshold).astype(int)
    
    # 计算混淆矩阵
    tp = np.sum((predictions == 1) & (gt_labels == 1))  # True Positive
    fp = np.sum((predictions == 1) & (gt_labels == 0))  # False Positive
    tn = np.sum((predictions == 0) & (gt_labels == 0))  # True Negative
    fn = np.sum((predictions == 0) & (gt_labels == 1))  # False Negative
    
    # 计算指标
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr  # Recall = TPR
    
    return tpr, fpr, precision, recall, tp, fp, tn, fn


def find_threshold_for_tpr(spike_probs, gt_labels, total_gt_spikes, target_tpr=0.95, threshold_range=(0.0, 1.0), n_points=10000):
    """
    找到达到目标TPR所需的阈值
    
    参数:
    spike_probs : numpy.ndarray, spike概率
    gt_labels : numpy.ndarray, ground truth标签（在检测样本中的标签）
    total_gt_spikes : int, 所有ground truth spike的总数（用于正确计算TPR）
    target_tpr : float, 目标TPR（默认0.95）
    threshold_range : tuple, 阈值搜索范围
    n_points : int, 搜索点数
    
    返回:
    threshold : float, 达到目标TPR的阈值
    actual_tpr : float, 实际达到的TPR
    metrics : dict, 该阈值下的所有指标
    """
    print(f"\n[INFO] Finding threshold for TPR = {target_tpr*100:.1f}%...")
    print(f"[INFO] Total GT spikes: {total_gt_spikes:,}")
    print(f"[INFO] Detected samples: {len(spike_probs):,} (其中 {np.sum(gt_labels):,} 匹配到GT spike)")
    
    # 生成候选阈值
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
    
    # 计算每个阈值下的TPR
    # 注意：TPR应该基于所有GT spike，而不是检测到的样本
    tprs = []
    for threshold in thresholds:
        # 预测：spike_prob > threshold -> 预测为spike
        predictions = (spike_probs > threshold).astype(int)
        
        # TP: 预测为spike且确实是spike（匹配到GT）
        tp = np.sum((predictions == 1) & (gt_labels == 1))
        
        # FN: 所有GT spike中，没有被检测到或检测到但被预测为noise的数量
        # 由于我们只能知道检测到的样本，FN = total_gt_spikes - tp
        # 但这是近似值，因为可能有些GT spike根本没有被检测到
        fn = total_gt_spikes - tp
        
        # TPR = TP / (TP + FN) = TP / total_gt_spikes
        tpr = tp / total_gt_spikes if total_gt_spikes > 0 else 0.0
        tprs.append(tpr)
    
    tprs = np.array(tprs)
    
    # 找到最接近目标TPR的阈值
    # 如果目标TPR是95%，我们找TPR >= 95%的最小阈值（更保守）
    valid_indices = np.where(tprs >= target_tpr)[0]
    
    if len(valid_indices) == 0:
        # 如果无法达到目标TPR，返回能达到的最大TPR对应的阈值
        max_tpr_idx = np.argmax(tprs)
        best_threshold = thresholds[max_tpr_idx]
        best_tpr = tprs[max_tpr_idx]
        print(f"[WARNING] Cannot achieve TPR = {target_tpr*100:.1f}%")
        print(f"[WARNING] Maximum achievable TPR = {best_tpr*100:.2f}% at threshold = {best_threshold:.6f}")
        print(f"[WARNING] This might be because not all GT spikes were detected.")
        return best_threshold, best_tpr, None
    else:
        # 选择能达到目标TPR的最小阈值（更保守，减少false positive）
        best_idx = valid_indices[0]
        best_threshold = thresholds[best_idx]
        best_tpr = tprs[best_idx]
    
    # 计算该阈值下的所有指标
    predictions = (spike_probs > best_threshold).astype(int)
    tp = np.sum((predictions == 1) & (gt_labels == 1))
    fp = np.sum((predictions == 1) & (gt_labels == 0))
    tn = np.sum((predictions == 0) & (gt_labels == 0))
    fn = total_gt_spikes - tp  # 近似值
    
    tpr = tp / total_gt_spikes if total_gt_spikes > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    
    metrics = {
        'threshold': float(best_threshold),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'total_detected_samples': len(spike_probs),
        'matched_gt_spikes': int(np.sum(gt_labels)),
        'total_gt_spikes': int(total_gt_spikes),
    }
    
    print(f"[INFO] Found threshold = {best_threshold:.6f}")
    print(f"[INFO]   TPR = {tpr*100:.2f}% (TP={tp:,} / Total GT={total_gt_spikes:,})")
    print(f"[INFO]   FPR = {fpr*100:.2f}%")
    print(f"[INFO]   Precision = {precision*100:.2f}%")
    print(f"[INFO]   TP = {tp:,}, FP = {fp:,}, TN = {tn:,}, FN = {fn:,}")
    
    return best_threshold, best_tpr, metrics


# ==================== ROC曲线分析 ====================

def calculate_roc_curve(spike_probs, gt_labels, n_points=1000):
    """
    计算ROC曲线
    
    参数:
    spike_probs : numpy.ndarray, spike概率
    gt_labels : numpy.ndarray, ground truth标签
    n_points : int, ROC曲线点数
    
    返回:
    fprs : numpy.ndarray, False Positive Rates
    tprs : numpy.ndarray, True Positive Rates
    thresholds : numpy.ndarray, 对应的阈值
    """
    thresholds = np.linspace(0.0, 1.0, n_points)
    fprs = []
    tprs = []
    
    for threshold in thresholds:
        tpr, fpr, _, _, _, _, _, _ = calculate_tpr_at_threshold(spike_probs, gt_labels, threshold)
        tprs.append(tpr)
        fprs.append(fpr)
    
    return np.array(fprs), np.array(tprs), thresholds


# ==================== 主函数 ====================

def main():
    print("="*80)
    print("Calculate Threshold for 95% TPR")
    print("="*80)
    
    # 配置路径
    base_dir = '/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels'
    logits_comparison_dir = os.path.join(base_dir, 'logits_comparison_results')
    output_dir = os.path.join(base_dir, 'tpr_threshold_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据路径
    logits_021322_path = os.path.join(logits_comparison_dir, 'logits_021322.npy')
    logits_022522_path = os.path.join(logits_comparison_dir, 'logits_022522.npy')
    spike_times_021322_path = os.path.join(logits_comparison_dir, 'spike_times_021322.npy')
    spike_times_022522_path = os.path.join(logits_comparison_dir, 'spike_times_022522.npy')
    
    spike_inf_021322_path = os.path.join(base_dir, 'kilosort_spike_sorting/sorting_new/021322/spike_inf.tsv')
    spike_inf_022522_path = os.path.join(base_dir, 'kilosort_spike_sorting/sorting_new/022522/spike_inf.tsv')
    
    # 检查文件是否存在
    if not os.path.exists(logits_021322_path):
        raise FileNotFoundError(f"Logits file not found: {logits_021322_path}")
    if not os.path.exists(logits_022522_path):
        raise FileNotFoundError(f"Logits file not found: {logits_022522_path}")
    if not os.path.exists(spike_times_021322_path):
        raise FileNotFoundError(f"Spike times file not found: {spike_times_021322_path}. Please run compare_logits_distribution.py first.")
    if not os.path.exists(spike_times_022522_path):
        raise FileNotFoundError(f"Spike times file not found: {spike_times_022522_path}. Please run compare_logits_distribution.py first.")
    
    # 加载logits和spike_times
    print(f"\n[INFO] Loading logits and spike_times...")
    logits_021322 = np.load(logits_021322_path)
    logits_022522 = np.load(logits_022522_path)
    spike_times_021322 = np.load(spike_times_021322_path)
    spike_times_022522 = np.load(spike_times_022522_path)
    
    # 计算概率
    probs_021322 = np.exp(logits_021322) / np.sum(np.exp(logits_021322), axis=1, keepdims=True)
    spike_probs_021322 = probs_021322[:, 1]
    
    probs_022522 = np.exp(logits_022522) / np.sum(np.exp(logits_022522), axis=1, keepdims=True)
    spike_probs_022522 = probs_022522[:, 1]
    
    # 加载ground truth
    print(f"\n[INFO] Loading ground truth for 021322...")
    spike_inf_021322 = pd.read_csv(spike_inf_021322_path, sep='\t')
    gt_times_021322 = spike_inf_021322['time'].values.astype(np.int64)
    
    print(f"[INFO] Loading ground truth for 022522...")
    spike_inf_022522 = pd.read_csv(spike_inf_022522_path, sep='\t')
    gt_times_022522 = spike_inf_022522['time'].values.astype(np.int64)
    
    # 匹配spike_times与ground truth
    print(f"\n[INFO] Matching detected spikes with ground truth...")
    
    def match_spikes_to_gt(detect_times, gt_times, time_tolerance=1):
        """将检测到的spike与ground truth匹配"""
        gt_labels = np.zeros(len(detect_times), dtype=int)
        matched_count = 0
        
        for i, detect_time in enumerate(detect_times):
            # 查找在容差范围内的GT spike
            matches = np.abs(gt_times - detect_time) <= time_tolerance
            if np.any(matches):
                gt_labels[i] = 1  # 匹配到GT spike，标记为spike
                matched_count += 1
        
        return gt_labels, matched_count
    
    gt_labels_021322, matched_021322 = match_spikes_to_gt(spike_times_021322, gt_times_021322)
    gt_labels_022522, matched_022522 = match_spikes_to_gt(spike_times_022522, gt_times_022522)
    
    # 获取所有GT spike的总数（用于正确计算TPR）
    total_gt_spikes_021322 = len(gt_times_021322)
    total_gt_spikes_022522 = len(gt_times_022522)
    
    print(f"[INFO] 021322: {matched_021322:,}/{len(spike_times_021322):,} detected spikes matched to GT ({matched_021322/len(spike_times_021322)*100:.2f}%)")
    print(f"[INFO] 022522: {matched_022522:,}/{len(spike_times_022522):,} detected spikes matched to GT ({matched_022522/len(spike_times_022522)*100:.2f}%)")
    print(f"[INFO] 021322: {np.sum(gt_labels_021322):,} matched spikes in detected samples, {len(gt_labels_021322) - np.sum(gt_labels_021322):,} noise")
    print(f"[INFO] 022522: {np.sum(gt_labels_022522):,} matched spikes in detected samples, {len(gt_labels_022522) - np.sum(gt_labels_022522):,} noise")
    print(f"[INFO] 021322: Total GT spikes = {total_gt_spikes_021322:,}")
    print(f"[INFO] 022522: Total GT spikes = {total_gt_spikes_022522:,}")
    
    # 计算阈值
    print("\n" + "="*80)
    print("021322 Dataset")
    print("="*80)
    threshold_021322, tpr_021322, metrics_021322 = find_threshold_for_tpr(
        spike_probs_021322, gt_labels_021322, total_gt_spikes_021322, target_tpr=0.95
    )
    
    print("\n" + "="*80)
    print("022522 Dataset")
    print("="*80)
    threshold_022522, tpr_022522, metrics_022522 = find_threshold_for_tpr(
        spike_probs_022522, gt_labels_022522, total_gt_spikes_022522, target_tpr=0.95
    )
    
    # 保存结果
    results = {
        '021322': {
            'threshold': float(threshold_021322),
            'tpr': float(tpr_021322),
            'metrics': metrics_021322
        },
        '022522': {
            'threshold': float(threshold_022522),
            'tpr': float(tpr_022522),
            'metrics': metrics_022522
        },
        'note': 'Thresholds calculated based on matching detected spikes with ground truth using time tolerance of 1 sample.'
    }
    
    results_path = os.path.join(output_dir, 'tpr_threshold_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\n[INFO] Results saved to: {results_path}")
    print(f"\n[INFO] Thresholds for 95% TPR:")
    print(f"  021322: threshold = {threshold_021322:.6f}, TPR = {tpr_021322*100:.2f}%")
    print(f"  022522: threshold = {threshold_022522:.6f}, TPR = {tpr_022522*100:.2f}%")
    print(f"\n[INFO] These thresholds are based on matching detected spikes with ground truth.")
    print(f"[INFO] Use these thresholds in your detection model to achieve 95% TPR.")


if __name__ == "__main__":
    main()

