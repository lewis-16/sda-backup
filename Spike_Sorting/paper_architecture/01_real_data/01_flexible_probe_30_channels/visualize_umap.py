#!/usr/bin/env python
# coding: utf-8
"""
使用训练好的模型绘制UMAP降维可视化图
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
import os
import sys
import json
from pathlib import Path

# 获取当前脚本所在目录并加入搜索路径，然后导入 train_spike_pipeline 中的类
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from train_spike_pipeline import AutoSort, AutoSortDataset, SimpleClassifier

def load_model(noise_model_path, label_model_path, input_size, num_classes, device):
    """Instantiate AutoSort and load separate weights for noise and label classifiers.

    noise_model_path: path to noise classifier state_dict (or model file saved)
    label_model_path: path to label classifier state_dict (or model file saved)
    """
    model = AutoSort(input_size, num_classes, device)

    # Load noise classifier weights
    if noise_model_path is not None:
        sd = torch.load(noise_model_path, map_location=device)
        # sd may be a state_dict for the classifier or a whole model object
        if isinstance(sd, dict) and all(k.startswith('clsfier_noise') for k in sd.keys()):
            # extract sub-dict
            sub = {k.split('clsfier_noise.')[-1]: v for k, v in sd.items() if k.startswith('clsfier_noise.')}
            if sub:
                model.clsfier_noise.load_state_dict(sub)
            else:
                model.clsfier_noise.load_state_dict(sd)
        else:
            try:
                model.clsfier_noise.load_state_dict(sd)
            except Exception:
                # try loading as whole model
                try:
                    model_state = sd.get('model_state_dict', sd)
                    model.clsfier_noise.load_state_dict(model_state)
                except Exception:
                    # last resort: load entire file into classifier if shapes match
                    model.clsfier_noise.load_state_dict(sd)

    # Load label classifier weights
    if label_model_path is not None:
        sd = torch.load(label_model_path, map_location=device)
        if isinstance(sd, dict) and all(k.startswith('clsfier_label') for k in sd.keys()):
            sub = {k.split('clsfier_label.')[-1]: v for k, v in sd.items() if k.startswith('clsfier_label.')}
            if sub:
                model.clsfier_label.load_state_dict(sub)
            else:
                model.clsfier_label.load_state_dict(sd)
        else:
            try:
                model.clsfier_label.load_state_dict(sd)
            except Exception:
                try:
                    model_state = sd.get('model_state_dict', sd)
                    model.clsfier_label.load_state_dict(model_state)
                except Exception:
                    model.clsfier_label.load_state_dict(sd)

    model.eval()
    model.to(device)
    return model

def load_data(data_path):
    """加载预处理的数据"""
    df = pd.read_pickle(data_path)
    return df

def extract_features(model, df, device, batch_size=512, max_samples=50000):
    """
    提取所有样本的中间层特征
    """
    # 准备数据
    all_waveforms = np.stack(df['waveform'].values)
    cluster_labels_raw = df['mapping'].values
    noise_labels = (cluster_labels_raw >= 0).astype(int)
    spike_mask = noise_labels == 1
    
    # 处理cluster_id映射
    valid_mask = cluster_labels_raw >= 0
    unique_clusters = np.unique(cluster_labels_raw[valid_mask])
    unique_clusters = np.sort(unique_clusters)
    
    cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    cluster_series = pd.Series(cluster_labels_raw)
    cluster_labels_mapped = cluster_series.map(cluster_to_index).fillna(-1).astype(int).values
    
    # 限制样本数量（如果太多的话）
    if len(all_waveforms) > max_samples:
        print(f"[INFO] Sampling {max_samples} samples from {len(all_waveforms)} total samples")
        indices = np.random.choice(len(all_waveforms), max_samples, replace=False)
        all_waveforms = all_waveforms[indices]
        noise_labels = noise_labels[indices]
        cluster_labels_mapped = cluster_labels_mapped[indices]
        spike_mask = spike_mask[indices]
    
    # 创建数据集
    num_classes = len(unique_clusters)
    dataset = AutoSortDataset(
        all_waveforms,
        noise_labels,
        cluster_labels_mapped,
        spike_mask,
        num_classes=num_classes
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 提取特征
    noise_features_list = []
    label_features_list = []
    noise_pred_list = []
    label_pred_list = []
    noise_gt_list = []
    label_gt_list = []
    spike_mask_list = []
    
    print("[INFO] Extracting features...")
    with torch.no_grad():
        for batch in dataloader:
            waveforms = batch['waveform'].to(device)
            noise_labels_batch = batch['noise_label'].to(device)
            cluster_labels_batch = batch['cluster_label'].to(device)
            is_spike = batch['is_spike']
            
            # 提取中间层特征
            noise_features, label_features = model.get_intermediate_features(waveforms)
            
            # 获取预测结果
            noise_output, label_output = model(waveforms)
            noise_pred = torch.argmax(noise_output, dim=1)
            label_pred = torch.argmax(label_output, dim=1)
            noise_gt = torch.argmax(noise_labels_batch, dim=1)
            label_gt = torch.argmax(cluster_labels_batch, dim=1)
            
            noise_features_list.append(noise_features.cpu().numpy())
            label_features_list.append(label_features.cpu().numpy())
            noise_pred_list.append(noise_pred.cpu().numpy())
            label_pred_list.append(label_pred.cpu().numpy())
            noise_gt_list.append(noise_gt.cpu().numpy())
            label_gt_list.append(label_gt.cpu().numpy())
            spike_mask_list.append(is_spike.numpy())
    
    # 合并所有batch的结果
    noise_features_all = np.vstack(noise_features_list)
    label_features_all = np.vstack(label_features_list)
    noise_pred_all = np.concatenate(noise_pred_list)
    label_pred_all = np.concatenate(label_pred_list)
    noise_gt_all = np.concatenate(noise_gt_list)
    label_gt_all = np.concatenate(label_gt_list)
    spike_mask_all = np.concatenate(spike_mask_list)
    
    return {
        'noise_features': noise_features_all,
        'label_features': label_features_all,
        'noise_pred': noise_pred_all,
        'label_pred': label_pred_all,
        'noise_gt': noise_gt_all,
        'label_gt': label_gt_all,
        'spike_mask': spike_mask_all,
        'unique_clusters': unique_clusters
    }

def plot_umap_visualization(features_dict, output_dir, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    绘制4个UMAP可视化图
    1. 有noise的GT noise分类器结果
    2. 有noise的predicted noise分类器结果
    3. 只包含spike的GT label分类器结果
    4. 只包含spike的predicted label分类器结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    noise_features = features_dict['noise_features']
    label_features = features_dict['label_features']
    noise_pred = features_dict['noise_pred']
    label_pred = features_dict['label_pred']
    noise_gt = features_dict['noise_gt']
    label_gt = features_dict['label_gt']
    spike_mask = features_dict['spike_mask']
    
    print("[INFO] Computing UMAP embeddings...")
    
    # 1. UMAP for noise features (all samples) - GT
    print("[INFO] Computing UMAP for noise features (GT)...")
    reducer_noise = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    noise_embedding = reducer_noise.fit_transform(noise_features)
    
    # 2. UMAP for noise features (all samples) - Predicted
    # 使用相同的reducer（已经fit过了）
    noise_embedding_pred = noise_embedding  # 使用相同的embedding
    
    # 3. UMAP for label features (spike samples only) - GT
    print("[INFO] Computing UMAP for label features (GT, spike only)...")
    label_features_spike = label_features[spike_mask]
    label_gt_spike = label_gt[spike_mask]
    label_pred_spike = label_pred[spike_mask]
    
    # 过滤掉无效的cluster（-1）
    valid_mask_spike = label_gt_spike >= 0
    label_features_spike_valid = label_features_spike[valid_mask_spike]
    label_gt_spike_valid = label_gt_spike[valid_mask_spike]
    label_pred_spike_valid = label_pred_spike[valid_mask_spike]
    
    reducer_label = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    label_embedding = reducer_label.fit_transform(label_features_spike_valid)
    
    # 4. UMAP for label features (spike samples only) - Predicted
    # 使用相同的embedding（因为特征相同）
    label_embedding_pred = label_embedding
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('UMAP Visualization of Model Features', fontsize=16, fontweight='bold')
    
    # 图1: Noise分类器 - GT (所有样本)
    ax1 = axes[0, 0]
    # map numeric labels to categorical colors/strings
    noise_labels_str = np.where(noise_gt == 1, 'spike', 'noise')
    noise_colors = np.where(noise_gt == 1, '#ff7f0e', '#d3d3d3')  # orange for spike, light gray for noise
    ax1.scatter(noise_embedding[:, 0], noise_embedding[:, 1], c=noise_colors, s=2, alpha=0.8)
    ax1.set_title('Noise Classifier Features - Ground Truth\n(All samples)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('UMAP 1', fontsize=10)
    ax1.set_ylabel('UMAP 2', fontsize=10)
    # legend for categorical labels
    legend_handles = [mpatches.Patch(color='#d3d3d3', label='noise'), mpatches.Patch(color='#ff7f0e', label='spike')]
    ax1.legend(handles=legend_handles, loc='best', fontsize=8)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal', adjustable='box')
    
    # 图2: Noise分类器 - Predicted (所有样本)
    ax2 = axes[0, 1]
    noise_pred_str = np.where(noise_pred == 1, 'spike', 'noise')
    noise_pred_colors = np.where(noise_pred == 1, '#ff7f0e', '#d3d3d3')
    ax2.scatter(noise_embedding_pred[:, 0], noise_embedding_pred[:, 1], c=noise_pred_colors, s=2, alpha=0.8)
    ax2.set_title('Noise Classifier Features - Predicted\n(All samples)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('UMAP 1', fontsize=10)
    ax2.set_ylabel('UMAP 2', fontsize=10)
    legend_handles_pred = [mpatches.Patch(color='#d3d3d3', label='noise'), mpatches.Patch(color='#ff7f0e', label='spike')]
    ax2.legend(handles=legend_handles_pred, loc='best', fontsize=8)
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect('equal', adjustable='box')
    
    # 图3: Label分类器 - GT (只包含spike)
    ax3 = axes[1, 0]
    # 为不同的cluster使用不同颜色
    unique_labels_gt = np.unique(label_gt_spike_valid)
    unique_labels_gt = unique_labels_gt[unique_labels_gt >= 0]  # 确保没有-1
    if len(unique_labels_gt) > 20:
        # 如果cluster太多，使用tab20b和tab20c
        colors_gt = list(plt.cm.tab20(np.linspace(0, 1, 20))) + \
                    list(plt.cm.tab20b(np.linspace(0, 1, min(20, len(unique_labels_gt)-20))))
    else:
        colors_gt = plt.cm.tab20(np.linspace(0, 1, len(unique_labels_gt)))
    
    for i, label in enumerate(unique_labels_gt):
        mask = label_gt_spike_valid == label
        if np.sum(mask) > 0:
            ax3.scatter(label_embedding[mask, 0], label_embedding[mask, 1], 
                       c=[colors_gt[i % len(colors_gt)]], label=f'Cluster {label}', s=2, alpha=0.8)
    ax3.set_title('Label Classifier Features - Ground Truth\n(Spike samples only)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('UMAP 1', fontsize=10)
    ax3.set_ylabel('UMAP 2', fontsize=10)
    if len(unique_labels_gt) <= 30:
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=3, ncol=1)
    ax3.grid(True, alpha=0.3)
    
    # 图4: Label分类器 - Predicted (只包含spike)
    ax4 = axes[1, 1]
    unique_labels_pred = np.unique(label_pred_spike_valid)
    unique_labels_pred = unique_labels_pred[unique_labels_pred >= 0]  # 确保没有-1
    if len(unique_labels_pred) > 20:
        colors_pred = list(plt.cm.tab20(np.linspace(0, 1, 20))) + \
                     list(plt.cm.tab20b(np.linspace(0, 1, min(20, len(unique_labels_pred)-20))))
    else:
        colors_pred = plt.cm.tab20(np.linspace(0, 1, len(unique_labels_pred)))
    
    for i, label in enumerate(unique_labels_pred):
        mask = label_pred_spike_valid == label
        if np.sum(mask) > 0:
            ax4.scatter(label_embedding_pred[mask, 0], label_embedding_pred[mask, 1], 
                       c=[colors_pred[i % len(colors_pred)]], label=f'Cluster {label}', s=2, alpha=0.8)
    ax4.set_title('Label Classifier Features - Predicted\n(Spike samples only)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('UMAP 1', fontsize=10)
    ax4.set_ylabel('UMAP 2', fontsize=10)
    if len(unique_labels_pred) <= 30:
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=3, ncol=1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片为 PDF
    output_path = os.path.join(output_dir, 'umap_visualization.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"[INFO] UMAP visualization saved to {output_path}")
    
    plt.close()
    
    # 打印统计信息
    print("\n[INFO] Visualization Statistics:")
    print(f"  Total samples: {len(noise_features):,}")
    print(f"  Noise samples (GT): {np.sum(noise_gt == 0):,}")
    print(f"  Spike samples (GT): {np.sum(noise_gt == 1):,}")
    print(f"  Noise samples (Pred): {np.sum(noise_pred == 0):,}")
    print(f"  Spike samples (Pred): {np.sum(noise_pred == 1):,}")
    print(f"  Unique clusters (GT, valid): {len(unique_labels_gt)}")
    print(f"  Unique clusters (Pred, valid): {len(unique_labels_pred)}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model features using UMAP')
    parser.add_argument('--noise_model_path', type=str, required=True,
                        help='Path to the saved noise classifier (.pth file)')
    parser.add_argument('--label_model_path', type=str, required=True,
                        help='Path to the saved label/classifier (.pth file)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the preprocessed data (.pkl file)')
    parser.add_argument('--cluster_mapping_path', type=str, required=True,
                        help='Path to cluster mapping JSON file')
    parser.add_argument('--output_dir', type=str, default='./umap_visualization',
                        help='Output directory for visualization')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Maximum number of samples to use (for speed)')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP min_dist parameter')
    
    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    # 加载cluster映射
    with open(args.cluster_mapping_path, 'r') as f:
        cluster_mapping = json.load(f)
    num_classes = len(cluster_mapping['cluster_to_index'])
    print(f"[INFO] Number of classes: {num_classes}")
    
    # 加载数据
    print(f"[INFO] Loading data from {args.data_path}")
    df = load_data(args.data_path)
    print(f"[INFO] Loaded {len(df):,} samples")
    
    # 加载模型（分别加载 noise 和 label 两个权重）
    print(f"[INFO] Loading noise model from {args.noise_model_path} and label model from {args.label_model_path}")
    input_size = 30 * 30  # waveform size
    model = load_model(args.noise_model_path, args.label_model_path, input_size, num_classes, device)
    print("[INFO] Model loaded successfully")
    
    # 提取特征
    features_dict = extract_features(model, df, device, max_samples=args.max_samples)
    
    # 绘制UMAP可视化
    plot_umap_visualization(
        features_dict, 
        args.output_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )
    
    print("\n[INFO] Visualization completed!")

if __name__ == "__main__":
    main()

