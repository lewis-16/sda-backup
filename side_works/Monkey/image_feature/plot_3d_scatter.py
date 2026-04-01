#!/usr/bin/env python3
"""
为每个 corr > 0.5 的神经元绘制3维散点图，根据反应强度着色。

使用image_feature的全局坐标体系（PCA前3个主成分），所有神经元使用相同的坐标空间，
根据每个神经元对不同图像的反应强度进行着色。
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免X11显示问题
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from analyze_neuron_preferences import load_features_csv, compute_pca_basis, project_to_subspace


def load_correlations(csv_path: str) -> np.ndarray:
    """
    从CSV文件加载预测相关性数据。
    
    参数:
        csv_path: CSV文件路径
    
    返回:
        相关性数组，形状 (n_neurons,)
    """
    corrs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            corrs.append(float(row['prediction_correlation']))
    return np.array(corrs)


def plot_3d_scatter(
    coords: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    neuron_id: int = None,
    corr: float = None,
    title: str = None,
    cmap: str = 'viridis',
    figsize: tuple = (12, 10),
    dpi: int = 300,
    alpha: float = 0.6,
    s: float = 20,
):
    """
    绘制3维散点图。
    
    参数:
        coords: 3维坐标，形状 (n_images, 3)，每个点代表一张图像
        colors: 颜色值，形状 (n_images,)，每个点的反应强度
        output_path: 输出图像路径
        neuron_id: 神经元ID
        corr: 相关性值
        title: 图像标题（如果为None则自动生成）
        cmap: 颜色映射
        figsize: 图像大小
        dpi: 分辨率
        alpha: 点的透明度
        s: 点的大小
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建散点图
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=colors,
        cmap=cmap,
        alpha=alpha,
        s=s,
        edgecolors='black',
        linewidths=0.3,
    )
    
    # 设置标签
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_zlabel('Dimension 3', fontsize=12)
    
    # 生成标题
    if title is None:
        if neuron_id is not None and corr is not None:
            title = f"Neuron {neuron_id} (corr = {corr:.3f})"
        elif neuron_id is not None:
            title = f"Neuron {neuron_id}"
        else:
            title = "3D Scatter Plot"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Response Strength', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="绘制3维散点图，显示 corr > 0.5 的神经元")
    parser.add_argument(
        "--corr-csv",
        default="/media/ubuntu/sda/Monkey/image_feature/neuron_preferred_axes_results.csv",
        help="相关性CSV文件路径",
    )
    parser.add_argument(
        "--feature-path",
        default="/media/ubuntu/sda/Monkey/image_feature/image_features_756d_clip_vitl14_F.csv",
        help="特征CSV文件路径",
    )
    parser.add_argument(
        "--response-path",
        default="/media/ubuntu/sda/Monkey/image_feature/train_MUA_MonkeyF.npy",
        help="神经响应npy文件路径",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="PCA主成分数量（用于3维坐标）",
    )
    parser.add_argument(
        "--output-dir",
        default="neuron_3d_scatter_plots",
        help="输出图像目录",
    )
    parser.add_argument(
        "--output-pattern",
        default="neuron_{neuron_id:03d}_corr_{corr:.3f}.png",
        help="输出文件名模式，可使用 {neuron_id} 和 {corr} 占位符",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.5,
        help="相关性阈值，只绘制 corr > threshold 的神经元",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="颜色映射（如 'viridis', 'plasma', 'coolwarm', 'RdYlBu_r' 等）",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 10],
        help="图像大小（宽度，高度）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="分辨率",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="点的透明度（0-1）",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=50,
        help="点的大小",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("加载数据...")
    # 加载相关性数据
    corrs = load_correlations(args.corr_csv)
    print(f"  加载了 {len(corrs)} 个神经元的相关性数据")
    
    # 加载image_feature数据
    X = load_features_csv(args.feature_path, dtype=np.float32)
    print(f"  加载了特征数据，形状: {X.shape}")
    
    # 加载神经响应数据
    y_all = np.load(args.response_path)
    print(f"  加载了神经响应数据，形状: {y_all.shape}")
    
    # 检查维度一致性
    n_neurons = len(corrs)
    if X.shape[0] != y_all.shape[0]:
        raise ValueError(
            f"特征与响应图片数不一致: {X.shape[0]} vs {y_all.shape[0]}"
        )
    if y_all.shape[1] != n_neurons:
        raise ValueError(
            f"响应数据与相关性数据的神经元数不一致: {y_all.shape[1]} vs {n_neurons}"
        )
    
    # 计算全局的PCA坐标体系（所有图像共享）
    print(f"\n计算全局PCA坐标体系（{args.pca_components}个主成分）...")
    pc_basis = compute_pca_basis(X, n_components=args.pca_components)
    global_coords = project_to_subspace(X, pc_basis)  # 形状 (n_samples, 3)
    print(f"  全局坐标形状: {global_coords.shape}")
    
    # 筛选 corr > threshold 的神经元
    mask = corrs > args.corr_threshold
    selected_indices = np.where(mask)[0]
    n_selected = len(selected_indices)
    print(f"\n筛选 corr > {args.corr_threshold} 的神经元: {n_selected}/{n_neurons}")
    
    if n_selected == 0:
        print(f"警告: 没有找到 corr > {args.corr_threshold} 的神经元！")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出目录: {args.output_dir}")
    
    # 为每个符合条件的神经元绘制一张图
    print(f"\n开始绘制 {n_selected} 个神经元的3维散点图...")
    print("  所有神经元使用相同的全局坐标体系（image_feature的PCA坐标）")
    for i, neuron_idx in enumerate(selected_indices):
        corr = corrs[neuron_idx]
        
        # 所有神经元使用相同的全局坐标
        # 获取该神经元对所有图像的响应
        neuron_responses = y_all[:, neuron_idx]  # 形状 (n_samples,)
        
        # 生成输出文件名
        output_filename = args.output_pattern.format(
            neuron_id=neuron_idx,
            corr=corr
        )
        output_path = os.path.join(args.output_dir, output_filename)
        
        # 绘制3维散点图
        plot_3d_scatter(
            global_coords,  # 使用全局坐标
            neuron_responses,
            output_path,
            neuron_id=neuron_idx,
            corr=corr,
            cmap=args.cmap,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            alpha=args.alpha,
            s=args.point_size,
        )
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{n_selected}")
    
    print(f"\n完成！所有图像已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

