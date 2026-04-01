#!/usr/bin/env python3
"""
基于偏好轴对图像排序并绘制热图。

对于每个神经元：
1. 计算每个图像在偏好轴上的投影值
2. 按投影值对图像排序
3. 使用排序后的神经响应值形成热图矩阵

支持按脑区分组绘制，并绘制偏好轴方向和正交方向的heatmap。
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免X11显示问题
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from analyze_neuron_preferences import load_features_csv


def load_preferred_axes(axes_path: str) -> np.ndarray:
    """加载偏好轴矩阵，形状 (n_neurons, n_features)"""
    return np.load(axes_path)


def compute_orthogonal_axis(preferred_axis: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    计算与偏好轴正交的方向。
    
    参数:
        preferred_axis: 偏好轴向量，形状 (n_features,)
        X: 特征矩阵，形状 (n_samples, n_features)，用于计算PCA
    
    返回:
        正交轴向量，形状 (n_features,)，已归一化
    """
    n_features = preferred_axis.shape[0]
    
    # 归一化偏好轴
    pref_norm = np.linalg.norm(preferred_axis)
    if pref_norm < 1e-10:
        # 如果偏好轴为零向量，返回第一个标准基向量
        orth_axis = np.zeros(n_features)
        orth_axis[0] = 1.0
        return orth_axis
    
    preferred_axis_norm = preferred_axis / pref_norm
    
    # 计算PCA基（使用前30个主成分）
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    n_components = min(30, vh.shape[0])
    vh_subset = vh[:n_components]  # 形状 (n_components, n_features)
    
    # 将偏好轴投影到PCA空间
    pref_in_pca = preferred_axis_norm @ vh_subset.T  # 形状 (n_components,)
    
    # 在PCA空间中找与pref_in_pca正交的方向
    # 使用Gram-Schmidt过程：从标准基向量开始
    orth_in_pca = None
    for i in range(n_components):
        # 尝试第i个标准基向量
        candidate = np.zeros(n_components)
        candidate[i] = 1.0
        
        # 投影到正交补空间
        proj = np.dot(candidate, pref_in_pca) * pref_in_pca
        orth_candidate = candidate - proj
        
        # 归一化
        orth_norm = np.linalg.norm(orth_candidate)
        if orth_norm > 1e-8:
            orth_in_pca = orth_candidate / orth_norm
            break
    
    # 如果没找到合适的，使用第一个主成分
    if orth_in_pca is None:
        orth_in_pca = np.zeros(n_components)
        orth_in_pca[0] = 1.0
        # 确保正交
        proj = np.dot(orth_in_pca, pref_in_pca) * pref_in_pca
        orth_in_pca = orth_in_pca - proj
        orth_norm = np.linalg.norm(orth_in_pca)
        if orth_norm > 1e-8:
            orth_in_pca = orth_in_pca / orth_norm
        else:
            # 如果还是不行，使用第二个主成分
            orth_in_pca = np.zeros(n_components)
            orth_in_pca[1] = 1.0
    
    # 转换回原始特征空间
    orth_axis = orth_in_pca @ vh_subset  # 形状 (n_features,)
    
    # 归一化
    orth_norm = np.linalg.norm(orth_axis)
    if orth_norm > 1e-10:
        orth_axis = orth_axis / orth_norm
    else:
        # 如果转换后为零向量，使用第一个特征维度
        orth_axis = np.zeros(n_features)
        orth_axis[0] = 1.0
    
    return orth_axis


def compute_sorted_responses(
    X: np.ndarray,
    y_all: np.ndarray,
    direction_vectors: np.ndarray,
    center_features: bool = True,
) -> np.ndarray:
    """
    对每个神经元，按照给定方向对图像排序，返回排序后的响应矩阵。
    
    参数:
        X: 特征矩阵，形状 (n_samples, n_features)
        y_all: 神经响应矩阵，形状 (n_samples, n_neurons)
        direction_vectors: 方向向量矩阵，形状 (n_neurons, n_features)
        center_features: 是否对特征进行中心化
    
    返回:
        排序后的响应矩阵，形状 (n_neurons, n_samples)
    """
    n_samples, n_features = X.shape
    n_neurons = direction_vectors.shape[0]
    
    if center_features:
        X_mean = X.mean(axis=0, keepdims=True)
        Xc = X - X_mean
    else:
        Xc = X
    
    sorted_responses = np.zeros((n_neurons, n_samples), dtype=y_all.dtype)
    
    print(f"对 {n_neurons} 个神经元按方向向量排序...")
    for neuron_idx in range(n_neurons):
        # 计算每个图像在方向向量上的投影
        projection = Xc @ direction_vectors[neuron_idx]  # 形状 (n_samples,)
        
        # 按投影值排序（从小到大）
        sort_indices = np.argsort(projection)
        
        # 使用排序索引重新排列该神经元的响应
        sorted_responses[neuron_idx] = y_all[sort_indices, neuron_idx]
        
        if (neuron_idx + 1) % 50 == 0 or neuron_idx == n_neurons - 1:
            print(f"  完成 {neuron_idx + 1}/{n_neurons} 个神经元")
    
    return sorted_responses


def plot_heatmap(
    sorted_responses: np.ndarray,
    output_path: str,
    figsize: tuple = None,
    cmap: str = "viridis",
    aspect: str = "auto",
    dpi: int = 300,
    vmin: float = None,
    vmax: float = None,
    title: str = None,
):
    """
    绘制热图。
    
    参数:
        sorted_responses: 排序后的响应矩阵，形状 (n_neurons, n_samples)
        output_path: 输出图像路径
        figsize: 图像大小 (宽度, 高度)，如果为None则自动计算
        cmap: 颜色映射
        aspect: 图像宽高比
        dpi: 分辨率
        vmin, vmax: 颜色范围，如果为None则使用数据的最小/最大值
        title: 图像标题
    """
    n_neurons, n_samples = sorted_responses.shape
    
    if figsize is None:
        # 根据数据维度自动计算合适的大小
        # 保持合理的宽高比
        width = 12
        height = 8
        figsize = (width, height)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 如果未指定颜色范围，使用数据的百分位数来确定范围（避免异常值影响）
    if vmin is None:
        vmin = np.percentile(sorted_responses, 1)
    if vmax is None:
        vmax = np.percentile(sorted_responses, 99)
    
    im = ax.imshow(
        sorted_responses,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect=aspect,
    )
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Response')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_heatmap_to_ax(
    ax,
    sorted_responses: np.ndarray,
    cmap: str = "viridis",
    vmin: float = None,
    vmax: float = None,
    title: str = None,
):
    """
    在给定的axes上绘制热图。
    
    参数:
        ax: matplotlib axes对象
        sorted_responses: 排序后的响应矩阵，形状 (n_neurons, n_samples)
        cmap: 颜色映射
        vmin, vmax: 颜色范围
        title: 图像标题
    """
    if vmin is None:
        vmin = np.percentile(sorted_responses, 1)
    if vmax is None:
        vmax = np.percentile(sorted_responses, 99)
    
    im = ax.imshow(
        sorted_responses,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    return im


def plot_region_heatmaps_to_pdf(
    X: np.ndarray,
    y_all: np.ndarray,
    preferred_axes: np.ndarray,
    region_ranges: dict,
    output_path: str,
    center_features: bool = True,
    cmap: str = "viridis",
    dpi: int = 300,
):
    """
    按脑区分组绘制偏好轴方向和正交方向的heatmap，保存到PDF。
    
    参数:
        X: 特征矩阵，形状 (n_samples, n_features)
        y_all: 神经响应矩阵，形状 (n_samples, n_neurons)
        preferred_axes: 偏好轴矩阵，形状 (n_neurons, n_features)
        region_ranges: 脑区索引范围字典，格式: {'V1': (0, 270), 'V4': (271, 395), 'IT': (396, 502)}
        output_path: 输出PDF路径
        center_features: 是否对特征进行中心化
        cmap: 颜色映射
        dpi: 分辨率
    """
    n_samples, n_features = X.shape
    
    # 计算全局颜色范围（用于所有heatmap使用相同的颜色范围）
    global_vmin = np.percentile(y_all, 1)
    global_vmax = np.percentile(y_all, 99)
    
    with PdfPages(output_path) as pdf:
        for region_name, (start_idx, end_idx) in region_ranges.items():
            print(f"\n处理脑区: {region_name} (yindex: {start_idx}-{end_idx})")
            
            # 提取该脑区的数据
            region_neurons = list(range(start_idx, end_idx + 1))
            region_preferred_axes = preferred_axes[region_neurons]  # (n_region_neurons, n_features)
            region_y_all = y_all[:, region_neurons]  # (n_samples, n_region_neurons)
            
            # 计算正交轴
            print(f"  计算 {len(region_neurons)} 个神经元的正交轴...")
            region_orthogonal_axes = np.zeros_like(region_preferred_axes)
            for i, neuron_idx in enumerate(region_neurons):
                region_orthogonal_axes[i] = compute_orthogonal_axis(
                    preferred_axes[neuron_idx], X
                )
                if (i + 1) % 50 == 0 or i == len(region_neurons) - 1:
                    print(f"    完成 {i + 1}/{len(region_neurons)} 个神经元")
            
            # 计算偏好轴方向的排序响应
            print(f"  计算偏好轴方向的排序响应...")
            sorted_responses_pref = compute_sorted_responses(
                X, region_y_all, region_preferred_axes, center_features=center_features
            )
            
            # 计算正交方向的排序响应
            print(f"  计算正交方向的排序响应...")
            sorted_responses_orth = compute_sorted_responses(
                X, region_y_all, region_orthogonal_axes, center_features=center_features
            )
            
            # 创建包含两个子图的figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi)
            
            # 绘制偏好轴方向的heatmap
            plot_heatmap_to_ax(
                axes[0],
                sorted_responses_pref,
                cmap=cmap,
                vmin=global_vmin,
                vmax=global_vmax,
                title=f"{region_name} - Preferred Axis Direction"
            )
            axes[0].set_xlabel('Images (sorted by projection)', fontsize=10)
            axes[0].set_ylabel('Neurons', fontsize=10)
            
            # 绘制正交方向的heatmap
            plot_heatmap_to_ax(
                axes[1],
                sorted_responses_orth,
                cmap=cmap,
                vmin=global_vmin,
                vmax=global_vmax,
                title=f"{region_name} - Orthogonal Direction"
            )
            axes[1].set_xlabel('Images (sorted by projection)', fontsize=10)
            axes[1].set_ylabel('Neurons', fontsize=10)
            
            plt.suptitle(f"Region: {region_name} ({len(region_neurons)} neurons)", 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            
            print(f"  {region_name} 的heatmap已添加到PDF")
    
    print(f"\n所有heatmap已保存到: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="基于偏好轴排序绘制神经元响应热图")
    parser.add_argument(
        "--feature-path",
        default="/media/ubuntu/sda/Monkey/image_feature/image_features_756d_clip_vitl14_F.csv",
        help="特征 CSV 路径",
    )
    parser.add_argument(
        "--response-path",
        default="/media/ubuntu/sda/Monkey/image_feature/train_MUA_MonkeyF.npy",
        help="神经响应 npy 路径",
    )
    parser.add_argument(
        "--axes-path",
        default="/media/ubuntu/sda/Monkey/image_feature/neuron_preferred_axes_results_axes.npy",
        help="偏好轴 npy 路径",
    )
    parser.add_argument(
        "--output",
        default="neuron_response_heatmap.png",
        help="输出热图路径",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="颜色映射（如 'viridis', 'plasma', 'coolwarm', 'RdYlBu_r' 等）",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=-1,
        help="仅绘制前 N 个神经元，-1 表示全部",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="颜色范围最小值（默认使用数据1%%分位数）",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="颜色范围最大值（默认使用数据99%%分位数）",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="不对特征进行中心化",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="随机抽取的图像数量，-1 表示使用全部图像",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于图像抽样）",
    )
    parser.add_argument(
        "--by-region",
        action="store_true",
        help="按脑区分组绘制heatmap（偏好轴方向和正交方向）",
    )
    parser.add_argument(
        "--region-ranges",
        type=str,
        default="V1:0-270,V4:271-395,IT:396-502",
        help="脑区索引范围，格式: 'V1:0-270,V4:271-395,IT:396-502'",
    )
    return parser.parse_args()


def parse_region_ranges(region_ranges_str: str) -> dict:
    """
    解析脑区索引范围字符串。
    
    参数:
        region_ranges_str: 格式为 'V1:0-270,V4:271-395,IT:396-502' 的字符串
    
    返回:
        字典，格式: {'V1': (0, 270), 'V4': (271, 395), 'IT': (396, 502)}
    """
    region_ranges = {}
    for part in region_ranges_str.split(','):
        part = part.strip()
        if ':' in part:
            region_name, range_str = part.split(':', 1)
            region_name = region_name.strip()
            if '-' in range_str:
                start_str, end_str = range_str.split('-', 1)
                start_idx = int(start_str.strip())
                end_idx = int(end_str.strip())
                region_ranges[region_name] = (start_idx, end_idx)
    return region_ranges


def main():
    args = parse_args()
    
    print("加载数据...")
    X = load_features_csv(args.feature_path, dtype=np.float32)
    y_all = np.load(args.response_path)
    preferred_axes = load_preferred_axes(args.axes_path)
    
    # 检查维度一致性
    if X.shape[0] != y_all.shape[0]:
        raise ValueError(f"特征与响应图片数不一致: {X.shape[0]} vs {y_all.shape[0]}")
    
    if preferred_axes.shape[0] != y_all.shape[1]:
        raise ValueError(
            f"偏好轴与响应的神经元数不一致: {preferred_axes.shape[0]} vs {y_all.shape[1]}"
        )
    
    if preferred_axes.shape[1] != X.shape[1]:
        raise ValueError(
            f"偏好轴与特征的特征维度不一致: {preferred_axes.shape[1]} vs {X.shape[1]}"
        )
    
    # 如果指定了最大神经元数，进行截取
    n_neurons = preferred_axes.shape[0]
    if args.max_neurons > 0:
        n_neurons = min(args.max_neurons, n_neurons)
        preferred_axes = preferred_axes[:n_neurons]
        y_all = y_all[:, :n_neurons]
    
    # 如果指定了最大图像数，进行随机抽样
    n_samples = X.shape[0]
    image_indices = np.arange(n_samples)
    if args.max_images > 0 and args.max_images < n_samples:
        rng = np.random.default_rng(args.seed)
        image_indices = rng.choice(n_samples, size=args.max_images, replace=False)
        image_indices = np.sort(image_indices)  # 排序以保持一致性
        X = X[image_indices]
        y_all = y_all[image_indices]
        print(f"随机抽取 {args.max_images} 个图像（随机种子: {args.seed}）")

    # 如果指定了按脑区绘制
    if args.by_region:
        region_ranges = parse_region_ranges(args.region_ranges)
        print(f"\n按脑区绘制heatmap，脑区范围: {region_ranges}")
        
        # 确保输出是PDF格式
        if not args.output.endswith('.pdf'):
            args.output = args.output.rsplit('.', 1)[0] + '.pdf'
        
        plot_region_heatmaps_to_pdf(
            X,
            y_all,
            preferred_axes,
            region_ranges,
            args.output,
            center_features=not args.no_center,
            cmap=args.cmap,
            dpi=300,
        )
    else:
        # 原有的单heatmap绘制逻辑
        sorted_responses = compute_sorted_responses(
            X,
            y_all,
            preferred_axes,
            center_features=not args.no_center,
        )
        
        # 绘制热图
        plot_heatmap(
            sorted_responses,
            args.output,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        
        # 可选：保存排序后的响应矩阵
        np.save(args.output.replace(".png", "_sorted_responses.npy"), sorted_responses)
        print(f"排序后的响应矩阵已保存到: {args.output.replace('.png', '_sorted_responses.npy')}")


if __name__ == "__main__":
    main()

