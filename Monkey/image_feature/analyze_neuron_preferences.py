#!/usr/bin/env python3
"""
基于 prompt.txt 要求的岭回归偏好轴分析，适配本地数据：
- 特征文件: /Users/jin/Downloads/image_feature/image_features_756d_clip_vitl14_F.csv
  形状 (22248, 760)，首列是图像路径，后 760 列是特征
- 神经响应: /Users/jin/Downloads/image_feature/train_MUA_MonkeyF.npy
  形状 (22248, n_neurons)，此处 n_neurons=503

仅依赖 numpy，可选通过 argparse 配置；默认使用 10 折外部 CV + 5 折内部 CV
对每个神经元选择最优 alpha，并计算预测相关性与偏好轴。
"""

import argparse
import csv
import os
from typing import Iterable, List, Tuple

import numpy as np


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """返回皮尔逊相关，若方差为 0 则返回 0."""
    a = np.asarray(a)
    b = np.asarray(b)
    a_std = a.std()
    b_std = b.std()
    if a_std == 0 or b_std == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ridge_weights(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """计算岭回归权重，显式中心化，返回权重."""
    X_mean = X.mean(axis=0, keepdims=True)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean
    # (X^T X + alpha I) w = X^T y
    xtx = Xc.T @ Xc
    xty = Xc.T @ yc
    reg = alpha * np.eye(X.shape[1], dtype=X.dtype)
    w = np.linalg.solve(xtx + reg, xty)
    # 将截距吸收进 y_mean 以方便预测：y_pred = (X - X_mean) @ w + y_mean
    return w, X_mean, y_mean


def predict(X: np.ndarray, w: np.ndarray, X_mean: np.ndarray, y_mean: float) -> np.ndarray:
    """使用中心化后的权重进行预测."""
    return (X - X_mean) @ w + y_mean


def kfold_indices(n_samples: int, n_splits: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
    """生成 K 折索引 (train_idx, test_idx)."""
    indices = np.arange(n_samples)
    perm = rng.permutation(indices)
    folds = np.array_split(perm, n_splits)
    splits = []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i + 1 :])
        splits.append((train_idx, test_idx))
    return splits


def select_alpha_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Iterable[float],
    n_splits: int,
    rng: np.random.Generator,
) -> float:
    """在训练集上做内部 CV，选出使相关性均值最大的 alpha."""
    splits = kfold_indices(len(y), n_splits, rng)
    alpha_scores = []
    for alpha in alphas:
        fold_corrs = []
        for train_idx, val_idx in splits:
            w, X_mean, y_mean = ridge_weights(X[train_idx], y[train_idx], alpha)
            y_pred = predict(X[val_idx], w, X_mean, y_mean)
            fold_corrs.append(pearson_corr(y[val_idx], y_pred))
        alpha_scores.append((np.mean(fold_corrs), alpha))
    alpha_scores.sort(key=lambda x: (-x[0], x[1]))
    return alpha_scores[0][1]


def compute_pca_basis(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """使用 SVD 计算前 n_components 个主成分基 (shape: n_components x n_features)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # 仅保留前 n_components 的右奇异向量
    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    return vh[:n_components]


def build_3d_subspace_in_30d(preferred_axis: np.ndarray, X: np.ndarray, pc30_basis: np.ndarray) -> np.ndarray:
    """在30维PCA空间中构建包含偏好轴的3维子空间基。
    
    参数:
        preferred_axis: 偏好轴向量（原始特征空间），形状 (n_features,)
        X: 特征矩阵，形状 (n_samples, n_features)
        pc30_basis: 30维PCA基，形状 (30, n_features)
    
    返回:
        3维子空间基（原始特征空间），形状 (3, n_features)，每行是一个单位向量
        - 第0行：偏好轴在30维空间中的投影方向（归一化）
        - 第1行：30维空间中与偏好轴正交的第一个主要方向
        - 第2行：30维空间中与偏好轴和前一个方向都正交的第二个方向
    """
    n_features = preferred_axis.shape[0]
    
    # 将偏好轴投影到30维PCA空间
    preferred_axis_30d = preferred_axis @ pc30_basis.T  # 形状 (30,)
    
    axis_norm_30d = np.linalg.norm(preferred_axis_30d)
    if axis_norm_30d < 1e-10:
        # 如果偏好轴在30维空间中为零向量，使用30维空间的前3个主成分
        basis_30d = np.zeros((3, 30), dtype=preferred_axis.dtype)
        basis_30d[0, 0] = 1.0
        basis_30d[1, 1] = 1.0
        basis_30d[2, 2] = 1.0
    else:
        # 第一个方向：偏好轴在30维空间中的归一化方向
        v1_30d = preferred_axis_30d / axis_norm_30d
        
        # 在30维空间中找与v1_30d正交的方向
        # 使用30维标准基向量（单位向量），通过Gram-Schmidt过程
        # 这样可以确保找到与偏好轴正交的方向
        v2_candidates_30d = []
        for i in range(30):
            candidate_30d = np.zeros(30, dtype=preferred_axis.dtype)
            candidate_30d[i] = 1.0
            # 投影到v1_30d的正交补空间
            proj_30d = candidate_30d - np.dot(candidate_30d, v1_30d) * v1_30d
            proj_norm = np.linalg.norm(proj_30d)
            if proj_norm > 1e-8:
                v2_candidates_30d.append((proj_norm, proj_30d / proj_norm))
        
        if not v2_candidates_30d:
            # 如果没有找到合适的候选，使用第二个标准基向量
            v2_30d = np.zeros(30, dtype=preferred_axis.dtype)
            v2_30d[1] = 1.0
            v2_30d = v2_30d - np.dot(v2_30d, v1_30d) * v1_30d
            v2_norm = np.linalg.norm(v2_30d)
            if v2_norm > 1e-8:
                v2_30d = v2_30d / v2_norm
            else:
                v2_30d[0] = 1.0
                v2_30d = v2_30d - np.dot(v2_30d, v1_30d) * v1_30d
                v2_30d = v2_30d / np.linalg.norm(v2_30d)
        else:
            # 选择投影长度最大的候选（最正交的方向）
            v2_candidates_30d.sort(key=lambda x: -x[0])
            v2_30d = v2_candidates_30d[0][1]
        
        # 第三个方向：与v1_30d和v2_30d都正交
        v3_candidates_30d = []
        for i in range(30):
            candidate_30d = np.zeros(30, dtype=preferred_axis.dtype)
            candidate_30d[i] = 1.0
            # 投影到v1_30d和v2_30d的正交补空间
            proj_30d = candidate_30d - np.dot(candidate_30d, v1_30d) * v1_30d - np.dot(candidate_30d, v2_30d) * v2_30d
            proj_norm = np.linalg.norm(proj_30d)
            if proj_norm > 1e-8:
                v3_candidates_30d.append((proj_norm, proj_30d / proj_norm))
        
        if not v3_candidates_30d:
            # 如果没有找到合适的候选，使用第三个标准基向量
            v3_30d = np.zeros(30, dtype=preferred_axis.dtype)
            v3_30d[2] = 1.0
            v3_30d = v3_30d - np.dot(v3_30d, v1_30d) * v1_30d - np.dot(v3_30d, v2_30d) * v2_30d
            v3_norm = np.linalg.norm(v3_30d)
            if v3_norm > 1e-8:
                v3_30d = v3_30d / v3_norm
            else:
                v3_30d[0] = 1.0
                v3_30d = v3_30d - np.dot(v3_30d, v1_30d) * v1_30d - np.dot(v3_30d, v2_30d) * v2_30d
                v3_30d = v3_30d / np.linalg.norm(v3_30d)
        else:
            # 选择投影长度最大的候选（最正交的方向）
            v3_candidates_30d.sort(key=lambda x: -x[0])
            v3_30d = v3_candidates_30d[0][1]
        
        basis_30d = np.vstack([v1_30d, v2_30d, v3_30d])  # 形状 (3, 30)
    
    # 将30维空间的3维子空间基转换回原始特征空间
    # basis_30d @ pc30_basis 得到原始特征空间的基向量
    basis = basis_30d @ pc30_basis  # 形状 (3, n_features)
    
    # 归一化（确保是单位向量）
    for i in range(3):
        norm = np.linalg.norm(basis[i])
        if norm > 1e-10:
            basis[i] = basis[i] / norm
    
    return basis


def project_to_subspace(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """将特征矩阵投影到给定的子空间。
    
    参数:
        X: 特征矩阵，形状 (n_samples, n_features)
        basis: 子空间基，形状 (n_dims, n_features)
    
    返回:
        投影坐标，形状 (n_samples, n_dims)
    """
    # 中心化
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    # 投影：Xc @ basis.T
    return Xc @ basis.T


def load_features_csv(path: str, dtype=np.float32) -> np.ndarray:
    """读取带表头的 CSV，跳过首列路径，返回二维特征矩阵。

    使用 csv.reader 手动解析，避免 numpy.loadtxt 对字符串路径的解析问题。
    """
    import csv

    rows: List[List[float]] = []
    numeric_idx: List[int] = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 丢弃表头
        first_data = next(reader, None)
        if first_data is None:
            raise ValueError("CSV 中没有数据行")

        # 确定哪些列是数值列：跳过首列路径，保留可以转成 float 的列
        for idx, val in enumerate(first_data[1:], start=1):
            try:
                float(val)
                numeric_idx.append(idx)
            except ValueError:
                # 非数值列（如类别/文件名等）直接跳过
                continue
        if not numeric_idx:
            raise ValueError("未找到可解析的数值特征列")

        # 处理首行
        rows.append([float(first_data[i]) for i in numeric_idx])

        # 继续读取剩余行
        for line_no, row in enumerate(reader, start=3):  # 之前已读两行
            try:
                rows.append([float(row[i]) for i in numeric_idx])
            except ValueError as e:
                raise ValueError(f"第 {line_no} 行存在无法转换为浮点的值: {e}") from e

    return np.asarray(rows, dtype=dtype)


def save_results_csv(
    path: str,
    neuron_ids: List[int],
    corrs: np.ndarray,
    axis_lengths: np.ndarray,
    preferred_axes: np.ndarray,
    pc_proj: np.ndarray,
    best_alphas: List[float],
    subspace_coords: np.ndarray = None,
) -> None:
    headers = [
        "neuron_id",
        "prediction_correlation",
        "axis_length",
        "best_alpha",
        "pc1_projection",
        "pc2_projection",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, nid in enumerate(neuron_ids):
            writer.writerow(
                [
                    nid,
                    corrs[i],
                    axis_lengths[i],
                    best_alphas[i],
                    pc_proj[i, 0] if pc_proj.size else "",
                    pc_proj[i, 1] if pc_proj.size else "",
                ]
            )
    # 可选：另存偏好轴矩阵，便于后续使用
    np.save(os.path.splitext(path)[0] + "_axes.npy", preferred_axes)
    
    # 保存3维子空间坐标：形状 (n_neurons, n_samples, 3)
    if subspace_coords is not None:
        subspace_path = os.path.splitext(path)[0] + "_subspace_coords.npy"
        np.save(subspace_path, subspace_coords)
        print(f"3维子空间坐标已保存到 {subspace_path}")
        print(f"  形状: {subspace_coords.shape} (神经元数 x 图像数 x 3维)")


def run_analysis(
    feature_path: str,
    response_path: str,
    alphas: List[float],
    outer_splits: int,
    inner_splits: int,
    seed: int,
    compute_pca: bool,
    max_neurons: int = -1,
):
    print("加载数据...")
    X = load_features_csv(feature_path, dtype=np.float32)
    y_all = np.load(response_path)

    if X.shape[0] != y_all.shape[0]:
        raise ValueError(f"特征与响应图片数不一致: {X.shape[0]} vs {y_all.shape[0]}")

    n_samples, n_features = X.shape
    n_neurons = y_all.shape[1] if max_neurons <= 0 else min(max_neurons, y_all.shape[1])

    print(f"特征矩阵: {X.shape}, 神经元数: {n_neurons}")
    rng = np.random.default_rng(seed)
    outer_cv = kfold_indices(n_samples, outer_splits, rng)

    all_preferred_axes = np.zeros((n_neurons, n_features), dtype=np.float32)
    all_corrs: List[float] = []
    best_alphas: List[float] = []

    for neuron_idx in range(n_neurons):
        y = y_all[:, neuron_idx].astype(np.float32)
        fold_corrs = []
        weight_accum = np.zeros(n_features, dtype=np.float64)

        for fold_id, (train_idx, test_idx) in enumerate(outer_cv):
            inner_rng = np.random.default_rng(seed + neuron_idx * 100 + fold_id)
            alpha = select_alpha_cv(X[train_idx], y[train_idx], alphas, inner_splits, inner_rng)
            w, X_mean, y_mean = ridge_weights(X[train_idx], y[train_idx], alpha)
            y_pred = predict(X[test_idx], w, X_mean, y_mean)
            fold_corrs.append(pearson_corr(y[test_idx], y_pred))
            weight_accum += w
        best_alpha = float(select_alpha_cv(X, y, alphas, inner_splits, np.random.default_rng(seed + neuron_idx)))

        all_preferred_axes[neuron_idx] = (weight_accum / outer_splits).astype(np.float32)
        all_corrs.append(float(np.mean(fold_corrs)))
        best_alphas.append(best_alpha)

        if (neuron_idx + 1) % 20 == 0 or neuron_idx == n_neurons - 1:
            print(f"完成神经元 {neuron_idx + 1}/{n_neurons}, 平均相关={all_corrs[-1]:.3f}, best_alpha={best_alpha}")

    all_corrs_arr = np.array(all_corrs, dtype=np.float32)
    axis_lengths = np.linalg.norm(all_preferred_axes, axis=1)

    if compute_pca:
        print("计算前 2 个主成分并投影偏好轴...")
        pc_basis = compute_pca_basis(X, n_components=2)  # 形状 (2, n_features)
        pc_proj = all_preferred_axes @ pc_basis.T  # (n_neurons, 2)
    else:
        pc_proj = np.zeros((n_neurons, 0), dtype=np.float32)

    # 计算每个神经元的3维子空间坐标（在30维PCA空间的3维子空间中）
    print("计算30维PCA基...")
    pc30_basis = compute_pca_basis(X, n_components=30)  # 形状 (30, n_features)
    print("计算每个神经元的3维子空间坐标（在30维PCA空间的3维子空间中）...")
    all_subspace_coords = np.zeros((n_neurons, n_samples, 3), dtype=np.float32)
    for neuron_idx in range(n_neurons):
        preferred_axis = all_preferred_axes[neuron_idx]
        basis = build_3d_subspace_in_30d(preferred_axis, X, pc30_basis)
        coords = project_to_subspace(X, basis)
        all_subspace_coords[neuron_idx] = coords
        if (neuron_idx + 1) % 20 == 0 or neuron_idx == n_neurons - 1:
            print(f"完成神经元 {neuron_idx + 1}/{n_neurons} 的3维子空间坐标计算")

    summary = {
        "corr_mean": float(all_corrs_arr.mean()),
        "corr_std": float(all_corrs_arr.std()),
        "corr_median": float(np.median(all_corrs_arr)),
        "best_neuron": int(np.argmax(all_corrs_arr)),
        "best_corr": float(all_corrs_arr.max()),
    }
    print("====== 结果概览 ======")
    print(f"神经元数: {n_neurons}")
    print(f"预测相关均值: {summary['corr_mean']:.4f} ± {summary['corr_std']:.4f}")
    print(f"预测相关中位数: {summary['corr_median']:.4f}")
    print(f"最佳神经元: #{summary['best_neuron']} (r = {summary['best_corr']:.4f})")

    return all_preferred_axes, all_corrs_arr, axis_lengths, pc_proj, best_alphas, summary, all_subspace_coords


def parse_args():
    parser = argparse.ArgumentParser(description="岭回归偏好轴分析 (纯 numpy 版本)")
    parser.add_argument(
        "--feature-path",
        default="/media/ubuntu/sda/Monkey/image_feature/image_features_756d_clip_vitl14_F.csv",
        help="特征 CSV 路径（首列为路径，后面为特征）",
    )
    parser.add_argument(
        "--response-path",
        default="/media/ubuntu/sda/Monkey/image_feature/train_MUA_MonkeyF.npy",
        help="神经响应 npy 路径",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2, 1e-1, 1, 10, 100],
        help="候选岭回归正则强度列表",
    )
    parser.add_argument("--outer-splits", type=int, default=10, help="外部折数")
    parser.add_argument("--inner-splits", type=int, default=5, help="内部折数用于选 alpha")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-neurons", type=int, default=-1, help="仅分析前 N 个神经元，-1 表示全部")
    parser.add_argument("--no-pca", action="store_true", help="跳过 PCA 投影（省时省内存）")
    parser.add_argument(
        "--output",
        default="neuron_preferred_axes_results.csv",
        help="结果 CSV 输出路径（旁边会生成 _axes.npy）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preferred_axes, corrs, axis_lengths, pc_proj, best_alphas, summary, subspace_coords = run_analysis(
        feature_path=args.feature_path,
        response_path=args.response_path,
        alphas=args.alphas,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        seed=args.seed,
        compute_pca=not args.no_pca,
        max_neurons=args.max_neurons,
    )

    neuron_ids = list(range(preferred_axes.shape[0]))
    save_results_csv(
        path=args.output,
        neuron_ids=neuron_ids,
        corrs=corrs,
        axis_lengths=axis_lengths,
        preferred_axes=preferred_axes,
        pc_proj=pc_proj,
        best_alphas=best_alphas,
        subspace_coords=subspace_coords,
    )
    print(f"结果已写入 {args.output} 与 {os.path.splitext(args.output)[0] + '_axes.npy'}")


if __name__ == "__main__":
    main()

