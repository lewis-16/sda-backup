#!/usr/bin/env python3
"""
AlexNet特征与语言特征提取的PLSR模型预测神经元活动

包含两种编码方式：
1. Vision-based encoding: AlexNet fc6层特征
2. Language-based encoding: all-mpnet-base-v2处理的coco caption特征

使用偏最小二乘回归(PLSR)预测神经元对图像的响应

测试版本特点：
- 随机采样1000个神经元进行测试
- 使用sklearn的PLSR（CPU版本）
- 成分数范围: 5-25
- 固定900训练/100测试分割

论文方法：
- 固定900训练/100测试分割
- 通过最小化训练集MSE选择最优成分数（5-25）
- 编码精度 = 预测与实际响应的相关性
- 标准化精度 = 编码精度 / √(reliability)
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
from datetime import datetime
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import warnings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')


NUM_WORKERS = 30  # 并行进程数
GPU_BATCH_SIZE = 10  # GPU上批量处理的神经元数量（根据GPU内存调整）
MIN_COMPONENTS = 5  # 最小成分数
MAX_COMPONENTS = 25  # 最大成分数


def setup_device():
    """设置计算设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    return device


def load_alexnet(device):
    """加载预训练AlexNet模型"""
    alexnet = models.alexnet(weights='IMAGENET1K_V1')
    alexnet.eval()
    alexnet = alexnet.to(device)
    
    return alexnet


def extract_fc6_features(image_files, stimuli_path, alexnet, device, batch_size=32):
    """提取图像的AlexNet fc6层特征
    
    遵循原始AlexNet架构：
    Input(224x224) → features → avgpool → flatten → fc6 → relu
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    n_images = len(image_files)
    fc6_features = np.zeros((n_images, 4096), dtype=np.float32)
    
    print(f"开始提取 {n_images} 张图像的AlexNet fc6特征...")
    
    for i in range(0, n_images, batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for filename in batch_files:
            img_path = os.path.join(stimuli_path, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            # 原始AlexNet架构：features → avgpool → flatten → fc6
            x = alexnet.features(batch_tensor)
            x = alexnet.avgpool(x)
            x = torch.flatten(x, 1)
            fc6_activations = alexnet.classifier[1](x)
            fc6_activations = nn.functional.relu(fc6_activations)
            
            fc6_features[i:i+len(batch_files)] = fc6_activations.cpu().numpy()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理 {min(i + batch_size, n_images)}/{n_images} 张图像")
    
    print(f"特征提取完成，特征矩阵形状: {fc6_features.shape}")
    return fc6_features


def load_sentence_model(device):
    """加载all-mpnet-base-v2 sentence transformer模型"""
    model = SentenceTransformer('/media/ubuntu/sda/TrippleN/model/all-mpnet-base-v2')
    model.eval()
    return model


def extract_caption_features(captions_path, model, batch_size=32):
    """提取coco captions的语义特征
    
    使用all-mpnet-base-v2模型处理captions
    对每张图片的5个caption取平均，得到768维特征向量
    """
    print("加载coco captions矩阵...")
    with open(captions_path, 'rb') as f:
        captions_matrix = pickle.load(f)
    
    n_images = captions_matrix.shape[0]
    print(f"Caption矩阵形状: {captions_matrix.shape}")
    
    # 存储所有图片的特征
    caption_features = np.zeros((n_images, 768), dtype=np.float32)
    
    print(f"开始使用all-mpnet-base-v2提取 {n_images} 张图片的caption特征...")
    
    for i in range(0, n_images, batch_size):
        end_idx = min(i + batch_size, n_images)
        
        # 获取当前batch的所有captions (5个 per image)
        batch_captions = []
        for j in range(i, end_idx):
            batch_captions.extend(captions_matrix[j].tolist())
        
        # 批量编码
        embeddings = model.encode(batch_captions, show_progress_bar=False)
        
        # 对每张图片的5个caption取平均
        for j in range(i, end_idx):
            start_emb = (j - i) * 5
            end_emb = start_emb + 5
            caption_features[j] = np.mean(embeddings[start_emb:end_emb], axis=0)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  已处理 {min(end_idx, n_images)}/{n_images} 张图片")
    
    print(f"Caption特征提取完成，特征矩阵形状: {caption_features.shape}")
    return caption_features


def load_neuron_responses(responses_path):
    """加载预计算的神经元响应数据"""
    print(f"加载神经元响应数据: {responses_path}")
    neuron_responses = np.load(responses_path)
    print(f"  响应矩阵形状: {neuron_responses.shape}")
    print(f"  是否包含NaN: {np.any(np.isnan(neuron_responses))}")
    return neuron_responses


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def compute_pls_predictions_all_components(X_train, Y_train, X_test, n_components_range):
    """使用GPU计算所有成分数的预测结果
    
    使用NIPALS算法一次性计算所有成分数的PLSR权重
    然后对每个成分数计算预测
    
    Parameters:
    -----------
    X_train : torch.Tensor
        训练特征 (n_train, n_features)
    Y_train : torch.Tensor
        训练目标 (n_train, n_neurons)
    X_test : torch.Tensor
        测试特征 (n_test, n_features)
    n_components_range : list
        成分数范围，如 [5, 6, ..., 25]
    
    Returns:
    --------
    Y_train_pred_all : torch.Tensor
        所有成分数的训练集预测 (len(n_components_range), n_train, n_neurons)
    Y_test_pred_all : torch.Tensor
        所有成分数的测试集预测 (len(n_components_range), n_test, n_neurons)
    """
    device = X_train.device
    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    n_neurons = Y_train.shape[1]
    n_test = X_test.shape[0]
    n_comps = len(n_components_range)
    max_comp = max(n_components_range)
    
    # 确保成分数不超过限制
    max_comp = min(max_comp, n_train - 1)
    
    # 使用NIPALS算法计算PLSR
    # 存储每个成分的权重和载荷
    W = torch.zeros(n_features, max_comp, n_neurons, device=device, dtype=X_train.dtype)
    C = torch.zeros(max_comp, n_neurons, n_neurons, device=device, dtype=X_train.dtype)
    
    # 残差矩阵
    X_residual = X_train.clone()
    Y_residual = Y_train.clone()
    
    # 迭代计算每个成分
    for a in range(max_comp):
        # 计算权重向量: w = X^T Y / ||X^T Y||
        C_mat = X_residual.T @ Y_residual  # (n_features, n_neurons)
        
        # 归一化: 对每个神经元分别归一化
        w_norm = torch.norm(C_mat, dim=0, keepdim=True) + 1e-8
        w = C_mat / w_norm  # (n_features, n_neurons)
        
        # 存储权重
        W[:, a, :] = w
        
        # 计算得分向量: t = X w
        t = X_residual @ w  # (n_train, n_neurons)
        
        # 计算每个神经元 t 的范数平方 (用于归一化)
        t_norm_sq = torch.sum(t ** 2, dim=0, keepdim=True) + 1e-8  # (1, n_neurons)
        
        # 计算载荷向量: p = (X^T t) / (t^T t)
        Xt = X_residual.T @ t  # (n_features, n_neurons)
        p = Xt / t_norm_sq.T  # (n_features, n_neurons), 正确广播
        
        # 回归系数: c = (Y^T t) / (t^T t)
        # Yt 形状是 (n_neurons, n_neurons)
        # t_norm_sq 形状是 (1, n_neurons)
        # 我们需要每列除以对应的 t_norm_sq，所以要转置
        Yt = Y_residual.T @ t  # (n_neurons, n_neurons)
        c = Yt / t_norm_sq.t()  # (n_neurons, n_neurons), 每列除以对应元素
        
        C[a, :, :] = c
        
        # 更新残差
        X_residual = X_residual - t @ p.T
        Y_residual = Y_residual - t @ c.T
    
    # 计算X的均值和标准差（用于标准化）
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    # 对每个成分数计算预测
    Y_train_pred_all = torch.zeros(n_comps, n_train, n_neurons, device=device, dtype=X_train.dtype)
    Y_test_pred_all = torch.zeros(n_comps, n_test, n_neurons, device=device, dtype=X_train.dtype)
    
    for i, n_comp in enumerate(n_components_range):
        if n_comp > max_comp:
            n_comp = max_comp
        
        # 计算回归系数矩阵 B = W @ (C[0:n_comp]的累积)
        # 使用B式回归: B = W (P^T W)^(-1) C^T
        # 简化版本: 直接累积计算
        
        B = torch.zeros(n_features, n_neurons, device=device, dtype=X_train.dtype)
        
        for a in range(n_comp):
            w_a = W[:, a, :]  # (n_features, n_neurons)
            c_a = C[a, :, :]  # (n_neurons, n_neurons)
            B = B + w_a @ c_a
        
        # 预测
        Y_train_pred_all[i] = X_train_scaled @ B
        Y_test_pred_all[i] = X_test_scaled @ B
    
    return Y_train_pred_all, Y_test_pred_all


def compute_all_mse_and_correlations_gpu(Y_train_true, Y_train_pred_all, Y_test_true, Y_test_pred_all, n_components_range):
    """计算所有成分数的MSE和相关系数
    
    Parameters:
    -----------
    Y_train_true : torch.Tensor
        训练集真实响应 (n_train, n_neurons)
    Y_train_pred_all : torch.Tensor
        所有成分数的训练集预测 (n_comps, n_train, n_neurons)
    Y_test_true : torch.Tensor
        测试集真实响应 (n_test, n_neurons)
    Y_test_pred_all : torch.Tensor
        所有成分数的测试集预测 (n_comps, n_test, n_neurons)
    n_components_range : list
        成分数范围
    
    Returns:
    --------
    train_mse : torch.Tensor
        训练集MSE (n_comps, n_neurons)
    test_correlations : torch.Tensor
        测试集相关系数 (n_comps, n_neurons)
    """
    n_comps = len(n_components_range)
    n_neurons = Y_train_true.shape[1]
    device = Y_train_true.device
    
    # 计算训练集MSE
    train_mse = torch.zeros(n_comps, n_neurons, device=device, dtype=torch.float32)
    for i in range(n_comps):
        mse = torch.mean((Y_train_true - Y_train_pred_all[i]) ** 2, dim=0)
        train_mse[i] = mse
    
    # 计算测试集相关系数（使用NumPy/Python计算，因为PyTorch没有内置的pearsonr）
    test_correlations = torch.zeros(n_comps, n_neurons, device=device, dtype=torch.float32)
    
    Y_test_true_np = Y_test_true.cpu().numpy()
    Y_test_pred_np = Y_test_pred_all.cpu().numpy()
    
    for i in range(n_comps):
        for j in range(n_neurons):
            y_true = Y_test_true_np[:, j]
            y_pred = Y_test_pred_np[i, :, j]
            
            if np.std(y_true) > 0 and np.std(y_pred) > 0:
                corr, _ = pearsonr(y_true, y_pred)
                test_correlations[i, j] = corr
            else:
                test_correlations[i, j] = 0.0
    
    return train_mse, test_correlations


def _train_neuron_batch_gpu(args):
    """GPU批量训练一批神经元的PLSR（一次性计算所有成分数的MSE和相关性）
    
    按照论文方法：
    - 固定900训练/100测试分割
    - 一次性计算所有成分数(5-25)的预测结果
    - 通过最小化训练集MSE选择最优成分数
    - 在测试集上评估相关性
    
    包含GPU内存管理和CPU回退机制
    """
    batch_idx, features, neuron_responses, n_component_candidates, reliability_best = args
    
    n_neurons_in_batch = len(batch_idx)
    
    # 验证 batch_idx 的有效性
    max_batch_idx = batch_idx.max() if len(batch_idx) > 0 else 0
    n_neurons_total = neuron_responses.shape[0]
    
    if max_batch_idx >= n_neurons_total:
        raise ValueError(f"ERROR: batch_idx max={max_batch_idx} >= neuron_responses rows={n_neurons_total}")
    
    encoding_correlations = np.zeros(n_neurons_in_batch)
    best_components = np.zeros(n_neurons_in_batch, dtype=int)
    
    # 检查NaN
    if np.any(np.isnan(neuron_responses[batch_idx])):
        raise ValueError(f"错误: 神经元响应数据包含NaN值！")
    
    # 固定随机种子
    np.random.seed(42)
    
    # 固定的训练/测试分割 (900/100)
    n_images = features.shape[0]
    indices = np.random.permutation(n_images)
    train_idx = indices[:900]
    test_idx = indices[900:]
    
    # 提取数据 - 使用更安全的索引方式避免内存引用问题
    X_train = features[train_idx]  # (900, n_features)
    X_test = features[test_idx]    # (100, n_features)
    
    # 关键：使用 .copy() 和正确的索引方式避免内存引用问题
    # neuron_responses[batch_idx] 返回的是视图，容易被垃圾回收修改
    Y_train = neuron_responses[batch_idx, :][:, train_idx].copy()  # (n_neurons_in_batch, 900)
    Y_test = neuron_responses[batch_idx, :][:, test_idx].copy()    # (n_neurons_in_batch, 100)
    
    # 交换维度：(n_samples, n_targets)
    Y_train_swapped = Y_train.T  # (900, n_neurons_in_batch)
    Y_test_swapped = Y_test.T    # (100, n_neurons_in_batch)
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 移到GPU
        X_train_gpu = torch.from_numpy(X_train).float().to(device)
        X_test_gpu = torch.from_numpy(X_test).float().to(device)
        Y_train_gpu = torch.from_numpy(Y_train_swapped).float().to(device)
        Y_test_gpu = torch.from_numpy(Y_test_swapped).float().to(device)
        
        # 一次性计算所有成分数的预测
        Y_train_pred_all, Y_test_pred_all = compute_pls_predictions_all_components(
            X_train_gpu, Y_train_gpu, X_test_gpu, n_component_candidates
        )
        
        # 计算所有成分数的MSE和相关系数
        train_mse, test_correlations = compute_all_mse_and_correlations_gpu(
            Y_train_gpu, Y_train_pred_all, Y_test_gpu, Y_test_pred_all, n_component_candidates
        )
        
        # 移到CPU进行最终筛选
        train_mse_np = train_mse.cpu().numpy()  # (n_comps, n_neurons)
        test_corrs_np = test_correlations.cpu().numpy()  # (n_comps, n_neurons)
        
        # 对每个神经元，选择MSE最小的成分数
        for i in range(n_neurons_in_batch):
            mse_for_neuron = train_mse_np[:, i]
            best_comp_idx = np.argmin(mse_for_neuron)
            best_components[i] = n_component_candidates[best_comp_idx]
            encoding_correlations[i] = test_corrs_np[best_comp_idx, i]
        
        # 统计最优成分数分布
        unique_comps, counts = np.unique(best_components, return_counts=True)
        comp_dist = ", ".join([f"{c}:{cnt}" for c, cnt in zip(unique_comps, counts)])
        print(f"    最优成分数分布: {comp_dist}")
        
        # 清理GPU内存
        del X_train_gpu, X_test_gpu, Y_train_gpu, Y_test_gpu
        del Y_train_pred_all, Y_test_pred_all, train_mse, test_correlations
        clear_gpu_memory()
        
    except RuntimeError as e:
        # 如果GPU内存不足，清理并回退到CPU
        print(f"    GPU内存不足，回退到CPU: {str(e)[:50]}...")
        clear_gpu_memory()
        
        # 使用CPU版本
        from sklearn.cross_decomposition import PLSRegression
        from scipy.stats import pearsonr
        
        print("    使用CPU版本继续处理...")
        
        # 标准化X
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_train_scaled = (X_train - X_mean) / X_std
        X_test_scaled = (X_test - X_mean) / X_std
        
        # 存储所有成分数的预测结果
        n_comps = len(n_component_candidates)
        all_train_pred = np.zeros((n_comps, 900, n_neurons_in_batch))
        all_test_pred = np.zeros((n_comps, 100, n_neurons_in_batch))
        
        # 一次性训练所有成分数的模型
        for comp_idx, n_comp in enumerate(n_component_candidates):
            max_comp = min(n_comp, X_train.shape[0] - 1)
            
            for i in range(n_neurons_in_batch):
                y_train = Y_train[i]
                
                pls = PLSRegression(n_components=max_comp, scale=True)
                pls.fit(X_train_scaled, y_train)
                
                all_train_pred[comp_idx, :, i] = pls.predict(X_train_scaled).flatten()
                all_test_pred[comp_idx, :, i] = pls.predict(X_test_scaled).flatten()
        
        # 计算MSE并选择最优成分数
        for i in range(n_neurons_in_batch):
            # 找MSE最小的成分数
            best_mse = float('inf')
            best_n_comp = n_component_candidates[0]
            
            for comp_idx, n_comp in enumerate(n_component_candidates):
                y_train = Y_train[i]
                y_pred_train = all_train_pred[comp_idx, :, i]
                mse = np.mean((y_train - y_pred_train) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_n_comp = n_comp
            
            best_components[i] = best_n_comp
            
            # 获取对应成分数的测试集相关系数
            comp_idx = n_component_candidates.index(best_n_comp)
            y_test = Y_test[i]
            y_pred_test = all_test_pred[comp_idx, :, i]
            
            if np.std(y_test) > 0 and np.std(y_pred_test) > 0:
                corr, _ = pearsonr(y_test, y_pred_test)
                encoding_correlations[i] = corr
            else:
                encoding_correlations[i] = 0.0
    
    return batch_idx, encoding_correlations, best_components


def _train_neuron_batch(args):
    """训练一批神经元的PLSR（逐个计算）"""
    batch_idx, features, neuron_responses, n_component_candidates, reliability_best = args
    
    n_neurons_in_batch = len(batch_idx)
    encoding_correlations = np.zeros(n_neurons_in_batch)
    best_components = np.zeros(n_neurons_in_batch, dtype=int)
    
    # 检查NaN
    if np.any(np.isnan(neuron_responses[batch_idx])):
        raise ValueError(f"错误: 神经元响应数据包含NaN值！")
    
    # 固定随机种子
    np.random.seed(42)
    
    # 固定的训练/测试分割 (900/100)
    n_images = features.shape[0]
    indices = np.random.permutation(n_images)
    train_idx = indices[:900]
    test_idx = indices[900:]
    
    X_train = features[train_idx]
    X_test = features[test_idx]
    
    # 标准化X
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    # 逐个神经元处理
    for batch_i, global_i in enumerate(batch_idx):
        y = neuron_responses[global_i]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # 找MSE最小的成分数
        best_mse = float('inf')
        best_n_comp = n_component_candidates[0]
        
        for n_comp in n_component_candidates:
            max_comp = min(n_comp, X_train.shape[0] - 1, X_train.shape[1])
            if max_comp < 1:
                continue
            
            pls = PLSRegression(n_components=max_comp, scale=True)
            pls.fit(X_train_scaled, y_train)
            
            y_pred_train = pls.predict(X_train_scaled).flatten()
            mse = np.mean((y_train - y_pred_train) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_n_comp = n_comp
        
        best_components[batch_i] = best_n_comp
        
        # 用最优成分数预测测试集
        max_comp = min(best_n_comp, X_train.shape[0] - 1, X_train.shape[1])
        pls = PLSRegression(n_components=max_comp, scale=True)
        pls.fit(X_train_scaled, y_train)
        
        y_pred_test = pls.predict(X_test_scaled).flatten()
        
        if np.std(y_test) > 0 and np.std(y_pred_test) > 0:
            corr, _ = pearsonr(y_test, y_pred_test)
            encoding_correlations[batch_i] = corr
        else:
            encoding_correlations[batch_i] = 0.0
    
    return batch_idx, encoding_correlations, best_components


def evaluate_encoding_performance(features, neuron_responses, unit_info, encoding_type="vision"):
    """评估每个神经元的编码性能（采样1000个神经元测试）"""
    n_neurons_total = neuron_responses.shape[0]
    reliability_best = unit_info['reliability_best'].values
    
    # 随机采样1000个神经元
    np.random.seed(42)
    sampled_indices = np.random.choice(n_neurons_total, size=1000, replace=False)
    sampled_indices = np.sort(sampled_indices)
    
    print(f"  总神经元数: {n_neurons_total}, 采样数: {len(sampled_indices)}")
    
    # 候选成分数 (5-25)
    n_component_candidates = list(range(MIN_COMPONENTS, MAX_COMPONENTS + 1))
    
    n_neurons = len(sampled_indices)
    encoding_correlations = np.zeros(n_neurons)
    normalized_correlations = np.zeros(n_neurons)
    best_components = np.zeros(n_neurons, dtype=int)
    
    print(f"\n开始为 {n_neurons} 个神经元训练PLSR模型...")
    print(f"  成分数范围: {n_component_candidates[0]}-{n_component_candidates[-1]}")
    print(f"  训练/测试分割: 900/100")
    
    # 使用tqdm显示每个神经元的进度
    pbar = tqdm(
        total=n_neurons,
        desc=f"  {encoding_type} encoding",
        unit="神经元",
        ncols=80,
        unit_scale=True
    )
    
    # 逐个神经元处理
    for neuron_i in range(n_neurons):
        y = neuron_responses[sampled_indices[neuron_i]]
        y_train = y[:900]
        y_test = y[900:]
        
        # 找MSE最小的成分数
        best_mse = float('inf')
        best_n_comp = n_component_candidates[0]
        
        for n_comp in n_component_candidates:
            max_comp = min(n_comp, 899, features.shape[1])
            if max_comp < 1:
                continue
            
            pls = PLSRegression(n_components=max_comp, scale=True)
            pls.fit(features[:900], y_train)
            
            y_pred_train = pls.predict(features[:900]).flatten()
            mse = np.mean((y_train - y_pred_train) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_n_comp = n_comp
        
        best_components[neuron_i] = best_n_comp
        
        # 用最优成分数预测测试集
        max_comp = min(best_n_comp, 899, features.shape[1])
        pls = PLSRegression(n_components=max_comp, scale=True)
        pls.fit(features[:900], y_train)
        
        y_pred_test = pls.predict(features[900:]).flatten()
        
        if np.std(y_test) > 0 and np.std(y_pred_test) > 0:
            corr, _ = pearsonr(y_test, y_pred_test)
            encoding_correlations[neuron_i] = corr
        else:
            encoding_correlations[neuron_i] = 0.0
        
        pbar.update(1)
    
    pbar.close()
    
    # 计算标准化精度
    for i in range(n_neurons):
        reliability = reliability_best[sampled_indices[i]]
        if reliability > 0:
            normalized_correlations[i] = encoding_correlations[i] / np.sqrt(reliability)
        else:
            normalized_correlations[i] = 0.0
    
    # 打印成分数统计
    unique_comps, counts = np.unique(best_components, return_counts=True)
    comp_dist = ", ".join([f"{c}:{cnt}" for c, cnt in zip(unique_comps, counts)])
    print(f"  最优成分数分布: {comp_dist}")
    print(f"  平均最优成分数: {np.mean(best_components):.1f}")
    
    print(f"  {encoding_type} encoding 完成!")
    
    return encoding_correlations, normalized_correlations, best_components, sampled_indices


def save_results(encoding_correlations, normalized_correlations, unit_info, output_path, encoding_type="vision", best_components=None, sampled_indices=None):
    """保存结果"""
    results_dict = {
        'encoding_correlation': encoding_correlations,
        'normalized_correlation': normalized_correlations,
        'reliability_best': unit_info['reliability_best'].values,
        'best_r_time1': unit_info['best_r_time1'].values,
        'best_r_time2': unit_info['best_r_time2'].values,
        'subject': unit_info['subject'].values if 'subject' in unit_info.columns else None
    }
    
    if best_components is not None:
        results_dict['best_n_components'] = best_components
    
    if sampled_indices is not None:
        results_dict['sampled_indices'] = sampled_indices
    
    results_df = pd.DataFrame(results_dict)
    
    results_df.to_pickle(output_path)
    print(f"{encoding_type}-based encoding 结果已保存到: {output_path}")
    
    return results_df


def print_summary(encoding_correlations, normalized_correlations, encoding_type="vision", best_components=None):
    """打印结果摘要"""
    print("\n" + "="*60)
    print(f"{encoding_type}-based Encoding 编码性能评估结果摘要")
    print("="*60)
    
    print(f"\n编码相关性 (Encoding Correlation):")
    print(f"  平均值: {np.mean(encoding_correlations):.4f}")
    print(f"  标准差: {np.std(encoding_correlations):.4f}")
    print(f"  最小值: {np.min(encoding_correlations):.4f}")
    print(f"  最大值: {np.max(encoding_correlations):.4f}")
    print(f"  中位数: {np.median(encoding_correlations):.4f}")
    
    print(f"\n标准化相关性 (Normalized Correlation):")
    print(f"  平均值: {np.mean(normalized_correlations):.4f}")
    print(f"  标准差: {np.std(normalized_correlations):.4f}")
    print(f"  最小值: {np.min(normalized_correlations):.4f}")
    print(f"  最大值: {np.max(normalized_correlations):.4f}")
    print(f"  中位数: {np.median(normalized_correlations):.4f}")
    
    sig_count = np.sum(encoding_correlations > 0.1)
    print(f"\n显著相关神经元数量 (r > 0.1): {sig_count}/{len(encoding_correlations)}")
    
    if best_components is not None:
        print(f"\n最优成分数统计:")
        print(f"  平均值: {np.mean(best_components):.1f}")
        print(f"  标准差: {np.std(best_components):.1f}")
        print(f"  最小值: {np.min(best_components)}")
        print(f"  最大值: {np.max(best_components)}")
        print(f"  中位数: {np.median(best_components)}")
        
        # 分布
        print(f"\n  成分数分布:")
        unique_comps, counts = np.unique(best_components, return_counts=True)
        for comp, count in zip(unique_comps, counts):
            pct = count / len(best_components) * 100
            bar = "█" * int(pct / 2)
            print(f"    {comp:2d}: {bar} {count} ({pct:.1f}%)")


def main():
    """主函数"""
    start_time = datetime.now()
    print("="*60)
    print("Vision & Language Encoding PLSR神经元活动预测")
    print("开始时间:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    stimuli_path = '/media/ubuntu/sda/TrippleN/stimuli'
    unit_info_path = '/media/ubuntu/sda/TrippleN/customize/all_subjects_unit_info.pkl'
    captions_path = '/media/ubuntu/sda/TrippleN/customize/coco_captions_1000x5.pkl'
    neuron_responses_path = '/media/ubuntu/sda/TrippleN/customize/neuron_responses_1000.npy'
    vision_output_path = '/media/ubuntu/sda/TrippleN/customize/vision_encoding_results.pkl'
    language_output_path = '/media/ubuntu/sda/TrippleN/customize/language_encoding_results.pkl'
    
    print("\n[1/6] 加载神经元响应数据...")
    neuron_responses = load_neuron_responses(neuron_responses_path)
    
    print("\n[2/6] 加载单元信息...")
    unit_info = pd.read_pickle(unit_info_path)
    print(f"  单元信息记录数: {len(unit_info)}")
    
    image_files = sorted([f for f in os.listdir(stimuli_path) if f.endswith('.bmp')])[:1000]
    print(f"  使用图像数量: {len(image_files)}")
    
    print("\n[3/6] 加载AlexNet模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alexnet = load_alexnet(device)
    print("  AlexNet模型加载完成")
    
    print("\n[4/6] 提取AlexNet fc6特征 (Vision-based encoding)...")
    fc6_features = extract_fc6_features(image_files, stimuli_path, alexnet, device)
    
    print("\n[5/6] 加载Sentence Transformer模型并提取Caption特征...")
    sentence_model = load_sentence_model(device)
    print("  all-mpnet-base-v2模型加载完成")
    caption_features = extract_caption_features(captions_path, sentence_model)
    
    print("\n[6/6] 评估编码性能...")
    print("  注意: 随机采样1000个神经元进行测试")
    
    # Vision-based encoding
    print("\n  --- Vision-based Encoding (AlexNet fc6) ---")
    vision_corr, vision_norm_corr, vision_comps, vision_sampled = evaluate_encoding_performance(
        fc6_features, neuron_responses, unit_info, encoding_type="vision"
    )
    save_results(vision_corr, vision_norm_corr, unit_info, vision_output_path, 
                 encoding_type="vision", best_components=vision_comps, sampled_indices=vision_sampled)
    print_summary(vision_corr, vision_norm_corr, encoding_type="vision", best_components=vision_comps)
    
    # Language-based encoding
    print("\n  --- Language-based Encoding (all-mpnet-base-v2) ---")
    language_corr, language_norm_corr, language_comps, language_sampled = evaluate_encoding_performance(
        caption_features, neuron_responses, unit_info, encoding_type="language"
    )
    save_results(language_corr, language_norm_corr, unit_info, language_output_path, 
                 encoding_type="language", best_components=language_comps, sampled_indices=language_sampled)
    print_summary(language_corr, language_norm_corr, encoding_type="language", best_components=language_comps)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n运行完成，耗时: {duration}")
    print("="*60)


if __name__ == '__main__':
    main()
