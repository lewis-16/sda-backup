#!/usr/bin/env python3
"""
评估AlexNet fc7和fc8层的编码性能
"""

import numpy as np
import pandas as pd
import os
import torch
from torchvision import models, transforms
from PIL import Image
from scipy.stats import pearsonr
from datetime import datetime
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

GPU_BATCH_SIZE = 10
MIN_COMPONENTS = 5
MAX_COMPONENTS = 25


def load_alexnet(device):
    """加载预训练AlexNet模型"""
    alexnet = models.alexnet(weights='IMAGENET1K_V1')
    alexnet.eval()
    alexnet = alexnet.to(device)
    return alexnet


def extract_fc7_features(image_files, stimuli_path, alexnet, device, batch_size=32):
    """提取图像的AlexNet fc7层特征"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    n_images = len(image_files)
    fc7_features = np.zeros((n_images, 4096), dtype=np.float32)
    
    print(f"开始提取 {n_images} 张图像的AlexNet fc7特征...")
    
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
            x = alexnet.features(batch_tensor)
            x = alexnet.avgpool(x)
            x = torch.flatten(x, 1)
            # fc6 -> relu -> fc7
            x = alexnet.classifier[1](x)
            x = torch.nn.functional.relu(x)
            fc7_activations = alexnet.classifier[4](x)
            fc7_activations = torch.nn.functional.relu(fc7_activations)
            fc7_features[i:i+len(batch_files)] = fc7_activations.cpu().numpy()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理 {min(i + batch_size, n_images)}/{n_images} 张图像")
    
    print(f"fc7特征提取完成，特征矩阵形状: {fc7_features.shape}")
    return fc7_features


def extract_fc8_features(image_files, stimuli_path, alexnet, device, batch_size=32):
    """提取图像的AlexNet fc8层特征"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    n_images = len(image_files)
    fc8_features = np.zeros((n_images, 1000), dtype=np.float32)
    
    print(f"开始提取 {n_images} 张图像的AlexNet fc8特征...")
    
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
            x = alexnet.features(batch_tensor)
            x = alexnet.avgpool(x)
            x = torch.flatten(x, 1)
            # fc6 -> relu -> fc7 -> relu -> fc8
            x = alexnet.classifier[1](x)
            x = torch.nn.functional.relu(x)
            x = alexnet.classifier[4](x)
            x = torch.nn.functional.relu(x)
            # fc8
            fc8_activations = alexnet.classifier[6](x)
            fc8_features[i:i+len(batch_files)] = fc8_activations.cpu().numpy()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理 {min(i + batch_size, n_images)}/{n_images} 张图像")
    
    print(f"fc8特征提取完成，特征矩阵形状: {fc8_features.shape}")
    return fc8_features


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def compute_pls_predictions_all_components(X_train, Y_train, X_test, n_components_range):
    """使用GPU计算所有成分数的预测结果"""
    device = X_train.device
    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    n_neurons = Y_train.shape[1]
    n_test = X_test.shape[0]
    n_comps = len(n_components_range)
    max_comp = max(n_components_range)
    
    max_comp = min(max_comp, n_train - 1)
    
    W = torch.zeros(n_features, max_comp, device=device, dtype=X_train.dtype)
    P = torch.zeros(n_features, max_comp, device=device, dtype=X_train.dtype)
    Q = torch.zeros(n_neurons, max_comp, device=device, dtype=X_train.dtype)
    
    X_residual = X_train.clone()
    Y_residual = Y_train.clone()
    
    for a in range(max_comp):
        C_mat = X_residual.T @ Y_residual
        
        try:
            U, S, Vh = torch.linalg.svd(C_mat, full_matrices=False)
            w = U[:, 0:1]
        except:
            w = C_mat[:, 0:1] / (torch.norm(C_mat[:, 0:1]) + 1e-8)
        
        W[:, a] = w.squeeze()
        
        t = X_residual @ w
        
        t_norm_sq = (t.T @ t).item() + 1e-8
        p = (X_residual.T @ t) / t_norm_sq
        P[:, a] = p.squeeze()
        
        q = (Y_residual.T @ t) / t_norm_sq
        Q[:, a] = q.squeeze()
        
        X_residual = X_residual - t @ p.T
        Y_residual = Y_residual - t @ q.T
    
    W_comp = W[:, :max_comp]
    P_comp = P[:, :max_comp]
    Q_comp = Q[:n_neurons, :max_comp]
    
    PWP_inv = torch.inverse(P_comp.T @ W_comp)
    B = W_comp @ PWP_inv @ Q_comp.T
    
    Y_train_pred_all = torch.zeros(n_comps, n_train, n_neurons, device=device, dtype=X_train.dtype)
    Y_test_pred_all = torch.zeros(n_comps, n_test, n_neurons, device=device, dtype=X_train.dtype)
    
    for i, n_comp in enumerate(n_components_range):
        if n_comp > max_comp:
            n_comp = max_comp
        
        W_i = W_comp[:, :n_comp]
        P_i = P_comp[:, :n_comp]
        Q_i = Q_comp[:, :n_comp]
        
        PWP_inv_i = torch.inverse(P_i.T @ W_i)
        B_i = W_i @ PWP_inv_i @ Q_i.T
        
        Y_train_pred_all[i] = X_train @ B_i
        Y_test_pred_all[i] = X_test @ B_i
    
    return Y_train_pred_all, Y_test_pred_all


def evaluate_encoding_performance_gpu(features, neuron_responses, unit_info, encoding_type="vision"):
    """使用GPU评估所有神经元的编码性能"""
    n_neurons_total = neuron_responses.shape[0]
    n_images = neuron_responses.shape[1]
    reliability_basic = unit_info['reliability_basic'].values
    
    print(f"  总神经元数: {n_neurons_total}")
    
    n_component_candidates = list(range(MIN_COMPONENTS, MAX_COMPONENTS + 1))
    
    np.random.seed(42)
    indices = np.random.permutation(n_images)
    train_img_idx = indices[:900]
    test_img_idx = indices[900:]
    
    X_train = features[train_img_idx]
    X_test = features[test_img_idx]
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    X_train_gpu = torch.from_numpy(X_train_scaled).float().to(device)
    X_test_gpu = torch.from_numpy(X_test_scaled).float().to(device)
    
    batch_size = GPU_BATCH_SIZE
    n_batches = int(np.ceil(n_neurons_total / batch_size))
    
    encoding_correlations = np.zeros(n_neurons_total)
    normalized_correlations = np.zeros(n_neurons_total)
    best_components = np.zeros(n_neurons_total, dtype=int)
    
    pbar = tqdm(
        total=n_neurons_total,
        desc=f"  {encoding_type} encoding",
        ncols=80,
        unit_scale=True
    )
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_neurons_total)
        n_neurons_in_batch = end_idx - start_idx
        
        Y_batch = neuron_responses[start_idx:end_idx, :]
        
        Y_train = Y_batch[:, train_img_idx]
        Y_test = Y_batch[:, test_img_idx]
        
        Y_train_swapped = Y_train.T
        Y_test_swapped = Y_test.T
        
        Y_train_gpu = torch.from_numpy(Y_train_swapped).float().to(device)
        Y_test_gpu = torch.from_numpy(Y_test_swapped).float().to(device)
        
        Y_train_pred_all, Y_test_pred_all = compute_pls_predictions_all_components(
            X_train_gpu, Y_train_gpu, X_test_gpu, n_component_candidates
        )
        
        Y_train_pred_np = Y_train_pred_all.cpu().numpy()
        Y_test_pred_np = Y_test_pred_all.cpu().numpy()
        
        del Y_train_pred_all, Y_test_pred_all, Y_train_gpu, Y_test_gpu
        clear_gpu_memory()
        
        for local_idx in range(n_neurons_in_batch):
            global_neuron_idx = start_idx + local_idx
            
            mse_for_comp = np.zeros(len(n_component_candidates))
            for comp_idx in range(len(n_component_candidates)):
                mse_for_comp[comp_idx] = np.mean(
                    (Y_train_swapped[:, local_idx] - Y_train_pred_np[comp_idx, :, local_idx]) ** 2
                )
            
            best_comp_idx = np.argmin(mse_for_comp)
            best_components[global_neuron_idx] = n_component_candidates[best_comp_idx]
            
            y_test_true = Y_test_swapped[:, local_idx]
            y_test_pred = Y_test_pred_np[best_comp_idx, :, local_idx]
            
            if np.std(y_test_true) > 0 and np.std(y_test_pred) > 0:
                corr, _ = pearsonr(y_test_true, y_test_pred)
                encoding_correlations[global_neuron_idx] = corr
            else:
                encoding_correlations[global_neuron_idx] = 0.0
            
            pbar.update(1)
    
    pbar.close()
    
    for i in range(n_neurons_total):
        reliability = reliability_basic[i]
        if reliability > 0:
            normalized_correlations[i] = encoding_correlations[i] / reliability
        else:
            normalized_correlations[i] = 0.0
    
    print(f"  平均最优成分数: {np.mean(best_components):.1f}")
    print(f"  {encoding_type} encoding 完成!")
    
    return encoding_correlations, normalized_correlations, best_components


def main():
    """主函数"""
    start_time = datetime.now()
    print("="*60)
    print("AlexNet fc7/fc8 编码性能评估")
    print("="*60)
    
    stimuli_path = '/media/ubuntu/sda/TrippleN/stimuli'
    unit_info_path = '/media/ubuntu/sda/TrippleN/customize/all_subjects_unit_info.pkl'
    neuron_responses_path = '/media/ubuntu/sda/TrippleN/customize/neuron_responses_1000.npy'
    
    # 输出文件路径
    fc7_output_path = '/media/ubuntu/sda/TrippleN/customize/alexnet_fc7_encoding_results_gpu.pkl'
    fc8_output_path = '/media/ubuntu/sda/TrippleN/customize/alexnet_fc8_encoding_results_gpu.pkl'
    
    print("\n[1/4] 加载数据...")
    neuron_responses = np.load(neuron_responses_path)
    print(f"  响应矩阵形状: {neuron_responses.shape}")
    
    unit_info = pd.read_pickle(unit_info_path)
    print(f"  单元信息记录数: {len(unit_info)}")
    
    image_files = sorted([f for f in os.listdir(stimuli_path) if f.endswith('.bmp')])[:1000]
    print(f"  使用图像数量: {len(image_files)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载AlexNet
    print("\n[2/4] 加载AlexNet模型...")
    alexnet = load_alexnet(device)
    
    # 提取fc7特征
    print("\n[3/4] 提取fc7特征...")
    fc7_features = extract_fc7_features(image_files, stimuli_path, alexnet, device)
    
    # 提取fc8特征
    print("\n[4/4] 提取fc8特征...")
    fc8_features = extract_fc8_features(image_files, stimuli_path, alexnet, device)
    
    del alexnet
    clear_gpu_memory()
    
    # 评估fc7
    print("\n评估fc7编码性能...")
    fc7_corr, fc7_norm_corr, fc7_comps = evaluate_encoding_performance_gpu(
        fc7_features, neuron_responses, unit_info, encoding_type="alexnet_fc7"
    )
    
    # 保存fc7结果
    results_df = pd.DataFrame({
        'encoding_correlation': fc7_corr,
        'normalized_correlation': fc7_norm_corr,
        'reliability_best': unit_info['reliability_best'].values,
        'best_r_time1': unit_info['best_r_time1'].values,
        'best_r_time2': unit_info['best_r_time2'].values,
        'best_n_components': fc7_comps,
    })
    results_df.to_pickle(fc7_output_path)
    print(f"fc7结果已保存到: {fc7_output_path}")
    
    print(f"\nfc7编码相关性: 均值={np.mean(fc7_corr):.4f}, 标准化={np.mean(fc7_norm_corr):.4f}")
    
    clear_gpu_memory()
    
    # 评估fc8
    print("\n评估fc8编码性能...")
    fc8_corr, fc8_norm_corr, fc8_comps = evaluate_encoding_performance_gpu(
        fc8_features, neuron_responses, unit_info, encoding_type="alexnet_fc8"
    )
    
    # 保存fc8结果
    results_df = pd.DataFrame({
        'encoding_correlation': fc8_corr,
        'normalized_correlation': fc8_norm_corr,
        'reliability_best': unit_info['reliability_best'].values,
        'best_r_time1': unit_info['best_r_time1'].values,
        'best_r_time2': unit_info['best_r_time2'].values,
        'best_n_components': fc8_comps,
    })
    results_df.to_pickle(fc8_output_path)
    print(f"fc8结果已保存到: {fc8_output_path}")
    
    print(f"\nfc8编码相关性: 均值={np.mean(fc8_corr):.4f}, 标准化={np.mean(fc8_norm_corr):.4f}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n运行完成，耗时: {duration}")
    print("="*60)


if __name__ == '__main__':
    main()
