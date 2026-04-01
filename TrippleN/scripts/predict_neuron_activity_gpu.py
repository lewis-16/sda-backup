#!/usr/bin/env python3
"""
使用PLSR模型预测神经元活动 - 5个模型

包含模型：
1. AlexNet fc6
2. CLIP ViT-L-14 (text)
3. CLIP ViT-L-14 (image)
4. CLIP RN50 (text)
5. CLIP RN50 (image)
6. CLIP RN101 (text)
7. CLIP RN101 (image)
8. all-mpnet-base-v2
9. dinov3_vitl16
10. dinov3_convnext_base
11. dinov3_vitb16

使用GPU加速PLSR计算
"""

import argparse

import numpy as np
import pandas as pd
import os
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
from datetime import datetime
import pickle
import tqdm
import warnings
from sentence_transformers import SentenceTransformer
import open_clip

warnings.filterwarnings('ignore')

def _open_image(img_path, gray=False):
    img = Image.open(img_path)
    if gray:
        img = img.convert('L').convert('RGB')
    else:
        img = img.convert('RGB')
    return img


NUM_WORKERS = 30
GPU_BATCH_SIZE = 10
MIN_COMPONENTS = 5
MAX_COMPONENTS = 25

DINOV3_REPO = '/media/ubuntu/sda/paper_code/dinov3-main/'
DINOV3_WEIGHTS_DIR = '/media/ubuntu/sda/TrippleN/model'
DINOV3_WEIGHT_FILES = {
    'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    'dinov3_convnext_base': 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
    'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
}
DINOV3_FEATURE_DIMS = {
    'dinov3_vitl16': 1024,
    'dinov3_convnext_base': 1024,
    'dinov3_vitb16': 768,
}

ALEXNET_AE_WEIGHTS = '/media/ubuntu/sda/TrippleN/model/fc6_ae_best.pth'
ALEXNET_AE_DIM = 4096

ALL_MODELS = [
    'alexnet',
    'alexnet_ae',
    'clip_vit_l14_text', 'clip_vit_l14_image',
    'clip_rn50_text', 'clip_rn50_image',
    'clip_rn101_text', 'clip_rn101_image',
    'all_mpnet_base_v2',
    'dinov3_vitl16', 'dinov3_convnext_base', 'dinov3_vitb16',
]


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


def load_alexnet_ae(device):
    """加载微调后的AlexNet AutoEncoder模型 (encoder部分)"""
    from torchvision import models

    base_alexnet = models.alexnet(weights='IMAGENET1K_V1')
    checkpoint = torch.load(ALEXNET_AE_WEIGHTS, map_location=device, weights_only=False)

    class AlexNetAEEncoder(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.classifier = torch.nn.Sequential(
                base_model.classifier[0],
                base_model.classifier[1],
                base_model.classifier[2],
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    encoder = AlexNetAEEncoder(base_alexnet)
    encoder.load_state_dict(checkpoint['encoder'], strict=False)
    encoder.eval()
    encoder = encoder.to(device)
    return encoder


def extract_fc6_features(image_files, stimuli_path, alexnet, device, batch_size=32, gray=False):
    """提取图像的AlexNet fc6层特征"""
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
            img = _open_image(img_path, gray=gray)
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            x = alexnet.features(batch_tensor)
            x = alexnet.avgpool(x)
            x = torch.flatten(x, 1)
            fc6_activations = alexnet.classifier[1](x)
            fc6_activations = torch.nn.functional.relu(fc6_activations)
            fc6_features[i:i+len(batch_files)] = fc6_activations.cpu().numpy()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理 {min(i + batch_size, n_images)}/{n_images} 张图像")
    
    print(f"fc6特征提取完成，特征矩阵形状: {fc6_features.shape}")
    return fc6_features


def extract_fc7_features(image_files, stimuli_path, alexnet, device, batch_size=32, gray=False):
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
            img = _open_image(img_path, gray=gray)
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            x = alexnet.features(batch_tensor)
            x = alexnet.avgpool(x)
            x = torch.flatten(x, 1)
            # fc6
            x = alexnet.classifier[1](x)
            x = torch.nn.functional.relu(x)
            # fc7
            fc7_activations = alexnet.classifier[4](x)
            fc7_activations = torch.nn.functional.relu(fc7_activations)
            fc7_features[i:i+len(batch_files)] = fc7_activations.cpu().numpy()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理 {min(i + batch_size, n_images)}/{n_images} 张图像")
    
    print(f"fc7特征提取完成，特征矩阵形状: {fc7_features.shape}")
    return fc7_features


def extract_fc8_features(image_files, stimuli_path, alexnet, device, batch_size=32, gray=False):
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
            img = _open_image(img_path, gray=gray)
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
            # fc8 (不带softmax)
            fc8_activations = alexnet.classifier[6](x)
            fc8_features[i:i+len(batch_files)] = fc8_activations.cpu().numpy()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理 {min(i + batch_size, n_images)}/{n_images} 张图像")
    
    print(f"fc8特征提取完成，特征矩阵形状: {fc8_features.shape}")
    return fc8_features


def load_clip_model(model_name, checkpoint_path, device):
    """加载CLIP模型"""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=checkpoint_path,
        weights_only=False
    )
    model.eval()
    model = model.to(device)
    return model, preprocess


def extract_clip_image_features(image_files, stimuli_path, clip_model, preprocess, device, clip_dim=1024, batch_size=32, gray=False):
    """提取图像的CLIP特征
    
    Parameters:
    -----------
    clip_dim : int
        CLIP模型的特征维度 (ViT-L-14: 768, RN50/RN101: 1024)
    """
    n_images = len(image_files)
    
    clip_features = np.zeros((n_images, clip_dim), dtype=np.float32)
    
    print(f"开始使用CLIP提取 {n_images} 张图像的特征...")
    
    for i in range(0, n_images, batch_size):
        end_idx = min(i + batch_size, n_images)
        batch_files = image_files[i:end_idx]
        batch_images = []
        
        for filename in batch_files:
            img_path = os.path.join(stimuli_path, filename)
            img = _open_image(img_path, gray=gray)
            img_tensor = preprocess(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(batch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            clip_features[i:end_idx] = image_features.cpu().numpy()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理 {min(end_idx, n_images)}/{n_images} 张图像")
    
    print(f"CLIP图像特征提取完成，特征矩阵形状: {clip_features.shape}")
    return clip_features


def extract_clip_text_features(captions_path, clip_model, device, clip_dim=1024, batch_size=32):
    """提取coco captions的CLIP文本特征
    
    Parameters:
    -----------
    clip_dim : int
        CLIP模型的文本特征维度 (ViT-L-14: 768, RN50/RN101: 1024)
    """
    print("加载coco captions矩阵...")
    with open(captions_path, 'rb') as f:
        captions_matrix = pickle.load(f)
    
    n_images = captions_matrix.shape[0]
    clip_text_features = np.zeros((n_images, clip_dim), dtype=np.float32)
    
    # 获取CLIP tokenizer
    tokenizer = open_clip.get_tokenizer('RN50')  # 使用RN50的tokenizer
    
    print(f"开始使用CLIP提取 {n_images} 张图片的caption文本特征...")
    
    for i in range(0, n_images, batch_size):
        end_idx = min(i + batch_size, n_images)
        
        # 获取当前batch的所有captions (5个 per image)
        batch_captions = []
        for j in range(i, end_idx):
            batch_captions.extend(captions_matrix[j].tolist())
        
        # Tokenize
        tokens = tokenizer(batch_captions).to(device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_np = text_features.cpu().numpy()
        
        # 对每张图片的5个caption取平均
        for j in range(i, end_idx):
            start_emb = (j - i) * 5
            end_emb = start_emb + 5
            clip_text_features[j] = np.mean(text_features_np[start_emb:end_emb], axis=0)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  已处理 {min(end_idx, n_images)}/{n_images} 张图片")
    
    print(f"CLIP文本特征提取完成，特征矩阵形状: {clip_text_features.shape}")
    return clip_text_features


def load_sentence_model(device):
    """加载all-mpnet-base-v2 sentence transformer模型"""
    model = SentenceTransformer('/media/ubuntu/sda/TrippleN/model/all-mpnet-base-v2')
    model.eval()
    return model


def extract_caption_features(captions_path, model, batch_size=32):
    """提取coco captions的语义特征"""
    print("加载coco captions矩阵...")
    with open(captions_path, 'rb') as f:
        captions_matrix = pickle.load(f)
    
    n_images = captions_matrix.shape[0]
    print(f"Caption矩阵形状: {captions_matrix.shape}")
    
    caption_features = np.zeros((n_images, 768), dtype=np.float32)
    
    print(f"开始使用all-mpnet-base-v2提取 {n_images} 张图片的caption特征...")
    
    for i in range(0, n_images, batch_size):
        end_idx = min(i + batch_size, n_images)
        
        batch_captions = []
        for j in range(i, end_idx):
            batch_captions.extend(captions_matrix[j].tolist())
        
        embeddings = model.encode(batch_captions, show_progress_bar=False)
        
        for j in range(i, end_idx):
            start_emb = (j - i) * 5
            end_emb = start_emb + 5
            caption_features[j] = np.mean(embeddings[start_emb:end_emb], axis=0)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  已处理 {min(end_idx, n_images)}/{n_images} 张图片")
    
    print(f"Caption特征提取完成，特征矩阵形状: {caption_features.shape}")
    return caption_features


def load_dinov3_model(model_name, weights_path, device):
    model = torch.hub.load(DINOV3_REPO, model_name, source='local', weights=weights_path)
    model.eval()
    model = model.to(device)
    return model


def extract_dinov3_features(image_files, stimuli_path, model, device, feature_dim, batch_size=32, gray=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    n_images = len(image_files)
    features = np.zeros((n_images, feature_dim), dtype=np.float32)
    for i in range(0, n_images, batch_size):
        end_idx = min(i + batch_size, n_images)
        batch_files = image_files[i:end_idx]
        batch_images = []
        for filename in batch_files:
            img_path = os.path.join(stimuli_path, filename)
            img = _open_image(img_path, gray=gray)
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            out = model(batch_tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            features[i:end_idx] = out.cpu().float().numpy()
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理 {min(end_idx, n_images)}/{n_images} 张图像")
    print(f"DINOv3特征提取完成，特征矩阵形状: {features.shape}")
    return features


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


def compute_all_mse_and_correlations_gpu(Y_train_true, Y_train_pred_all, Y_test_true, Y_test_pred_all, n_components_range):
    """计算所有成分数的MSE和相关系数"""
    n_comps = len(n_components_range)
    n_neurons = Y_train_true.shape[1]
    device = Y_train_true.device
    
    train_mse = torch.zeros(n_comps, n_neurons, device=device, dtype=torch.float32)
    for i in range(n_comps):
        mse = torch.mean((Y_train_true - Y_train_pred_all[i]) ** 2, dim=0)
        train_mse[i] = mse
    
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


def evaluate_encoding_performance_gpu(features, neuron_responses, unit_info, encoding_type="vision"):
    """使用GPU评估所有神经元的编码性能"""
    n_neurons_total = neuron_responses.shape[0]
    n_images = neuron_responses.shape[1]
    reliability_best = unit_info['reliability_best'].values
    
    print(f"  总神经元数: {n_neurons_total}")
    print(f"  总图像数: {n_images}")
    
    n_component_candidates = list(range(MIN_COMPONENTS, MAX_COMPONENTS + 1))
    
    np.random.seed(42)
    indices = np.random.permutation(n_images)
    train_img_idx = indices[:900]
    test_img_idx = indices[900:]
    
    X_train = features[train_img_idx]
    X_test = features[test_img_idx]
    
    print(f"  图像训练集: {X_train.shape[0]} 样本")
    print(f"  图像测试集: {X_test.shape[0]} 样本")
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  使用设备: {device}")
    
    X_train_gpu = torch.from_numpy(X_train_scaled).float().to(device)
    X_test_gpu = torch.from_numpy(X_test_scaled).float().to(device)
    
    batch_size = GPU_BATCH_SIZE
    n_batches = int(np.ceil(n_neurons_total / batch_size))
    
    print(f"  GPU批次数: {n_batches}, 每批 {batch_size} 个神经元")
    
    encoding_correlations = np.zeros(n_neurons_total)
    normalized_correlations = np.zeros(n_neurons_total)
    best_components = np.zeros(n_neurons_total, dtype=int)
    
    pbar = tqdm.tqdm(
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
        reliability = reliability_best[i]
        if reliability > 0:
            normalized_correlations[i] = encoding_correlations[i] / reliability
        else:
            normalized_correlations[i] = 0.0
    
    unique_comps, counts = np.unique(best_components, return_counts=True)
    comp_dist = ", ".join([f"{c}:{cnt}" for c, cnt in zip(unique_comps, counts)])
    print(f"  最优成分数分布: {comp_dist}")
    print(f"  平均最优成分数: {np.mean(best_components):.1f}")
    
    print(f"  {encoding_type} encoding 完成!")
    
    return encoding_correlations, normalized_correlations, best_components


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
        
        print(f"\n  成分数分布:")
        unique_comps, counts = np.unique(best_components, return_counts=True)
        for comp, count in zip(unique_comps, counts):
            pct = count / len(best_components) * 100
            bar = "█" * int(pct / 2)
            print(f"    {comp:2d}: {bar} {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='GPU加速PLSR神经元活动预测')
    parser.add_argument('--models', nargs='*', default=None,
                        help='要运行的模型标识，不传则运行全部。可选: %s' % ', '.join(ALL_MODELS))
    parser.add_argument('--gray', action='store_true',
                        help='图像特征提取时使用灰度输入(L->RGB)，并在输出文件名/encoding_type后添加_gray')
    args = parser.parse_args()
    selected = args.models if args.models else ALL_MODELS
    for m in selected:
        if m not in ALL_MODELS:
            raise ValueError('未知模型: %s，可选: %s' % (m, ', '.join(ALL_MODELS)))

    start_time = datetime.now()
    print("="*60)
    print("GPU加速PLSR神经元活动预测 - 运行 %d 个模型" % len(selected))
    print("开始时间:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    stimuli_path = '/media/ubuntu/sda/TrippleN/stimuli'
    unit_info_path = '/media/ubuntu/sda/TrippleN/customize/aggregate_response/all_subjects_unit_info.pkl'
    captions_path = '/media/ubuntu/sda/TrippleN/customize/coco_captions_1000x5.pkl'
    neuron_responses_path = '/media/ubuntu/sda/TrippleN/customize/neuron_responses_1000.npy'
    encoding_dir = '/media/ubuntu/sda/TrippleN/customize/encoding_analysis'
    os.makedirs(encoding_dir, exist_ok=True)

    print("\n[1/3] 加载神经元响应数据...")
    neuron_responses = load_neuron_responses(neuron_responses_path)
    print("\n[2/3] 加载单元信息...")
    unit_info = pd.read_pickle(unit_info_path)
    print(f"  单元信息记录数: {len(unit_info)}")
    image_files = sorted([f for f in os.listdir(stimuli_path) if f.endswith('.bmp')])[:1000]
    print(f"  使用图像数量: {len(image_files)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    print("\n[3/3] 按模型提取特征并评估编码性能...")
    print("  前900张图片用于训练，后100张图片用于测试")
    models_results = {}
    suffix = "_gray" if args.gray else ""

    if 'alexnet' in selected:
        print("\n  --- AlexNet fc6 ---")
        alexnet = load_alexnet(device)
        fc6_features = extract_fc6_features(image_files, stimuli_path, alexnet, device, gray=args.gray)
        del alexnet
        clear_gpu_memory()
        corr, norm_corr, comps = evaluate_encoding_performance_gpu(
            fc6_features, neuron_responses, unit_info, encoding_type="alexnet" + suffix)
        save_results(corr, norm_corr, unit_info,
                     os.path.join(encoding_dir, 'alexnet%s_encoding_results_gpu.pkl' % suffix),
                     encoding_type="alexnet" + suffix, best_components=comps)
        print_summary(corr, norm_corr, encoding_type="alexnet" + suffix, best_components=comps)
        models_results["AlexNet fc6" + suffix] = (np.mean(corr), np.mean(norm_corr))
        del fc6_features
        clear_gpu_memory()

    if 'alexnet_ae' in selected:
        print("\n  --- AlexNet AE (微调) ---")
        alexnet_ae = load_alexnet_ae(device)
        fc6_ae_features = extract_fc6_features(image_files, stimuli_path, alexnet_ae, device, gray=args.gray)
        del alexnet_ae
        clear_gpu_memory()
        corr, norm_corr, comps = evaluate_encoding_performance_gpu(
            fc6_ae_features, neuron_responses, unit_info, encoding_type="alexnet_ae" + suffix)
        save_results(corr, norm_corr, unit_info,
                     os.path.join(encoding_dir, 'alexnet_ae%s_encoding_results_gpu.pkl' % suffix),
                     encoding_type="alexnet_ae" + suffix, best_components=comps)
        print_summary(corr, norm_corr, encoding_type="alexnet_ae" + suffix, best_components=comps)
        models_results["AlexNet AE" + suffix] = (np.mean(corr), np.mean(norm_corr))
        del fc6_ae_features
        clear_gpu_memory()

    if 'clip_vit_l14_image' in selected or 'clip_vit_l14_text' in selected:
        print("\n  --- CLIP ViT-L-14 ---")
        clip_vit_l14_model, clip_vit_l14_preprocess = load_clip_model(
            'ViT-L-14', '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt', device)
        if 'clip_vit_l14_image' in selected:
            vit_img = extract_clip_image_features(
                image_files, stimuli_path, clip_vit_l14_model, clip_vit_l14_preprocess, device, clip_dim=768, gray=args.gray)
            corr, norm_corr, comps = evaluate_encoding_performance_gpu(
                vit_img, neuron_responses, unit_info, encoding_type="clip_vit_l14_image" + suffix)
            save_results(corr, norm_corr, unit_info,
                         os.path.join(encoding_dir, 'clip_vit_l14_image%s_encoding_results_gpu.pkl' % suffix),
                         encoding_type="clip_vit_l14_image" + suffix, best_components=comps)
            print_summary(corr, norm_corr, encoding_type="clip_vit_l14_image" + suffix, best_components=comps)
            models_results["CLIP ViT-L-14 Image" + suffix] = (np.mean(corr), np.mean(norm_corr))
            del vit_img
            clear_gpu_memory()
        if 'clip_vit_l14_text' in selected:
            vit_txt = extract_clip_text_features(captions_path, clip_vit_l14_model, device, clip_dim=768)
            corr, norm_corr, comps = evaluate_encoding_performance_gpu(
                vit_txt, neuron_responses, unit_info, encoding_type="clip_vit_l14_text")
            save_results(corr, norm_corr, unit_info,
                         os.path.join(encoding_dir, 'clip_vit_l14_text_encoding_results_gpu.pkl'),
                         encoding_type="clip_vit_l14_text", best_components=comps)
            print_summary(corr, norm_corr, encoding_type="clip_vit_l14_text", best_components=comps)
            models_results["CLIP ViT-L-14 Text"] = (np.mean(corr), np.mean(norm_corr))
            del vit_txt
            clear_gpu_memory()
        del clip_vit_l14_model, clip_vit_l14_preprocess
        clear_gpu_memory()

    if 'clip_rn50_image' in selected or 'clip_rn50_text' in selected:
        print("\n  --- CLIP RN50 ---")
        clip_rn50_model, clip_rn50_preprocess = load_clip_model(
            'RN50', '/media/ubuntu/sda/TrippleN/model/RN50.pt', device)
        if 'clip_rn50_image' in selected:
            rn50_img = extract_clip_image_features(
                image_files, stimuli_path, clip_rn50_model, clip_rn50_preprocess, device, clip_dim=1024, gray=args.gray)
            corr, norm_corr, comps = evaluate_encoding_performance_gpu(
                rn50_img, neuron_responses, unit_info, encoding_type="clip_rn50_image" + suffix)
            save_results(corr, norm_corr, unit_info,
                         os.path.join(encoding_dir, 'clip_rn50_image%s_encoding_results_gpu.pkl' % suffix),
                         encoding_type="clip_rn50_image" + suffix, best_components=comps)
            print_summary(corr, norm_corr, encoding_type="clip_rn50_image" + suffix, best_components=comps)
            models_results["CLIP RN50 Image" + suffix] = (np.mean(corr), np.mean(norm_corr))
            del rn50_img
            clear_gpu_memory()
        if 'clip_rn50_text' in selected:
            rn50_txt = extract_clip_text_features(captions_path, clip_rn50_model, device, clip_dim=1024)
            corr, norm_corr, comps = evaluate_encoding_performance_gpu(
                rn50_txt, neuron_responses, unit_info, encoding_type="clip_rn50_text")
            save_results(corr, norm_corr, unit_info,
                         os.path.join(encoding_dir, 'clip_rn50_text_encoding_results_gpu.pkl'),
                         encoding_type="clip_rn50_text", best_components=comps)
            print_summary(corr, norm_corr, encoding_type="clip_rn50_text", best_components=comps)
            models_results["CLIP RN50 Text"] = (np.mean(corr), np.mean(norm_corr))
            del rn50_txt
            clear_gpu_memory()
        del clip_rn50_model, clip_rn50_preprocess
        clear_gpu_memory()

    if 'clip_rn101_image' in selected or 'clip_rn101_text' in selected:
        print("\n  --- CLIP RN101 ---")
        clip_rn101_model, clip_rn101_preprocess = load_clip_model(
            'RN101', '/media/ubuntu/sda/TrippleN/model/RN101.pt', device)
        if 'clip_rn101_image' in selected:
            rn101_img = extract_clip_image_features(
                image_files, stimuli_path, clip_rn101_model, clip_rn101_preprocess, device, clip_dim=512, gray=args.gray)
            corr, norm_corr, comps = evaluate_encoding_performance_gpu(
                rn101_img, neuron_responses, unit_info, encoding_type="clip_rn101_image" + suffix)
            save_results(corr, norm_corr, unit_info,
                         os.path.join(encoding_dir, 'clip_rn101_image%s_encoding_results_gpu.pkl' % suffix),
                         encoding_type="clip_rn101_image" + suffix, best_components=comps)
            print_summary(corr, norm_corr, encoding_type="clip_rn101_image" + suffix, best_components=comps)
            models_results["CLIP RN101 Image" + suffix] = (np.mean(corr), np.mean(norm_corr))
            del rn101_img
            clear_gpu_memory()
        if 'clip_rn101_text' in selected:
            rn101_txt = extract_clip_text_features(captions_path, clip_rn101_model, device, clip_dim=512)
            corr, norm_corr, comps = evaluate_encoding_performance_gpu(
                rn101_txt, neuron_responses, unit_info, encoding_type="clip_rn101_text")
            save_results(corr, norm_corr, unit_info,
                         os.path.join(encoding_dir, 'clip_rn101_text_encoding_results_gpu.pkl'),
                         encoding_type="clip_rn101_text", best_components=comps)
            print_summary(corr, norm_corr, encoding_type="clip_rn101_text", best_components=comps)
            models_results["CLIP RN101 Text"] = (np.mean(corr), np.mean(norm_corr))
            del rn101_txt
            clear_gpu_memory()
        del clip_rn101_model, clip_rn101_preprocess
        clear_gpu_memory()

    if 'all_mpnet_base_v2' in selected:
        print("\n  --- all-mpnet-base-v2 ---")
        sentence_model = load_sentence_model(device)
        sent_feat = extract_caption_features(captions_path, sentence_model)
        del sentence_model
        clear_gpu_memory()
        corr, norm_corr, comps = evaluate_encoding_performance_gpu(
            sent_feat, neuron_responses, unit_info, encoding_type="sentence")
        save_results(corr, norm_corr, unit_info,
                     os.path.join(encoding_dir, 'sentence_encoding_results_gpu.pkl'),
                     encoding_type="sentence", best_components=comps)
        print_summary(corr, norm_corr, encoding_type="sentence", best_components=comps)
        models_results["all-mpnet-base-v2"] = (np.mean(corr), np.mean(norm_corr))
        del sent_feat
        clear_gpu_memory()

    for dinov3_id in ('dinov3_vitl16', 'dinov3_convnext_base', 'dinov3_vitb16'):
        if dinov3_id not in selected:
            continue
        print("\n  --- %s ---" % dinov3_id)
        weights_path = os.path.join(DINOV3_WEIGHTS_DIR, DINOV3_WEIGHT_FILES[dinov3_id])
        dim = DINOV3_FEATURE_DIMS[dinov3_id]
        model = load_dinov3_model(dinov3_id, weights_path, device)
        features = extract_dinov3_features(
            image_files, stimuli_path, model, device, feature_dim=dim, gray=args.gray)
        del model
        clear_gpu_memory()
        corr, norm_corr, comps = evaluate_encoding_performance_gpu(
            features, neuron_responses, unit_info, encoding_type=dinov3_id + suffix)
        save_results(corr, norm_corr, unit_info,
                     os.path.join(encoding_dir, '%s%s_encoding_results_gpu.pkl' % (dinov3_id, suffix)),
                     encoding_type=dinov3_id + suffix, best_components=comps)
        print_summary(corr, norm_corr, encoding_type=dinov3_id + suffix, best_components=comps)
        models_results[dinov3_id + suffix] = (np.mean(corr), np.mean(norm_corr))
        del features
        clear_gpu_memory()

    print("\n" + "="*60)
    print("编码性能汇总 (%d 个模型)" % len(models_results))
    print("="*60)
    print(f"\n{'Model':<25} {'Corr':<12} {'Norm Corr':<12}")
    print("-" * 50)
    for model_name, (corr, norm_corr) in models_results.items():
        print(f"{model_name:<25} {corr:<12.4f} {norm_corr:<12.4f}")
    end_time = datetime.now()
    print(f"\n运行完成，耗时: {end_time - start_time}")
    print("="*60)


if __name__ == '__main__':
    main()
