#!/usr/bin/env python3
"""
使用RSA（Representational Similarity Analysis）分析模型嵌入预测脑活动的能力

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

分析类型：
1) 按cluster_tuning_type: 从all_subjects_unit_info_SI.pkl读取3种tuning类型，
   每种随机抽取1000个unit计算RDM并与各模型RSA，重复--n-repeats次(默认1000)
2) 按脑区: middle(含MF,MB,MO,LPP,CLC)与anterior(含AF,AB,AO,PITP,AMC)两组，
   每组随机抽取3000个神经元计算RDM并与各模型RSA，重复--n-repeats次(默认1000)
3) all: encoding筛选后随机抽取全体unit的20%%，与各模型RSA，固定重复200次(与--n-repeats无关)
4) session: 按unit_info的session_id分组，每个session用该session全部unit计算RSA，不重复抽样

用法: python compute_rsa.py [--models ...] [--n-repeats 1000] [--mode tuning|area|area_finer|all|session|both]
"""

import argparse
from typing import Any
import numpy as np
import pandas as pd
import os
import torch
from torchvision import models, transforms
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from datetime import datetime
import pickle
import tqdm
import warnings
from sentence_transformers import SentenceTransformer
import open_clip

warnings.filterwarnings('ignore')

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
ALL_MODELS = [
    'alexnet',
    'clip_vit_l14_text', 'clip_vit_l14_image',
    'clip_rn50_text', 'clip_rn50_image',
    'clip_rn101_text', 'clip_rn101_image',
    'all_mpnet_base_v2',
    'dinov3_vitl16', 'dinov3_convnext_base', 'dinov3_vitb16',
]

MODEL_DISPLAY_NAMES = {
    'alexnet': 'AlexNet fc6',
    'clip_vit_l14_text': 'CLIP ViT-L-14 Text',
    'clip_vit_l14_image': 'CLIP ViT-L-14 Image',
    'clip_rn50_text': 'CLIP RN50 Text',
    'clip_rn50_image': 'CLIP RN50 Image',
    'clip_rn101_text': 'CLIP RN101 Text',
    'clip_rn101_image': 'CLIP RN101 Image',
    'all_mpnet_base_v2': 'all-mpnet-base-v2',
    'dinov3_vitl16': 'dinov3_vitl16',
    'dinov3_convnext_base': 'dinov3_convnext_base',
    'dinov3_vitb16': 'dinov3_vitb16',
}

MIDDLE_AREAS = ['MF', 'MB', 'MO', 'LPP', 'CLC']
ANTERIOR_AREAS = ['AF', 'AB', 'AO', 'PITP', 'AMC']
REGION_GROUPS = {'middle': MIDDLE_AREAS, 'anterior': ANTERIOR_AREAS}

TUNING_SAMPLE = 1000
AREA_SAMPLE = 3000
AREA_FINER_SAMPLE = 200
ALL_SUBSAMPLE_FRACTION = 0.2
ALL_SUBSAMPLE_REPEATS = 200
N_REPEATS_DEFAULT = 1000


def compute_rdm(features, method='correlation'):
    """
    计算表示异质性矩阵（RDM）

    Parameters:
    -----------
    features : np.ndarray
        特征矩阵，shape为(n_samples, n_features)
    method : str
        计算距离的方法
        - 'correlation': 1 - correlation
        - 'euclidean': Euclidean distance
        - 'cosine': 1 - cosine similarity

    Returns:
    --------
    rdm : np.ndarray
        表示异质性矩阵，shape为(n_samples, n_samples)
    """
    if method == 'correlation':
        n_samples = features.shape[0]
        rdm = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if np.std(features[i]) > 0 and np.std(features[j]) > 0:
                    corr, _ = pearsonr(features[i], features[j])
                    dissimilarity = 1 - corr
                else:
                    dissimilarity = 0.0
                rdm[i, j] = dissimilarity
                rdm[j, i] = dissimilarity

    elif method == 'euclidean':
        pairwise_distances = pdist(features, metric='euclidean')
        rdm = squareform(pairwise_distances)

    elif method == 'cosine':
        normalized_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        cosine_similarity = np.dot(normalized_features, normalized_features.T)
        rdm = 1 - cosine_similarity

    return rdm


def compute_rdm_vectorized(features, method='correlation'):
    """
    使用向量化的方式计算RDM，更快

    Parameters:
    -----------
    features : np.ndarray
        特征矩阵，shape为(n_samples, n_features)
    method : str
        计算距离的方法

    Returns:
    --------
    rdm : np.ndarray
        表示异质性矩阵
    """
    n_samples = features.shape[0]

    if method == 'correlation':
        features_centered = features - features.mean(axis=1, keepdims=True)
        features_normalized = features_centered / (features.std(axis=1, keepdims=True) + 1e-8)

        corr_matrix = np.dot(features_normalized, features_normalized.T) / features.shape[1]
        rdm = 1 - corr_matrix

    elif method == 'euclidean':
        sq_dists = np.sum(features ** 2, axis=1)[:, np.newaxis] + \
                   np.sum(features ** 2, axis=1)[np.newaxis, :] - \
                   2 * np.dot(features, features.T)
        sq_dists = np.maximum(sq_dists, 0)
        rdm = np.sqrt(sq_dists)

    elif method == 'cosine':
        features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        cosine_sim = np.dot(features_normalized, features_normalized.T)
        rdm = 1 - cosine_sim

    np.fill_diagonal(rdm, 0)

    return rdm


def compute_neuron_rdm(neuron_responses, method='correlation'):
    """
    计算神经元响应的RDM

    Parameters:
    -----------
    neuron_responses : np.ndarray
        神经元响应矩阵，shape为(n_neurons, n_samples)
    method : str
        计算距离的方法

    Returns:
    --------
    rdm : np.ndarray
        表示异质性矩阵，shape为(n_samples, n_samples)
    """
    n_neurons, n_samples = neuron_responses.shape

    rdm = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            responses_i = neuron_responses[:, i]
            responses_j = neuron_responses[:, j]

            if np.std(responses_i) > 0 and np.std(responses_j) > 0:
                corr, _ = pearsonr(responses_i, responses_j)
                dissimilarity = 1 - corr
            else:
                dissimilarity = 0.0

            rdm[i, j] = dissimilarity
            rdm[j, i] = dissimilarity

    return rdm


def compute_neuron_rdm_vectorized(neuron_responses):
    """
    向量化计算神经元RDM

    Parameters:
    -----------
    neuron_responses : np.ndarray
        神经元响应矩阵，shape为(n_neurons, n_samples)

    Returns:
    --------
    rdm : np.ndarray
        表示异质性矩阵，shape为(n_samples, n_samples)
    """
    n_neurons, n_samples = neuron_responses.shape

    responses_centered = neuron_responses - neuron_responses.mean(axis=1, keepdims=True)
    responses_normalized = responses_centered / (responses_centered.std(axis=1, keepdims=True) + 1e-8)

    corr_matrix = np.dot(responses_normalized.T, responses_normalized) / n_neurons
    rdm = 1 - corr_matrix

    np.fill_diagonal(rdm, 0)

    return rdm


def compute_rsa_correlation(model_rdm, brain_rdm, method='spearman'):
    """
    计算模型RDM与脑活动RDM之间的RSA相关性

    Parameters:
    -----------
    model_rdm : np.ndarray
        模型嵌入的RDM，shape为(n_samples, n_samples)
    brain_rdm : np.ndarray
        脑活动的RDM，shape为(n_samples, n_samples)
    method : str
        相关性计算方法
        - 'spearman': Spearman相关系数（推荐）
        - 'pearson': Pearson相关系数

    Returns:
    --------
    correlation : float
        RSA相关系数
    p_value : float
        p值
    """
    upper_tri_model = model_rdm[np.triu_indices_from(model_rdm, k=1)]
    upper_tri_brain = brain_rdm[np.triu_indices_from(brain_rdm, k=1)]

    if method == 'spearman':
        correlation, p_value = spearmanr(upper_tri_model, upper_tri_brain)
    else:
        correlation, p_value = pearsonr(upper_tri_model, upper_tri_brain)

    return correlation, p_value


def load_alexnet(device):
    """加载预训练AlexNet模型"""
    alexnet = models.alexnet(weights='IMAGENET1K_V1')
    alexnet.eval()
    alexnet = alexnet.to(device)
    return alexnet


def extract_fc6_features(image_files, stimuli_path, alexnet, device, batch_size=32):
    """提取图像的AlexNet fc6层特征"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    n_images = len(image_files)
    fc6_features = np.zeros((n_images, 4096), dtype=np.float32)

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
            fc6_activations = alexnet.classifier[1](x)
            fc6_activations = torch.nn.functional.relu(fc6_activations)
            fc6_features[i:i+len(batch_files)] = fc6_activations.cpu().numpy()

    return fc6_features


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


def extract_clip_image_features(image_files, stimuli_path, clip_model, preprocess, device, clip_dim=1024, batch_size=32):
    """提取图像的CLIP特征"""
    n_images = len(image_files)
    clip_features = np.zeros((n_images, clip_dim), dtype=np.float32)

    for i in range(0, n_images, batch_size):
        end_idx = min(i + batch_size, n_images)
        batch_files = image_files[i:end_idx]
        batch_images = []

        for filename in batch_files:
            img_path = os.path.join(stimuli_path, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img)
            batch_images.append(img_tensor)

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(batch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            clip_features[i:end_idx] = image_features.cpu().numpy()

    return clip_features


def extract_clip_text_features(captions_path, clip_model, device, clip_dim=1024, batch_size=32):
    """提取coco captions的CLIP文本特征"""
    print("加载coco captions矩阵...")
    with open(captions_path, 'rb') as f:
        captions_matrix = pickle.load(f)

    n_images = captions_matrix.shape[0]
    clip_text_features = np.zeros((n_images, clip_dim), dtype=np.float32)

    tokenizer = open_clip.get_tokenizer('RN50')

    for i in range(0, n_images, batch_size):
        end_idx = min(i + batch_size, n_images)

        batch_captions = []
        for j in range(i, end_idx):
            batch_captions.extend(captions_matrix[j].tolist())

        tokens = tokenizer(batch_captions).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_np = text_features.cpu().numpy()

        for j in range(i, end_idx):
            start_emb = (j - i) * 5
            end_emb = start_emb + 5
            clip_text_features[j] = np.mean(text_features_np[start_emb:end_emb], axis=0)

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
    caption_features = np.zeros((n_images, 768), dtype=np.float32)

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

    return caption_features


def load_dinov3_model(model_name, weights_path, device):
    model = torch.hub.load(DINOV3_REPO, model_name, source='local', weights=weights_path)
    model.eval()
    model = model.to(device)
    return model


def extract_dinov3_features(image_files, stimuli_path, model, device, feature_dim, batch_size=32):
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
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            out = model(batch_tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            features[i:end_idx] = out.cpu().float().numpy()
    return features


def load_neuron_responses(responses_path):
    """加载预计算的神经元响应数据"""
    print(f"加载神经元响应数据: {responses_path}")
    neuron_responses = np.load(responses_path)
    print(f"  响应矩阵形状: {neuron_responses.shape}")
    print(f"  是否包含NaN: {np.any(np.isnan(neuron_responses))}")
    return neuron_responses


def load_unit_info(unit_info_path):
    """加载神经元单元信息"""
    print(f"加载神经元单元信息: {unit_info_path}")
    unit_info = pd.read_pickle(unit_info_path)
    print(f"  单元信息形状: {unit_info.shape}")
    print(f"  AREALABEL列唯一值: {unit_info['AREALABEL'].unique()}")
    return unit_info


def get_brain_region_indices(unit_info, prefix_list):
    """
    获取指定脑区前缀的神经元索引

    Parameters:
    -----------
    unit_info : pd.DataFrame
        单元信息DataFrame，包含AREALABEL列
    prefix_list : list
        脑区前缀列表，如['MF', 'MB', 'MO', 'AF', 'AB', 'AO']

    Returns:
    --------
    indices : np.ndarray
        符合条件的前缀对应的神经元索引
    """
    mask = unit_info['AREALABEL'].apply(lambda x: any(str(x).startswith(prefix) for prefix in prefix_list))
    indices = np.where(mask)[0]
    return indices


def get_tuning_type_indices(unit_info_si, tuning_type):
    """
    获取指定cluster_tuning_type的神经元索引
    """
    mask = unit_info_si['cluster_tuning_type'] == tuning_type
    return np.where(mask)[0]


def get_single_area_indices(unit_info, area_prefix):
    """
    获取单个脑区（AREALABEL以prefix开头）的神经元索引
    """
    mask = unit_info['AREALABEL'].apply(lambda x: str(x).startswith(area_prefix))
    return np.where(mask)[0]


def compute_brain_region_rdm(neuron_responses, indices):
    """
    计算特定脑区神经元的RDM

    Parameters:
    -----------
    neuron_responses : np.ndarray
        神经元响应矩阵，shape为(n_neurons, n_samples)
    indices : np.ndarray
        选中的神经元索引

    Returns:
    --------
    rdm : np.ndarray
        表示异质性矩阵，shape为(n_samples, n_samples)
    """
    if len(indices) == 0:
        return None

    region_responses = neuron_responses[indices, :]
    n_neurons, n_samples = region_responses.shape

    if n_neurons == 0:
        return None

    responses_centered = region_responses - region_responses.mean(axis=1, keepdims=True)
    responses_normalized = responses_centered / (responses_centered.std(axis=1, keepdims=True) + 1e-8)

    corr_matrix = np.dot(responses_normalized.T, responses_normalized) / n_neurons
    rdm = 1 - corr_matrix

    np.fill_diagonal(rdm, 0)

    return rdm


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def run_rsa_analysis(model_name, model_features, brain_rdm, output_path):
    """
    运行RSA分析

    Parameters:
    -----------
    model_name : str
        模型名称
    model_features : np.ndarray
        模型特征矩阵
    brain_rdm : np.ndarray
        脑活动RDM
    output_path : str
        输出文件路径

    Returns:
    --------
    results : dict
        RSA分析结果
    """
    print(f"\n  分析模型: {model_name}")

    print(f"    特征矩阵形状: {model_features.shape}")

    print("    计算模型嵌入RDM...")
    model_rdm = compute_rdm_vectorized(model_features, method='correlation')

    print("    计算RSA相关性...")
    spearman_corr, spearman_p = compute_rsa_correlation(model_rdm, brain_rdm, method='spearman')
    pearson_corr, pearson_p = compute_rsa_correlation(model_rdm, brain_rdm, method='pearson')

    results = {
        'model_name': model_name,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p,
        'pearson_r': pearson_corr,
        'pearson_p': pearson_p,
        'model_rdm': model_rdm,
        'n_features': model_features.shape[1]
    }

    print(f"    Spearman相关系数: {spearman_corr:.4f} (p = {spearman_p:.2e})")
    print(f"    Pearson相关系数:  {pearson_corr:.4f} (p = {pearson_p:.2e})")

    return results


def run_rsa_analysis_region(model_name, region_name, model_features, region_rdm, output_path):
    """
    运行特定脑区的RSA分析

    Parameters:
    -----------
    model_name : str
        模型名称
    region_name : str
        脑区名称
    model_features : np.ndarray
        模型特征矩阵
    region_rdm : np.ndarray
        脑区活动RDM
    output_path : str
        输出文件路径

    Returns:
    --------
    results : dict
        RSA分析结果
    """
    if region_rdm is None:
        return None

    print(f"\n    [{region_name}] 分析模型: {model_name}")

    print(f"      特征矩阵形状: {model_features.shape}")

    print("      计算模型嵌入RDM...")
    model_rdm = compute_rdm_vectorized(model_features, method='correlation')

    print("      计算RSA相关性...")
    spearman_corr, spearman_p = compute_rsa_correlation(model_rdm, region_rdm, method='spearman')
    pearson_corr, pearson_p = compute_rsa_correlation(model_rdm, region_rdm, method='pearson')

    results = {
        'model_name': model_name,
        'region_name': region_name,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p,
        'pearson_r': pearson_corr,
        'pearson_p': pearson_p,
        'model_rdm': model_rdm,
        'n_features': model_features.shape[1]
    }

    print(f"      Spearman相关系数: {spearman_corr:.4f} (p = {spearman_p:.2e})")
    print(f"      Pearson相关系数:  {pearson_corr:.4f} (p = {pearson_p:.2e})")

    return results


def run_rsa_batch(model_rdms, brain_rdm, category_name, category_value, repeat_idx):
    """
    使用预计算的模型RDM与单个brain_rdm计算所有模型的RSA，用于批量循环
    """
    results = []
    for model_name, model_rdm in model_rdms.items():
        spearman_corr, spearman_p = compute_rsa_correlation(model_rdm, brain_rdm, method='spearman')
        pearson_corr, pearson_p = compute_rsa_correlation(model_rdm, brain_rdm, method='pearson')
        results.append({
            'model_name': model_name,
            category_name: category_value,
            'repeat': repeat_idx,
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p,
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
        })
    return results


def print_rsa_summary(all_results, region_results=None):
    """打印RSA分析汇总"""
    print("\n" + "="*70)
    print("RSA分析结果汇总 - 全体神经元")
    print("="*70)

    print(f"\n{'Model':<30} {'Spearman r':<15} {'Pearson r':<15}")
    print("-" * 60)

    for result in all_results:
        print(f"{result['model_name']:<30} {result['spearman_r']:<15.4f} {result['pearson_r']:<15.4f}")

    print("\n" + "="*70)
    print("说明: RSA相关系数越高，表示模型嵌入与脑活动的表示结构越相似")
    print("      使用Spearman相关是因为它对RDM中的排名关系更敏感")
    print("="*70)

    if region_results is not None and len(region_results) > 0:
        unique_regions = set([r['region_name'] for r in region_results])
        for region in sorted(unique_regions):
            print(f"\n{'='*70}")
            print(f"RSA分析结果汇总 - {region} 脑区 ".format(
                len(set(r.get('repeat') for r in region_results if r['region_name'] == region))))
            print("="*70)

            region_filtered = [r for r in region_results if r['region_name'] == region]
            agg = pd.DataFrame(region_filtered).groupby('model_name').agg(
                spearman_r_mean=('spearman_r', 'mean'),
                spearman_r_std=('spearman_r', 'std'),
                pearson_r_mean=('pearson_r', 'mean'),
                pearson_r_std=('pearson_r', 'std')
            ).reset_index()
            print(f"\n{'Model':<30} {'Spearman r (mean±std)':<25} {'Pearson r (mean±std)':<25}")
            print("-" * 80)
            for _, row in agg.iterrows():
                sr = f"{row['spearman_r_mean']:.4f}±{row['spearman_r_std']:.4f}" if pd.notna(row['spearman_r_std']) else f"{row['spearman_r_mean']:.4f}"
                pr = f"{row['pearson_r_mean']:.4f}±{row['pearson_r_std']:.4f}" if pd.notna(row['pearson_r_std']) else f"{row['pearson_r_mean']:.4f}"
                print(f"{row['model_name']:<30} {sr:<25} {pr:<25}")


def save_rsa_results(all_results, output_path, region_results=None):
    """保存RSA分析结果"""
    summary_data = []
    for result in all_results:
        summary_data.append({
            'model_name': result['model_name'],
            'spearman_r': result['spearman_r'],
            'spearman_p': result['spearman_p'],
            'pearson_r': result['pearson_r'],
            'pearson_p': result['pearson_p'],
            'n_features': result['n_features'],
            'region_name': 'all'
        })

    summary_df = pd.DataFrame(summary_data)

    if region_results is not None and len(region_results) > 0:
        region_data = []
        for result in region_results:
            region_data.append({
                'model_name': result['model_name'],
                'spearman_r': result['spearman_r'],
                'spearman_p': result['spearman_p'],
                'pearson_r': result['pearson_r'],
                'pearson_p': result['pearson_p'],
                'n_features': result['n_features'],
                'region_name': result['region_name'],
                'repeat': result.get('repeat', None)
            })
        region_df = pd.DataFrame(region_data)
        summary_df = pd.concat([summary_df, region_df], ignore_index=True)

    summary_df.to_pickle(output_path)
    print(f"RSA分析结果已保存到: {output_path}")

    return summary_df


def compute_and_save_relative_rsa(region_results, output_dir, alexnet_name='AlexNet fc6'):
    """
    以 AlexNet 为分母计算相对 RSA (rsa_model / rsa_alexnet)，
    保存形状为 (n_rep, n_model, 2) 的矩阵，最后一维为 [middle, anterior]。
    """
    if not region_results:
        return
    region_order = ['middle', 'anterior']
    df = pd.DataFrame(region_results)
    reps = sorted(df['repeat'].dropna().unique())
    n_rep = len(reps)
    model_names = sorted([m for m in df['model_name'].unique() if m != alexnet_name])
    n_model = len(model_names)
    alexnet_r = np.zeros((n_rep, 2))
    for i, rep in enumerate(reps):
        for j, reg in enumerate(region_order):
            row = df[(df['repeat'] == rep) & (df['region_name'] == reg) & (df['model_name'] == alexnet_name)]
            if len(row) > 0:
                alexnet_r[i, j] = row['spearman_r'].iloc[0]
            else:
                alexnet_r[i, j] = np.nan
    rel_rsa = np.full((n_rep, n_model, 2), np.nan)
    for rep_i, rep in enumerate(reps):
        for model_j, model_name in enumerate[Any](model_names):
            for reg_k, reg in enumerate(region_order):
                row = df[(df['repeat'] == rep) & (df['region_name'] == reg) & (df['model_name'] == model_name)]
                if len(row) > 0 and np.isfinite(alexnet_r[rep_i, reg_k]) and alexnet_r[rep_i, reg_k] != 0:
                    rel_rsa[rep_i, model_j, reg_k] = row['spearman_r'].iloc[0] / alexnet_r[rep_i, reg_k]
    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, 'relative_rsa_matrix.npy')
    names_path = os.path.join(output_dir, 'relative_rsa_model_names.pkl')
    np.save(matrix_path, rel_rsa)
    with open(names_path, 'wb') as f:
        pickle.dump(model_names, f)
    print(f"相对RSA矩阵 (n_rep={n_rep}, n_model={n_model}, 2=[middle,anterior]) 已保存: {matrix_path}")
    print(f"模型顺序 (分母为 {alexnet_name}) 已保存: {names_path}")


def save_tuning_rsa_results(tuning_results, output_path):
    """保存按tuning type分组的RSA结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame([{
        'model_name': r['model_name'],
        'tuning_type': r['tuning_type'],
        'repeat': r['repeat'],
        'spearman_r': r['spearman_r'],
        'spearman_p': r['spearman_p'],
        'pearson_r': r['pearson_r'],
        'pearson_p': r['pearson_p'],
    } for r in tuning_results])
    df.to_pickle(output_path)
    print(f"Tuning RSA结果已保存: {output_path}")


def save_area_rsa_results(area_results, output_path):
    """保存按脑区分组的RSA结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame([{
        'model_name': r['model_name'],
        'area': r['area'],
        'repeat': r['repeat'],
        'spearman_r': r['spearman_r'],
        'spearman_p': r['spearman_p'],
        'pearson_r': r['pearson_r'],
        'pearson_p': r['pearson_p'],
    } for r in area_results])
    df.to_pickle(output_path)
    print(f"Area RSA结果已保存: {output_path}")


def save_all_subsample_rsa_results(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame([{
        'model_name': r['model_name'],
        'repeat': r['repeat'],
        'n_units_sampled': r['n_units_sampled'],
        'spearman_r': r['spearman_r'],
        'spearman_p': r['spearman_p'],
        'pearson_r': r['pearson_r'],
        'pearson_p': r['pearson_p'],
    } for r in rows])
    df.to_pickle(output_path)
    print(f"All subsample RSA结果已保存: {output_path}")


def save_session_rsa_results(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame([{
        'model_name': r['model_name'],
        'session_id': r['session_id'],
        'n_units': r['n_units'],
        'repeat': r['repeat'],
        'spearman_r': r['spearman_r'],
        'spearman_p': r['spearman_p'],
        'pearson_r': r['pearson_r'],
        'pearson_p': r['pearson_p'],
    } for r in rows])
    df.to_pickle(output_path)
    print(f"Session RSA结果已保存: {output_path}")


def load_encoding_filter_mask(encoding_dir, threshold=0.3):
    """
    加载encoding结果并返回normalized_correlation阈值过滤mask
    """
    print(f"加载encoding结果用于筛选unit: {encoding_dir}")
    encoding_files = [f for f in os.listdir(encoding_dir) if f.endswith('.pkl')]
    encoding_results = {}
    for f in sorted(encoding_files):
        file_path = os.path.join(encoding_dir, f)
        model_name = f.replace('_encoding_results_gpu.pkl', '').replace('_', ' ')
        with open(file_path, 'rb') as fp:
            data = pickle.load(fp)
        encoding_results[model_name] = data

    alexnet_key = None
    for name in encoding_results.keys():
        if 'alexnet' in name.lower():
            alexnet_key = name
            break
    if alexnet_key is None:
        raise ValueError("未找到 alexnet 的 encoding 结果")

    alexnet_nc = np.asarray(encoding_results[alexnet_key]['normalized_correlation']).ravel()
    mask = alexnet_nc > threshold
    print(f"  normalized_correlation > {threshold}: {int(mask.sum())}/{len(mask)}")
    return mask


def compute_all_model_rdms(selected, image_files, stimuli_path, captions_path, device):
    """
    加载所有选中模型，提取特征并计算RDM，返回{display_name: model_rdm}
    """
    model_rdms = {}
    n_samples = len(image_files)

    if 'alexnet' in selected:
        print("  计算 AlexNet fc6 RDM...")
        alexnet = load_alexnet(device)
        fc6 = extract_fc6_features(image_files, stimuli_path, alexnet, device)
        del alexnet
        clear_gpu_memory()
        model_rdms['AlexNet fc6'] = compute_rdm_vectorized(fc6, method='correlation')
        del fc6
        clear_gpu_memory()

    if 'clip_vit_l14_image' in selected or 'clip_vit_l14_text' in selected:
        clip_m, clip_p = load_clip_model('ViT-L-14', '/media/ubuntu/sda/TrippleN/model/ViT-L-14.pt', device)
        if 'clip_vit_l14_image' in selected:
            print("  计算 CLIP ViT-L-14 Image RDM...")
            f = extract_clip_image_features(image_files, stimuli_path, clip_m, clip_p, device, clip_dim=768)
            model_rdms['CLIP ViT-L-14 Image'] = compute_rdm_vectorized(f, method='correlation')
            del f
            clear_gpu_memory()
        if 'clip_vit_l14_text' in selected:
            print("  计算 CLIP ViT-L-14 Text RDM...")
            f = extract_clip_text_features(captions_path, clip_m, device, clip_dim=768)
            model_rdms['CLIP ViT-L-14 Text'] = compute_rdm_vectorized(f, method='correlation')
            del f
            clear_gpu_memory()
        del clip_m, clip_p
        clear_gpu_memory()

    if 'clip_rn50_image' in selected or 'clip_rn50_text' in selected:
        clip_m, clip_p = load_clip_model('RN50', '/media/ubuntu/sda/TrippleN/model/RN50.pt', device)
        if 'clip_rn50_image' in selected:
            print("  计算 CLIP RN50 Image RDM...")
            f = extract_clip_image_features(image_files, stimuli_path, clip_m, clip_p, device, clip_dim=1024)
            model_rdms['CLIP RN50 Image'] = compute_rdm_vectorized(f, method='correlation')
            del f
            clear_gpu_memory()
        if 'clip_rn50_text' in selected:
            print("  计算 CLIP RN50 Text RDM...")
            f = extract_clip_text_features(captions_path, clip_m, device, clip_dim=1024)
            model_rdms['CLIP RN50 Text'] = compute_rdm_vectorized(f, method='correlation')
            del f
            clear_gpu_memory()
        del clip_m, clip_p
        clear_gpu_memory()

    if 'clip_rn101_image' in selected or 'clip_rn101_text' in selected:
        clip_m, clip_p = load_clip_model('RN101', '/media/ubuntu/sda/TrippleN/model/RN101.pt', device)
        if 'clip_rn101_image' in selected:
            print("  计算 CLIP RN101 Image RDM...")
            f = extract_clip_image_features(image_files, stimuli_path, clip_m, clip_p, device, clip_dim=512)
            model_rdms['CLIP RN101 Image'] = compute_rdm_vectorized(f, method='correlation')
            del f
            clear_gpu_memory()
        if 'clip_rn101_text' in selected:
            print("  计算 CLIP RN101 Text RDM...")
            f = extract_clip_text_features(captions_path, clip_m, device, clip_dim=512)
            model_rdms['CLIP RN101 Text'] = compute_rdm_vectorized(f, method='correlation')
            del f
            clear_gpu_memory()
        del clip_m, clip_p
        clear_gpu_memory()

    if 'all_mpnet_base_v2' in selected:
        print("  计算 all-mpnet-base-v2 RDM...")
        sent_m = load_sentence_model(device)
        f = extract_caption_features(captions_path, sent_m)
        model_rdms['all-mpnet-base-v2'] = compute_rdm_vectorized(f, method='correlation')
        del sent_m, f
        clear_gpu_memory()

    for dinov3_id in ('dinov3_vitl16', 'dinov3_convnext_base', 'dinov3_vitb16'):
        if dinov3_id not in selected:
            continue
        print(f"  计算 {dinov3_id} RDM...")
        wp = os.path.join(DINOV3_WEIGHTS_DIR, DINOV3_WEIGHT_FILES[dinov3_id])
        dim = DINOV3_FEATURE_DIMS[dinov3_id]
        m = load_dinov3_model(dinov3_id, wp, device)
        f = extract_dinov3_features(image_files, stimuli_path, m, device, dim)
        del m
        clear_gpu_memory()
        model_rdms[dinov3_id] = compute_rdm_vectorized(f, method='correlation')
        del f
        clear_gpu_memory()

    return model_rdms


def main():
    parser = argparse.ArgumentParser(description='RSA分析')
    parser.add_argument('--models', nargs='*', default=None,
                        help='要运行的模型标识，不传则运行全部。可选: %s' % ', '.join(ALL_MODELS))
    parser.add_argument('--n-repeats', type=int, default=N_REPEATS_DEFAULT,
                        help='抽样重复次数，默认%d' % N_REPEATS_DEFAULT)
    parser.add_argument('--mode', choices=['tuning', 'area', 'area_finer', 'all', 'session', 'both'], default='both',
                        help='tuning/area/area_finer/all/session/both(仅tuning+area); all固定200次20%%抽样')
    args = parser.parse_args()
    selected = args.models if args.models else ALL_MODELS
    for m in selected:
        if m not in ALL_MODELS:
            raise ValueError('未知模型: %s，可选: %s' % (m, ', '.join(ALL_MODELS)))

    n_repeats = args.n_repeats
    start_time = datetime.now()
    print("="*70)
    print("RSA（Representational Similarity Analysis）分析")
    print("1) 按cluster_tuning_type: 每类随机1000 units x %d次" % n_repeats)
    print("2) 按脑区: middle/anterior各随机3000 neurons x %d次" % n_repeats)
    print("3) all: 筛后全体随机20%% x %d次" % ALL_SUBSAMPLE_REPEATS)
    print("4) session: 按session_id全unit各一次")
    print("开始时间:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)

    base_dir = '/media/ubuntu/sda/TrippleN'
    stimuli_path = os.path.join(base_dir, 'stimuli')
    captions_path = os.path.join(base_dir, 'customize/coco_captions_1000x5.pkl')
    neuron_responses_path = os.path.join(base_dir, 'customize/neuron_responses_1000.npy')
    unit_info_path = os.path.join(base_dir, 'customize/aggregate_response/all_subjects_unit_info.pkl')
    unit_info_si_path = os.path.join(base_dir, 'customize/aggregate_response/all_subjects_unit_info_SI.pkl')
    encoding_dir = os.path.join(base_dir, 'customize/encoding_analysis/encoding_results')
    rsa_output_dir = os.path.join(base_dir, 'customize/RSA_analysis')
    tuning_output_path = os.path.join(rsa_output_dir, 'rsa_results_by_tuning.pkl')
    area_output_path = os.path.join(rsa_output_dir, 'rsa_results_by_area.pkl')
    area_finer_output_path = os.path.join(rsa_output_dir, 'rsa_results_by_area_finer.pkl')
    all_subsample_output_path = os.path.join(rsa_output_dir, 'rsa_results_all_subsample20pct.pkl')
    session_output_path = os.path.join(rsa_output_dir, 'rsa_results_by_session.pkl')

    image_files = sorted([f for f in os.listdir(stimuli_path) if f.endswith('.bmp')])[:1000]
    print(f"\n使用图像数量: {len(image_files)}")

    print("\n[1/5] 加载神经元响应和单元信息...")
    neuron_responses = load_neuron_responses(neuron_responses_path)
    unit_info = load_unit_info(unit_info_path)
    print(f"加载unit_info_SI: {unit_info_si_path}")
    unit_info_si = pd.read_pickle(unit_info_si_path)
    if unit_info_si.shape[0] != neuron_responses.shape[0]:
        raise ValueError("unit_info_si行数(%d)与neuron_responses(%d)不一致" %
                         (unit_info_si.shape[0], neuron_responses.shape[0]))
    if 'cluster_tuning_type' not in unit_info_si.columns:
        raise ValueError("unit_info_SI缺少cluster_tuning_type列")

    encoding_mask = load_encoding_filter_mask(encoding_dir, threshold=0.3)
    if len(encoding_mask) != neuron_responses.shape[0]:
        raise ValueError("encoding筛选mask长度(%d)与neuron_responses(%d)不一致" %
                         (len(encoding_mask), neuron_responses.shape[0]))
    neuron_responses = neuron_responses[encoding_mask]
    unit_info = unit_info.iloc[encoding_mask].reset_index(drop=True)
    unit_info_si = unit_info_si.iloc[encoding_mask].reset_index(drop=True)
    print(f"  筛选后units: {neuron_responses.shape[0]}")

    tuning_types = sorted(unit_info_si['cluster_tuning_type'].unique().tolist())
    print(f"  cluster_tuning_type: {tuning_types}")
    for t in tuning_types:
        n = (unit_info_si['cluster_tuning_type'] == t).sum()
        print(f"    {t}: {n} 个units")

    print("\n[2/5] 获取各类索引...")
    tuning_indices = {t: get_tuning_type_indices(unit_info_si, t) for t in tuning_types}
    region_indices = {r: get_brain_region_indices(unit_info, prefixes) for r, prefixes in REGION_GROUPS.items()}
    area_finer_prefixes = MIDDLE_AREAS + ANTERIOR_AREAS
    area_finer_indices = {a: get_single_area_indices(unit_info, a) for a in area_finer_prefixes}
    for r in REGION_GROUPS:
        print(f"  {r}: {len(region_indices[r])} 个neurons")
    for a in area_finer_prefixes:
        print(f"  finer-{a}: {len(area_finer_indices[a])} 个neurons")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    print("\n[3/5] 计算所有模型RDM...")
    model_rdms = compute_all_model_rdms(selected, image_files, stimuli_path, captions_path, device)
    print(f"  已计算 {len(model_rdms)} 个模型的RDM")

    tuning_results = []
    area_results = []
    area_finer_results = []
    all_subsample_results = []
    session_results = []

    if args.mode in ('tuning', 'both'):
        print("\n[4/5] 按tuning type计算RSA (每类%d units x %d repeats)..." % (TUNING_SAMPLE, n_repeats))
        for rep in tqdm.tqdm(range(n_repeats), desc="tuning"):
            rng = np.random.default_rng(rep)
            for tuning_type in tuning_types:
                idx = tuning_indices[tuning_type]
                if len(idx) == 0:
                    continue
                replace = len(idx) < TUNING_SAMPLE
                sampled = rng.choice(idx, size=TUNING_SAMPLE, replace=replace)
                brain_rdm = compute_brain_region_rdm(neuron_responses, sampled)
                if brain_rdm is None:
                    continue
                batch = run_rsa_batch(model_rdms, brain_rdm, 'tuning_type', tuning_type, rep)
                tuning_results.extend(batch)

    if args.mode in ('area', 'both'):
        print("\n[4/5] 按脑区计算RSA (middle/anterior各%d neurons x %d repeats)..." % (AREA_SAMPLE, n_repeats))
        for rep in tqdm.tqdm(range(n_repeats), desc="area"):
            rng = np.random.default_rng(rep + 10000)
            for region_name in REGION_GROUPS:
                idx = region_indices[region_name]
                if len(idx) == 0:
                    continue
                replace = len(idx) < AREA_SAMPLE
                sampled = rng.choice(idx, size=AREA_SAMPLE, replace=replace)
                brain_rdm = compute_brain_region_rdm(neuron_responses, sampled)
                if brain_rdm is None:
                    continue
                batch = run_rsa_batch(model_rdms, brain_rdm, 'area', region_name, rep)
                area_results.extend(batch)

    if args.mode in ('area_finer',):
        print("\n[4/5] 按细脑区计算RSA (10个脑区各%d neurons x %d repeats)..." % (AREA_FINER_SAMPLE, n_repeats))
        for rep in tqdm.tqdm(range(n_repeats), desc="area_finer"):
            rng = np.random.default_rng(rep + 20000)
            for area_name in area_finer_prefixes:
                idx = area_finer_indices[area_name]
                if len(idx) == 0:
                    continue
                replace = len(idx) < AREA_FINER_SAMPLE
                sampled = rng.choice(idx, size=AREA_FINER_SAMPLE, replace=replace)
                brain_rdm = compute_brain_region_rdm(neuron_responses, sampled)
                if brain_rdm is None:
                    continue
                batch = run_rsa_batch(model_rdms, brain_rdm, 'area', area_name, rep)
                area_finer_results.extend(batch)

    if args.mode in ('all',):
        n_all = neuron_responses.shape[0]
        if n_all < 2:
            raise ValueError('筛选后unit不足2个，无法计算RSA(all模式)')
        k = max(1, int(round(n_all * ALL_SUBSAMPLE_FRACTION)))
        print("\n[4/5] all模式: 每次随机%d/%d个unit (约%.0f%%), 重复%d次..." % (
            k, n_all, 100.0 * k / n_all, ALL_SUBSAMPLE_REPEATS))
        all_idx = np.arange(n_all)
        for rep in tqdm.tqdm(range(ALL_SUBSAMPLE_REPEATS), desc='all_subsample'):
            rng = np.random.default_rng(rep + 40000)
            replace = n_all < k
            sampled = rng.choice(all_idx, size=k, replace=replace)
            brain_rdm = compute_brain_region_rdm(neuron_responses, sampled)
            if brain_rdm is None:
                continue
            batch = run_rsa_batch(model_rdms, brain_rdm, 'scope', 'all_20pct', rep)
            for row in batch:
                row['n_units_sampled'] = k
            all_subsample_results.extend(batch)

    if args.mode in ('session',):
        if 'session_id' not in unit_info.columns:
            raise ValueError('unit_info缺少session_id列，无法运行session模式')
        sessions = unit_info['session_id'].unique()
        print("\n[4/5] session模式: %d 个session，每session用全部unit..." % len(sessions))
        for sid in tqdm.tqdm(sessions, desc='session'):
            idx = np.where(unit_info['session_id'].to_numpy() == sid)[0]
            if len(idx) < 2:
                continue
            brain_rdm = compute_brain_region_rdm(neuron_responses, idx)
            if brain_rdm is None:
                continue
            batch = run_rsa_batch(model_rdms, brain_rdm, 'session_id', sid, 0)
            for row in batch:
                row['n_units'] = len(idx)
            session_results.extend(batch)

    print("\n[5/5] 保存结果...")
    if tuning_results:
        save_tuning_rsa_results(tuning_results, tuning_output_path)
    if area_results:
        save_area_rsa_results(area_results, area_output_path)
    if area_finer_results:
        save_area_rsa_results(area_finer_results, area_finer_output_path)
    if all_subsample_results:
        save_all_subsample_rsa_results(all_subsample_results, all_subsample_output_path)
    if session_results:
        save_session_rsa_results(session_results, session_output_path)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nRSA分析完成，耗时: {duration}")
    print("="*70)


if __name__ == '__main__':
    main()
