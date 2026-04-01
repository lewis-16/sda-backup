#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
import os
from scipy.stats import spearmanr
import torch
from torchvision import models, transforms
from PIL import Image

BASE_DIR = '/media/ubuntu/sda/TrippleN'
neuron_responses_path = os.path.join(BASE_DIR, 'customize', 'neuron_responses_1000.npy')
encoding_dir = os.path.join(BASE_DIR, 'customize', 'encoding_analysis')
rsa_results_path = os.path.join(BASE_DIR, 'customize', 'RSA_analysis', 'rsa_results.pkl')
stimuli_path = os.path.join(BASE_DIR, 'stimuli')

print("="*70)
print("分析：按encoding性能分组的RSA差异")
print("="*70)

print("\n[1/5] 加载数据...")
neuron_responses = np.load(neuron_responses_path)
print(f"  神经元响应形状: {neuron_responses.shape} (n_neurons, n_images)")

alexnet_encoding = pd.read_pickle(os.path.join(encoding_dir, 'alexnet_encoding_results_gpu.pkl'))
print(f"  AlexNet encoding结果: {len(alexnet_encoding)} 个units")

print("\n[2/5] 按encoding性能排序并分组...")
encoding_perf = alexnet_encoding['normalized_correlation'].values
sorted_indices = np.argsort(encoding_perf)[::-1]
n_units = len(sorted_indices)
n_top = n_units // 2

top50_indices = sorted_indices[:n_top]
bottom50_indices = sorted_indices[n_top:]

print(f"  总units数: {n_units}")
print(f"  前50% units: {n_top} 个")
print(f"    平均encoding性能: {encoding_perf[top50_indices].mean():.4f}")
print(f"    性能范围: {encoding_perf[top50_indices].min():.4f} - {encoding_perf[top50_indices].max():.4f}")
print(f"  后50% units: {n_units - n_top} 个")
print(f"    平均encoding性能: {encoding_perf[bottom50_indices].mean():.4f}")
print(f"    性能范围: {encoding_perf[bottom50_indices].min():.4f} - {encoding_perf[bottom50_indices].max():.4f}")

print("\n[3/5] 计算RDM...")
def compute_rdm_vectorized(responses):
    n_neurons, n_samples = responses.shape
    responses_centered = responses - responses.mean(axis=1, keepdims=True)
    responses_normalized = responses_centered / (responses_centered.std(axis=1, keepdims=True) + 1e-8)
    corr_matrix = np.dot(responses_normalized.T, responses_normalized) / n_neurons
    rdm = 1 - corr_matrix
    np.fill_diagonal(rdm, 0)
    return rdm

print("  计算全体RDM...")
all_rdm = compute_rdm_vectorized(neuron_responses)
print("  计算前50% RDM...")
top50_rdm = compute_rdm_vectorized(neuron_responses[top50_indices])
print("  计算后50% RDM...")
bottom50_rdm = compute_rdm_vectorized(neuron_responses[bottom50_indices])

print("\n[4/5] 加载模型RDM...")
rsa_data = pd.read_pickle(rsa_results_path)
alexnet_all_rsa = rsa_data[(rsa_data['model_name'] == 'AlexNet fc6') & (rsa_data['region_name'] == 'all')]['spearman_r'].values[0]
print(f"  全体units的RSA (已计算): {alexnet_all_rsa:.4f}")

image_files = sorted([f for f in os.listdir(stimuli_path) if f.endswith('.bmp')])[:1000]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  使用设备: {device}")

alexnet = models.alexnet(pretrained=True).to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

fc6_features = []
batch_size = 32
print("  提取AlexNet fc6特征...")
with torch.no_grad():
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        for filename in batch_files:
            img_path = os.path.join(stimuli_path, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        x = alexnet.features(batch_tensor)
        x = alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        fc6 = alexnet.classifier[1](x)
        fc6 = torch.nn.functional.relu(fc6)
        fc6_features.append(fc6.cpu().numpy())
        
        if (i // batch_size + 1) % 20 == 0:
            print(f"    已处理 {min(i+batch_size, len(image_files))}/{len(image_files)} 张图像")

fc6_features = np.vstack(fc6_features)
print(f"  AlexNet fc6特征形状: {fc6_features.shape}")

def compute_model_rdm(features):
    n_samples = features.shape[0]
    features_centered = features - features.mean(axis=0, keepdims=True)
    features_normalized = features_centered / (features_centered.std(axis=0, keepdims=True) + 1e-8)
    corr_matrix = np.dot(features_normalized, features_normalized.T) / features.shape[1]
    rdm = 1 - corr_matrix
    np.fill_diagonal(rdm, 0)
    return rdm

model_rdm = compute_model_rdm(fc6_features)
print(f"  模型RDM形状: {model_rdm.shape}")

print("\n[5/5] 计算RSA相关性...")
def compute_rsa_correlation(model_rdm, brain_rdm):
    upper_tri_model = model_rdm[np.triu_indices_from(model_rdm, k=1)]
    upper_tri_brain = brain_rdm[np.triu_indices_from(brain_rdm, k=1)]
    corr, p_val = spearmanr(upper_tri_model, upper_tri_brain)
    return corr, p_val

top50_rsa, top50_p = compute_rsa_correlation(model_rdm, top50_rdm)
bottom50_rsa, bottom50_p = compute_rsa_correlation(model_rdm, bottom50_rdm)

print("\n" + "="*70)
print("结果对比")
print("="*70)
print(f"全体units RSA:     {alexnet_all_rsa:.4f}")
print(f"前50% units RSA:   {top50_rsa:.4f} (p = {top50_p:.2e})")
print(f"后50% units RSA:   {bottom50_rsa:.4f} (p = {bottom50_p:.2e})")
print(f"\n差异分析:")
print(f"  前50% vs 全体:   {top50_rsa - alexnet_all_rsa:+.4f} ({(top50_rsa/alexnet_all_rsa - 1)*100:+.1f}%)")
print(f"  后50% vs 全体:   {bottom50_rsa - alexnet_all_rsa:+.4f} ({(bottom50_rsa/alexnet_all_rsa - 1)*100:+.1f}%)")
print(f"  前50% vs 后50%:  {top50_rsa - bottom50_rsa:+.4f} ({(top50_rsa/bottom50_rsa - 1)*100:+.1f}%)")
print("="*70)
