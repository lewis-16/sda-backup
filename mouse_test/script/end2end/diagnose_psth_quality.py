import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUTPUT_DIR = '/media/ubuntu/sda/mouse_test/processed_results/psth_results'

session_names = [
    'mouse6_021322_natural_image_001',
    'mouse6_022522_natural_image_001',
    'mouse6_031722_natural_image_001',
    'mouse6_042422_natural_image_001',
    'mouse6_052422_natural_image_001',
    'mouse6_062422_natural_image_001',
    'mouse6_072322_natural_image_001',
    'mouse6_082322_natural_image_001',
    'mouse6_092422_natural_image_001',
    'mouse6_102122_natural_image_001',
    'mouse6_112022_natural_image_001',
    'mouse6_122022_natural_image_001',
]

EXTEND_TIME = 0.25
STIMULUS_DURATION = 1.0
bin_size_ms = 50
bin_size_s = bin_size_ms / 1000.0
extend_bins = int(EXTEND_TIME / bin_size_s)
stimulus_bins = int(STIMULUS_DURATION / bin_size_s)
stimulus_start_bin = extend_bins
stimulus_end_bin = extend_bins + stimulus_bins

print("="*70)
print("PSTH质量诊断")
print("="*70)

all_psth_list = []
all_img_list = []
all_session_list = []

for sess_name in session_names:
    psth_path = os.path.join(OUTPUT_DIR, sess_name, 'psth_matrix_from_realtime.npy')
    img_path = os.path.join(OUTPUT_DIR, sess_name, 'trial_image_id_from_realtime.pkl')
    
    if not os.path.exists(psth_path) or not os.path.exists(img_path):
        continue
    
    psth = np.load(psth_path)
    with open(img_path, 'rb') as f:
        img_ids = pickle.load(f)
    
    all_psth_list.append(psth)
    all_img_list.extend(img_ids)
    all_session_list.extend([sess_name] * len(img_ids))

if len(all_psth_list) == 0:
    print("未找到PSTH数据！")
    exit()

psth_full = np.concatenate(all_psth_list, axis=0)
psth_stimulus = psth_full[:, stimulus_start_bin:stimulus_end_bin, :]

print(f"\n【1. PSTH矩阵基本信息】")
print(f"  总trials: {psth_full.shape[0]}")
print(f"  总时间bins: {psth_full.shape[1]}")
print(f"  神经元数: {psth_full.shape[2]}")
print(f"  刺激期bins: {psth_stimulus.shape[1]}")

print(f"\n【2. Firing Rate统计】")
print(f"  全矩阵firing rate范围: [{psth_full.min():.2f}, {psth_full.max():.2f}] Hz")
print(f"  全矩阵firing rate均值: {psth_full.mean():.2f} Hz")
print(f"  全矩阵firing rate标准差: {psth_full.std():.2f} Hz")
print(f"  刺激期firing rate范围: [{psth_stimulus.min():.2f}, {psth_stimulus.max():.2f}] Hz")
print(f"  刺激期firing rate均值: {psth_stimulus.mean():.2f} Hz")
print(f"  刺激期firing rate标准差: {psth_stimulus.std():.2f} Hz")

print(f"\n【3. 稀疏度检查】")
non_zero_ratio = np.count_nonzero(psth_stimulus) / psth_stimulus.size
print(f"  非零值比例: {non_zero_ratio*100:.2f}%")
print(f"  零值比例: {(1-non_zero_ratio)*100:.2f}%")

print(f"\n【4. 每个Neuron的Firing Rate统计】")
neuron_mean_fr = psth_stimulus.mean(axis=(0, 1))
neuron_std_fr = psth_stimulus.std(axis=(0, 1))
print(f"  各neuron平均firing rate范围: [{neuron_mean_fr.min():.2f}, {neuron_mean_fr.max():.2f}] Hz")
print(f"  各neuron平均firing rate均值: {neuron_mean_fr.mean():.2f} Hz")
print(f"  各neuron平均firing rate标准差: {neuron_std_fr.mean():.2f} Hz")
print(f"  有spike的neuron数: {np.count_nonzero(neuron_mean_fr > 0.1)}/{len(neuron_mean_fr)}")

print(f"\n【5. 不同Image之间的Firing Rate差异】")
unique_imgs = sorted(list(set(all_img_list)))
img_mean_fr = {}
for img_id in unique_imgs[:10]:
    img_mask = np.array(all_img_list) == img_id
    if img_mask.sum() > 0:
        img_psth = psth_stimulus[img_mask]
        img_mean_fr[img_id] = img_psth.mean()
        print(f"  Image {img_id}: {img_mask.sum()} trials, 平均firing rate: {img_mean_fr[img_id]:.2f} Hz")

img_fr_array = np.array([img_mean_fr[img] for img in unique_imgs[:10] if img in img_mean_fr])
if len(img_fr_array) > 1:
    img_fr_std = img_fr_array.std()
    print(f"  前10个image的平均firing rate标准差: {img_fr_std:.2f} Hz")
    if img_fr_std < 1.0:
        print(f"  ⚠ 警告：不同image之间的firing rate差异很小，可能难以区分")

print(f"\n【6. PSTH特征的可分性（PCA）】")
psth_flat = psth_stimulus.reshape(psth_stimulus.shape[0], -1)
pca = PCA(n_components=10)
pca_result = pca.fit_transform(psth_flat)
print(f"  PCA前10个主成分解释的方差比例:")
for i, var_ratio in enumerate(pca.explained_variance_ratio_[:10]):
    print(f"    PC{i+1}: {var_ratio*100:.2f}%")
print(f"  前10个主成分累计解释方差: {pca.explained_variance_ratio_[:10].sum()*100:.2f}%")

if pca.explained_variance_ratio_[0] > 0.5:
    print(f"  ⚠ 警告：第一个主成分解释方差>50%，可能数据维度较低")

print(f"\n【7. 与Ground Truth对比（如果可用）】")
gt_psth_path = os.path.join(OUTPUT_DIR, session_names[0], 'psth_matrix.npy')
if os.path.exists(gt_psth_path):
    gt_psth = np.load(gt_psth_path)
    gt_psth_stimulus = gt_psth[:, stimulus_start_bin:stimulus_end_bin, :]
    
    print(f"  GT PSTH矩阵形状: {gt_psth.shape}")
    print(f"  GT刺激期firing rate均值: {gt_psth_stimulus.mean():.2f} Hz")
    print(f"  GT刺激期firing rate标准差: {gt_psth_stimulus.std():.2f} Hz")
    
    if gt_psth_stimulus.shape == psth_stimulus[:gt_psth_stimulus.shape[0]].shape:
        correlation = np.corrcoef(
            gt_psth_stimulus.flatten(),
            psth_stimulus[:gt_psth_stimulus.shape[0]].flatten()
        )[0, 1]
        print(f"  GT与real_time_processing的PSTH相关性: {correlation:.4f}")
        if correlation < 0.5:
            print(f"  ⚠ 警告：相关性较低，说明real_time_processing的PSTH与GT差异较大")
else:
    print(f"  未找到GT PSTH数据（psth_matrix.npy）")

print("\n" + "="*70)
print("诊断完成")
print("="*70)








