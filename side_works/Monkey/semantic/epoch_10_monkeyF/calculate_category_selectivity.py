import numpy as np
import matplotlib.pyplot as plt

# 加载数据
mua = np.load("/media/ubuntu/sda/Monkey/data/train_MUA_MonkeyF.npy")
pc_features = np.load('/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/pc_results/pca_results.npz', allow_pickle=True)

print(f"MUA数据形状: {mua.shape}")

# 提取特征
features_40d = pc_features['features_40d']
features_10d = features_40d[:, :10]

# 手动实现K-means聚类
def kmeans_manual(X, n_clusters, random_state=42, n_init=10):
    np.random.seed(random_state)
    best_inertia = None
    best_labels = None
    best_centers = None
    
    for init in range(n_init):
        indices = np.random.choice(len(X), n_clusters, replace=False)
        centers = X[indices].copy()
        labels = np.zeros(len(X), dtype=np.int32)
        inertia = 0
        
        for _ in range(100):
            distances = np.zeros((len(X), n_clusters))
            for k in range(n_clusters):
                distances[:, k] = np.sqrt(np.sum((X - centers[k]) ** 2, axis=1))
            
            new_labels = np.argmin(distances, axis=1)
            
            inertia = np.sum((X - centers[new_labels]) ** 2)
            
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_labels = new_labels.copy()
                best_centers = centers.copy()
            
            for k in range(n_clusters):
                mask = new_labels == k
                if np.sum(mask) > 0:
                    centers[k] = np.mean(X[mask], axis=0)
            
            if np.all(new_labels == labels):
                break
            
            labels = new_labels
    
    return best_labels, best_centers

# K-means聚类分成10个cluster
n_clusters = 10
cluster_labels, cluster_centers = kmeans_manual(features_10d, n_clusters)
print(f"聚类完成，原始cluster数量: {len(np.unique(cluster_labels))}")

# 合并cluster：cluster 1和6合并，cluster 7和4合并
merged_cluster_labels = cluster_labels.copy()
merged_cluster_labels[merged_cluster_labels == 7] = 1
merged_cluster_labels[merged_cluster_labels == 9] = 6
merged_cluster_labels[merged_cluster_labels == 5] = 4

# 重新映射cluster编号（0-6，共7个类别）
cluster_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    6: 5,
    8: 6
}

category_labels = np.array([cluster_mapping[label] for label in merged_cluster_labels])

n_neurons = mua.shape[1]
n_categories = 7
n_iterations = 100

# 计算每个类别的图像数量，并选择最少数量的类别
category_counts = []
for i in range(n_categories):
    count = np.sum(category_labels == i)
    category_counts.append(count)
    print(f"  类别 {i}: {count} 张图像")

category_counts = np.array(category_counts)
n_samples_per_class = int(np.min(category_counts))
print(f"\n每个类别抽样数量（最少类别数量）: {n_samples_per_class}")

print(f"神经元数量: {n_neurons}")
print(f"类别数量: {n_categories}")
print(f"图像数量: {len(category_labels)}")

np.random.seed(42)

# 存储每次迭代的SI
all_si = np.zeros((n_iterations, n_neurons, n_categories), dtype=np.float32)

for iteration in range(n_iterations):
    si_iter = np.zeros((n_neurons, n_categories), dtype=np.float32)
    
    for neuron_idx in range(n_neurons):
        neuron_responses = mua[:, neuron_idx]
        
        for category_id in range(n_categories):
            category_mask = category_labels == category_id
            category_indices = np.where(category_mask)[0]
            
            if len(category_indices) >= n_samples_per_class:
                sampled_indices = np.random.choice(category_indices, size=n_samples_per_class, replace=False)
                r_category = neuron_responses[sampled_indices]
            else:
                r_category = neuron_responses[category_mask]
            
            noncategory_mask = category_labels != category_id
            r_noncategory = neuron_responses[noncategory_mask]
            
            if len(r_category) > 0 and len(r_noncategory) > 0:
                mean_category = np.mean(r_category)
                mean_noncategory = np.mean(r_noncategory)
                var_category = np.var(r_category)
                var_noncategory = np.var(r_noncategory)
                
                denominator = np.sqrt(0.5 * (var_category + var_noncategory))
                if denominator > 0:
                    si = (mean_category - mean_noncategory) / denominator
                else:
                    si = 0.0
            else:
                si = 0.0
            
            si_iter[neuron_idx, category_id] = si
    
    all_si[iteration] = si_iter
    if (iteration + 1) % 20 == 0:
        print(f"已完成 {iteration + 1}/{n_iterations} 次迭代")

# 计算100次的平均值
selectivity = np.mean(all_si, axis=0)

print(f"\n选择性指数形状: {selectivity.shape}")
print(f"选择性指数范围: [{selectivity.min():.4f}, {selectivity.max():.4f}]")
print(f"选择性指数均值: {selectivity.mean():.4f}")
print(f"选择性指数标准差: {selectivity.std():.4f}")

# 保存结果
np.save('/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity.npy', selectivity)
print(f"\n结果已保存到: /media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity.npy")

# 同时保存类别标签便于后续使用
np.save('/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/image_category_labels.npy', category_labels)
print(f"类别标签已保存到: /media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/image_category_labels.npy")

# 定义脑区信息
brain_regions = {
    'V1': {'range': [0, 270], 'color': '#E74C3C', 'n_neurons': 271},
    'IT': {'range': [271, 395], 'color': '#3498DB', 'n_neurons': 125},
    'V4': {'range': [396, 502], 'color': '#2ECC71', 'n_neurons': 107}
}

# 为每个脑区创建颜色映射
region_colors = {'V1': '#E74C3C', 'IT': '#3498DB', 'V4': '#2ECC71'}

# 绘制每个类别的直方图（按脑区分别绘制）
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for category_id in range(n_categories):
    ax = axes[category_id]
    
    # 分别绘制每个脑区的直方图
    for region_name, region_info in brain_regions.items():
        start, end = region_info['range']
        si_values_region = selectivity[start:end+1, category_id]
        ax.hist(si_values_region, bins=30, color=region_info['color'], 
                edgecolor='black', alpha=0.6, label=f"{region_name} (n={region_info['n_neurons']})")
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    
    # 计算并显示总体均值
    overall_mean = np.mean(selectivity[:, category_id])
    ax.axvline(x=overall_mean, color='purple', linestyle='-', linewidth=2, label=f'Overall Mean: {overall_mean:.2f}')
    
    ax.set_xlabel('Selectivity Index', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Category {category_id}\n(n={category_counts[category_id]})', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

# 隐藏多余的子图
for i in range(n_categories, len(axes)):
    axes[i].axis('off')

plt.suptitle('Category Selectivity Index Distribution for Each Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity_histogram.png', dpi=150, bbox_inches='tight')
plt.savefig('/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity_histogram.pdf', bbox_inches='tight')
print(f"\n直方图已保存到:")
print(f"  - /media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity_histogram.png")
print(f"  - /media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity_histogram.pdf")

plt.close()

# 绘制总体统计图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：所有类别的箱线图（按脑区着色）
ax1 = axes[0]

# 准备按脑区分组的数据
box_data = []
box_colors = []
box_labels = []

for cat_id in range(n_categories):
    for region_name, region_info in brain_regions.items():
        start, end = region_info['range']
        box_data.append(selectivity[start:end+1, cat_id])
        box_colors.append(region_info['color'])
        box_labels.append(f'{region_name}\nCat{cat_id}')

bp = ax1.boxplot(box_data, patch_artist=True, showfliers=False)

for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Brain Region - Category', fontsize=11)
ax1.set_ylabel('Selectivity Index', fontsize=11)
ax1.set_title('Category Selectivity Index by Brain Region', fontsize=12, fontweight='bold')
ax1.set_xticklabels(box_labels, fontsize=6, rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=info['color'], label=f"{name} (n={info['n_neurons']})") 
                   for name, info in brain_regions.items()]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

# 右图：热力图显示每个神经元对各类别的SI
ax2 = axes[1]
im = ax2.imshow(selectivity.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)

# 添加脑区分隔线和标注
v1_end = brain_regions['V1']['range'][1]
it_end = brain_regions['IT']['range'][1]

# 绘制分隔线
ax2.axhline(y=-0.5, color='black', linewidth=2)
ax2.axhline(y=6.5, color='black', linewidth=2)

# 在顶部添加脑区标注
ax2_twin = ax2.twiny()
ax2_twin.set_xlim(ax2.get_xlim())
ax2_twin.set_xticks([v1_end//2, (v1_end + it_end)//2 + 1, (it_end + 502)//2])
ax2_twin.set_xticklabels(['V1', 'IT', 'V4'], fontsize=10, fontweight='bold')

ax2.set_xlabel('Neuron Index', fontsize=11)
ax2.set_ylabel('Category', fontsize=11)
ax2.set_title('Selectivity Index Heatmap (Neurons x Categories)', fontsize=12, fontweight='bold')
ax2.set_yticks(range(n_categories))
ax2.set_yticklabels([f'Cat {i}' for i in range(n_categories)])
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Selectivity Index', fontsize=10)

plt.tight_layout()
plt.savefig('/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity_summary.png', dpi=150, bbox_inches='tight')
plt.savefig('/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity_summary.pdf', bbox_inches='tight')
print(f"\n汇总图已保存到:")
print(f"  - /media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity_summary.png")
print(f"  - /media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyF/category_selectivity_summary.pdf")

plt.close()

print("\n计算完成！")
