import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class SimpleClassificationDataset:
    """简化的分类数据集，只使用神经元活动数据"""
    def __init__(self, trail_activity, task_type='category'):
        self.trail_activity = trail_activity
        self.task_type = task_type
        
        # 准备数据和标签
        self.data = []
        self.labels = []
        
        for key, trials in trail_activity.items():
            trial_condition = int(key.split('_')[0])
            trial_target = int(key.split('_')[1])
            
            # 根据trial_condition确定图像类别
            if trial_condition <= 4:  # animals
                category = 'animals'
                if trial_condition == 1:
                    image_name = 'Dee1'
                elif trial_condition == 2:
                    image_name = 'Ele'
                elif trial_condition == 3:
                    image_name = 'Pig'
                elif trial_condition == 4:
                    image_name = 'Rhi'
            elif trial_condition <= 8:  # faces
                category = 'faces'
                if trial_condition == 5:
                    image_name = 'MA'
                elif trial_condition == 6:
                    image_name = 'MB'
                elif trial_condition == 7:
                    image_name = 'MC'
                elif trial_condition == 8:
                    image_name = 'WA'
            elif trial_condition <= 12:  # fruits
                category = 'fruits'
                if trial_condition == 9:
                    image_name = 'App1'
                elif trial_condition == 10:
                    image_name = 'Ban1'
                elif trial_condition == 11:
                    image_name = 'Pea1'
                elif trial_condition == 12:
                    image_name = 'Pin1'
            elif trial_condition <= 16:  # manmade
                category = 'manmade'
                if trial_condition == 13:
                    image_name = 'Bed1'
                elif trial_condition == 14:
                    image_name = 'Cha1'
                elif trial_condition == 15:
                    image_name = 'Dis1'
                elif trial_condition == 16:
                    image_name = 'Sof1'
            elif trial_condition <= 20:  # plants
                category = 'plants'
                if trial_condition == 17:
                    image_name = 'A'
                elif trial_condition == 18:
                    image_name = 'B'
                elif trial_condition == 19:
                    image_name = 'C'
                elif trial_condition == 20:
                    image_name = 'D'
            elif trial_condition <= 24:  # shape2d
                category = 'shape2d'
                if trial_condition == 21:
                    image_name = 'Cir'
                elif trial_condition == 22:
                    image_name = 'Oth'
                elif trial_condition == 23:
                    image_name = 'Squ'
                elif trial_condition == 24:
                    image_name = 'Tri'
            
            # 根据trial_target确定角度/颜色
            if category == 'shape2d':
                if trial_target == 1:
                    angle = 'B1'
                elif trial_target == 2:
                    angle = 'G1'
                elif trial_target == 3:
                    angle = 'R1'
            else:
                if trial_target == 1:
                    angle = '0'
                elif trial_target == 2:
                    angle = '315'
                elif trial_target == 3:
                    angle = '45'
            
            # 为每个trial添加数据
            for trial_data in trials:
                self.data.append(trial_data)
                
                # 根据任务类型确定标签
                if task_type == 'category':
                    label = self._get_category_label(category)
                elif task_type == 'image':
                    label = self._get_image_label(image_name)
                elif task_type == 'angle':
                    label = self._get_angle_label(image_name, angle)
                
                self.labels.append(label)
    
    def _get_category_label(self, category):
        category_mapping = {
            'animals': 0, 'faces': 1, 'fruits': 2, 
            'manmade': 3, 'plants': 4, 'shape2d': 5
        }
        return category_mapping[category]
    
    def _get_image_label(self, image_name):
        image_mapping = {
            'Dee1': 0, 'Ele': 1, 'Pig': 2, 'Rhi': 3,
            'MA': 4, 'MB': 5, 'MC': 6, 'WA': 7,
            'App1': 8, 'Ban1': 9, 'Pea1': 10, 'Pin1': 11,
            'Bed1': 12, 'Cha1': 13, 'Dis1': 14, 'Sof1': 15,
            'A': 16, 'B': 17, 'C': 18, 'D': 19,
            'Cir': 20, 'Oth': 21, 'Squ': 22, 'Tri': 23
        }
        return image_mapping[image_name]
    
    def _get_angle_label(self, image_name, angle):
        angle_mapping = {}
        angle_idx = 0
        for category in ['animals', 'faces', 'fruits', 'manmade', 'plants', 'shape2d']:
            if category == 'animals':
                images = ['Dee1', 'Ele', 'Pig', 'Rhi']
            elif category == 'faces':
                images = ['MA', 'MB', 'MC', 'WA']
            elif category == 'fruits':
                images = ['App1', 'Ban1', 'Pea1', 'Pin1']
            elif category == 'manmade':
                images = ['Bed1', 'Cha1', 'Dis1', 'Sof1']
            elif category == 'plants':
                images = ['A', 'B', 'C', 'D']
            elif category == 'shape2d':
                images = ['Cir', 'Oth', 'Squ', 'Tri']
                
            for img in images:
                if category == 'shape2d':
                    angles = ['B1', 'G1', 'R1']
                else:
                    angles = ['0', '315', '45']
                for ang in angles:
                    angle_mapping[f"{img}_{ang}"] = angle_idx
                    angle_idx += 1
        return angle_mapping[f"{image_name}_{angle}"]

    def get_data_and_labels(self):
        """获取所有数据和标签"""
        all_data = []
        all_labels = []
        
        for i in range(len(self.data)):
            data_array = np.array(self.data[i], dtype=np.float32).flatten()
            all_data.append(data_array)
            all_labels.append(self.labels[i])
        
        return np.array(all_data), np.array(all_labels)

def plot_accuracy_comparison():
    """绘制三个尺度的accuracy柱状图"""
    # 训练结果数据
    tasks = ['Category', 'Image', 'Angle']
    accuracies = [92.46, 80.16, 49.60]
    num_classes = [6, 24, 72]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：准确率柱状图
    bars = ax1.bar(tasks, accuracies, color=['#2E8B57', '#4169E1', '#DC143C'], alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Classification Accuracy Across Different Scales', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # 在柱子上添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 右图：类别数量对比
    bars2 = ax2.bar(tasks, num_classes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax2.set_ylabel('Number of Classes', fontsize=12)
    ax2.set_title('Number of Classes Across Different Scales', fontsize=14, fontweight='bold')
    
    # 在柱子上添加数值标签
    for bar, num in zip(bars2, num_classes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{num}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/media/ubuntu/sda/duan/figure/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Accuracy comparison chart saved!")

def create_feature_extraction_models():
    """创建用于特征提取的简单模型"""
    models = {}
    
    # Category模型
    category_model = nn.Sequential(
        nn.Linear(1720, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 6)
    )
    models['category'] = category_model
    
    # Image模型
    image_model = nn.Sequential(
        nn.Linear(1720, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 24)
    )
    models['image'] = image_model
    
    # Angle模型
    angle_model = nn.Sequential(
        nn.Linear(1720, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 72)
    )
    models['angle'] = angle_model
    
    return models

def extract_features_from_model(model, data, layer_indices):
    """从模型中提取指定层的特征"""
    features = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        x = data_tensor
        for i, layer in enumerate(model):
            x = layer(x)
            if i in layer_indices:
                features[f'Layer_{i}'] = x.cpu().numpy()
    
    return features

def visualize_features():
    """进行特征提取和降维可视化"""
    # 加载数据
    with open('/media/ubuntu/sda/duan/script/trail_activity_500.pkl', 'rb') as f:
        trail_activity = pickle.load(f)
    
    # 创建特征提取模型
    models = create_feature_extraction_models()
    
    tasks = ['category', 'image', 'angle']
    task_names = ['Category', 'Image', 'Angle']
    
    for task, task_name in zip(tasks, task_names):
        print(f"\n处理 {task_name} 任务...")
        
        # 创建数据集
        dataset = SimpleClassificationDataset(trail_activity, task_type=task)
        data, labels = dataset.get_data_and_labels()
        
        print(f"数据形状: {data.shape}, 标签形状: {labels.shape}")
        
        # 1. 原始数据降维
        print("对原始数据进行降维...")
        
        # PCA降维
        pca_original = PCA(n_components=2)
        data_pca_original = pca_original.fit_transform(data)
        
        # t-SNE降维
        tsne_original = TSNE(n_components=2, random_state=42, perplexity=30)
        data_tsne_original = tsne_original.fit_transform(data)
        
        # 2. 模型各层特征提取和降维
        model = models[task]
        
        # 定义要提取的层索引（每层的输出）
        layer_indices = [0, 3, 6, 9] if task == 'angle' else [0, 3, 6]  # Linear层的索引
        
        print("提取模型各层特征...")
        layer_features = extract_features_from_model(model, data, layer_indices)
        
        # 对每层特征进行降维
        layer_pca_results = {}
        layer_tsne_results = {}
        
        for layer_name, features in layer_features.items():
            print(f"对 {layer_name} 特征进行降维...")
            
            # PCA降维
            pca = PCA(n_components=2)
            layer_pca_results[layer_name] = pca.fit_transform(features)
            
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            layer_tsne_results[layer_name] = tsne.fit_transform(features)
        
        # 创建可视化图表
        create_visualization_plots(task_name, labels, 
                                 data_pca_original, data_tsne_original,
                                 layer_pca_results, layer_tsne_results)

def create_visualization_plots(task_name, labels, 
                             original_pca, original_tsne,
                             layer_pca_results, layer_tsne_results):
    """创建可视化图表"""
    
    # 创建大图
    n_layers = len(layer_pca_results) + 1  # +1 for original data
    fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 10))
    
    if n_layers == 1:
        axes = axes.reshape(2, 1)
    
    # 原始数据可视化
    # PCA
    scatter = axes[0, 0].scatter(original_pca[:, 0], original_pca[:, 1], 
                               c=labels, cmap='tab10', alpha=0.7, s=20)
    axes[0, 0].set_title(f'{task_name} - Original Data (PCA)', fontweight='bold')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    
    # t-SNE
    scatter = axes[1, 0].scatter(original_tsne[:, 0], original_tsne[:, 1], 
                               c=labels, cmap='tab10', alpha=0.7, s=20)
    axes[1, 0].set_title(f'{task_name} - Original Data (t-SNE)', fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    
    # 各层特征可视化
    for i, (layer_name, pca_data) in enumerate(layer_pca_results.items()):
        col_idx = i + 1
        
        # PCA
        scatter = axes[0, col_idx].scatter(pca_data[:, 0], pca_data[:, 1], 
                                         c=labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, col_idx].set_title(f'{task_name} - {layer_name} (PCA)', fontweight='bold')
        axes[0, col_idx].set_xlabel('PC1')
        axes[0, col_idx].set_ylabel('PC2')
        
        # t-SNE
        tsne_data = layer_tsne_results[layer_name]
        scatter = axes[1, col_idx].scatter(tsne_data[:, 0], tsne_data[:, 1], 
                                         c=labels, cmap='tab10', alpha=0.7, s=20)
        axes[1, col_idx].set_title(f'{task_name} - {layer_name} (t-SNE)', fontweight='bold')
        axes[1, col_idx].set_xlabel('t-SNE 1')
        axes[1, col_idx].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(f'/media/ubuntu/sda/duan/figure/{task_name.lower()}_feature_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{task_name} feature visualization saved!")

def main():
    """主函数"""
    print("开始生成accuracy对比图和特征可视化...")
    
    # 1. 绘制accuracy柱状图
    print("1. 绘制accuracy柱状图...")
    plot_accuracy_comparison()
    
    # 2. 提取特征并进行降维可视化
    print("2. 提取特征并进行降维可视化...")
    visualize_features()
    
    print("\n所有任务完成！")
    print("生成的文件:")
    print("- /media/ubuntu/sda/duan/figure/accuracy_comparison.png")
    print("- /media/ubuntu/sda/duan/figure/category_feature_visualization.png")
    print("- /media/ubuntu/sda/duan/figure/image_feature_visualization.png")
    print("- /media/ubuntu/sda/duan/figure/angle_feature_visualization.png")

if __name__ == "__main__":
    main()

