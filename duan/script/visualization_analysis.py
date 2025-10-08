import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class SimpleClassificationDataset(Dataset):
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

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data_array = np.array(self.data[idx], dtype=np.float32).flatten()
        data_tensor = torch.from_numpy(data_array)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        
        return data_tensor, label

class OptimizedNeuralClassifier(nn.Module):
    """优化的神经网络分类器"""
    def __init__(self, input_dim, num_classes, hidden_dims, dropout_rate, activation='relu'):
        super(OptimizedNeuralClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def extract_features(self, x, layer_idx=-2):
        """提取指定层的特征"""
        features = x
        for i, layer in enumerate(self.network):
            features = layer(features)
            if i == layer_idx:
                break
        return features

def plot_accuracy_comparison():
    """绘制三个尺度的accuracy柱状图"""
    # 训练结果数据
    tasks = ['Category', 'Image', 'Angle']
    accuracies = [92.46, 80.16, 49.60]
    num_classes = [6, 24, 72]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    bars = ax1.bar(tasks, accuracies, color=['#2E8B57', '#4169E1', '#DC143C'], alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    bars2 = ax2.bar(tasks, num_classes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax2.set_ylabel('Number of Classes', fontsize=12)
    
    # 在柱子上添加数值标签
    for bar, num in zip(bars2, num_classes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{num}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/media/ubuntu/sda/duan/figure/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Accuracy comparison chart saved!")


def main():
    """主函数"""
    print("开始生成accuracy对比图和特征可视化...")
    
    print("1. 绘制accuracy柱状图...")
    plot_accuracy_comparison()

    print("\n所有任务完成！")
    print("生成的文件:")
    print("- /media/ubuntu/sda/duan/figure/accuracy_comparison.png")

if __name__ == "__main__":
    main()
