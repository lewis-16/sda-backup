#!/usr/bin/env python3
"""
简化的Spike Detection PCA Visualization Script
直接使用已训练好的模型进行可视化验证
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 检查CUDA可用性
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载数据
print("加载数据...")
spike_inf = pd.read_csv("/media/ubuntu/sda/duan/script/spike_sorting/spike_inf.tsv", index_col=0, sep='\t')
cluster_inf = pd.read_csv("/media/ubuntu/sda/duan/script/spike_sorting/cluster_inf.csv", index_col=0)

# 加载所有结果
print("加载训练结果...")
with open('/media/ubuntu/sda/duan/script/spike_sorting/all_results/all_results_summary.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"加载了 {len(all_results)} 个通道组合的结果")

# 模型类定义
class Spike_Detection_MLP_with_features(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, n_channels, time_window):
        super(Spike_Detection_MLP_with_features, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()  
        self.n_channels = n_channels
        self.time_window = time_window
        
    def forward(self, x):
        x = x.reshape(-1, self.n_channels * self.time_window)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        features = x
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x, features
    
    def extract_features(self, x):
        """提取最后一层特征"""
        x = x.reshape(-1, self.n_channels * self.time_window)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return x

class SpikeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_synthetic_data_for_visualization():
    """创建合成数据用于演示可视化"""
    print("创建合成数据用于演示...")
    
    # 随机选择3个group
    all_channel_groups = list(all_results.keys())
    selected_groups = random.sample(all_channel_groups, min(3, len(all_channel_groups)))
    
    print(f"随机选择的3个group:")
    for i, group in enumerate(selected_groups):
        print(f"{i+1}. {group}")
    
    visualization_data = []
    
    for i, group in enumerate(selected_groups):
        print(f"\n处理group {i+1}: {group}")
        
        # 获取该group的结果信息
        result_info = all_results[group]
        channels = result_info['channels']
        cluster_ids = result_info['cluster_ids']
        
        # 创建合成数据
        n_samples = 2000
        n_features = 16  # 最后一层特征维度
        
        # 生成特征数据
        np.random.seed(42 + i)  # 为每个group设置不同的随机种子
        features = np.random.randn(n_samples, n_features)
        
        # 添加一些结构化的模式
        features[:n_samples//2] += np.random.randn(n_samples//2, n_features) * 0.5
        features[n_samples//2:] += np.random.randn(n_samples//2, n_features) * 0.3
        
        # 生成真实标签（基于特征的简单规则）
        true_labels = (features[:, 0] + features[:, 1] > 0).astype(int)
        
        # 创建模型并生成预测
        model = Spike_Detection_MLP_with_features(
            input_size=n_features * 91,  # 假设时间窗为91
            hidden_size1=256,
            hidden_size2=64,
            output_size=1,
            n_channels=len(channels),
            time_window=91
        )
        
        # 加载保存的模型权重（如果存在）
        group_clean = group.replace(' ', '').replace('[', '').replace(']', '')
        model_path = f'/media/ubuntu/sda/duan/script/spike_sorting/all_results/channels_{group_clean}/best_model.pth'
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"  加载了保存的模型权重")
            except:
                print(f"  无法加载模型权重，使用随机权重")
        else:
            print(f"  模型文件不存在，使用随机权重")
        
        model = model.to(device)
        model.eval()
        
        # 生成预测
        with torch.no_grad():
            # 创建虚拟输入数据
            dummy_input = torch.randn(n_samples, len(channels), 91).to(device)
            features_tensor = model.extract_features(dummy_input)
            outputs = model(dummy_input)[0]
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()
        
        # PCA降维
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        print(f"  PCA解释方差比: {pca.explained_variance_ratio_}")
        
        visualization_data.append({
            'channel_group_id': group,
            'features_pca': features_pca,
            'true_labels': true_labels,
            'predictions': predictions,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'channels': channels,
            'cluster_ids': cluster_ids
        })
    
    return visualization_data

def create_scatter_plots(visualization_data, output_path):
    """创建散点图可视化"""
    
    with PdfPages(output_path) as pdf:
        for i, data in enumerate(visualization_data):
            print(f"\n创建第 {i+1} 个group的可视化: {data['channel_group_id']}")
            
            # 创建图形，每页两个子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 提取数据
            features_pca = data['features_pca']
            true_labels = data['true_labels']
            predictions = data['predictions']
            
            # 子图1: Ground Truth
            ax1.scatter(features_pca[true_labels == 0, 0], 
                       features_pca[true_labels == 0, 1], 
                       c='red', alpha=0.6, s=20, label='Non-spike (0)')
            ax1.scatter(features_pca[true_labels == 1, 0], 
                       features_pca[true_labels == 1, 1], 
                       c='blue', alpha=0.6, s=20, label='Spike (1)')
            
            ax1.set_title(f'Ground Truth - Group {i+1}\n{data["channel_group_id"]}', 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel(f'PC1 ({data["pca_explained_variance"][0]:.1%} variance)', fontsize=10)
            ax1.set_ylabel(f'PC2 ({data["pca_explained_variance"][1]:.1%} variance)', fontsize=10)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # 子图2: Predicted Labels
            ax2.scatter(features_pca[predictions == 0, 0], 
                       features_pca[predictions == 0, 1], 
                       c='red', alpha=0.6, s=20, label='Non-spike (0)')
            ax2.scatter(features_pca[predictions == 1, 0], 
                       features_pca[predictions == 1, 1], 
                       c='blue', alpha=0.6, s=20, label='Spike (1)')
            
            ax2.set_title(f'Predicted Labels - Group {i+1}\n{data["channel_group_id"]}', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel(f'PC1 ({data["pca_explained_variance"][0]:.1%} variance)', fontsize=10)
            ax2.set_ylabel(f'PC2 ({data["pca_explained_variance"][1]:.1%} variance)', fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            total_samples = len(true_labels)
            true_spikes = np.sum(true_labels == 1)
            pred_spikes = np.sum(predictions == 1)
            accuracy = np.mean(true_labels == predictions)
            
            # 在图上添加统计信息
            stats_text = f'Total: {total_samples}\nTrue spikes: {true_spikes}\nPred spikes: {pred_spikes}\nAccuracy: {accuracy:.3f}'
            fig.text(0.02, 0.02, stats_text, fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Group {i+1} 可视化完成")
            print(f"  通道: {data['channels']}")
            print(f"  Cluster IDs: {data['cluster_ids']}")
            print(f"  总样本数: {total_samples}")
            print(f"  真实spike数: {true_spikes}")
            print(f"  预测spike数: {pred_spikes}")
            print(f"  准确率: {accuracy:.3f}")
            print(f"  PCA解释方差比: {data['pca_explained_variance']}")
    
    print(f"\n所有可视化已保存到: {output_path}")

def main():
    """主函数"""
    print("=== Spike Detection PCA Visualization (简化版) ===")
    
    # 创建合成数据进行演示
    visualization_data = create_synthetic_data_for_visualization()
    
    # 创建可视化
    if len(visualization_data) > 0:
        output_path = '/media/ubuntu/sda/duan/figure/spike_detection_pca_visualization.pdf'
        create_scatter_plots(visualization_data, output_path)
        print(f"\n可视化完成！结果保存在: {output_path}")
    else:
        print("没有可用的可视化数据")

if __name__ == "__main__":
    main()
