#!/usr/bin/env python3
"""
创建PyTorch Dataset类用于加载联合VISp数据集
"""

import numpy as np
import pandas as pd
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class JointVISpDataset(Dataset):
    """
    联合VISp数据集类
    每个样本包含所有session的神经元对同一次image刺激实验的联合反应
    """
    def __init__(self, data, labels, transform=None):
        """
        参数:
        - data: (n_samples, n_neurons, n_timesteps) 二值化矩阵
        - labels: (n_samples,) 图像ID标签
        - transform: 可选的变换函数
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
    
    def get_stats(self):
        """获取数据集统计信息"""
        stats = {
            'n_samples': len(self.data),
            'n_neurons': self.data.shape[1],
            'n_timesteps': self.data.shape[2],
            'unique_labels': torch.unique(self.labels).tolist(),
            'n_classes': len(torch.unique(self.labels)),
            'spike_rate_per_sample': torch.sum(self.data, dim=(1,2)).float() / (self.data.shape[1] * self.data.shape[2]),
            'spike_rate_per_neuron': torch.sum(self.data, dim=(0,2)).float() / (self.data.shape[0] * self.data.shape[2]),
            'spike_rate_per_timestep': torch.sum(self.data, dim=(0,1)).float() / (self.data.shape[0] * self.data.shape[1])
        }
        return stats

def load_joint_visp_dataset(data_dir="/media/ubuntu/sda/neuropixels/correct_visp_dataset"):
    """
    加载联合VISp数据集
    """
    print("加载联合VISp数据集...")
    
    # 加载数据
    data = np.load(os.path.join(data_dir, "correct_visp_data.npy"))
    labels = np.load(os.path.join(data_dir, "correct_visp_labels.npy"))
    
    # 加载元数据
    with open(os.path.join(data_dir, "correct_visp_neuron_info.pkl"), 'rb') as f:
        neuron_info = pickle.load(f)
    
    with open(os.path.join(data_dir, "correct_visp_metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)
    
    # 创建数据集
    dataset = JointVISpDataset(data, labels)
    
    print(f"联合VISp数据集加载完成!")
    print(f"数据形状: {data.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"神经元数量: {len(neuron_info)}")
    print(f"样本数量: {len(metadata)}")
    
    return dataset, neuron_info, metadata

def main():
    """主函数"""
    print("=== 联合VISp数据集使用示例 ===")
    
    # 加载数据集
    dataset, neuron_info, metadata = load_joint_visp_dataset()
    
    # 获取统计信息
    stats = dataset.get_stats()
    print(f"\n数据集统计信息:")
    print(f"样本数量: {stats['n_samples']}")
    print(f"神经元数量: {stats['n_neurons']}")
    print(f"时间步数: {stats['n_timesteps']}")
    print(f"类别数量: {stats['n_classes']}")
    print(f"平均每样本spike率: {stats['spike_rate_per_sample'].mean():.4f}")
    print(f"平均每神经元spike率: {stats['spike_rate_per_neuron'].mean():.4f}")
    print(f"平均每时间步spike率: {stats['spike_rate_per_timestep'].mean():.4f}")
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    
    # 查看一个批次的数据
    print("\n查看一个批次的数据:")
    for batch_data, batch_labels in train_loader:
        print(f"批次数据形状: {batch_data.shape}")
        print(f"批次标签形状: {batch_labels.shape}")
        print(f"批次标签范围: {batch_labels.min().item()} - {batch_labels.max().item()}")
        print(f"批次中spike总数: {torch.sum(batch_data).item()}")
        print(f"批次中活跃神经元数: {torch.sum(torch.sum(batch_data, dim=(0,2)) > 0).item()}")
        
        # 检查数据结构
        print(f"\n数据结构验证:")
        print(f"每个样本包含 {batch_data.shape[1]} 个神经元（来自所有session）")
        print(f"每个样本的时间步数: {batch_data.shape[2]}")
        print(f"数据范围: {batch_data.min().item()} - {batch_data.max().item()}")
        break
    
    print(f"\n=== 数据集特点总结 ===")
    print(f"✅ 每个data样本: (n_neurons, n_timesteps) = ({stats['n_neurons']}, {stats['n_timesteps']})")
    print(f"✅ 包含所有session的神经元对同一次image刺激实验的联合反应")
    print(f"✅ 标签: 图像ID (0-117)")
    print(f"✅ 时间分辨率: 1ms bins")
    print(f"✅ 刺激时长: 250ms")
    print(f"✅ 总样本数: {stats['n_samples']}")
    
    return dataset, neuron_info, metadata

if __name__ == "__main__":
    dataset, neuron_info, metadata = main()


