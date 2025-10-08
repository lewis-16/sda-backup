#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定的LSTM路径整合网络训练脚本
使用更稳定的训练策略
"""

import sys
import matplotlib.pyplot as plt
import json
from scipy.io import loadmat
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 加载数据
print("加载数据...")
grid_dir_separated = loadmat("/media/ubuntu/sda/AD_grid/Figure_1/grid_dir_separated_into_hexagonal_and_rectangular.mat")
wtydir1 = pd.DataFrame(grid_dir_separated['wtydir1'])
wtydir1_data = loadmat(f"/media/ubuntu/sda/AD_grid/cleaned_mat/{str(wtydir1[2][0][0])}_cleaned.mat")
wtydir1_data = wtydir1_data['cleaned_data']
wtydir1_data = pd.DataFrame(wtydir1_data, columns=['x', 'y', 'vel', 'sx', 'sy', 'svel', 'headdir', 'sheaddir', 'ind', 'ts'])

print(f"数据形状: {wtydir1_data.shape}")

# 数据预处理：提取运动轨迹特征
def preprocess_trajectory_data(data):
    """预处理轨迹数据，提取速度、角速度等特征"""
    # 计算线性速度（使用已有的vel列）
    linear_velocity = data['vel'].values
    
    # 计算角速度（头朝向的变化率）
    headdir_rad = np.deg2rad(data['headdir'].values)
    angular_velocity = np.diff(headdir_rad, prepend=headdir_rad[0])
    
    # 处理角速度的跳跃（从-π到π或反之）
    angular_velocity = np.arctan2(np.sin(angular_velocity), np.cos(angular_velocity))
    
    # 构建输入特征：[linear_velocity, sin(angular_velocity), cos(angular_velocity)]
    features = np.column_stack([
        linear_velocity,
        np.sin(angular_velocity),
        np.cos(angular_velocity)
    ])
    
    # 位置信息
    positions = np.column_stack([data['x'].values, data['y'].values])
    
    return features, positions

# 简化的细胞活动生成器（不使用softmax）
class SimpleCellActivityGenerator:
    """简化的细胞活动生成器，不使用softmax归一化"""
    def __init__(self, n_position_cells=100, n_head_direction_cells=36, 
                 position_std=0.1, head_direction_kappa=2.0):
        self.n_position_cells = n_position_cells
        self.n_head_direction_cells = n_head_direction_cells
        self.position_std = position_std
        self.head_direction_kappa = head_direction_kappa
        
        # 生成位置细胞的感受野中心
        self.position_centers = self._generate_position_centers()
        
        # 生成头方向细胞的偏好方向
        self.head_direction_centers = np.linspace(0, 2*np.pi, n_head_direction_cells, endpoint=False)
    
    def _generate_position_centers(self):
        """生成位置细胞的感受野中心"""
        x_min, x_max = -50, 50
        y_min, y_max = -50, 50
        
        x_centers = np.random.uniform(x_min, x_max, self.n_position_cells)
        y_centers = np.random.uniform(y_min, y_max, self.n_position_cells)
        
        return np.column_stack([x_centers, y_centers])
    
    def generate_position_activity(self, positions):
        """根据位置生成位置细胞活动（不使用softmax）"""
        activity = np.zeros((len(positions), self.n_position_cells))
        
        for i, pos in enumerate(positions):
            # 计算到所有感受野中心的距离
            distances_squared = np.sum((self.position_centers - pos)**2, axis=1)
            
            # 计算高斯函数值：exp(-||x - μ_i||² / (2σ²))
            gaussian_values = np.exp(-distances_squared / (2 * self.position_std**2))
            
            # 直接使用高斯值，不进行softmax归一化
            activity[i] = gaussian_values
        
        return activity
    
    def generate_head_direction_activity(self, head_directions):
        """根据头方向生成头方向细胞活动（不使用softmax）"""
        activity = np.zeros((len(head_directions), self.n_head_direction_cells))
        
        for i, hd in enumerate(head_directions):
            # 计算与所有偏好方向的角度差
            angle_diffs = hd - self.head_direction_centers
            
            # 计算冯·米塞斯分布值：exp(κ * cos(φ - μ_j))
            von_mises_values = np.exp(self.head_direction_kappa * np.cos(angle_diffs))
            
            # 直接使用冯·米塞斯值，不进行softmax归一化
            activity[i] = von_mises_values
        
        return activity

# 简化的LSTM网络
class SimplePathIntegrationLSTM(nn.Module):
    """简化的LSTM路径整合网络"""
    def __init__(self, input_dim=3, hidden_dim=64, linear_dim=256, 
                 n_position_cells=100, n_head_direction_cells=36, dropout=0.3):
        super(SimplePathIntegrationLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_position_cells = n_position_cells
        self.n_head_direction_cells = n_head_direction_cells
        
        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # 线性表征层
        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_dim, linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.position_output = nn.Linear(linear_dim, n_position_cells)
        self.head_direction_output = nn.Linear(linear_dim, n_head_direction_cells)
        
    def forward(self, x):
        """前向传播"""
        # LSTM前向传播
        lstm_output, _ = self.lstm(x)
        
        # 线性表征层
        linear_output = self.linear_layer(lstm_output)
        
        # 输出层
        position_outputs = self.position_output(linear_output)
        head_direction_outputs = self.head_direction_output(linear_output)
        
        return position_outputs, head_direction_outputs

# 简化的损失函数
class SimplePathIntegrationLoss(nn.Module):
    """简化的路径整合损失函数：MSE损失"""
    def __init__(self, position_weight=1.0, head_direction_weight=1.0):
        super(SimplePathIntegrationLoss, self).__init__()
        self.position_weight = position_weight
        self.head_direction_weight = head_direction_weight
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, position_pred, head_direction_pred, 
                position_target, head_direction_target):
        """计算损失"""
        position_loss = self.mse_loss(position_pred, position_target)
        head_direction_loss = self.mse_loss(head_direction_pred, head_direction_target)
        
        total_loss = (self.position_weight * position_loss + 
                     self.head_direction_weight * head_direction_loss)
        
        return total_loss, position_loss, head_direction_loss

# 数据预处理和序列生成
def create_sequences(features, position_activity, head_direction_activity, 
                    seq_length=50, batch_size=32):
    """创建训练序列"""
    # 确保数据长度一致
    min_length = min(len(features), len(position_activity), len(head_direction_activity))
    features = features[:min_length]
    position_activity = position_activity[:min_length]
    head_direction_activity = head_direction_activity[:min_length]
    
    # 创建序列
    sequences = []
    for i in range(0, len(features) - seq_length, seq_length // 2):
        seq_features = features[i:i+seq_length]
        seq_position = position_activity[i:i+seq_length]
        seq_head_direction = head_direction_activity[i:i+seq_length]
        
        sequences.append({
            'features': torch.FloatTensor(seq_features),
            'position_target': torch.FloatTensor(seq_position),
            'head_direction_target': torch.FloatTensor(seq_head_direction)
        })
    
    # 创建数据加载器
    dataset = TensorDataset(
        torch.stack([s['features'] for s in sequences]),
        torch.stack([s['position_target'] for s in sequences]),
        torch.stack([s['head_direction_target'] for s in sequences])
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练循环
def train_model(model, train_loader, criterion, optimizer, num_epochs=20, 
                clip_grad_norm=0.5, device=device):
    """训练LSTM路径整合模型"""
    model.train()
    training_history = {
        'epoch': [],
        'total_loss': [],
        'position_loss': [],
        'head_direction_loss': []
    }
    
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_position_loss = 0.0
        epoch_head_direction_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, position_target, head_direction_target) in enumerate(train_loader):
            # 移动到设备
            features = features.to(device)
            position_target = position_target.to(device)
            head_direction_target = head_direction_target.to(device)
            
            # 前向传播
            position_pred, head_direction_pred = model(features)
            
            # 计算损失
            total_loss, position_loss, head_direction_loss = criterion(
                position_pred, head_direction_pred, 
                position_target, head_direction_target
            )
            
            # 检查损失是否为NaN
            if torch.isnan(total_loss):
                print(f"检测到NaN损失，跳过此批次")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # 更新参数
            optimizer.step()
            
            # 记录损失
            epoch_total_loss += total_loss.item()
            epoch_position_loss += position_loss.item()
            epoch_head_direction_loss += head_direction_loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {total_loss.item():.4f}')
        
        if num_batches > 0:
            # 计算平均损失
            avg_total_loss = epoch_total_loss / num_batches
            avg_position_loss = epoch_position_loss / num_batches
            avg_head_direction_loss = epoch_head_direction_loss / num_batches
            
            # 记录历史
            training_history['epoch'].append(epoch + 1)
            training_history['total_loss'].append(avg_total_loss)
            training_history['position_loss'].append(avg_position_loss)
            training_history['head_direction_loss'].append(avg_head_direction_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs} 完成 - '
                  f'总损失: {avg_total_loss:.4f}, '
                  f'位置损失: {avg_position_loss:.4f}, '
                  f'头方向损失: {avg_head_direction_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs} 完成 - 无有效批次')
        
        print('-' * 60)
    
    return training_history

# 分析LSTM单元的空间活动
def analyze_lstm_spatial_activity(model, features, positions, seq_length=100, device=device):
    """分析LSTM单元在空间中的活动模式"""
    model.eval()
    
    # 选择测试序列
    test_seq_features = features[:seq_length]
    test_seq_positions = positions[:seq_length]
    
    # 转换为张量
    test_input = torch.FloatTensor(test_seq_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 前向传播到LSTM层
        lstm_output, _ = model.lstm(test_input)
        
        # 获取LSTM输出 (1, seq_len, hidden_dim)
        lstm_activations = lstm_output.squeeze(0).cpu().numpy()  # (seq_len, hidden_dim)
        
        return lstm_activations, test_seq_positions

# 生成空间活动PDF
def generate_spatial_activity_pdf(lstm_activations, positions, save_path="/media/ubuntu/sda/AD_grid/lstm_spatial_activity_stable.pdf"):
    """生成LSTM单元空间活动的PDF"""
    
    with PdfPages(save_path) as pdf:
        # 总览页面
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 真实轨迹
        axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1, alpha=0.7)
        axes[0, 0].set_title('小鼠运动轨迹', fontsize=14)
        axes[0, 0].set_xlabel('X 位置')
        axes[0, 0].set_ylabel('Y 位置')
        axes[0, 0].grid(True, alpha=0.3)
        
        # LSTM激活热图
        im = axes[0, 1].imshow(lstm_activations.T, aspect='auto', cmap='viridis')
        axes[0, 1].set_title('LSTM单元激活热图', fontsize=14)
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('LSTM单元索引')
        plt.colorbar(im, ax=axes[0, 1])
        
        # 激活强度分布
        valid_activations = lstm_activations[~np.isnan(lstm_activations)]
        if len(valid_activations) > 0:
            axes[1, 0].hist(valid_activations, bins=50, alpha=0.7, color='blue')
            axes[1, 0].set_title('LSTM激活强度分布', fontsize=14)
            axes[1, 0].set_xlabel('激活强度')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '无有效激活数据', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('LSTM激活强度分布', fontsize=14)
        
        # 单元激活随时间变化
        if not np.all(np.isnan(lstm_activations)):
            axes[1, 1].plot(lstm_activations[:, 0], label='单元 0', alpha=0.7)
            axes[1, 1].plot(lstm_activations[:, 1], label='单元 1', alpha=0.7)
            axes[1, 1].plot(lstm_activations[:, 2], label='单元 2', alpha=0.7)
            axes[1, 1].set_title('LSTM单元激活随时间变化', fontsize=14)
            axes[1, 1].set_xlabel('时间步')
            axes[1, 1].set_ylabel('激活强度')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '无有效激活数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('LSTM单元激活随时间变化', fontsize=14)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 每个LSTM单元的空间活动模式
        n_units = lstm_activations.shape[1]
        units_per_page = 16
        
        for page_start in range(0, n_units, units_per_page):
            page_end = min(page_start + units_per_page, n_units)
            n_units_this_page = page_end - page_start
            
            # 计算子图布局
            n_cols = 4
            n_rows = (n_units_this_page + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, unit_idx in enumerate(range(page_start, page_end)):
                row = i // n_cols
                col = i % n_cols
                
                if n_rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]
                
                # 创建散点图，颜色表示单元活动
                unit_activations = lstm_activations[:, unit_idx]
                if not np.all(np.isnan(unit_activations)):
                    scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                                       c=unit_activations, 
                                       cmap='viridis', s=2, alpha=0.7)
                    ax.set_title(f'LSTM单元 {unit_idx}', fontsize=10)
                    ax.set_xlabel('X 位置')
                    ax.set_ylabel('Y 位置')
                    ax.grid(True, alpha=0.3)
                    
                    # 添加颜色条
                    plt.colorbar(scatter, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'LSTM单元 {unit_idx}', fontsize=10)
                    ax.set_xlabel('X 位置')
                    ax.set_ylabel('Y 位置')
            
            # 隐藏多余的子图
            for i in range(n_units_this_page, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                if n_rows == 1:
                    axes[col].set_visible(False)
                else:
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"LSTM空间活动PDF已保存到: {save_path}")

# 主训练流程
def main():
    print("开始稳定的LSTM路径整合网络训练...")
    
    # 处理数据
    features, positions = preprocess_trajectory_data(wtydir1_data)
    print(f"特征形状: {features.shape}")
    print(f"位置形状: {positions.shape}")
    
    # 创建简化的细胞活动生成器
    cell_generator = SimpleCellActivityGenerator(
        n_position_cells=100, 
        n_head_direction_cells=36,
        position_std=0.1,
        head_direction_kappa=2.0
    )
    
    # 生成位置细胞活动
    position_activity = cell_generator.generate_position_activity(positions)
    
    # 生成头方向细胞活动
    headdir_rad = np.deg2rad(wtydir1_data['headdir'].values)
    head_direction_activity = cell_generator.generate_head_direction_activity(headdir_rad)
    
    print(f"位置细胞活动形状: {position_activity.shape}")
    print(f"头方向细胞活动形状: {head_direction_activity.shape}")
    
    # 创建简化的网络实例
    model = SimplePathIntegrationLSTM(
        input_dim=3,
        hidden_dim=64,
        linear_dim=256,
        n_position_cells=100,
        n_head_direction_cells=36,
        dropout=0.3
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    criterion = SimplePathIntegrationLoss(position_weight=1.0, head_direction_weight=1.0)
    
    # 创建训练数据
    train_loader = create_sequences(features, position_activity, head_direction_activity, 
                                   seq_length=50, batch_size=32)
    
    print(f"训练批次数量: {len(train_loader)}")
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练模型
    training_history = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=15,
        clip_grad_norm=0.5,
        device=device
    )
    
    # 分析LSTM单元的空间活动
    print("分析LSTM单元的空间活动...")
    lstm_activations, test_positions = analyze_lstm_spatial_activity(
        model, features, positions, seq_length=100
    )
    
    # 生成空间活动PDF
    print("生成LSTM空间活动PDF...")
    generate_spatial_activity_pdf(lstm_activations, test_positions)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 3,
            'hidden_dim': 64,
            'linear_dim': 256,
            'n_position_cells': 100,
            'n_head_direction_cells': 36,
            'dropout': 0.3
        }
    }, "/media/ubuntu/sda/AD_grid/simple_path_integration_lstm_model.pth")
    
    print("训练完成！")
    if training_history['total_loss']:
        print(f"最终损失: {training_history['total_loss'][-1]:.4f}")
    else:
        print("训练过程中出现NaN，但模型已保存")
    print("模型和PDF已保存")

if __name__ == "__main__":
    main()
