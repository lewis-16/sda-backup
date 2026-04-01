# MUA分类网络训练和评估
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import time
import os

class MUAClassificationDataset(Dataset):
    def __init__(self, mua_data, labels, transform=None):
        """
        MUA分类数据集
        
        Args:
            mua_data: MUA数据，形状为 (n_samples, n_channels)
            labels: 标签列表
            transform: 数据变换（可选）
        """
        self.mua_data = torch.tensor(mua_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        
    def __len__(self):
        return len(self.mua_data)
    
    def __getitem__(self, idx):
        sample = self.mua_data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

class ResidualBlock(nn.Module):
    """ResNet风格的残差块"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 残差连接
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.linear2(out))
        out = self.dropout2(out)
        
        out += residual
        out = F.relu(out)
        
        return out

class ResNetMLPClassifier(nn.Module):
    """4层ResNet MLP分类器"""
    def __init__(self, input_dim, hidden_dims=[512, 1024, 1024, 1024], num_classes=91, dropout_rate=0.1):
        super(ResNetMLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        
        # 4个ResNet块
        self.resnet_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.resnet_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout_rate)
            )
        
        # 最终特征层（1024维）
        self.feature_layer = nn.Linear(hidden_dims[-1], 1024)
        self.feature_bn = nn.BatchNorm1d(1024)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # 分类器层
        self.classifier = nn.Linear(1024, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入投影
        x = F.relu(self.input_bn(self.input_projection(x)))
        
        # 通过ResNet块
        for block in self.resnet_blocks:
            x = block(x)
        
        # 特征提取（1024维）
        features = F.relu(self.feature_bn(self.feature_layer(x)))
        features = self.feature_dropout(features)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, features
    
    def get_features(self, x):
        """只返回特征，不进行分类"""
        with torch.no_grad():
            x = F.relu(self.input_bn(self.input_projection(x)))
            
            for block in self.resnet_blocks:
                x = block(x)
            
            features = F.relu(self.feature_bn(self.feature_layer(x)))
            return features

def create_label_mapping(original_labels, num_classes=91):
    """
    创建标签重新映射
    
    Args:
        original_labels: 原始标签数组
        num_classes: 目标类别数量
    
    Returns:
        mapped_labels: 重新映射后的标签
        label_mapping: 标签映射字典
    """
    unique_labels = np.unique(original_labels)
    print(f"原始标签数量: {len(unique_labels)}")
    
    # 创建映射字典
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    # 重新映射标签
    mapped_labels = np.array([label_mapping[label] for label in original_labels])
    
    print(f"映射后标签数量: {len(np.unique(mapped_labels))}")
    print(f"标签映射范围: {mapped_labels.min()} - {mapped_labels.max()}")
    
    return mapped_labels, label_mapping

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        logits, features = model(data)
        loss = criterion(logits, target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_features = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        logits, features = model(data)
        loss = criterion(logits, target)
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 收集预测结果和特征
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        all_features.extend(features.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_predictions, all_targets, all_features

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path=None):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def print_classification_report(y_true, y_pred, class_names=None):
    """打印分类报告"""
    print("\n=== Classification Report ===")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

def main():
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据（这里需要根据实际情况调整数据加载）
    print("请确保已经运行了数据筛选代码，获得了 final_test_MUA 和 final_labels")
    print("或者使用 run_training.py 脚本来运行完整的训练流程")
    
    # 示例：如何加载数据
    # final_test_MUA = np.load("filtered_test_MUA_oracle.npy")
    # final_labels = np.load("filtered_labels.npy")
    
    print("请使用 run_training.py 脚本来运行训练")

if __name__ == "__main__":
    main()
