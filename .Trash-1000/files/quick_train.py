import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import optuna
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class SimpleClassificationDataset(Dataset):
    """简化的分类数据集，只使用神经元活动数据"""
    def __init__(self, trail_activity, task_type='category'):
        """
        trail_activity: 从pkl文件加载的数据字典
        task_type: 'category', 'image', 'angle' 三种分类任务
        """
        self.trail_activity = trail_activity
        self.task_type = task_type
        
        # 准备数据和标签
        self.data = []
        self.labels = []
        
        for key, trials in trail_activity.items():
            # key格式: "trial_condition_trial_target" (如 "1_1", "1_2", "1_3")
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
                # trial_data shape: (86, 20, 1) - 86个神经元，20个时间bin
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
        # 数据形状: (86, 20, 1) -> 展平为 (1720,)
        # 预先转换为numpy数组以提高性能
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

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience=10):
    """训练模型"""
    model.to(device)
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return best_val_acc, train_losses, val_losses, val_accuracies

def quick_train(task_type='image', epochs=50):
    """快速训练单个任务"""
    print(f"开始快速训练 {task_type} 分类任务...")
    
    # 加载数据
    with open('/media/ubuntu/sda/duan/script/trail_activity_500.pkl', 'rb') as f:
        trail_activity = pickle.load(f)
    
    # 创建数据集
    dataset = SimpleClassificationDataset(trail_activity, task_type=task_type)
    print(f"数据集大小: {len(dataset)}")
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 根据任务类型确定类别数和网络结构
    if task_type == 'category':
        num_classes = 6
        hidden_dims = [512, 256, 128]
    elif task_type == 'image':
        num_classes = 24
        hidden_dims = [1024, 512, 256]
    elif task_type == 'angle':
        num_classes = 72
        hidden_dims = [1024, 512, 256, 128]
    
    # 创建模型
    model = OptimizedNeuralClassifier(
        input_dim=1720,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout_rate=0.3,
        activation='relu'
    )
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    best_val_acc, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, optimizer, criterion, device, epochs=epochs, patience=15
    )
    
    # 保存模型
    model_path = f'/media/ubuntu/sda/duan/script/best_model_{task_type}_quick.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{task_type.capitalize()} Classification - Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{task_type.capitalize()} Classification - Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'/media/ubuntu/sda/duan/figure/{task_type}_quick_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, best_val_acc

def main():
    """主函数"""
    print("开始快速训练剩余任务...")
    
    # 训练剩余的两个任务
    tasks = ['image', 'angle']
    results = {}
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"快速训练 {task} 分类任务")
        print(f"{'='*60}")
        
        model, best_acc = quick_train(task, epochs=50)
        results[task] = {
            'model': model,
            'best_accuracy': best_acc
        }
        
        print(f"{task} 任务完成! 最佳准确率: {best_acc:.2f}%")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("训练总结")
    print(f"{'='*60}")
    print("Category 分类: 92.46% (已完成)")
    for task, result in results.items():
        print(f"{task.capitalize()} 分类: {result['best_accuracy']:.2f}%")
    
    return results

if __name__ == "__main__":
    results = main()
