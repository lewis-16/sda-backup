#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUA分类网络训练脚本
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 导入模型类
from mua_classification import (
    MUAClassificationDataset, 
    ResNetMLPClassifier, 
    create_label_mapping,
    train_epoch, 
    evaluate,
    plot_training_history,
    plot_confusion_matrix,
    print_classification_report
)

def main():
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载筛选后的数据
    print("加载筛选后的数据...")
    try:
        final_test_MUA = np.load("filtered_test_MUA_oracle.npy")
        final_labels = np.load("filtered_labels.npy")
        print(f"数据形状: {final_test_MUA.shape}")
        print(f"标签形状: {final_labels.shape}")
    except FileNotFoundError:
        print("未找到筛选后的数据文件，请先运行数据筛选代码")
        print("需要文件: filtered_test_MUA_oracle.npy, filtered_labels.npy")
        return
    
    # 创建标签映射
    print("\n=== 创建标签映射 ===")
    mapped_labels, label_mapping = create_label_mapping(final_labels, num_classes=91)
    
    # 分割训练和测试集
    train_indices, test_indices = train_test_split(
        range(len(final_test_MUA)), 
        test_size=0.2, 
        random_state=42, 
        stratify=mapped_labels
    )
    
    train_dataset = MUAClassificationDataset(
        final_test_MUA[train_indices], 
        mapped_labels[train_indices]
    )
    
    test_dataset = MUAClassificationDataset(
        final_test_MUA[test_indices], 
        mapped_labels[test_indices]
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"特征维度: {final_test_MUA.shape[1]}")
    print(f"类别数量: {len(np.unique(mapped_labels))}")
    
    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"\n训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    print(f"批次大小: {batch_size}")
    
    # 创建模型
    input_dim = final_test_MUA.shape[1]
    num_classes = len(np.unique(mapped_labels))
    
    model = ResNetMLPClassifier(
        input_dim=input_dim,
        hidden_dims=[512, 1024, 1024, 1024],
        num_classes=num_classes,
        dropout_rate=0.1
    ).to(device)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 设置优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    print("优化器和损失函数设置完成")
    
    # 训练循环
    num_epochs = 50
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\n开始训练，共 {num_epochs} 个epoch...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mua_classifier.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 绘制训练历史
    plot_training_history(train_losses, train_accs, val_losses, val_accs, 'training_history.png')
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('best_mua_classifier.pth'))
    val_loss, val_acc, predictions, targets, features = evaluate(model, test_loader, criterion, device)
    
    print(f"\n=== 最终结果 ===")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最终测试准确率: {val_acc:.2f}%")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(targets, predictions, save_path='confusion_matrix.png')
    
    # 打印分类报告
    print_classification_report(targets, predictions)
    
    # 保存特征和预测结果
    np.save('test_features.npy', np.array(features))
    np.save('test_predictions.npy', np.array(predictions))
    np.save('test_targets.npy', np.array(targets))
    
    print("\n训练完成！模型和结果已保存。")
    print("保存的文件:")
    print("- best_mua_classifier.pth: 最佳模型权重")
    print("- training_history.png: 训练历史图表")
    print("- confusion_matrix.png: 混淆矩阵")
    print("- test_features.npy: 测试集特征")
    print("- test_predictions.npy: 测试集预测结果")
    print("- test_targets.npy: 测试集真实标签")

if __name__ == "__main__":
    main()

