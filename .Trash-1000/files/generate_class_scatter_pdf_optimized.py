#!/usr/bin/env python3
"""
优化版本：为每个class_name生成一页散点图的PDF文件
包含内存优化和进度显示
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tqdm import tqdm
import os
import gc

def generate_class_scatter_pdf_optimized():
    """
    优化版本：生成包含每个class_name散点图的PDF文件
    """
    # 读取数据
    print("正在读取数据...")
    clustering_results = pd.read_csv("/media/ubuntu/sda/visual_stimuli_pattern/dynamic/clustering_results.csv")
    tsne_features = pd.read_csv("/media/ubuntu/sda/visual_stimuli_pattern/dynamic/tsne_features.csv")
    
    # 确保数据对齐
    assert len(clustering_results) == len(tsne_features), "数据长度不匹配"
    
    # 获取所有唯一的class_name
    unique_classes = sorted(clustering_results['class_name'].unique())
    print(f"找到 {len(unique_classes)} 个不同的类别")
    
    # 设置matplotlib参数
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建PDF文件
    output_path = "/media/ubuntu/sda/visual_stimuli_pattern/dynamic/class_scatter_plots_optimized.pdf"
    
    # 预计算坐标范围
    x_min, x_max = tsne_features['tSNE1'].min() - 5, tsne_features['tSNE1'].max() + 5
    y_min, y_max = tsne_features['tSNE2'].min() - 5, tsne_features['tSNE2'].max() + 5
    total_points = len(clustering_results)
    
    with PdfPages(output_path) as pdf:
        for i, class_name in enumerate(tqdm(unique_classes, desc="生成散点图")):
            # 创建新图形
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 获取当前类别的数据
            current_class_mask = clustering_results['class_name'] == class_name
            other_class_mask = ~current_class_mask
            
            # 绘制其他类别的点（浅灰色）- 使用更小的点以提高性能
            if other_class_mask.sum() > 0:
                ax.scatter(tsne_features.loc[other_class_mask, 'tSNE1'],
                          tsne_features.loc[other_class_mask, 'tSNE2'],
                          c='lightgray', alpha=0.2, s=0.5, label=f'Other classes ({other_class_mask.sum():,} points)')
            
            # 绘制当前类别的点（彩色）
            if current_class_mask.sum() > 0:
                ax.scatter(tsne_features.loc[current_class_mask, 'tSNE1'],
                          tsne_features.loc[current_class_mask, 'tSNE2'],
                          c='red', alpha=0.8, s=2, label=f'{class_name} ({current_class_mask.sum():,} points)')
            
            # 设置图形属性
            ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
            ax.set_title(f'Class: {class_name}', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴范围
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 添加统计信息
            class_points = current_class_mask.sum()
            percentage = (class_points / total_points) * 100
            
            info_text = f'Total points: {total_points:,}\nCurrent class: {class_points:,}\nPercentage: {percentage:.2f}%'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 保存到PDF
            pdf.savefig(fig, bbox_inches='tight', dpi=200)  # 降低DPI以提高性能
            plt.close(fig)
            
            # 定期清理内存
            if (i + 1) % 50 == 0:
                gc.collect()
                print(f"已完成 {i + 1}/{len(unique_classes)} 个类别")
    
    print(f"\nPDF文件已生成: {output_path}")
    print(f"总共包含 {len(unique_classes)} 页散点图")
    
    # 显示文件大小
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")

if __name__ == "__main__":
    generate_class_scatter_pdf_optimized()
