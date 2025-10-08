import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

def analyze_classification_performance():
    """分析分类性能与可视化分离度的关系"""
    # 加载数据
    with open('/media/ubuntu/sda/duan/script/trail_activity_500.pkl', 'rb') as f:
        trail_activity = pickle.load(f)
    
    tasks = ['category', 'image', 'angle']
    task_names = ['Category', 'Image', 'Angle']
    
    results = {}
    
    for task, task_name in zip(tasks, task_names):
        print(f"\n分析 {task_name} 任务...")
        
        # 创建数据集
        dataset = SimpleClassificationDataset(trail_activity, task_type=task)
        data, labels = dataset.get_data_and_labels()
        
        # 1. 使用简单分类器测试原始数据的分类能力
        print("测试原始数据的分类能力...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 测试多个分类器
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        classifier_scores = {}
        for name, clf in classifiers.items():
            try:
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                classifier_scores[name] = score
                print(f"  {name}: {score:.4f}")
            except Exception as e:
                print(f"  {name}: Failed - {e}")
                classifier_scores[name] = 0
        
        # 2. 分析数据的统计特性
        print("分析数据统计特性...")
        
        # 计算类间距离
        class_means = []
        class_covs = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            class_data = data[labels == label]
            class_means.append(np.mean(class_data, axis=0))
            class_covs.append(np.cov(class_data.T))
        
        class_means = np.array(class_means)
        
        # 计算类间距离矩阵
        class_distances = np.zeros((len(unique_labels), len(unique_labels)))
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                if i != j:
                    class_distances[i, j] = np.linalg.norm(class_means[i] - class_means[j])
        
        avg_class_distance = np.mean(class_distances[class_distances > 0])
        
        # 计算类内方差
        class_variances = []
        for label in unique_labels:
            class_data = data[labels == label]
            class_var = np.var(class_data, axis=0)
            class_variances.append(np.mean(class_var))
        
        avg_class_variance = np.mean(class_variances)
        
        # 计算分离度指标
        separation_ratio = avg_class_distance / np.sqrt(avg_class_variance)
        
        print(f"  平均类间距离: {avg_class_distance:.4f}")
        print(f"  平均类内方差: {avg_class_variance:.4f}")
        print(f"  分离度比率: {separation_ratio:.4f}")
        
        # 3. 降维后的分离度分析
        print("分析降维后的分离度...")
        
        # PCA降维
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
        
        # 计算PCA后的类间距离
        pca_class_means = []
        for label in unique_labels:
            class_data = data_pca[labels == label]
            pca_class_means.append(np.mean(class_data, axis=0))
        
        pca_class_means = np.array(pca_class_means)
        pca_class_distances = np.zeros((len(unique_labels), len(unique_labels)))
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                if i != j:
                    pca_class_distances[i, j] = np.linalg.norm(pca_class_means[i] - pca_class_means[j])
        
        avg_pca_class_distance = np.mean(pca_class_distances[pca_class_distances > 0])
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        data_tsne = tsne.fit_transform(data)
        
        # 计算t-SNE后的类间距离
        tsne_class_means = []
        for label in unique_labels:
            class_data = data_tsne[labels == label]
            tsne_class_means.append(np.mean(class_data, axis=0))
        
        tsne_class_means = np.array(tsne_class_means)
        tsne_class_distances = np.zeros((len(unique_labels), len(unique_labels)))
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                if i != j:
                    tsne_class_distances[i, j] = np.linalg.norm(tsne_class_means[i] - tsne_class_means[j])
        
        avg_tsne_class_distance = np.mean(tsne_class_distances[tsne_class_distances > 0])
        
        print(f"  PCA后平均类间距离: {avg_pca_class_distance:.4f}")
        print(f"  t-SNE后平均类间距离: {avg_tsne_class_distance:.4f}")
        
        # 4. 保存结果
        results[task_name] = {
            'classifier_scores': classifier_scores,
            'separation_ratio': separation_ratio,
            'avg_class_distance': avg_class_distance,
            'avg_class_variance': avg_class_variance,
            'pca_class_distance': avg_pca_class_distance,
            'tsne_class_distance': avg_tsne_class_distance,
            'data': data,
            'labels': labels,
            'data_pca': data_pca,
            'data_tsne': data_tsne
        }
    
    return results

def create_detailed_analysis_plots(results):
    """创建详细的分析图表"""
    
    # 1. 分类器性能对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    tasks = ['Category', 'Image', 'Angle']
    classifiers = ['Logistic Regression', 'SVM', 'Random Forest']
    
    for i, task in enumerate(tasks):
        scores = results[task]['classifier_scores']
        bars = axes[i].bar(classifiers, [scores[clf] for clf in classifiers], 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        axes[i].set_title(f'{task} Classification Performance', fontweight='bold')
        axes[i].set_ylabel('Accuracy')
        axes[i].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, score in zip(bars, [scores[clf] for clf in classifiers]):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/media/ubuntu/sda/duan/figure/classifier_performance_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 分离度分析
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 分离度比率
    separation_ratios = [results[task]['separation_ratio'] for task in tasks]
    bars = axes[0, 0].bar(tasks, separation_ratios, color=['#2E8B57', '#4169E1', '#DC143C'], alpha=0.8)
    axes[0, 0].set_title('Separation Ratio (Class Distance / sqrt(Class Variance))', fontweight='bold')
    axes[0, 0].set_ylabel('Separation Ratio')
    
    for bar, ratio in zip(bars, separation_ratios):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 类间距离对比
    distances = [results[task]['avg_class_distance'] for task in tasks]
    pca_distances = [results[task]['pca_class_distance'] for task in tasks]
    tsne_distances = [results[task]['tsne_class_distance'] for task in tasks]
    
    x = np.arange(len(tasks))
    width = 0.25
    
    axes[0, 1].bar(x - width, distances, width, label='Original', alpha=0.8)
    axes[0, 1].bar(x, pca_distances, width, label='PCA', alpha=0.8)
    axes[0, 1].bar(x + width, tsne_distances, width, label='t-SNE', alpha=0.8)
    
    axes[0, 1].set_title('Average Inter-class Distance', fontweight='bold')
    axes[0, 1].set_ylabel('Distance')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(tasks)
    axes[0, 1].legend()
    
    # 3. 可视化分离度对比
    for i, task in enumerate(tasks):
        data_pca = results[task]['data_pca']
        data_tsne = results[task]['data_tsne']
        labels = results[task]['labels']
        
        # PCA可视化
        scatter = axes[1, 0].scatter(data_pca[:, 0], data_pca[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.6, s=10)
        axes[1, 0].set_title(f'{task} - PCA Visualization', fontweight='bold')
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        
        # t-SNE可视化
        scatter = axes[1, 1].scatter(data_tsne[:, 0], data_tsne[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.6, s=10)
        axes[1, 1].set_title(f'{task} - t-SNE Visualization', fontweight='bold')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig('/media/ubuntu/sda/duan/figure/separation_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Detailed analysis plots saved!")

def main():
    """主函数"""
    print("开始分析分类性能与可视化分离度的关系...")
    
    # 分析分类性能
    results = analyze_classification_performance()
    
    # 创建详细分析图表
    create_detailed_analysis_plots(results)
    
    # 打印总结
    print("\n" + "="*60)
    print("分析总结")
    print("="*60)
    
    for task in ['Category', 'Image', 'Angle']:
        print(f"\n{task} 任务:")
        print(f"  分离度比率: {results[task]['separation_ratio']:.4f}")
        print(f"  原始数据类间距离: {results[task]['avg_class_distance']:.4f}")
        print(f"  PCA后类间距离: {results[task]['pca_class_distance']:.4f}")
        print(f"  t-SNE后类间距离: {results[task]['tsne_class_distance']:.4f}")
        
        print("  分类器性能:")
        for clf_name, score in results[task]['classifier_scores'].items():
            print(f"    {clf_name}: {score:.4f}")
    
    print(f"\n生成的文件:")
    print(f"- /media/ubuntu/sda/duan/figure/classifier_performance_analysis.png")
    print(f"- /media/ubuntu/sda/duan/figure/separation_analysis.png")

if __name__ == "__main__":
    main()

