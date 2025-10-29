#!/usr/bin/env python3
"""
点过程模型可视化工具
用于可视化优化过程和结果

"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PointProcessVisualizer:
    """点过程模型可视化器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        
    def plot_optimization_convergence(self, optimization_history: List[float]):
        """绘制优化收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(optimization_history, 'b-', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title('参数优化收敛曲线')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        output_path = os.path.join(self.results_dir, 'optimization_convergence.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"收敛曲线已保存到: {output_path}")
    
    def plot_feature_comparison(self, target_features: Dict, model_features: Dict):
        """绘制特征对比图"""
        # 提取共同特征
        common_features = set(target_features.keys()) & set(model_features.keys())
        
        if not common_features:
            print("没有找到共同特征进行对比")
            return
        
        n_features = len(common_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature_name in enumerate(sorted(common_features)):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row, col]
            
            # 目标值（带误差条）
            target_mean = target_features[feature_name][0]
            target_std = target_features[feature_name][1]
            model_value = model_features[feature_name]
            
            # 绘制对比
            x_pos = [0, 1]
            y_values = [target_mean, model_value]
            y_errors = [target_std, 0]
            
            bars = ax.bar(x_pos, y_values, yerr=y_errors, 
                         capsize=5, alpha=0.7, 
                         color=['skyblue', 'lightcoral'])
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['目标值', '模型值'])
            ax.set_title(f'{feature_name}')
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, (bar, value) in enumerate(zip(bars, y_values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_errors[j],
                       f'{value:.2f}', ha='center', va='bottom')
        
        # 隐藏多余的子图
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, 'feature_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"特征对比图已保存到: {output_path}")
    
    def plot_voltage_traces(self, time: np.ndarray, target_voltage: np.ndarray, 
                           model_voltage: np.ndarray, stimulus: np.ndarray):
        """绘制电压轨迹对比"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 刺激电流
        ax1.plot(time, stimulus, 'k-', linewidth=1)
        ax1.set_ylabel('刺激电流 (nA)')
        ax1.set_title('刺激协议')
        ax1.grid(True, alpha=0.3)
        
        # 目标电压轨迹
        ax2.plot(time, target_voltage, 'b-', linewidth=2, label='实验数据')
        ax2.set_ylabel('膜电位 (mV)')
        ax2.set_title('实验电压轨迹')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 模型电压轨迹
        ax3.plot(time, model_voltage, 'r-', linewidth=2, label='模型预测')
        ax3.set_xlabel('时间 (ms)')
        ax3.set_ylabel('膜电位 (mV)')
        ax3.set_title('模型电压轨迹')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, 'voltage_traces.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"电压轨迹图已保存到: {output_path}")
    
    def plot_parameter_distribution(self, parameter_history: Dict[str, List[float]]):
        """绘制参数分布图"""
        n_params = len(parameter_history)
        n_cols = 2
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (param_name, values) in enumerate(parameter_history.items()):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row, col]
            
            # 绘制参数变化
            ax.plot(values, 'b-', linewidth=2)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel(param_name)
            ax.set_title(f'{param_name} 优化过程')
            ax.grid(True, alpha=0.3)
            
            # 添加最终值标签
            final_value = values[-1]
            ax.text(0.02, 0.98, f'最终值: {final_value:.4f}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, 'parameter_optimization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"参数优化图已保存到: {output_path}")
    
    def create_summary_report(self, optimization_result: Dict, target_features: Dict):
        """创建总结报告"""
        report_path = os.path.join(self.results_dir, 'optimization_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("点过程模型优化报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 优化结果
            f.write("优化结果:\n")
            f.write(f"  成功: {optimization_result['success']}\n")
            f.write(f"  最终误差: {optimization_result['error']:.6f}\n")
            f.write(f"  迭代次数: {optimization_result.get('iterations', 'N/A')}\n\n")
            
            # 优化参数
            f.write("优化参数:\n")
            param_names = ['Cm', 'Rm', 'V_rest', 'g_Na_max', 'g_K_max', 'g_L']
            for i, param_name in enumerate(param_names):
                if i < len(optimization_result['parameters']):
                    f.write(f"  {param_name}: {optimization_result['parameters'][i]:.6f}\n")
            f.write("\n")
            
            # 特征对比
            f.write("特征对比:\n")
            f.write(f"{'特征名称':<25} {'目标值':<15} {'模型值':<15} {'误差':<15}\n")
            f.write("-" * 70 + "\n")
            
            for feature_name, target_value in target_features.items():
                target_mean = target_value[0]
                target_std = target_value[1]
                # 这里需要模型特征值，简化处理
                f.write(f"{feature_name:<25} {target_mean:<15.4f} {'N/A':<15} {'N/A':<15}\n")
        
        print(f"总结报告已保存到: {report_path}")

def main():
    """示例用法"""
    # 创建结果目录
    results_dir = "point_process_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建可视化器
    visualizer = PointProcessVisualizer(results_dir)
    
    # 示例数据
    optimization_history = [10.0, 8.5, 6.2, 4.1, 3.2, 2.8, 2.5, 2.3, 2.1, 2.0]
    
    target_features = {
        'voltage_base': [-70.0, 2.0],
        'mean_frequency': [20.0, 5.0],
        'AP_amplitude': [80.0, 10.0]
    }
    
    model_features = {
        'voltage_base': -69.5,
        'mean_frequency': 18.5,
        'AP_amplitude': 82.0
    }
    
    # 生成示例图表
    visualizer.plot_optimization_convergence(optimization_history)
    visualizer.plot_feature_comparison(target_features, model_features)
    
    print("可视化完成!")

if __name__ == "__main__":
    main()

