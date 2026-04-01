#!/usr/bin/env python3
"""
生成Fig 2e-2i的脚本
根据customize目录下的分析结果生成所有需要的图
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/media/ubuntu/sda/TrippleN'
OUTPUT_DIR = os.path.join(BASE_DIR, 'customize', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_rsa_results():
    """加载RSA分析结果"""
    rsa_path = os.path.join(BASE_DIR, 'customize', 'RSA_analysis', 'rsa_results.pkl')
    return pd.read_pickle(rsa_path)


def load_encoding_results():
    """加载所有encoding分析结果"""
    encoding_dir = os.path.join(BASE_DIR, 'customize', 'encoding_analysis')
    encoding_files = [f for f in os.listdir(encoding_dir) if f.endswith('_gpu.pkl')]
    
    encoding_results = {}
    for f in sorted(encoding_files):
        file_path = os.path.join(encoding_dir, f)
        model_name = f.replace('_encoding_results_gpu.pkl', '').replace('_', ' ')
        
        data = pd.read_pickle(file_path)
        encoding_results[model_name] = data
    
    return encoding_results


def load_decoding_results():
    """加载decoding分析结果"""
    decoding_path = os.path.join(BASE_DIR, 'customize', 'decoding_analysis', 'decoding_results_loocv.pkl')
    with open(decoding_path, 'rb') as f:
        return pickle.load(f)


def load_slope_data():
    """加载slope数据"""
    slope_path = os.path.join(BASE_DIR, 'customize', 'encoding_analysis', 'slope_vs_alexnet.csv')
    return pd.read_csv(slope_path)


def load_unit_info():
    """加载unit信息"""
    unit_info_path = os.path.join(BASE_DIR, 'customize', 'aggregate_response', 'all_subjects_unit_info.pkl')
    return pd.read_pickle(unit_info_path)


def plot_fig_2e(rsa_data, output_path):
    """Fig 2e: RSA searchlight分析结果"""
    all_data = rsa_data[rsa_data['region_name'] == 'all'].copy()
    
    model_order = all_data.groupby('model_name')['spearman_r'].mean().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(model_order))
    bars = ax.bar(x_pos, all_data.groupby('model_name')['spearman_r'].mean()[model_order], 
                   color='#5E9FD1', alpha=0.8, width=0.6)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('RSA Spearman Correlation', fontsize=12)
    ax.set_title('RSA Searchlight Analysis Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Fig 2e to {output_path}")


def plot_fig_2f(rsa_data, output_path):
    """Fig 2f: Region-specific RSA对比"""
    region_data = rsa_data[rsa_data['region_name'].isin(['middle', 'anterior'])].copy()
    
    middle_data = region_data[region_data['region_name'] == 'middle'].groupby('model_name')['spearman_r'].mean()
    anterior_data = region_data[region_data['region_name'] == 'anterior'].groupby('model_name')['spearman_r'].mean()
    
    models = sorted(set(middle_data.index) & set(anterior_data.index))
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, [middle_data[m] for m in models], width, 
                    label='Middle IT', color='#5E9FD1', alpha=0.8)
    bars2 = ax.bar(x + width/2, [anterior_data[m] for m in models], width,
                    label='Anterior IT', color='#EC6F7E', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('RSA Spearman Correlation', fontsize=12)
    ax.set_title('Region-Specific RSA Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Fig 2f to {output_path}")


def plot_fig_2g(encoding_results, unit_info, output_path):
    """Fig 2g: Encoding model performance comparison"""
    model_names = []
    mean_performances = []
    
    for model_name, data in encoding_results.items():
        if 'normalized_correlation' in data.columns:
            mean_perf = data['normalized_correlation'].mean()
            model_names.append(model_name)
            mean_performances.append(mean_perf)
    
    df = pd.DataFrame({'model': model_names, 'performance': mean_performances})
    df = df.sort_values('performance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(df)), df['performance'], color='#5E9FD1', alpha=0.8, width=0.6)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean Normalized Correlation', fontsize=12)
    ax.set_title('Encoding Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Fig 2g to {output_path}")


def plot_fig_2h(slope_data, output_path):
    """Fig 2h: Model prediction index (slope)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    bar_df = slope_data[['model', 'slope_all']].dropna().sort_values('slope_all', ascending=True)
    
    axes[0].barh(range(len(bar_df)), bar_df['slope_all'], color='#5E9FD1', alpha=0.8, height=0.6)
    axes[0].set_yticks(range(len(bar_df)))
    axes[0].set_yticklabels(bar_df['model'])
    axes[0].set_xlabel('Slope (vs AlexNet)', fontsize=12)
    axes[0].set_title('Model Prediction Index', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0.5, 1.0)
    axes[0].grid(axis='x', alpha=0.3)
    
    pair_df = slope_data[['model', 'slope_middle', 'slope_anterior']].dropna()
    x1 = pair_df['slope_middle'].values
    x2 = pair_df['slope_anterior'].values
    
    t_stat, p_val = stats.ttest_rel(x1, x2)
    
    long_df = pd.DataFrame({
        'group': np.repeat(['middle', 'anterior'], repeats=[len(x1), len(x2)]),
        'slope': np.concatenate([x1, x2])
    })
    
    sns.boxplot(data=long_df, x='group', y='slope', ax=axes[1], color='#5E9FD1', width=0.5)
    sns.stripplot(data=long_df, x='group', y='slope', ax=axes[1], color='black', size=4, alpha=0.7, jitter=0.08)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_ylabel('Slope (vs AlexNet)', fontsize=12)
    axes[1].set_xlabel('Region', fontsize=12)
    axes[1].set_title('Slope Comparison: Middle vs Anterior', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    y_max = np.nanmax(long_df['slope'].values)
    y_line = min(0.98, y_max + 0.05)
    axes[1].plot([0, 0, 1, 1], [y_line-0.01, y_line, y_line, y_line-0.01], color='black', linewidth=1)
    axes[1].text(0.5, y_line + 0.01, f'p={p_val:.3g}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Fig 2h to {output_path}")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_val:.4e}, n_models={len(pair_df)}")


def plot_fig_2i(decoding_results, output_path):
    """Fig 2i: Decoding analysis results"""
    model_names = []
    mean_corrs = []
    
    for model_name, result in decoding_results.items():
        if 'mean_corr' in result:
            model_names.append(model_name.replace('_', ' ').title())
            mean_corrs.append(result['mean_corr'])
    
    df = pd.DataFrame({'model': model_names, 'mean_corr': mean_corrs})
    df = df.sort_values('mean_corr', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(df)), df['mean_corr'], color='#5E9FD1', alpha=0.8, width=0.6)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Decoding Accuracy (Mean Correlation)', fontsize=12)
    ax.set_title('Decoding Analysis Results', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Fig 2i to {output_path}")


def main():
    """主函数"""
    print("="*70)
    print("生成Fig 2e-2i")
    print("="*70)
    
    print("\n[1/6] 加载RSA分析结果...")
    rsa_data = load_rsa_results()
    print(f"  加载了 {len(rsa_data)} 条RSA记录")
    
    print("\n[2/6] 加载Encoding分析结果...")
    encoding_results = load_encoding_results()
    print(f"  加载了 {len(encoding_results)} 个模型的encoding结果")
    
    print("\n[3/6] 加载Decoding分析结果...")
    decoding_results = load_decoding_results()
    print(f"  加载了 {len(decoding_results)} 个模型的decoding结果")
    
    print("\n[4/6] 加载Slope数据...")
    slope_data = load_slope_data()
    print(f"  加载了 {len(slope_data)} 个模型的slope数据")
    
    print("\n[5/6] 加载Unit信息...")
    unit_info = load_unit_info()
    print(f"  加载了 {len(unit_info)} 个unit的信息")
    
    print("\n[6/6] 生成图片...")
    plot_fig_2e(rsa_data, os.path.join(OUTPUT_DIR, 'Fig_2e_RSA_searchlight.pdf'))
    plot_fig_2f(rsa_data, os.path.join(OUTPUT_DIR, 'Fig_2f_region_specific_RSA.pdf'))
    plot_fig_2g(encoding_results, unit_info, os.path.join(OUTPUT_DIR, 'Fig_2g_encoding_performance.pdf'))
    plot_fig_2h(slope_data, os.path.join(OUTPUT_DIR, 'Fig_2h_model_prediction_index.pdf'))
    plot_fig_2i(decoding_results, os.path.join(OUTPUT_DIR, 'Fig_2i_decoding_results.pdf'))
    
    print("\n" + "="*70)
    print("所有图片已生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
