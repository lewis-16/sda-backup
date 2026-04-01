"""
按单个脑区计算 Alexnet/Sentence Encoding 斜率
脑区: MF, AF, MB, AB, MO, AO
"""
import numpy as np
import pandas as pd
from scipy import stats

# 定义六个独立的脑区
single_areas = ['MF', 'AF', 'MB', 'AB', 'MO', 'AO']

# 获取数据
alexnet_data = encoding_results['alexnet']['normalized_correlation']
sentence_data = encoding_results['sentence']['normalized_correlation']

# 创建 DataFrame
data_df = pd.DataFrame({
    'alexnet': alexnet_data,
    'sentence': sentence_data,
    'AREALABEL': unit_info['AREALABEL'].values
})

# 计算各脑区的斜率
slope_results = {}

for area in single_areas:
    # 筛选该脑区的单元
    mask = data_df['AREALABEL'].apply(lambda x: x == area)
    area_df = data_df[mask]

    # 过滤数据：只保留alexnet和sentence都在0-1之间的点
    filtered_df = area_df[(area_df['alexnet'] >= 0) & (area_df['alexnet'] <= 1) & 
                          (area_df['sentence'] >= 0) & (area_df['sentence'] <= 1)]

    x_data = filtered_df['alexnet'].values
    y_data = filtered_df['sentence'].values

    # 计算通过原点的线性回归系数 a (y = ax)
    if len(x_data) > 0:
        a = np.sum(x_data * y_data) / np.sum(x_data**2)

        # 计算R²值
        y_pred = a * x_data
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 计算皮尔逊相关系数
        correlation = np.corrcoef(x_data, y_data)[0, 1]

        # 包含截距的线性拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)

        slope_results[area] = {
            '斜率(无截距)': a,
            '斜率(含截距)': slope,
            '截距': intercept,
            'R²': r_squared,
            '相关系数r': correlation,
            'p值': p_value,
            '样本数': len(x_data)
        }

        print(f"\n{'='*50}")
        print(f"脑区: {area}")
        print(f"{'='*50}")
        print(f"  原始单元数: {len(area_df)}")
        print(f"  过滤后样本数: {len(x_data)}")
        print(f"  斜率 (无截距 y=ax): {a:.4f}")
        print(f"  斜率 (含截距 y=ax+b): {slope:.4f}")
        print(f"  截距: {intercept:.4f}")
        print(f"  R²: {r_squared:.4f}")
        print(f"  皮尔逊相关系数 r: {correlation:.4f}")
        print(f"  p值: {p_value:.4e}")
    else:
        print(f"\n脑区 {area} 过滤后没有数据！")

# 创建汇总表格
slope_df = pd.DataFrame(slope_results).T
print("\n" + "="*60)
print("各脑区斜率汇总表")
print("="*60)
slope_df
