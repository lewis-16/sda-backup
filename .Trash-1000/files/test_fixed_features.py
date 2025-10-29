#!/usr/bin/env python3
"""
测试修复后的特征提取功能
验证sag_amplitude警告是否减少

"""

import json
import logging
from feature_extractor import PointProcessFeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_feature_extraction():
    """测试修复后的特征提取"""
    
    print("="*60)
    print("测试修复后的特征提取功能")
    print("="*60)
    
    # NWB文件路径
    nwb_path = "/media/ubuntu/sda/Patch-seq/data/Patch/601506492_icephys.nwb"
    
    # 创建特征提取器
    extractor = PointProcessFeatureExtractor(nwb_path, junction_potential=-14.0)
    
    print(f"从 {nwb_path} 提取台阶电流特征...")
    print("注意观察sag_amplitude警告是否减少...")
    
    # 使用台阶电流方法提取特征
    features = extractor.extract_step_current_features()
    
    print(f"\n成功提取 {len(features)} 个幅度组的特征:")
    
    # 统计不同幅度组的特征
    negative_amplitudes = []
    positive_amplitudes = []
    
    for amplitude_key, stim_features in features.items():
        amplitude = int(amplitude_key.split('_')[1])
        if amplitude < 0:
            negative_amplitudes.append(amplitude)
        else:
            positive_amplitudes.append(amplitude)
        
        print(f"\n{amplitude_key}:")
        print(f"  特征数量: {len(stim_features)}")
        
        # 检查是否有sag特征
        has_sag = any('sag' in feature_name for feature_name in stim_features.keys())
        print(f"  包含sag特征: {'是' if has_sag else '否'}")
        
        # 显示前几个特征
        for i, (feature_name, feature_value) in enumerate(stim_features.items()):
            if i < 3:  # 只显示前3个特征
                print(f"    {feature_name}: {feature_value[0]:.2f} ± {feature_value[1]:.2f}")
    
    print(f"\n负电流幅度组: {sorted(negative_amplitudes)}")
    print(f"正电流幅度组: {sorted(positive_amplitudes)}")
    
    # 保存特征
    output_file = "fixed_step_current_features.json"
    extractor.save_features(features, output_file)
    print(f"\n特征已保存到: {output_file}")
    
    return features

if __name__ == "__main__":
    test_fixed_feature_extraction()

