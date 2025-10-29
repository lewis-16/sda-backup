#!/usr/bin/env python3
"""
测试台阶电流特征提取功能
参考neuron_model.ipynb的方法

"""

import json
import logging
from feature_extractor import PointProcessFeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_step_current_extraction():
    """测试台阶电流特征提取"""
    
    print("="*60)
    print("台阶电流特征提取测试")
    print("="*60)
    
    # NWB文件路径
    nwb_path = "/media/ubuntu/sda/Patch-seq/data/Patch/601506492_icephys.nwb"
    
    # 创建特征提取器
    extractor = PointProcessFeatureExtractor(nwb_path, junction_potential=-14.0)
    
    print(f"从 {nwb_path} 提取台阶电流特征...")
    
    # 使用台阶电流方法提取特征
    features = extractor.extract_step_current_features()
    
    print(f"\n成功提取 {len(features)} 个幅度组的特征:")
    for amplitude, stim_features in features.items():
        print(f"\n{amplitude}:")
        for feature_name, feature_value in stim_features.items():
            print(f"  {feature_name}: {feature_value[0]:.2f} ± {feature_value[1]:.2f}")
    
    # 保存特征
    output_file = "step_current_features.json"
    extractor.save_features(features, output_file)
    print(f"\n特征已保存到: {output_file}")
    
    return features

if __name__ == "__main__":
    test_step_current_extraction()

