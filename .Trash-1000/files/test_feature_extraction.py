#!/usr/bin/env python3
"""
测试多通道点过程模型工作流程
使用pynwb读取NWB文件

"""

import json
import logging
from feature_extractor import PointProcessFeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_extraction():
    """测试特征提取功能"""
    
    # NWB文件路径
    nwb_path = "/media/ubuntu/sda/Patch-seq/data/Patch/601506492_icephys.nwb"
    
    # 创建特征提取器
    extractor = PointProcessFeatureExtractor(nwb_path, junction_potential=-14.0)
    
    # 测试提取特征
    stimulus_types = ["sweep_0", "sweep_1", "sweep_2", "sweep_3", "sweep_4"]
    
    print("开始提取电生理特征...")
    features = extractor.extract_all_features(stimulus_types)
    
    print(f"\n提取的特征:")
    for stim_type, stim_features in features.items():
        print(f"\n{stim_type}:")
        for feature_name, feature_value in stim_features.items():
            print(f"  {feature_name}: {feature_value}")
    
    # 保存特征
    output_file = "test_extracted_features.json"
    extractor.save_features(features, output_file)
    print(f"\n特征已保存到: {output_file}")

if __name__ == "__main__":
    test_feature_extraction()
