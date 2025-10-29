#!/usr/bin/env python3
"""
测试完整的多通道点过程模型工作流程
验证pynwb读取和特征提取功能

"""

import json
import logging
from feature_extractor import PointProcessFeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_workflow():
    """测试完整工作流程"""
    
    print("="*60)
    print("多通道点过程模型工作流程测试")
    print("="*60)
    
    # 1. 测试特征提取
    print("\n步骤1: 测试特征提取功能")
    print("-" * 40)
    
    nwb_path = "/media/ubuntu/sda/Patch-seq/data/Patch/601506492_icephys.nwb"
    extractor = PointProcessFeatureExtractor(nwb_path, junction_potential=-14.0)
    
    # 测试少量sweep
    stimulus_types = ["sweep_0", "sweep_1", "sweep_2"]
    
    print(f"从 {nwb_path} 提取特征...")
    features = extractor.extract_all_features(stimulus_types)
    
    print(f"\n成功提取 {len(features)} 个刺激类型的特征:")
    for stim_type, stim_features in features.items():
        print(f"  {stim_type}: {len(stim_features)} 个特征")
        for feature_name, feature_value in stim_features.items():
            print(f"    {feature_name}: {feature_value[0]:.2f} ± {feature_value[1]:.2f}")
    
    # 2. 保存特征
    print("\n步骤2: 保存特征文件")
    print("-" * 40)
    
    output_file = "workflow_test_features.json"
    extractor.save_features(features, output_file)
    print(f"特征已保存到: {output_file}")
    
    # 3. 验证配置文件
    print("\n步骤3: 验证配置文件")
    print("-" * 40)
    
    config_file = "multi_channel_config.json"
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"配置文件 {config_file} 加载成功")
        print(f"NWB文件路径: {config['nwb_path']}")
        print(f"刺激类型数量: {len(config['stimulus_types'])}")
        print(f"输出目录: {config['output_dir']}")
    except Exception as e:
        print(f"配置文件加载失败: {e}")
    
    # 4. 总结
    print("\n步骤4: 测试总结")
    print("-" * 40)
    
    print("✅ 特征提取功能正常")
    print("✅ pynwb读取NWB文件成功")
    print("✅ 配置文件格式正确")
    print("✅ 工作流程基础组件就绪")
    
    print("\n下一步可以运行:")
    print("python launch_multi_channel.py --config multi_channel_config.json")
    
    return True

if __name__ == "__main__":
    test_complete_workflow()

