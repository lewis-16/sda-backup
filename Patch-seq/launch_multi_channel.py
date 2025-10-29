#!/usr/bin/env python3
"""
多通道点过程模型启动脚本
基于All-active-Workflow-master的多通道设计，但无需形态学数据

使用方法:
python launch_multi_channel.py --config multi_channel_config.json

"""

import argparse
import json
import os
import logging
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from multi_channel_workflow_main import MultiChannelWorkflow
from feature_extractor import PointProcessFeatureExtractor

def setup_logging(log_level: str = 'INFO'):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('multi_channel.log')
        ]
    )

def validate_config(config: dict) -> bool:
    """验证配置文件"""
    required_fields = ['nwb_path']
    
    for field in required_fields:
        if field not in config:
            print(f"错误: 配置文件缺少必需字段 '{field}'")
            return False
    
    if not os.path.exists(config['nwb_path']):
        print(f"错误: NWB文件不存在: {config['nwb_path']}")
        return False
    
    return True

def extract_features_step(config: dict) -> str:
    """步骤1: 提取电生理特征"""
    print("\n" + "="*60)
    print("步骤1: 提取电生理特征")
    print("="*60)
    
    extractor = PointProcessFeatureExtractor(
        config['nwb_path'],
        config.get('junction_potential', -14.0)
    )
    
    features = extractor.extract_all_features(config.get('stimulus_types'))
    
    # 保存特征
    features_file = os.path.join(config.get('output_dir', 'multi_channel_results'), 'extracted_features.json')
    os.makedirs(os.path.dirname(features_file), exist_ok=True)
    extractor.save_features(features, features_file)
    
    print(f"特征提取完成，保存到: {features_file}")
    return features_file

def optimization_step(config: dict, features_file: str) -> dict:
    """步骤2: 多通道参数优化"""
    print("\n" + "="*60)
    print("步骤2: 多通道参数优化")
    print("="*60)
    
    # 加载特征
    with open(features_file, 'r') as f:
        features = json.load(f)
    
    # 创建优化配置
    opt_config = config.copy()
    opt_config['target_features'] = features
    
    # 创建临时配置文件
    temp_config_file = os.path.join(config.get('output_dir', 'multi_channel_results'), 'temp_config.json')
    os.makedirs(os.path.dirname(temp_config_file), exist_ok=True)
    with open(temp_config_file, 'w') as f:
        json.dump(opt_config, f, indent=2)
    
    # 运行优化
    workflow = MultiChannelWorkflow(temp_config_file)
    result = workflow.run_optimization()
    
    # 清理临时文件
    os.remove(temp_config_file)
    
    print("多通道参数优化完成!")
    return result

def visualization_step(config: dict, optimization_result: dict, features_file: str):
    """步骤3: 结果可视化"""
    print("\n" + "="*60)
    print("步骤3: 结果可视化")
    print("="*60)
    
    results_dir = config.get('output_dir', 'multi_channel_results')
    
    print("可视化功能已禁用（visualizer模块已删除）")
    print("优化结果已保存在以下文件中:")
    print(f"  - {results_dir}/optimization_result.json")
    print(f"  - {results_dir}/optimized_parameters.json")
    print(f"  - {results_dir}/model_performance.json")
    
    print("可视化完成!")

def print_channel_summary(optimization_result: dict):
    """打印通道参数摘要"""
    print("\n" + "="*60)
    print("多通道参数摘要")
    print("="*60)
    
    model = optimization_result['model']
    params = optimization_result['parameters']
    
    # 通道参数名称
    param_names = [
        'Cm', 'Ra', 'g_pas', 'e_pas',
        'gbar_NaTs2_t', 'gbar_NaTa_t', 'gbar_Nap_Et2',
        'gbar_K_Tst', 'gbar_Kv3_1', 'gbar_K_Pst', 'gbar_Kd',
        'gbar_Ca_HVA', 'gbar_Ca_LVA',
        'gbar_SK', 'gbar_BK_gc',
        'gbar_HCN', 'gbar_Ih',
        'gbar_Im', 'gbar_Kir21_gc'
    ]
    
    print("被动参数:")
    print(f"  膜电容 (Cm): {params[0]:.4f} μF/cm²")
    print(f"  轴向电阻 (Ra): {params[1]:.4f} Ω·cm")
    print(f"  被动电导 (g_pas): {params[2]:.6f} S/cm²")
    print(f"  被动平衡电位 (e_pas): {params[3]:.2f} mV")
    
    print("\n钠通道:")
    print(f"  NaTs2_t: {params[4]:.6f} S/cm²")
    print(f"  NaTa_t: {params[5]:.6f} S/cm²")
    print(f"  Nap_Et2: {params[6]:.6f} S/cm²")
    
    print("\n钾通道:")
    print(f"  K_Tst: {params[7]:.6f} S/cm²")
    print(f"  Kv3_1: {params[8]:.6f} S/cm²")
    print(f"  K_Pst: {params[9]:.6f} S/cm²")
    print(f"  Kd: {params[10]:.6f} S/cm²")
    
    print("\n钙通道:")
    print(f"  Ca_HVA: {params[11]:.6f} S/cm²")
    print(f"  Ca_LVA: {params[12]:.6f} S/cm²")
    
    print("\n钙依赖性通道:")
    print(f"  SK: {params[13]:.6f} S/cm²")
    print(f"  BK_gc: {params[14]:.6f} S/cm²")
    
    print("\nHCN通道:")
    print(f"  HCN: {params[15]:.6f} S/cm²")
    print(f"  Ih: {params[16]:.6f} S/cm²")
    
    print("\n其他通道:")
    print(f"  Im: {params[17]:.6f} S/cm²")
    print(f"  Kir21_gc: {params[18]:.6f} S/cm²")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多通道点过程模型工作流程')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='跳过特征提取步骤（如果已有特征文件）')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='跳过优化步骤')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='跳过可视化步骤')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载配置文件 {args.config}: {e}")
        return 1
    
    # 验证配置
    if not validate_config(config):
        return 1
    
    # 创建输出目录
    output_dir = config.get('output_dir', 'multi_channel_results')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 步骤1: 特征提取
        if not args.skip_extraction:
            features_file = extract_features_step(config)
        else:
            features_file = os.path.join(output_dir, 'extracted_features.json')
            if not os.path.exists(features_file):
                print(f"错误: 特征文件不存在: {features_file}")
                return 1
        
        # 步骤2: 参数优化
        if not args.skip_optimization:
            optimization_result = optimization_step(config, features_file)
        else:
            # 加载已有结果
            result_file = os.path.join(output_dir, 'optimized_parameters.json')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    optimization_result = json.load(f)
            else:
                print(f"错误: 优化结果文件不存在: {result_file}")
                return 1
        
        # 步骤3: 可视化
        if not args.skip_visualization:
            visualization_step(config, optimization_result, features_file)
        
        # 打印通道参数摘要
        print_channel_summary(optimization_result)
        
        print("\n" + "="*60)
        print("多通道点过程模型工作流程完成!")
        print("="*60)
        print(f"结果保存在: {output_dir}")
        print(f"优化误差: {optimization_result.get('error', 'N/A')}")
        print(f"优化成功: {optimization_result.get('success', 'N/A')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"工作流程执行出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
