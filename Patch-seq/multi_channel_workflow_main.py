#!/usr/bin/env python3
"""
多通道点过程模型工作流程主类
基于All-active-Workflow-master的多通道设计，但无需形态学数据

"""

import numpy as np
import json
import os
import logging
from typing import Dict, List
from multi_channel_workflow import MultiChannelModel, MultiChannelOptimizer
from feature_extractor import PointProcessFeatureExtractor
from memory_monitor import setup_memory_monitor

logger = logging.getLogger(__name__)

class MultiChannelWorkflow:
    """多通道点过程模型工作流程主类"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = MultiChannelModel()
        self.extractor = PointProcessFeatureExtractor(
            self.config['nwb_path'],
            self.config.get('junction_potential', -14.0)
        )
        
        # 初始化内存监控
        max_memory_gb = self.config.get('max_memory_gb', 60.0)
        warning_threshold_gb = self.config.get('warning_memory_gb', 50.0)
        
        logger.info(f"设置内存监控: 最大{max_memory_gb}GB, 警告阈值{warning_threshold_gb}GB")
        self.memory_monitor = setup_memory_monitor(
            max_memory_gb=max_memory_gb,
            warning_threshold_gb=warning_threshold_gb
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def run_optimization(self) -> Dict:
        """运行完整的优化流程"""
        logger.info("开始多通道点过程模型优化流程...")
        
        try:
            # 1. 提取电生理特征
            logger.info("步骤1: 提取电生理特征")
            target_features = self.extractor.extract_all_features(
                self.config.get('stimulus_types')
            )
            
            # 检查内存使用
            self._check_memory_usage("特征提取后")
            
            # 2. 创建优化器
            logger.info("步骤2: 创建多通道参数优化器")
            optimizer = MultiChannelOptimizer(self.model, target_features)
            
            # 检查内存使用
            self._check_memory_usage("优化器创建后")
            
            # 3. 执行优化
            logger.info("步骤3: 执行多通道参数优化")
            result = optimizer.optimize()
            
            # 检查内存使用
            self._check_memory_usage("优化完成后")
            
            # 4. 保存结果
            logger.info("步骤4: 保存优化结果")
            self._save_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"优化过程出错: {e}")
            # 显示最终内存使用情况
            self._check_memory_usage("出错时")
            raise
        finally:
            # 停止内存监控
            if self.memory_monitor:
                self.memory_monitor.stop_monitoring()
                logger.info("内存监控已停止")
    
    def _check_memory_usage(self, stage: str):
        """检查内存使用情况"""
        if self.memory_monitor:
            info = self.memory_monitor.get_memory_info()
            logger.info(f"内存使用情况 ({stage}):")
            logger.info(f"  当前进程内存: {info['process_memory_gb']:.2f} GB")
            logger.info(f"  系统总内存: {info['system_memory_gb']:.2f} GB")
            logger.info(f"  内存使用率: {info['memory_usage_percent']:.1f}%")
            
            # 如果接近警告阈值，发出警告
            if info['memory_usage_percent'] > 80:
                logger.warning(f"⚠️  内存使用率较高: {info['memory_usage_percent']:.1f}%")
    
    def _save_results(self, result: Dict):
        """保存优化结果"""
        output_dir = self.config.get('output_dir', 'multi_channel_results')
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存参数
        params_file = os.path.join(output_dir, 'optimized_parameters.json')
        with open(params_file, 'w') as f:
            json.dump({
                'parameters': result['parameters'].tolist(),
                'error': result['error'],
                'success': result['success']
            }, f, indent=2)
        
        # 保存模型配置
        model_file = os.path.join(output_dir, 'multi_channel_model.json')
        model_dict = {
            'Cm': result['model'].Cm,
            'Ra': result['model'].Ra,
            'g_pas': result['model'].g_pas,
            'e_pas': result['model'].e_pas,
            'ena': result['model'].ena,
            'ek': result['model'].ek,
            'eca': result['model'].eca,
            'gbar_NaTs2_t': result['model'].gbar_NaTs2_t,
            'gbar_NaTa_t': result['model'].gbar_NaTa_t,
            'gbar_Nap_Et2': result['model'].gbar_Nap_Et2,
            'gbar_K_Tst': result['model'].gbar_K_Tst,
            'gbar_Kv3_1': result['model'].gbar_Kv3_1,
            'gbar_K_Pst': result['model'].gbar_K_Pst,
            'gbar_Kd': result['model'].gbar_Kd,
            'gbar_Ca_HVA': result['model'].gbar_Ca_HVA,
            'gbar_Ca_LVA': result['model'].gbar_Ca_LVA,
            'gbar_SK': result['model'].gbar_SK,
            'gbar_BK_gc': result['model'].gbar_BK_gc,
            'gbar_HCN': result['model'].gbar_HCN,
            'gbar_Ih': result['model'].gbar_Ih,
            'gbar_Im': result['model'].gbar_Im,
            'gbar_Kir21_gc': result['model'].gbar_Kir21_gc,
            'gamma_CaDynamics': result['model'].gamma_CaDynamics,
            'decay_CaDynamics': result['model'].decay_CaDynamics,
            'celsius': result['model'].celsius,
            'v_init': result['model'].v_init
        }
        
        with open(model_file, 'w') as f:
            json.dump(model_dict, f, indent=2)
        
        logger.info(f"结果已保存到 {output_dir}")

if __name__ == "__main__":
    # 创建示例配置文件
    config = {
        'nwb_path': 'example_data.nwb',
        'junction_potential': -14.0,
        'stimulus_types': ['Long Square', 'Ramp'],
        'output_dir': 'multi_channel_results'
    }
    
    config_file = 'multi_channel_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 运行工作流程
    workflow = MultiChannelWorkflow(config_file)
    result = workflow.run_optimization()
    
    print("多通道优化完成!")
    print(f"最终误差: {result['error']}")
    print(f"优化成功: {result['success']}")
