#!/usr/bin/env python3
"""
点过程模型工作流程 - 基于电生理数据的神经元模型拟合
无需形态学数据，仅使用Patch-seq电生理记录进行参数优化

作者: AI Assistant
基于: All-active-Workflow-master 简化版本
"""

import numpy as np
import json
import os
import logging
from collections import defaultdict
from scipy.optimize import minimize
from scipy.stats import poisson, gamma
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import efel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PointProcessModel:
    """点过程神经元模型参数"""
    # 被动参数
    Cm: float = 1.0  # 膜电容 (μF/cm²)
    Rm: float = 100.0  # 膜电阻 (MΩ·cm²)
    V_rest: float = -70.0  # 静息电位 (mV)
    V_thresh: float = -50.0  # 阈值电位 (mV)
    
    # 主动参数 (简化的HH模型)
    g_Na_max: float = 120.0  # 最大钠电导 (mS/cm²)
    g_K_max: float = 36.0   # 最大钾电导 (mS/cm²)
    g_L: float = 0.3        # 漏电导 (mS/cm²)
    
    # 离子平衡电位
    E_Na: float = 50.0      # 钠平衡电位 (mV)
    E_K: float = -77.0      # 钾平衡电位 (mV)
    E_L: float = -54.4      # 漏平衡电位 (mV)
    
    # 温度系数
    Q10: float = 3.0        # 温度系数

class ElectrophysiologyExtractor:
    """电生理特征提取器"""
    
    def __init__(self, nwb_path: str, junction_potential: float = -14.0):
        self.nwb_path = nwb_path
        self.junction_potential = junction_potential
        self.features = {}
        
    def extract_features(self, stimulus_types: List[str]) -> Dict:
        """从NWB文件提取电生理特征"""
        logger.info(f"从 {self.nwb_path} 提取电生理特征...")
        
        # 这里需要根据实际的NWB文件格式进行调整
        # 简化版本：假设已经预处理好的数据
        features = {}
        
        for stim_type in stimulus_types:
            features[stim_type] = self._extract_stimulus_features(stim_type)
            
        return features
    
    def _extract_stimulus_features(self, stim_type: str) -> Dict:
        """提取特定刺激类型的特征"""
        # 模拟特征提取过程
        # 实际实现中需要读取NWB文件并提取真实数据
        
        if stim_type == "Long Square":
            return {
                "voltage_base": [-70.0, 2.0],
                "steady_state_voltage": [-65.0, 3.0],
                "voltage_deflection": [5.0, 1.0],
                "decay_time_constant": [10.0, 2.0],
                "spike_count": [0, 0],  # 非发放
                "sag_amplitude": [2.0, 0.5],
                "sag_ratio": [0.1, 0.02]
            }
        elif stim_type == "Ramp":
            return {
                "voltage_base": [-70.0, 2.0],
                "spike_count": [5, 1],  # 发放
                "mean_frequency": [20.0, 5.0],
                "time_to_first_spike": [100.0, 20.0],
                "AP_amplitude": [80.0, 10.0],
                "ISI_CV": [0.3, 0.1],
                "adaptation_index": [0.5, 0.2]
            }
        else:
            return {}

class PointProcessSimulator:
    """点过程神经元模拟器"""
    
    def __init__(self, model: PointProcessModel):
        self.model = model
        
    def simulate_response(self, stimulus: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """模拟神经元对刺激的响应"""
        n_steps = len(stimulus)
        V = np.zeros(n_steps)
        spikes = np.zeros(n_steps, dtype=bool)
        
        # 初始化
        V[0] = self.model.V_rest
        
        # 简化的HH模型参数
        m = 0.0
        h = 1.0
        n = 0.0
        
        for i in range(1, n_steps):
            # 计算门控变量
            alpha_m = 0.1 * (V[i-1] + 40) / (1 - np.exp(-(V[i-1] + 40) / 10))
            beta_m = 4 * np.exp(-(V[i-1] + 65) / 18)
            alpha_h = 0.07 * np.exp(-(V[i-1] + 65) / 20)
            beta_h = 1 / (1 + np.exp(-(V[i-1] + 35) / 10))
            alpha_n = 0.01 * (V[i-1] + 55) / (1 - np.exp(-(V[i-1] + 55) / 10))
            beta_n = 0.125 * np.exp(-(V[i-1] + 65) / 80)
            
            # 更新门控变量
            m += dt * (alpha_m * (1 - m) - beta_m * m)
            h += dt * (alpha_h * (1 - h) - beta_h * h)
            n += dt * (alpha_n * (1 - n) - beta_n * n)
            
            # 计算电流
            I_Na = self.model.g_Na_max * m**3 * h * (V[i-1] - self.model.E_Na)
            I_K = self.model.g_K_max * n**4 * (V[i-1] - self.model.E_K)
            I_L = self.model.g_L * (V[i-1] - self.model.E_L)
            
            # 更新膜电位
            dV_dt = (stimulus[i-1] - I_Na - I_K - I_L) / self.model.Cm
            V[i] = V[i-1] + dt * dV_dt
            
            # 检测动作电位
            if V[i] > self.model.V_thresh and V[i-1] <= self.model.V_thresh:
                spikes[i] = True
                V[i] = 20.0  # 动作电位峰值
                
        return V, spikes
    
    def extract_model_features(self, stimulus: np.ndarray, dt: float = 0.1) -> Dict:
        """从模拟响应中提取特征"""
        V, spikes = self.simulate_response(stimulus, dt)
        
        features = {}
        
        # 基础特征
        features["voltage_base"] = np.mean(V[:100])  # 前100个点的平均值
        features["steady_state_voltage"] = np.mean(V[-100:])  # 后100个点的平均值
        
        # 发放特征
        spike_times = np.where(spikes)[0] * dt
        features["spike_count"] = len(spike_times)
        
        if len(spike_times) > 0:
            features["mean_frequency"] = len(spike_times) / (len(stimulus) * dt) * 1000  # Hz
            features["time_to_first_spike"] = spike_times[0]
            
            if len(spike_times) > 1:
                isi = np.diff(spike_times)
                features["ISI_CV"] = np.std(isi) / np.mean(isi)
                features["adaptation_index"] = self._calculate_adaptation_index(isi)
            else:
                features["ISI_CV"] = 0.0
                features["adaptation_index"] = 0.0
        else:
            features["mean_frequency"] = 0.0
            features["time_to_first_spike"] = np.nan
            features["ISI_CV"] = 0.0
            features["adaptation_index"] = 0.0
            
        return features
    
    def _calculate_adaptation_index(self, isi: np.ndarray) -> float:
        """计算适应指数"""
        if len(isi) < 2:
            return 0.0
        return (isi[-1] - isi[0]) / (isi[-1] + isi[0])

class PointProcessOptimizer:
    """点过程模型参数优化器"""
    
    def __init__(self, model: PointProcessModel, target_features: Dict):
        self.model = model
        self.target_features = target_features
        self.simulator = PointProcessSimulator(model)
        
    def objective_function(self, params: np.ndarray) -> float:
        """目标函数：计算模型特征与目标特征的差异"""
        # 更新模型参数
        self._update_model_parameters(params)
        
        total_error = 0.0
        
        for stim_type, target_feat in self.target_features.items():
            # 生成刺激
            stimulus = self._generate_stimulus(stim_type)
            
            # 模拟响应并提取特征
            model_features = self.simulator.extract_model_features(stimulus)
            
            # 计算误差
            error = self._calculate_feature_error(model_features, target_feat)
            total_error += error
            
        return total_error
    
    def _update_model_parameters(self, params: np.ndarray):
        """更新模型参数"""
        # 参数顺序：[Cm, Rm, V_rest, g_Na_max, g_K_max, g_L]
        self.model.Cm = params[0]
        self.model.Rm = params[1]
        self.model.V_rest = params[2]
        self.model.g_Na_max = params[3]
        self.model.g_K_max = params[4]
        self.model.g_L = params[5]
        
    def _generate_stimulus(self, stim_type: str) -> np.ndarray:
        """生成刺激电流"""
        duration = 1000  # ms
        dt = 0.1
        n_steps = int(duration / dt)
        
        if stim_type == "Long Square":
            # 长方波刺激
            stimulus = np.zeros(n_steps)
            stim_start = int(100 / dt)  # 100ms后开始
            stim_end = int(900 / dt)    # 900ms结束
            amplitude = 0.1  # nA
            stimulus[stim_start:stim_end] = amplitude
            
        elif stim_type == "Ramp":
            # 斜坡刺激
            stimulus = np.zeros(n_steps)
            stim_start = int(100 / dt)
            stim_end = int(900 / dt)
            ramp_duration = stim_end - stim_start
            amplitude = 0.2  # nA
            stimulus[stim_start:stim_end] = np.linspace(0, amplitude, ramp_duration)
            
        else:
            stimulus = np.zeros(n_steps)
            
        return stimulus
    
    def _calculate_feature_error(self, model_features: Dict, target_features: Dict) -> float:
        """计算特征误差"""
        error = 0.0
        
        for feature_name, target_value in target_features.items():
            if feature_name in model_features:
                model_value = model_features[feature_name]
                target_mean = target_value[0]
                target_std = target_value[1]
                
                # 归一化误差
                if target_std > 0:
                    error += ((model_value - target_mean) / target_std) ** 2
                else:
                    error += (model_value - target_mean) ** 2
                    
        return error
    
    def optimize(self, initial_params: Optional[np.ndarray] = None) -> Dict:
        """执行参数优化"""
        if initial_params is None:
            initial_params = np.array([
                self.model.Cm,
                self.model.Rm,
                self.model.V_rest,
                self.model.g_Na_max,
                self.model.g_K_max,
                self.model.g_L
            ])
        
        # 参数边界
        bounds = [
            (0.1, 10.0),    # Cm
            (10.0, 1000.0), # Rm
            (-90.0, -50.0), # V_rest
            (10.0, 200.0),  # g_Na_max
            (5.0, 100.0),   # g_K_max
            (0.01, 1.0)     # g_L
        ]
        
        logger.info("开始参数优化...")
        result = minimize(
            self.objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        logger.info(f"优化完成，最终误差: {result.fun}")
        
        # 更新最终参数
        self._update_model_parameters(result.x)
        
        return {
            'success': result.success,
            'error': result.fun,
            'parameters': result.x,
            'model': self.model
        }

class PointProcessWorkflow:
    """点过程模型工作流程主类"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = PointProcessModel()
        self.extractor = ElectrophysiologyExtractor(
            self.config['nwb_path'],
            self.config.get('junction_potential', -14.0)
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def run_optimization(self) -> Dict:
        """运行完整的优化流程"""
        logger.info("开始点过程模型优化流程...")
        
        # 1. 提取电生理特征
        logger.info("步骤1: 提取电生理特征")
        target_features = self.extractor.extract_features(
            self.config['stimulus_types']
        )
        
        # 2. 创建优化器
        logger.info("步骤2: 创建参数优化器")
        optimizer = PointProcessOptimizer(self.model, target_features)
        
        # 3. 执行优化
        logger.info("步骤3: 执行参数优化")
        result = optimizer.optimize()
        
        # 4. 保存结果
        logger.info("步骤4: 保存优化结果")
        self._save_results(result)
        
        return result
    
    def _save_results(self, result: Dict):
        """保存优化结果"""
        output_dir = self.config.get('output_dir', 'point_process_results')
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
        model_file = os.path.join(output_dir, 'point_process_model.json')
        model_dict = {
            'Cm': result['model'].Cm,
            'Rm': result['model'].Rm,
            'V_rest': result['model'].V_rest,
            'V_thresh': result['model'].V_thresh,
            'g_Na_max': result['model'].g_Na_max,
            'g_K_max': result['model'].g_K_max,
            'g_L': result['model'].g_L,
            'E_Na': result['model'].E_Na,
            'E_K': result['model'].E_K,
            'E_L': result['model'].E_L
        }
        
        with open(model_file, 'w') as f:
            json.dump(model_dict, f, indent=2)
        
        logger.info(f"结果已保存到 {output_dir}")

def main():
    """主函数示例"""
    # 创建示例配置文件
    config = {
        'nwb_path': 'example_data.nwb',
        'junction_potential': -14.0,
        'stimulus_types': ['Long Square', 'Ramp'],
        'output_dir': 'point_process_results'
    }
    
    config_file = 'point_process_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 运行工作流程
    workflow = PointProcessWorkflow(config_file)
    result = workflow.run_optimization()
    
    print("优化完成!")
    print(f"最终误差: {result['error']}")
    print(f"优化成功: {result['success']}")

if __name__ == "__main__":
    main()
