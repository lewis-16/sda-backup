#!/usr/bin/env python3
"""
点过程模型特征提取工具
从NWB文件中提取电生理特征，用于点过程模型拟合

基于eFEL库进行特征提取
"""

import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import efel
from nwb_reader import GenericNWBReader

logger = logging.getLogger(__name__)

class PointProcessFeatureExtractor:
    """点过程模型特征提取器"""
    
    def __init__(self, nwb_path: str, junction_potential: float = -14.0):
        self.nwb_path = nwb_path
        self.junction_potential = junction_potential
        self.nwb_file = None
        
    def extract_step_current_features(self) -> Dict:
        """提取台阶电流特征（参考neuron_model.ipynb的方法）"""
        features = {}
        
        with GenericNWBReader(self.nwb_path) as reader:
            # 获取所有stimulus数据
            acquisition_data = {}
            count = 0
            
            # 遍历所有stimulus
            for stimulus_key in reader.nwb_file.stimulus.keys():
                try:
                    stimulus = reader.nwb_file.get_stimulus(stimulus_key)
                    
                    # 只处理CurrentClampStimulusSeries
                    if stimulus.data_type == 'CurrentClampStimulusSeries':
                        # 检查数据长度（台阶电流通常是特定长度）
                        if len(stimulus.data) in [301000, 201000, 401000]:
                            # 获取对应的acquisition数据
                            acquisition_key = stimulus_key.split("DA")[0] + "AD0"
                            if acquisition_key in reader.nwb_file.acquisition:
                                acquisition = reader.nwb_file.get_acquisition(acquisition_key)
                                
                                # 提取刺激幅度（从60000:70000区间的最大值）
                                stimulus_data = str(int(np.array(stimulus.data)[60000:70000].max()))
                                
                                # 存储acquisition数据（40000:125000区间）
                                if stimulus_data not in acquisition_data.keys():
                                    acquisition_data[f'{stimulus_key}_{stimulus_data}'] = np.array(acquisition.data)[40000:125000]
                                else:
                                    acquisition_data[f'{stimulus_key}_{stimulus_data}_{count}'] = np.array(acquisition.data)[40000:125000]
                                    count += 1
                                    
                except Exception as e:
                    logger.debug(f"处理stimulus {stimulus_key} 时出错: {e}")
                    continue
            
            logger.info(f"找到 {len(acquisition_data)} 个台阶电流sweep")
            
            # 按幅度分组
            amplitude_groups = {}
            for key, data in acquisition_data.items():
                amplitude = int(key.split("_")[-1])
                if amplitude not in amplitude_groups:
                    amplitude_groups[amplitude] = []
                amplitude_groups[amplitude].append(data)
            
            # 为每个幅度组提取特征
            for amplitude, data_list in amplitude_groups.items():
                logger.info(f"处理幅度 {amplitude} pA，共 {len(data_list)} 个sweep")
                
                all_features = []
                for data in data_list:
                    # 创建trace格式
                    trace = {
                        'V': data,
                        'T': np.arange(len(data)) / 50.0,  # 50kHz采样率
                        'stim_start': [10000 / 50],  # 200ms开始
                        'stim_end': [60000 / 50]     # 1200ms结束
                    }
                    
                    # 提取特征，传递幅度信息
                    trace_features = self._extract_efel_features(trace, amplitude)
                    if trace_features:
                        all_features.append(trace_features)
                
                if all_features:
                    # 计算统计值
                    features[f'Amplitude_{amplitude}'] = self._calculate_feature_statistics(all_features)
        
        return features
    
    def extract_all_features(self, stimulus_types: List[str] = None) -> Dict:
        """提取所有刺激类型的特征"""
        features = {}
        
        # 如果指定了stimulus_types，使用传统方法
        if stimulus_types:
            with GenericNWBReader(self.nwb_path) as reader:
                for stim_type in stimulus_types:
                    logger.info(f"提取 {stim_type} 特征...")
                    stim_features = self._extract_stimulus_features(stim_type, reader)
                    if stim_features:
                        features[stim_type] = stim_features
        else:
            # 否则使用台阶电流方法
            logger.info("使用台阶电流特征提取方法...")
            features = self.extract_step_current_features()
                    
        return features
    
    def _extract_stimulus_features(self, stim_type: str, reader: GenericNWBReader) -> Dict:
        """提取特定刺激类型的特征"""
        sweeps = reader.get_sweeps_by_stimulus_type(stim_type)
        
        if not sweeps:
            logger.warning(f"未找到 {stim_type} 类型的sweep")
            return {}
        
        all_features = []
        
        for sweep_num in sweeps:
            try:
                features = self._extract_sweep_features(sweep_num, reader)
                if features:
                    all_features.append(features)
            except Exception as e:
                logger.error(f"提取sweep {sweep_num} 特征时出错: {e}")
                continue
        
        if not all_features:
            return {}
        
        # 计算特征的平均值和标准差
        return self._calculate_feature_statistics(all_features)
    
    def _extract_sweep_features(self, sweep_num: int, reader: GenericNWBReader) -> Dict:
        """从单个sweep提取特征"""
        try:
            # 获取sweep数据
            sweep_data = reader.get_sweep_data(sweep_num)
            
            if not sweep_data:
                logger.warning(f"未找到sweep {sweep_num} 的数据")
                return {}
            
            # 获取电压和刺激数据
            voltage = sweep_data.get('response')
            stimulus = sweep_data.get('stimulus')
            sampling_rate = sweep_data.get('sampling_rate', 20000.0)
            
            if voltage is None:
                logger.warning(f"sweep {sweep_num} 没有电压数据")
                return {}
            
            # 时间轴
            time = np.arange(len(voltage)) / sampling_rate
            
            # 校正连接电位
            voltage_corrected = voltage + self.junction_potential
            
            # 转换为mV
            voltage_corrected *= 1000
            
            # 计算刺激参数
            stim_start, stim_end = self._calculate_stimulus_timing(stimulus, time)
            
            # 准备eFEL输入
            trace = {
                'T': time,
                'V': voltage_corrected,
                'stim_start': [stim_start],
                'stim_end': [stim_end]
            }
            
            # 提取特征
            features = self._extract_efel_features(trace)
            
            return features
            
        except Exception as e:
            logger.error(f"提取sweep {sweep_num} 特征时出错: {e}")
            return {}
    
    def _calculate_stimulus_timing(self, stimulus: np.ndarray, time: np.ndarray) -> Tuple[float, float]:
        """计算刺激开始和结束时间"""
        if stimulus is None:
            # 无刺激数据，使用默认时间
            return time[20000] if len(time) > 20000 else time[0], time[-1]
        
        # 找到非零刺激的索引
        nonzero_indices = np.where(stimulus != 0)[0]
        
        if len(nonzero_indices) == 0:
            # 无刺激
            return time[20000] if len(time) > 20000 else time[0], time[-1]
        
        stim_start_idx = nonzero_indices[0]
        stim_end_idx = nonzero_indices[-1]
        
        return time[stim_start_idx], time[stim_end_idx]
    
    def _extract_efel_features(self, trace: Dict, amplitude: int = None) -> Dict:
        """使用eFEL提取特征"""
        # 基础特征（所有刺激都有）
        basic_features = [
            'voltage_base',
            'steady_state_voltage',
            'voltage_after_stim',
            'voltage_deflection_vb_ssse',
            'decay_time_constant_after_stim',
            'Spikecount'
        ]
        
        # 主动特征（有动作电位时）
        active_features = [
            'mean_frequency',
            'time_to_first_spike',
            'AP_amplitude_from_voltagebase',
            'ISI_CV',
            'AP_width',
            'adaptation_index2',
            'AHP_depth'
        ]
        
        # 凹陷特征（仅负电流/超极化刺激）
        sag_features = [
            'sag_amplitude',
            'sag_ratio1'
        ]
        
        # 根据刺激幅度选择特征
        if amplitude is not None and amplitude < 0:
            # 负电流：包含所有特征
            feature_names = basic_features + active_features + sag_features
        else:
            # 正电流：不包含凹陷特征
            feature_names = basic_features + active_features
        
        try:
            feature_values = efel.getFeatureValues([trace], feature_names)
            
            if not feature_values or not feature_values[0]:
                return {}
            
            features = {}
            for feature_name in feature_names:
                if feature_name in feature_values[0] and feature_values[0][feature_name] is not None:
                    value = feature_values[0][feature_name]
                    if isinstance(value, np.ndarray) and len(value) > 0:
                        features[feature_name] = float(np.mean(value))
                    elif isinstance(value, (int, float)):
                        features[feature_name] = float(value)
            
            return features
            
        except Exception as e:
            logger.error(f"eFEL特征提取出错: {e}")
            return {}
    
    def _calculate_feature_statistics(self, all_features: List[Dict]) -> Dict:
        """计算特征统计量"""
        if not all_features:
            return {}
        
        # 收集所有特征值
        feature_values = defaultdict(list)
        
        for features in all_features:
            for feature_name, value in features.items():
                if not np.isnan(value) and not np.isinf(value):
                    feature_values[feature_name].append(value)
        
        # 计算统计量
        statistics = {}
        
        for feature_name, values in feature_values.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # 如果只有一个值，设置一个小的标准差
                if len(values) == 1:
                    std_val = 0.05 * abs(mean_val) if mean_val != 0 else 0.05
                
                statistics[feature_name] = [mean_val, std_val]
        
        return statistics
    
    def save_features(self, features: Dict, output_path: str):
        """保存特征到JSON文件"""
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=2)
        
        logger.info(f"特征已保存到 {output_path}")

def main():
    """示例用法"""
    # 配置
    nwb_path = "example_data.nwb"  # 替换为实际的NWB文件路径
    output_path = "extracted_features.json"
    stimulus_types = ["Long Square", "Ramp", "Noise 1", "Noise 2"]
    
    # 创建提取器
    extractor = PointProcessFeatureExtractor(nwb_path)
    
    # 提取特征
    features = extractor.extract_all_features(stimulus_types)
    
    # 保存特征
    extractor.save_features(features, output_path)
    
    print("特征提取完成!")
    print(f"提取的刺激类型: {list(features.keys())}")

if __name__ == "__main__":
    main()
