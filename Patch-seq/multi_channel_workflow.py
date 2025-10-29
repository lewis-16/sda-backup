#!/usr/bin/env python3
"""
多通道点过程模型工作流程
基于All-active-Workflow-master的多通道设计，但无需形态学数据
使用单点神经元模型，但保留所有离子通道的复杂性

作者: AI Assistant
基于: All-active-Workflow-master 多通道版本
"""

import numpy as np
import json
import os
import logging
from collections import defaultdict
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import efel
import pynwb
from pynwb import NWBHDF5IO
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数值计算安全函数
def safe_exp(x, max_exp=700):
    """安全的指数函数，防止溢出"""
    x = np.clip(x, -max_exp, max_exp)
    return np.exp(x)

def safe_divide(a, b, default=1e-6):
    """安全的除法，防止除零"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.divide(a, b, out=np.full_like(a, default), where=b!=0)
    return result

def safe_sigmoid(x, scale=1.0):
    """安全的sigmoid函数"""
    x_clipped = np.clip(x, -700, 700)
    return 1.0 / (1.0 + safe_exp(-x_clipped / scale))

def safe_tau_calculation(alpha, beta, min_tau=0.01, max_tau=1000.0):
    """安全的时间常数计算"""
    tau = safe_divide(1.0, alpha + beta, default=min_tau)
    return np.clip(tau, min_tau, max_tau)

@dataclass
class MultiChannelModel:
    """多通道点过程神经元模型参数"""
    
    # 被动参数
    Cm: float = 1.0  # 膜电容 (μF/cm²)
    Ra: float = 100.0  # 轴向电阻 (Ω·cm)
    g_pas: float = 0.0001  # 被动电导 (S/cm²)
    e_pas: float = -70.0  # 被动平衡电位 (mV)
    
    # 离子平衡电位
    ena: float = 53.0  # 钠平衡电位 (mV)
    ek: float = -107.0  # 钾平衡电位 (mV)
    eca: float = 120.0  # 钙平衡电位 (mV)
    
    # 钠通道参数
    gbar_NaTs2_t: float = 0.0  # NaTs2_t通道密度
    gbar_NaTa_t: float = 0.0   # NaTa_t通道密度
    gbar_Nap_Et2: float = 0.0  # Nap_Et2通道密度
    
    # 钾通道参数
    gbar_K_Tst: float = 0.0    # K_Tst通道密度
    gbar_Kv3_1: float = 0.0    # Kv3_1通道密度
    gbar_K_Pst: float = 0.0    # K_Pst通道密度
    gbar_Kd: float = 0.0       # Kd通道密度
    gbar_K_P: float = 0.0      # K_P通道密度
    
    # 钙通道参数
    gbar_Ca_HVA: float = 0.0   # Ca_HVA通道密度
    gbar_Ca_LVA: float = 0.0   # Ca_LVA通道密度
    
    # 钙依赖性通道参数
    gbar_SK: float = 0.0       # SK通道密度
    gbar_BK_gc: float = 0.0    # BK_gc通道密度
    
    # HCN通道参数
    gbar_HCN: float = 0.0      # HCN通道密度
    gbar_Ih: float = 0.0       # Ih通道密度
    
    # 其他通道参数
    gbar_Im: float = 0.0       # Im通道密度
    gbar_Kir21_gc: float = 0.0 # Kir21_gc通道密度
    
    # 钙动力学参数
    gamma_CaDynamics: float = 0.0005  # 钙缓冲参数
    decay_CaDynamics: float = 20.0    # 钙衰减时间常数
    
    # 温度
    celsius: float = 34.0
    v_init: float = -80.0

class MultiChannelSimulator:
    """多通道神经元模拟器"""
    
    def __init__(self, model: MultiChannelModel):
        self.model = model
        self.dt = 0.1  # 时间步长 (ms)
        
    def simulate_response(self, stimulus: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """模拟神经元对刺激的响应"""
        self.dt = dt
        n_steps = len(stimulus)
        
        # 初始化状态变量
        V = np.zeros(n_steps)
        spikes = np.zeros(n_steps, dtype=bool)
        
        # 初始化膜电位
        V[0] = self.model.v_init
        
        # 初始化离子通道状态变量
        states = self._initialize_channel_states()
        
        # 初始化钙浓度
        cai = 0.0001  # 初始钙浓度 (mM)
        
        # 模拟时间步进
        for i in range(1, n_steps):
            # 更新离子通道状态
            states = self._update_channel_states(V[i-1], states, cai)
            
            # 计算各种电流
            currents = self._calculate_currents(V[i-1], states, cai)
            
            # 更新钙浓度
            cai = self._update_calcium_concentration(cai, currents['ica'])
            
            # 更新膜电位
            dV_dt = (stimulus[i-1] - sum(currents.values())) / self.model.Cm
            V[i] = V[i-1] + dt * dV_dt
            
            # 检测动作电位
            if V[i] > -20.0 and V[i-1] <= -20.0:
                spikes[i] = True
                V[i] = 20.0  # 动作电位峰值
        
        return V, spikes, states
    
    def _initialize_channel_states(self) -> Dict:
        """初始化离子通道状态变量"""
        states = {}
        
        # 钠通道状态 (m, h)
        states['NaTs2_t_m'] = 0.0
        states['NaTs2_t_h'] = 1.0
        states['NaTa_t_m'] = 0.0
        states['NaTa_t_h'] = 1.0
        states['Nap_Et2_m'] = 0.0
        
        # 钾通道状态 (n, p等)
        states['K_Tst_n'] = 0.0
        states['Kv3_1_n'] = 0.0
        states['K_Pst_n'] = 0.0
        states['Kd_n'] = 0.0
        states['Kd_h'] = 1.0
        
        # 钙通道状态
        states['Ca_HVA_m'] = 0.0
        states['Ca_HVA_h'] = 1.0
        states['Ca_LVA_m'] = 0.0
        states['Ca_LVA_h'] = 1.0
        
        # 钙依赖性通道状态
        states['SK_z'] = 0.0
        states['BK_gc_c'] = 0.0
        
        # HCN通道状态
        states['HCN_l1'] = 0.0
        states['HCN_l2'] = 0.0
        states['Ih_l'] = 0.0
        
        # 其他通道状态
        states['Im_m'] = 0.0
        states['Kir21_gc_n'] = 0.0
        
        return states
    
    def _update_channel_states(self, V: float, states: Dict, cai: float) -> Dict:
        """更新离子通道状态变量"""
        new_states = states.copy()
        
        # 钠通道动力学
        new_states.update(self._update_NaTs2_t_states(V, states))
        new_states.update(self._update_NaTa_t_states(V, states))
        new_states.update(self._update_Nap_Et2_states(V, states))
        
        # 钾通道动力学
        new_states.update(self._update_K_Tst_states(V, states))
        new_states.update(self._update_Kv3_1_states(V, states))
        new_states.update(self._update_K_Pst_states(V, states))
        new_states.update(self._update_Kd_states(V, states))
        
        # 钙通道动力学
        new_states.update(self._update_Ca_HVA_states(V, states))
        new_states.update(self._update_Ca_LVA_states(V, states))
        
        # 钙依赖性通道动力学
        new_states.update(self._update_SK_states(cai, states))
        new_states.update(self._update_BK_gc_states(V, cai, states))
        
        # HCN通道动力学
        new_states.update(self._update_HCN_states(V, states))
        new_states.update(self._update_Ih_states(V, states))
        
        # 其他通道动力学
        new_states.update(self._update_Im_states(V, states))
        new_states.update(self._update_Kir21_gc_states(V, states))
        
        return new_states
    
    def _update_NaTs2_t_states(self, V: float, states: Dict) -> Dict:
        """更新NaTs2_t通道状态"""
        # 限制电压范围
        V = np.clip(V, -200, 200)
        
        # 使用安全的sigmoid函数
        m_inf = safe_sigmoid(-(V + 38.0), 6.0)
        h_inf = safe_sigmoid(V + 66.0, 6.0)
        
        m_tau = 0.1
        h_tau = 1.0
        
        m = np.clip(states['NaTs2_t_m'], 0.0, 1.0)
        h = np.clip(states['NaTs2_t_h'], 0.0, 1.0)
        
        return {
            'NaTs2_t_m': np.clip(m + self.dt * (m_inf - m) / m_tau, 0.0, 1.0),
            'NaTs2_t_h': np.clip(h + self.dt * (h_inf - h) / h_tau, 0.0, 1.0)
        }
    
    def _update_NaTa_t_states(self, V: float, states: Dict) -> Dict:
        """更新NaTa_t通道状态"""
        V = np.clip(V, -200, 200)
        
        m_inf = safe_sigmoid(-(V + 38.0), 6.0)
        h_inf = safe_sigmoid(V + 66.0, 6.0)
        
        m_tau = 0.1
        h_tau = 1.0
        
        m = np.clip(states['NaTa_t_m'], 0.0, 1.0)
        h = np.clip(states['NaTa_t_h'], 0.0, 1.0)
        
        return {
            'NaTa_t_m': np.clip(m + self.dt * (m_inf - m) / m_tau, 0.0, 1.0),
            'NaTa_t_h': np.clip(h + self.dt * (h_inf - h) / h_tau, 0.0, 1.0)
        }
    
    def _update_Nap_Et2_states(self, V: float, states: Dict) -> Dict:
        """更新Nap_Et2通道状态"""
        m_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 5.0))
        m_tau = 0.5
        
        m = states['Nap_Et2_m']
        
        return {
            'Nap_Et2_m': m + self.dt * (m_inf - m) / m_tau
        }
    
    def _update_K_Tst_states(self, V: float, states: Dict) -> Dict:
        """更新K_Tst通道状态"""
        n_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 10.0))
        n_tau = 1.0
        
        n = states['K_Tst_n']
        
        return {
            'K_Tst_n': n + self.dt * (n_inf - n) / n_tau
        }
    
    def _update_Kv3_1_states(self, V: float, states: Dict) -> Dict:
        """更新Kv3_1通道状态"""
        n_inf = 1.0 / (1.0 + np.exp(-(V + 30.0) / 10.0))
        n_tau = 0.5
        
        n = states['Kv3_1_n']
        
        return {
            'Kv3_1_n': n + self.dt * (n_inf - n) / n_tau
        }
    
    def _update_K_Pst_states(self, V: float, states: Dict) -> Dict:
        """更新K_Pst通道状态"""
        n_inf = 1.0 / (1.0 + np.exp(-(V + 40.0) / 8.0))
        n_tau = 1.0
        
        n = states['K_Pst_n']
        
        return {
            'K_Pst_n': n + self.dt * (n_inf - n) / n_tau
        }
    
    def _update_Kd_states(self, V: float, states: Dict) -> Dict:
        """更新Kd通道状态"""
        n_inf = 1.0 / (1.0 + np.exp(-(V + 43.0) / 8.0))
        n_tau = 1.0
        h_inf = 1.0 / (1.0 + np.exp((V + 67.0) / 7.3))
        h_tau = 1500.0
        
        n = states['Kd_n']
        h = states['Kd_h']
        
        return {
            'Kd_n': n + self.dt * (n_inf - n) / n_tau,
            'Kd_h': h + self.dt * (h_inf - h) / h_tau
        }
    
    def _update_Ca_HVA_states(self, V: float, states: Dict) -> Dict:
        """更新Ca_HVA通道状态"""
        V = np.clip(V, -200, 200)
        
        # 使用安全的指数函数
        exp_term1 = safe_exp(-(V + 27.0) / 3.8)
        exp_term2 = safe_exp(-(V + 75.0) / 17.0)
        exp_term3 = safe_exp(-(V + 13.0) / 50.0)
        exp_term4 = safe_exp(-(V + 15.0) / 28.0)
        
        # 计算alpha和beta
        m_alpha = 0.055 * exp_term1
        m_beta = 0.94 * exp_term2
        h_alpha = 0.000457 * exp_term3
        h_beta = 0.0065 / (1.0 + exp_term4)
        
        # 计算稳态值和时间常数
        m_inf = safe_divide(m_alpha, m_alpha + m_beta, default=0.5)
        h_inf = safe_divide(h_alpha, h_alpha + h_beta, default=0.5)
        
        m_tau = safe_tau_calculation(m_alpha, m_beta)
        h_tau = safe_tau_calculation(h_alpha, h_beta)
        
        m = np.clip(states['Ca_HVA_m'], 0.0, 1.0)
        h = np.clip(states['Ca_HVA_h'], 0.0, 1.0)
        
        return {
            'Ca_HVA_m': np.clip(m + self.dt * (m_inf - m) / m_tau, 0.0, 1.0),
            'Ca_HVA_h': np.clip(h + self.dt * (h_inf - h) / h_tau, 0.0, 1.0)
        }
    
    def _update_Ca_LVA_states(self, V: float, states: Dict) -> Dict:
        """更新Ca_LVA通道状态"""
        m_inf = 1.0 / (1.0 + np.exp(-(V + 40.0) / 5.0))
        m_tau = 1.0
        h_inf = 1.0 / (1.0 + np.exp((V + 60.0) / 5.0))
        h_tau = 10.0
        
        m = states['Ca_LVA_m']
        h = states['Ca_LVA_h']
        
        return {
            'Ca_LVA_m': m + self.dt * (m_inf - m) / m_tau,
            'Ca_LVA_h': h + self.dt * (h_inf - h) / h_tau
        }
    
    def _update_SK_states(self, cai: float, states: Dict) -> Dict:
        """更新SK通道状态"""
        z_inf = 1.0 / (1.0 + (0.00043 / max(cai, 1e-7)) ** 4.8)
        z_tau = 1.0
        
        z = states['SK_z']
        
        return {
            'SK_z': z + self.dt * (z_inf - z) / z_tau
        }
    
    def _update_BK_gc_states(self, V: float, cai: float, states: Dict) -> Dict:
        """更新BK_gc通道状态"""
        c_inf = 1.0 / (1.0 + np.exp(-(V + 20.0) / 10.0))
        c_tau = 1.0
        
        c = states['BK_gc_c']
        
        return {
            'BK_gc_c': c + self.dt * (c_inf - c) / c_tau
        }
    
    def _update_HCN_states(self, V: float, states: Dict) -> Dict:
        """更新HCN通道状态"""
        V = np.clip(V, -200, 200)
        
        # 计算稳态值
        l_inf = safe_sigmoid(V + 75.3, 8.0)
        
        # 安全的时间常数计算
        exp_term1 = safe_exp(-V / 30.4)
        exp_term2 = safe_exp(V / 30.4)
        
        l_tau = safe_divide(1.0, 0.00052 * exp_term1 + 0.2151 * exp_term2, default=100.0)
        l_tau = np.clip(l_tau, 1.0, 10000.0)
        
        l1 = np.clip(states['HCN_l1'], 0.0, 1.0)
        l2 = np.clip(states['HCN_l2'], 0.0, 1.0)
        
        return {
            'HCN_l1': np.clip(l1 + self.dt * (l_inf - l1) / l_tau, 0.0, 1.0),
            'HCN_l2': np.clip(l2 + self.dt * (l_inf - l2) / (l_tau * 6.4), 0.0, 1.0)
        }
    
    def _update_Ih_states(self, V: float, states: Dict) -> Dict:
        """更新Ih通道状态"""
        l_inf = 1.0 / (1.0 + np.exp((V + 75.0) / 8.0))
        l_tau = 100.0
        
        l = states['Ih_l']
        
        return {
            'Ih_l': l + self.dt * (l_inf - l) / l_tau
        }
    
    def _update_Im_states(self, V: float, states: Dict) -> Dict:
        """更新Im通道状态"""
        m_inf = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
        m_tau = 100.0
        
        m = states['Im_m']
        
        return {
            'Im_m': m + self.dt * (m_inf - m) / m_tau
        }
    
    def _update_Kir21_gc_states(self, V: float, states: Dict) -> Dict:
        """更新Kir21_gc通道状态"""
        n_inf = 1.0 / (1.0 + np.exp((V + 80.0) / 10.0))
        n_tau = 1.0
        
        n = states['Kir21_gc_n']
        
        return {
            'Kir21_gc_n': n + self.dt * (n_inf - n) / n_tau
        }
    
    def _calculate_currents(self, V: float, states: Dict, cai: float) -> Dict:
        """计算各种离子电流"""
        # 限制电压范围
        V = np.clip(V, -200, 200)
        
        currents = {}
        
        # 确保所有状态变量都是有效的
        for key, value in states.items():
            if np.isnan(value) or np.isinf(value):
                states[key] = 0.0
        
        # 被动电流
        currents['i_pas'] = self.model.g_pas * (V - self.model.e_pas)
        
        # 钠电流 - 使用安全的状态变量
        m_NaTs2_t = np.clip(states['NaTs2_t_m'], 0.0, 1.0)
        h_NaTs2_t = np.clip(states['NaTs2_t_h'], 0.0, 1.0)
        m_NaTa_t = np.clip(states['NaTa_t_m'], 0.0, 1.0)
        h_NaTa_t = np.clip(states['NaTa_t_h'], 0.0, 1.0)
        m_Nap_Et2 = np.clip(states['Nap_Et2_m'], 0.0, 1.0)
        
        currents['i_NaTs2_t'] = self.model.gbar_NaTs2_t * m_NaTs2_t**3 * h_NaTs2_t * (V - self.model.ena)
        currents['i_NaTa_t'] = self.model.gbar_NaTa_t * m_NaTa_t**3 * h_NaTa_t * (V - self.model.ena)
        currents['i_Nap_Et2'] = self.model.gbar_Nap_Et2 * m_Nap_Et2 * (V - self.model.ena)
        
        # 钾电流
        n_K_Tst = np.clip(states['K_Tst_n'], 0.0, 1.0)
        n_Kv3_1 = np.clip(states['Kv3_1_n'], 0.0, 1.0)
        n_K_Pst = np.clip(states['K_Pst_n'], 0.0, 1.0)
        n_Kd = np.clip(states['Kd_n'], 0.0, 1.0)
        h_Kd = np.clip(states['Kd_h'], 0.0, 1.0)
        
        currents['i_K_Tst'] = self.model.gbar_K_Tst * n_K_Tst**4 * (V - self.model.ek)
        currents['i_Kv3_1'] = self.model.gbar_Kv3_1 * n_Kv3_1**4 * (V - self.model.ek)
        currents['i_K_Pst'] = self.model.gbar_K_Pst * n_K_Pst * (V - self.model.ek)
        currents['i_Kd'] = self.model.gbar_Kd * n_Kd * h_Kd * (V - self.model.ek)
        
        # 钙电流
        m_Ca_HVA = np.clip(states['Ca_HVA_m'], 0.0, 1.0)
        h_Ca_HVA = np.clip(states['Ca_HVA_h'], 0.0, 1.0)
        m_Ca_LVA = np.clip(states['Ca_LVA_m'], 0.0, 1.0)
        h_Ca_LVA = np.clip(states['Ca_LVA_h'], 0.0, 1.0)
        
        currents['i_Ca_HVA'] = self.model.gbar_Ca_HVA * m_Ca_HVA**2 * h_Ca_HVA * (V - self.model.eca)
        currents['i_Ca_LVA'] = self.model.gbar_Ca_LVA * m_Ca_LVA**2 * h_Ca_LVA * (V - self.model.eca)
        
        # 钙依赖性钾电流
        z_SK = np.clip(states['SK_z'], 0.0, 1.0)
        c_BK_gc = np.clip(states['BK_gc_c'], 0.0, 1.0)
        
        currents['i_SK'] = self.model.gbar_SK * z_SK * (V - self.model.ek)
        currents['i_BK_gc'] = self.model.gbar_BK_gc * c_BK_gc * (V - self.model.ek)
        
        # HCN电流
        l1_HCN = np.clip(states['HCN_l1'], 0.0, 1.0)
        l2_HCN = np.clip(states['HCN_l2'], 0.0, 1.0)
        l_Ih = np.clip(states['Ih_l'], 0.0, 1.0)
        
        currents['i_HCN'] = self.model.gbar_HCN * (0.8 * l1_HCN + 0.2 * l2_HCN) * (V - (-41.9))
        currents['i_Ih'] = self.model.gbar_Ih * l_Ih * (V - (-41.9))
        
        # 其他电流
        m_Im = np.clip(states['Im_m'], 0.0, 1.0)
        n_Kir21_gc = np.clip(states['Kir21_gc_n'], 0.0, 1.0)
        
        currents['i_Im'] = self.model.gbar_Im * m_Im * (V - self.model.ek)
        currents['i_Kir21_gc'] = self.model.gbar_Kir21_gc * n_Kir21_gc * (V - self.model.ek)
        
        # 总钙电流
        currents['ica'] = currents['i_Ca_HVA'] + currents['i_Ca_LVA']
        
        return currents
    
    def _update_calcium_concentration(self, cai: float, ica: float) -> float:
        """更新钙浓度"""
        # 简化的钙动力学
        dca_dt = -self.model.gamma_CaDynamics * ica - (cai - 0.0001) / self.model.decay_CaDynamics
        return cai + self.dt * dca_dt
    
    def extract_model_features(self, stimulus: np.ndarray, dt: float = 0.1) -> Dict:
        """从模拟响应中提取特征"""
        V, spikes, states = self.simulate_response(stimulus, dt)
        
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

class MultiChannelOptimizer:
    """多通道模型参数优化器"""
    
    def __init__(self, model: MultiChannelModel, target_features: Dict):
        self.model = model
        self.target_features = target_features
        self.simulator = MultiChannelSimulator(model)
        
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
        # 参数顺序：[Cm, Ra, g_pas, e_pas, gbar_NaTs2_t, gbar_NaTa_t, gbar_Nap_Et2, 
        #           gbar_K_Tst, gbar_Kv3_1, gbar_K_Pst, gbar_Kd, gbar_Ca_HVA, gbar_Ca_LVA,
        #           gbar_SK, gbar_BK_gc, gbar_HCN, gbar_Ih, gbar_Im, gbar_Kir21_gc]
        
        self.model.Cm = params[0]
        self.model.Ra = params[1]
        self.model.g_pas = params[2]
        self.model.e_pas = params[3]
        self.model.gbar_NaTs2_t = params[4]
        self.model.gbar_NaTa_t = params[5]
        self.model.gbar_Nap_Et2 = params[6]
        self.model.gbar_K_Tst = params[7]
        self.model.gbar_Kv3_1 = params[8]
        self.model.gbar_K_Pst = params[9]
        self.model.gbar_Kd = params[10]
        self.model.gbar_Ca_HVA = params[11]
        self.model.gbar_Ca_LVA = params[12]
        self.model.gbar_SK = params[13]
        self.model.gbar_BK_gc = params[14]
        self.model.gbar_HCN = params[15]
        self.model.gbar_Ih = params[16]
        self.model.gbar_Im = params[17]
        self.model.gbar_Kir21_gc = params[18]
    
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
                self.model.Ra,
                self.model.g_pas,
                self.model.e_pas,
                self.model.gbar_NaTs2_t,
                self.model.gbar_NaTa_t,
                self.model.gbar_Nap_Et2,
                self.model.gbar_K_Tst,
                self.model.gbar_Kv3_1,
                self.model.gbar_K_Pst,
                self.model.gbar_Kd,
                self.model.gbar_Ca_HVA,
                self.model.gbar_Ca_LVA,
                self.model.gbar_SK,
                self.model.gbar_BK_gc,
                self.model.gbar_HCN,
                self.model.gbar_Ih,
                self.model.gbar_Im,
                self.model.gbar_Kir21_gc
            ])
        
        # 参数边界
        bounds = [
            (0.1, 10.0),    # Cm
            (10.0, 1000.0), # Ra
            (1e-7, 0.01),   # g_pas
            (-110, -60),    # e_pas
            (0, 5),         # gbar_NaTs2_t
            (0, 10),        # gbar_NaTa_t
            (0, 5),         # gbar_Nap_Et2
            (0, 1),         # gbar_K_Tst
            (0, 2),         # gbar_Kv3_1
            (0, 1),         # gbar_K_Pst
            (0, 1),         # gbar_Kd
            (1e-7, 0.001),  # gbar_Ca_HVA
            (1e-7, 0.01),   # gbar_Ca_LVA
            (1e-7, 0.1),    # gbar_SK
            (1e-7, 0.1),    # gbar_BK_gc
            (1e-7, 0.0001), # gbar_HCN
            (1e-7, 0.0001), # gbar_Ih
            (1e-7, 0.01),   # gbar_Im
            (1e-7, 0.01)    # gbar_Kir21_gc
        ]
        
        logger.info("开始多通道参数优化...")
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

def main():
    """主函数示例"""
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

if __name__ == "__main__":
    main()
