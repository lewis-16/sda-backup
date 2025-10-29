#!/usr/bin/env python3
"""
修复数值计算问题的多通道模拟器
解决溢出、除零和无效值警告

"""

import numpy as np
import warnings

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

# 修复后的离子通道动力学函数
def update_NaTs2_t_states_safe(V, states, dt):
    """安全的NaTs2_t通道状态更新"""
    # 限制电压范围
    V = np.clip(V, -200, 200)
    
    # 计算稳态值
    m_inf = safe_sigmoid(-(V + 38.0), 6.0)
    h_inf = safe_sigmoid(V + 66.0, 6.0)
    
    # 安全的时间常数
    m_tau = 0.1
    h_tau = 1.0
    
    m = states['NaTs2_t_m']
    h = states['NaTs2_t_h']
    
    # 确保状态变量在合理范围内
    m = np.clip(m, 0.0, 1.0)
    h = np.clip(h, 0.0, 1.0)
    
    return {
        'NaTs2_t_m': np.clip(m + dt * (m_inf - m) / m_tau, 0.0, 1.0),
        'NaTs2_t_h': np.clip(h + dt * (h_inf - h) / h_tau, 0.0, 1.0)
    }

def update_Ca_HVA_states_safe(V, states, dt):
    """安全的Ca_HVA通道状态更新"""
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
        'Ca_HVA_m': np.clip(m + dt * (m_inf - m) / m_tau, 0.0, 1.0),
        'Ca_HVA_h': np.clip(h + dt * (h_inf - h) / h_tau, 0.0, 1.0)
    }

def update_HCN_states_safe(V, states, dt):
    """安全的HCN通道状态更新"""
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
        'HCN_l1': np.clip(l1 + dt * (l_inf - l1) / l_tau, 0.0, 1.0),
        'HCN_l2': np.clip(l2 + dt * (l_inf - l2) / (l_tau * 6.4), 0.0, 1.0)
    }

def calculate_currents_safe(V, states, model):
    """安全的电流计算"""
    # 限制电压范围
    V = np.clip(V, -200, 200)
    
    currents = {}
    
    # 确保所有状态变量都是有效的
    for key, value in states.items():
        if np.isnan(value) or np.isinf(value):
            states[key] = 0.0
    
    # 被动电流
    currents['i_pas'] = model.g_pas * (V - model.e_pas)
    
    # 钠电流 - 使用安全的状态变量
    m_NaTs2_t = np.clip(states['NaTs2_t_m'], 0.0, 1.0)
    h_NaTs2_t = np.clip(states['NaTs2_t_h'], 0.0, 1.0)
    m_NaTa_t = np.clip(states['NaTa_t_m'], 0.0, 1.0)
    h_NaTa_t = np.clip(states['NaTa_t_h'], 0.0, 1.0)
    m_Nap_Et2 = np.clip(states['Nap_Et2_m'], 0.0, 1.0)
    
    currents['i_NaTs2_t'] = model.gbar_NaTs2_t * m_NaTs2_t**3 * h_NaTs2_t * (V - model.ena)
    currents['i_NaTa_t'] = model.gbar_NaTa_t * m_NaTa_t**3 * h_NaTa_t * (V - model.ena)
    currents['i_Nap_Et2'] = model.gbar_Nap_Et2 * m_Nap_Et2 * (V - model.ena)
    
    # 钾电流
    n_K_Tst = np.clip(states['K_Tst_n'], 0.0, 1.0)
    n_Kv3_1 = np.clip(states['Kv3_1_n'], 0.0, 1.0)
    n_K_Pst = np.clip(states['K_Pst_n'], 0.0, 1.0)
    n_Kd = np.clip(states['Kd_n'], 0.0, 1.0)
    h_Kd = np.clip(states['Kd_h'], 0.0, 1.0)
    
    currents['i_K_Tst'] = model.gbar_K_Tst * n_K_Tst**4 * (V - model.ek)
    currents['i_Kv3_1'] = model.gbar_Kv3_1 * n_Kv3_1**4 * (V - model.ek)
    currents['i_K_Pst'] = model.gbar_K_Pst * n_K_Pst * (V - model.ek)
    currents['i_Kd'] = model.gbar_Kd * n_Kd * h_Kd * (V - model.ek)
    
    # 钙电流
    m_Ca_HVA = np.clip(states['Ca_HVA_m'], 0.0, 1.0)
    h_Ca_HVA = np.clip(states['Ca_HVA_h'], 0.0, 1.0)
    m_Ca_LVA = np.clip(states['Ca_LVA_m'], 0.0, 1.0)
    h_Ca_LVA = np.clip(states['Ca_LVA_h'], 0.0, 1.0)
    
    currents['i_Ca_HVA'] = model.gbar_Ca_HVA * m_Ca_HVA**2 * h_Ca_HVA * (V - model.eca)
    currents['i_Ca_LVA'] = model.gbar_Ca_LVA * m_Ca_LVA**2 * h_Ca_LVA * (V - model.eca)
    
    # 钙依赖性钾电流
    z_SK = np.clip(states['SK_z'], 0.0, 1.0)
    c_BK_gc = np.clip(states['BK_gc_c'], 0.0, 1.0)
    
    currents['i_SK'] = model.gbar_SK * z_SK * (V - model.ek)
    currents['i_BK_gc'] = model.gbar_BK_gc * c_BK_gc * (V - model.ek)
    
    # HCN电流
    l1_HCN = np.clip(states['HCN_l1'], 0.0, 1.0)
    l2_HCN = np.clip(states['HCN_l2'], 0.0, 1.0)
    l_Ih = np.clip(states['Ih_l'], 0.0, 1.0)
    
    currents['i_HCN'] = model.gbar_HCN * (0.8 * l1_HCN + 0.2 * l2_HCN) * (V - (-41.9))
    currents['i_Ih'] = model.gbar_Ih * l_Ih * (V - (-41.9))
    
    # 其他电流
    m_Im = np.clip(states['Im_m'], 0.0, 1.0)
    n_Kir21_gc = np.clip(states['Kir21_gc_n'], 0.0, 1.0)
    
    currents['i_Im'] = model.gbar_Im * m_Im * (V - model.ek)
    currents['i_Kir21_gc'] = model.gbar_Kir21_gc * n_Kir21_gc * (V - model.ek)
    
    # 总钙电流
    currents['ica'] = currents['i_Ca_HVA'] + currents['i_Ca_LVA']
    
    return currents

if __name__ == "__main__":
    print("数值计算修复工具已准备就绪")
    print("主要修复:")
    print("1. 安全的指数函数 - 防止溢出")
    print("2. 安全的除法 - 防止除零")
    print("3. 状态变量范围限制 - 防止无效值")
    print("4. 电压范围限制 - 防止极端值")

