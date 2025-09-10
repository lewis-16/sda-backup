# 导入必要的库
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm
import pynwb

import bluepyopt as bpop
import bluepyopt.ephys as ephys
from neuron import h
import collections
import json
import efel

# 添加Brain 2库支持
import brainpy as bp
import brainpy.math as bm

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 设置随机种子，确保结果可重复
np.random.seed(42)
bm.random.seed(42)

# 加载数据
def load_patch_data(file_path):
    """加载patch-clamp实验数据"""
    try:
        if file_path.endswith('.nwb') or file_path.endswith('.nwb.h5'):
            io = pynwb.NWBHDF5IO(file_path, 'r')
            data = io.read()
            print(f"成功加载NWB数据文件: {file_path}")
            return data
        elif file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            print(f"成功加载CSV数据文件: {file_path}")
            return data
        elif file_path.endswith('.npy'):
            data = np.load(file_path)
            print(f"成功加载NPY数据文件: {file_path}")
            return data
        else:
            print(f"不支持的文件格式: {file_path}")
            return None
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

# 处理数据以提取记录和刺激
def process_patch_data(data):
    """处理patch-clamp数据，提取记录和刺激信息"""
    acquisition_data = {}
    count = 0
    
    # 处理NWB格式数据
    if hasattr(data, 'stimulus'):
        for i in data.stimulus.keys():
            try:
                stimulus = data.get_stimulus(i)
                if stimulus.data_type == 'CurrentClampStimulusSeries':
                    if len(stimulus.data) in [301000, 201000, 401000]:
                        acquisition_key = f"{i.split('_')[0]}_{i.split('_')[1]}_AD{i.split('_')[2].split('A')[1]}"
                        if acquisition_key in data.acquisition:
                            acquisition = data.get_acquisition(acquisition_key)
                            stimulus_data = str(int(np.array(stimulus.data)[60000:70000].max()))
                            
                            if stimulus_data not in acquisition_data.keys():
                                acquisition_data[f'{i}_{stimulus_data}'] = np.array(acquisition.data)[40000:125000]
                            else:
                                acquisition_data[f'{i}_{stimulus_data}_{count}'] = np.array(acquisition.data)[40000:125000]
                                count += 1
            except Exception as e:
                print(f"处理刺激 {i} 时出错: {e}")
                continue
    # 处理CSV格式数据（假设已经包含了记录的电压和时间）
    elif isinstance(data, pd.DataFrame):
        # 简单示例，具体格式可能需要根据实际数据调整
        if 'time' in data.columns and 'voltage' in data.columns:
            acquisition_data['trace_1'] = data['voltage'].values
            # 假设时间步长为0.02ms (50kHz采样率)
            time = data['time'].values
    # 处理NPY格式数据
    elif isinstance(data, np.ndarray):
        # 假设是2D数组，第一维是时间，第二维是记录
        if data.ndim == 2:
            for i in range(data.shape[1]):
                acquisition_data[f'trace_{i+1}'] = data[:, i]
        else:
            acquisition_data['trace_1'] = data
    
    return acquisition_data

# 生成协议
def generate_protocol(acquisition_data):
    """根据记录数据生成刺激协议"""
    protocol = {}
    
    for amplitude in acquisition_data.keys():
        try:
            # 尝试从键名中提取刺激幅度
            if '_' in amplitude:
                parts = amplitude.split('_')
                if len(parts) >= 4 and parts[3].isdigit():
                    amp_value = int(parts[3])
                else:
                    amp_value = 100  # 默认值
            else:
                amp_value = 100  # 默认值
                
            protocol[f'Amplitude_{int(amp_value)}'] = {}
            protocol[f'Amplitude_{int(amp_value)}']['stimuli'] = [
                {
                    'amp': amp_value / 1000,
                    'amp_end': amp_value / 1000,
                    'delay': 200,
                    'duration': 1000,
                    'stim_end': 1500,
                    'totduration': 3000,
                    'type': 'SquarePulse'
                }
            ]
        except Exception as e:
            print(f"生成协议时出错: {e}")
            continue
    
    return protocol

# 准备用于EFEL分析的迹数据
def prepare_traces_for_efel(acquisition_data, sample_rate=50):
    """准备用于EFEL分析的迹数据"""
    traces = []
    for key in acquisition_data.keys():
        trace1 = {}
        trace1['V'] = acquisition_data[key]
        trace1['T'] = np.array(range(1, len(acquisition_data[key]) + 1)) / sample_rate
        trace1['stim_start'] = [10000 / sample_rate]
        trace1['stim_end'] = [60000 / sample_rate]
        traces.append(trace1)
    
    return traces

# 使用Brain 2实现的AdEx神经元模型
class Brain2AdExNeuron(bp.NeuGroup):
    """使用Brain 2实现的AdEx神经元模型"""
    def __init__(self, size, **kwargs):
        super(Brain2AdExNeuron, self).__init__(size, **kwargs)
        
        # 初始化参数
        self.a = bm.Variable(4. * bm.ones(size))          # [nS] 适应耦合常数
        self.b = bm.Variable(0.0805 * bm.ones(size))       # [nA] 动作电位触发的适应
        self.tau_w = bm.Variable(144. * bm.ones(size))     # [ms] 适应变量w的时间常数
        self.Delta_T = bm.Variable(2. * bm.ones(size))     # [mV] 指数参数
        self.el = bm.Variable(-70.6 * bm.ones(size))       # [mV] 静息电位
        self.VT_rest = bm.Variable(-50.4 * bm.ones(size))  # [mV] 重置电压
        self.g_L = bm.Variable(30. * bm.ones(size))        # [nS] 膜电导
        self.C = bm.Variable(281. * bm.ones(size))         # [pF] 膜电容
        self.th = bm.Variable(20. * bm.ones(size))         # [mV] 动作电位阈值
        self.w_jump = bm.Variable(400. * bm.ones(size))    # [pA] 动作电位后去极化电流
        self.tau_wtail = bm.Variable(40. * bm.ones(size))  # [ms] 动作电位后去极化时间常数
        self.tau_VT = bm.Variable(50. * bm.ones(size))     # [ms] VT的时间常数
        self.VT_jump = bm.Variable(20. * bm.ones(size))    # 自适应阈值
        
        # 初始化状态变量
        self.V = bm.Variable(self.el + 5. * bm.random.random(size))  # 膜电位
        self.w = bm.Variable(bm.zeros(size))                          # 适应变量
        self.VT = bm.Variable(self.VT_rest)                          # 动态阈值
        
        # 输入电流
        self.I = bm.Variable(bm.zeros(size))
        
        # 积分函数
        self.integral = bp.odeint(f=self.derivative, method='exp_auto')
    
    def derivative(self, V, w, VT, t, a, b, tau_w, Delta_T, el, VT_rest, g_L, C, I):
        """AdEx模型的微分方程"""
        # 膜电位动力学
        dVdt = (g_L * (el - V) + g_L * Delta_T * bm.exp((V - VT) / Delta_T) - w + I) / C
        # 适应变量动力学
        dwdt = (a * (V - el) - w) / tau_w
        # 阈值动力学
        dVTdt = (VT_rest - VT) / self.tau_VT
        
        return dVdt, dwdt, dVTdt
    
    def update(self, tdi):
        """更新神经元状态"""
        V, w, VT = self.integral(self.V, self.w, self.VT, tdi.t, tdi.dt, 
                               self.a, self.b, self.tau_w, self.Delta_T, 
                               self.el, self.VT_rest, self.g_L, self.C, self.I)
        
        # 检测动作电位
        spike = V >= self.th
        
        # 动作电位后重置
        self.V.value = bm.where(spike, self.el, V)
        self.w.value = bm.where(spike, w + self.b, w)
        self.VT.value = bm.where(spike, VT + self.VT_jump, VT_rest)
        
        # 记录发放的动作电位
        self.spike = spike

# 使用NEURON实现的AdEx神经元模型
class AdExNeuronModel:
    """AdEx神经元模型类，用于创建和管理神经元模型"""
    
    def __init__(self):
        """初始化神经元模型"""
        self.cell = None
        self.brain2_model = None
    
    def create_neuron(self):
        """创建单个胞体的AdEx神经元模型"""
        # 初始化神经元
        self.cell = h.Section(name='soma')
        self.cell.L = 40  # 长度 (μm)
        self.cell.diam = 40  # 直径 (μm)
        self.cell.Ra = 100  # 轴向电阻 (Ω·cm)
        self.cell.cm = 1  # 膜电容 (μF/cm²)
        
        # 插入AdEx机制
        self.cell.insert('adex')
        
        # 设置AdEx模型参数的初始值
        self.cell.th_adex = 20        # [mV] 动作电位阈值
        self.cell.C_adex = 281        # [pF] 膜电容
        self.cell.g_L_adex = 30       # [nS] 膜电导
        self.cell.el_adex = -70.6     # [mV] 静息电位
        self.cell.VT_rest_adex = -50.4# [mV] 重置电压
        self.cell.Delta_T_adex = 2    # [mV] 指数参数
        self.cell.tau_w_adex = 144    # [ms] 适应变量w的时间常数
        self.cell.a_adex = 4          # [nS] 适应耦合常数
        self.cell.b_adex = 0.0805     # [nA] 动作电位触发的适应
        self.cell.w_jump_adex = 400   # [pA] 动作电位后去极化电流
        self.cell.tau_wtail_adex = 40 # [ms] 动作电位后去极化时间常数
        self.cell.tau_VT_adex = 50    # [ms] VT的时间常数
        self.cell.VT_jump_adex = 20   # 自适应阈值
        
        print("已创建单个胞体的AdEx神经元模型")
        return self.cell
    
    def create_brain2_neuron(self):
        """创建Brain 2版本的AdEx神经元模型"""
        self.brain2_model = Brain2AdExNeuron(size=1)
        print("已创建Brain 2版本的AdEx神经元模型")
        return self.brain2_model
    
    def set_parameters(self, params):
        """设置神经元模型参数"""
        if self.cell is None:
            self.create_neuron()
        
        for param_name, param_value in params.items():
            if hasattr(self.cell, param_name):
                setattr(self.cell, param_name, param_value)
    
    def set_brain2_parameters(self, params):
        """设置Brain 2神经元模型参数"""
        if self.brain2_model is None:
            self.create_brain2_neuron()
        
        for param_name, param_value in params.items():
            # 转换参数名（去掉_adex后缀）
            brain2_param_name = param_name.replace('_adex', '')
            if hasattr(self.brain2_model, brain2_param_name):
                param_var = getattr(self.brain2_model, brain2_param_name)
                if isinstance(param_var, bm.Variable):
                    param_var.value = param_value
    
    def simulate(self, current_amplitude=0.1, delay=100, duration=800, total_time=1000):
        """使用NEURON运行神经元模拟"""
        if self.cell is None:
            self.create_neuron()
        
        # 设置记录
        v_vec = h.Vector()
        t_vec = h.Vector()
        v_vec.record(self.cell(0.5)._ref_v)
        t_vec.record(h._ref_t)
        
        # 注入电流
        stim = h.IClamp(self.cell(0.5))
        stim.delay = delay  # ms
        stim.dur = duration  # ms
        stim.amp = current_amplitude  # nA
        
        # 运行模拟
        h.tstop = total_time  # ms
        h.run()
        
        # 返回模拟结果
        return np.array(t_vec), np.array(v_vec)
    
    def simulate_brain2(self, current_amplitude=0.1, delay=100, duration=800, total_time=1000, dt=0.025):
        """使用Brain 2运行神经元模拟"""
        if self.brain2_model is None:
            self.create_brain2_neuron()
        
        # 准备时间
        times = np.arange(0, total_time, dt)
        
        # 准备电流输入
        current = np.zeros_like(times)
        delay_idx = int(delay / dt)
        dur_idx = int(duration / dt)
        current[delay_idx:delay_idx+dur_idx] = current_amplitude * 1000  # 转换为pA
        
        # 准备记录
        voltage = np.zeros_like(times)
        
        # 初始化
        self.brain2_model.V.value = self.brain2_model.el.value
        self.brain2_model.w.value = 0.
        self.brain2_model.VT.value = self.brain2_model.VT_rest.value
        
        # 运行模拟
        for i, t in enumerate(times):
            self.brain2_model.I.value = current[i]
            self.brain2_model.update(bp.tdi(dt=dt))
            voltage[i] = self.brain2_model.V.value
        
        return times, voltage

# 创建Bluepyopt细胞模型
def create_cell_model():
    """创建Bluepyopt细胞模型"""
    # 创建形态
    morph = ephys.morphologies.NrnFileMorphology(
        None,  # 无形态文件，使用简单胞体
        do_replace_axon=True, 
        soma_at_origin=True)
    
    # 创建机制 - 使用adex机制
    mechs = [
        ephys.mechanisms.NrnMODMechanism(
            name='adex',
            suffix='adex',
            locations=[ephys.locations.NrnSeclistLocation('soma', seclist_name='somatic')]
        )
    ]
    
    # 创建参数 - AdEx模型的关键参数
    parameters = [
        ephys.parameters.NrnSectionParameter(
            name='a_adex',
            param_name='a_adex',
            locations=[ephys.locations.NrnSeclistLocation('soma', seclist_name='somatic')],
            bounds=[0.1, 10.0],
            value=4.0,
            frozen=False
        ),
        ephys.parameters.NrnSectionParameter(
            name='b_adex',
            param_name='b_adex',
            locations=[ephys.locations.NrnSeclistLocation('soma', seclist_name='somatic')],
            bounds=[0.001, 0.2],
            value=0.0805,
            frozen=False
        ),
        ephys.parameters.NrnSectionParameter(
            name='tau_w_adex',
            param_name='tau_w_adex',
            locations=[ephys.locations.NrnSeclistLocation('soma', seclist_name='somatic')],
            bounds=[50.0, 300.0],
            value=144.0,
            frozen=False
        ),
        ephys.parameters.NrnSectionParameter(
            name='Delta_T_adex',
            param_name='Delta_T_adex',
            locations=[ephys.locations.NrnSeclistLocation('soma', seclist_name='somatic')],
            bounds=[0.5, 5.0],
            value=2.0,
            frozen=False
        ),
        ephys.parameters.NrnSectionParameter(
            name='el_adex',
            param_name='el_adex',
            locations=[ephys.locations.NrnSeclistLocation('soma', seclist_name='somatic')],
            bounds=[-80.0, -60.0],
            value=-70.6,
            frozen=False
        )
    ]
    
    # 创建细胞模型
    cell_model = ephys.models.CellModel(
        name='adex_neuron',
        morph=morph,
        mechs=mechs,
        params=parameters
    )
    
    print("已设置Bluepyopt细胞模型")
    return cell_model, parameters

# 创建刺激协议
def create_protocol(amplitude=0.1, delay=100, duration=800, total_duration=1000):
    """创建电生理刺激协议"""
    # 创建注入电流刺激
    stim = ephys.stimuli.NrnSquarePulse(
        step_amplitude=amplitude,  # nA
        step_delay=delay,      # ms
        step_duration=duration,   # ms
        location=ephys.locations.NrnSectionLocation(
            name='soma_center',
            sec_name='soma',
            sec_index=0,
            loc=0.5
        ),
        total_duration=total_duration  # ms
    )
    
    # 创建记录
    rec = ephys.recordings.CompRecording(
        name='v_soma',
        location=ephys.locations.NrnSectionLocation(
            name='soma_center',
            sec_name='soma',
            sec_index=0,
            loc=0.5
        ),
        variable='v'
    )
    
    # 创建协议
    ephys_protocol = ephys.protocols.SweepProtocol(
        name='step_protocol',
        stimuli=[stim],
        recordings=[rec]
    )
    
    print("已设置电生理刺激协议")
    return ephys_protocol

# 创建优化目标
def create_objectives(ephys_protocol, target_data=None, use_features=True, feature_names=None):
    """设置优化目标"""
    objectives = []
    
    if target_data is not None:
        if use_features and feature_names is not None:
            # 使用EFEL特征作为优化目标
            for feature_name in feature_names:
                objectives.append(
                    ephys.objectives.SingletonObjective(
                        name=f'{feature_name}_objective',
                        protocol=ephys_protocol,
                        recording_names=['v_soma'],
                        stimulus_name='step_protocol.stimulus',
                        feature_name=feature_name,
                        # 这里我们将实验数据与模拟数据比较
                        evaluator=lambda x, feature=feature_name: np.sqrt(np.mean((x - target_data) ** 2))
                    )
                )
        else:
            # 创建目标函数 - 基于膜电位的均方误差
            objectives = [
                ephys.objectives.SingletonObjective(
                    name='v_error',
                    protocol=ephys_protocol,
                    recording_names=['v_soma'],
                    stimulus_name='step_protocol.stimulus',
                    feature_name='voltage',
                    # 这里我们将实验数据与模拟数据比较
                    evaluator=lambda x: np.sqrt(np.mean((x - target_data) ** 2))
                )
            ]
    else:
        # 如果没有目标数据，使用基于特征的目标函数
        # 这里仅作为示例，实际应根据需要调整
        objectives = []
    
    # 创建适应度计算器
    fitness_calculator = ephys.objectivescalculators.ObjectivesCalculator(
        objectives=objectives
    )
    
    print("已设置优化目标")
    return objectives, fitness_calculator

# 创建优化器
def create_optimizer(cell_model, parameters, fitness_calculator, ephys_protocol, algorithm='deap'):
    """设置优化器"""
    if algorithm == 'deap':
        # 创建优化器 - 使用DEAP的差分进化算法
        optimizer = bpop.optimisations.DEAPOptimisation(
            evaluator=ephys.evaluators.CellEvaluator(
                cell_model=cell_model,
                param_names=[param.name for param in parameters],
                fitness_calculator=fitness_calculator,
                protocols={'step_protocol': ephys_protocol}
            ),
            offspring_size=10,  # 每一代的后代数量
            map_function=None,  # 使用默认映射函数
            seed=1  # 随机种子，保证结果可重复
        )
    else:
        # 默认使用DEAP优化器
        optimizer = bpop.optimisations.DEAPOptimisation(
            evaluator=ephys.evaluators.CellEvaluator(
                cell_model=cell_model,
                param_names=[param.name for param in parameters],
                fitness_calculator=fitness_calculator,
                protocols={'step_protocol': ephys_protocol}
            ),
            offspring_size=10,
            map_function=None,
            seed=1
        )
    
    print("已设置优化器")
    return optimizer

# 运行优化过程
def run_optimization(optimizer, max_ngen=50, verbose=True):
    """运行优化过程
    
    Args:
        optimizer: 优化器实例
        max_ngen: 最大进化代数
        verbose: 是否打印详细信息
    
    Returns:
        优化后的参数
    """
    print(f"开始优化过程，最大代数: {max_ngen}")
    
    # 运行优化
    final_pop, hall_of_fame, logs, hist = optimizer.run(max_ngen=max_ngen)
    
    # 获取最优参数
    best_individual = hall_of_fame[0]
    best_params = best_individual.to_dict()
    
    print("优化完成，最优参数:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    
    # 绘制适应度历史
    plot_fitness_history(logs)
    
    return best_params


# 测试最优模型


# Stage0拟合流程
def run_stage0_fitting(target_data=None, max_ngen=30):
    """运行Stage0拟合流程，针对亚阈值特征"""
    print("开始Stage0拟合流程...")
    
    # 创建细胞模型
    cell_model, parameters = create_cell_model()
    
    # 创建协议 - 使用较小的电流，避免产生动作电位
    ephys_protocol = create_protocol(amplitude=0.05)
    
    # 创建优化目标
    # Stage0使用亚阈值特征
    stage0_feature = ['voltage_base', 'steady_state_voltage', 'voltage_deflection_vb_ssse', 
                      'sag_amplitude', 'sag_ratio1', 'decay_time_constant_after_stim']
    objectives, fitness_calculator = create_objectives(ephys_protocol, target_data, 
                                                      use_features=False)
    
    # 创建优化器
    optimizer = create_optimizer(cell_model, parameters, fitness_calculator, ephys_protocol)
    
    # 运行优化
    best_params = run_optimization(optimizer, max_ngen=max_ngen)
    
    print("Stage0拟合流程已完成")
    return best_params

# Stage1拟合流程
def run_stage1_fitting(initial_params=None, target_data=None, max_ngen=50):
    """运行Stage1拟合流程，针对全特征集"""
    print("开始Stage1拟合流程...")
    
    # 创建细胞模型
    cell_model, parameters = create_cell_model()
    
    # 如果提供了初始参数，更新参数的初始值
    if initial_params is not None:
        for param in parameters:
            if param.name in initial_params:
                param.value = initial_params[param.name]
    
    # 创建协议 - 使用较大的电流，诱导产生动作电位
    ephys_protocol = create_protocol(amplitude=0.1)
    
    # 创建优化目标
    # Stage1使用全特征集
    stage1_feature = ['voltage_base', 'steady_state_voltage', 'voltage_deflection_vb_ssse', 'sag_amplitude', 
                      'sag_ratio1', 'decay_time_constant_after_stim', 'Spikecount', 'mean_frequency', 
                      'time_to_first_spike', 'AP_amplitude_from_voltagebase', 'AP_width', 'AHP_depth', 
                      'adaptation_index2']
    objectives, fitness_calculator = create_objectives(ephys_protocol, target_data, 
                                                      use_features=False)
    
    # 创建优化器
    optimizer = create_optimizer(cell_model, parameters, fitness_calculator, ephys_protocol)
    
    # 运行优化
    best_params = run_optimization(optimizer, max_ngen=max_ngen)
    
    print("Stage1拟合流程已完成")
    return best_params

# 主函数
def main(data_file=None, use_brain2=False, max_ngen_stage0=30, max_ngen_stage1=50):
    """主函数，运行整个神经元模型拟合流程"""
    # 定义特征集
    stage0_feature = ['voltage_base', 'steady_state_voltage', 'voltage_deflection_vb_ssse', 'sag_amplitude', 'sag_ratio1', 
                      'decay_time_constant_after_stim']
    
    stage1_feature = ['voltage_base', 'steady_state_voltage', 'voltage_deflection_vb_ssse', 'sag_amplitude', 'sag_ratio1', 
                      'decay_time_constant_after_stim', 'Spikecount', 'mean_frequency', 'time_to_first_spike',
                      'AP_amplitude_from_voltagebase', 'AP_width', 'AHP_depth', 'adaptation_index2']
    
    # 准备目标数据
    target_voltage = None
    target_time = None
    acquisition_data = None
    
    # 如果提供了数据文件路径，加载和处理数据
    if data_file and os.path.exists(data_file):
        data = load_patch_data(data_file)
        if data is not None:
            acquisition_data = process_patch_data(data)
            if acquisition_data:
                stage0_protocol = generate_protocol(acquisition_data)
                traces = prepare_traces_for_efel(acquisition_data)
                
                try:
                    # 使用EFEL计算特征值
                    traces_results = efel.get_feature_values(traces, stage1_feature)
                    
                    # 组织特征结果
                    traces_results_dict = {}
                    id = 0
                    for key in acquisition_data.keys():
                        traces_results_dict[key] = traces_results[id]
                        id += 1
                except Exception as e:
                    print(f"EFEL特征计算失败: {e}")
                    traces_results_dict = None
                
                # 选择第一个记录作为目标数据
                first_key = next(iter(acquisition_data.keys()))
                target_voltage = acquisition_data[first_key]
                # 假设采样率为50kHz (根据之前的代码中的/50)
                target_time = np.array(range(1, len(target_voltage) + 1)) / 50
    else:
        print("警告: 未提供有效数据文件，将使用模拟数据进行演示")
        # 生成模拟数据
        target_time = np.arange(0, 1000, 0.025)  # 40kHz采样率
        target_voltage = -70 * np.ones_like(target_time)
        
        # 添加一些模拟的动作电位
        for i in range(5):
            start_idx = int((200 + i * 150) / 0.025)  # 转换为索引
            if start_idx + 200 < len(target_voltage):
                target_voltage[start_idx:start_idx+200] = -70 + 100 * np.exp(-((np.arange(200) - 50) ** 2) / (2 * 10 ** 2))
    
    # 运行两阶段拟合
    # Stage0: 拟合亚阈值特征
    stage0_params = run_stage0_fitting(target_data=target_voltage, max_ngen=max_ngen_stage0)
    
    # Stage1: 使用Stage0的结果作为初始值，拟合全特征集
    best_params = run_stage1_fitting(initial_params=stage0_params, target_data=target_voltage, max_ngen=max_ngen_stage1)
    
    # 测试最优模型
    t_vec, v_vec = test_best_model(best_params, target_voltage, target_time, use_brain2=use_brain2)
    
    # 如果启用了Brain 2，比较两个模型
    if use_brain2:
        print("比较NEURON和Brain 2模型结果...")
        neuron_results, brain2_results = compare_models(best_params, target_voltage, target_time)
    
    # 分析结果
    if acquisition_data is not None:
        results = analyze_results(best_params, acquisition_data)
    
    # 保存结果
    save_fitting_results(best_params)
    
    print("神经元模型拟合流程已完成")
    return best_params

# 运行主函数
if __name__ == '__main__':
    # 可以在这里指定您的patch-clamp数据文件路径
    # data_file = '/path/to/your/patch_clamp_data.nwb'
    # best_params = main(data_file, use_brain2=True)
    
    # 直接运行，使用默认数据或生成模拟数据
    best_params = main(use_brain2=True)