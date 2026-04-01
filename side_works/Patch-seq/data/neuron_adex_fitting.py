# 导入必要的库
import numpy as np
import neuron
import bluepyopt as bpop
import bluepyopt.ephys as ephys
import os
import matplotlib.pyplot as plt
from neuron import h

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

class AdExNeuronModel:
    """AdEx神经元模型类，用于创建和管理神经元模型"""
    
    def __init__(self, data_file=None):
        """初始化神经元模型
        
        Args:
            data_file: patch-clamp数据文件路径
        """
        self.data_file = data_file
        self.cell = None
        self.cell_model = None
        self.ephys_protocol = None
        self.parameters = None
        self.responses = None
        self.objectives = None
        self.fitness_calculator = None
        self.optimizer = None
        
        # 加载实验数据
        if self.data_file and os.path.exists(self.data_file):
            self.load_data()
        else:
            # 生成模拟数据用于演示
            self.generate_sample_data()
            print(f"警告: 未提供数据文件，将使用模拟数据进行演示")
    
    def load_data(self):
        """加载patch-clamp实验数据"""
        # 这里根据实际数据格式修改
        # 假设数据是电压-时间序列，存储在CSV文件中
        try:
            data = np.loadtxt(self.data_file, delimiter=',')
            self.time = data[:, 0]
            self.voltage = data[:, 1]
            print(f"成功加载数据文件: {self.data_file}")
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """生成模拟的patch-clamp数据用于演示"""
        self.time = np.arange(0, 1000, 0.025)  # 40ms采样率
        self.voltage = -70 * np.ones_like(self.time)
        
        # 添加一些模拟的动作电位
        for i in range(5):
            start_idx = int(200 + i * 150) // 0.025
            if start_idx + 200 < len(self.voltage):
                self.voltage[start_idx:start_idx+200] = -70 + 100 * np.exp(-((np.arange(200) - 50) ** 2) / (2 * 10 ** 2))
    
    def create_neuron(self):
        """创建单个胞体的神经元模型"""
        # 初始化神经元
        self.cell = h.Section(name='soma')
        self.cell.L = 40  # 长度 (μm)
        self.cell.diam = 40  # 直径 (μm)
        self.cell.Ra = 100  # 轴向电阻 (Ω·cm)
        self.cell.cm = 1  # 膜电容 (μF/cm²)
        
        # 插入我们新创建的AdEx机制
        self.cell.insert('adex')
        
        # 设置AdEx模型参数的初始值
        # 这些参数从MATLAB文件中提取
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
    
    def setup_bluepyopt_model(self):
        """设置Bluepyopt优化框架"""
        # 创建形态
        morph = ephys.morphologies.NrnFileMorphology(
            None,  # 无形态文件，使用简单胞体
            do_replace_axon=True, 
            soma_at_origin=True)
        
        # 创建机制 - 使用我们新的adex机制替换原来的hh机制
        mechs = [
            ephys.mechanisms.NrnMODMechanism(
                name='adex',
                suffix='adex',
                locations=[ephys.locations.NrnSeclistLocation('soma', seclist_name='somatic')]
            )
        ]
        
        # 创建参数 - AdEx模型的关键参数
        self.parameters = [
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
        self.cell_model = ephys.models.CellModel(
            name='adex_neuron',
            morph=morph,
            mechs=mechs,
            params=self.parameters
        )
        
        print("已设置Bluepyopt细胞模型")
    
    def setup_protocol(self):
        """设置电生理刺激协议"""
        # 创建注入电流刺激
        stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=0.1,  # nA
            step_delay=100,      # ms
            step_duration=800,   # ms
            location=ephys.locations.NrnSectionLocation(
                name='soma_center',
                sec_name='soma',
                sec_index=0,
                loc=0.5
            ),
            total_duration=1000  # ms
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
        self.ephys_protocol = ephys.protocols.SweepProtocol(
            name='step_protocol',
            stimuli=[stim],
            recordings=[rec]
        )
        
        print("已设置电生理刺激协议")
    
    def setup_objectives(self):
        """设置优化目标"""
        # 创建目标函数 - 基于膜电位的均方误差
        self.objectives = [
            ephys.objectives.SingletonObjective(
                name='v_error',
                protocol=self.ephys_protocol,
                recording_names=['v_soma'],
                stimulus_name='step_protocol.stimulus',
                feature_name='voltage',
                # 这里我们将实验数据与模拟数据比较
                # 实际应用中，您可能需要提取特征（如峰频率、峰宽等）进行比较
                evaluator=lambda x: np.sqrt(np.mean((x - self.voltage) ** 2))
            )
        ]
        
        # 创建适应度计算器
        self.fitness_calculator = ephys.objectivescalculators.ObjectivesCalculator(
            objectives=self.objectives
        )
        
        print("已设置优化目标")
    
    def setup_optimizer(self):
        """设置优化器"""
        # 创建优化器 - 使用DEAP的差分进化算法
        self.optimizer = bpop.optimisations.DEAPOptimisation(
            evaluator=ephys.evaluators.CellEvaluator(
                cell_model=self.cell_model,
                param_names=[param.name for param in self.parameters],
                fitness_calculator=self.fitness_calculator,
                protocols={'step_protocol': self.ephys_protocol}
            ),
            offspring_size=10,  # 每一代的后代数量
            map_function=None,  # 使用默认映射函数
            seed=1  # 随机种子，保证结果可重复
        )
        
        print("已设置优化器")
    
    def run_optimization(self, max_ngen=50):
        """运行优化过程
        
        Args:
            max_ngen: 最大进化代数
        
        Returns:
            优化后的参数
        """
        print(f"开始优化过程，最大代数: {max_ngen}")
        
        # 运行优化
        final_pop, hall_of_fame, logs, hist = self.optimizer.run(max_ngen=max_ngen)
        
        # 获取最优参数
        best_individual = hall_of_fame[0]
        best_params = best_individual.to_dict()
        
        print("优化完成，最优参数:")
        for param_name, param_value in best_params.items():
            print(f"  {param_name}: {param_value}")
        
        # 保存结果
        self.plot_fitness_history(logs)
        self.test_best_model(best_params)
        
        return best_params
    
    def plot_fitness_history(self, logs):
        """绘制适应度历史曲线"""
        gen = logs.select('gen')
        min_fitness = logs.select('min')
        avg_fitness = logs.select('avg')
        max_fitness = logs.select('max')
        
        plt.figure(figsize=(10, 6))
        plt.plot(gen, min_fitness, 'b-', label='最小适应度')
        plt.plot(gen, avg_fitness, 'g-', label='平均适应度')
        plt.plot(gen, max_fitness, 'r-', label='最大适应度')
        plt.xlabel('代数')
        plt.ylabel('适应度值')
        plt.title('优化过程中的适应度变化')
        plt.legend()
        plt.grid(True)
        plt.savefig('fitness_history.png')
        plt.show()
    
    def test_best_model(self, best_params):
        """测试最优模型"""
        # 创建新的神经元模型
        test_cell = self.create_neuron()
        
        # 应用最优参数
        for param_name, param_value in best_params.items():
            if hasattr(test_cell, param_name):
                setattr(test_cell, param_name, param_value)
        
        # 设置记录
        v_vec = h.Vector()
        t_vec = h.Vector()
        v_vec.record(test_cell(0.5)._ref_v)
        t_vec.record(h._ref_t)
        
        # 注入电流
        stim = h.IClamp(test_cell(0.5))
        stim.delay = 100  # ms
        stim.dur = 800    # ms
        stim.amp = 0.1    # nA
        
        # 运行模拟
        h.tstop = 1000  # ms
        h.run()
        
        # 绘制结果
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(t_vec, v_vec, 'b-', label='优化后模型')
        plt.plot(self.time, self.voltage, 'r--', label='实验数据')
        plt.xlabel('时间 (ms)')
        plt.ylabel('膜电位 (mV)')
        plt.title('最优模型与实验数据比较')
        plt.legend()
        plt.grid(True)
        
        # 绘制参数对比
        plt.subplot(2, 1, 2)
        initial_params = {'a_adex': 4.0, 'b_adex': 0.0805, 'tau_w_adex': 144.0, 'Delta_T_adex': 2.0, 'el_adex': -70.6}
        param_names = list(initial_params.keys())
        initial_values = [initial_params[name] for name in param_names]
        best_values = [best_params.get(name, initial_params[name]) for name in param_names]
        
        x = np.arange(len(param_names))
        width = 0.35
        
        plt.bar(x - width/2, initial_values, width, label='初始参数')
        plt.bar(x + width/2, best_values, width, label='优化后参数')
        plt.xticks(x, param_names)
        plt.ylabel('参数值')
        plt.title('参数优化前后对比')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.show()

# 主函数
def main(data_file=None):
    """主函数，运行整个神经元模型拟合流程"""
    # 创建AdEx神经元模型实例
    model = AdExNeuronModel(data_file)
    
    # 创建神经元
    model.create_neuron()
    
    # 设置Bluepyopt框架
    model.setup_bluepyopt_model()
    model.setup_protocol()
    model.setup_objectives()
    model.setup_optimizer()
    
    # 运行优化
    best_params = model.run_optimization(max_ngen=50)
    
    print("神经元模型拟合流程已完成")
    return best_params

if __name__ == '__main__':
    # 可以在这里指定您的patch-clamp数据文件路径
    # data_file = 'your_patch_clamp_data.csv'
    # main(data_file)
    
    # 或者直接运行，使用模拟数据
    main()