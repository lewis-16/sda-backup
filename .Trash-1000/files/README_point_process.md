# 点过程模型工作流程

## 概述

这是一个基于点过程模型的神经元拟合工作流程，**无需形态学数据**，仅使用Patch-seq电生理记录进行参数优化。该工作流程是对All-active-Workflow-master的简化版本，专门针对缺乏形态学数据的情况设计。

## 核心特点

- ✅ **无需形态学数据**：仅使用电生理记录
- ✅ **简化的神经元模型**：基于Hodgkin-Huxley模型的单点神经元
- ✅ **自动化特征提取**：使用eFEL库提取电生理特征
- ✅ **参数优化**：基于scipy的优化算法
- ✅ **结果可视化**：自动生成对比图表和报告

## 文件结构

```
point_process_workflow/
├── point_process_workflow.py    # 主工作流程类
├── feature_extractor.py         # 特征提取工具
├── visualizer.py               # 可视化工具
├── launch_point_process.py     # 启动脚本
├── point_process_config.json   # 配置文件模板
└── README.md                   # 说明文档
```

## 安装依赖

```bash
pip install numpy scipy matplotlib seaborn efel allensdk
```

## 使用方法

### 1. 准备配置文件

复制并修改 `point_process_config.json`：

```json
{
    "nwb_path": "your_data.nwb",
    "junction_potential": -14.0,
    "stimulus_types": [
        "Long Square",
        "Ramp",
        "Noise 1",
        "Noise 2"
    ],
    "output_dir": "point_process_results"
}
```

### 2. 运行工作流程

```bash
python launch_point_process.py --config point_process_config.json
```

### 3. 查看结果

结果将保存在 `point_process_results/` 目录中：
- `extracted_features.json` - 提取的电生理特征
- `optimized_parameters.json` - 优化后的参数
- `point_process_model.json` - 最终模型配置
- `optimization_convergence.png` - 优化收敛曲线
- `feature_comparison.png` - 特征对比图
- `optimization_report.txt` - 总结报告

## 工作流程详解

### 步骤1: 特征提取
- 从NWB文件中读取电生理数据
- 使用eFEL库提取各种电生理特征
- 计算特征的平均值和标准差

### 步骤2: 参数优化
- 使用简化的Hodgkin-Huxley模型
- 通过scipy.optimize.minimize进行参数优化
- 最小化模型特征与实验特征的差异

### 步骤3: 结果可视化
- 生成优化收敛曲线
- 创建特征对比图表
- 输出详细的优化报告

## 模型参数

优化的参数包括：
- `Cm`: 膜电容 (μF/cm²)
- `Rm`: 膜电阻 (MΩ·cm²)
- `V_rest`: 静息电位 (mV)
- `g_Na_max`: 最大钠电导 (mS/cm²)
- `g_K_max`: 最大钾电导 (mS/cm²)
- `g_L`: 漏电导 (mS/cm²)

## 提取的特征

### 被动特征
- `voltage_base`: 基础膜电位
- `steady_state_voltage`: 稳态膜电位
- `voltage_deflection`: 电压偏转
- `decay_time_constant`: 衰减时间常数
- `sag_amplitude`: 凹陷幅度
- `sag_ratio`: 凹陷比率

### 主动特征
- `spike_count`: 动作电位数量
- `mean_frequency`: 平均发放频率
- `time_to_first_spike`: 首次发放时间
- `AP_amplitude`: 动作电位幅度
- `ISI_CV`: 发放间隔变异系数
- `adaptation_index`: 适应指数

## 高级选项

### 跳过特定步骤
```bash
# 跳过特征提取
python launch_point_process.py --config config.json --skip-extraction

# 跳过优化
python launch_point_process.py --config config.json --skip-optimization

# 跳过可视化
python launch_point_process.py --config config.json --skip-visualization
```

### 设置日志级别
```bash
python launch_point_process.py --config config.json --log-level DEBUG
```

## 与原始工作流程的对比

| 特性 | 原始工作流程 | 点过程工作流程 |
|------|-------------|---------------|
| 形态学数据 | 必需 | 不需要 |
| 离子通道数量 | 38种 | 3种（简化HH模型） |
| 空间复杂度 | 高（多区域） | 低（单点） |
| 计算复杂度 | 高 | 低 |
| 生物真实性 | 高 | 中等 |
| 适用场景 | 详细研究 | 快速拟合 |

## 限制和注意事项

1. **简化模型**：使用简化的Hodgkin-Huxley模型，无法捕捉复杂的离子通道动力学
2. **空间效应**：无法模拟树突整合、动作电位反向传播等空间现象
3. **通道多样性**：仅包含基本的钠、钾、漏通道，缺乏钙通道等
4. **适用性**：适合快速拟合和初步分析，不适合高精度研究

## 扩展建议

1. **增加离子通道**：可以添加钙通道、HCN通道等
2. **改进优化算法**：使用遗传算法或粒子群优化
3. **多目标优化**：同时优化多个刺激协议
4. **不确定性量化**：提供参数估计的置信区间

## 故障排除

### 常见问题

1. **NWB文件读取错误**
   - 检查文件路径是否正确
   - 确认文件格式是否支持

2. **特征提取失败**
   - 检查刺激类型名称是否匹配
   - 确认数据质量是否良好

3. **优化不收敛**
   - 调整参数边界
   - 增加最大迭代次数
   - 检查初始参数设置

### 获取帮助

如果遇到问题，请检查：
1. 日志文件 `point_process.log`
2. 配置文件格式是否正确
3. 依赖包是否完整安装

## 引用

如果您使用了这个工作流程，请引用原始论文：
- Nandi et al. (2020): https://www.biorxiv.org/content/10.1101/2020.04.09.030239v1
