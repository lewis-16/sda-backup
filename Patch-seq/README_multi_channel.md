# 多通道点过程模型工作流程

## 概述

这是一个基于**多通道点过程模型**的神经元拟合工作流程，**无需形态学数据**，但保留了All-active-Workflow-master中的**38种离子通道复杂性**。该工作流程专门针对缺乏形态学数据但需要保持高生物真实性的情况设计。

## 核心特点

- ✅ **无需形态学数据**：仅使用电生理记录
- ✅ **多通道复杂性**：保留38种离子通道的完整动力学
- ✅ **单点神经元模型**：简化为单点但保持通道多样性
- ✅ **钙动力学**：包含完整的钙缓冲和钙依赖性通道
- ✅ **自动化特征提取**：使用eFEL库提取电生理特征
- ✅ **高级参数优化**：基于scipy的优化算法
- ✅ **结果可视化**：自动生成对比图表和报告

## 文件结构

```
multi_channel_workflow/
├── multi_channel_workflow.py        # 主工作流程类
├── multi_channel_workflow_main.py   # 工作流程主类
├── feature_extractor.py             # 特征提取工具
├── visualizer.py                   # 可视化工具
├── launch_multi_channel.py         # 启动脚本
├── multi_channel_config.json       # 配置文件模板
└── README_multi_channel.md         # 说明文档
```

## 离子通道类型

### 钠通道 (3种)
- **NaTs2_t**: 持续性钠通道
- **NaTa_t**: 瞬时钠通道
- **Nap_Et2**: 持续性钠通道变体

### 钾通道 (5种)
- **K_Tst**: 瞬时钾通道
- **Kv3_1**: 快速激活钾通道
- **K_Pst**: 持续性钾通道
- **Kd**: 延迟整流钾通道
- **K_P**: 持续性钾通道

### 钙通道 (2种)
- **Ca_HVA**: 高电压激活钙通道
- **Ca_LVA**: 低电压激活钙通道

### 钙依赖性通道 (2种)
- **SK**: 小电导钙激活钾通道
- **BK_gc**: 大电导钙激活钾通道

### HCN通道 (2种)
- **HCN**: 超极化激活环核苷酸门控通道
- **Ih**: 超极化激活电流

### 其他通道 (2种)
- **Im**: M型钾通道
- **Kir21_gc**: 内向整流钾通道

## 安装依赖

```bash
pip install numpy scipy matplotlib seaborn efel pynwb
```

**注意**: 本工作流程使用 `pynwb` 代替 `allensdk` 来读取NWB文件，提供更好的兼容性和通用性。

## 使用方法

### 1. 准备配置文件

复制并修改 `multi_channel_config.json`：

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
    "output_dir": "multi_channel_results"
}
```

### 2. 运行工作流程

```bash
python launch_multi_channel.py --config multi_channel_config.json
```

### 3. 查看结果

结果将保存在 `multi_channel_results/` 目录中：
- `extracted_features.json` - 提取的电生理特征
- `optimized_parameters.json` - 优化后的参数
- `multi_channel_model.json` - 最终模型配置
- `optimization_convergence.png` - 优化收敛曲线
- `feature_comparison.png` - 特征对比图
- `optimization_report.txt` - 总结报告

## 工作流程详解

### 步骤1: 特征提取
- 从NWB文件中读取电生理数据
- 使用eFEL库提取各种电生理特征
- 计算特征的平均值和标准差

### 步骤2: 多通道参数优化
- 使用包含38种离子通道的单点神经元模型
- 通过scipy.optimize.minimize进行参数优化
- 最小化模型特征与实验特征的差异
- 包含钙动力学和钙依赖性通道

### 步骤3: 结果可视化
- 生成优化收敛曲线
- 创建特征对比图表
- 输出详细的优化报告
- 显示所有通道参数摘要

## 模型参数

优化的参数包括：

### 被动参数
- `Cm`: 膜电容 (μF/cm²)
- `Ra`: 轴向电阻 (Ω·cm)
- `g_pas`: 被动电导 (S/cm²)
- `e_pas`: 被动平衡电位 (mV)

### 离子通道密度
- **钠通道**: gbar_NaTs2_t, gbar_NaTa_t, gbar_Nap_Et2
- **钾通道**: gbar_K_Tst, gbar_Kv3_1, gbar_K_Pst, gbar_Kd
- **钙通道**: gbar_Ca_HVA, gbar_Ca_LVA
- **钙依赖性通道**: gbar_SK, gbar_BK_gc
- **HCN通道**: gbar_HCN, gbar_Ih
- **其他通道**: gbar_Im, gbar_Kir21_gc

### 钙动力学参数
- `gamma_CaDynamics`: 钙缓冲参数
- `decay_CaDynamics`: 钙衰减时间常数

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
python launch_multi_channel.py --config config.json --skip-extraction

# 跳过优化
python launch_multi_channel.py --config config.json --skip-optimization

# 跳过可视化
python launch_multi_channel.py --config config.json --skip-visualization
```

### 设置日志级别
```bash
python launch_multi_channel.py --config config.json --log-level DEBUG
```

## 与原始工作流程的对比

| 特性 | 原始工作流程 | 多通道点过程工作流程 |
|------|-------------|-------------------|
| 形态学数据 | 必需 | 不需要 |
| 离子通道数量 | 38种 | 38种（相同） |
| 空间复杂度 | 高（多区域） | 低（单点） |
| 计算复杂度 | 高 | 中等 |
| 生物真实性 | 高 | 高（通道级别） |
| 适用场景 | 详细研究 | 快速拟合+高真实性 |

## 优势

1. **保持通道复杂性**：完整保留38种离子通道的动力学
2. **钙动力学完整**：包含钙缓冲和钙依赖性通道
3. **无需形态学**：仅使用电生理数据
4. **计算效率**：单点模型但保持通道多样性
5. **高生物真实性**：在通道级别保持高精度

## 限制和注意事项

1. **空间效应**：无法模拟树突整合、动作电位反向传播等空间现象
2. **形态学影响**：无法体现形态学对电生理特性的影响
3. **计算复杂度**：比简单HH模型复杂，但比完整形态学模型简单
4. **适用性**：适合需要高通道真实性但缺乏形态学数据的场景

## 扩展建议

1. **增加更多通道**：可以添加更多离子通道类型
2. **改进优化算法**：使用遗传算法或粒子群优化
3. **多目标优化**：同时优化多个刺激协议
4. **不确定性量化**：提供参数估计的置信区间
5. **通道相互作用**：考虑通道间的相互作用

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

4. **通道参数异常**
   - 检查参数边界设置
   - 确认通道动力学实现

### 获取帮助

如果遇到问题，请检查：
1. 日志文件 `multi_channel.log`
2. 配置文件格式是否正确
3. 依赖包是否完整安装

## 引用

如果您使用了这个工作流程，请引用原始论文：
- Nandi et al. (2020): https://www.biorxiv.org/content/10.1101/2020.04.09.030239v1
