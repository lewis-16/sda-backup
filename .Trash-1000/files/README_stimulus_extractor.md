# Monkey Stimulus Matrix Extractor

这个脚本可以从猴子的实验日志文件中提取刺激信息，生成包含刺激开始时间、结束时间和图像ID的矩阵。

## 功能

- 从MATLAB .mat格式的日志文件中读取刺激信息
- 生成包含 `[start_time, end_time, image_id]` 的矩阵
- 支持批量处理多个日志文件
- 输出为文本文件和CSV文件两种格式

## 使用方法

### 1. 处理单个日志文件

```bash
python monkey_stimulus_extractor.py /path/to/THINGS_monkeyF_20240112_B1.mat
```

### 2. 批量处理所有日志文件

```bash
python monkey_stimulus_extractor.py
```

## 输出格式

### 矩阵格式
```
[start_time, end_time, image_id]
```

- `start_time`: 刺激开始时间（秒）
- `end_time`: 刺激结束时间（秒）  
- `image_id`: 刺激图像ID

### 示例输出
```
Start(s)  End(s)    Image_ID
------------------------------
   0.000    0.200    16504
   0.400    0.600    16470
   0.800    1.000    15094
   1.200    1.400     2514
   1.600    1.800     4860
```

## 参数说明

- `stimulus_duration`: 每个刺激的持续时间（默认：0.2秒）
- `trial_interval`: 试验间隔时间（默认：0.4秒）

## 输出文件

对于每个处理的日志文件，会生成两个输出文件：

1. `*_stimulus_matrix.txt` - 文本格式
2. `*_stimulus_matrix.csv` - CSV格式（便于Excel查看）

## 示例

处理2024年1月12日Block 1的数据：

```bash
python monkey_stimulus_extractor.py /media/ubuntu/sda/Monkey/TVSD/monkeyF/_logs/THINGS_monkeyF_20240112_B1.mat
```

输出：
- 1657个刺激
- 1454个唯一图像
- 时间范围：0.000 - 662.600秒
- 总持续时间：662.600秒

## 依赖

- Python 3.6+
- numpy
- pandas  
- scipy

安装依赖：
```bash
pip install numpy pandas scipy
```
