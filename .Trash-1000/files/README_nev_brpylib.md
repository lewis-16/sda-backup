# NEV文件读取指南

现在您可以使用 `brpylib` 库来直接读取NEV文件，获取精确的刺激时间戳。

## 基本用法

```python
from brpylib import NevFile

# 打开NEV文件
file = NevFile("/media/ubuntu/sda/Monkey/TVSD/monkeyF/20240112/Block_1/NSP-instance1_B001.nev")

# 获取数字I/O事件
digital_events = file.getdata('digitalserial')

# 提取刺激时间戳
timestamps = digital_events['TimeStamp']
unparsed_data = digital_events['UnparsedData']

# 找到刺激事件（假设bit 0是刺激标记）
stim_mask = (unparsed_data & 1) > 0
stim_timestamps = timestamps[stim_mask]
```

## 提供的脚本

### 1. `simple_nev_reader.py`
- 简单的NEV文件读取脚本
- 直接使用您提供的代码示例
- 自动查找对应的日志文件
- 生成刺激矩阵

### 2. `nev_brpy_extractor.py`
- 更完整的NEV文件处理脚本
- 支持命令行参数
- 错误处理和验证

## 使用方法

### 运行简单脚本
```bash
python simple_nev_reader.py
```

### 运行完整脚本
```bash
python nev_brpy_extractor.py /path/to/your/file.nev
```

## 输出格式

脚本会生成包含以下信息的矩阵：
- `start_time`: 刺激开始时间（秒）
- `end_time`: 刺激结束时间（秒）
- `image_id`: 刺激图像ID

## 优势

使用brpylib直接读取NEV文件的优势：

1. **精确时间戳**: 直接从电生理记录中获取刺激时间戳
2. **无需估算**: 不需要根据试验间隔估算时间
3. **实时同步**: 确保刺激时间与电生理数据完全同步
4. **验证机制**: 可以验证刺激事件的数量和时序

## 注意事项

1. 确保brpylib库已正确安装
2. NEV文件路径必须正确
3. 数字I/O事件必须包含刺激标记
4. 建议同时使用日志文件来获取图像ID信息

## 示例输出

```
Found 1657 digital events
Found 1657 stimulus events
Sample rate: 30000 Hz

Stimulus Matrix Shape: (1657, 3)
Format: [start_time, end_time, image_id]

First 10 stimuli:
Start(s)  End(s)    Image_ID
------------------------------
   0.000    0.200    16504
   0.400    0.600    16470
   0.800    1.000    15094
   1.200    1.400     2514
   1.600    1.800     4860
```

这样您就可以获得最准确的刺激-电生理数据对应关系了！
