# MRI 目录说明与读取方法

## 目录结构（DICOM 光盘）

`不同年龄段的SEEG原始数据2026-2-1/于涵 男 14y/MRI/` 下是**医院刻录的 DICOM 光盘**内容：

| 内容 | 说明 |
|------|------|
| **DICOMDIR** | DICOM 目录索引文件，记录整盘影像的层级关系 |
| **PAT00001/STD00001/** | 患者/检查（Study）目录 |
| **SER00001 ~ SER00014** | 14 个扫描序列（Series），每个序列下为多张切片 |
| **IMG00001, IMG00002, ...** | 单张 DICOM 图像，**无扩展名**，需按二进制头判断 |

影像文件路径示例：
```
MRI/PAT00001/STD00001/SER00013/IMG00001   （DICOM medical imaging data）
```

## 用 Python 读取

### 1. 安装依赖

```bash
conda activate spike_sorting
pip install pydicom numpy
```

### 2. 单张切片

```python
from pydicom import dcmread

path = "不同年龄段的SEEG原始数据2026-2-1/于涵 男 14y/MRI/PAT00001/STD00001/SER00013/IMG00001"
ds = dcmread(path, force=True)

# 元信息
print(ds.PatientName, ds.Modality, ds.SeriesDescription)

# 像素数组 (numpy)
arr = ds.pixel_array   # shape 如 (512, 512)
```

### 3. 整序列（多张切片 → 3D 体数据）

遍历某个 `SERxxxxx` 目录下所有 DICOM 文件（无扩展名，可用 `file` 或读 0x80 处 4 字节是否为 `DICM` 判断），按 `InstanceNumber` 排序后逐张 `dcmread`，再堆叠成 3D 数组：

```python
import os
from pydicom import dcmread
import numpy as np

def read_series(series_dir):
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir)
             if os.path.isfile(os.path.join(series_dir, f))]
    slices = []
    for f in files:
        try:
            ds = dcmread(f, force=True)
            if hasattr(ds, 'pixel_array'):
                slices.append((getattr(ds, 'InstanceNumber', 0), ds.pixel_array))
        except Exception:
            pass
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices])

series_path = "不同年龄段的SEEG原始数据2026-2-1/于涵 男 14y/MRI/PAT00001/STD00001/SER00013"
vol = read_series(series_path)   # shape 如 (20, 512, 512)
```

### 4. 通过 DICOMDIR（可选）

若希望按“光盘标准”解析整盘结构，可使用 pydicom 的 **DICOM File-set** 支持（含 `DICOMDIR`）：

- 文档：[DICOM File-sets and DICOMDIR](https://pydicom.github.io/pydicom/stable/tutorials/filesets.html)
- 指定 MRI 目录为 File-set 根目录，再通过索引访问各 SOP 实例。

## 使用提供的脚本

项目内脚本 `read_mri_dicom.py` 会：

- 扫描 `PAT00001/STD00001` 下各 `SERxxxxx`
- 识别无扩展名的 DICOM 文件（按文件头 `DICM`）
- 读取每个序列的元数据与 pixel array，并打印前 3 个序列信息

运行（在 spike_sorting 环境）：

```bash
conda run -n spike_sorting python read_mri_dicom.py
```

如需读其他患者，修改脚本中的 `MRI_BASE` 或通过命令行参数传入患者 MRI 目录即可。

---

## 如何区分 T1 / T2 / FLAIR

DICOM 里可用以下信息区分权重：

| 依据 | 说明 |
|------|------|
| **SeriesDescription / ProtocolName** | 常含 "T1"、"T2"、"FLAIR"、"MPRage"（多为 T1）、"T2W"（T2 加权）等 |
| **EchoTime (TE)** | 单位 ms。T1 通常短 TE（&lt;30）；T2/FLAIR 长 TE（&gt;80） |
| **RepetitionTime (TR)** | 单位 ms。T1 通常短 TR（&lt;600）；T2/FLAIR 长 TR（&gt;1500） |
| **ScanningSequence** | 如 SE、IR、GR；MPRage 多为 GR，多为 T1 |

**于涵 男 14y 本例**（运行 `inspect_mri_series.py` 可得）：

| 类型 | 序列示例 | 说明 |
|------|----------|------|
| **T1** | SER00002 (Survey CLEAR), **SER00008/09/10** (MPRageAX SENSE) | TR≈8–11 ms, TE≈3.8–4.6 ms |
| **T2** | **SER00011/12/13** (T2W_DRIVE_2mm) | TR≈6122 ms, TE≈103 ms, 描述含 T2W |
| **FLAIR** | **SER00003/05/06** (3D_FLAIR_SHC), SER00007 (Synergy) | TR≈4800 ms, TE≈320 ms, 描述含 FLAIR 或 IR |
| **其他** | SER00014 (DWI) | 扩散加权 |

结论：**T1 选 SER00008/09/10（MPRage）；T2 选 SER00011/12/13（T2W_DRIVE）；FLAIR 选 SER00003/05/06。**
