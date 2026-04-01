# sEEG 导览软件导出目录说明（于涵_20240603074026）

该目录为 sEEG 手术导览/规划软件（名称中含 Robot.SEEG）的导出包，内含**重建的 3D 脑影像**和软件内部索引；**电极轨迹**很可能编码在二进制 `export_info` 中，需厂商格式说明或进一步解析。

---

## 目录结构

```
于涵_20240603074026/
├── export_info          # 二进制索引/配置（含 Robot.SEEG、display、series 等 UTF-16 片段）
└── files/
    ├── volumeModel/     # 3D 体数据（VTK ImageData .vti）
    │   └── {UUID}/
    │       ├── {uuid1}.vti   # 体积 1：与 T1 一致的脑影像
    │       └── {uuid2}.vti   # 体积 2：另一套分辨率/范围
    └── importFile/      # 按日期/UUID 存放的导入文件（含 JPEG 等，多为缩略图/辅助）
```

---

## 1. 读取 3D 大脑影像（.vti）

两个 `.vti` 均为 **VTK ImageData**（可能带 zlib 压缩），可用 **PyVista** 或 **VTK** 直接读成 3D 数组。

### 方式一：PyVista（推荐）

```bash
pip install pyvista
```

```python
import pyvista as pv
import numpy as np

# 路径示例（实际 UUID 以你目录为准）
vti_path = "于涵_20240603074026/files/volumeModel/{92150358-99eb-453e-9762-da18404752e3}/{63d78b21-0071-4362-b5cf-0717b44be134}.vti"
grid = pv.read(vti_path)
arr = grid.point_data.get_array(0)
arr = np.reshape(arr, (grid.dimensions[2], grid.dimensions[1], grid.dimensions[0]))
# 本例中为 (372, 313, 307)，对应 spacing ≈ (0.86, 0.86, 0.56) mm
```

### 方式二：VTK（Python）

```python
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
reader = vtkXMLImageDataReader()
reader.SetFileName(vti_path)
reader.Update()
grid = reader.GetOutput()
arr = vtk_to_numpy(grid.GetPointData().GetScalars())
arr = arr.reshape(grid.GetDimensions()[2], grid.GetDimensions()[1], grid.GetDimensions()[0])
```

### 本例两个 .vti 的差异

| 文件 (UUID 后四位…) | 尺寸 (nx×ny×nz) | 体素间距 (mm) | 说明 |
|----------------------|------------------|----------------|------|
| …0717b44be134        | 307×313×372      | 0.86, 0.86, 0.56 | 与 T1 SER00008 间距一致，应为**同一脑 3D 重建** |
| …c52179ba0b4d        | 512×512×260      | 0.46, 0.46, 0.7  | 更高分辨率/不同 FOV，可能为局部或另一序列 |

---

## 2. SEEG 电极轨迹

- **当前情况**：未发现单独的轨迹文件（如 .vtk polyline、.json、.csv）。`export_info` 为**二进制**，内含 "Robot.SEEG"、"display"、"series"、"window" 等，电极/轨迹信息很可能以**私有格式**写在该文件中。
- **建议**：
  1. 若有该导览软件的名称与版本，可查厂商是否提供“导出轨迹”的格式说明或 API。
  2. 若仅有此导出包，可尝试对 `export_info` 做**逆向**：例如查找连续 float32/float64 数组（每 3 个为一组坐标）、或与界面显示电极数/通道数一致的结构。
  3. 轨迹也可从**术后 CT + 术前 MRI 配准**或**EDF 通道位置信息**中另行计算，不依赖此导出。

---

## 3. 项目内脚本

- **read_seeg_export.py**：扫描导出目录、解析所有 `.vti` 的 XML 头（尺寸、Origin、Spacing），并尝试用 PyVista 读第一个 .vti 为 numpy 数组；同时说明 `export_info` 为二进制、可能含轨迹。
- 运行示例：
  ```bash
  python read_seeg_export.py
  # 或指定路径
  python read_seeg_export.py /path/to/于涵_20240603074026
  ```
- 读取 .vti 体数据前需安装：`pip install pyvista`。

---

## 小结

| 内容           | 位置              | 读取方式                          |
|----------------|-------------------|-----------------------------------|
| 3D 脑影像      | `files/volumeModel/*/*.vti` | PyVista 或 VTK，见上文            |
| 电极/轨迹      | 很可能在 `export_info`     | 需厂商文档或逆向解析，暂无现成解析器 |
