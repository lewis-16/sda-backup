# export_info 中电极坐标的坐标系说明

## 坐标是相对于什么的？

规划里导出的 **entry / target 坐标** 是**规划基准影像（Basis）所在物理空间**下的坐标。  
Basis 在 export_info 里对应 `<template>Basis</template>`，且 `<registration>` 为单位矩阵，因此**没有再做一次变换**，坐标就是 **Basis 体积的“原生”物理空间**。

该物理空间与 **DICOM 标准中的“患者坐标系”（Patient Coordinate System）** 一致，即与生成 Basis 的那组 DICOM 影像使用同一套坐标系。

## 规划基准影像的 0 点在哪里？

- **DICOM 患者坐标系** 的 **(0, 0, 0)** 由**设备/厂家**定义，一般是**扫描架等中心（isocenter）**或设备参考点，**不在患者身上**，也不在影像的角点上。
- 因此：**规划基准影像的“0 点” = 该患者坐标系的 (0, 0, 0) = 设备参考点（多为等中心）**；单位是 **mm**。

## 于涵本例中 Basis 的几何（供参考）

Basis 对应序列 UUID：`19a7d2ff-b1de-46e7-84eb-14b990061f94`，为 **MR（WIP MPRageAX SENSE）**。  
该序列**首帧** DICOM 的几何为：

| 属性 | 值 |
|------|-----|
| ImagePositionPatient | 约 (-96.55, -103.78, 68.16) mm |
| ImageOrientationPatient | 行/列方向余弦（略） |
| PixelSpacing | 0.859375 × 0.859375 mm |
| SliceThickness | 1 mm |

即：**第一张切片中心**在患者坐标系下约为 **(-96.55, -103.78, 68.16) mm**。  
所以影像（和脑组织）大致分布在该点附近，而 **(0, 0, 0)** 在设备参考点，通常在头部附近但不一定在影像范围内。

## 小结

| 问题 | 答案 |
|------|------|
| 坐标相对于什么？ | 相对于 **Basis 体积的物理空间**，与 **DICOM 患者坐标系**一致。 |
| 单位 | **毫米 (mm)**。 |
| 0 点在哪里？ | 在 **DICOM 患者坐标系原点**，一般为**扫描设备等中心/参考点**，不在患者体表或影像角点上。 |
| 如何得到“影像体素坐标”？ | 用该序列的 ImagePositionPatient、ImageOrientationPatient、PixelSpacing、SliceThickness 建立**体素指标 (i,j,k) → 物理 (x,y,z) mm** 的变换，再对 entry/target 做逆变换即可。 |

## 换算到 MRI 体素（规划用 Basis MR）

轨迹与 Basis 在同一物理空间，只需用 Basis 序列的 DICOM 几何做 **物理(mm) → 体素(k,j,i)** 即可。

项目内脚本 **`trajectories_to_mri_voxels.py`** 会：

- 从 export_info 解析轨迹与 Basis 系列 UUID；
- 自动找到 `files/importFile/2024/{Basis的UUID}/` 下 DICOM，按 SliceLocation 排序建 3D 几何；
- 将每条轨迹的 entry/target 从 mm 换算为体素坐标 (k,j,i)，对应 numpy 数组 `vol[k,j,i]`（k=切片，j=行，i=列）。

用法示例：

```bash
python trajectories_to_mri_voxels.py "于涵_20240603074026/export_info"
python trajectories_to_mri_voxels.py "于涵_20240603074026/export_info" --csv
```

若要把轨迹画到**其他 MRI 序列**（如医院光盘里的另一套 DICOM），需要先做**配准**，得到「Basis 空间 → 目标 MRI 空间」的变换矩阵后再换算。
