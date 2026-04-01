import glob
import os

import pyvista as pv

export_root = "/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1/于涵 男 14y/于涵_20240603074026"
pattern = os.path.join(export_root, "files", "volumeModel", "*", "*.vti")
vti_list = sorted(glob.glob(pattern))
if not vti_list:
    raise FileNotFoundError("未找到 .vti 文件")
first_vti = vti_list[0]
print("加载:", first_vti)

grid = pv.read(first_vti)
print("尺寸:", grid.dimensions)
print("范围:", grid.bounds)

plotter = pv.Plotter()
plotter.add_volume(grid, cmap="gray", opacity="linear")
plotter.show()
