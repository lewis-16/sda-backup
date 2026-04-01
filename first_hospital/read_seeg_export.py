"""
读取 sEEG 导览软件导出目录 于涵_20240603074026：
  - files/volumeModel/*.vti：重建的 3D 脑影像（VTK ImageData）
  - export_info：二进制索引/配置（含 Robot.SEEG、序列等），轨迹可能在其中

用法：
  python read_seeg_export.py [导出目录路径]
  不传路径则用默认：不同年龄段的SEEG原始数据2026-2-1/于涵 男 14y/于涵_20240603074026
"""
import os
import sys
import glob
import struct

EXPORT_ROOT = "/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1/于涵 男 14y/于涵_20240603074026"


def read_vti_headers(export_root):
    """扫描 volumeModel 下所有 .vti，只解析 XML 头得到尺寸与物理信息（不读体数据）。"""
    pattern = os.path.join(export_root, "files", "volumeModel", "*", "*.vti")
    vti_files = glob.glob(pattern)
    if not vti_files:
        return []
    results = []
    for path in sorted(vti_files):
        with open(path, "rb") as f:
            raw = f.read(2048)
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            continue
        if "ImageData" not in text or "WholeExtent" not in text:
            continue
        extent = None
        origin = None
        spacing = None
        for line in text.split(">"):
            if "WholeExtent=" in line:
                import re
                m = re.search(r'WholeExtent="([^"]+)"', line)
                if m:
                    parts = m.group(1).split()
                    if len(parts) == 6:
                        extent = (
                            int(parts[1]) - int(parts[0]) + 1,
                            int(parts[3]) - int(parts[2]) + 1,
                            int(parts[5]) - int(parts[4]) + 1,
                        )
            if "Origin=" in line:
                import re
                m = re.search(r'Origin="([^"]+)"', line)
                if m:
                    origin = m.group(1).strip()
            if "Spacing=" in line:
                import re
                m = re.search(r'Spacing="([^"]+)"', line)
                if m:
                    spacing = m.group(1).strip()
        name = os.path.basename(path)
        results.append({
            "path": path,
            "name": name,
            "extent": extent,
            "origin": origin,
            "spacing": spacing,
        })
    return results


def load_vti_volume(vti_path):
    """用 PyVista 读取单个 .vti 为 3D 数组。需安装: pip install pyvista."""
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("读取 .vti 需安装 pyvista: pip install pyvista")
    grid = pv.read(vti_path)
    if hasattr(grid, "get_array") and grid.array_names:
        arr = grid.get_array(grid.array_names[0])
    else:
        arr = grid.point_data.get_array(0) if grid.point_data.keys() else None
    if arr is not None:
        arr = arr.reshape(grid.dimensions[2], grid.dimensions[1], grid.dimensions[0])
    return grid, arr


def inspect_export_info(export_root):
    """简单查看 export_info 二进制中的可读片段（UTF-16 LE）。"""
    path = os.path.join(export_root, "export_info")
    if not os.path.isfile(path):
        return []
    with open(path, "rb") as f:
        raw = f.read(50000)
    found = []
    try:
        s = raw.decode("utf-16-le", errors="ignore")
        for token in ["Robot.SEEG", "display", "series", "window", "preview", "trajectory", "electrode", "point", "line"]:
            if token in s:
                found.append(token)
    except Exception:
        pass
    return found


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else EXPORT_ROOT
    if not os.path.isdir(root):
        print("目录不存在:", root)
        return
    print("导出目录:", root)
    print()

    print("=== 1. 3D 体数据 (volumeModel/*.vti) ===")
    headers = read_vti_headers(root)
    for h in headers:
        print("  文件:", h["name"])
        print("    extent (nx ny nz):", h["extent"])
        print("    origin (mm):", h["origin"])
        print("    spacing (mm):", h["spacing"])
        print()
    if not headers:
        print("  未找到 .vti 文件")
        print()

    print("=== 2. 用 PyVista 读取第一个 .vti 为 numpy 数组（可选）===")
    if headers:
        vti_path = headers[0]["path"]
        try:
            grid, arr = load_vti_volume(vti_path)
            if arr is not None:
                print("  shape:", arr.shape, "dtype:", arr.dtype)
            else:
                print("  已打开 grid，未提取 point_data 数组")
        except Exception as e:
            print("  错误:", e)
    print()

    print("=== 3. export_info ===")
    print("  类型: 二进制（内含 UTF-16 文本片段，如 Robot.SEEG、display、series）")
    tokens = inspect_export_info(root)
    if tokens:
        print("  出现的片段:", tokens)
    print("  SEEG 电极轨迹若由该软件导出，可能编码在 export_info 中，需软件文档或逆向解析。")
    print()
    print("读取方式小结:")
    print("  - 3D 大脑/影像: 使用 files/volumeModel/*/*.vti，用 pyvista.read(...) 或 vtk 读入。")
    print("  - 轨迹/电极: 当前仅知 export_info 含 Robot.SEEG 等，具体格式需厂商说明或进一步解析。")


if __name__ == "__main__":
    main()
