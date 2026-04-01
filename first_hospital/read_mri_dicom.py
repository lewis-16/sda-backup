"""
读取 不同年龄段的SEEG原始数据2026-2-1 下患者 MRI 目录中的 DICOM 数据。
MRI 目录结构：医院刻录的 DICOM 光盘，含 DICOMDIR、PAT00001/STD00001/SERxxxxx/IMGxxxxx（无扩展名）。
"""
import os
from pathlib import Path

try:
    import pydicom
    from pydicom import dcmread
except ImportError:
    raise ImportError("请先安装 pydicom: pip install pydicom")

MRI_BASE = "/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1/于涵 男 14y/MRI"
PAT_DIR = os.path.join(MRI_BASE, "PAT00001", "STD00001")


def is_dicom(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            f.seek(0x80)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def collect_dicom_files(series_dir):
    files = []
    for name in sorted(os.listdir(series_dir)):
        fp = os.path.join(series_dir, name)
        if os.path.isfile(fp) and is_dicom(fp):
            files.append(fp)
    return files


def read_series(series_path, load_pixels=False):
    """读取一个 DICOM 序列（同一序列多张切片），返回 (datasets, pixel_arrays 可选)。"""
    if not os.path.isdir(series_path):
        return [], None
    paths = collect_dicom_files(series_path)
    if not paths:
        return [], None
    datasets = []
    for p in paths:
        try:
            ds = dcmread(p, force=True)
            datasets.append((p, ds))
        except Exception as e:
            print(f"  跳过 {p}: {e}")
    datasets.sort(key=lambda x: getattr(x[1], "InstanceNumber", 0))
    pixel_arrays = []
    if load_pixels:
        for _, ds in datasets:
            if hasattr(ds, "pixel_array"):
                try:
                    pixel_arrays.append(ds.pixel_array)
                except Exception as e:
                    print(f"  无法读取 pixel_array: {e}")
    return datasets, pixel_arrays if pixel_arrays else None


def read_one_slice(dicom_path):
    """读取单张 DICOM 切片，返回 pydicom Dataset。"""
    return dcmread(dicom_path, force=True)


def main():
    print("MRI 目录:", MRI_BASE)
    print("DICOM 根目录 (Study):", PAT_DIR)
    if not os.path.isdir(PAT_DIR):
        print("未找到 PAT00001/STD00001 目录")
        return

    series_list = sorted([d for d in os.listdir(PAT_DIR) if os.path.isdir(os.path.join(PAT_DIR, d))])
    print(f"共 {len(series_list)} 个序列: {series_list}\n")

    for i, ser in enumerate(series_list[:5]):
        ser_path = os.path.join(PAT_DIR, ser)
        load_pixels = i == 0
        datasets, pixels = read_series(ser_path, load_pixels=load_pixels)
        if not datasets:
            print(f"{ser}: 未找到有效 DICOM 文件")
            continue
        ds0 = datasets[0][1]
        mod = getattr(ds0, "Modality", "?")
        desc = getattr(ds0, "SeriesDescription", "") or getattr(ds0, "ProtocolName", "")
        print(f"{ser}: Modality={mod}, 张数={len(datasets)}, SeriesDescription/ProtocolName={desc}")
        if pixels is not None:
            import numpy as np
            arr = np.stack(pixels)
            print(f"  体数据 shape: {arr.shape}, dtype={arr.dtype}")
        else:
            shp = getattr(ds0, "pixel_array", None)
            if hasattr(ds0, "Rows") and hasattr(ds0, "Columns"):
                print(f"  单张尺寸: Rows={ds0.Rows}, Columns={ds0.Columns}")
        print()

    example_path = os.path.join(PAT_DIR, "SER00013", "IMG00001")
    if os.path.isfile(example_path):
        print("--- 单张切片示例 (SER00013/IMG00001) ---")
        ds = read_one_slice(example_path)
        print(f"  PatientName: {getattr(ds, 'PatientName', '')}")
        print(f"  Modality: {getattr(ds, 'Modality', '')}")
        print(f"  Shape: {getattr(ds, 'pixel_array', None).shape if hasattr(ds, 'pixel_array') else 'N/A'}")


if __name__ == "__main__":
    main()
