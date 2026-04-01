"""
比较 T1 相关序列 SER00002, SER00008, SER00009, SER00010 的 DICOM 差异。
"""
import os
from pydicom import dcmread

MRI_BASE = "/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1/于涵 男 14y/MRI"
PAT_DIR = os.path.join(MRI_BASE, "PAT00001", "STD00001")

T1_SERIES = ["SER00002", "SER00008", "SER00009", "SER00010"]


def is_dicom(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            f.seek(0x80)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def get_first_dicom(series_dir):
    for name in sorted(os.listdir(series_dir)):
        fp = os.path.join(series_dir, name)
        if os.path.isfile(fp) and is_dicom(fp):
            return dcmread(fp, force=True)
    return None


def count_slices(series_dir):
    n = 0
    for name in os.listdir(series_dir):
        fp = os.path.join(series_dir, name)
        if os.path.isfile(fp) and is_dicom(fp):
            n += 1
    return n


def main():
    tags = [
        "SeriesDescription", "ProtocolName", "SeriesNumber",
        "Rows", "Columns", "NumberOfFrames",
        "SliceThickness", "PixelSpacing", "SpacingBetweenSlices",
        "ImageOrientationPatient", "ImagePositionPatient",
        "EchoTime", "RepetitionTime", "InversionTime",
        "ScanningSequence", "SequenceVariant", "ScanOptions",
        "MRAcquisitionType", "EchoTrainLength",
    ]
    print("T1 序列对比 (每序列取第一张切片的 DICOM 元数据)\n")
    for ser in T1_SERIES:
        ser_path = os.path.join(PAT_DIR, ser)
        if not os.path.isdir(ser_path):
            print(f"{ser}: 目录不存在\n")
            continue
        ds = get_first_dicom(ser_path)
        n_slices = count_slices(ser_path)
        if ds is None:
            print(f"{ser}: 无有效 DICOM, 张数={n_slices}\n")
            continue
        print(f"===== {ser} (共 {n_slices} 张) =====")
        for tag in tags:
            val = getattr(ds, tag, None)
            if val is not None:
                if tag == "PixelSpacing" and hasattr(val, "__iter__"):
                    val = [float(x) for x in val]
                if tag in ("ImageOrientationPatient", "ImagePositionPatient") and hasattr(val, "__iter__"):
                    val = [float(x) for x in val]
                print(f"  {tag}: {val}")
        print()


if __name__ == "__main__":
    main()
