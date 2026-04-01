"""
检查各 MRI 序列的 DICOM 元数据，用于区分 T1 / T2 / FLAIR 等。
依据：SeriesDescription、ProtocolName、TE/TR、ScanningSequence 等。
"""
import os
from pydicom import dcmread

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


def get_first_dicom(series_dir):
    for name in sorted(os.listdir(series_dir)):
        fp = os.path.join(series_dir, name)
        if os.path.isfile(fp) and is_dicom(fp):
            return dcmread(fp, force=True)
    return None


def infer_weighting(te_ms, tr_ms, desc):
    if te_ms is None or tr_ms is None:
        return "?"
    desc_upper = (desc or "").upper()
    if "FLAIR" in desc_upper or "FLAIR" in (desc or ""):
        return "FLAIR"
    if "T2" in desc_upper or "T2 " in (desc or "") or " T2" in (desc or ""):
        return "T2"
    if "T1" in desc_upper or "T1 " in (desc or "") or " T1" in (desc or ""):
        return "T1"
    if tr_ms < 600 and te_ms < 30:
        return "T1 (短TR/短TE)"
    if tr_ms > 1500 and te_ms > 80:
        return "T2 (长TR/长TE)"
    if tr_ms > 2500 and 80 < te_ms < 200:
        return "FLAIR 可能 (长TR/长TE)"
    return "需结合描述判断"


def main():
    if not os.path.isdir(PAT_DIR):
        print("未找到 PAT 目录:", PAT_DIR)
        return
    series_list = sorted([d for d in os.listdir(PAT_DIR) if os.path.isdir(os.path.join(PAT_DIR, d))])
    print("序列 | Modality | 描述/协议名 | TE(ms) | TR(ms) | 序列类型 | 推断权重")
    print("-" * 100)
    for ser in series_list:
        ser_path = os.path.join(PAT_DIR, ser)
        ds = get_first_dicom(ser_path)
        if ds is None:
            print(f"{ser} | (无DICOM)")
            continue
        mod = getattr(ds, "Modality", "?")
        desc = getattr(ds, "SeriesDescription", "") or ""
        proto = getattr(ds, "ProtocolName", "") or ""
        desc_str = (desc or proto or "-")[:40]
        te = getattr(ds, "EchoTime", None)
        tr = getattr(ds, "RepetitionTime", None)
        if te is not None and hasattr(te, "original_string"):
            try:
                te_ms = float(te)
            except Exception:
                te_ms = None
        else:
            te_ms = float(te) if te is not None else None
        if tr is not None and hasattr(tr, "original_string"):
            try:
                tr_ms = float(tr)
            except Exception:
                tr_ms = None
        else:
            tr_ms = float(tr) if tr is not None else None
        scan_seq = getattr(ds, "ScanningSequence", "") or "-"
        weight = infer_weighting(te_ms, tr_ms, desc or proto)
        te_s = f"{te_ms:.1f}" if te_ms is not None else "-"
        tr_s = f"{tr_ms:.0f}" if tr_ms is not None else "-"
        print(f"{ser} | {mod} | {desc_str} | {te_s} | {tr_s} | {scan_seq} | {weight}")
    print()
    print("说明: T1 多为短 TR/短 TE; T2 多为长 TR/长 TE; FLAIR 常为长 TR、长 TE，且描述含 FLAIR。")

if __name__ == "__main__":
    main()
