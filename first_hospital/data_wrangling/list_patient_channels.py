# -*- coding: utf-8 -*-
"""
遍历 不同年龄段的SEEG原始数据2026-2-1 下各患者，列出每名患者各 EDF 的通道名，输出为 MD 文件供人工筛选。
"""
import os
import sys
import argparse

try:
    import mne
except ImportError:
    print("需要安装 mne: pip install mne", file=sys.stderr)
    sys.exit(1)

BASE = "/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1"
OUT_MD = "/media/ubuntu/sda/first_hospital/data_wrangling/patient_channels_list.md"


def main():
    parser = argparse.ArgumentParser(description="列出各患者 EDF 通道名 -> MD")
    parser.add_argument("--base", default=BASE, help="患者根目录")
    parser.add_argument("--out", default=OUT_MD, help="输出 MD 路径")
    parser.add_argument("--first_only", action="store_true", help="每患者仅列第一个 EDF（加快生成）")
    parser.add_argument("--max_patients", type=int, default=None, help="仅处理前 N 个患者")
    args = parser.parse_args()

    base = os.path.abspath(args.base)
    if not os.path.isdir(base):
        print("目录不存在:", base)
        return
    patient_dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    if args.max_patients is not None:
        patient_dirs = patient_dirs[: args.max_patients]
    lines = [
        "# 各患者 EDF 通道列表",
        "",
        "以下按患者目录列出 EDF 的通道名称，便于与 coordination.md 对照、人工筛选需保留的通道。",
        "",
    ]
    if args.first_only:
        lines.append("（每患者仅列出**第一个 EDF** 的通道；若有多文件可去掉 `--first_only` 重新生成。）")
    lines.extend(["", "---", ""])
    for patient_name in patient_dirs:
        patient_path = os.path.join(base, patient_name)
        edf_files = sorted([f for f in os.listdir(patient_path) if f.lower().endswith(".edf")])
        if not edf_files:
            lines.append("## {}\n\n（无 EDF 文件）\n".format(patient_name))
            continue
        if args.first_only:
            edf_files = edf_files[:1]
        lines.append("## {}".format(patient_name))
        lines.append("")
        for edf_name in edf_files:
            edf_path = os.path.join(patient_path, edf_name)
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, encoding="gb18030", verbose=False)
                ch_names = raw.ch_names
                n = len(ch_names)
            except Exception as e:
                lines.append("- **{}**：读取失败 `{}`\n".format(edf_name, e))
                continue
            lines.append("### {}（共 {} 通道）".format(edf_name, n))
            lines.append("")
            lines.append("| 序号 | 通道名 |")
            lines.append("|------|--------|")
            for i, ch in enumerate(ch_names, 1):
                lines.append("| {} | {} |".format(i, ch))
            lines.append("")
        lines.append("---")
        lines.append("")
    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("已写入:", out_path)


if __name__ == "__main__":
    main()
