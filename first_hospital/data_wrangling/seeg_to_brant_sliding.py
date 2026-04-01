# -*- coding: utf-8 -*-
"""
将 不同年龄段的SEEG原始数据2026-2-1 下的 EDF 整理为 Brant 可用的滑动窗口数据。
标签规则见 SEEG_TO_BRANT_DATA_README.md。
"""
import os
import re
import sys
import gc
import argparse
import warnings
import numpy as np
from scipy import signal as scipy_signal
from tqdm import tqdm

try:
    import mne
except ImportError:
    print("需要安装 mne: pip install mne", file=sys.stderr)
    sys.exit(1)

WINDOW_DURATION_SEC = 90.0
SEG_DURATION_SEC = 6.0
N_SEG = int(WINDOW_DURATION_SEC / SEG_DURATION_SEC)
POINTS_PER_SEG = 1500
TARGET_FS = 250
STEP_SEC = 3.0
POWER_FS = 256
COORDINATION_FILENAME = "coordination.md"
ELECTRODE_ORDER = "ABCDEFGH"


def parse_coordination_md(patient_path):
    """
    解析患者目录下的 coordination.md，返回与电极触点一一对应的 (字母, 触点号) 列表，顺序为 A1..A14, B1..B8, ...（与文档一致）。
    返回: [(letter, contact_num), ...]，以及 valid_set = set of (letter, contact_num)。
    """
    path = os.path.join(patient_path, COORDINATION_FILENAME)
    if not os.path.isfile(path):
        return [], set()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    canonical = []
    pin_re = re.compile(r"SEEG-(\d+)PIN(?:-8-8)?", re.IGNORECASE)
    for letter in ELECTRODE_ORDER:
        pat = r"###\s*电极\s*" + re.escape(letter) + r"\s*\(([^)]+)\)"
        m = re.search(pat, text)
        if not m:
            continue
        header = m.group(1)
        pin_m = pin_re.search(header)
        n_contacts = int(pin_m.group(1)) if pin_m else 0
        for contact in range(1, n_contacts + 1):
            canonical.append((letter, contact))
    return canonical, set(canonical)


def should_exclude_channel(ch_name):
    """按 SEEG_TO_BRANT_DATA_README 第 3 节排除 DC、心电、辅助、参考通道"""
    if not ch_name or not isinstance(ch_name, str):
        return True
    ch = ch_name.strip()
    if re.match(r"POL\s+DC\d+", ch, re.IGNORECASE):
        return True
    if re.match(r"POL\s+ECG", ch, re.IGNORECASE):
        return True
    if ch in ("POL 0", "POL 0V"):
        return True
    if re.match(r"POL\s+[Xx]\d+-\d+", ch):
        return True
    if re.match(r"POL\s+-\d+-\d+", ch):
        return True
    if re.match(r"POL\s+[Ll]\d+", ch):
        return True
    if re.match(r"POL\s+[Xx]\d+\b", ch):
        return True
    if re.match(r"POL\s+-\d+\b", ch):
        return True
    return False


def channel_name_to_elec_contact(ch_name):
    """
    从 EDF 通道名中解析出 (电极字母, 触点号)，支持如 EEG A1-Ref, POL A10, A1, A 1, A-1 等格式。
    若无法解析则返回 None。
    """
    if not ch_name or not isinstance(ch_name, str):
        return None
    ch = ch_name.strip()
    m = re.search(r"\b([A-Ha-h])\s*[-]?\s*(\d+)\b", ch)
    if not m:
        return None
    letter = m.group(1).upper()
    num = int(m.group(2))
    if 1 <= num <= 20:
        return (letter, num)
    return None


def pick_seeg_channels(raw, canonical_order, valid_set):
    """
    按 coordination 定义的顺序筛选通道。返回要保留的通道名列表（按 canonical_order 顺序），
    若某 (letter, num) 在 EDF 中无对应通道则跳过该触点。
    """
    ch_to_pair = {}
    for ch in raw.ch_names:
        pair = channel_name_to_elec_contact(ch)
        if pair is not None and pair in valid_set:
            if pair not in ch_to_pair:
                ch_to_pair[pair] = ch
    ordered_chs = []
    for pair in canonical_order:
        if pair in ch_to_pair:
            ordered_chs.append(ch_to_pair[pair])
    return ordered_chs


def is_ictal_start(desc):
    desc = (desc or "").strip()
    return desc.startswith("◆发作") or desc == "◆发作"


def is_ictal_end(desc):
    desc = (desc or "").strip().lower()
    return desc in ("end", "发作")


def is_interictal(desc):
    desc = (desc or "").strip()
    if not desc or len(desc) > 20:
        return False
    return bool(re.match(r"^[A-Z]+$", desc))


def is_stim_start(desc):
    return desc and "Stim Start" in desc


def is_stim_stop(desc):
    return desc and "Stim Stop" in desc


def build_discharge_and_stim(annotations):
    discharge = []
    stim = []
    ann = annotations
    n = len(ann)
    if n == 0:
        return discharge, stim
    onset = np.asarray(ann.onset).ravel()
    duration = np.asarray(ann.duration).ravel()
    description = np.asarray(ann.description).ravel()
    for i in range(n):
        onset_i = float(onset[i])
        dur_i = float(duration[i])
        desc = str(description[i]).strip() if description[i] is not None else ""
        if is_ictal_start(desc):
            end_time = None
            for j in range(i + 1, n):
                d = str(description[j]).strip().lower() if description[j] is not None else ""
                if is_ictal_end(d):
                    end_time = float(onset[j])
                    break
            if end_time is not None:
                discharge.append((onset_i, end_time))
            else:
                discharge.append((onset_i, onset_i + dur_i))
        elif is_interictal(desc):
            discharge.append((onset_i, onset_i + dur_i))
        elif is_stim_start(desc):
            stop_time = None
            for j in range(i + 1, n):
                d = str(description[j]) if description[j] is not None else ""
                if is_stim_stop(d):
                    stop_time = float(onset[j]) + float(duration[j])
                    break
            if stop_time is not None:
                stim.append((onset_i, stop_time))
    return discharge, stim


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start


def window_overlaps_discharge(win_start, win_end, discharge_list):
    for (d_start, d_end) in discharge_list:
        if overlaps(win_start, win_end, d_start, d_end):
            return True
    return False


def window_in_stim(win_start, win_end, stim_list):
    for (s_start, s_end) in stim_list:
        if overlaps(win_start, win_end, s_start, s_end):
            return True
    return False


def compute_power(data, fs):
    """与 brant/Brant_src/pretrain/pre_utils.compute_power 完全一致。注意：计算频谱时使用 256 Hz（与 Brant 一致）。"""
    f, Pxx_den = scipy_signal.periodogram(data, fs)
    f_thres = [4, 8, 13, 30, 50, 70, 90, 110, 128]
    poses = []
    for fi in range(len(f_thres) - 1):
        cond1_pos = np.where(f_thres[fi] < f)[0]
        cond2_pos = np.where(f_thres[fi + 1] >= f)[0]
        poses.append(np.intersect1d(cond1_pos, cond2_pos))
    ori_shape = Pxx_den.shape[:-1]
    Pxx_den = Pxx_den.reshape(-1, len(f))
    band_sum = [np.sum(Pxx_den[:, band_pos], axis=-1) + 1 for band_pos in poses]
    band_sum = [np.log10(_band_sum)[:, np.newaxis] for _band_sum in band_sum]
    band_sum = np.concatenate(band_sum, axis=-1)
    ori_shape += (8,)
    band_sum = band_sum.reshape(ori_shape)
    return band_sum


def process_one_edf(edf_path, out_dir, target_fs=TARGET_FS, step_sec=STEP_SEC, power_fs=POWER_FS, canonical_order=None, valid_set=None):
    raw = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Channel names are not unique.*", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*Omitted.*annotation.*outside data range.*", category=RuntimeWarning)
            raw = mne.io.read_raw_edf(edf_path, preload=True, encoding="gb18030", verbose=False)
        exclude = [ch for ch in raw.ch_names if should_exclude_channel(ch)]
        if exclude:
            raw.drop_channels(exclude)
        if canonical_order is not None and valid_set is not None and len(valid_set) > 0:
            pick_chs = pick_seeg_channels(raw, canonical_order, valid_set)
            if len(pick_chs) == 0:
                raise ValueError("coordination 筛选后无有效通道，请检查 coordination.md 与 EDF 通道名")
            raw.pick(pick_chs)
        fs_orig = float(raw.info["sfreq"])
        n_ch = len(raw.ch_names)
        ann = raw.annotations
        discharge = []
        stim = []
        if ann is not None and len(ann) > 0:
            discharge, stim = build_discharge_and_stim(ann)
        raw.resample(target_fs, npad="auto", verbose=False)
        fs = target_fs

        duration_sec = raw.times[-1] - raw.times[0] if len(raw.times) else 0
        step_samples = int(round(step_sec * fs))
        window_samples = N_SEG * POINTS_PER_SEG
        if window_samples != int(WINDOW_DURATION_SEC * fs):
            window_samples = int(WINDOW_DURATION_SEC * fs)

        win_starts_sec = []
        t = 0.0
        while t + WINDOW_DURATION_SEC <= duration_sec:
            if not window_in_stim(t, t + WINDOW_DURATION_SEC, stim):
                win_starts_sec.append(t)
            t += step_sec

        if len(win_starts_sec) == 0:
            return 0, 0

        data_list = []
        power_list = []
        label_list = []
        for win_start in win_starts_sec:
            win_end = win_start + WINDOW_DURATION_SEC
            start_samp = int(round(win_start * fs))
            end_samp = int(start_samp + window_samples)
            if end_samp > raw.n_times:
                break
            seg_len_samp = window_samples // N_SEG
            dat, _ = raw[:, start_samp:end_samp]
            if dat.shape[1] != window_samples:
                pad = window_samples - dat.shape[1]
                dat = np.pad(dat, ((0, 0), (0, pad)), mode="edge")
            dat = dat.reshape(n_ch, N_SEG, seg_len_samp)
            if seg_len_samp != POINTS_PER_SEG:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, seg_len_samp)
                x_new = np.linspace(0, 1, POINTS_PER_SEG)
                dat_new = np.zeros((n_ch, N_SEG, POINTS_PER_SEG), dtype=np.float32)
                for c in range(n_ch):
                    for s in range(N_SEG):
                        f = interp1d(x_old, dat[c, s, :], kind="linear", fill_value="extrapolate")
                        dat_new[c, s, :] = f(x_new)
                dat = dat_new
            data_list.append(dat)
            power_seg_list = []
            for seg in range(N_SEG):
                seg_dat = dat[:, seg, :]
                pw = compute_power(seg_dat, power_fs)
                power_seg_list.append(pw)
            power_list.append(np.stack(power_seg_list, axis=1))
            lab = 1 if window_overlaps_discharge(win_start, win_end, discharge) else 0
            label_list.append(lab)

        if len(data_list) == 0:
            return 0, 0

        data = np.stack(data_list, axis=1).astype(np.float32)
        power = np.stack(power_list, axis=1).astype(np.float32)
        label = np.array(label_list, dtype=np.int64)
        n_win = data.shape[1]
        label = label.reshape(1, n_win, 1).repeat(N_SEG, axis=2)

        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "data.npy"), data)
        np.save(os.path.join(out_dir, "power.npy"), power)
        np.save(os.path.join(out_dir, "label.npy"), label)
        return n_win, len(win_starts_sec)
    finally:
        if raw is not None:
            del raw
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="SEEG EDF -> Brant sliding-window data")
    parser.add_argument("--base", default="/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1", help="根目录（患者目录的父目录）")
    parser.add_argument("--out", default="/media/ubuntu/sda/first_hospital/data_wrangling/brant_seeg_out", help="输出根目录")
    parser.add_argument("--fs", type=int, default=250, help="目标采样率 Hz")
    parser.add_argument("--power_fs", type=int, default=POWER_FS, help="计算功率谱时的 fs，固定 256 Hz（与 Brant 一致）")
    parser.add_argument("--step", type=float, default=3.0, help="滑动步长（秒）")
    parser.add_argument("--max_patients", type=int, default=None, help="仅处理前 N 个患者（用于测试，默认全部）")
    args = parser.parse_args()

    base = os.path.abspath(args.base)
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    if not os.path.isdir(base):
        print("目录不存在:", base)
        return

    patient_dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    if args.max_patients is not None:
        patient_dirs = patient_dirs[: args.max_patients]
    total_windows = 0
    for patient_name in tqdm(patient_dirs, desc="患者"):
        patient_path = os.path.join(base, patient_name)
        canonical_order, valid_set = parse_coordination_md(patient_path)
        if len(valid_set) == 0:
            canonical_order, valid_set = None, None
        edf_files = sorted([f for f in os.listdir(patient_path) if f.lower().endswith(".edf")])
        for edf_name in edf_files:
            edf_path = os.path.join(patient_path, edf_name)
            safe_name = edf_name.replace(".edf", "").replace(".EDF", "")
            out_dir = os.path.join(out_root, patient_name, safe_name)
            try:
                n_win, _ = process_one_edf(
                    edf_path, out_dir,
                    target_fs=args.fs,
                    step_sec=args.step,
                    power_fs=args.power_fs,
                    canonical_order=canonical_order,
                    valid_set=valid_set,
                )
                total_windows += n_win
            except Exception as e:
                import traceback
                tqdm.write("失败 {} / {}: {}".format(patient_name, edf_name, e))
                tqdm.write(traceback.format_exc())

    print("完成. 总窗口数:", total_windows)


if __name__ == "__main__":
    main()
