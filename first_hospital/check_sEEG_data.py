from mne.io import read_raw_edf
import mne
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

base_path = "/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1"
output_file = "/media/ubuntu/sda/first_hospital/sEEG_check_output.txt"
with open(output_file, "w", encoding="utf-8") as _:
    pass

def log(msg):
    print(msg, flush=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

patient_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
results = []

for patient_name in patient_dirs:
    patient_path = os.path.join(base_path, patient_name)
    edf_files = sorted([f for f in os.listdir(patient_path) if f.lower().endswith('.edf')])
    for edf_name in edf_files:
        edf_path = os.path.join(patient_path, edf_name)
        try:
            raw = read_raw_edf(edf_path, preload=False, encoding='gb18030')
            n_channels = len(raw.ch_names)
            events_from_annot, event_dict = mne.events_from_annotations(raw)
            n_events = len(events_from_annot)
            event_descriptions = list(event_dict.keys())
            results.append({
                "patient": patient_name,
                "edf_file": edf_name,
                "n_channels": n_channels,
                "n_events": n_events,
                "event_descriptions": event_descriptions,
                "event_dict": event_dict
            })
            log(f"患者: {patient_name}")
            log(f"  文件: {edf_name}")
            log(f"  通道数: {n_channels}")
            log(f"  事件数: {n_events}")
            log(f"  事件类型: {event_descriptions}")
            log("-" * 60)
        except Exception as e:
            log(f"患者: {patient_name}, 文件: {edf_name} - 读取失败: {e}")
            log("-" * 60)

log(f"\n共处理 {len(results)} 段sEEG数据")
