"""
端到端分类评估：使用 trial 4000:5000 进行 real_time_classification。
对每个 trial 遍历所有 clique，逐 clique 运行 real_time_classifier 得到 PSTH，整合后送入已训练 7 类分类器，统计准确率。
依赖：eval 已生成各 clique 的 model；clique_classify 已训练并保存 psth_cluster_classifier_best.pth。
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from scipy.io import loadmat

MOUNTAINSORT_DIR = "/media/ubuntu/sda/visual_generation/results/neuroscroll_260122/mountainsort"
CLIQUE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
REC_PARAMS_PATH = "/media/ubuntu/sda/duan/result/260121/rec_params.csv"
CONDITION_PATH = "/media/ubuntu/sda/duan/result/260121/images_sequence_10000.csv"
PKL_CLUSTER_PATH = "/media/ubuntu/sda/Monkey/semantic/epoch_10_monkeyN/merged_cluster_class_counts.pkl"
CLASSIFIER_DIR = os.path.join(MOUNTAINSORT_DIR, "clique_classify_4cliques")
MODEL_PATH = os.path.join(CLASSIFIER_DIR, "psth_cluster_classifier_best.pth")
INTAN_PATH = "/home/ubuntu/Documents/jct/project/251205/M190011_260121_150111_merged_130.rhd"
PROBE_MAT_PATH = "/media/ubuntu/sda/duan/raw_data/chanMap_DCX_5mm.mat"
CH_MAP_CSV_PATH = "/media/ubuntu/sda/duan/raw_data/ch_map_R.csv"

TRIAL_START = 4000
TRIAL_END = 5000
N_EVAL_TRIALS = TRIAL_END - TRIAL_START
WINDOW_SAMPLES = 6000
N_TIME_BINS_MS = 600
PSTH_WINDOW_MS = 20
N_BINS_TARGET = 30
GAUSSIAN_SIGMA_BINS = 10
SAMPLE_TO_MS = 0.1
TIME_BINS_INPUT = 26


def load_probe():
    from probeinterface import Probe
    probe_data = loadmat(PROBE_MAT_PATH)
    probe_x = probe_data["xcoords"]
    probe_y = probe_data["ycoords"]
    probe_position = pd.DataFrame(probe_x)
    probe_position[1] = probe_y
    probe_position["chan_map"] = probe_data["chanMap0ind"].astype(int)
    chan_map = pd.read_csv(CH_MAP_CSV_PATH)
    merged = chan_map.merge(probe_position, left_on="probeloc", right_on="chan_map").iloc[chan_map.index].reset_index(drop=True)
    probe = Probe()
    probe.set_contacts(positions=merged.iloc[:, 2:4])
    probe.set_device_channel_indices(range(256))
    return probe


def load_recording_segment_4000_5000(recording_f, rec_params, n_jobs=30):
    stim_points = rec_params["rec_codes_points_10000"].iloc[TRIAL_START:TRIAL_END].values.astype(int)
    segment_start = int(stim_points.min()) - 1500
    segment_end = int(stim_points.max()) + 4500
    segment = recording_f.frame_slice(start_frame=segment_start, end_frame=segment_end)
    segment = segment.save(format="binary", n_jobs=n_jobs)
    return segment, segment_start, stim_points


def load_rec_params():
    rec_params = pd.read_csv(REC_PARAMS_PATH)
    rec_params = rec_params[rec_params["bhv_codes"] == 10].reset_index(drop=True)
    n_rec = len(rec_params)
    condition = pd.read_csv(CONDITION_PATH)
    n_cond = len(condition)
    if n_cond < n_rec:
        condition = pd.concat([condition] * (n_rec // n_cond + 1), ignore_index=True)
    condition = condition.iloc[:n_rec].reset_index(drop=True)
    rec_params = pd.concat([rec_params, condition], axis=1)
    fs_ratio = 30000 / 10000
    rec_params["rec_codes_points_10000"] = (rec_params["rec_codes_points"] / fs_ratio).astype(int)
    rec_params = rec_params[(rec_params["trial_ids"] >= 300) & (rec_params["trial_ids"] < 5300)].reset_index(drop=True)
    return rec_params


def spiketrain_to_response_30(spiketrain, all_neuron_ids, neuron_to_idx, n_neurons):
    n_time_bins_ms = N_TIME_BINS_MS
    raster = np.zeros((n_neurons, n_time_bins_ms), dtype=np.float32)
    for unit_id, times in spiketrain.items():
        if unit_id not in neuron_to_idx:
            continue
        ni = neuron_to_idx[unit_id]
        for t in times:
            bin_idx = int(np.floor(t * SAMPLE_TO_MS))
            if 0 <= bin_idx < n_time_bins_ms:
                raster[ni, bin_idx] += 1
    psth = np.zeros((n_neurons, n_time_bins_ms), dtype=np.float32)
    for time_point_ms in range(n_time_bins_ms):
        if time_point_ms - PSTH_WINDOW_MS // 2 < 0:
            ws, we = 0, PSTH_WINDOW_MS
        elif time_point_ms + PSTH_WINDOW_MS // 2 + 1 > n_time_bins_ms:
            ws, we = n_time_bins_ms - PSTH_WINDOW_MS, n_time_bins_ms
        else:
            ws = time_point_ms - PSTH_WINDOW_MS // 2
            we = time_point_ms + PSTH_WINDOW_MS // 2 + 1
        tw = np.arange(ws, we)
        psth[:, time_point_ms] = 1000 * np.sum(raster[:, tw], axis=1) / len(tw)
    psth = gaussian_filter1d(psth, sigma=GAUSSIAN_SIGMA_BINS, axis=1, mode="nearest")
    bin_size = n_time_bins_ms // N_BINS_TARGET
    n_valid = N_BINS_TARGET * bin_size
    psth = psth[:, :n_valid].reshape(n_neurons, N_BINS_TARGET, bin_size).mean(axis=-1)
    return psth


class TemporalEPEncoder(nn.Module):
    def __init__(self, input_dim=244, time_bins=30, d_model=256, n_token=128,
                 num_conv_layers=3, dropout=0.2, output_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.time_bins = time_bins
        self.d_model = d_model
        self.n_token = n_token
        self.output_dim = output_dim
        if input_dim > 200:
            hidden_dim = min(input_dim // 4, d_model * 8)
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model * 4),
                nn.LayerNorm(d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model * 4),
                nn.LayerNorm(d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
        conv_layers = []
        for i in range(num_conv_layers):
            in_ch = d_model if i == 0 else d_model * 2
            out_ch = d_model if i == num_conv_layers - 1 else d_model * 2
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.temporal_conv = nn.Sequential(*conv_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_token)
        self.final_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = self.adaptive_pool(x)
        x = x.transpose(1, 2)
        x = self.final_proj(x)
        x = self.feature_proj(x)
        return x.mean(dim=1)


class PSTHClusterClassifier(nn.Module):
    def __init__(self, encoder, num_classes=7):
        super().__init__()
        self.encoder = encoder
        feat_dim = getattr(encoder, "output_dim", 768)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_features=False):
        feat = self.encoder(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


def main():
    from utils_clique import get_recording_clique, real_time_classifier, neuron_inf_dict_to_dataframe
    from utils_clique import SimpleAutoSort

    print("1. 加载 probe 与 recording (Intan + 预处理)...")
    probe = load_probe()
    recording_raw = se.read_intan(INTAN_PATH, stream_id="0", ignore_integrity_checks=True)
    recording_raw = spre.unsigned_to_signed(recording_raw)
    recording_raw = spre.resample(recording_raw, 10000)
    recording_recorded = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)
    recording_recorded = spre.notch_filter(recording_recorded, freq=50)
    recording_f = spre.common_reference(recording_recorded, reference="global", operator="median")
    recording_f = recording_f.set_probegroup(probe)
    print("   recording 加载完成")

    print("2. 获取 trial 4000:5000 的 stim 与 recording segment...")
    rec_params = load_rec_params()
    recording_segment, segment_start, stim_points = load_recording_segment_4000_5000(recording_f, rec_params)
    image_col = "image_name" if "image_name" in rec_params.columns else [c for c in rec_params.columns if "image" in c.lower()][0]
    stimulus_ids_eval = rec_params[image_col].iloc[TRIAL_START:TRIAL_END].astype(str).values
    print(f"   rec_params: {len(rec_params)} 行 (bhv_codes==10 后合并 condition, 再筛 trial_ids 300~5299), 评估 trial: iloc[{TRIAL_START}:{TRIAL_END}]")
    print(f"   segment 样本数: {recording_segment.get_num_samples()}, 评估 trial 数: {N_EVAL_TRIALS}")

    print("3. 加载 clique_info 与各 clique 的 model、detect_channel、classification_mapping...")
    with open(os.path.join(MOUNTAINSORT_DIR, "clique_info.pkl"), "rb") as f:
        clique_info = pickle.load(f)
    cliques = clique_info["cliques"]
    clique_configs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for clique in cliques:
        cid = clique.clique_id
        if cid not in CLIQUE_IDS:
            continue
        data_folder = os.path.join(MOUNTAINSORT_DIR, f"clique_{cid}")
        with open(os.path.join(data_folder, "neuron_inf_all.pickle"), "rb") as f:
            neuron_inf_dict = pickle.load(f)
        neuron_inf = neuron_inf_dict_to_dataframe(neuron_inf_dict)
        test_rec = get_recording_clique(recording_segment, clique)
        ch_ids = test_rec.get_channel_ids()
        probe_to_clique_index = {str(ch_id): i for i, ch_id in enumerate(ch_ids)}
        detect_channel = {-1: [], 1: []}
        for _, row in neuron_inf.iterrows():
            ext_ch = str(row["extremum_channel"])
            if ext_ch not in probe_to_clique_index:
                continue
            sign = row.get("sign", -1)
            if sign is None:
                sign = -1
            detect_channel[sign].append(probe_to_clique_index[ext_ch])
        detect_channel[1] = list(set(detect_channel[1]))
        detect_channel[-1] = list(set(detect_channel[-1]))
        model_save_dir = os.path.join(data_folder, "model_1")
        with open(os.path.join(model_save_dir, "classification_mapping.pkl"), "rb") as f:
            classification_mapping = pickle.load(f)
        n_channels = test_rec.get_num_channels()
        keep_id_list = classification_mapping["label_list"]
        autosort_model = SimpleAutoSort(
            ch_num=n_channels,
            samplepoints=30,
            device=device,
            set_shank_id=keep_id_list,
            save_dir=model_save_dir,
            pos_weight_noise=None,
            pos_weight_label=None,
        )
        autosort_model.clsfier_noise.load_state_dict(
            torch.load(os.path.join(model_save_dir, "multitask_single_wave_clsfier_noise_clsfier.pth"), map_location=device)
        )
        autosort_model.clsfier_label.load_state_dict(
            torch.load(os.path.join(model_save_dir, "multitask_single_wave_clsfier_label_clsfier.pth"), map_location=device)
        )
        autosort_model.eval()
        all_neuron_ids = sorted(classification_mapping["label_to_unit"].values())
        neuron_to_idx = {nid: i for i, nid in enumerate(all_neuron_ids)}
        clique_configs.append({
            "clique": clique,
            "autosort_model": autosort_model,
            "classification_mapping": classification_mapping,
            "detect_channel": detect_channel,
            "all_neuron_ids": all_neuron_ids,
            "neuron_to_idx": neuron_to_idx,
            "n_neurons": len(all_neuron_ids),
        })
    n_neurons_total = sum(c["n_neurons"] for c in clique_configs)
    print(f"   共 {len(clique_configs)} 个 clique, 总神经元数: {n_neurons_total}")

    print("4. 加载 7 类分类器与标签映射...")
    with open(PKL_CLUSTER_PATH, "rb") as f:
        cluster_data = pickle.load(f)
    class_to_cluster = {}
    for cluster_id in range(7):
        for class_name in cluster_data[cluster_id]:
            class_to_cluster[class_name] = cluster_id
    condition_df = pd.read_csv(CONDITION_PATH)
    image_path_dict = {row["image_name"]: row["image_path"] for _, row in condition_df.iterrows()}

    true_labels = []
    valid_trial_indices = []
    for t in range(N_EVAL_TRIALS):
        img_name = stimulus_ids_eval[t]
        if img_name not in image_path_dict:
            continue
        path = image_path_dict[img_name]
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class_to_cluster:
            continue
        true_labels.append(class_to_cluster[class_name])
        valid_trial_indices.append(t)
    true_labels = np.array(true_labels)
    print(f"   有效评估样本: {len(true_labels)} 个")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到分类器: {MODEL_PATH}")
    encoder = TemporalEPEncoder(
        input_dim=n_neurons_total,
        time_bins=TIME_BINS_INPUT,
        d_model=64,
        n_token=128,
        num_conv_layers=2,
        dropout=0.2,
        output_dim=768,
    )
    classifier_model = PSTHClusterClassifier(encoder, num_classes=7)
    classifier_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    classifier_model = classifier_model.to(device)
    classifier_model.eval()

    print("5. 逐 trial 遍历所有 clique 做 real_time_classifier，整合 PSTH 后分类...")
    all_preds = []
    for trial_idx in tqdm(range(N_EVAL_TRIALS), desc="Trials"):
        stim = int(stim_points[trial_idx])
        start_in_seg = stim - segment_start - 1500
        end_in_seg = stim - segment_start + 4500
        rec_trial = recording_segment.frame_slice(start_frame=start_in_seg, end_frame=end_in_seg)
        responses = []
        for cfg in clique_configs:
            rec_clique = get_recording_clique(rec_trial, cfg["clique"])
            spiketrain = real_time_classifier(
                recording_f=rec_clique,
                autosort_model=cfg["autosort_model"],
                start_frame=0,
                end_frame=WINDOW_SAMPLES,
                id_to_neuron=cfg["classification_mapping"]["label_to_unit"],
                detection_params={"thr_min": 5, "thr_max": 35, "distance": 3, "wlen": 5, "prominence": 15},
                window_params={"left_sample": 10, "right_sample": 20},
                detect_channel=cfg["detect_channel"],
                device=device,
                verbose=False,
            )
            resp_30 = spiketrain_to_response_30(
                spiketrain,
                cfg["all_neuron_ids"],
                cfg["neuron_to_idx"],
                cfg["n_neurons"],
            )
            responses.append(resp_30)
        combined = np.concatenate(responses, axis=0)
        psth_input = combined[:, 4:4 + TIME_BINS_INPUT]
        if psth_input.shape[1] != TIME_BINS_INPUT:
            psth_input = combined[:, 4:]
        x = torch.tensor(psth_input.T, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = classifier_model(x)
        pred = logits.argmax(dim=1).item()
        all_preds.append(pred)

    preds_valid = np.array([all_preds[i] for i in valid_trial_indices])
    correct = (preds_valid == true_labels).sum()
    total = len(true_labels)
    acc = correct / total if total > 0 else 0.0
    print(f"\n分类准确率: {correct}/{total} = {acc:.4f} ({acc*100:.2f}%)")
    print(f"各类样本数: {np.bincount(true_labels, minlength=7)}")
    per_class_correct = np.zeros(7, dtype=np.int64)
    for c in range(7):
        mask = true_labels == c
        if mask.sum() > 0:
            per_class_correct[c] = (preds_valid[mask] == c).sum()
    print(f"各类正确数: {per_class_correct}")


if __name__ == "__main__":
    main()
