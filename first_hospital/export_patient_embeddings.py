# -*- coding: utf-8 -*-
"""
使用预训练 Brant 权重，从每个患者中每类抽取 100 条，生成 embedding，
保存为 (n_patient, 200, d_model) 的 npy 文件，以及同名的 _info.npz。

_info.npz 内容：
  - patient_names: (n_patient,) 各样本对应的患者名（患者目录名）
  - labels: (n_patient, 200) 每条样本的标签，前 per_class 个为 0，后 per_class 个为 1

加载示例：
  emb = np.load("patient_embeddings.npy")
  info = np.load("patient_embeddings_info.npz", allow_pickle=True)
  names = info["patient_names"]
  labels = info["labels"]
  # emb[i, j] 来自患者 names[i]，标签为 labels[i, j]

数据目录结构：data_dir / 患者名 / 记录名 / data.npy, power.npy, label.npy
"""
import os
import sys
import argparse
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRANT_SRC = os.path.join(PROJECT_ROOT, "paper_code", "Brant_src")
sys.path.insert(0, BRANT_SRC)
os.chdir(BRANT_SRC)

from pretrain.pre_model import TimeEncoder, ChannelEncoder
from utils import get_emb, unwrap_ddp


def list_record_dirs(data_dir):
    out = []
    for p in sorted(os.listdir(data_dir)):
        pdir = os.path.join(data_dir, p)
        if not os.path.isdir(pdir):
            continue
        for r in sorted(os.listdir(pdir)):
            rpath = os.path.join(pdir, r)
            if os.path.isdir(rpath) and os.path.isfile(os.path.join(rpath, "data.npy")):
                out.append(os.path.join(p, r))
    return out


def get_patient(rel_path):
    return rel_path.split(os.sep)[0]


def get_items_with_labels(data_dir, record_list):
    out = []
    for rel_path in tqdm(record_list, desc="Items with labels"):
        lab_path = os.path.join(data_dir, rel_path, "label.npy")
        if not os.path.isfile(lab_path):
            continue
        y = np.load(lab_path, mmap_mode="r")
        if y.ndim == 3:
            n_boards = y.shape[1]
            for bi in range(n_boards):
                lab = int(y[0, bi, 0])
                out.append((rel_path, bi, lab))
        else:
            n_boards = y.shape[0]
            for bi in range(n_boards):
                lab = int(y.ravel()[bi])
                out.append((rel_path, bi, lab))
    return out


class MmapBoardDataset(Dataset):
    def __init__(self, data_dir, items_list):
        self.data_dir = data_dir
        self.items_list = items_list
        self._mmap_cache = {}

    def _get_mmaps(self, rec_path):
        if rec_path not in self._mmap_cache:
            base = os.path.join(self.data_dir, rec_path)
            d = np.load(os.path.join(base, "data.npy"), mmap_mode="r")
            p = np.load(os.path.join(base, "power.npy"), mmap_mode="r")
            self._mmap_cache[rec_path] = (d, p)
        return self._mmap_cache[rec_path]

    def __len__(self):
        return len(self.items_list)

    def __getitem__(self, i):
        rec_path, board_idx = self.items_list[i]
        d, p = self._get_mmaps(rec_path)
        x = np.ascontiguousarray(d[:, board_idx, :, :].astype(np.float32))
        pw = np.ascontiguousarray(p[:, board_idx, :, :].astype(np.float32))
        return torch.from_numpy(x), torch.from_numpy(pw)


def load_encoder(args):
    encoder_t = TimeEncoder(
        in_dim=args.seg_len,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        seq_len=args.seq_len,
        n_layer=args.time_ar_layer,
        nhead=args.time_ar_head,
        band_num=args.band_num,
        project_mode=args.input_emb_mode,
        learnable_mask=args.learnable_mask,
    ).to(args.device)
    encoder_ch = ChannelEncoder(
        out_dim=args.seg_len,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        n_layer=args.ch_ar_layer,
        nhead=args.ch_ar_head,
    ).to(args.device)
    if args.load_pretrained and args.ckpt_dir:
        map_loc = {"cuda:0": "cuda:{}".format(args.gpu_id)} if torch.cuda.is_available() else "cpu"
        t_path = os.path.join(args.ckpt_dir, "time_encoder_{}.pt".format(args.start_epo_idx))
        ch_path = os.path.join(args.ckpt_dir, "channel_encoder_{}.pt".format(args.start_epo_idx))
        if not os.path.isfile(t_path) or not os.path.isfile(ch_path):
            t_path = os.path.join(args.ckpt_dir, "time_encoder.pt")
            ch_path = os.path.join(args.ckpt_dir, "channel_encoder.pt")
        if os.path.isfile(t_path) and os.path.isfile(ch_path):
            t_sd = torch.load(t_path, map_location=map_loc)
            ch_sd = torch.load(ch_path, map_location=map_loc)
            if args.unwrap_ddp:
                t_sd = unwrap_ddp(t_sd)
                ch_sd = unwrap_ddp(ch_sd)
            encoder_t.load_state_dict(t_sd)
            encoder_ch.load_state_dict(ch_sd)
            print("Pretrained encoders loaded.")
    encoder_t.eval()
    encoder_ch.eval()
    return encoder_t, encoder_ch


def sample_per_patient(all_items_with_label, per_class=100, seed=1):
    random.seed(seed)
    by_patient = defaultdict(list)
    for p, b, lab in all_items_with_label:
        by_patient[get_patient(p)].append((p, b, lab))
    out = []
    for pat, items in sorted(by_patient.items()):
        items0 = [(p, b) for p, b, lab in items if lab == 0]
        items1 = [(p, b) for p, b, lab in items if lab == 1]
        if len(items0) == 0 or len(items1) == 0:
            continue
        n0 = min(per_class, len(items0))
        n1 = min(per_class, len(items1))
        s0 = random.sample(items0, n0)
        s1 = random.sample(items1, n1)
        if n0 < per_class:
            s0 = s0 + random.choices(items0, k=per_class - n0)
        if n1 < per_class:
            s1 = s1 + random.choices(items1, k=per_class - n1)
        out.append((pat, s0 + s1))
    return out


def run_embed_one_patient(items_list, data_dir, encoder_t, encoder_ch, device, seg_len=1500, seq_len=15):
    embeddings = []
    ds = MmapBoardDataset(data_dir, items_list)
    with torch.no_grad():
        for i in range(len(ds)):
            x, pw = ds[i]
            if x.shape[1] != seq_len or x.shape[2] != seg_len:
                continue
            x = x.unsqueeze(0).to(device)
            pw = pw.unsqueeze(0).to(device)
            emb = get_emb(x, pw, encoder_t, encoder_ch)
            emb = emb.mean(dim=(1, 2))
            embeddings.append(emb.cpu().numpy().squeeze(0))
    return np.stack(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/solid1/first_hospital/preprocessed_data/sgement_data")
    parser.add_argument("--out_npy", type=str, default="/mnt/solid1/first_hospital/scripts/finetune/patient_embeddings.npy")
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(PROJECT_ROOT, "paper_code", "Brant_src", "pre_trained_weights"))
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--per_class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--load_pretrained", type=bool, default=True)
    parser.add_argument("--start_epo_idx", type=int, default=29)
    parser.add_argument("--unwrap_ddp", type=bool, default=True)
    parser.add_argument("--seg_len", type=int, default=1500)
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--dim_feedforward", type=int, default=3072)
    parser.add_argument("--time_ar_layer", type=int, default=12)
    parser.add_argument("--time_ar_head", type=int, default=8)
    parser.add_argument("--ch_ar_layer", type=int, default=5)
    parser.add_argument("--ch_ar_head", type=int, default=8)
    parser.add_argument("--band_num", type=int, default=8)
    parser.add_argument("--input_emb_mode", type=str, default="linear")
    parser.add_argument("--learnable_mask", type=bool, default=False)
    args = parser.parse_args()
    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    record_list = list_record_dirs(args.data_dir)
    if not record_list:
        print("No record dirs in", args.data_dir)
        return
    all_items = get_items_with_labels(args.data_dir, record_list)
    patient_samples = sample_per_patient(all_items, per_class=args.per_class, seed=args.seed)
    if not patient_samples:
        print("No patient with both labels, exit.")
        return
    print("Patients with both labels: {}, per patient samples: {}".format(len(patient_samples), 2 * args.per_class))

    encoder_t, encoder_ch = load_encoder(args)
    target_per_patient = 2 * args.per_class
    d_model = args.d_model
    patient_embeddings = []
    patient_names = []
    labels_list = []
    for pat, items in tqdm(patient_samples, desc="Patients"):
        emb = run_embed_one_patient(items, args.data_dir, encoder_t, encoder_ch, args.device, args.seg_len, args.seq_len)
        if emb.size == 0:
            continue
        if emb.shape[0] < target_per_patient:
            pad = np.zeros((target_per_patient - emb.shape[0], d_model), dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=0)
        elif emb.shape[0] > target_per_patient:
            emb = emb[:target_per_patient]
        patient_embeddings.append(emb)
        patient_names.append(pat)
        labels_list.append(np.array([0] * args.per_class + [1] * args.per_class, dtype=np.int64))
    out_arr = np.stack(patient_embeddings, axis=0)
    labels_arr = np.stack(labels_list, axis=0)
    out_dir = os.path.dirname(os.path.abspath(args.out_npy)) or "."
    os.makedirs(out_dir, exist_ok=True)
    np.save(args.out_npy, out_arr.astype(np.float32))
    base = os.path.splitext(args.out_npy)[0]
    np.savez(
        base + "_info.npz",
        patient_names=np.array(patient_names, dtype=object),
        labels=labels_arr,
    )
    print("Saved shape {} to {}".format(out_arr.shape, args.out_npy))
    print("Saved info (patient_names, labels) to {}".format(base + "_info.npz"))


if __name__ == "__main__":
    main()
