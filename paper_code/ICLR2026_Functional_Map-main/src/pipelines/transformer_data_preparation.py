import numpy as np
import torch
from torch.utils.data import Dataset

class MRCPStreamDataset(Dataset):
    """
    Continuous segmentation (no events). Each sample returns:
      waves (C,T), fembs (C,F), region_ids (C,)
    """
    def __init__(self, window_ms=500.0, hop_ms=500.0, fs=None):
        self.samples = []
        self.window_ms = window_ms
        self.hop_ms = hop_ms
        self.fs_default = fs
        self.region_vocab = {}  # region_name -> id

    def _region_id(self, name: str) -> int:
        name = str(name)
        if name not in self.region_vocab:
            self.region_vocab[name] = len(self.region_vocab)
        return self.region_vocab[name]

    def add_session(self, *, session_path: str, fs: int,
                    signals: dict,        # {channel: 1D np.array}
                    embeds: dict,         # {channel: 1D np.array}
                    region_map: dict      # {channel: "GPi"/"STN"/"VO"/...}
                   ) -> int:
        if fs is None and self.fs_default is None:
            raise ValueError("fs must be provided.")
        fs = fs or self.fs_default

        # Use only channels that have signal, embedding, and region label
        ch_names = sorted([ch for ch in signals.keys() if ch in embeds and ch in region_map])
        if len(ch_names) == 0:
            print(f"[WARN] no labeled channels in {session_path}")
            return 0

        # Truncate to the minimum length across channels
        lengths = [len(signals[ch]) for ch in ch_names]
        L = int(min(lengths))
        win_len = int(round((self.window_ms/1000.0) * fs))
        hop_len = int(round((self.hop_ms/1000.0) * fs))
        if L < win_len:
            print(f"[WARN] {session_path}: too short for one window; skipping")
            return 0

        # Region ids aligned to ch_names
        reg_ids = np.array([self._region_id(region_map[ch]) for ch in ch_names], dtype=np.int64)
        fembs_full = np.stack([embeds[ch].astype(np.float32) for ch in ch_names], axis=0)  # (C,F)

        added = 0
        for s0 in range(0, L - win_len + 1, hop_len):
            s1 = s0 + win_len
            waves = np.stack([signals[ch][s0:s1].astype(np.float32) for ch in ch_names], axis=0)  # (C,T)
            self.samples.append((
                torch.from_numpy(waves),                      # (C,T)
                torch.from_numpy(fembs_full.copy()),          # (C,F)
                torch.from_numpy(reg_ids.copy())              # (C,)
            ))
            added += 1
        return added

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

