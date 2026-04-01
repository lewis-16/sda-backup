import numpy as np
import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict

def robust_std(x):
    """
    Compute a robust estimate of standard deviation using the median absolute deviation (MAD).
    
    Args:
        x (array-like): Input data.
    
    Returns:
        float: Robust estimate of standard deviation.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def segment_data(data, segment_length, shuffle=True, threshold=10, min_segments_for_filtering=100):
    """
    Segment time-series data into fixed-length chunks and remove segments with outliers.
    
    Args:
        data (array-like): Input 1D time-series data.
        segment_length (int): Length of each segment(in samples).
        shuffle (bool): Whether to shuffle segments after filtering.
        threshold (float): Outlier rejection threshold in MAD units.
    
    Returns:
        np.ndarray: Filtered and optionally shuffled segments.
    """
    num_segments = len(data) // segment_length
    if num_segments == 0:
        print("⚠️ Not enough data to form even one segment.")
        return np.array([])

    segments = np.array(np.split(data[:num_segments * segment_length], num_segments))

    if num_segments < min_segments_for_filtering:
        print(f"⚠️ Only {num_segments} segments available. Skipping noise filtering.")
        clean_segments = segments
    else:
        # Compute robust threshold
        median_val = np.median(data)
        std_est = robust_std(data)
        upper_thresh = median_val + threshold * std_est
        lower_thresh = median_val - threshold * std_est

        # Vectorized filtering: find segments where all values are within thresholds
        mask = np.all((segments >= lower_thresh) & (segments <= upper_thresh), axis=1)
        clean_segments = segments[mask]
        print(f"Segments before cleaning: {len(segments)} | After cleaning: {len(clean_segments)}")

    if shuffle:
        np.random.shuffle(clean_segments)

    return clean_segments


def create_balanced_pairs(region_segments_dict, max_dataset_size=20000, seed=42):
    """
    Create a balanced dataset of similar and dissimilar pairs for contrastive training.
    
    Args:
        region_segments_dict (dict): Keys are region names, values are np.arrays of segments.
        max_dataset_size (int): Max total number of training pairs.
        seed (int): Random seed for reproducibility.
    
    Returns:
        (list of tuple, list of int): List of (segment1, segment2) pairs and corresponding binary labels (0 = similar, 1 = dissimilar).
    """
    random.seed(seed)
    np.random.seed(seed)
    region_names = list(region_segments_dict.keys())
    num_regions = len(region_names)

    # ---- SIMILAR PAIRS ----
    similar_pairs = []
    per_region_quota = max_dataset_size // 2 // num_regions

    for region in region_names:
        segments = region_segments_dict[region]
        indices = list(range(len(segments)))
        random.shuffle(indices)  # Avoid sorted bias

        # Sample random non-repeating index pairs (like combinations but shuffled)
        seen_pairs = set()
        max_unique_pairs = len(indices) * (len(indices) - 1) // 2
        while len(similar_pairs) < per_region_quota * (region_names.index(region) + 1):
            i, j = random.sample(indices, 2)
            if (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                similar_pairs.append((segments[i], segments[j]))
                seen_pairs.add((i, j))
            if len(seen_pairs) >= len(indices) * (len(indices) - 1) // 2 or len(seen_pairs) >= max_unique_pairs:
                break

    similar_labels = [0] * len(similar_pairs)

    # ---- DISSIMILAR PAIRS ----
    dissimilar_pairs = []
    seen_dissimilar = set()
    per_region_combo_quota = max_dataset_size // 2 // (num_regions * (num_regions - 1) // 2)

    for i, region_a in enumerate(region_names):
        for region_b in region_names[i+1:]:
            segs_a = region_segments_dict[region_a]
            segs_b = region_segments_dict[region_b]

            count = 0
            while count < per_region_combo_quota:
                a = random.choice(segs_a)
                b = random.choice(segs_b)
                key = (id(a), id(b))
                key_rev = (id(b), id(a))
                if key not in seen_dissimilar and key_rev not in seen_dissimilar:
                    dissimilar_pairs.append((a, b))
                    seen_dissimilar.add(key)
                    count += 1

    dissimilar_labels = [1] * len(dissimilar_pairs)

    # ---- Final Combine & Shuffle ----
    total_pairs = min(len(similar_pairs), len(dissimilar_pairs))
    final_pairs = similar_pairs[:total_pairs] + dissimilar_pairs[:total_pairs]
    final_labels = [0] * total_pairs + [1] * total_pairs

    combined = list(zip(final_pairs, final_labels))
    random.shuffle(combined)
    final_pairs, final_labels = zip(*combined)

    print(f"Created {len(final_pairs)} total pairs ({total_pairs} similar, {total_pairs} dissimilar) from {num_regions} regions.")

    return list(final_pairs), list(final_labels)


def make_balanced_reference(embeddings, labels, per_class=None, seed=42):
    """
    Returns (ref_emb, ref_labels) where each class contributes the same number of samples.
    If per_class is None, we use the minimum class count.
    """
    # Ensure numpy array for easy indexing
    emb = np.asarray(embeddings)

    by_cls = defaultdict(list)
    for i, y in enumerate(labels):
        by_cls[y].append(i)

    if per_class is None:
        per_class = min(len(v) for v in by_cls.values())

    rng = np.random.default_rng(seed)
    sel_idx = []
    for y, idxs in by_cls.items():
        take = min(per_class, len(idxs))
        sel = rng.choice(idxs, size=take, replace=False)
        sel_idx.extend(sel.tolist())

    sel_idx = np.asarray(sel_idx, dtype=int)
    ref_emb = emb[sel_idx]
    ref_labels = [labels[i] for i in sel_idx]
    return ref_emb, ref_labels


class LFPDataset(Dataset):
    """
    PyTorch Dataset class for loading LFP contrastive pairs.
    
    Each item is a tuple of two time-series segments and a binary label:
    - (segment1, segment2, label)
    - label = 0 for similar (same region), 1 for dissimilar (different regions)
    """
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        x1 = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x1, x2, label
    
def _augment_timeseries(x_np, jitter_std=0.05, scale_min=0.9, scale_max=1.1, time_shift_frac=0.2, mask_frac=0.05):
    """
    Light, physiology-friendly augmentations: scale, jitter, small circular shift, tiny time mask.
    x_np: (T,)
    """
    x = x_np.copy()

    # amplitude scale
    scale = np.random.uniform(scale_min, scale_max)
    x = x * scale

    # small circular time shift
    T = x.shape[-1]
    max_shift = max(1, int(time_shift_frac * T))
    if max_shift > 0:
        s = np.random.randint(-max_shift, max_shift+1)
        x = np.roll(x, s)

    # small Gaussian jitter
    if jitter_std > 0:
        x = x + np.random.randn(*x.shape) * jitter_std * (np.std(x_np) + 1e-8)

    # tiny contiguous mask
    mlen = max(1, int(mask_frac * T))
    start = np.random.randint(0, max(1, T - mlen + 1))
    x[start:start+mlen] = 0.0

    return x

class SupConDataset(Dataset):
    """
    For each segment, returns two augmented views and a region label index.
    region_segments_dict: { region_name: np.ndarray[num_segments, T] }
    """
    def __init__(self, region_segments_dict, region_order=None, transform=_augment_timeseries):
        self.transform = transform
        self.samples = []   # list of (segment_np, region_idx)
        if region_order is None:
            region_order = list(region_segments_dict.keys())
        self.region_to_idx = {r:i for i,r in enumerate(region_order)}
        for r in region_order:
            segs = region_segments_dict[r]
            for s in segs:
                self.samples.append( (s.astype(np.float32), self.region_to_idx[r]) )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]                   # x: (T,), y: int
        x1 = self.transform(x)
        x2 = self.transform(x)
        x1 = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)  # (1, T)
        x2 = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)
        y  = torch.tensor(y, dtype=torch.long)
        return x1, x2, y
    


