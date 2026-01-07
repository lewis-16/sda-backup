
import numpy as np
import torch

def collate_region_exclude_train_multi(batch, device, excluded_region_ids: set, Kmax=None):
    """
    TRAIN collate (multi-target): for each sample/window, targets = ALL channels in excluded region.
    Returns:
      waves_src:   (B, Csrc_max, T)
      fembs_src:   (B, Csrc_max, F)
      ch_counts:   (B,)
      pad_mask:    (B, Csrc_max)   True=PAD (sources)
      fembs_tgt:   (B, Kmax, F)
      tgt_pad_mask:(B, Kmax)       True=PAD (targets)
      y_target:    (B, Kmax, T)
    """
    waves_list  = [b[0] for b in batch]
    fembs_list  = [b[1] for b in batch]
    regs_list   = [b[2] for b in batch]

    items = []
    T = None; F = None
    for w, f, r in zip(waves_list, fembs_list, regs_list):
        if T is None: T = w.shape[1]
        if F is None: F = f.shape[1]
        r_np = r.cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
        idx_tgt_pool = np.where(np.isin(r_np, list(excluded_region_ids)))[0]
        idx_src_pool = np.where(~np.isin(r_np, list(excluded_region_ids)))[0]
        if len(idx_tgt_pool) == 0 or len(idx_src_pool) == 0:
            continue
        items.append((w, f, idx_tgt_pool, idx_src_pool))

    if len(items) == 0:
        return (
            torch.empty(0, 0, 0, dtype=torch.float32, device=device),
            torch.empty(0, 0, 0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, 0, dtype=torch.bool, device=device),
            torch.empty(0, 0, F, dtype=torch.float32, device=device),
            torch.empty(0, 0, dtype=torch.bool, device=device),
            torch.empty(0, 0, T, dtype=torch.float32, device=device),
        )

    B = len(items)
    Csrc_max = max(len(src_idx) for _, _, _, src_idx in items)
    Kmax = Kmax or max(len(tgt_idx) for _, _, tgt_idx, _ in items)

    waves_src = torch.zeros((B, Csrc_max, T), dtype=torch.float32)
    fembs_src = torch.zeros((B, Csrc_max, F), dtype=torch.float32)
    ch_counts = torch.zeros((B,), dtype=torch.long)
    pad_mask  = torch.ones((B, Csrc_max), dtype=torch.bool)

    fembs_tgt    = torch.zeros((B, Kmax, F), dtype=torch.float32)
    tgt_pad_mask = torch.ones((B, Kmax), dtype=torch.bool)
    y_target     = torch.zeros((B, Kmax, T), dtype=torch.float32)

    for i, (w, f, idx_tgt_pool, idx_src_pool) in enumerate(items):
        # sources
        Cs = len(idx_src_pool)
        waves_src[i, :Cs] = w[idx_src_pool]
        fembs_src[i, :Cs] = f[idx_src_pool]
        ch_counts[i] = Cs
        pad_mask[i, :Cs] = False
        # targets
        k = min(len(idx_tgt_pool), Kmax)
        tgt_sel = idx_tgt_pool[:k]
        y_target[i, :k]     = w[tgt_sel]
        fembs_tgt[i, :k]    = f[tgt_sel]
        tgt_pad_mask[i, :k] = False

    return (waves_src.to(device), fembs_src.to(device), ch_counts.to(device),
            pad_mask.to(device), fembs_tgt.to(device), tgt_pad_mask.to(device),
            y_target.to(device))


def collate_region_exclude_eval_multi(batch, device, excluded_region_ids: set, Kmax=None, order="maxvar"):
    """
    EVAL collate (multi-target, deterministic order).
    order: 'maxvar' (descending variance) or 'as_is'
    """
    waves_list  = [b[0] for b in batch]
    fembs_list  = [b[1] for b in batch]
    regs_list   = [b[2] for b in batch]

    items = []
    T = None; F = None
    for w, f, r in zip(waves_list, fembs_list, regs_list):
        if T is None: T = w.shape[1]
        if F is None: F = f.shape[1]
        r_np = r.cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
        idx_tgt_pool = np.where(np.isin(r_np, list(excluded_region_ids)))[0]
        idx_src_pool = np.where(~np.isin(r_np, list(excluded_region_ids)))[0]
        if len(idx_tgt_pool) == 0 or len(idx_src_pool) == 0:
            continue
        if order == "maxvar":
            vari = w.float().var(dim=1, unbiased=False).cpu().numpy()
            sort_idx = np.argsort(vari[idx_tgt_pool])[::-1]  # high→low
            idx_tgt_pool = idx_tgt_pool[sort_idx]
        items.append((w, f, idx_tgt_pool, idx_src_pool))

    if len(items) == 0:
        return (
            torch.empty(0, 0, 0, dtype=torch.float32, device=device),
            torch.empty(0, 0, 0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, 0, dtype=torch.bool, device=device),
            torch.empty(0, 0, F, dtype=torch.float32, device=device),
            torch.empty(0, 0, dtype=torch.bool, device=device),
            torch.empty(0, 0, T, dtype=torch.float32, device=device),
        )

    B = len(items)
    Csrc_max = max(len(src_idx) for _, _, _, src_idx in items)
    Kmax = Kmax or max(len(tgt_idx) for _, _, tgt_idx, _ in items)

    waves_src = torch.zeros((B, Csrc_max, T), dtype=torch.float32)
    fembs_src = torch.zeros((B, Csrc_max, F), dtype=torch.float32)
    ch_counts = torch.zeros((B,), dtype=torch.long)
    pad_mask  = torch.ones((B, Csrc_max), dtype=torch.bool)

    fembs_tgt    = torch.zeros((B, Kmax, F), dtype=torch.float32)
    tgt_pad_mask = torch.ones((B, Kmax), dtype=torch.bool)
    y_target     = torch.zeros((B, Kmax, T), dtype=torch.float32)

    for i, (w, f, idx_tgt_pool, idx_src_pool) in enumerate(items):
        # sources
        Cs = len(idx_src_pool)
        waves_src[i, :Cs] = w[idx_src_pool]
        fembs_src[i, :Cs] = f[idx_src_pool]
        ch_counts[i] = Cs
        pad_mask[i, :Cs] = False
        # targets
        k = min(len(idx_tgt_pool), Kmax)
        tgt_sel = idx_tgt_pool[:k]
        y_target[i, :k]     = w[tgt_sel]
        fembs_tgt[i, :k]    = f[tgt_sel]
        tgt_pad_mask[i, :k] = False

    return (waves_src.to(device), fembs_src.to(device), ch_counts.to(device),
            pad_mask.to(device), fembs_tgt.to(device), tgt_pad_mask.to(device),
            y_target.to(device))


def compute_excluded_region_ids(region_vocab: dict, predicate=None):
    """
    region_vocab: dict {region_name -> id} from MRCPStreamDataset.region_vocab
    predicate: callable(str)->bool; default excludes names starting with 'VO' (case-insensitive)
    returns: set of region ids to exclude
    """
    if predicate is None:
        predicate = lambda name: str(name).upper().startswith("VO")
    return {rid for name, rid in region_vocab.items() if predicate(name)}

def needs_wd(name, p):
    if p.ndim == 1: return False
    if any(k in name for k in ["embed","pos","norm","query_base","is_source","is_query"]): return False
    return True


def flatten_valid(y_hat, y_true, tgt_pad_mask):
    """
    y_hat, y_true: (B, K, Tm)
    tgt_pad_mask:  (B, K) True=PAD
    -> y_hat_v, y_true_v: (N, Tm) with only valid targets
    """
    valid = ~tgt_pad_mask  # (B,K)
    if not valid.any():
        return None, None
    y_hat = y_hat[valid]           # (N, Tm)
    y_true = y_true[valid]         # (N, Tm)
    return y_hat, y_true

@torch.no_grad()
def nmse_nrmse_r(y_hat, y_true, eps=1e-8):
    """
    y_hat, y_true: (N, T)  (already flattened to valid targets)
    Returns scalar means: nmse_mean, nrmse_mean, r_mean
    """
    # Per-sample variance
    var = y_true.var(dim=1, unbiased=False) + eps
    mse = ((y_hat - y_true) ** 2).mean(dim=1)
    nmse = mse / var
    nrmse = torch.sqrt(nmse)

    # Pearson r per sample
    yh = y_hat - y_hat.mean(dim=1, keepdim=True)
    yt = y_true - y_true.mean(dim=1, keepdim=True)
    denom = (yh.std(dim=1, keepdim=False) * yt.std(dim=1, keepdim=False)) + eps
    r = (yh * yt).mean(dim=1) / denom

    return nmse.mean().item(), nrmse.mean().item(), r.mean().item()

def corr_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8, reduction: str = "mean"):
    """
    1 - Pearson correlation per sample (over time), then reduce.
    Accepts (B,T) or (B,K,T); flattens leading dims to (N,T).

    Args:
        pred, tgt: tensors with time in the last dim.
        eps: small constant for numerical stability.
        reduction: "mean" | "sum" | "none"

    Returns:
        scalar if reduction != "none", else (N,) tensor of per-sample losses.
    """
    # Ensure same shape & dtype
    assert pred.shape == tgt.shape, "pred and tgt must have the same shape"
    pred = pred.float()
    tgt  = tgt.float()

    # Flatten any leading dims except time: (..., T) -> (N, T)
    if pred.dim() > 2:
        N, T = pred.numel() // pred.size(-1), pred.size(-1)
        pred = pred.reshape(N, T)
        tgt  = tgt.reshape(N, T)

    # Normalize per sample
    pred = pred - pred.mean(dim=1, keepdim=True)
    tgt  = tgt  - tgt.mean(dim=1, keepdim=True)
    pred_std = pred.std(dim=1, keepdim=True).clamp_min(eps)
    tgt_std  = tgt.std(dim=1, keepdim=True).clamp_min(eps)

    # Pearson r per sample
    r = (pred * tgt).mean(dim=1) / (pred_std.squeeze(1) * tgt_std.squeeze(1) + eps)
    loss_vec = 1.0 - r  # higher when less correlated

    if reduction == "mean":
        return loss_vec.mean()
    if reduction == "sum":
        return loss_vec.sum()
    return loss_vec