import matplotlib.pyplot as plt
import numpy as np
import torch
import random, itertools

@torch.no_grad()
def plot_query_example_region_exclude_multi(
    model,
    loader,
    fs,
    device,
    sample_idx: int = 0,
    max_k: int = None,
    title: str = "Region-excluded multi-target reconstruction",
    random_batch: bool = False,
):
    """
    Works with BOTH:
      - single-target batches: (waves_src, fembs_src, ch_counts, pad_mask, femb_tgt, y_true)
      - multi-target batches:  (waves_src, fembs_src, ch_counts, pad_mask, fembs_tgt, tgt_pad_mask, y_true)

    Plots True vs Pred for all valid targets in one sample (indexed by sample_idx),
    each on its own subplot. Displays NMSE and Pearson r per target in the legend.
    """
    model.eval()
    if random_batch:
        
        nb = len(loader)
        skip = random.randrange(nb) if nb > 0 else 0
        it = iter(loader)
        batch = next(it, None)
        for _ in range(skip):
            nxt = next(it, None)
            if nxt is None: break
            batch = nxt
    else:
        batch = next(iter(loader))
    if not isinstance(batch, (list, tuple)) or len(batch) < 6:
        print("Batch format not recognized.")
        return

    # Empty/too-short guard
    waves_src = batch[0]
    if waves_src.numel() == 0:
        print("No valid samples in this batch (excluded region missing).")
        return
    if waves_src.shape[-1] < getattr(model, "patch_len", 1):
        print("Batch windows shorter than model.patch_len—nothing to plot.")
        return

    # Unpack and forward
    if len(batch) == 6:
        # Single-target path
        waves_src, fembs_src, ch_counts, pad_mask, femb_tgt, y_true = batch
        y_hat = model(waves_src, fembs_src, pad_mask, femb_tgt)  # (B, Tm)
        B, Tm = y_hat.shape
        if sample_idx >= B:
            print(f"sample_idx={sample_idx} out of range (B={B}).")
            return
        t_ms = np.arange(Tm) * (1000.0 / fs)

        plt.figure(figsize=(8, 3))
        yt = y_true[sample_idx, :Tm].detach().cpu().numpy()
        yp = y_hat[sample_idx].detach().cpu().numpy()

        # Metrics
        yt_t = torch.tensor(yt).unsqueeze(0)
        yp_t = torch.tensor(yp).unsqueeze(0)
        var = yt_t.var(dim=1, unbiased=False) + 1e-8
        mse = ((yp_t - yt_t) ** 2).mean(dim=1)
        nmse = (mse / var).item()
        # Pearson r
        yh = yp_t - yp_t.mean(dim=1, keepdim=True)
        yt0 = yt_t - yt_t.mean(dim=1, keepdim=True)
        r = ((yh * yt0).mean(dim=1) / (yh.std(dim=1) * yt0.std(dim=1) + 1e-8)).item()

        plt.plot(t_ms, yt, label=f"True", linewidth=1.1)
        plt.plot(t_ms, yp, label=f"Pred (NMSE={nmse:.3f}, r={r:.3f})", linewidth=1.1)
        plt.xlabel("Time (ms)"); plt.ylabel("Amplitude (z)")
        plt.title(title + " (single target)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
        return

    # Multi-target path
    waves_src, fembs_src, ch_counts, pad_mask, fembs_tgt, tgt_pad_mask, y_true = batch
    y_hat = model(waves_src, fembs_src, pad_mask, fembs_tgt, tgt_pad_mask)  # (B, K, Tm)

    B, K, Tm = y_hat.shape
    if sample_idx >= B:
        print(f"sample_idx={sample_idx} out of range (B={B}).")
        return
    t_ms = np.arange(Tm) * (1000.0 / fs)

    # Pick valid target indices for this sample
    valid_mask = (~tgt_pad_mask[sample_idx]).detach().cpu()
    valid_idx = torch.where(valid_mask)[0].tolist()
    if len(valid_idx) == 0:
        print("No valid targets for this sample.")
        return
    if max_k is not None:
        valid_idx = valid_idx[:max_k]

    nplots = len(valid_idx)
    fig, axes = plt.subplots(nplots, 1, figsize=(9, 2.6 * nplots), sharex=True)
    if nplots == 1:
        axes = [axes]

    for ax, k in zip(axes, valid_idx):
        yt = y_true[sample_idx, k, :Tm].detach().cpu().numpy()
        yp = y_hat[sample_idx, k].detach().cpu().numpy()

        # Metrics per target
        yt_t = torch.tensor(yt).unsqueeze(0)
        yp_t = torch.tensor(yp).unsqueeze(0)
        var = yt_t.var(dim=1, unbiased=False) + 1e-8
        mse = ((yp_t - yt_t) ** 2).mean(dim=1)
        nmse = (mse / var).item()
        yh = yp_t - yp_t.mean(dim=1, keepdim=True)
        yt0 = yt_t - yt_t.mean(dim=1, keepdim=True)
        r = ((yh * yt0).mean(dim=1) / (yh.std(dim=1) * yt0.std(dim=1) + 1e-8)).item()

        ax.plot(t_ms, yt, label="True", linewidth=1.0)
        ax.plot(t_ms, yp, label=f"Pred (NMSE={nmse:.3f}, r={r:.3f})", linewidth=1.0)
        ax.set_ylabel("z")
        ax.legend(frameon=False)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"{title} | sample={sample_idx} | K={nplots}", y=1.02, fontsize=11)
    plt.tight_layout()
    plt.show()