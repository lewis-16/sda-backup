import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import Dataset


def supcon_with_nn_loss(z1, z2, y, temperature=0.1, add_nn_positives=True, eps=1e-8):
    """
    z1, z2: (B, D) normalized
    y:  (B,)  integer region labels
    Returns scalar SupCon loss with multi-positives (same label across views)
    and optional NN-positive mining within the current batch.
    """
    # z = torch.cat([z1, z2], dim=0)             # (2B, D)
    # y = torch.cat([y, y], dim=0)               # (2B,)

    # augmented view is not used here
    z= z1.clone()
    y= y.clone()

    B2 = z.size(0)

    # cosine similarity matrix
    sim = z @ z.t()                            # (2B, 2B)
    # mask self-similarity
    self_mask = torch.eye(B2, dtype=torch.bool, device=z.device)
    sim = sim / temperature

    # supervised positives: same label
    labels_equal = y.unsqueeze(0) == y.unsqueeze(1)  # (2B, 2B)
    pos_mask = labels_equal & (~self_mask)

    # optional NN positives (NNCLR-style, mined in-batch)
    # For each row i, add argmax_j sim(i,j), j != i
    if add_nn_positives:
        with torch.no_grad():
            sim_detached = sim.clone().detach()
            sim_detached[self_mask] = -1e9
            nn_idx = torch.argmax(sim_detached, dim=1)  # (2B,)
        extra = torch.zeros_like(pos_mask)
        extra[torch.arange(B2, device=z.device), nn_idx] = True
        pos_mask = pos_mask | extra

    # For each anchor i: loss_i = -log( sum_{p} exp(sim_i,p) / sum_{a!=i} exp(sim_i,a) )
    exp_sim = torch.exp(sim)                                 # (2B,2B)
    # numerator: sum over positives
    numerator = (exp_sim * pos_mask).sum(dim=1) + eps
    # denominator: sum over all except self
    denominator = (exp_sim * (~self_mask)).sum(dim=1) + eps
    loss = -torch.log(numerator / denominator)
    # Only anchors that actually have at least one positive contribute
    valid = pos_mask.sum(dim=1) > 0
    loss = loss[valid].mean()
    return loss


def uniformity_loss(z_norm, t=2.0, eps=1e-8):
    """
    z_norm: (B, D) L2-normalized
    Minimizes log E_{i<j} exp(-t ||zi - zj||^2)
    """
    pdist2 = torch.pdist(z_norm, p=2).pow(2)                # [B*(B-1)/2]
    return torch.log(torch.exp(-t * pdist2).mean() + eps)

def vicreg_var_cov(z_raw, var_target=1.0, eps=1e-4):
    """
    z_raw: (B, D) pre-normalized projections
    Returns: var_loss, cov_loss
    """
    zc = z_raw - z_raw.mean(dim=0, keepdim=True)
    # per-dim std
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + eps)   # (D,)
    # encourage std >= var_target
    var_loss = F.relu(var_target - std).mean()

    # decorrelate (off-diagonal covariance -> 0)
    B, D = zc.shape
    cov = (zc.T @ zc) / (B - 1)                             # (D, D)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / D                    # normalized scale
    return var_loss, cov_loss

def intra_class_variance_loss(z, y):
    """
    z: (B, D) L2-normalized embeddings
    y: (B,) long region labels
    returns mean per-class variance (pulls same-class points together)
    """
    loss, groups = 0.0, 0
    for r in y.unique():
        mask = (y == r)
        if mask.sum() < 2:
            continue
        zr = z[mask]
        mr = zr.mean(dim=0, keepdim=True)         # class mean in batch
        loss += ((zr - mr)**2).sum(dim=1).mean()  # L2 variance
        groups += 1
    return loss / max(groups, 1)


def train_supcon(model, dataloader, optimizer, device, writer, epoch, temperature=0.1, add_nn=True):
    model.train()
    # print("Model training mode:", model.training)
    # print("Encoder training mode:", model.encoder.training)
    running = 0.0
    for i, (x1, x2, y_region) in enumerate(dataloader):
        x1, x2, y_region = x1.to(device), x2.to(device), y_region.to(device)
        optimizer.zero_grad()

        z1 = model(x1)  # (B,D)
        z2 = model(x2)

        ## adding additional loss for subject spread
        z_cat = torch.cat([z1, z2], dim=0)
        y_cat = torch.cat([y_region, y_region], dim=0)
        loss_sup = supcon_with_nn_loss(z1, z2, y_region, temperature=temperature, add_nn_positives=False)
        loss_var = intra_class_variance_loss(z1, y_region) # augmented vew not used here
        loss = loss_sup + 0.05 * loss_var      # start with 0.1–0.3

        loss.backward()
        optimizer.step()

        running += loss.item()
        writer.add_scalar('Train/Batch_Loss', loss.item(), epoch * len(dataloader) + i)

    avg = running / len(dataloader)
    writer.add_scalar('Train/Epoch_Loss', avg, epoch)
    print(f"Epoch {epoch+1} SupCon Train Loss: {avg:.4f}")
    return avg


@torch.no_grad()
def evaluate_pairs_with_encoder(model, pair_loader, device, writer=None, epoch=0, threshold=0.5):
    """
    Reuse PSC pair-based validation for better comparison: embed both, compute distances, and do the same thresholding accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    for x1, x2, label in pair_loader:
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
        e1 = model.encoder.embed(x1)
        e2 = model.encoder.embed(x2)
        dist = (e1 - e2).norm(p=2, dim=1)
        # use PSC contrastive loss for reporting (optional)
        loss_sim = (1 - label) * dist.pow(2)
        loss_dis = label * F.relu(1 - dist).pow(2)
        loss = (loss_sim + loss_dis).mean()
        total_loss += loss.item()
        preds = (dist > threshold).float()
        correct += (preds == label).sum().item()
        total += label.size(0)
    avg_loss = total_loss / len(pair_loader)
    acc = correct / total if total > 0 else 0.0
    if writer is not None:
        writer.add_scalar('Test/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('Test/Accuracy', acc, epoch)
    print(f"Epoch {epoch+1} PairEval Acc: {acc:.4f}  (loss {avg_loss:.4f})")
    return avg_loss, acc


def training_validate_loop_supcon(model, train_loader, pair_val_loader, device, run_dir, epochs=100, lr=1e-3,
                                  temperature=0.1, add_nn=True):
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5, min_lr=1e-5)

    model_save_path = os.path.join(run_dir, "supcon_model_best.pt")
    best_val_acc = 0
    patience = 1000
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_supcon(model, train_loader, optimizer, device, writer, epoch,
                                  temperature=temperature, add_nn=add_nn)  # keep False for single-subject

        val_loss, val_acc = evaluate_pairs_with_encoder(model, pair_val_loader, device, writer, epoch)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate', current_lr, epoch)
        print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f} - Train: {train_loss:.4f} - Val: {val_loss:.4f} (acc {val_acc:.3f})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, model_save_path)
            print(f"  New best model saved at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f" Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    writer.close()
    return torch.load(model_save_path, weights_only=False)


def zscore_transform(x, eps=1e-8):
    x = x.astype(np.float32)
    s = x.std()
    return (x - x.mean()) / (s if s > eps else eps)