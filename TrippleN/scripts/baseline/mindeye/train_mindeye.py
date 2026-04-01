#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

MINDEYE_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "paper_code", "MindsEye-main", "src"
)
if os.path.isdir(MINDEYE_SRC):
    sys.path.insert(0, MINDEYE_SRC)
else:
    MINDEYE_SRC = os.path.expanduser("~/paper_code/MindsEye-main/src")
    if os.path.isdir(MINDEYE_SRC):
        sys.path.insert(0, MINDEYE_SRC)

import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, VersatileDiffusionPriorNetwork

torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description="MindEye baseline training on TrippleN")
parser.add_argument("--neuron_responses", type=str, default=None)
parser.add_argument("--stimuli_dir", type=str, default=None)
parser.add_argument("--vd_cache_dir", type=str, required=True)
parser.add_argument("--prior_ckpt_dir", type=str, default=None)
parser.add_argument("--ckpt_dir", type=str, default=None)
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=120)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mixup_pct", type=float, default=0.33)
parser.add_argument("--max_lr", type=float, default=3e-4)
parser.add_argument("--ckpt_interval", type=int, default=10)
parser.add_argument("--use_image_aug", action="store_true", default=True)
parser.add_argument("--n_samples_save", type=int, default=0)
args = parser.parse_args()

if args.ckpt_dir is None:
    args.ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(args.ckpt_dir, exist_ok=True)

utils.seed_everything(args.seed, cudnn_deterministic=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import TrippleNDataset, train_val_split, default_paths

paths = default_paths()
neuron_path = args.neuron_responses or paths["neuron_responses"]
stimuli_dir = args.stimuli_dir or paths["stimuli_dir"]

full_dataset = TrippleNDataset(
    neuron_responses_path=neuron_path,
    stimuli_dir=stimuli_dir,
    indices=None,
)
n_samples = len(full_dataset)
num_neurons = full_dataset.n_neurons
train_idx, val_idx = train_val_split(n_samples, train_ratio=args.train_ratio, seed=args.seed)
train_dataset = TrippleNDataset(
    neuron_responses_path=neuron_path,
    stimuli_dir=stimuli_dir,
    indices=train_idx,
)
val_dataset = TrippleNDataset(
    neuron_responses_path=neuron_path,
    stimuli_dir=stimuli_dir,
    indices=val_idx,
)
num_train = len(train_dataset)
num_val = len(val_dataset)

train_dl = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
val_dl = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
)

clip_size = 768
out_dim = 257 * clip_size
clip_extractor = Clipper("ViT-L/14", device=device, hidden_state=True, norm_embs=True)
voxel2clip = BrainNetwork(
    in_dim=num_neurons,
    out_dim=out_dim,
    clip_size=clip_size,
    use_projector=True,
).to(device)

depth = 6
dim_head = 64
heads = clip_size // 64
prior_network = VersatileDiffusionPriorNetwork(
    dim=clip_size,
    depth=depth,
    dim_head=dim_head,
    heads=heads,
    causal=False,
    num_tokens=257,
    learned_query_mode="pos_emb",
).to(device)
diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=clip_size,
    condition_on_text_encodings=False,
    timesteps=100,
    cond_drop_prob=0.2,
    image_embed_scale=None,
    voxel2clip=voxel2clip,
).to(device)

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
opt_grouped_parameters = [
    {"params": [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 1e-2},
    {"params": [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    {"params": [p for n, p in diffusion_prior.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 1e-2},
    {"params": [p for n, p in diffusion_prior.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)
total_steps = int(args.num_epochs * (num_train // args.batch_size))
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.max_lr,
    total_steps=total_steps,
    final_div_factor=1000,
    last_epoch=-1,
    pct_start=2 / args.num_epochs,
)

if args.use_image_aug:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
    img_augment = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224, 224), (0.6, 1), p=0.3),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        data_keys=["input"],
    ).to(device)

prior_mult = 30
soft_loss_temps = utils.cosine_anneal(
    0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs)
)
best_val_loss = float("inf")
epoch = 0


def save_ckpt(tag):
    ckpt_path = os.path.join(args.ckpt_dir, f"{tag}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": diffusion_prior.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        ckpt_path,
    )


for epoch in tqdm(range(epoch, args.num_epochs), desc="Epoch"):
    diffusion_prior.train()
    losses = []
    val_losses = []
    loss_nce_sum = 0.0
    loss_prior_sum = 0.0
    val_loss_nce_sum = 0.0
    val_loss_prior_sum = 0.0
    epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)] if epoch >= int(args.mixup_pct * args.num_epochs) else 0.006

    for train_i, (voxel, image) in enumerate(train_dl):
        voxel = voxel.to(device).float()
        image = image.to(device)

        with torch.cuda.amp.autocast():
            optimizer.zero_grad()
            if args.use_image_aug:
                image = img_augment(image)
            if epoch < int(args.mixup_pct * args.num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel)
            clip_target = clip_extractor.embed_image(image).float()
            clip_voxels, clip_voxels_proj = diffusion_prior.voxel2clip(voxel)
            clip_voxels = clip_voxels.view(len(voxel), -1, clip_size)

            loss_prior, _ = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
            clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if epoch < int(args.mixup_pct * args.num_epochs):
                loss_nce = utils.mixco_nce(
                    clip_voxels_norm, clip_target_norm, temp=0.006,
                    perm=perm, betas=betas, select=select,
                )
            else:
                loss_nce = utils.soft_clip_loss(
                    clip_voxels_norm, clip_target_norm, temp=epoch_temp,
                )
            loss_nce_sum += loss_nce.item()
            loss_prior_sum += loss_prior.item()
            loss = loss_nce + prior_mult * loss_prior
            utils.check_loss(loss)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.item())

    diffusion_prior.eval()
    with torch.no_grad():
        for val_i, (voxel, image) in enumerate(val_dl):
            voxel = voxel.to(device).float()
            image = image.to(device)
            if args.use_image_aug:
                image = img_augment(image)
            with torch.cuda.amp.autocast():
                clip_target = clip_extractor.embed_image(image).float()
                clip_voxels, clip_voxels_proj = diffusion_prior.voxel2clip(voxel)
                clip_voxels = clip_voxels.view(len(voxel), -1, clip_size)
                val_loss_prior, _ = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                val_loss_nce = utils.soft_clip_loss(
                    clip_voxels_norm, clip_target_norm, temp=epoch_temp,
                )
                val_loss_nce_sum += val_loss_nce.item()
                val_loss_prior_sum += val_loss_prior.item()
                val_loss = val_loss_nce + prior_mult * val_loss_prior
                val_losses.append(val_loss.item())

    mean_val_loss = np.mean(val_losses)
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        save_ckpt("best")
    save_ckpt("last")
    if (epoch + 1) % args.ckpt_interval == 0:
        save_ckpt("last_backup")

    tqdm.write(
        f"epoch {epoch} train_loss={np.mean(losses):.4f} val_loss={mean_val_loss:.4f} "
        f"nce={loss_nce_sum/(train_i+1):.4f} prior={loss_prior_sum/(train_i+1):.4f}"
    )

print("Training finished.")
