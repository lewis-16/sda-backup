#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="MindEye reconstruction on TrippleN")
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--vd_cache_dir", type=str, required=True)
parser.add_argument("--neuron_responses", type=str, default=None)
parser.add_argument("--stimuli_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--recons_per_sample", type=int, default=4)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--guidance_scale", type=float, default=3.5)
parser.add_argument("--save_images", action="store_true")
args = parser.parse_args()

from dataset import TrippleNDataset, train_val_split, default_paths

paths = default_paths()
neuron_path = args.neuron_responses or paths["neuron_responses"]
stimuli_dir = args.stimuli_dir or paths["stimuli_dir"]
if args.output_dir is None:
    args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reconstructions")
os.makedirs(args.output_dir, exist_ok=True)

utils.seed_everything(args.seed)
num_neurons = 15652
clip_size = 768
out_dim = 257 * clip_size

clip_extractor = Clipper("ViT-L/14", device=device, hidden_state=True, norm_embs=True)
voxel2clip = BrainNetwork(
    in_dim=num_neurons,
    out_dim=out_dim,
    clip_size=clip_size,
    use_projector=True,
).to(device)
prior_network = VersatileDiffusionPriorNetwork(
    dim=clip_size,
    depth=6,
    dim_head=64,
    heads=clip_size // 64,
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

ckpt = torch.load(args.ckpt_path, map_location=device)
diffusion_prior.load_state_dict(ckpt["model_state_dict"], strict=False)
diffusion_prior.eval()

from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel

try:
    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(args.vd_cache_dir).to(device).to(torch.float16)
except Exception:
    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
        "shi-labs/versatile-diffusion",
        cache_dir=args.vd_cache_dir,
    ).to(device).to(torch.float16)
vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(args.vd_cache_dir, subfolder="scheduler")
text_image_ratio = 0.0
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = text_image_ratio
        module.condition_lengths[0] = 257
        module.condition_lengths[1] = 77
        module.transformer_index_for_condition[0] = 0
        module.transformer_index_for_condition[1] = 1
unet = vd_pipe.image_unet
vae = vd_pipe.vae
noise_scheduler = vd_pipe.scheduler

full_dataset = TrippleNDataset(neuron_responses_path=neuron_path, stimuli_dir=stimuli_dir, indices=None)
train_idx, val_idx = train_val_split(len(full_dataset), train_ratio=args.train_ratio, seed=args.seed)
indices = val_idx if args.split == "val" else train_idx
dataset = TrippleNDataset(neuron_responses_path=neuron_path, stimuli_dir=stimuli_dir, indices=indices)

all_recons = []
all_images = []

for i in tqdm(range(len(dataset)), desc="Reconstruct"):
    voxel, image = dataset[i]
    voxel = voxel.unsqueeze(0).to(device).float()
    image_batch = image.unsqueeze(0).to(device)
    image_01 = (image_batch / 2 + 0.5).clamp(0, 1)

    with torch.no_grad():
        grid, brain_recons, best_picks, _ = utils.reconstruction(
            image_01,
            voxel,
            clip_extractor,
            unet,
            vae,
            noise_scheduler,
            diffusion_priors=[diffusion_prior],
            num_inference_steps=args.num_inference_steps,
            n_samples_save=1,
            recons_per_sample=args.recons_per_sample,
            guidance_scale=args.guidance_scale,
            timesteps_prior=100,
            seed=args.seed + i,
            retrieve=False,
            plotting=False,
            img_variations=False,
        )
    best_idx = int(best_picks[0])
    recon = brain_recons[0, best_idx].unsqueeze(0)
    all_recons.append(recon.cpu())
    all_images.append(image_01.cpu())

all_recons = torch.cat(all_recons, dim=0)
all_images = torch.cat(all_images, dim=0)

out_pt = os.path.join(args.output_dir, "recons.pt")
torch.save({"recons": all_recons, "images": all_images}, out_pt)
print(f"Saved {out_pt}")

if args.save_images:
    from torchvision.utils import save_image
    recons_dir = os.path.join(args.output_dir, "recons")
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(recons_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    for i in range(all_recons.shape[0]):
        save_image(all_recons[i], os.path.join(recons_dir, f"{i:04d}.png"))
        save_image(all_images[i], os.path.join(images_dir, f"{i:04d}.png"))
    print(f"Saved PNGs to {recons_dir} and {images_dir}")
