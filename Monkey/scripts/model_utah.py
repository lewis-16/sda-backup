
import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms

import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms


class IntegratedNeuronDataset(Dataset):
    def __init__(self, neuro_data, labels, image_source):

        self.neuro_data = neuro_data
        self.labels = labels
        
        self.image_paths = image_source

        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  
        ])
        
        self.empty_image_tensor = torch.zeros(3, 256, 256, dtype=torch.float32)
    
    def __len__(self):
        return len(self.neuro_data)
    
    def __getitem__(self, idx):
        neuro_tensor = torch.tensor(self.neuro_data[idx], dtype=torch.float32).T
        
        image_label = self.labels[idx]
        
        image_path = self.image_paths[idx]
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((256, 256))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            img_tensor = img_tensor * 2 - 1
        except (FileNotFoundError, OSError) as e:
            img_tensor = self.empty_image_tensor.clone()
            print(f"Warning: {image_path} not found or corrupted: {e}")
        
        return neuro_tensor, img_tensor, image_label

def load_firing_rate_data(firing_rate_path, train=True):
    with open(firing_rate_path, 'rb')as f:
        firing_rate = pickle.load(f)

    firing_rate_list = []
    label = []
    
    if train:
        split_str = 'train'
    else:
        split_str = 'test'
    
    for key in firing_rate.keys():
        if split_str in key:
            firing_rate_data = firing_rate[key]['firing_rate']
            if len(firing_rate_data.shape) == 2 and firing_rate_data.shape[0] == 244 and firing_rate_data.shape[1] == 20:
                label.append(firing_rate[key]['image_id'])
                firing_rate_list.append(firing_rate_data)
            elif len(firing_rate_data.shape) == 2 and firing_rate_data.shape[0] == 244 and firing_rate_data.shape[1] == 21:
                label.append(firing_rate[key]['image_id'])
                firing_rate_list.append(firing_rate_data[:, :-1])
    
    print(f"Loaded {len(firing_rate_list)} {split_str} trials")
    return firing_rate_list, label


def load_image_data(csv_path):
    df = pd.read_csv(csv_path)

    image_labels = np.array(df.index) + 1
    image_paths = df['local_path'].values
    image_class = df['class'].values
    
    print(f"Loaded {len(image_labels)} images")
    
    return image_labels, image_paths, image_class


def create_integrated_datasets(firing_rate_path=None, csv_path=None,
                             test_size=0.2, random_state=42):
    
    neuro_data, labels = load_firing_rate_data(firing_rate_path, train=False)
    image_labels, image_paths, _ = load_image_data(csv_path)
    image_source = []

    for label in labels:
        image_source.append(image_paths[np.where(image_labels == int(label))[0]][0])
    
    train_indices, val_indices = train_test_split(
        range(len(labels)), test_size=test_size, random_state=random_state
    )
    
    train_dataset = IntegratedNeuronDataset(
        neuro_data=[neuro_data[i] for i in train_indices],
        labels=[labels[i] for i in train_indices],
        image_source=[image_source[i] for i in train_indices],
    )
    
    val_dataset = IntegratedNeuronDataset(
        neuro_data=[neuro_data[i] for i in val_indices],
        labels=[labels[i] for i in val_indices],
        image_source=[image_source[i] for i in val_indices],
    )
    
    return train_dataset, val_dataset


firing_rate_path = "./firing_rate_summary_0112.pkl"
csv_path = "/media/ubuntu/sda/Monkey/scripts/test_image.csv"

train_dataset_csv, val_dataset_csv = create_integrated_datasets(
    firing_rate_path=firing_rate_path, 
    csv_path=csv_path
)

print(f"CSV Mode - Train set: {len(train_dataset_csv)}")
print(f"CSV Mode - Val set: {len(val_dataset_csv)}")

import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
from models.EP_encoder import ModelConfig


ep_config = ModelConfig(
        input_neuron=train_dataset_csv.neuro_data[0].shape[0],  # 244个神经元
        time_bins=train_dataset_csv.neuro_data[0].shape[1],    # 20个时间bin
        d_model=150,
        nhead=10,
        num_transformer_layers=2, 
        conv_channels=64,
        num_conv_blocks=3,
        num_classes=117,
        residual_dims=[256, 512, 1024],
        use_positional_encoding=True,
        dim_feedforward_ratio=4,
        activation='relu',
        lr=1e-5,  
        epochs=50
    )

MODEL_DEPTH = 16   


# download checkpoint
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'

FOR_512_px = MODEL_DEPTH == 16

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,   
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=FOR_512_px,
    config= ep_config
)

vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)

class TrainingConfig:
    batch_size = 8
    num_workers = 0
    image_size = 256
    dataset_path = "/disk1/jinchentao/visual_decode/visual_reconstruction"
    
    # 多GPU配置
    distributed = True
    backend = 'nccl' 
    init_method = 'env://'
    
    vae_config = {
        "in_channel": 3,
        "channel": 128,
        "n_res_block": 2,
        "n_res_channel": 64,
        "embed_dim": 64,
        "n_embed": 8192,
        "decay": 0.99
    }
    var_config = {
        "num_classes": 1000,
        "depth": 16,
        "embed_dim": 1024,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "norm_eps": 1e-6,
        "shared_aln": True,
        "cond_drop_rate": 0.1,
        "attn_l2_norm": False,
        "patch_nums": (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        "flash_if_available": True,
        "fused_if_available": True,
    }
    
    lr = 1e-4
    weight_decay = 0.05
    betas = (0.9, 0.95)
    
    epochs = 60
    grad_accum = 1
    label_smooth = 0.1
    amp_enabled = True
    
    prog_epochs = 5 
    prog_warmup_iters = 1000  
    
    log_interval = 50
    eval_interval = 1
    save_interval = 5
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

class VARTrainer(object):
    def __init__(
        self, device, patch_nums, resos,
        vae_local, var_model,
        optimizer: torch.optim.Optimizer, label_smooth: float,
        amp_enabled: bool = False, rank: int = 0
    ):
        super(VARTrainer, self).__init__()
        
        self.var_model = var_model
        self.vae_local = vae_local
        self.quantize_local = vae_local.quantize
        self.optimizer = optimizer
        self.amp_enabled = amp_enabled
        self.device = device
        self.rank = rank
        
        if hasattr(self.var_model, 'rng'):
            self.var_model.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        
        # 使用新的GradScaler API
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_model.training
        self.var_model.eval()
        
        for neuron_activity, inp_B3HW, _  in ld_val:
            B, V = neuron_activity.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(self.device)
            neuron_activity = neuron_activity.to(self.device)
            
            # 修复：检查模型是否有img_to_idxBl方法
            if hasattr(self.vae_local, 'img_to_idxBl'):
                gt_idx_Bl: List[torch.Tensor] = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            else:
                # 如果没有img_to_idxBl方法，使用简化的处理
                gt_BL = torch.zeros(B, self.L, dtype=torch.long, device=self.device)
                x_BLCv_wo_first_l = torch.zeros(B, self.L, 32, device=self.device)
            
            logits_BLV = self.var_model(neuron_activity, x_BLCv_wo_first_l)
            
            L_mean += self.val_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).item() * B
            L_tail += self.val_loss(
                logits_BLV[:, -self.last_l:].reshape(-1, V), 
                gt_BL[:, -self.last_l:].reshape(-1)
            ).item() * B
            acc_mean += (logits_BLV.argmax(dim=-1) == gt_BL).float().mean().item() * 100 * B
            acc_tail += (
                logits_BLV[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]
            ).float().mean().item() * 100 * B
            tot += B
        
        self.var_model.train(training)
        
        L_mean /= tot
        L_tail /= tot
        acc_mean /= tot
        acc_tail /= tot
        
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt

    def train_step(
        self, it: int, g_it: int, stepping: bool,
        inp_B3HW: torch.Tensor, neuron_activity: torch.Tensor, prog_si: int, prog_wp_it: float
    ):
        # 修复：检查模型是否有prog_si属性
        if hasattr(self.var_model, 'prog_si'):
            self.var_model.prog_si = prog_si
        if hasattr(self.vae_local.quantize, 'prog_si'):
            self.vae_local.quantize.prog_si = prog_si
            
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: 
                self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: 
            prog_wp = 1
        if prog_si == len(self.patch_nums) - 1: 
            prog_si = -1 

        B, V = neuron_activity.shape[0], self.vae_local.vocab_size
        
        # 使用新的autocast API
        with torch.amp.autocast('cuda', enabled=False):
            # 修复：检查模型是否有img_to_idxBl方法
            if hasattr(self.vae_local, 'img_to_idxBl'):
                gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            else:
                # 如果没有img_to_idxBl方法，使用简化的处理
                gt_BL = torch.zeros(B, self.L, dtype=torch.long, device=self.device)
                x_BLCv_wo_first_l = torch.zeros(B, self.L, 32, device=self.device)
            
            logits_BLV = self.var_model(neuron_activity, x_BLCv_wo_first_l)

            pred_BL = logits_BLV.argmax(dim=-1)
            accuracy = (pred_BL == gt_BL).float().mean().item() * 100
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= prog_wp
            else:
                lw = self.loss_weight
                
            loss = loss.mul(lw).sum(dim=-1).mean()

        self.scaler.scale(loss).backward()
        
        grad_norm = 0
        if stepping:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.var_model.parameters(), max_norm=1.0
            ).item()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        # 修复：检查模型是否有prog_si属性
        if hasattr(self.var_model, 'prog_si'):
            self.var_model.prog_si = -1
        if hasattr(self.vae_local.quantize, 'prog_si'):
            self.vae_local.quantize.prog_si = -1
            
        return loss.item(), accuracy, grad_norm, self.scaler.get_scale()


def train_single_gpu(config, vae, var_model, train_dataset, val_dataset):
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    
    optimizer = torch.optim.AdamW(
        var_model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
    trainer = VARTrainer(
        device=config.device,
        patch_nums=config.var_config["patch_nums"],
        resos=(16, 32, 48, 64, 80, 96, 128, 160, 208, 256),
        vae_local=vae,
        var_model=var,
        optimizer=optimizer,
        label_smooth=config.label_smooth,
        amp_enabled=config.amp_enabled
    )
    
    
    global_step = 0
    for epoch in range(config.epochs):
        var_model.train()        
        prog_si = min(epoch // config.prog_epochs + 1, len(config.var_config["patch_nums"]) - 1)
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_samples = 0
        
        for i, (neuron_activity, images, _) in enumerate(train_loader):
            images = images.to(config.device, non_blocking=True)
            neuron_activity = neuron_activity.to(config.device, non_blocking=True)
            
            stepping = (i + 1) % config.grad_accum == 0
            
            loss_value, accuracy, grad_norm, scale = trainer.train_step(
                it=i,
                g_it=global_step,
                stepping=stepping,
                inp_B3HW=images,
                neuron_activity=neuron_activity,
                prog_si=prog_si,
                prog_wp_it=config.prog_warmup_iters
            )
            
            batch_size = images.size(0)
            epoch_loss += loss_value * batch_size
            epoch_acc += accuracy * batch_size
            epoch_samples += batch_size

            global_step += 1
        
        avg_epoch_loss = epoch_loss / epoch_samples
        avg_epoch_acc = epoch_acc / epoch_samples
        
        print(f"\nEpoch {epoch}/{config.epochs} Training: "
              f"Loss = {avg_epoch_loss:.4f} "
              f"Acc = {avg_epoch_acc:.2f}%")

        if epoch % config.eval_interval == 0:
            var_model.eval()
            L_mean, L_tail, acc_mean, acc_tail, tot, eval_time = trainer.eval_ep(val_loader)
            print(f"\nEpoch {epoch} Validation: "
                  f"Loss = {L_mean:.4f}/{L_tail:.4f} "
                  f"Acc = {acc_mean:.2f}%/{acc_tail:.2f}% "
                  f"Time = {eval_time:.1f}s\n")
            
        
        # if epoch % config.save_interval == 0:
        #     checkpoint = {
        #         "epoch": epoch,
        #         "model": trainer.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "scaler": trainer.scaler.state_dict()
        #     }
        #     torch.save(checkpoint, f"{config.checkpoint_dir}/ckpt_epoch{epoch}.pth")
        #     print(f"Saved checkpoint at epoch {epoch}")




config = TrainingConfig()

train_single_gpu(config, vae=vae, var_model=var, train_dataset=train_dataset_csv, val_dataset=val_dataset_csv)
