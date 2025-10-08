from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import gc
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
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import random
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var


filtered_test_MUA = np.load("/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/filtered_test_MUA_MonkeyN.npy")
filtered_labels = np.load("/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/filtered_labels_MonkeyN.npy")

# 添加图像路径加载
def load_image_paths_for_utah():
    """
    为Utah数据集加载对应的图像路径
    返回图像路径列表，与filtered_labels对应
    """
    # 读取CSV文件获取图像路径
    csv_path = "/disk1/jinchentao/visual_decode/visual_reconstruction/dataset/test_image_MonkeyN.csv"
    df = pd.read_csv(csv_path)
    
    # 创建图像路径列表，与filtered_labels对应
    image_paths = []
    for label in filtered_labels:
        if label < len(df):
            image_path = df.iloc[label]['local_path']
            image_paths.append(image_path)
        else:
            # 如果标签超出范围，使用空路径
            image_paths.append("")
    
    return image_paths

# 加载图像路径
image_paths = load_image_paths_for_utah()
print(f"加载了 {len(image_paths)} 个图像路径")

# 检查数据维度
print(f"filtered_test_MUA shape: {filtered_test_MUA.shape}")
print(f"filtered_labels shape: {filtered_labels.shape}")
print(f"unique labels: {len(np.unique(filtered_labels))}")


class MUAClassificationDataset(Dataset):
    def __init__(self, mua_data, labels, image_paths=None, transform=None):
        self.mua_data = torch.tensor(mua_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.image_paths = image_paths
        self.transform = transform
        
        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到[-1, 1]
        ])
        
        # 空图像张量作为备用
        self.empty_image_tensor = torch.zeros(3, 256, 256, dtype=torch.float32)
        
    def __len__(self):
        return len(self.mua_data)
    
    def __getitem__(self, idx):
        mua_sample = self.mua_data[idx]
        label = self.labels[idx]
        
        # 处理图像
        if self.image_paths is not None and idx < len(self.image_paths):
            image_path = self.image_paths[idx]
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((256, 256))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                img_tensor = img_tensor * 2 - 1  # 归一化到[-1, 1]
            except (FileNotFoundError, OSError) as e:
                img_tensor = self.empty_image_tensor.clone()
                print(f"Warning: {image_path} not found or corrupted: {e}")
        else:
            img_tensor = self.empty_image_tensor.clone()
        
        if self.transform:
            mua_sample = self.transform(mua_sample)
            
        return mua_sample, img_tensor, label

def create_label_mapping(original_labels, num_classes=91):

    unique_labels = np.unique(original_labels)
    
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    mapped_labels = np.array([label_mapping[label] for label in original_labels])
    
    
    return mapped_labels, label_mapping

mapped_labels, label_mapping = create_label_mapping(filtered_labels, num_classes=91)

train_indices, test_indices = train_test_split(
    range(len(filtered_test_MUA)), 
    test_size=0.2, 
    random_state=42, 
    stratify=mapped_labels
)

train_dataset = MUAClassificationDataset(
    filtered_test_MUA[train_indices], 
    mapped_labels[train_indices],
    image_paths=[image_paths[i] for i in train_indices]
)

test_dataset = MUAClassificationDataset(
    filtered_test_MUA[test_indices], 
    mapped_labels[test_indices],
    image_paths=[image_paths[i] for i in test_indices]
)


MODEL_DEPTH = 16   

actual_input_dim = filtered_test_MUA.shape[1]  # 神经元数量
num_classes = len(np.unique(filtered_labels))  # 实际类别数量

print(f"EP配置 - 输入维度: {actual_input_dim}, 类别数量: {num_classes}")

vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'

FOR_512_px = MODEL_DEPTH == 16

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,   
    device=device, patch_nums=patch_nums, input_dim = 669, num_classes_ep = 98,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=FOR_512_px
)

vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)

class TrainingConfig:
    batch_size = 8
    num_workers = 0
    image_size = 256
    
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
    
    epochs = 55
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
            
            # 强制使用VAE的img_to_idxBl方法，不使用零张量
            try:
                gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            except Exception as e:
                print(f"Error in VAE img_to_idxBl: {e}")
                # 如果VAE方法失败，跳过这个batch
                continue
            
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
            # 强制使用VAE的img_to_idxBl方法，不使用零张量
            try:
                gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            except Exception as e:
                print(f"Error in VAE img_to_idxBl during training: {e}")
                # 如果VAE方法失败，跳过这个batch
                return 0.0, 0.0, 0.0, 1.0
            
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

train_single_gpu(config, vae=vae, var_model=var, train_dataset=train_dataset, val_dataset=test_dataset)

torch.save(var, 'var_20250914.pth')

val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

vae.eval()
var.eval()
pdf_path = "/disk1/jinchentao/visual_decode/visual_reconstruction/my_VAR_final/reconstruction_results_utah_MonkeyN.pdf"
with PdfPages(pdf_path) as pdf:
    for i, (neuron_activity, images, labels) in enumerate(val_loader):
        neuron_activity = neuron_activity.to(device)
        images = images.to(device)
        label_B = neuron_activity
        B = len(label_B)
        cfg = 5
        recon_B3HW = var.autoregressive_infer_cfg(B=B, neuron_activity=neuron_activity, cfg=cfg, top_k=1000, top_p=0.99, more_smooth=True)
        
        for i in range(B):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            # 显示原始RGB图像
            original_img = images[i].cpu().detach().numpy()
            original_img = (original_img + 1) / 2  # 从[-1,1]转换到[0,1]
            original_img = np.transpose(original_img, (1, 2, 0))  # 从CHW转换为HWC
            axes[0].imshow(original_img)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            # 显示重建的RGB图像
            recon_img = recon_B3HW[i].cpu().detach().numpy()
            recon_img = (recon_img + 1) / 2  # 从[-1,1]转换到[0,1]
            recon_img = np.transpose(recon_img, (1, 2, 0))  # 从CHW转换为HWC
            axes[1].imshow(recon_img)
            axes[1].set_title("Reconstructed")
            axes[1].axis('off')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)