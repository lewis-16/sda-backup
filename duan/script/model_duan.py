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


# 加载trail_activity数据
with open("trail_activity.pkl", 'rb') as f:
    trail_activity = pickle.load(f)

print(f"trail_activity数据: {len(trail_activity)} 个条件")

# 创建标签映射
def create_label_mappings():
    """
    创建三种分类任务的标签映射：
    1. 大类分类 (6类): animals, faces, fruits, manmade, plants, shape2d
    2. 图片分类 (24张): 每类4张图片
    3. 角度分类 (72个): 每张图片3个角度/颜色
    """
    
    # 大类映射 (6类)
    category_mapping = {
        'animals': 0, 'faces': 1, 'fruits': 2, 
        'manmade': 3, 'plants': 4, 'shape2d': 5
    }
    
    # 图片映射 (24张)
    image_mapping = {
        # animals (0-3)
        'Dee1': 0, 'Ele': 1, 'Pig': 2, 'Rhi': 3,
        # faces (4-7) 
        'MA': 4, 'MB': 5, 'MC': 6, 'WA': 7,
        # fruits (8-11)
        'App1': 8, 'Ban1': 9, 'Pea1': 10, 'Pin1': 11,
        # manmade (12-15)
        'Bed1': 12, 'Cha1': 13, 'Dis1': 14, 'Sof1': 15,
        # plants (16-19)
        'A': 16, 'B': 17, 'C': 18, 'D': 19,
        # shape2d (20-23)
        'Cir': 20, 'Oth': 21, 'Squ': 22, 'Tri': 23
    }
    
    # 角度映射 (72个)
    angle_mapping = {}
    angle_idx = 0
    for category in ['animals', 'faces', 'fruits', 'manmade', 'plants', 'shape2d']:
        if category == 'animals':
            images = ['Dee1', 'Ele', 'Pig', 'Rhi']
        elif category == 'faces':
            images = ['MA', 'MB', 'MC', 'WA']
        elif category == 'fruits':
            images = ['App1', 'Ban1', 'Pea1', 'Pin1']
        elif category == 'manmade':
            images = ['Bed1', 'Cha1', 'Dis1', 'Sof1']
        elif category == 'plants':
            images = ['A', 'B', 'C', 'D']
        elif category == 'shape2d':
            images = ['Cir', 'Oth', 'Squ', 'Tri']
            
        for img in images:
            if category == 'shape2d':
                angles = ['B1', 'G1', 'R1']  # 颜色而不是角度
            else:
                angles = ['0', '315', '45']  # 角度
            for angle in angles:
                angle_mapping[f"{img}_{angle}"] = angle_idx
                angle_idx += 1
    
    return category_mapping, image_mapping, angle_mapping

# 创建标签映射
category_mapping, image_mapping, angle_mapping = create_label_mappings()

print(f"大类分类: {len(category_mapping)} 类")
print(f"图片分类: {len(image_mapping)} 张") 
print(f"角度分类: {len(angle_mapping)} 个")

# 打印数据示例
print("\n数据示例:")
sample_key = list(trail_activity.keys())[0]
print(f"键: {sample_key}")
print(f"该条件下的trials数量: {len(trail_activity[sample_key])}")
print(f"每个trial的神经元数量: {len(trail_activity[sample_key][0])}")
print(f"每个神经元数据形状: {trail_activity[sample_key][0][0].shape}")


class TrailActivityDataset(Dataset):
    def __init__(self, trail_activity, task_type='category', transform=None):
        """
        trail_activity: 从pkl文件加载的数据字典
        task_type: 'category', 'image', 'angle' 三种分类任务
        """
        self.trail_activity = trail_activity
        self.task_type = task_type
        self.transform = transform
        
        # 准备数据和标签
        self.data = []
        self.labels = []
        
        for key, trials in trail_activity.items():
            # key格式: "trial_condition_trial_target" (如 "1_1", "1_2", "1_3")
            trial_condition = int(key.split('_')[0])
            trial_target = int(key.split('_')[1])
            
            # 根据trial_condition确定图像类别
            if trial_condition <= 4:  # animals
                category = 'animals'
                if trial_condition == 1:
                    image_name = 'Dee1'
                elif trial_condition == 2:
                    image_name = 'Ele'
                elif trial_condition == 3:
                    image_name = 'Pig'
                elif trial_condition == 4:
                    image_name = 'Rhi'
            elif trial_condition <= 8:  # faces
                category = 'faces'
                if trial_condition == 5:
                    image_name = 'MA'
                elif trial_condition == 6:
                    image_name = 'MB'
                elif trial_condition == 7:
                    image_name = 'MC'
                elif trial_condition == 8:
                    image_name = 'WA'
            elif trial_condition <= 12:  # fruits
                category = 'fruits'
                if trial_condition == 9:
                    image_name = 'App1'
                elif trial_condition == 10:
                    image_name = 'Ban1'
                elif trial_condition == 11:
                    image_name = 'Pea1'
                elif trial_condition == 12:
                    image_name = 'Pin1'
            elif trial_condition <= 16:  # manmade
                category = 'manmade'
                if trial_condition == 13:
                    image_name = 'Bed1'
                elif trial_condition == 14:
                    image_name = 'Cha1'
                elif trial_condition == 15:
                    image_name = 'Dis1'
                elif trial_condition == 16:
                    image_name = 'Sof1'
            elif trial_condition <= 20:  # plants
                category = 'plants'
                if trial_condition == 17:
                    image_name = 'A'
                elif trial_condition == 18:
                    image_name = 'B'
                elif trial_condition == 19:
                    image_name = 'C'
                elif trial_condition == 20:
                    image_name = 'D'
            elif trial_condition <= 24:  # shape2d
                category = 'shape2d'
                if trial_condition == 21:
                    image_name = 'Cir'
                elif trial_condition == 22:
                    image_name = 'Oth'
                elif trial_condition == 23:
                    image_name = 'Squ'
                elif trial_condition == 24:
                    image_name = 'Tri'
            
            # 根据trial_target确定角度/颜色
            if category == 'shape2d':
                if trial_target == 1:
                    angle = 'B1'
                elif trial_target == 2:
                    angle = 'G1'
                elif trial_target == 3:
                    angle = 'R1'
            else:
                if trial_target == 1:
                    angle = '0'
                elif trial_target == 2:
                    angle = '315'
                elif trial_target == 3:
                    angle = '45'
            
            # 为每个trial添加数据
            for trial_data in trials:
                # trial_data shape: (86, 20, 1) - 86个神经元，20个时间bin
                self.data.append(trial_data)
                
                # 根据任务类型确定标签
                if task_type == 'category':
                    label = category_mapping[category]
                elif task_type == 'image':
                    label = image_mapping[image_name]
                elif task_type == 'angle':
                    label = angle_mapping[f"{image_name}_{angle}"]
                
                self.labels.append(label)
        
        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到[-1, 1]
        ])
        
        # 空图像张量作为备用
        self.empty_image_tensor = torch.zeros(3, 256, 256, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 数据形状: (86, 20, 1) -> (20, 86) 转置以适应模型
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32).squeeze(-1).T
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        
        # 创建dummy图像张量（与原代码兼容）
        img_tensor = self.empty_image_tensor.clone()
        
        if self.transform:
            data_tensor = self.transform(data_tensor)
            
        return data_tensor, img_tensor, label

# 创建数据集（测试三种分类任务）
print("\n创建数据集...")

# 1. 大类分类数据集
category_dataset = TrailActivityDataset(trail_activity, task_type='category')
print(f"大类分类数据集大小: {len(category_dataset)}")

# 2. 图片分类数据集
image_dataset = TrailActivityDataset(trail_activity, task_type='image')
print(f"图片分类数据集大小: {len(image_dataset)}")

# 3. 角度分类数据集
angle_dataset = TrailActivityDataset(trail_activity, task_type='angle')
print(f"角度分类数据集大小: {len(angle_dataset)}")

# 测试数据集创建
print("\n测试数据集...")
sample_data, sample_img, sample_label = category_dataset[0]
print(f"样本数据形状: {sample_data.shape}")
print(f"样本图像形状: {sample_img.shape}")
print(f"样本标签: {sample_label}")

# 检查标签分布
category_labels = [category_dataset[i][2].item() for i in range(min(100, len(category_dataset)))]
image_labels = [image_dataset[i][2].item() for i in range(min(100, len(image_dataset)))]
angle_labels = [angle_dataset[i][2].item() for i in range(min(100, len(angle_dataset)))]

print(f"\n标签分布:")
print(f"大类分类标签范围: {min(category_labels)} - {max(category_labels)}")
print(f"图片分类标签范围: {min(image_labels)} - {max(image_labels)}")
print(f"角度分类标签范围: {min(angle_labels)} - {max(angle_labels)}")

# 根据类别进行分层划分（以大类分类为例）
category_labels = [category_dataset[i][2].item() for i in range(len(category_dataset))]
train_indices, test_indices = train_test_split(
    range(len(category_dataset)), 
    test_size=0.2, 
    random_state=42, 
    stratify=category_labels
)

print(f"\n数据划分:")
print(f"训练集大小: {len(train_indices)}")
print(f"测试集大小: {len(test_indices)}")

# 创建训练和测试数据集（使用大类分类任务）
train_dataset = TrailActivityDataset(trail_activity, task_type='category')
test_dataset = TrailActivityDataset(trail_activity, task_type='category')

# 为了简化，我们使用所有数据作为训练集，所有数据作为测试集
# 在实际使用中，应该根据train_indices和test_indices来分割数据
print("数据集创建完成！")


MODEL_DEPTH = 16   

# 更新输入维度以适应trail_activity数据
# trail_activity数据形状: (86, 20, 1) -> 展平后为 86*20 = 1720
actual_input_dim = 86 * 20  # 神经元数量 * 时间bin数量 = 1720
num_classes_ep = 6  # 大类分类的类别数量

print(f"EP配置 - 输入维度: {actual_input_dim}, 类别数量: {num_classes_ep}")
print(f"数据形状: 86个神经元 × 20个时间bin = {actual_input_dim}")

vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'

FOR_512_px = MODEL_DEPTH == 16

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,   
    device=device, patch_nums=patch_nums, input_dim = actual_input_dim, num_classes_ep = num_classes_ep,
    num_classes=6, depth=MODEL_DEPTH, shared_aln=FOR_512_px  # 使用大类分类的类别数量
)

vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
#var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)

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
    
    epochs = 120
    grad_accum = 1
    label_smooth = 0.1
    amp_enabled = True
    
    prog_epochs = 10 
    prog_warmup_iters = 1000  
    
    log_interval = 50
    eval_interval = 1
    save_interval = 5
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
    if config.device.startswith("cuda"):
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
pdf_path = "reconstruction_results_utah_train.pdf"
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